import json
import logging
import random
import re
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field, ValidationError

from ..base.base_generator import BaseGenerator
from ..base.base_optimizer import BaseOptimizer
from ..datamappers.basic_mapper import BasicDataMapper
from ..base.evaluator import Evaluator
from ..generators.litellm import LiteLLMGenerator
from ..types import IterationHistory, OptimizationResult

MUTATE_PROMPT = """
You are an expert in prompt engineering. Your task is to generate {num_variations} diverse variations of the following instruction by adaptively mixing in the provided thinking styles.

[Task Description]: {task_description}
[Thinking Styles]: {meta_prompts}
[Prompt Instruction]: {prompt_instruction}

Return ONLY a valid JSON object with a single key "variations" containing a list of the new prompt strings.
"""

CRITIQUE_PROMPT = """
You are an expert prompt engineering analyst. My current prompt is:
---
{instruction}
---
This prompt performed poorly on the following examples:
---
{examples}
---
Provide a detailed critique explaining the potential reasons for failure.

Return ONLY a valid JSON object with a single key "variations" containing a list with ONE string: your critique.
"""

REFINE_PROMPT = """
You are an expert prompt engineer. My current prompt is:
---
{instruction}
---
It failed on these examples:
---
{examples}
---
Here is a critique of the prompt's weaknesses: "{critique}"

Based on this critique, write {steps_per_sample} different, improved versions of the prompt.

Return ONLY a valid JSON object with a single key "variations" containing a list of the new prompt strings.
"""


class Variations(BaseModel):
    variations: List[str]


class PromptWizardOptimizer(BaseOptimizer):
    """
    An adapter for the PromptWizard optimization algorithm, using a multi-stage
    process of mutation, critique, and refinement for prompt instructions.
    """

    def __init__(
        self,
        teacher_generator: BaseGenerator,
        mutate_rounds: int = 3,
        refine_iterations: int = 2,
        beam_size: int = 1,
    ):
        self.teacher = teacher_generator
        self.mutate_rounds = mutate_rounds
        self.refine_iterations = refine_iterations
        self.beam_size = beam_size
        self.thinking_styles = THINKING_STYLES
        logger.info("--- PromptWizard Optimizer Initialized ---")

    def optimize(
        self,
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
        initial_prompts: List[str],
        **kwargs: Any,
    ) -> OptimizationResult:
        task_description = kwargs.get("task_description", "No description provided.")
        eval_subset_size = kwargs.get("eval_subset_size", 25)

        if not initial_prompts:
            raise ValueError("Initial prompts list cannot be empty.")

        current_best_instruction = initial_prompts[0]
        history: List[IterationHistory] = []

        for i in range(self.refine_iterations):
            logger.info(
                f"\n--- Instruction Refinement Iteration {i + 1}/{self.refine_iterations} ---"
            )

            # 1. Mutate
            mutated_prompts = self._mutate_instruction(
                current_best_instruction, task_description
            )
            candidate_pool = {current_best_instruction, *mutated_prompts}

            # 2. Score
            eval_subset = random.sample(dataset, min(len(dataset), eval_subset_size))
            iteration_history = self._score_candidates(
                list(candidate_pool), evaluator, data_mapper, eval_subset
            )
            history.extend(iteration_history)

            sorted_by_score = sorted(
                iteration_history, key=lambda x: x.average_score, reverse=True
            )
            top_prompts_this_round = [
                item.prompt for item in sorted_by_score[: self.beam_size]
            ]

            # 3. Critique and Refine
            refined_prompts = set()
            for prompt_to_refine in top_prompts_this_round:
                errors = self._get_errors(
                    prompt_to_refine, evaluator, data_mapper, dataset
                )
                if errors:
                    refined = self._critique_and_refine(prompt_to_refine, errors)
                    if refined:
                        refined_prompts.add(refined)

            # Determine the best instruction for the next iteration
            if refined_prompts:
                final_candidates_this_round = {
                    current_best_instruction,
                    *refined_prompts,
                }
                final_history = self._score_candidates(
                    list(final_candidates_this_round),
                    evaluator,
                    data_mapper,
                    eval_subset,
                )
                history.extend(final_history)
                current_best_instruction = sorted(
                    final_history, key=lambda x: x.average_score, reverse=True
                )[0].prompt
            else:
                current_best_instruction = top_prompts_this_round[0]

            logger.info(
                f"Best instruction after iteration {i + 1}: '{current_best_instruction[:100]}...'"
            )

        final_history = sorted(history, key=lambda x: x.average_score, reverse=True)
        best_prompt = final_history[0].prompt
        best_score = final_history[0].average_score

        final_best_generator = LiteLLMGenerator("final-model", best_prompt)
        return OptimizationResult(
            best_generator=final_best_generator,
            history=history,
            final_score=best_score,
        )

    def _mutate_instruction(
        self, base_instruction: str, task_description: str
    ) -> Set[str]:
        all_variations = set()
        for _ in range(self.mutate_rounds):
            prompt = MUTATE_PROMPT.format(
                num_variations=len(self.thinking_styles),
                task_description=task_description,
                meta_prompts="\n".join(self.thinking_styles),
                prompt_instruction=base_instruction,
            )
            response_text = self.teacher.generate(
                {"prompt": prompt}, response_format={"type": "json_object"}
            )
            variations = self._parse_variations_from_json(response_text)
            all_variations.update(variations)
        return all_variations

    def _critique_and_refine(
        self, prompt: str, errors: List[Dict[str, Any]]
    ) -> str | None:
        error_str = json.dumps(errors, indent=2, ensure_ascii=False)

        critique_prompt = CRITIQUE_PROMPT.format(instruction=prompt, examples=error_str)
        critique_response = self.teacher.generate(
            {"prompt": critique_prompt}, response_format={"type": "json_object"}
        )
        critiques = self._parse_variations_from_json(critique_response)
        if not critiques:
            logger.warning("Critique generation failed, skipping refinement.")
            return None
        critique_text = critiques[0]  # Expecting a single critique string in the list

        refine_prompt = REFINE_PROMPT.format(
            instruction=prompt,
            examples=error_str,
            critique=critique_text,
            steps_per_sample=1,
        )
        refined_text = self.teacher.generate(
            {"prompt": refine_prompt}, response_format={"type": "json_object"}
        )

        refined_prompts = self._parse_variations_from_json(refined_text)
        return refined_prompts[0] if refined_prompts else None

    # Helper methods (shared with ProTeGi, kept here for encapsulation)
    def _get_errors(
        self,
        prompt: str,
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
        sample_size: int = 10,
    ) -> List[Dict[str, Any]]:
        subset = random.sample(dataset, min(len(dataset), sample_size))
        temp_generator = LiteLLMGenerator("gpt-4o-mini", prompt)
        generated_outputs = [temp_generator.generate(example) for example in subset]
        eval_inputs = [
            data_mapper.map(gen_out, ex)
            for gen_out, ex in zip(generated_outputs, subset)
        ]
        results = evaluator.evaluate(eval_inputs)
        return [
            {"inputs": subset[i], "output": generated_outputs[i], "score": res.score}
            for i, res in enumerate(results)
            if res.score < 0.5
        ]

    def _score_candidates(
        self,
        prompts: List[str],
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
    ) -> List[IterationHistory]:
        histories = []
        for prompt in prompts:
            temp_generator = LiteLLMGenerator("gpt-4o-mini", prompt)
            generated_outputs = [
                temp_generator.generate(example) for example in dataset
            ]
            eval_inputs = [
                data_mapper.map(gen_out, ex)
                for gen_out, ex in zip(generated_outputs, dataset)
            ]
            results = evaluator.evaluate(eval_inputs)
            avg_score = (
                sum(res.score for res in results) / len(results) if results else 0.0
            )
            histories.append(
                IterationHistory(
                    prompt=prompt, average_score=avg_score, individual_results=results
                )
            )
        return histories

    @staticmethod
    def _parse_variations_from_json(text: str) -> List[str]:
        """Parses a JSON string containing a 'variations' key."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                logger.warning(f"Could not find any JSON in the teacher's response.")
                return []
            json_str = match.group(0)
            data = json.loads(json_str)
            return Variations.model_validate(data).variations
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse teacher model JSON response: {e}")
            logger.debug(f"Raw problematic output:\n{text}")
            return []


# Static list of thinking styles from PromptWizard's config
THINKING_STYLES = [
    "How could I devise an experiment to help solve that problem?",
    "Make a list of ideas for solving this problem, and apply them one by one.",
    "How can I simplify the problem so that it is easier to solve?",
    "What are the key assumptions underlying this problem?",
    "Critical Thinking: Analyze the problem from different perspectives, questioning assumptions.",
    "Try creative thinking, generate innovative and out-of-the-box ideas.",
    "Use systems thinking: Consider the problem as part of a larger interconnected system.",
    "Let's think step by step.",
]
