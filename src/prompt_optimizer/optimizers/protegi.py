import json
import logging
import random
import time
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field, ValidationError

from ..base.base_generator import BaseGenerator
from ..base.base_optimizer import BaseOptimizer
from ..datamappers.basic_mapper import BasicDataMapper
from ..base.evaluator import Evaluator
from ..generators.litellm import LiteLLMGenerator
from ..types import IterationHistory, OptimizationResult


GET_GRADIENTS_PROMPT = """
You are an expert in prompt engineering. I'm trying to write a zero-shot classifier prompt.
My current prompt is:
---
{prompt}
---

This prompt failed on the following examples:
---
{error_examples}
---

Provide {num_feedbacks} distinct reasons why the prompt could have failed on these examples.
Each reason should be a concise critique of the prompt's structure or wording.

Return a JSON object with a single key "variations" containing a list of strings.
Each string in the list should be a critique.
"""

APPLY_GRADIENT_PROMPT = """
You are an expert in prompt engineering. I'm trying to improve a zero-shot classifier prompt.

My current prompt is:
---
{prompt}
---

It failed on these examples:
---
{error_examples}
---

A key reason for the failure is: "{feedback}"

Based on this, generate {num_new_prompts} different, improved versions of the prompt.
The new prompts should directly address the feedback.

Return a JSON object with a single key "variations" containing a list of strings.
Each string in the list should be a new prompt.
"""


class GradientVariations(BaseModel):
    variations: List[str] = Field(description="A list of generated text strings.")


class ProTeGi(BaseOptimizer):
    """
    Optimizes prompts using a "textual gradient" approach inspired by ProTeGi.

    This method involves:
    1. Finding examples where the current prompt fails.
    2. Using a powerful "teacher" LLM to generate critiques ("gradients") of the prompt.
    3. Using the teacher LLM again to rewrite the prompt based on those critiques.
    """

    def __init__(
        self,
        teacher_generator: LiteLLMGenerator,
        num_gradients: int = 4,
        errors_per_gradient: int = 4,
        prompts_per_gradient: int = 1,
        beam_size: int = 4,
    ):
        """
        Initializes the Gradient Optimizer.

        Args:
            teacher_generator: A powerful generator (e.g., GPT-4, Claude 3 Opus)
                used to generate critiques and new prompt variations.
            num_gradients: The number of critiques to generate for each prompt.
            errors_per_gradient: The number of failure examples to show the teacher
                when generating a critique.
            prompts_per_gradient: The number of new prompts to generate for each critique.
            beam_size: The number of best prompts to keep for the next round.
        """
        self.teacher = teacher_generator
        self.num_gradients = num_gradients
        self.errors_per_gradient = errors_per_gradient
        self.prompts_per_gradient = prompts_per_gradient
        self.beam_size = beam_size

        super().__init__()

        logging.info("--- ProTeGi Optimizer Initialized ---")
        logging.info(f"Teacher Model: {self.teacher.model_name}")
        logging.info(f"Number of Gradients per Prompt: {self.num_gradients}")
        logging.info(f"Errors per Gradient: {self.errors_per_gradient}")
        logging.info(f"Prompts per Gradient: {self.prompts_per_gradient}")
        logging.info(f"Beam Size: {self.beam_size}")
        logging.info("------------------------------------")

    def optimize(
        self,
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
        initial_prompts: List[str],
        num_rounds: int = 3,
        eval_subset_size: int = 32,
    ) -> OptimizationResult:
        logging.info("--- Starting ProTeGi Prompt Optimization ---")
        logging.info(f"Initial prompts: {len(initial_prompts)}")
        logging.info(f"Number of rounds: {num_rounds}")
        logging.info(f"Evaluation subset size: {eval_subset_size}")

        candidates = initial_prompts
        best_overall_score = -1.0
        best_overall_prompt = initial_prompts[0]
        history: List[IterationHistory] = []

        for round_num in range(num_rounds):
            logging.info(
                f"\n--- Starting Optimization Round {round_num + 1}/{num_rounds} ---"
            )

            # 1. Expand the set of candidate prompts using textual gradients
            logging.info(
                f"Expanding {len(candidates)} prompts into a new set of candidates..."
            )
            candidates = self._expand_candidates(
                candidates, evaluator, data_mapper, dataset
            )
            logging.info(
                f"Generated a new pool of {len(candidates)} candidate prompts."
            )

            # 2. Score all candidates to find the best ones for the next round
            logging.info(
                f"Scoring {len(candidates)} candidates on a subset of the data..."
            )
            eval_subset = random.sample(dataset, min(len(dataset), eval_subset_size))
            logging.info(f"Evaluation subset size for this round: {len(eval_subset)}")

            iteration_history = self._score_candidates(
                candidates, evaluator, data_mapper, eval_subset
            )
            history.extend(iteration_history)

            # 3. Select the top N prompts for the next round (beam search)
            sorted_history = sorted(
                iteration_history, key=lambda x: x.average_score, reverse=True
            )

            if not sorted_history:
                logging.warning(
                    "No successful evaluations in this round. Halting optimization."
                )
                break

            candidates = [item.prompt for item in sorted_history[: self.beam_size]]
            best_round_score = sorted_history[0].average_score
            best_round_prompt = sorted_history[0].prompt

            logging.info(f"Best score in round {round_num + 1}: {best_round_score:.4f}")
            logging.info(f"Selected top {len(candidates)} prompts for the next round.")

            if best_round_score > best_overall_score:
                best_overall_score = best_round_score
                best_overall_prompt = best_round_prompt
                logging.info(
                    f"New best overall prompt found with score: {best_overall_score:.4f}"
                )

        final_best_generator = LiteLLMGenerator(
            self.teacher.model_name, best_overall_prompt
        )

        logging.info("--- ProTeGi Prompt Optimization Finished ---")
        logging.info(f"Final best score: {best_overall_score:.4f}")
        logging.info(f"Final best prompt: \n{best_overall_prompt}")
        logging.info("-----------------------------------------")

        return OptimizationResult(
            best_generator=final_best_generator,
            history=history,
            final_score=best_overall_score,
        )

    def _expand_candidates(
        self,
        prompts: List[str],
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
    ) -> List[str]:
        """Generates new prompt variations based on critiques of failures."""
        new_prompts = set(prompts)

        for i, prompt in enumerate(prompts):
            logging.info(f"--> Expanding prompt {i + 1}/{len(prompts)}...")
            # Find errors for the current prompt
            errors = self._get_errors(prompt, evaluator, data_mapper, dataset)
            if not errors:
                logging.info(
                    f"Prompt produced no errors on the dataset sample. Keeping as is."
                )
                continue
            logging.info(f"Found {len(errors)} examples where the prompt failed.")

            # Generate critiques ("gradients") based on these errors
            critiques = self._get_gradients(prompt, errors)
            logging.info(f"Generated {len(critiques)} critiques (gradients).")

            # Generate new prompts by "applying" these critiques
            for j, feedback in enumerate(critiques):
                logging.info(
                    f"Applying gradient {j + 1}/{len(critiques)}: '{feedback[:80]}...'"
                )
                generated = self._apply_gradient(prompt, errors, feedback)
                logging.info(
                    f"Generated {len(generated)} new prompts from this gradient."
                )

                new_prompts.update(generated)

        return list(new_prompts)

    def _get_errors(
        self,
        prompt: str,
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
        sample_size: int = 64,
    ) -> List[Dict[str, Any]]:
        """Finds examples where the prompt results in a low score."""
        subset = random.sample(dataset, min(len(dataset), sample_size))
        logging.info(f"Getting errors from a subset of {len(subset)} examples.")

        # We need a temporary generator to evaluate the specific prompt
        temp_generator = LiteLLMGenerator("gpt-4o-mini", prompt)

        start_time_gen = time.time()
        generated_outputs = [temp_generator.generate(example) for example in subset]
        end_time_gen = time.time()
        logging.info(
            f"    Generation for error search took: {end_time_gen - start_time_gen:.2f}s"
        )

        eval_inputs = [
            data_mapper.map(gen_out, ex)
            for gen_out, ex in zip(generated_outputs, subset)
        ]

        start_time_eval = time.time()
        results = evaluator.evaluate(eval_inputs)
        end_time_eval = time.time()
        logging.info(
            f"    Evaluation for error search took: {end_time_eval - start_time_eval:.2f}s"
        )

        errors = [subset[i] for i, res in enumerate(results) if res.score < 0.5]
        logging.info(f"Found {len(errors)} errors with score < 0.5.")
        return errors

    def _get_gradients(self, prompt: str, errors: List[Dict[str, Any]]) -> List[str]:
        """Uses the teacher model to generate critiques of the prompt."""
        error_sample = random.sample(errors, min(len(errors), self.errors_per_gradient))
        logging.info(f"Generating gradients from {len(error_sample)} error examples.")
        error_examples_str = json.dumps(error_sample, indent=2)

        prompt_vars = {
            "prompt": prompt,
            "error_examples": error_examples_str,
            "num_feedbacks": self.num_gradients,
        }

        # We use a temporary generator for the teacher model call
        critique_prompt = GET_GRADIENTS_PROMPT.format(**prompt_vars)
        response_text = self.teacher.generate({"prompt": critique_prompt})

        return self._parse_variations(response_text)

    def _apply_gradient(
        self, prompt: str, errors: List[Dict[str, Any]], feedback: str
    ) -> List[str]:
        """Uses the teacher model to rewrite the prompt based on a critique."""
        error_sample = random.sample(errors, min(len(errors), self.errors_per_gradient))
        logging.info(
            f"Applying gradient with {len(error_sample)} error examples and feedback: '{feedback[:80]}...'"
        )
        error_examples_str = json.dumps(error_sample, indent=2)

        prompt_vars = {
            "prompt": prompt,
            "error_examples": error_examples_str,
            "feedback": feedback,
            "num_new_prompts": self.prompts_per_gradient,
        }

        rewrite_prompt = APPLY_GRADIENT_PROMPT.format(**prompt_vars)
        response_text = self.teacher.generate({"prompt": rewrite_prompt})

        return self._parse_variations(response_text)

    def _score_candidates(
        self,
        prompts: List[str],
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
    ) -> List[IterationHistory]:
        """Scores a list of prompts and returns the detailed history."""
        histories = []
        for i, prompt in enumerate(prompts):
            logging.info(f"--> Scoring prompt {i + 1}/{len(prompts)}...")
            temp_generator = LiteLLMGenerator("gpt-4o-mini", prompt)

            start_time_gen = time.time()
            generated_outputs = [
                temp_generator.generate(example) for example in dataset
            ]
            end_time_gen = time.time()
            logging.info(f"    Generation took: {end_time_gen - start_time_gen:.2f}s")

            eval_inputs = [
                data_mapper.map(gen_out, ex)
                for gen_out, ex in zip(generated_outputs, dataset)
            ]

            start_time_eval = time.time()
            results = evaluator.evaluate(eval_inputs)
            end_time_eval = time.time()
            logging.info(f"    Evaluation took: {end_time_eval - start_time_eval:.2f}s")

            avg_score = (
                sum(res.score for res in results) / len(results) if results else 0.0
            )
            logging.info(f"    Average score: {avg_score:.4f}")

            histories.append(
                IterationHistory(
                    prompt=prompt, average_score=avg_score, individual_results=results
                )
            )
        return histories

    def _parse_variations(self, text: str) -> List[str]:
        try:
            # First, try to find a JSON code block
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
                return GradientVariations.model_validate(data).variations

            # If no code block, try to parse the whole string
            return GradientVariations.model_validate_json(text).variations

        except (json.JSONDecodeError, ValidationError, IndexError) as e:
            logging.error(f"Failed to parse model output: {e}")
            logging.error(f"Raw output:\n{text}")
            return []
