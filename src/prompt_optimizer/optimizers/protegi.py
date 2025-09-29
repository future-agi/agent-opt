import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Set

import numpy as np
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
This prompt performed poorly on the following examples:
---
{error_examples}
---
Provide {num_feedbacks} distinct reasons why the prompt could have failed. Each reason should be a concise critique.
Return ONLY a valid JSON object with a single key "variations" containing a list of strings (the critiques).
"""

APPLY_GRADIENT_PROMPT = """
You are an expert in prompt engineering. I'm trying to improve a zero-shot classifier prompt.
My current prompt is:
---
{prompt}
---
It performed poorly on these examples:
---
{error_examples}
---
A key reason for the failure is the following critique: "{feedback}"
Based on this critique, generate {num_new_prompts} different, improved versions of the prompt.
Return ONLY a valid JSON object with a single key "variations" containing a list of strings (the new prompts).
"""

PARAPHRASE_PROMPT = """
Generate {num_variations} semantic paraphrases of the following prompt. The meaning should be identical, but the wording should be different.
---
{prompt}
---
Return ONLY a valid JSON object with a single key "variations" containing a list of strings (the paraphrased prompts).
"""


class GradientVariations(BaseModel):
    variations: List[str] = Field(description="A list of generated text strings.")


class BanditSelector:
    """Implements the UCB bandit algorithm for efficient prompt selection."""

    def __init__(self, candidate_prompts: List[str], c: float = 2.0):
        self.prompts = candidate_prompts
        self.n_prompts = len(candidate_prompts)
        self.c = c  # Exploration parameter
        self.prompt_to_idx = {p: i for i, p in enumerate(self.prompts)}

        self.counts = np.zeros(self.n_prompts)
        self.scores = np.zeros(self.n_prompts)
        self.total_pulls = 0

    def select_arms(self, num_arms: int):
        """Selects the best arms (prompts) to pull based on UCB scores."""
        # Give every arm at least one chance
        untested_indices = np.where(self.counts == 0)[0]
        if len(untested_indices) > 0:
            return [(i, self.prompts[i]) for i in untested_indices[:num_arms]]

        # UCB calculation
        average_scores = self.scores / self.counts
        exploration_term = self.c * np.sqrt(np.log(self.total_pulls) / self.counts)
        ucb_scores = average_scores + exploration_term

        # Get the indices of the top arms
        top_indices = np.argsort(ucb_scores)[-num_arms:][::-1]
        return [(i, self.prompts[i]) for i in top_indices]

    def update(self, prompt_idx: int, score: float, num_samples: int):
        """Updates the scores and counts for a pulled arm."""
        self.scores[prompt_idx] += score * num_samples
        self.counts[prompt_idx] += num_samples
        self.total_pulls += num_samples

    def get_best_prompts(self, num_prompts: int) -> List[str]:
        """Returns the top prompts based on their empirical average score."""
        if self.n_prompts == 0:
            return []
        average_scores = np.divide(
            self.scores,
            self.counts,
            out=np.zeros_like(self.scores),
            where=self.counts != 0,
        )
        top_indices = np.argsort(average_scores)[-num_prompts:][::-1]
        return [self.prompts[i] for i in top_indices]


class ProTeGi(BaseOptimizer):
    """
    A corrected and robust implementation of the ProTeGi optimizer.
    """

    def __init__(
        self,
        teacher_generator: LiteLLMGenerator,
        num_gradients: int = 4,
        errors_per_gradient: int = 4,
        prompts_per_gradient: int = 1,
        beam_size: int = 4,
    ):
        self.teacher = teacher_generator
        self.num_gradients = num_gradients
        self.errors_per_gradient = errors_per_gradient
        self.prompts_per_gradient = prompts_per_gradient
        self.beam_size = beam_size
        logging.info("--- ProTeGi Optimizer Initialized ---")

    def optimize(
        self,
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
        initial_prompts: List[str],
        **kwargs: Any,
    ) -> OptimizationResult:
        num_rounds = kwargs.get("num_rounds", 3)
        eval_subset_size = kwargs.get("eval_subset_size", 32)

        beam = set(initial_prompts)
        best_overall_score = -1.0
        best_overall_prompt = initial_prompts[0] if initial_prompts else ""
        history: List[IterationHistory] = []

        for round_num in range(num_rounds):
            logging.info(
                f"\n--- Starting Optimization Round {round_num + 1}/{num_rounds} ---"
            )

            # 1. EXPANSION: Generate new candidates from the current beam
            current_prompts = list(beam)
            logging.info(
                f"Expanding {len(current_prompts)} prompts into new candidates..."
            )
            expanded_prompts = self._expand_candidates(
                current_prompts, evaluator, data_mapper, dataset
            )

            # The candidate pool for this round is the union of the old beam and new prompts
            candidate_pool = beam.union(expanded_prompts)
            logging.info(
                f"Candidate pool for this round has {len(candidate_pool)} unique prompts."
            )

            # 2. SELECTION: Score all candidates in the pool
            eval_subset = random.sample(dataset, min(len(dataset), eval_subset_size))
            iteration_history = self._score_candidates(
                list(candidate_pool), evaluator, data_mapper, eval_subset
            )
            history.extend(iteration_history)

            # 3. BEAM UPDATE: Select the top N prompts for the next round
            sorted_history = sorted(
                iteration_history, key=lambda x: x.average_score, reverse=True
            )
            if not sorted_history:
                logging.warning("No successful evaluations in this round. Halting.")
                break

            beam = {item.prompt for item in sorted_history[: self.beam_size]}
            best_round_score = sorted_history[0].average_score
            best_round_prompt = sorted_history[0].prompt

            logging.info(f"Best score in round {round_num + 1}: {best_round_score:.4f}")
            logging.info(f"New beam selected with {len(beam)} prompts.")

            if best_round_score > best_overall_score:
                best_overall_score = best_round_score
                best_overall_prompt = best_round_prompt

        final_best_generator = LiteLLMGenerator(
            self.teacher.model_name, best_overall_prompt
        )
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
    ) -> Set[str]:
        new_prompts = set()
        for i, prompt in enumerate(prompts):
            logging.debug(f"--> Expanding prompt {i + 1}/{len(prompts)}...")
            errors = self._get_errors(prompt, evaluator, data_mapper, dataset)
            if not errors:
                logging.debug(f"Prompt produced no errors. No expansion.")
                continue

            critiques = self._get_gradients(prompt, errors)
            logging.debug(f"Generated {len(critiques)} critiques (gradients).")

            for feedback in critiques:
                generated = self._apply_gradient(prompt, errors, feedback)
                if generated:
                    logging.debug(
                        f"Generated {len(generated)} new prompts from critique: '{feedback[:50]}...'"
                    )
                    new_prompts.update(generated)
        return new_prompts

    def _get_errors(
        self,
        prompt: str,
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
        sample_size: int = 32,
    ) -> List[Dict[str, Any]]:
        subset = random.sample(dataset, min(len(dataset), sample_size))
        temp_generator = LiteLLMGenerator("gpt-4o-mini", prompt)

        generated_outputs = [temp_generator.generate(example) for example in subset]
        eval_inputs = [
            data_mapper.map(gen_out, ex)
            for gen_out, ex in zip(generated_outputs, subset)
        ]
        results = evaluator.evaluate(eval_inputs)

        errors = [subset[i] for i, res in enumerate(results) if res.score < 0.5]
        logging.debug(
            f"Found {len(errors)} errors with score < 0.5 from a subset of {len(subset)}."
        )
        return errors

    def _get_gradients(self, prompt: str, errors: List[Dict[str, Any]]) -> List[str]:
        error_sample = random.sample(errors, min(len(errors), self.errors_per_gradient))
        critique_prompt = GET_GRADIENTS_PROMPT.format(
            prompt=prompt,
            error_examples=json.dumps(error_sample, indent=2),
            num_feedbacks=self.num_gradients,
        )
        response_text = self.teacher.generate(
            prompt_vars={"prompt": critique_prompt},
            response_format={"type": "json_object"},
        )
        return self._parse_variations_from_json(response_text)

    def _apply_gradient(
        self, prompt: str, errors: List[Dict[str, Any]], feedback: str
    ) -> List[str]:
        error_sample = random.sample(errors, min(len(errors), self.errors_per_gradient))
        rewrite_prompt = APPLY_GRADIENT_PROMPT.format(
            prompt=prompt,
            error_examples=json.dumps(error_sample, indent=2),
            feedback=feedback,
            num_new_prompts=self.prompts_per_gradient,
        )
        response_text = self.teacher.generate(
            prompt_vars={"prompt": rewrite_prompt},
            response_format={"type": "json_object"},
        )
        return self._parse_variations_from_json(response_text)

    def _score_candidates(
        self,
        prompts: List[str],
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
    ) -> List[IterationHistory]:
        histories = []
        for i, prompt in enumerate(prompts):
            logging.info(
                f"--> Scoring prompt {i + 1}/{len(prompts)}: '{prompt[:100]}...'"
            )
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
            logging.info(f"    Average score: {avg_score:.4f}")
            histories.append(
                IterationHistory(
                    prompt=prompt, average_score=avg_score, individual_results=results
                )
            )
        return histories

    @staticmethod
    def _parse_variations_from_json(text: str) -> List[str]:
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                logging.warning(f"Could not find any JSON in the teacher's response.")
                return []
            json_str = match.group(0)
            data = json.loads(json_str)
            return GradientVariations.model_validate(data).variations
        except (json.JSONDecodeError, ValidationError) as e:
            logging.error(f"Failed to parse teacher model JSON response: {e}")
            logging.debug(f"Raw problematic output:\n{text}")
            return []
