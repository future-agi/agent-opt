import logging
import time
from typing import Any, Dict, List, Optional

# Import GEPA's core components. A try/except block makes this a soft dependency.
try:
    import gepa
    from gepa.core.adapter import GEPAAdapter, EvaluationBatch, DataInst
except ImportError:
    raise ImportError(
        "To use GEPAOptimizer, please install the 'gepa' library with: pip install gepa"
    )

from ..base.base_optimizer import BaseOptimizer
from ..datamappers.basic_mapper import BasicDataMapper
from ..base.evaluator import Evaluator
from ..generators.litellm import LiteLLMGenerator
from ..types import OptimizationResult, IterationHistory


class _InternalGEPAAdapter(GEPAAdapter[DataInst, Dict[str, Any], Dict[str, Any]]):
    """
    An internal adapter that translates our framework's components (Evaluator,
    DataMapper) into the interface GEPA's optimization engine expects.
    """

    def __init__(
        self, generator_model: str, evaluator: Evaluator, data_mapper: BasicDataMapper
    ):
        self.generator_model = generator_model
        self.evaluator = evaluator
        self.data_mapper = data_mapper
        logging.info(
            f"[_InternalGEPAAdapter] Initialized with generator_model: {generator_model}"
        )

    def evaluate(
        self,
        batch: List[Dict[str, Any]],
        candidate: Dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Dict[str, Any], Dict[str, Any]]:
        """
        This method is called by GEPA during its optimization loop. It uses
        our framework's components to perform the evaluation.
        """
        eval_start_time = time.time()
        logging.info(
            f"[_InternalGEPAAdapter] Starting evaluation for a candidate prompt."
        )

        # GEPA provides the prompt as the first (and only) value in the candidate dict.
        prompt_text = next(iter(candidate.values()))
        logging.info(
            f"[_InternalGEPAAdapter] Evaluating prompt: '{prompt_text[:100]}...'"
        )
        logging.info(f"[_InternalGEPAAdapter] Batch size: {len(batch)}")

        temp_generator = LiteLLMGenerator(
            model=self.generator_model, prompt_template=prompt_text
        )

        logging.info(f"[_InternalGEPAAdapter] Generating outputs...")
        gen_start_time = time.time()
        generated_outputs = [temp_generator.generate(example) for example in batch]
        gen_end_time = time.time()
        logging.info(
            f"[_InternalGEPAAdapter] Output generation finished in {gen_end_time - gen_start_time:.2f}s."
        )

        logging.info(f"[_InternalGEPAAdapter] Mapping evaluation inputs...")
        eval_inputs = [
            self.data_mapper.map(gen_out, ex)
            for gen_out, ex in zip(generated_outputs, batch)
        ]

        logging.info(f"[_InternalGEPAAdapter] Evaluating generated outputs...")
        evaluator_start_time = time.time()
        results = self.evaluator.evaluate(eval_inputs)
        evaluator_end_time = time.time()
        logging.info(
            f"[_InternalGEPAAdapter] Evaluation with framework evaluator finished in {evaluator_end_time - evaluator_start_time:.2f}s."
        )

        scores = [res.score for res in results]
        logging.info(f"[_InternalGEPAAdapter] Scores: {scores}")
        outputs = [
            {"generated_text": out, "full_result": res.model_dump()}
            for out, res in zip(generated_outputs, results)
        ]

        trajectories = []
        if capture_traces:
            logging.info(f"[_InternalGEPAAdapter] Capturing traces.")
            for i in range(len(batch)):
                trajectories.append(
                    {
                        "inputs": batch[i],
                        "generated_output": generated_outputs[i],
                        "evaluation_result": results[i].model_dump(),
                    }
                )

        eval_end_time = time.time()
        logging.info(
            f"[_InternalGEPAAdapter] Evaluation finished in {eval_end_time - eval_start_time:.2f}s."
        )
        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=trajectories
        )

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Creates the dataset for GEPA's reflective LLM to analyze.
        """
        logging.info(f"[_InternalGEPAAdapter] Creating reflective dataset.")
        reflective_data = {comp: [] for comp in components_to_update}

        if not eval_batch.trajectories:
            logging.warning(
                "[_InternalGEPAAdapter] No trajectories found to create reflective dataset."
            )
            return reflective_data

        logging.info(
            f"[_InternalGEPAAdapter] Processing {len(eval_batch.trajectories)} trajectories."
        )
        for trajectory in eval_batch.trajectories:
            result = trajectory.get("evaluation_result", {})
            score = result.get("score", 0.0)
            reason = result.get("reason", "No reason provided.")

            if score >= 0.8:
                feedback = f"This output was successful (score={score:.2f}). The reasoning for this score was: {reason}"
            else:
                feedback = f"This output performed poorly (score={score:.2f}). The key reason for the failure was: {reason}. The prompt needs to be improved to avoid this specific failure mode."

            example = {
                "Inputs": trajectory.get("inputs", {}),
                "Generated Outputs": trajectory.get("generated_output", ""),
                "Feedback": feedback,
            }

            for comp in components_to_update:
                reflective_data[comp].append(example)

        logging.info(
            f"[_InternalGEPAAdapter] Reflective dataset created for components: {components_to_update}"
        )
        return reflective_data


# ==============================================================================
# The User-Facing GEPA Optimizer Wrapper
# ==============================================================================


class GEPAOptimizer(BaseOptimizer):
    """
    An adapter that integrates the powerful GEPA evolutionary optimization
    algorithm into the prompt-optimizer framework.
    """

    def __init__(self, reflection_model: str, generator_model: str = "gpt-4o-mini"):
        """
        Initializes the GEPA Optimizer wrapper.

        Args:
            reflection_model (str): The name of a powerful LLM (e.g., "gpt-4-turbo")
                that GEPA will use for its reflection and mutation steps.
            generator_model (str): The name of the model that will be used by the
                prompts being optimized (the "task language model").
        """
        self.reflection_model = reflection_model
        self.generator_model = generator_model
        logging.info(
            f"[GEPAOptimizer] Initialized with reflection_model: {reflection_model}, generator_model: {generator_model}"
        )

    def optimize(
        self,
        evaluator: Evaluator,
        data_mapper: BasicDataMapper,
        dataset: List[Dict[str, Any]],
        initial_prompts: List[str],
        max_metric_calls: Optional[int] = 50,
    ) -> OptimizationResult:
        opt_start_time = time.time()
        logging.info("--- Starting GEPA Prompt Optimization ---")
        logging.info(f"[GEPAOptimizer] Dataset size: {len(dataset)}")
        logging.info(f"[GEPAOptimizer] Initial prompts: {initial_prompts}")
        logging.info(f"[GEPAOptimizer] Max metric calls: {max_metric_calls}")

        if not initial_prompts:
            raise ValueError("Initial prompts list cannot be empty for GEPAOptimizer.")

        # 1. Create the internal adapter that bridges our framework to GEPA
        logging.info("[GEPAOptimizer] Creating internal GEPA adapter...")
        adapter = _InternalGEPAAdapter(
            generator_model=self.generator_model,
            evaluator=evaluator,
            data_mapper=data_mapper,
        )

        # 2. Prepare the inputs for gepa.optimize
        seed_candidate = {"prompt": initial_prompts[0]}
        logging.info(f"[GEPAOptimizer] Seed candidate for GEPA: {seed_candidate}")

        # 3. Call the external GEPA library's optimize function
        logging.info("[GEPAOptimizer] Calling gepa.optimize...")
        gepa_start_time = time.time()
        gepa_result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=dataset,
            valset=dataset,
            adapter=adapter,
            reflection_lm=self.reflection_model,
            max_metric_calls=max_metric_calls,
            display_progress_bar=True,
        )
        gepa_end_time = time.time()
        logging.info(
            f"[GEPAOptimizer] gepa.optimize finished in {gepa_end_time - gepa_start_time:.2f}s."
        )
        logging.info(
            f"[GEPAOptimizer] GEPA result best score: {gepa_result.val_aggregate_scores[gepa_result.best_idx]}"
        )
        logging.info(
            f"[GEPAOptimizer] GEPA best candidate: {gepa_result.best_candidate}"
        )

        # 4. Translate GEPA's result back into our framework's standard format
        logging.info("[GEPAOptimizer] Translating GEPA result to OptimizationResult...")
        history = [
            IterationHistory(
                prompt=cand.get("prompt", ""),
                average_score=score,
                individual_results=[],  # GEPA's final result doesn't expose per-example scores easily
            )
            for cand, score in zip(
                gepa_result.candidates, gepa_result.val_aggregate_scores
            )
        ]

        final_best_generator = LiteLLMGenerator(
            model=self.generator_model,
            prompt_template=gepa_result.best_candidate.get("prompt", ""),
        )

        result = OptimizationResult(
            best_generator=final_best_generator,
            history=history,
            final_score=gepa_result.val_aggregate_scores[gepa_result.best_idx],
        )

        opt_end_time = time.time()
        logging.info(
            f"--- GEPA Prompt Optimization finished in {opt_end_time - opt_start_time:.2f}s ---"
        )
        logging.info(f"Final best score: {result.final_score}")

        return result
