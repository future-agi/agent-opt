import os
import csv
import logging
import random
import time
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv

# --- Framework Imports ---
from prompt_optimizer.base.base_optimizer import BaseOptimizer
from prompt_optimizer.generators import LiteLLMGenerator
from prompt_optimizer.datamappers import BasicDataMapper
from prompt_optimizer.base.evaluator import Evaluator
from prompt_optimizer.types import OptimizationResult

# --- Import All Optimizers ---
from prompt_optimizer.optimizers import (
    RandomSearchOptimizer,
    ProTeGi,
    MetaPromptOptimizer,
    GEPAOptimizer,
)
from prompt_optimizer import setup_logging

# ==============================================================================
# Configuration
# ==============================================================================

# --- Dataset Configuration ---
DATASET_FILE = "experiments/datasets/d2.csv"
DATASET_SAMPLE_SIZE = 2

# --- Optimization Configuration ---
INITIAL_PROMPT = "Given the context:{context}, answer the question: {question}"
OPTIMIZER_KWARGS = {
    "num_rounds": 2,  # For ProTeGi and MetaPrompt
    "max_metric_calls": 40,  # For GEPA
    "num_variations": 3,  # For RandomSearch
}

# --- Model Configuration ---
GENERATOR_MODEL = "gpt-4o-mini"  # The model we are optimizing the prompt for
TEACHER_MODEL = "gpt-5-mini"  # A powerful model for the optimizers
EVALUATOR_MODEL = "turing_flash"  # A fast, cheap model for the online evaluator

# ==============================================================================
# Main Script Logic
# ==============================================================================
logger = logging.getLogger("testing_script")


def load_dataset(file_path: str, sample_size: int) -> List[Dict[str, Any]]:
    """Loads a dataset from a CSV file and returns a random sample."""
    logger.info(f"Loading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        df.dropna(subset=["context", "question", "answer"], inplace=True)
        # Convert to a flat dictionary structure
        full_dataset = df.to_dict("records")

        if not full_dataset:
            logger.error(
                "Dataset is empty after processing. Please check the CSV file."
            )
            return []

        if len(full_dataset) < sample_size:
            logger.warning(f"Dataset has {len(full_dataset)} rows, using all of them.")
            return full_dataset

        return random.sample(full_dataset, sample_size)
    except FileNotFoundError:
        logger.error(f"Dataset file not found at: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading or processing dataset: {e}")
        return []


def print_summary(optimizer_name: str, result: OptimizationResult, duration: float):
    """Prints a summary of the optimization results."""
    print("\n" + "=" * 80)
    print(f"✅ BAKE-OFF COMPLETE: {optimizer_name}")
    print(f"⏱️  Execution Time: {duration:.2f} seconds")
    print(f"🏆 Final Best Score: {result.final_score:.4f}")
    print("✨ Best Prompt Found:")
    print(result.best_generator.get_prompt_template())
    print("=" * 80)


def main() -> None:
    """Main function to run the optimizer bake-off."""
    load_dotenv()
    setup_logging(logging.INFO, log_to_file=True, log_file="test_optim.log")

    if not os.getenv("OPENAI_API_KEY") or not os.getenv("FI_API_KEY"):
        logger.error("API keys not found in .env file.")
        return

    # --- 1. Load Data and Set Up Common Components ---
    dataset = load_dataset(DATASET_FILE, DATASET_SAMPLE_SIZE)
    if not dataset:
        return

    logger.info("Setting up common components (Evaluator and Data Mapper)...")

    # The 'context_adherence' evaluator expects keys 'output' and 'context'.
    # We will map our generated text to 'output' and the original context to 'context'.
    evaluator = Evaluator(
        fi_api_key=os.getenv("FI_API_KEY"),
        fi_secret_key=os.getenv("FI_SECRET_KEY"),
        eval_template="context_adherence",
        eval_model_name="turing_flash",
    )
    data_mapper = BasicDataMapper(
        key_map={"output": "generated_output", "context": "context"}
    )

    # This will be the powerful teacher model for ProTeGi and MetaPromptOptimizer
    teacher_generator = LiteLLMGenerator(
        model=TEACHER_MODEL, prompt_template="{prompt}"
    )

    # --- 2. Initialize All Optimizers ---
    optimizers_to_test = {
        # "Random Search": RandomSearchOptimizer(
        #     generator=LiteLLMGenerator(GENERATOR_MODEL, INITIAL_PROMPT),
        #     teacher_model=TEACHER_MODEL,
        #     num_variations=OPTIMIZER_KWARGS["num_variations"],
        # ),
        "ProTeGi": ProTeGi(teacher_generator=teacher_generator, beam_size=2),
        "Meta-Prompt": MetaPromptOptimizer(teacher_generator=teacher_generator),
        "GEPA": GEPAOptimizer(
            reflection_model=TEACHER_MODEL, generator_model=GENERATOR_MODEL
        ),
    }

    all_results = {}

    # --- 3. Run Each Optimizer on the SAME Dataset ---
    for name, optimizer in optimizers_to_test.items():
        try:
            logger.info(f"\n--- Running Optimizer: {name} ---")
            start_time = time.time()

            # Call optimize with a consistent signature
            results = optimizer.optimize(
                evaluator=evaluator,
                data_mapper=data_mapper,
                dataset=dataset,
                initial_prompts=[INITIAL_PROMPT],
                **OPTIMIZER_KWARGS,
            )

            duration = time.time() - start_time
            all_results[name] = (results, duration)
            print_summary(name, results, duration)
        except Exception as e:
            logger.error(f"Optimizer '{name}' failed with an error: {e}", exc_info=True)

    # --- 4. Final Summary ---
    print("\n\n" + "#" * 80)
    print("###               OPTIMIZER BAKE-OFF FINAL SUMMARY                ###")
    print("#" * 80)

    if not all_results:
        print("No optimizers completed successfully.")
        return

    sorted_optimizers = sorted(
        all_results.items(), key=lambda item: item[1][0].final_score, reverse=True
    )

    for i, (name, (result, duration)) in enumerate(sorted_optimizers):
        print(f"\n--- Rank #{i + 1}: {name} ---")
        print(f"  Final Score: {result.final_score:.4f}")
        print(f"  Time Taken: {duration:.2f}s")
        print("  Best Prompt:")
        print(result.best_generator.get_prompt_template())


if __name__ == "__main__":
    main()
