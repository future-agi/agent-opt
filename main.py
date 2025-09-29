import os
import logging
import time
from dotenv import load_dotenv
from fi.evals import Evaluator as AIEvaluator
from prompt_optimizer.optimizers import RandomSearchOptimizer
from prompt_optimizer.generators import LiteLLMGenerator
from prompt_optimizer.datamappers import BasicDataMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main function to run the prompt optimization."""
    logging.info("--- Starting Prompt Optimization ---")
    start_time = time.time()

    # Load API keys from .env file
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("FI_API_KEY"):
        logging.error(
            "API keys not found. Please create a .env file with OPENAI_API_KEY and FI_API_KEY."
        )
        return

    # 1. Set up the Generator
    logging.info("Setting up the Generator...")
    initial_prompt = "Write a short story based on the following idea: {prompt}"
    generator = LiteLLMGenerator(model="gpt-4o-mini", prompt_template=initial_prompt)

    # 2. Set up the Evaluator
    logging.info("Setting up the Evaluator...")
    evaluator = AIEvaluator(
        fi_api_key=os.getenv("FI_API_KEY"), fi_secret_key=os.getenv("FI_SECRET_KEY")
    )

    # 3. Set up the Data Mapper
    logging.info("Setting up the Data Mapper...")
    key_map = {"input": "prompt", "output": "generated_output"}
    data_mapper = BasicDataMapper(key_map=key_map)

    # 4. Set up the Optimizer
    logging.info("Setting up the Optimizer...")
    optimizer = RandomSearchOptimizer(
        generator=generator,
        teacher_model="gemini/gemini-2.5-flash-lite",
        num_variations=3,
        eval_template="summary_quality",
        eval_model_name="turing_flash",
    )

    # Define a simple dataset
    dataset = [
        {"prompt": "A robot who dreams of becoming a chef."},
        {"prompt": "A magical forest where the trees can talk."},
    ]
    logging.info(f"Using a dataset with {len(dataset)} examples.")

    # Run optimization
    logging.info("--- Starting Optimization Loop ---")
    optimization_start_time = time.time()
    results = optimizer.optimize(
        evaluator=evaluator, data_mapper=data_mapper, dataset=dataset
    )
    optimization_end_time = time.time()
    logging.info(
        f"--- Optimization Loop Finished in {optimization_end_time - optimization_start_time:.2f} seconds ---"
    )

    # Process and log results
    if results and results.final_score > -1:
        logging.info(f"Final Score: {results.final_score:.4f}")
        logging.info("Best Prompt Found:")
        logging.info(results.best_generator.get_prompt_template())

        logging.info("--- History of Prompts Tried ---")
        for item in results.history:
            logging.info(f"Score: {item['score']:.4f}, Prompt: {item['prompt']}")
    else:
        logging.warning("Optimization did not find a successful prompt.")

    end_time = time.time()
    logging.info(f"--- Total Execution Time: {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()
