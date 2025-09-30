import os
import logging
import time
from dotenv import load_dotenv
from fi.evals import Evaluator as AIEvaluator
from prompt_optimizer.optimizers import (
    RandomSearchOptimizer,
    ProTeGi,
    MetaPromptOptimizer,
    GEPAOptimizer,
)
from prompt_optimizer.generators import LiteLLMGenerator
from prompt_optimizer.datamappers import BasicDataMapper
from prompt_optimizer.base.evaluator import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
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
    initial_prompt = "Summarize in exactly one sentence: {text}"
    generator = LiteLLMGenerator(model="gpt-4o-mini", prompt_template=initial_prompt)

    # 2. Set up the Evaluator
    logging.info("Setting up the Evaluator...")

    # evaluator = AIEvaluator(
    #     fi_api_key=os.getenv("FI_API_KEY"), fi_secret_key=os.getenv("FI_SECRET_KEY")
    # )

    evaluator = Evaluator(
        fi_api_key=os.getenv("FI_API_KEY"),
        fi_secret_key=os.getenv("FI_SECRET_KEY"),
        eval_template="summary_quality",
        eval_model_name="turing_flash",
    )
    # 3. Set up the Data Mapper
    logging.info("Setting up the Data Mapper...")
    key_map = {"input": "text", "output": "generated_output"}
    data_mapper = BasicDataMapper(key_map=key_map)

    # 4. Set up the Optimizer
    logging.info("Setting up the Optimizer...")
    # optimizer = RandomSearchOptimizer(
    #     generator=generator,
    #     teacher_model="gemini/gemini-2.5-flash-lite",
    #     num_variations=4,
    # )
    # results = optimizer.optimize(
    #     evaluator=evaluator, data_mapper=data_mapper, dataset=dataset
    # )
    # optimizer = ProTeGi(
    #     teacher_generator=LiteLLMGenerator(
    #         "gemini/gemini-2.5-flash-lite", prompt_template=initial_prompt
    #     )
    # ))
    optimizer = GEPAOptimizer(reflection_model="gpt-5")
    # Define a simple dataset
    dataset = [
        {
            "text": "The sun is a star at the center of the Solar System. It is a nearly perfect sphere of hot plasma, with internal convective motion that generates a magnetic field via a dynamo process."
        },
        {
            "text": "Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant whitespace."
        },
        {
            "text": "The mitochondria is the powerhouse of the cell, responsible for generating most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy."
        },
        {
            "text": "The Roman Empire was the post-Republican period of ancient Rome. As a polity it included large territorial holdings around the Mediterranean Sea in Europe, Northern Africa, and Western Asia ruled by emperors."
        },
        {
            "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals, which involves consciousness and emotionality."
        },
        {
            "text": "To Kill a Mockingbird is a novel by Harper Lee published in 1960. Instantly successful, widely read in high schools and middle schools in the United States, it has become a classic of modern American literature, winning the Pulitzer Prize."
        },
        {
            "text": "Supply and demand is a microeconomic model of price determination in a market. It postulates that, holding all else equal, in a competitive market, the unit price for a particular good, or other traded item such as labor or liquid financial assets, will vary until it settles at a point where the quantity demanded will equal the quantity supplied."
        },
        {
            "text": "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science."
        },
    ]
    logging.info(f"Using a dataset with {len(dataset)} examples.")
    results = optimizer.optimize(
        evaluator=evaluator,
        data_mapper=data_mapper,
        dataset=dataset,
        initial_prompts=[initial_prompt],
        max_metric_calls=20,
    )
    # Run optimization
    logging.info("--- Starting Optimization Loop ---")
    optimization_start_time = time.time()
    optimization_end_time = time.time()
    logging.info(
        f"--- Optimization Loop Finished in {optimization_end_time - optimization_start_time:.2f} seconds ---"
    )

    # Process and log results
    if results and results.final_score > -1:
        logging.info(f"🏆 Final Best Score: {results.final_score:.4f}")
        logging.info("🏆 Best Prompt Found:")
        logging.info(results.best_generator.get_prompt_template())

        logging.info("\n--- 📜 Full Optimization History ---")
        for i, iteration in enumerate(results.history):
            logging.info(f"\n  --- Iteration {i + 1} ---")
            logging.info(f"  Prompt: {iteration.prompt}")
            logging.info(f"  Average Score: {iteration.average_score:.4f}")
            logging.info("  Individual Results:")
            for j, res in enumerate(iteration.individual_results):
                logging.info(
                    f"    - Example {j + 1} Score: {res.score:.2f}, Reason: {res.reason}"
                )
    else:
        logging.warning("Optimization did not find a successful prompt.")

    end_time = time.time()
    logging.info(f"--- Total Execution Time: {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()
