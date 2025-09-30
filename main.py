import os
import logging
import time
from dotenv import load_dotenv
from prompt_optimizer.optimizers import BayesianSearchOptimizer
from prompt_optimizer.generators import LiteLLMGenerator
from prompt_optimizer.datamappers import BasicDataMapper
from prompt_optimizer.base.evaluator import Evaluator

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
    initial_prompt = (
        "Summarize the following article in 3-4 concise bullet points capturing key facts: {text}"
    )
    generator = LiteLLMGenerator(model="gpt-4o-mini", prompt_template=initial_prompt)

    # 2. Set up the Evaluator
    logging.info("Setting up the Evaluator...")
    evaluator = Evaluator(
        fi_api_key=os.getenv("FI_API_KEY"),
        fi_secret_key=os.getenv("FI_SECRET_KEY"),
        eval_template="task_completion",
        eval_model_name="turing_flash",
    )
    
    # 3. Set up the Data Mapper
    logging.info("Setting up the Data Mapper...")
    key_map = {"input": "text", "output": "generated_output"}
    data_mapper = BasicDataMapper(key_map=key_map)

    # 4. Set up the Optimizer
    logging.info("Setting up the Optimizer...")
    optimizer = BayesianSearchOptimizer(
        n_trials=3,
        min_examples=1,
        max_examples=3,
        eval_subset_size=3,
        eval_subset_strategy="random",
        # Enable teacher-guided template inference
        infer_example_template_via_teacher=True,
        teacher_model_name="gpt-5",
        teacher_model_kwargs={"temperature": 1.0, "max_tokens": 16000},
        few_shot_title="Few-shot Examples:",
    )
    
    # Define a dataset with placeholder stories for few-shot example construction
    dataset = [
        {
            "text": "OpenAI introduced a new multimodal model with improved vision and audio. The model supports real-time tool use and lower latency.",
        },
        {
            "text": "The ECB kept interest rates unchanged, citing persistent inflation concerns while forecasting moderate growth for the eurozone.",
        },
        {
            "text": "Apple unveiled new MacBook models featuring M-series chips, promising better battery life and enhanced AI workloads.",
        },
        {
            "text": "Researchers discovered a room-temperature superconductor claim was flawed, after replication attempts failed and key data was questioned.",
        },
        {
            "text": "A major airline announced a carbon-neutral plan by 2035, including sustainable aviation fuel investments and fleet upgrades.",
        },
        {
            "text": "NASA postponed the Artemis mission due to safety checks, targeting a new launch window pending further tests.",
        },
        {
            "text": "A cybersecurity firm reported a critical vulnerability in a popular VPN provider; patches were released and users urged to update.",
        },
        {
            "text": "The UN climate summit concluded with pledges to accelerate renewable energy deployment and phase down unabated coal.",
        },
        {
            "text": "A biotech startup received FDA approval for a gene therapy targeting a rare inherited disorder, marking a clinical milestone.",
        },
        {
            "text": "Developers launched a new open-source vector database with hybrid search, aiming at scalable RAG applications.",
        },
    ]
    logging.info(f"Using a dataset with {len(dataset)} examples.")

    # Run optimization
    logging.info("--- Starting Optimization Loop ---")
    optimization_start_time = time.time()
    results = optimizer.optimize(
        evaluator=evaluator,
        data_mapper=data_mapper,
        dataset=dataset,
        initial_prompts=[initial_prompt],
    )
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
