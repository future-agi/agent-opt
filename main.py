import os
from dotenv import load_dotenv
from fi.evals import Evaluator as AIEvaluator
from prompt_optimizer.optimizers import RandomSearchOptimizer
from prompt_optimizer.generators import LiteLLMGenerator
from prompt_optimizer.datamappers import BasicDataMapper


def main():
    # Load API keys from .env file
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("FI_API_KEY"):
        print(
            "API keys not found. Please create a .env file with OPENAI_API_KEY and FI_API_KEY."
        )
        return

    # 1. Set up the Generator
    initial_prompt = "Write a short story based on the following idea: {prompt}"
    generator = LiteLLMGenerator(model="gpt-4o-mini", prompt_template=initial_prompt)

    # 2. Set up the Evaluator (from the external library)
    evaluator = AIEvaluator(
        fi_api_key=os.getenv("FI_API_KEY"), fi_secret_key=os.getenv("FI_SECRET_KEY")
    )

    # 3. Set up the Data Mapper
    key_map = {"input": "prompt", "output": "generated_output"}
    data_mapper = BasicDataMapper(key_map=key_map)

    # 4. Set up and run the Optimizer
    optimizer = RandomSearchOptimizer(
        generator=generator,
        teacher_model="gpt-5",
        num_variations=5,
        eval_template="summary_quality",  # User can now specify the template
        eval_model_name="turing_flash",  # And the evaluation model
    )

    # Define a simple dataset
    dataset = [
        {"prompt": "A robot who dreams of becoming a chef."},
        {"prompt": "A magical forest where the trees can talk."},
    ]

    results = optimizer.optimize(
        evaluator=evaluator, data_mapper=data_mapper, dataset=dataset
    )

    print("\n--- Optimization Complete ---")
    if results.final_score > -1:
        print(f"Final Score: {results.final_score:.4f}")
        print("Best Prompt Found:")
        print(results.best_generator.get_prompt_template())

        print("\n--- History of Prompts Tried ---")
        for item in results.history:
            print(f"Score: {item['score']:.4f}, Prompt: {item['prompt']}")
    else:
        print("Optimization did not find a successful prompt.")


if __name__ == "__main__":
    main()
