# agent-opt

A powerful Python SDK for automated prompt optimization using state-of-the-art algorithms and LLM-as-a-judge evaluation.

## Overview

**agent-opt** is a comprehensive framework for optimizing prompts through iterative refinement. It supports multiple optimization algorithms, flexible evaluation strategies, and seamless integration with popular LLM providers through LiteLLM.

## Features

- **Multiple Optimization Algorithms**: Choose from 6 different optimization strategies

  - Random Search
  - Bayesian Search (with Optuna)
  - ProTeGi (Prompt Optimization with Textual Gradients)
  - Meta-Prompt
  - PromptWizard
  - GEPA (Genetic Pareto)

- **Flexible Evaluation**: Support for multiple evaluation backends

  - Heuristic metrics (BLEU, ROUGE, etc.)
  - Custom LLM-as-a-judge metrics
  - FutureAGI platform integration

- **Easy Integration**: Works with any LLM through LiteLLM
- **Extensible Architecture**: Clean abstractions for custom optimizers, generators, and evaluators

## Installation

```bash
pip install agent-opt
```

## Quick Start

```python
from fi.opt.generators import LiteLLMGenerator
from fi.opt.optimizers import BayesianSearchOptimizer
from fi.opt.datamappers import BasicDataMapper
from fi.opt.base.evaluator import Evaluator
from fi.evals.metrics import BLEUScore

# 1. Set up your dataset
dataset = [
    {"context": "Paris is the capital of France", "question": "What is the capital of France?", "answer": "Paris"},
    # ... more examples
]

# 2. Configure the evaluator
metric = BLEUScore()
evaluator = Evaluator(metric)

# 3. Set up data mapping
data_mapper = BasicDataMapper(
    key_map={
        "response": "generated_output",
        "expected_response": "answer"
    }
)

# 4. Choose and configure an optimizer
optimizer = BayesianSearchOptimizer(
    inference_model_name="gpt-4o-mini",
    teacher_model_name="gpt-5",
    n_trials=10
)

# 5. Run optimization
initial_prompt = "Given the context: {context}, answer the question: {question}"
result = optimizer.optimize(
    evaluator=evaluator,
    data_mapper=data_mapper,
    dataset=dataset,
    initial_prompts=[initial_prompt]
)

# 6. Get the best prompt
print(f"Best Score: {result.final_score:.4f}")
print(f"Best Prompt: {result.best_generator.get_prompt_template()}")
```

## Core Components

### Generators

Generators execute prompts and return responses. The framework includes `LiteLLMGenerator` for easy integration with any LLM provider.

```python
from fi.opt.generators import LiteLLMGenerator

generator = LiteLLMGenerator(
    model="gpt-4o-mini",
    prompt_template="Summarize this text: {text}"
)
```

### Evaluators

Evaluators score generated outputs using various strategies:

**Local Metrics (Heuristic)**

```python
from fi.opt.base.evaluator import Evaluator
from fi.evals.metrics import BLEUScore

evaluator = Evaluator(metric=BLEUScore())
```

**LLM-as-a-Judge**

```python
from fi.evals.llm import LiteLLMProvider
from fi.evals.metrics import CustomLLMJudge
# LLM provider used by the judge
provider = LiteLLMProvider()
# Create custom LLM judge metric
correctness_judge_config = {
    "name": "correctness_judge",
    "grading_criteria": '''You are evaluating an AI's answer to a question. The score must be 1.0 if the 'response'
is semantically equivalent to the 'expected_response' (the ground truth). The score should be 0.0 if it is incorrect.
Partial credit is acceptable. For example, if the expected answer is "Gustave Eiffel" and the response is
"The tower was designed by Eiffel", a score of 0.8 is appropriate.''',
}

# Instantiate the judge and pass the metric to the evaluator
correctness_judge = CustomLLMJudge(
    provider=provider,
    config=correctness_judge_config,
    # pass litellm completion params as kwargs!
    model="gemini/gemini-2.5-flash",
    temperature=0.4)
evaluator = Evaluator(metric=correctness_judge)
```

**FutureAGI Platform**

Choose from our wide range of evaluations

```python
evaluator = Evaluator(
    eval_template="summary_quality",
    eval_model_name="turing_flash",
    fi_api_key="your_key",
    fi_secret_key="your_secret"
)
```

### Data Mappers

Data mappers transform your data into the format expected by evaluators:

```python
from fi.opt.datamappers import BasicDataMapper

mapper = BasicDataMapper(
    key_map={
        "output": "generated_output",  # Maps generator output
        "input": "question",            # Maps from dataset
        "ground_truth": "answer"        # Maps from dataset
    }
)
```

## Optimization Algorithms

### Bayesian Search

Uses Optuna for intelligent hyperparameter optimization of few-shot example selection.

```python
from fi.opt.optimizers import BayesianSearchOptimizer

optimizer = BayesianSearchOptimizer(
    min_examples=2,
    max_examples=8,
    n_trials=20,
    inference_model_name="gpt-4o-mini",
    teacher_model_name="gpt-5"
)
```

### ProTeGi

Gradient-based prompt optimization that iteratively refines prompts based on error analysis.

```python
from fi.opt.optimizers import ProTeGi
from fi.opt.generators import LiteLLMGenerator

teacher = LiteLLMGenerator(model="gpt-5", prompt_template="{prompt}")
optimizer = ProTeGi(
    teacher_generator=teacher,
    num_gradients=4,
    beam_size=4
)
```

### Meta-Prompt

Uses a powerful teacher model to analyze performance and rewrite prompts.

```python
from fi.opt.optimizers import MetaPromptOptimizer

optimizer = MetaPromptOptimizer(
    teacher_generator=teacher,
    num_rounds=5
)
```

### GEPA

Evolutionary optimization using the GEPA library.

```python
from fi.opt.optimizers import GEPAOptimizer

optimizer = GEPAOptimizer(
    reflection_model="gpt-5",
    generator_model="gpt-4o-mini"
)
```

### PromptWizard

Multi-stage optimization with mutation, critique, and refinement.

```python
from fi.opt.optimizers import PromptWizardOptimizer

optimizer = PromptWizardOptimizer(
    teacher_generator=teacher,
    mutate_rounds=3,
    refine_iterations=2
)
```

### Random Search

Simple baseline that tries random prompt variations.

```python
from fi.opt.optimizers import RandomSearchOptimizer

optimizer = RandomSearchOptimizer(
    generator=generator,
    teacher_model="gpt-5",
    num_variations=5
)
```

## Advanced Usage

### Custom Evaluation Metrics

Create custom heuristic metrics by extending `BaseMetric`:

```python
from fi.evals.metrics.base_metric import BaseMetric

class CustomMetric(BaseMetric):
    @property
    def metric_name():
        return "your custom metric name"
    def compute_one(self, inputs):
        # Your evaluation logic here
        pass
```

### Logging Configuration

```python
from fi.opt.utils import setup_logging
import logging

setup_logging(
    level=logging.INFO,
    log_to_console=True,
    log_to_file=True,
    log_file="optimization.log"
)
```

### Custom Prompt Builders

For complex prompt construction:

```python
def custom_prompt_builder(base_prompt: str, few_shot_examples: List[str]) -> str:
    examples = "\n\n".join(few_shot_examples)
    return f"{base_prompt}\n\nExamples:\n{examples}"

optimizer = BayesianSearchOptimizer(
    prompt_builder=custom_prompt_builder
)
```

## Environment Variables

Set up your API keys:

```bash
export OPENAI_API_KEY="your_openai_key" # Or GEMINI_API_KEY if using gemini models etc.
export FI_API_KEY="your_futureagi_key"
export FI_SECRET_KEY="your_futureagi_secret"
```

Or use a `.env` file:

```
OPENAI_API_KEY=your_openai_key
FI_API_KEY=your_futureagi_key
FI_SECRET_KEY=your_futureagi_secret
```

## Examples

Check out the `examples/FutureAGI_Agent_Optimizer.ipynb` notebook for a complete example!

## Requirements

- Python >= 3.10
- ai-evaluation >= 0.1.9
- gepa >= 0.0.17
- litellm >= 1.35.2
- optuna >= 3.6.1
- Additional dependencies listed in `pyproject.toml`

## Project Structure

```
src/fi/opt/
├── base/              # Abstract base classes
├── datamappers/       # Data transformation utilities
├── generators/        # LLM generator implementations
├── optimizers/        # Optimization algorithms
├── utils/            # Helper utilities
└── types.py          # Type definitions
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Support

For questions and support, please contact: support@futureagi.com

---

Built with ❤️ by Future AGI
