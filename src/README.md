# agent-opt

![Company Logo](https://fi-content.s3.ap-south-1.amazonaws.com/Logo.png)

[![PyPI version](https://badge.fury.io/py/agent-opt.svg)](https://badge.fury.io/py/agent-opt)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XjOUlEwUk-S0nl8dBb16iZHgr-XpXsnp?usp=sharing)

**Automated workflow optimization using state-of-the-art optimization algorithms and evaluation metrics.**

Built by [Future AGI](https://futureagi.com) - Empowering GenAI Teams with Advanced Performance Management

---

## Overview

**Future AGI** provides a cutting-edge platform designed to help GenAI teams maintain peak model accuracy in production environments. Our solution is purpose-built, scalable, and delivers results 10x faster than traditional methods.

**agent-opt** is a comprehensive Python SDK for optimizing prompts through iterative refinement. Powered by state-of-the-art optimization algorithms and flexible evaluation strategies from our [`ai-evaluation`](https://github.com/future-agi/ai-evaluation) library, agent-opt helps you discover the best prompts for your LLM workflows automatically.

### Key Features

- **Simplified GenAI Performance Management**: Streamline your workflow and focus on developing cutting-edge AI models
- **Multiple Optimization Algorithms**: Choose from 6 different optimization strategies (Random Search, Bayesian Search, ProTeGi, Meta-Prompt, PromptWizard, GEPA)
- **Flexible Evaluation**: All evaluation backends powered by FutureAGI's `ai-evaluation` library - score outputs without human-in-the-loop or ground truth
- **Easy Integration**: Works with any LLM through LiteLLM
- **Advanced Error Analytics**: Gain ready-to-use insights with comprehensive error tagging and segmentation
- **Extensible Architecture**: Clean abstractions for custom optimizers, generators, and evaluators

---

## Installation

```bash
pip install agent-opt
```

**Requirements:**

- Python >= 3.10
- ai-evaluation >= 0.1.9
- gepa >= 0.0.17
- litellm >= 1.35.2
- optuna >= 3.6.1

---

## Quick Start

```python
from fi.opt.generators import LiteLLMGenerator
from fi.opt.optimizers import BayesianSearchOptimizer
from fi.opt.datamappers import BasicDataMapper
from fi.opt.base.evaluator import Evaluator
from fi.evals.metrics import BLEUScore

# 1. Set up your dataset
dataset = [
    {
        "context": "Paris is the capital of France",
        "question": "What is the capital of France?",
        "answer": "Paris"
    },
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
    teacher_model_name="gpt-4o",
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

---

## Optimization Algorithms

### Available Optimizers

| Algorithm           | Description                                   | Best For                    |
| ------------------- | --------------------------------------------- | --------------------------- |
| **Random Search**   | Simple random variations                      | Quick baselines             |
| **Bayesian Search** | Intelligent hyperparameter tuning with Optuna | Few-shot optimization       |
| **ProTeGi**         | Textual gradients for iterative improvement   | Gradient-based refinement   |
| **Meta-Prompt**     | Uses powerful models to analyze and rewrite   | Teacher-driven optimization |
| **PromptWizard**    | Mutation, critique, and refinement pipeline   | Multi-stage refinement      |
| **GEPA**            | Genetic Pareto evolutionary optimization      | Complex solution spaces     |

### Example: Bayesian Search

```python
from fi.opt.optimizers import BayesianSearchOptimizer

optimizer = BayesianSearchOptimizer(
    min_examples=2,
    max_examples=8,
    n_trials=20,
    inference_model_name="gpt-4o-mini",
    teacher_model_name="gpt-4o"
)
```

---

## Evaluation Options

### Heuristic Metrics

```python
from fi.opt.base.evaluator import Evaluator
from fi.evals.metrics import BLEUScore

evaluator = Evaluator(metric=BLEUScore())
```

### LLM-as-a-Judge

```python
from fi.evals.llm import LiteLLMProvider
from fi.evals.metrics import CustomLLMJudge

provider = LiteLLMProvider()

correctness_judge_config = {
    "name": "correctness_judge",
    "grading_criteria": '''You are evaluating an AI's answer to a question.
    The score must be 1.0 if the 'response' is semantically equivalent to the
    'expected_response' (the ground truth). The score should be 0.0 if incorrect.
    Partial credit is acceptable.'''
}

correctness_judge = CustomLLMJudge(
    provider=provider,
    config=correctness_judge_config,
    model="gemini/gemini-2.5-flash",
    temperature=0.4
)
evaluator = Evaluator(metric=correctness_judge)
```

### FutureAGI Platform

Access 50+ pre-built evaluation templates:

```python
evaluator = Evaluator(
    eval_template="summary_quality",
    eval_model_name="turing_flash",
    fi_api_key="your_key",
    fi_secret_key="your_secret"
)
```

---

## Environment Setup

Set up your API keys:

```bash
export OPENAI_API_KEY="your_openai_key"
export FI_API_KEY="your_futureagi_key"
export FI_SECRET_KEY="your_futureagi_secret"
```

---

## Documentation & Resources

- **Documentation**: [https://docs.futureagi.com](https://docs.futureagi.com)
- **Platform**: [https://app.futureagi.com](https://app.futureagi.com)
- **GitHub**: [https://github.com/future-agi/agent-opt](https://github.com/future-agi/agent-opt)

---

## Related Projects

- **ai-evaluation**: [Comprehensive LLM evaluation framework](https://github.com/future-agi/ai-evaluation)
- **traceAI**: [Add tracing & observability](https://github.com/future-agi/traceAI)

---

## Support

For questions and support:

- **Email**: support@futureagi.com
- **Documentation**: [docs.futureagi.com](https://docs.futureagi.com)

---

Built with ❤️ by [Future AGI](https://futureagi.com)
