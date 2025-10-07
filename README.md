<div align="center">

# 🎯 Agent Opt

**Automated Workflow Optimization with State-of-the-Art Algorithms**  
Built by [Future AGI](https://futureagi.com) | [Docs](https://docs.futureagi.com) | [Platform](https://app.futureagi.com)

---

### 🚀 Try it Now

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XjOUlEwUk-S0nl8dBb16iZHgr-XpXsnp?usp=sharing)

</div>

---

## 🚀 Overview

**agent-opt** is a comprehensive Python SDK for optimizing prompts through iterative refinement. Powered by state-of-the-art optimization algorithms and flexible evaluation strategies from our [`ai-evaluation`](https://github.com/future-agi/ai-evaluation) library, agent-opt helps you discover the best prompts for your LLM workflows automatically.

- 🧬 **Smart Optimization**: 6 proven algorithms from random search to genetic evolution
- 📊 **Flexible Evaluation**: Heuristic metrics, LLM-as-a-judge, and platform integration
- ⚡ **Easy Integration**: Works with any LLM through LiteLLM
- 🔧 **Extensible Design**: Clean abstractions for custom optimizers and evaluators

---

## 🎨 Features

### 🧬 Multiple Optimization Algorithms

Choose from 6 battle-tested optimization strategies:

| Algorithm           | Best For                    | Key Feature                                   |
| ------------------- | --------------------------- | --------------------------------------------- |
| **Random Search**   | Quick baselines             | Simple random variations                      |
| **Bayesian Search** | Few-shot optimization       | Intelligent hyperparameter tuning with Optuna |
| **ProTeGi**         | Gradient-based refinement   | Textual gradients for iterative improvement   |
| **Meta-Prompt**     | Teacher-driven optimization | Uses powerful models to analyze and rewrite   |
| **PromptWizard**    | Multi-stage refinement      | Mutation, critique, and refinement pipeline   |
| **GEPA**            | Complex solution spaces     | Genetic Pareto evolutionary optimization      |

### 📊 Flexible Evaluation

All evaluation backends powered by FutureAGI's [`ai-evaluation`](https://github.com/future-agi/ai-evaluation) library:

- ✅ **Heuristic Metrics**: BLEU, ROUGE, embedding similarity, and more
- 🧠 **LLM-as-a-Judge**: Custom criteria with any LLM provider
- 🎯 **FutureAGI Platform**: 50+ pre-built evaluation templates
- 🔌 **Custom Metrics**: Build your own evaluation logic

### 🔧 Easy Integration

- Works with **any LLM** through LiteLLM (OpenAI, Anthropic, Google, etc.)
- Simple Python API with sensible defaults
- Comprehensive logging and progress tracking
- Clean separation of concerns

---

## 📦 Installation

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

## 🧑‍💻 Quick Start

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

## 🏗️ Core Components

### 🤖 Generators

Generators execute prompts and return responses. Use `LiteLLMGenerator` for seamless integration with any LLM provider.

```python
from fi.opt.generators import LiteLLMGenerator

generator = LiteLLMGenerator(
    model="gpt-4o-mini",
    prompt_template="Summarize this text: {text}"
)
```

---

### 📊 Evaluators

Evaluators score generated outputs using various strategies:

#### **Heuristic Metrics**

```python
from fi.opt.base.evaluator import Evaluator
from fi.evals.metrics import BLEUScore

evaluator = Evaluator(metric=BLEUScore())
```

#### **LLM-as-a-Judge**

```python
from fi.evals.llm import LiteLLMProvider
from fi.evals.metrics import CustomLLMJudge

# LLM provider used by the judge
provider = LiteLLMProvider()

# Create custom LLM judge metric
correctness_judge_config = {
    "name": "correctness_judge",
    "grading_criteria": '''You are evaluating an AI's answer to a question.
    The score must be 1.0 if the 'response' is semantically equivalent to the
    'expected_response' (the ground truth). The score should be 0.0 if incorrect.
    Partial credit is acceptable.'''
}

# Instantiate the judge and pass to evaluator
correctness_judge = CustomLLMJudge(
    provider=provider,
    config=correctness_judge_config,
    model="gemini/gemini-2.5-flash",
    temperature=0.4
)
evaluator = Evaluator(metric=correctness_judge)
```

#### **FutureAGI Platform**

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

### 🗺️ Data Mappers

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

---

## ⚙️ Optimization Algorithms

### 🔍 Bayesian Search

Uses Optuna for intelligent hyperparameter optimization of few-shot example selection.

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

**Best for:** Few-shot prompt optimization with automatic example selection

---

### 🎯 ProTeGi

Gradient-based prompt optimization that iteratively refines prompts through error analysis.

```python
from fi.opt.optimizers import ProTeGi
from fi.opt.generators import LiteLLMGenerator

teacher = LiteLLMGenerator(
    model="gpt-4o",
    prompt_template="{prompt}"
)
optimizer = ProTeGi(
    teacher_generator=teacher,
    num_gradients=4,
    beam_size=4
)
```

**Best for:** Iterative refinement with textual gradients

---

### 🧠 Meta-Prompt

Uses a powerful teacher model to analyze performance and rewrite prompts.

```python
from fi.opt.optimizers import MetaPromptOptimizer

optimizer = MetaPromptOptimizer(
    teacher_generator=teacher,
    num_rounds=5
)
```

**Best for:** Leveraging powerful models for prompt refinement

---

### 🧬 GEPA (Genetic Pareto)

Evolutionary optimization using the GEPA library for complex solution spaces.

```python
from fi.opt.optimizers import GEPAOptimizer

optimizer = GEPAOptimizer(
    reflection_model="gpt-5",
    generator_model="gpt-4o-mini"
)
```

**Best for:** Multi-objective optimization with genetic algorithms

---

### 🪄 PromptWizard

Multi-stage optimization with mutation, critique, and refinement.

```python
from fi.opt.optimizers import PromptWizardOptimizer

optimizer = PromptWizardOptimizer(
    teacher_generator=teacher,
    mutate_rounds=3,
    refine_iterations=2
)
```

**Best for:** Comprehensive multi-phase optimization pipeline

---

### 🎲 Random Search

Simple baseline that tries random prompt variations.

```python
from fi.opt.optimizers import RandomSearchOptimizer

optimizer = RandomSearchOptimizer(
    generator=generator,
    teacher_model="gpt-4o",
    num_variations=5
)
```

**Best for:** Quick baselines and sanity checks

---

## 🔧 Advanced Usage

### 🎨 Custom Evaluation Metrics

Create custom heuristic metrics by extending `BaseMetric`:

```python
from fi.evals.metrics.base_metric import BaseMetric

class CustomMetric(BaseMetric):
    @property
    def metric_name(self):
        return "your_custom_metric"

    def compute_one(self, inputs):
        # Your evaluation logic here
        score = your_scoring_logic(inputs)
        return score
```

---

### 📝 Logging Configuration

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

---

### 🏗️ Custom Prompt Builders

For complex prompt construction:

```python
def custom_prompt_builder(base_prompt: str, few_shot_examples: List[str]) -> str:
    examples = "\n\n".join(few_shot_examples)
    return f"{base_prompt}\n\nExamples:\n{examples}"

optimizer = BayesianSearchOptimizer(
    prompt_builder=custom_prompt_builder
)
```

---

## 🔑 Environment Setup

### API Keys

Set up your API keys for LLM providers and FutureAGI:

```bash
export OPENAI_API_KEY="your_openai_key"
export GEMINI_API_KEY="your_gemini_key"  # If using Gemini
export FI_API_KEY="your_futureagi_key"
export FI_SECRET_KEY="your_futureagi_secret"
```

Or use a `.env` file:

```
OPENAI_API_KEY=your_openai_key
FI_API_KEY=your_futureagi_key
FI_SECRET_KEY=your_futureagi_secret
```

---

## 📚 Examples & Tutorials

🎯 **Complete Example**: Check out `examples/FutureAGI_Agent_Optimizer.ipynb` for a comprehensive walkthrough!

---

## 📁 Project Structure

```
src/fi/opt/
├── base/              # Abstract base classes
├── datamappers/       # Data transformation utilities
├── generators/        # LLM generator implementations
├── optimizers/        # Optimization algorithms
├── utils/             # Helper utilities
└── types.py           # Type definitions
```

---

## 🔌 Related Projects

- 🧪 [ai-evaluation](https://github.com/future-agi/ai-evaluation): Comprehensive LLM evaluation framework with 50+ metrics
- 🚦 [traceAI](https://github.com/future-agi/traceAI): Add tracing & observability to your optimized workflows

---

## 🗺️ Roadmap

- [x] **Core Optimization Algorithms**
- [x] **ai-evaluation Integration**
- [x] **LiteLLM Support**
- [x] **Bayesian Optimization**
- [x] **ProTeGi & Meta-Prompt**
- [x] **GEPA Integration**

---

## 🤝 Contributing

We welcome contributions! To report issues, suggest features, or contribute improvements:

1. Open a [GitHub issue](https://github.com/future-agi/agent-opt/issues)
2. Submit a pull request
3. Join our community discussions

---

## 💬 Support

For questions and support:

📧 **Email**: support@futureagi.com  
📚 **Documentation**: [docs.futureagi.com](https://docs.futureagi.com)  
🌐 **Platform**: [app.futureagi.com](https://app.futureagi.com)

---

<div align="center">

Built with ❤️ by [Future AGI](https://futureagi.com)

</div>
