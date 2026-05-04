<div align="center">

<a href="https://futureagi.com">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="Logo.png">
    <img alt="Future AGI" src="Logo.png" width="100%">
  </picture>
</a>

# agent-opt

**Close the loop: six prompt-optimization algorithms, any LLM, any metric.**

Part of the [Future AGI](https://github.com/future-agi/future-agi) open-source platform for making AI agents reliable.

<p>
  <a href="https://pypi.org/project/agent-opt/"><img src="https://img.shields.io/pypi/v/agent-opt?style=flat-square&label=pypi" alt="PyPI"></a>
  <a href="https://pypi.org/project/agent-opt/"><img src="https://img.shields.io/pypi/pyversions/agent-opt?style=flat-square" alt="Python versions"></a>
  <a href="https://github.com/future-agi/agent-opt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="Apache 2.0 License"></a>
  <a href="https://pepy.tech/project/agent-opt"><img src="https://img.shields.io/pypi/dm/agent-opt?style=flat-square&color=blue" alt="Downloads"></a>
  <a href="https://discord.gg/UjZ2gRT5p"><img src="https://img.shields.io/badge/discord-join-5865F2?style=flat-square" alt="Discord"></a>
</p>

<p>
  <a href="https://app.futureagi.com/auth/jwt/register"><b>Try Cloud (Free)</b></a> ·
  <a href="https://docs.futureagi.com/docs/optimization"><b>Docs</b></a> ·
  <a href="https://colab.research.google.com/drive/1XjOUlEwUk-S0nl8dBb16iZHgr-XpXsnp?usp=sharing"><b>Colab</b></a> ·
  <a href="https://futureagi.com/blog"><b>Blog</b></a> ·
  <a href="https://discord.gg/UjZ2gRT5p"><b>Discord</b></a> ·
  <a href="https://github.com/orgs/future-agi/discussions"><b>Discussions</b></a>
</p>

</div>

---

## Why agent-opt?

Prompts are how ambiguity sneaks into an agent. You can tweak one by hand. You can't tweak a hundred, and you definitely can't re-tweak them every time the model behind them changes. `agent-opt` does the tweaking for you: pick an algorithm, pick a metric, feed it a dataset, and it returns a prompt that beats the one you wrote.

Six algorithms, one API. Plug in any LLM via LiteLLM. Score against any of the 50+ metrics from [`ai-evaluation`](https://github.com/future-agi/ai-evaluation), or write your own. Production traces feed back in as training data.

<table>
<tr>
<td width="33%" valign="top">

### Six real algorithms
Not one toy loop with six labels. Random Search, Bayesian (Optuna), **ProTeGi** (textual gradients), Meta-Prompt, **PromptWizard** (mutate-critique-refine), and **GEPA** (evolutionary Pareto). Pick by problem shape.

</td>
<td width="33%" valign="top">

### Any model, any metric
LiteLLM under the hood, so OpenAI, Anthropic, Gemini, Bedrock, Azure, Groq, and self-hosted all just work. Score with BLEU, ROUGE, embedding similarity, LLM-as-judge, or any of 50+ [`ai-evaluation`](https://github.com/future-agi/ai-evaluation) metrics. Or write your own.

</td>
<td width="33%" valign="top">

### Built for the Future AGI loop
Optimize against traces captured by [`traceAI`](https://github.com/future-agi/traceAI). Score with [`ai-evaluation`](https://github.com/future-agi/ai-evaluation). Deploy the winning prompt through the [Agent Command Center](https://docs.futureagi.com/docs/command-center) gateway. One loop, on your infrastructure.

</td>
</tr>
</table>

<div align="center">
  <img alt="agent-opt — six prompt-optimization algorithms, any LLM, any metric" src="agent-opt-repo.gif" width="100%">
</div>

---

## Install

```bash
pip install agent-opt
```

**Requirements:** Python ≥ 3.10 · `ai-evaluation` ≥ 0.2.2 · `litellm` ≥ 1.80 · `optuna` ≥ 3.6 · `gepa` ≥ 0.0.17.

---

## Quickstart

Optimize a RAG prompt against BLEU in 60 seconds.

```python
from fi.opt.optimizers import BayesianSearchOptimizer
from fi.opt.datamappers import BasicDataMapper
from fi.opt.base.evaluator import Evaluator
from fi.evals.metrics import BLEUScore

dataset = [
    {"context": "Paris is the capital of France.",
     "question": "What is the capital of France?", "answer": "Paris"},
    # ... more examples
]

evaluator = Evaluator(BLEUScore())
mapper = BasicDataMapper(key_map={
    "response": "generated_output",
    "expected_response": "answer",
})

optimizer = BayesianSearchOptimizer(
    inference_model_name="gpt-4o-mini",
    teacher_model_name="gpt-4o",
    n_trials=10,
)

result = optimizer.optimize(
    evaluator=evaluator,
    data_mapper=mapper,
    dataset=dataset,
    initial_prompts=["Given the context: {context}, answer: {question}"],
)

print(f"Best score:  {result.final_score:.4f}")
print(f"Best prompt: {result.best_generator.get_prompt_template()}")
```

<sub>Full walkthrough: [`examples/FutureAGI_Agent_Optimizer.ipynb`](./examples/FutureAGI_Agent_Optimizer.ipynb) · [Open in Colab](https://colab.research.google.com/drive/1XjOUlEwUk-S0nl8dBb16iZHgr-XpXsnp?usp=sharing)</sub>

---

## The six algorithms

Each algorithm is a drop-in `optimize()` call. Swap without touching your dataset, evaluator, or data mapper.

| Algorithm | Best for | Key idea |
|---|---|---|
| **Random Search** | Baselines and sanity checks | Random prompt variations around a seed |
| **Bayesian Search** | Few-shot example selection | Optuna TPE over example subsets and ordering |
| **ProTeGi** | Iterative refinement | Textual gradients from error analysis, beam-searched |
| **Meta-Prompt** | Teacher-model rewrites | Strong teacher analyzes failures, rewrites the prompt |
| **PromptWizard** | Multi-stage pipelines | Mutate → critique → refine, N rounds |
| **GEPA** | Complex solution spaces | Genetic Pareto evolution across multiple objectives |

<details><summary>Quick snippets for each</summary>

```python
from fi.opt.optimizers import (
    RandomSearchOptimizer, BayesianSearchOptimizer,
    ProTeGi, MetaPromptOptimizer,
    PromptWizardOptimizer, GEPAOptimizer,
)
from fi.opt.generators import LiteLLMGenerator

teacher = LiteLLMGenerator(model="gpt-4o", prompt_template="{prompt}")

# Random — fastest baseline
RandomSearchOptimizer(generator=teacher, teacher_model="gpt-4o", num_variations=5)

# Bayesian — few-shot selection via Optuna
BayesianSearchOptimizer(min_examples=2, max_examples=8, n_trials=20,
                        inference_model_name="gpt-4o-mini", teacher_model_name="gpt-4o")

# ProTeGi — textual gradient refinement
ProTeGi(teacher_generator=teacher, num_gradients=4, beam_size=4)

# Meta-Prompt — teacher-driven rewrites
MetaPromptOptimizer(teacher_generator=teacher, num_rounds=5)

# PromptWizard — mutate / critique / refine
PromptWizardOptimizer(teacher_generator=teacher, mutate_rounds=3, refine_iterations=2)

# GEPA — evolutionary Pareto
GEPAOptimizer(reflection_model="gpt-5", generator_model="gpt-4o-mini")
```

</details>

---

## Core concepts

### Generators

Execute a prompt, return a response. `LiteLLMGenerator` works with every LiteLLM-supported provider.

```python
from fi.opt.generators import LiteLLMGenerator

generator = LiteLLMGenerator(
    model="gpt-4o-mini",
    prompt_template="Summarize this text: {text}",
)
```

### Evaluators

Score a generated output. Three flavors (heuristic, LLM-as-judge, and the Future AGI platform's pre-built templates), all behind one `Evaluator` API.

```python
# Heuristic
from fi.evals.metrics import BLEUScore
evaluator = Evaluator(BLEUScore())

# LLM-as-judge
from fi.evals.llm import LiteLLMProvider
from fi.evals.metrics import CustomLLMJudge

judge = CustomLLMJudge(
    provider=LiteLLMProvider(),
    config={
        "name": "correctness_judge",
        "grading_criteria": (
            "Score 1.0 if 'response' is semantically equivalent to "
            "'expected_response'. 0.0 if incorrect. Partial credit OK."
        ),
    },
    model="gemini/gemini-2.5-flash",
    temperature=0.4,
)
evaluator = Evaluator(metric=judge)

# Future AGI platform — 50+ pre-built templates
evaluator = Evaluator(
    eval_template="summary_quality",
    eval_model_name="turing_flash",
    fi_api_key="...", fi_secret_key="...",
)
```

### Data mappers

Translate your dataset's shape into the keys the evaluator expects.

```python
from fi.opt.datamappers import BasicDataMapper

mapper = BasicDataMapper(key_map={
    "output":       "generated_output",  # from the generator
    "input":        "question",          # from the dataset row
    "ground_truth": "answer",            # from the dataset row
})
```

---

## Advanced usage

### Custom heuristic metric

```python
from fi.evals.metrics.base_metric import BaseMetric

class ExactMatchWithNormalization(BaseMetric):
    @property
    def metric_name(self):
        return "exact_match_norm"

    def compute_one(self, inputs):
        return float(inputs["response"].strip().lower()
                     == inputs["expected_response"].strip().lower())
```

### Custom prompt builder (few-shot composition)

```python
def builder(base_prompt: str, few_shot: list[str]) -> str:
    return f"{base_prompt}\n\nExamples:\n" + "\n\n".join(few_shot)

BayesianSearchOptimizer(prompt_builder=builder, ...)
```

### Logging

```python
from fi.opt.utils import setup_logging
import logging

setup_logging(level=logging.INFO,
              log_to_console=True, log_to_file=True,
              log_file="optimization.log")
```

### Environment

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."        # if using Gemini
export FI_API_KEY="..."            # for Future AGI platform evaluators
export FI_SECRET_KEY="..."
```

---

## Where agent-opt fits in the Future AGI loop

**simulate → evaluate → control → monitor → optimize.** This SDK is the `optimize` step.

- [`traceAI`](https://github.com/future-agi/traceAI) captures production traces of every LLM call.
- [`ai-evaluation`](https://github.com/future-agi/ai-evaluation) scores them with 50+ metrics.
- **`agent-opt`** turns those scored traces into a better prompt.
- The [Agent Command Center](https://docs.futureagi.com/docs/command-center) ships the new prompt behind an OpenAI-compatible endpoint.

Use one SDK or all of them. Each is independently packaged and Apache 2.0-licensed.

---

## Project structure

```
src/fi/opt/
├── base/              # Abstract base classes (Evaluator, Optimizer, …)
├── datamappers/       # Dataset-shape → evaluator-key translators
├── generators/        # LiteLLM-backed LLM callers
├── optimizers/        # Random, Bayesian, ProTeGi, Meta-Prompt, PromptWizard, GEPA
├── utils/             # Logging, IO, small helpers
└── types.py           # Shared type defs
```

---

## Roadmap

<table>
<tr>
<th width="25%">Shipped</th>
<th width="25%">In progress</th>
<th width="25%">Coming up</th>
<th width="25%">Exploring</th>
</tr>
<tr valign="top">
<td>

- [x] Six algorithms (RS, Bayesian, ProTeGi, Meta-Prompt, PromptWizard, GEPA)
- [x] LiteLLM generator
- [x] `ai-evaluation` integration (heuristic + LLM-judge + platform)
- [x] Early-stopping config
- [x] GEPA iteration history

</td>
<td>

- [ ] Public OSS launch
- [ ] Async optimization loop
- [ ] Multi-objective result surface
- [ ] Trace-ingestion connector (`traceAI` → dataset)

</td>
<td>

- [ ] Prompt version control with branches
- [ ] Cost-aware optimization budgets
- [ ] Resumable runs from checkpoint
- [ ] CLI (`agent-opt optimize …`)

</td>
<td>

- [ ] Auto-tuned rubrics from human feedback
- [ ] Multi-turn dialogue optimization
- [ ] Voice-agent prompt optimization
- [ ] Federated optimization across tenants

</td>
</tr>
</table>

---


## ❓ Frequently Asked Questions

### What is agent-opt?

Agent-opt is a prompt optimization library that automatically improves your agent prompts using six different algorithms. Instead of manually tweaking prompts, you pick an algorithm, a metric, feed it a dataset, and it returns an optimized prompt that outperforms your original.

### Which optimization algorithms are available?

Six algorithms are supported:
- **Random Search**: Baseline exploration of the prompt space
- **Bayesian (Optuna)**: Efficient search using probabilistic modeling
- **ProTeGi**: Textual gradient-based optimization
- **Meta-Prompt**: Uses meta-prompts to guide optimization
- **PromptWizard**: Mutate-critique-refine cycle
- **GEPA**: Evolutionary Pareto optimization for multi-objective scenarios

### What LLM providers are supported?

Agent-opt uses LiteLLM under the hood, so it supports any provider LiteLLM supports: OpenAI, Anthropic, Gemini, Bedrock, Azure, Groq, and self-hosted models. Just configure the appropriate API keys and model names.

### How do I evaluate optimized prompts?

Agent-opt integrates with [`ai-evaluation`](https://github.com/future-agi/ai-evaluation) which provides 50+ metrics including:
- **Heuristic metrics**: BLEU, ROUGE, embedding similarity
- **LLM-as-judge**: Use an LLM to score outputs against rubrics
- **Platform metrics**: Integration with evaluation platforms

You can also write custom metrics by implementing the evaluator interface.

### Can I use agent-opt with my own dataset?

Yes. Feed any dataset in the expected format — the library maps your data through configurable data mappers. Production traces from [`traceAI`](https://github.com/future-agi/traceAI) can also feed back as training data.

### How is agent-opt different from manual prompt engineering?

Manual prompt engineering works for one or two prompts. Agent-opt is designed for when you have many prompts that need to be optimized and re-optimized as models change. It automates the trial-and-error process, systematically exploring the prompt space and returning the best configuration for your specific metric.

### Can I run agent-opt locally?

Yes. Install via pip and run locally. You can also try the cloud version for free at [app.futureagi.com](https://app.futureagi.com/auth/jwt/register). A Google Colab notebook is also available for quick experimentation.

### What Python version is required?

Agent-opt supports Python 3.9+. Check the PyPI page for the latest version requirements.

### How does the Future AGI loop work?

The complete loop is:
1. **Trace**: Capture production agent traces with [`traceAI`](https://github.com/future-agi/traceAI)
2. **Optimize**: Use agent-opt to find the best prompt for your metric
3. **Deploy**: Push the winning prompt through the [Agent Command Center](https://docs.futureagi.com/docs/command-center) gateway
4. **Evaluate**: Score results with [`ai-evaluation`](https://github.com/future-agi/ai-evaluation)
5. **Iterate**: Production traces feed back as new training data

### Where can I get help?

- **Real-time help**: [Discord](https://discord.gg/UjZ2gRT5p)
- **Discussions**: [GitHub Discussions](https://github.com/orgs/future-agi/discussions)
- **Documentation**: [docs.futureagi.com](https://docs.futureagi.com/docs/optimization)
- **Blog**: [futureagi.com/blog](https://futureagi.com/blog)
- **Support**: support@futureagi.com
- **Security**: security@futureagi.com

## Contributing

Bug fixes, new algorithms, new metrics, docs, examples: all welcome.

1. [Browse `good first issue`](https://github.com/future-agi/agent-opt/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
2. Read the [main repo Contributing Guide](https://github.com/future-agi/future-agi/blob/main/CONTRIBUTING.md) — same CLA, same workflow.
3. Say hi on [Discord](https://discord.gg/UjZ2gRT5p) or [Discussions](https://github.com/orgs/future-agi/discussions).

---

## Community & support

| | |
|---|---|
| 💬 [**Discord**](https://discord.gg/UjZ2gRT5p) | Real-time help from the team and community |
| 🗨️ [**GitHub Discussions**](https://github.com/orgs/future-agi/discussions) | Ideas, questions, roadmap input |
| 📝 [**Blog**](https://futureagi.com/blog) | Engineering & research posts |
| 📧 **support@futureagi.com** | Cloud account / billing |
| 🔐 **security@futureagi.com** | Private vulnerability disclosure — see [SECURITY.md](SECURITY.md) |

---

## License

Licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Part of the [Future AGI](https://github.com/future-agi/future-agi) open-source ecosystem.

---

<div align="center">

Built by the Future AGI team and [contributors](https://github.com/future-agi/agent-opt/graphs/contributors).

If `agent-opt` helps you ship better agents, a ⭐ helps more teams find us.

[🌐 futureagi.com](https://futureagi.com) · [📖 docs.futureagi.com](https://docs.futureagi.com) · [☁️ app.futureagi.com](https://app.futureagi.com)

</div>
