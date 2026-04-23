# Changelog

All notable changes to `agent-opt` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- (nothing yet — open a PR!)

---

## [0.1.0] — 2026-04-23

First public release on PyPI. Six prompt-optimization algorithms behind one `optimize()` API, with `ai-evaluation` for scoring and LiteLLM for generation.

### Added
- **Six optimizers** — `RandomSearchOptimizer`, `BayesianSearchOptimizer`, `ProTeGi`, `MetaPromptOptimizer`, `PromptWizardOptimizer`, `GEPAOptimizer`.
- **LiteLLM generator** (`LiteLLMGenerator`) — works with any LiteLLM-supported provider (OpenAI, Anthropic, Gemini, Bedrock, Azure, Groq, self-hosted, …).
- **Unified evaluator** (`fi.opt.base.evaluator.Evaluator`) — plug in heuristic metrics (BLEU, ROUGE, embedding similarity), custom `LLM-as-judge` rubrics, or the Future AGI platform's 50+ pre-built templates.
- **Data mapper** (`BasicDataMapper`) — key-remap dataset rows to evaluator inputs without custom glue code.
- **Early-stopping config** — `patience`, `min_delta`, and `max_evaluations` for any optimizer, via a dedicated `EarlyStoppingException`.
- **GEPA iteration history** — capture per-generation progress for post-run analysis.
- **Custom prompt builders** — pass a `prompt_builder` callable to control few-shot composition.
- **Logging helper** (`fi.opt.utils.setup_logging`) — one-call console + file logging with sensible defaults.

### Changed
- Bumped `litellm` to `1.80.0`.
- Bumped `ai-evaluation` to `>= 0.2.2`.

### Fixed
- Evaluator secret-key handling for Future AGI platform templates.
- Pass/fail output normalization for rubric-style evaluators.
- Cleaner logging and better variation parsing in `RandomSearchOptimizer`.

---

[Unreleased]: https://github.com/future-agi/agent-opt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/future-agi/agent-opt/releases/tag/v0.1.0
