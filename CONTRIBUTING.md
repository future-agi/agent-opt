# Contributing to agent-opt

Thanks for your interest in contributing. `agent-opt` is part of the [Future AGI](https://github.com/future-agi/future-agi) open-source ecosystem, and we welcome bug fixes, new optimization algorithms, new evaluation adapters, docs improvements, and examples.

---

## Quick links

- 🐛 [Report a bug](https://github.com/future-agi/agent-opt/issues/new)
- ✨ [Request a feature](https://github.com/future-agi/agent-opt/issues/new)
- 🔖 [Good first issues](https://github.com/future-agi/agent-opt/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- 💬 [Join Discord](https://discord.gg/UjZ2gRT5p)
- 📖 [Optimization docs](https://docs.futureagi.com/docs/optimization)

---

## Code of Conduct & CLA

This project follows the [main repo's Contributing Guide](https://github.com/future-agi/future-agi/blob/main/CONTRIBUTING.md) — same Code of Conduct, same Contributor License Agreement. The CLA signs automatically on your first PR.

---

## Development setup

```bash
git clone https://github.com/future-agi/agent-opt
cd agent-opt
python -m venv .venv && source .venv/bin/activate
pip install -e "src[dev]"
```

Run the tests:

```bash
pytest tests/
```

---

## Pull request checklist

- [ ] Tests pass locally (`pytest tests/`)
- [ ] New optimizer? Add it under `src/fi/opt/optimizers/`, register it in `optimizers/__init__.py`, and add at least one end-to-end test.
- [ ] New metric adapter? Prefer extending `ai-evaluation` upstream first — ping on Discord if unsure.
- [ ] Docstrings on public classes and methods.
- [ ] README updated if the public API changed.

---

## Adding a new optimizer — the shape

Every optimizer subclasses `fi.opt.base.optimizer.Optimizer` and returns an `OptimizationResult`. Follow the existing ones (`bayesian_search.py`, `protegi.py`, `gepa.py`) — same constructor conventions, same `optimize(evaluator, data_mapper, dataset, initial_prompts)` signature, same result shape.

Questions about algorithm fit, API boundaries, or research references — open a Discussion or ping `@future-agi/opt-maintainers` on Discord before writing the code.
