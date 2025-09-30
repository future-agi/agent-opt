import json
import unittest
from unittest.mock import MagicMock, patch

from prompt_optimizer.optimizers.bayesian_search import BayesianSearchOptimizer
from prompt_optimizer.datamappers import BasicDataMapper
from prompt_optimizer.base.evaluator import Evaluator
from prompt_optimizer.types import EvaluationResult


class DummyEvaluator(Evaluator):
    def __init__(self):
        # Bypass parent init
        self._strategy = "dummy"

    def evaluate(self, inputs):
        # Return deterministic scores based on hash of input
        results = []
        for single in inputs:
            content = json.dumps(single, sort_keys=True)
            score = (hash(content) % 100) / 100.0
            results.append(EvaluationResult(score=score, reason="ok"))
        return results


class TestBayesianSearchOptimizer(unittest.TestCase):
    def setUp(self):
        self.dataset = [
            {"prompt": f"p{i}", "story": f"s{i}", "label": i} for i in range(10)
        ]
        self.data_mapper = BasicDataMapper({"input": "prompt", "output": "generated_output"})
        self.evaluator = DummyEvaluator()

    @patch("prompt_optimizer.generators.litellm.LiteLLMGenerator.generate")
    def test_basic_optimize_flow(self, mock_generate):
        mock_generate.side_effect = lambda vars, **kw: f"out:{vars}"

        opt = BayesianSearchOptimizer(n_trials=3, min_examples=1, max_examples=3)
        res = opt.optimize(
            evaluator=self.evaluator,
            data_mapper=self.data_mapper,
            dataset=self.dataset,
            initial_prompts=["Base: {prompt}"],
        )

        self.assertIsNotNone(res)
        self.assertTrue(res.final_score >= 0.0)
        self.assertTrue(len(res.history) >= 1)
        self.assertTrue(hasattr(res.best_generator, "get_prompt_template"))

    def test_formatting_with_fields_and_aliases(self):
        opt = BayesianSearchOptimizer(
            example_template_fields=["prompt", "story"],
            field_aliases={"prompt": "Input", "story": "Output"},
        )
        s = opt._format_example({"prompt": "p", "story": "s", "x": 1})
        self.assertIn("Input: p", s)
        self.assertIn("Output: s", s)

    def test_formatting_with_template(self):
        opt = BayesianSearchOptimizer(example_template="Q: {prompt}\nA: {story}")
        s = opt._format_example({"prompt": "p", "story": "s"})
        self.assertEqual(s, "Q: p\nA: s")

    def test_select_eval_subset(self):
        opt = BayesianSearchOptimizer(eval_subset_size=3, eval_subset_strategy="first")
        subset = opt._select_eval_subset(self.dataset)
        self.assertEqual(len(subset), 3)
        self.assertEqual(subset[0]["prompt"], "p0")

    @patch("prompt_optimizer.generators.litellm.LiteLLMGenerator.generate")
    def test_teacher_infer_template(self, mock_generate):
        mock_generate.return_value = json.dumps({"example_template": "Q: {prompt} -> {story}"})
        opt = BayesianSearchOptimizer(
            infer_example_template_via_teacher=True,
            template_infer_n_samples=5,
        )
        tmpl = opt._infer_example_template(self.dataset)
        self.assertEqual(tmpl, "Q: {prompt} -> {story}")

    def test_build_prompt_append(self):
        opt = BayesianSearchOptimizer(few_shot_position="append")
        prompt = opt._build_prompt("Base", "FS")
        self.assertTrue(prompt.startswith("Base"))
        self.assertIn("FS", prompt)

    def test_build_prompt_prepend(self):
        opt = BayesianSearchOptimizer(few_shot_position="prepend")
        prompt = opt._build_prompt("Base", "FS")
        self.assertTrue(prompt.startswith("FS"))
        self.assertIn("Base", prompt)


if __name__ == "__main__":
    unittest.main()


