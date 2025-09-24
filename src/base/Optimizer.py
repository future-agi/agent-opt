from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .Generator import Generator
from .Evaluator import Evaluator
from .Runner import Runner

class Optimizer(ABC):
    """
    Base Abstract Optimizer Class.

    The Optimizer orchestrates the prompt optimization process. It uses a
    Generator to create new prompts, a Runner to execute them against a dataset,
    and an Evaluator to score the results. Based on the scores and feedback,
    it guides the Generator in subsequent iterations.
    """

    def __init__(self, generator: Generator, evaluator: Evaluator, runner: Runner):
        """
        Initializes the Optimizer.

        Args:
            generator: An instance of a Generator class to produce new prompts.
            evaluator: An instance of an Evaluator class to score prompts.
            runner: An instance of a Runner class to execute prompts.
        """
        self.generator = generator
        self.evaluator = evaluator
        self.runner = runner

    @abstractmethod
    def optimize(self, initial_prompt: str, dataset: List[Dict[str, Any]], max_iterations: int) -> str:
        """
        Runs the optimization process.

        Args:
            initial_prompt: The starting prompt for the optimization.
            dataset: A dataset to use for evaluation.
            max_iterations: The maximum number of optimization iterations.

        Returns:
            The best prompt found during optimization.
        """
        pass
