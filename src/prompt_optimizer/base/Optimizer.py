from typing import List, Dict, Any
from .Generator import BaseGenerator
from .Strategy import BaseStrategy
from .dataclasses import OptimizationResult
from .mapper import DataMapper

class Optimizer:
    """
    Orchestrates the prompt optimization process by coordinating a Generator,
    an external Evaluator, a DataMapper, and an optimization Strategy.
    """

    def __init__(
        self,
        generator: BaseGenerator,
        evaluator: Any, # The user provides their own evaluator instance
        strategy: BaseStrategy,
        data_mapper: DataMapper
    ):
        """
        Initializes the Optimizer.

        Args:
            generator: An instance of a Generator class.
            evaluator: An instance of an evaluator object.
            strategy: The optimization strategy to use.
            data_mapper: An object that maps data to the evaluator's format.
        """
        self.generator = generator
        self.evaluator = evaluator
        self.strategy = strategy
        self.data_mapper = data_mapper

    def run(
        self, 
        trainset: List[Dict[str, Any]], 
        valset: List[Dict[str, Any]]
    ) -> OptimizationResult:
        """
        Executes the optimization process using the provided strategy and data.
        """
        return self.strategy.optimize(
            self.generator,
            self.evaluator,
            self.data_mapper,
            trainset,
            valset
        )
