from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .Generator import BaseGenerator
from .dataclasses import OptimizationResult
from .mapper import DataMapper

class BaseStrategy(ABC):
    """
    Abstract base class for all optimization strategies.
    This class contains the core logic for the optimization algorithm.
    """

    @abstractmethod
    def optimize(
        self,
        generator: BaseGenerator,
        evaluator: Any,
        data_mapper: DataMapper,
        trainset: List[Dict[str, Any]],
        valset: List[Dict[str, Any]]
    ) -> OptimizationResult:
        """
        Runs the optimization loop.

        Args:
            generator: The generator to be optimized.
            evaluator: The user-provided evaluator object.
            data_mapper: The user-provided data mapper.
            trainset: A list of training examples.
            valset: A list of validation examples.

        Returns:
            An OptimizationResult object.
        """
        pass
