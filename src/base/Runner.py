from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Runner(ABC):
    """
    Base Abstract Class for a Prompt Runner.
    Its role is to execute a prompt against a dataset and return the outputs.
    """

    @abstractmethod
    def run(self, prompt: str, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Runs the prompt on the dataset.

        Args:
            prompt: The prompt to execute.
            dataset: A list of data points to run the prompt on.

        Returns:
            A list of outputs, one for each data point.
        """
        pass
