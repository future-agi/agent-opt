from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDataMapper(ABC):
    """
    Abstract base class for Data Mappers. A Data Mapper is responsible for
    transforming data into the format expected by an evaluator.
    """

    @abstractmethod
    def map(
        self, generated_output: str, ground_truth_example: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Maps the generated output and a ground truth example to the format
        expected by the evaluator's `inputs`.

        Args:
            generated_output: The output from the Generator.
            ground_truth_example: A single example from the dataset.

        Returns:
            A dictionary formatted for the evaluator's `inputs` argument.
        """
        pass


class DataMapper:
    """
    A Data Mapper that transforms data into the format expected by an evaluator.
    The user provides a mapping dictionary to define how to structure the
    evaluator's `inputs`.
    """

    def __init__(self, key_map: Dict[str, str]):
        """
        Initializes the Data Mapper.

        Args:
            key_map: A dictionary that defines the mapping.
                     Example: {"output": "generated_story", "input": "prompt"}
                     This would map the generator's output to the "generated_story"
                     key and the ground truth "prompt" field to the "input" key.
        """
        self.key_map = key_map

    def map(
        self, generated_output: str, ground_truth_example: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Maps the data using the key_map provided during initialization.

        Returns:
            A dictionary formatted for the evaluator's `inputs` argument.
        """
        mapped_data = {}
        for new_key, original_key in self.key_map.items():
            if original_key == "generated_output":
                mapped_data[new_key] = generated_output
            elif original_key in ground_truth_example:
                mapped_data[new_key] = ground_truth_example[original_key]

        return mapped_data
