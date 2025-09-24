from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from pydantic import BaseModel

class Generator(ABC):
    """
    Base Abstract Generator Class.

    The Generator's role is to produce new candidate prompts that the Optimizer
    will then test using an Evaluator. This class is designed to be modular,
    allowing for different strategies for prompt generation.

    For example, a Generator could:
    - Use a powerful LLM to generate creative variations of a prompt.
    - Apply specific, rule-based transformations (e.g., rephrasing, adding examples).
    - Use simple templating to fill in variables.

    This abstract class defines the interface that all Generators must implement.
    """

    def __init__(self, gen_name: str, inputs: Type[BaseModel], outputs: Type[BaseModel]):
        """
        Initializes the Generator.

        Args:
            gen_name: The name of the generator for identification.
            inputs: A Pydantic model defining the expected input structure for generation.
            outputs: A Pydantic model defining the structure of the generated output.
        """
        self.gen_name = gen_name
        self.inputs = inputs
        self.outputs = outputs

    @abstractmethod
    def generate(self, inputs: BaseModel, critique: str, **kwargs) -> BaseModel:
        """
        Generates new prompts or data based on the inputs and a critique.

        Args:
            inputs: An instance of the Pydantic model defined in `self.inputs`.
                    This contains the data needed to generate new prompts.
            critique: Feedback from the evaluator on the previous generation.
            **kwargs: Allows for passing additional, implementation-specific
                      parameters to the generator.

        Returns:
            An instance of the Pydantic model defined in `self.outputs`,
            containing the generated prompts or data.
        """
        pass
