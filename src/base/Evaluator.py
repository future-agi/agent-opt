from abc import ABC
from abc import abstractmethod
from typing import Any, Dict, List, Type
from pydantic import BaseModel

class Evaluator(ABC):
    """
    Base Abstract Evaluator Class
    """
    def __init__(self, eval_name: str, inputs: Type[BaseModel], outputs: Type[BaseModel]):
        self.eval_name = eval_name
        self.inputs = inputs
        self.outputs = outputs
    
    @abstractmethod
    def evaluate(self, outputs: List[Dict[str, Any]], **kwargs):
        pass
