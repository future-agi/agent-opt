from typing import Any, List, Dict
from .Generator import BaseGenerator
from pydantic import BaseModel, Field


class Feedback(BaseModel):
    """A standardized object to carry evaluation results."""

    score: float
    reason: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OptimizationResult(BaseModel):
    """A standardized object to hold the results of an optimization run."""

    best_generator: BaseGenerator
    history: List[Dict[str, Any]] = Field(default_factory=list)
    final_score: float = 0.0
