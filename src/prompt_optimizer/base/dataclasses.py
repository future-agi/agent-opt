from dataclasses import dataclass, field
from typing import Any, List, Dict
from .Generator import BaseGenerator

@dataclass
class Feedback:
    """A standardized object to carry evaluation results."""
    score: float
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    """A standardized object to hold the results of an optimization run."""
    best_generator: BaseGenerator
    history: List[Dict[str, Any]] = field(default_factory=list)
    final_score: float = 0.0
