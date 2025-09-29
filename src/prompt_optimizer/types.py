from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
from .base.base_generator import BaseGenerator


class Feedback(BaseModel):
    """Output model to carry evaluation results."""

    score: float
    reason: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMMessage(BaseModel):
    """Every message sent and received by the LLM MUST follow this format."""

    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[str] = None
    tool_call_id: Optional[str] = None


class OptimizationResult(BaseModel):
    """Output Model to hold the results of an optimization run."""

    best_generator: BaseGenerator
    message_history: List[Dict[str, str]]
    final_score: float = 0.0

    model_config = {"arbitrary_types_allowed": True}
