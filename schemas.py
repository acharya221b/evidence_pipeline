# schemas.py
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

@dataclass
class EvidenceNode:
    eid: str
    route: str
    score01: float
    text: str
    meta: Dict[str, Any]
    trace: Optional[Dict[str, Any]] = None

class MCQZipAnswer(BaseModel):
    cop_index: str = Field(..., description="Correct option index as string, or '-1'")
    answer: str = Field(..., description="Option text if cop_index != -1 else ''")
    why_correct: str = Field(..., description="Short explanation with evidence citations")
    why_others_incorrect: Dict[str, str] = Field(..., description="Per-option why incorrect")
    evidence_used: List[str] = Field(default_factory=list, description="Evidence IDs actually used (E#)")
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Map of evidence_id -> {text, route, score, ids, trace(optional)}; ONLY for evidence_used"
    )