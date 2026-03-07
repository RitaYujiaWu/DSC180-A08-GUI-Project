from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedCase:
    trajectory_id: str
    label: str  # "success" or "failure"
    task: Optional[str] = None
    summary: Optional[str] = None
    score: Optional[float] = None
    raw_content: Optional[Dict[str, Any]] = None


@dataclass
class CandidateAction:
    text: str
    normalized_text: Optional[str] = None
    source: Optional[str] = None  # generated / fallback / replayed


@dataclass
class CandidateScore:
    candidate: CandidateAction
    success_score: float
    failure_score: float
    final_score: float
    reasoning: Optional[str] = None
    supporting_successes: List[RetrievedCase] = field(default_factory=list)
    supporting_failures: List[RetrievedCase] = field(default_factory=list)


@dataclass
class PlannerOutput:
    selected_action_text: Optional[str]
    candidate_scores: List[CandidateScore] = field(default_factory=list)
    retrieved_successes: List[RetrievedCase] = field(default_factory=list)
    retrieved_failures: List[RetrievedCase] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)