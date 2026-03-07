from __future__ import annotations

import re
from typing import Any, List, Optional

from .prompt_builder import build_candidate_scoring_prompt
from .schemas import CandidateAction, CandidateScore, RetrievedCase


class CandidateScorer:
    """
    Score each candidate using contrastive success/failure memories.
    Supports:
    - heuristic scoring
    - LLM-based scoring
    """

    def __init__(
        self,
        llm: Any = None,
        score_method: str = "heuristic",
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        self.llm = llm
        self.score_method = score_method
        self.alpha = alpha
        self.beta = beta

    def score_candidates(
        self,
        candidates: List[CandidateAction],
        success_cases: List[RetrievedCase],
        failure_cases: List[RetrievedCase],
        intent: str,
        current_url: Optional[str] = None,
        observation_summary: Optional[str] = None,
    ) -> List[CandidateScore]:
        scores: List[CandidateScore] = []

        for candidate in candidates:
            if self.score_method == "llm":
                score = self._score_candidate_llm(
                    candidate=candidate,
                    success_cases=success_cases,
                    failure_cases=failure_cases,
                    intent=intent,
                    current_url=current_url,
                    observation_summary=observation_summary,
                )
            else:
                score = self._score_candidate_heuristic(
                    candidate=candidate,
                    success_cases=success_cases,
                    failure_cases=failure_cases,
                )
            scores.append(score)

        return scores

    def select_best(self, scores: List[CandidateScore]) -> Optional[CandidateScore]:
        if not scores:
            return None
        return max(scores, key=lambda s: s.final_score)

    def _score_candidate_heuristic(
        self,
        candidate: CandidateAction,
        success_cases: List[RetrievedCase],
        failure_cases: List[RetrievedCase],
    ) -> CandidateScore:
        candidate_text = (candidate.normalized_text or candidate.text.lower()).strip()

        success_score = self._keyword_overlap_score(candidate_text, success_cases)
        failure_score = self._keyword_overlap_score(candidate_text, failure_cases)
        final_score = self.alpha * success_score - self.beta * failure_score

        reasoning = (
            f"Heuristic score based on overlap with success ({success_score:.2f}) "
            f"and failure ({failure_score:.2f}) precedents."
        )

        return CandidateScore(
            candidate=candidate,
            success_score=success_score,
            failure_score=failure_score,
            final_score=final_score,
            reasoning=reasoning,
            supporting_successes=success_cases,
            supporting_failures=failure_cases,
        )

    def _score_candidate_llm(
        self,
        candidate: CandidateAction,
        success_cases: List[RetrievedCase],
        failure_cases: List[RetrievedCase],
        intent: str,
        current_url: Optional[str] = None,
        observation_summary: Optional[str] = None,
    ) -> CandidateScore:
        prompt = build_candidate_scoring_prompt(
            intent=intent,
            candidate_action=candidate.text,
            success_cases=success_cases,
            failure_cases=failure_cases,
            current_url=current_url,
            observation_summary=observation_summary,
        )

        raw_text = self._call_llm(prompt)

        success_score = self._extract_number(raw_text, "SUCCESS_ALIGNMENT", default=0.0)
        failure_score = self._extract_number(raw_text, "FAILURE_RISK", default=0.0)
        final_score = self._extract_number(
            raw_text,
            "FINAL_SCORE",
            default=self.alpha * success_score - self.beta * failure_score,
        )
        reasoning = self._extract_reasoning(raw_text)

        return CandidateScore(
            candidate=candidate,
            success_score=success_score,
            failure_score=failure_score,
            final_score=final_score,
            reasoning=reasoning,
            supporting_successes=success_cases,
            supporting_failures=failure_cases,
        )

    def _keyword_overlap_score(self, candidate_text: str, cases: List[RetrievedCase]) -> float:
        if not candidate_text or not cases:
            return 0.0

        candidate_tokens = set(self._tokenize(candidate_text))
        if not candidate_tokens:
            return 0.0

        total = 0.0
        counted = 0

        for case in cases:
            base_text = f"{case.task or ''} {case.summary or ''}".strip().lower()
            case_tokens = set(self._tokenize(base_text))
            if not case_tokens:
                continue
            overlap = len(candidate_tokens & case_tokens) / max(1, len(candidate_tokens))
            total += overlap
            counted += 1

        if counted == 0:
            return 0.0
        return 10.0 * (total / counted)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def _call_llm(self, prompt: str) -> str:
        if self.llm is None:
            return ""

        try:
            result = self.llm(prompt)
            if isinstance(result, str):
                return result
        except Exception:
            pass

        for method_name in ("generate", "chat", "complete", "invoke"):
            method = getattr(self.llm, method_name, None)
            if method is None:
                continue
            try:
                result = method(prompt)
                if isinstance(result, str):
                    return result
                if hasattr(result, "text"):
                    return str(result.text)
                if isinstance(result, dict):
                    for key in ("text", "output", "response", "content"):
                        if key in result:
                            return str(result[key])
            except Exception:
                continue

        return ""

    def _extract_number(self, text: str, key: str, default: float = 0.0) -> float:
        if not text:
            return default
        pattern = rf"{re.escape(key)}\s*:\s*(-?\d+(?:\.\d+)?)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return default
        try:
            return float(match.group(1))
        except Exception:
            return default

    def _extract_reasoning(self, text: str) -> str:
        if not text:
            return "No scorer reasoning returned."
        match = re.search(r"REASONING\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()[:500]