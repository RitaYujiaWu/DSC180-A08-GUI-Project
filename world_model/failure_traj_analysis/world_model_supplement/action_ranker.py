from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .contrastive_memory import ContrastiveTrajectoryStore
from .utils import coerce_action_history_text


@dataclass
class ActionEvidence:
    action: Any
    success_scores: List[float]
    failure_scores: List[float]
    success_evidence: float
    failure_evidence: float
    margin: float
    gated: bool
    gate_reason: str


class ActionRanker:
    """
    Ranks candidate actions using contrastive evidence from success/failure memories.

    Evidence:
      - success_evidence = max(scores_success[:k]) OR mean(topk)
      - failure_evidence = max(scores_failure[:k]) OR mean(topk)
      - margin = success_evidence - failure_evidence

    Confidence gate:
      - If evidence is weak (both below min_sim) OR margin below min_margin,
        don't over-trust memory (gated=True).
    """

    def __init__(
        self,
        store: ContrastiveTrajectoryStore,
        *,
        top_k: int = 3,
        agg: str = "max",                # "max" or "mean"
        min_sim: float = 0.22,           # weak retrieval threshold
        min_margin: float = 0.05,        # require separation between success and failure
        ambiguous_margin: float = 0.02,  # margin near 0 is ambiguous
    ) -> None:
        self.store = store
        self.top_k = int(top_k)
        self.agg = agg.strip().lower()
        if self.agg not in ("max", "mean"):
            raise ValueError("agg must be 'max' or 'mean'")
        self.min_sim = float(min_sim)
        self.min_margin = float(min_margin)
        self.ambiguous_margin = float(ambiguous_margin)

    def rank_actions(
        self,
        *,
        task: str,
        dataset: str,
        domain: str,
        candidate_actions: List[Any],
        current_screenshot_b64: Optional[str] = None,
        action_history: Union[List[Any], str, None] = None,
        state_description: str = "",
        use_image: bool = True,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Returns:
          (ranked_actions, meta)

        meta contains per-action evidence and whether rerank should be applied.
        """
        if not candidate_actions:
            return [], {"applied": False, "reason": "no candidates"}

        # For action scoring, we query memory per action using a task+state+action-conditioned query.
        # This is a simple way to ground the action against past trajectories.
        # It does NOT require step-level indexing, but is a reasonable baseline.
        from .utils import safe_b64_to_pil
        img = safe_b64_to_pil(current_screenshot_b64) if (use_image and current_screenshot_b64) else None
        hist_txt = coerce_action_history_text(action_history)

        evidences: List[ActionEvidence] = []
        for a in candidate_actions:
            a_txt = self._action_to_text(a)
            query_text = self._build_query_text(
                task=task,
                dataset=dataset,
                domain=domain,
                action_text=a_txt,
                action_history=hist_txt,
                state_description=state_description,
            )

            succ_scores, fail_scores = self.store.score_text_query(
                query_text=query_text,
                domain=domain,
                dataset=dataset,
                query_image=img,
                top_k=self.top_k,
            )

            succ_ev = self._aggregate(succ_scores)
            fail_ev = self._aggregate(fail_scores)
            margin = succ_ev - fail_ev

            gated, reason = self._gate(succ_ev, fail_ev, margin)

            evidences.append(
                ActionEvidence(
                    action=a,
                    success_scores=succ_scores,
                    failure_scores=fail_scores,
                    success_evidence=succ_ev,
                    failure_evidence=fail_ev,
                    margin=margin,
                    gated=gated,
                    gate_reason=reason,
                )
            )

        # Decide whether we apply reranking at all.
        # If all candidates are gated, we return original order and mark not applied.
        if all(e.gated for e in evidences):
            return candidate_actions, {
                "applied": False,
                "reason": "all candidates gated (weak/ambiguous evidence)",
                "evidence": [self._evidence_to_dict(e) for e in evidences],
            }

        # Rank by margin descending; gated items stay but are pushed down.
        ranked = sorted(
            evidences,
            key=lambda e: (0 if e.gated else 1, e.margin),
            reverse=True,
        )

        return [e.action for e in ranked], {
            "applied": True,
            "reason": "contrastive margin rerank",
            "evidence": [self._evidence_to_dict(e) for e in ranked],
        }

    # -------------------
    # internals
    # -------------------
    def _aggregate(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        if self.agg == "max":
            return float(max(scores))
        # mean over available topk
        return float(sum(scores) / max(1, len(scores)))

    def _gate(self, succ: float, fail: float, margin: float) -> Tuple[bool, str]:
        # weak evidence: both low
        if succ < self.min_sim and fail < self.min_sim:
            return True, "weak evidence (both < min_sim)"
        # ambiguous: margin too small
        if abs(margin) < self.ambiguous_margin:
            return True, "ambiguous margin near 0"
        # require minimum positive margin to trust rerank
        if margin < self.min_margin:
            return True, "margin below min_margin"
        return False, "ok"

    def _build_query_text(
        self,
        *,
        task: str,
        dataset: str,
        domain: str,
        action_text: str,
        action_history: str,
        state_description: str,
    ) -> str:
        # Keep it compact: too much text can drown the action signal.
        # Prefix matches your retrieval convention.
        base = f"{dataset}_{domain}: {task}"
        parts = [base, f"PROPOSED_ACTION: {action_text}"]
        if state_description.strip():
            parts.append(f"STATE: {state_description.strip()[:200]}")
        if action_history and action_history != "(none)":
            parts.append(f"HISTORY: {action_history[-300:]}")
        return " | ".join(parts)

    def _action_to_text(self, a: Any) -> str:
        if isinstance(a, str):
            return a.strip()
        if isinstance(a, dict):
            name = a.get("name") or a.get("action") or "action"
            args = a.get("arguments") or a.get("args") or {}
            if isinstance(args, dict):
                kv = ", ".join([f"{k}={str(v)[:30]}" for k, v in list(args.items())[:4]])
                return f"{name}({kv})" if kv else f"{name}()"
            return f"{name}({str(args)[:60]})"
        return str(a)[:120]

    def _evidence_to_dict(self, e: ActionEvidence) -> Dict[str, Any]:
        return {
            "action": self._action_to_text(e.action),
            "success_scores": e.success_scores,
            "failure_scores": e.failure_scores,
            "success_evidence": e.success_evidence,
            "failure_evidence": e.failure_evidence,
            "margin": e.margin,
            "gated": e.gated,
            "gate_reason": e.gate_reason,
        }
