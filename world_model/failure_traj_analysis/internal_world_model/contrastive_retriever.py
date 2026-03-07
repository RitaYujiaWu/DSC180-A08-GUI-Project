from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .schemas import RetrievedCase


class ContrastiveRetriever:
    """
    Wrapper around the existing reasoning bank to retrieve success/failure cases separately.
    """

    def __init__(
        self,
        reasoning_bank: Any,
        topk_success: int = 3,
        topk_failure: int = 3,
        use_multimodal: bool = True,
    ) -> None:
        self.reasoning_bank = reasoning_bank
        self.topk_success = topk_success
        self.topk_failure = topk_failure
        self.use_multimodal = use_multimodal

    def retrieve(
        self,
        intent: str,
        trajectory: Optional[Any] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        current_image: Optional[Any] = None,
    ) -> Tuple[List[RetrievedCase], List[RetrievedCase]]:
        if self.reasoning_bank is None:
            return [], []

        query_text = self._build_query_text(intent=intent, trajectory=trajectory, meta_data=meta_data)
        meta_data = meta_data or {}
        domain = meta_data.get("domain") or meta_data.get("website") or meta_data.get("site")

        success_cases = self._retrieve_by_label(
            query_text=query_text,
            label="success",
            topk=self.topk_success,
            current_image=current_image,
            domain=domain,
        )
        failure_cases = self._retrieve_by_label(
            query_text=query_text,
            label="failure",
            topk=self.topk_failure,
            current_image=current_image,
            domain=domain,
        )
        return success_cases, failure_cases

    def _build_query_text(
        self,
        intent: str,
        trajectory: Optional[Any] = None,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        meta_data = meta_data or {}
        domain = meta_data.get("domain") or meta_data.get("website") or meta_data.get("site")
        current_url = meta_data.get("url") or meta_data.get("current_url")

        parts: List[str] = []
        if domain:
            parts.append(f"domain: {domain}")
        parts.append(f"task: {intent}")

        recent_actions = self._extract_recent_actions(trajectory)
        if recent_actions:
            parts.append("recent actions: " + " | ".join(recent_actions[-3:]))

        if current_url:
            parts.append(f"url: {current_url}")

        return "\n".join(parts)

    def _extract_recent_actions(self, trajectory: Optional[Any]) -> List[str]:
        if trajectory is None:
            return []

        actions: List[str] = []

        # conservative handling for common trajectory shapes
        if isinstance(trajectory, dict):
            raw_steps = trajectory.get("steps") or trajectory.get("history") or []
            for step in raw_steps:
                if isinstance(step, dict):
                    action = step.get("action") or step.get("response")
                    if action:
                        actions.append(str(action))
            return actions

        if hasattr(trajectory, "steps"):
            raw_steps = getattr(trajectory, "steps", [])
            for step in raw_steps:
                if isinstance(step, dict):
                    action = step.get("action") or step.get("response")
                    if action:
                        actions.append(str(action))
                elif hasattr(step, "action"):
                    action = getattr(step, "action", None)
                    if action:
                        actions.append(str(action))
        return actions

    def _retrieve_by_label(
        self,
        query_text: str,
        label: str,
        topk: int,
        current_image: Optional[Any] = None,
        domain: Optional[str] = None,
    ) -> List[RetrievedCase]:
        if topk <= 0:
            return []

        query_image_base64 = None
        if self.use_multimodal and current_image is not None:
            query_image_base64 = current_image

        try:
            results = self.reasoning_bank.retrieve(
                query_text=query_text,
                top_k=topk,
                domain=domain,
                label=label,
                query_image_base64=query_image_base64,
            )
        except Exception:
            return []

        if results is None:
            return []

        if isinstance(results, tuple):
            results = results[0]

        if not isinstance(results, list):
            return []

        return [self._convert_reasoning_bank_result(item, label=label) for item in results]

    def _convert_reasoning_bank_result(self, item: Any, label: str) -> RetrievedCase:
        if isinstance(item, RetrievedCase):
            return item

        if isinstance(item, dict):
            trajectory_id = (
                item.get("trajectory_id")
                or item.get("id")
                or item.get("traj_id")
                or "unknown"
            )
            task = item.get("task") or item.get("intent") or item.get("query")
            summary = (
                item.get("summary")
                or item.get("reasoning")
                or item.get("text")
                or item.get("content")
            )
            score = item.get("score") or item.get("similarity") or item.get("distance")
            return RetrievedCase(
                trajectory_id=str(trajectory_id),
                label=label,
                task=str(task) if task is not None else None,
                summary=str(summary) if summary is not None else None,
                score=float(score) if isinstance(score, (int, float)) else None,
                raw_content=item,
            )

        # generic object fallback
        trajectory_id = getattr(item, "trajectory_id", None) or getattr(item, "id", None) or "unknown"
        task = getattr(item, "task", None)
        summary = (
            getattr(item, "summary", None)
            or getattr(item, "reasoning", None)
            or getattr(item, "text", None)
        )
        score = getattr(item, "score", None)

        raw_content = item.__dict__ if hasattr(item, "__dict__") else {"value": str(item)}

        return RetrievedCase(
            trajectory_id=str(trajectory_id),
            label=label,
            task=str(task) if task is not None else None,
            summary=str(summary) if summary is not None else None,
            score=float(score) if isinstance(score, (int, float)) else None,
            raw_content=raw_content,
        )