from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .prompt_builder import build_candidate_generation_prompt
from .schemas import CandidateAction


class CandidateGenerator:
    """
    Generate multiple candidate next actions before execution.
    """

    def __init__(
        self,
        llm: Any,
        k_candidates: int = 3,
    ) -> None:
        self.llm = llm
        self.k_candidates = k_candidates

    def generate(
        self,
        intent: str,
        trajectory: Optional[Any] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[CandidateAction]:
        meta_data = meta_data or {}
        action_history = self._extract_recent_actions(trajectory)
        observation_summary = self._extract_observation_summary(messages)
        current_url = meta_data.get("url") or meta_data.get("current_url")

        prompt = build_candidate_generation_prompt(
            intent=intent,
            current_url=current_url,
            action_history=action_history,
            observation_summary=observation_summary,
            k_candidates=self.k_candidates,
        )

        raw_text = self._call_llm(prompt)
        candidates = self._parse_numbered_candidates(raw_text)
        candidates = self._deduplicate_candidates(candidates)

        if not candidates and raw_text.strip():
            candidates = [
                CandidateAction(
                    text=raw_text.strip(),
                    normalized_text=self._normalize_action_text(raw_text.strip()),
                    source="fallback",
                )
            ]

        return candidates

    def _call_llm(self, prompt: str) -> str:
        if self.llm is None:
            return ""

        try:
            # Common pattern: llm(prompt) -> str
            result = self.llm(prompt)
            if isinstance(result, str):
                return result
        except Exception:
            pass

        # common fallback methods
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

    def _parse_numbered_candidates(self, text: str) -> List[CandidateAction]:
        candidates: List[CandidateAction] = []
        if not text:
            return candidates

        pattern = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$", re.MULTILINE)
        matches = pattern.findall(text)

        for _, candidate_text in matches:
            candidate_text = candidate_text.strip()
            if not candidate_text:
                continue
            candidates.append(
                CandidateAction(
                    text=candidate_text,
                    normalized_text=self._normalize_action_text(candidate_text),
                    source="generated",
                )
            )

        # fallback: lines if numbering failed
        if not candidates:
            lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
            for line in lines[: self.k_candidates]:
                candidates.append(
                    CandidateAction(
                        text=line,
                        normalized_text=self._normalize_action_text(line),
                        source="generated",
                    )
                )

        return candidates[: self.k_candidates]

    def _normalize_action_text(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _deduplicate_candidates(self, candidates: List[CandidateAction]) -> List[CandidateAction]:
        deduped: List[CandidateAction] = []
        seen = set()

        for candidate in candidates:
            key = candidate.normalized_text or self._normalize_action_text(candidate.text)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)

        return deduped

    def _extract_recent_actions(self, trajectory: Optional[Any]) -> List[str]:
        if trajectory is None:
            return []

        actions: List[str] = []

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

    def _extract_observation_summary(self, messages: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        if not messages:
            return None

        text_parts: List[str] = []
        for msg in messages[-4:]:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text")
                        if text:
                            text_parts.append(str(text))

        if not text_parts:
            return None

        merged = " ".join(" ".join(text_parts).split())
        return merged[:1000]