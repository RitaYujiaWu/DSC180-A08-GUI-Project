from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .contrastive_memory import RetrievedTrajectory
from .contrastive_analyzer import ContrastiveInsight
from .prompt_templates import PromptTemplates
from .llm_utils import LLMClient
from .utils import clip_words


@dataclass
class GuidancePacket:
    guidance_text: str
    meta: Dict[str, Any]


class GuidanceGenerator:
    """
    Two-stage guidance:
      - initial: uses contrastive summary prompt (<=200 words)
      - step: uses step guidance prompt (<=150 words)

    Adds similarity gating to reduce distracting guidance when retrieval is weak.
    """
    def __init__(
        self,
        templates: PromptTemplates,
        min_sim: float = 0.22,
        low_confidence_max_words: int = 60,
    ) -> None:
        self.templates = templates
        self.min_sim = float(min_sim)
        self.low_conf_words = int(low_confidence_max_words)

    def format_initial(
        self,
        task: str,
        domain: str,
        success_trajs: List[RetrievedTrajectory],
        failure_trajs: List[RetrievedTrajectory],
        insights: ContrastiveInsight,
        min_similarity: float,
    ) -> GuidancePacket:
        # For initial guidance we return the structured text itself (from insight fields).
        # This keeps it deterministic and avoids second-round LLM calls.
        if min_similarity < self.min_sim:
            short = self._low_confidence_initial(task, insights, min_similarity)
            return GuidancePacket(short, meta={"mode": "initial", "min_similarity": min_similarity, "low_confidence": True})

        lines: List[str] = []
        if insights.success_patterns:
            lines.append("SUCCESS PATTERNS:")
            for p in insights.success_patterns[:3]:
                lines.append(f"- {p}")
        if insights.common_mistakes:
            lines.append("COMMON MISTAKES:")
            for m in insights.common_mistakes[:3]:
                lines.append(f"- {m}")
        if insights.key_divergence:
            lines.append(f"KEY DIVERGENCE: {insights.key_divergence}")
        if insights.recommendation:
            lines.append(f"RECOMMENDATION: {insights.recommendation}")

        text = "\n".join(lines).strip()
        return GuidancePacket(text, meta={"mode": "initial", "min_similarity": min_similarity, "low_confidence": False})

    def format_step(
        self,
        task: str,
        step_num: int,
        action_history_text: str,
        state_description: str,
        success_trajs: List[RetrievedTrajectory],
        failure_trajs: List[RetrievedTrajectory],
        min_similarity: float,
        avg_steps: Optional[int],
        llm: LLMClient,
    ) -> GuidancePacket:
        if min_similarity < self.min_sim:
            short = self._low_confidence_step(step_num, action_history_text, min_similarity)
            return GuidancePacket(short, meta={"mode": "step", "min_similarity": min_similarity, "low_confidence": True})

        # Build step-specific evidence snippets
        success_at_step = self._steps_near_k(success_trajs, step_num, label="SUCCESS")
        failures_at_step = self._steps_near_k(failure_trajs, step_num, label="FAILURE")

        prompt = self.templates.step_guidance.format(
            task=task,
            step_num=step_num,
            action_history=action_history_text,
            state_description=state_description,
            success_at_step=success_at_step,
            failures_at_step=failures_at_step,
            avg_steps=avg_steps if avg_steps is not None else "unknown",
        )

        out = llm.generate(prompt)
        out = out.strip()

        # Hard clip to keep it concise if model rambles
        out = clip_words(out, 150)  # ~150 words cap
        return GuidancePacket(out, meta={"mode": "step", "min_similarity": min_similarity, "low_confidence": False})

    # -----------------------
    # helpers
    # -----------------------
    def _steps_near_k(self, trajs: List[RetrievedTrajectory], step_num: int, label: str) -> str:
        if not trajs:
            return "(none)"
        chunks: List[str] = []
        for t in trajs:
            i = max(0, min(step_num - 1, len(t.steps) - 1))
            near = t.steps[i:i+2]
            text_steps = []
            for r in near:
                resp = (r.get("response") or "").strip()
                if resp:
                    text_steps.append(resp[:140])
            chunks.append(f"- ({label} score={t.score:.3f}) { ' | '.join(text_steps) if text_steps else '(no text)' }")
        return "\n".join(chunks)

    def _low_confidence_initial(self, task: str, insights: ContrastiveInsight, min_similarity: float) -> str:
        base = (
            f"Low-confidence retrieval (min_sim={min_similarity:.3f}). "
            "Use general safe strategy: identify the correct page/state, verify key fields, then act."
        )
        extra = ""
        if insights.common_mistakes:
            extra = " Avoid: " + "; ".join(insights.common_mistakes[:2])
        return clip_words(base + extra, self.low_conf_words)

    def _low_confidence_step(self, step_num: int, action_history_text: str, min_similarity: float) -> str:
        base = (
            f"Step {step_num}: Low-confidence retrieval (min_sim={min_similarity:.3f}). "
            "Re-orient: confirm current page/state matches the goal; avoid repeating the same action; prefer a reversible check (scroll/search/back)."
        )
        return clip_words(base, self.low_conf_words)
