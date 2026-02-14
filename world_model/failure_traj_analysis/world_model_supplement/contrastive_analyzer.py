from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json

from .llm_utils import LLMClient
from .prompt_templates import PromptTemplates
from .contrastive_memory import RetrievedTrajectory
from .utils import extract_json_object


@dataclass
class ContrastiveInsight:
    success_patterns: List[str]
    common_mistakes: List[str]
    key_divergence: str
    recommendation: str


class ContrastiveAnalyzer:
    """
    Uses the Contrastive Summary prompt (<=200 words) to produce structured insights.
    Falls back to a simple heuristic if the LLM output is malformed.
    """
    def __init__(self, llm: LLMClient, templates: PromptTemplates) -> None:
        self.llm = llm
        self.templates = templates

    def contrastive_summary(
        self,
        task: str,
        domain: str,
        success_trajs: List[RetrievedTrajectory],
        failure_trajs: List[RetrievedTrajectory],
    ) -> ContrastiveInsight:
        succ_txt = self._format_traj_summaries(success_trajs, max_each=4)
        fail_txt = self._format_traj_summaries(failure_trajs, max_each=4)

        prompt = self.templates.contrastive_summary.format(
            task=task,
            domain=domain,
            success_summaries=succ_txt,
            failure_summaries=fail_txt,
        )

        raw = self.llm.generate(prompt)

        insight = self._parse_structured(raw)
        if insight:
            return insight

        # fallback heuristic: use top retrieved steps to build minimal advice
        return ContrastiveInsight(
            success_patterns=self._fallback_patterns(success_trajs, positive=True),
            common_mistakes=self._fallback_patterns(failure_trajs, positive=False),
            key_divergence="Success trajectories verify key UI state before committing; failures often act prematurely.",
            recommendation="Follow the successful ordering: locate the right UI element, verify the correct page/state, then perform the action.",
        )

    def _format_traj_summaries(self, trajs: List[RetrievedTrajectory], max_each: int = 4) -> str:
        chunks = []
        for t in trajs:
            steps = self._extract_action_lines(t, max_each=max_each)
            chunks.append(
                f"- (score={t.score:.3f}) task='{t.task}' steps: {', '.join(steps) if steps else 'N/A'}"
            )
        return "\n".join(chunks) if chunks else "(none)"

    def _extract_action_lines(self, traj: RetrievedTrajectory, max_each: int = 4) -> List[str]:
        out: List[str] = []
        for r in (traj.steps or [])[:max_each]:
            resp = r.get("response") or ""
            # many logs include "Action: {...}"
            if "Action:" in resp:
                out.append(resp.split("Action:", 1)[-1].strip()[:80])
            else:
                out.append(resp.strip()[:80])
        return out

    def _parse_structured(self, text: str) -> Optional[ContrastiveInsight]:
        """
        Accepts the markdown-ish structure defined in contrastive_summary.md.
        We parse sections by headers; if missing, return None.
        """
        t = (text or "").strip()
        if not t:
            return None

        # crude section parsing
        def grab_section(header: str) -> str:
            idx = t.upper().find(header)
            if idx < 0:
                return ""
            sub = t[idx + len(header):]
            # stop at next all-caps header keyword
            stops = ["SUCCESS PATTERNS", "COMMON MISTAKES", "KEY DIVERGENCE", "RECOMMENDATION"]
            next_pos = None
            for s in stops:
                j = sub.upper().find(s)
                if j >= 0:
                    next_pos = j if next_pos is None else min(next_pos, j)
            return sub[:next_pos].strip() if next_pos is not None else sub.strip()

        sp = grab_section("SUCCESS PATTERNS")
        cm = grab_section("COMMON MISTAKES")
        kd = grab_section("KEY DIVERGENCE")
        rec = grab_section("RECOMMENDATION")

        if not (sp or cm or kd or rec):
            return None

        def bullets(x: str) -> List[str]:
            lines = []
            for ln in x.splitlines():
                ln = ln.strip()
                if ln.startswith("-"):
                    lines.append(ln[1:].strip())
            return lines[:3]

        return ContrastiveInsight(
            success_patterns=bullets(sp) if sp else [],
            common_mistakes=bullets(cm) if cm else [],
            key_divergence=" ".join(kd.split())[:240] if kd else "",
            recommendation=" ".join(rec.split())[:400] if rec else "",
        )

    def _fallback_patterns(self, trajs: List[RetrievedTrajectory], positive: bool) -> List[str]:
        if not trajs:
            return ["(no examples retrieved)"]
        # very lightweight: summarize first action types seen
        pats: List[str] = []
        for t in trajs[:3]:
            actions = self._extract_action_lines(t, max_each=3)
            if actions:
                pats.append(("Do: " if positive else "Avoid: ") + actions[0])
        return pats[:3]
