from __future__ import annotations

from typing import Iterable, List, Optional

from .schemas import RetrievedCase


def _truncate(text: Optional[str], max_chars: int = 240) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def format_retrieved_cases(
    cases: Iterable[RetrievedCase],
    title: str,
    max_cases: int = 3,
    max_chars_per_case: int = 220,
) -> str:
    lines: List[str] = [title]
    trimmed_cases = list(cases)[:max_cases]

    if not trimmed_cases:
        lines.append("- None")
        return "\n".join(lines)

    for idx, case in enumerate(trimmed_cases, start=1):
        task = _truncate(case.task, 100)
        summary = _truncate(case.summary, max_chars_per_case)
        score_str = f"{case.score:.4f}" if isinstance(case.score, (int, float)) else "N/A"
        lines.append(
            f"- Case {idx} | id={case.trajectory_id} | score={score_str} | task={task} | summary={summary}"
        )
    return "\n".join(lines)


def build_candidate_generation_prompt(
    intent: str,
    current_url: Optional[str] = None,
    action_history: Optional[List[str]] = None,
    observation_summary: Optional[str] = None,
    k_candidates: int = 3,
) -> str:
    action_history = action_history or []

    history_block = "\n".join(f"- {a}" for a in action_history[-5:]) if action_history else "- None"

    prompt = f"""You are helping a GUI agent think before acting.

Task:
{intent}

Current URL:
{current_url or "Unknown"}

Recent action history:
{history_block}

Current observation summary:
{observation_summary or "Not provided"}

Propose exactly {k_candidates} plausible next GUI actions for the agent.
The actions should be concrete, executable, and formatted similarly to the agent's normal action style.

Return exactly this format:
1. <action>
2. <action>
3. <action>
"""

    return prompt


def build_candidate_scoring_prompt(
    intent: str,
    candidate_action: str,
    success_cases: List[RetrievedCase],
    failure_cases: List[RetrievedCase],
    current_url: Optional[str] = None,
    observation_summary: Optional[str] = None,
) -> str:
    success_block = format_retrieved_cases(success_cases, "Retrieved successful precedents:")
    failure_block = format_retrieved_cases(failure_cases, "Retrieved failure precedents:")

    prompt = f"""You are evaluating one candidate GUI action before execution.

Task:
{intent}

Current URL:
{current_url or "Unknown"}

Current observation summary:
{observation_summary or "Not provided"}

Candidate action:
{candidate_action}

{success_block}

{failure_block}

Evaluate the candidate contrastively:
- How well does it align with the successful precedents?
- How much does it risk repeating failure precedents?

Return exactly this format:
SUCCESS_ALIGNMENT: <number from 0 to 10>
FAILURE_RISK: <number from 0 to 10>
FINAL_SCORE: <number from -10 to 10>
REASONING: <short explanation>
"""
    return prompt