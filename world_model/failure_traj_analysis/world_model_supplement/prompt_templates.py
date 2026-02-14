from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os


DEFAULT_STEP_GUIDANCE = """# Step Guidance Prompt


## Context

**Task:** {task}
**Current Step:** {step_num}
**Actions Taken So Far:** {action_history}
**Current State:** {state_description}

## Similar Successful Trajectories at This Step

{success_at_step}

## Common Failures at This Step

{failures_at_step}

## Output Format

1. **Progress Assessment** (1 sentence)

2. **Next Action Recommendation** (1-2 sentences)

3. **Pitfall Warning** (1 sentence)

4. **Verification Reminder** (1 sentence)
"""


DEFAULT_CONTRASTIVE_SUMMARY = """# Contrastive Summary Prompt

You are an expert at analyzing GUI automation trajectories.

## Your Goal

Generate a concise contrastive summary that helps an agent succeed at the current task by learning from similar past attempts.

## Input

**Current Task:** {task}
**Domain:** {domain}

**Successful Trajectories:**
{success_summaries}

**Failed Trajectories:**
{failure_summaries}

## Output Format

Generate a structured analysis with:

1. **SUCCESS PATTERNS** (2-3 bullet points):


2. **COMMON MISTAKES** (2-3 bullet points):


3. **KEY DIVERGENCE** (1 sentence):


4. **RECOMMENDATION** (1-2 sentences):


## Guidelines

- Keep output under 200 words
- Focus on actionable, specific insights
- Avoid generic advice
- Reference actual actions when possible (click, type, scroll, etc.)
- Consider the order and timing of actions
"""


@dataclass
class PromptTemplates:
    step_guidance: str
    contrastive_summary: str

    @staticmethod
    def from_files_or_defaults(step_path: Optional[str], contrastive_path: Optional[str]) -> "PromptTemplates":
        step = _read_text(step_path) if step_path else DEFAULT_STEP_GUIDANCE
        con = _read_text(contrastive_path) if contrastive_path else DEFAULT_CONTRASTIVE_SUMMARY
        return PromptTemplates(step_guidance=step, contrastive_summary=con)


def _read_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt template path not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
