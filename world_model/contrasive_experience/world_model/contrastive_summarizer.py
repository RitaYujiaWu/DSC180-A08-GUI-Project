"""
ContrastiveSummarizer: Generate text summaries for agent prompt injection.

Creates structured guidance based on contrastive analysis of success/failure trajectories.
"""

import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ContrastiveSummarizer:
    """
    Generates text summaries for injection into agent prompts.

    Produces ~200 token summaries at each step with:
    - Success patterns to follow
    - Failure patterns to avoid
    - Key insights and recommendations
    """

    def __init__(self, tool_llm: 'DirectVLLMModel', prompts_dir: str = None):
        """
        Initialize the ContrastiveSummarizer.

        Args:
            tool_llm: LLM for generating summaries (currently unused, summaries are template-based)
            prompts_dir: Directory containing prompt templates
        """
        self.tool_llm = tool_llm
        self.prompts_dir = prompts_dir or os.path.join(
            os.path.dirname(__file__), 'prompts'
        )

    def summarize_initial_guidance(
        self,
        task: str,
        contrastive_insights: List['ContrastiveInsight'],
        success_patterns: List[str],
        failure_patterns: List[str]
    ) -> str:
        """
        Generate initial guidance summary (before first action).

        This is injected into the system prompt at task start.

        Args:
            task: Current task description
            contrastive_insights: Insights from contrastive analysis
            success_patterns: Extracted success patterns
            failure_patterns: Extracted failure patterns

        Returns:
            Text summary (~200 tokens) to inject into agent prompt
        """
        lines = ["**World Model Guidance:**\n"]
        lines.append("Based on similar tasks, here's what typically works and what to avoid:\n")

        # Success patterns section
        lines.append("**SUCCESS PATTERNS:**")
        success_insights = [i for i in contrastive_insights if i.pattern_type == "success_pattern"]
        if success_insights:
            for insight in success_insights[:3]:
                lines.append(f"- {insight.description}")
        elif success_patterns:
            for pattern in success_patterns[:2]:
                # Extract just the action part for brevity
                if " | Actions: " in pattern:
                    actions = pattern.split(" | Actions: ")[1]
                    lines.append(f"- Follow pattern: {actions}")
                else:
                    lines.append(f"- {pattern[:100]}")
        else:
            lines.append("- (No specific success patterns available)")
        lines.append("")

        # Failure patterns section
        lines.append("**COMMON MISTAKES TO AVOID:**")
        failure_insights = [i for i in contrastive_insights if i.pattern_type == "failure_pattern"]
        if failure_insights:
            for insight in failure_insights[:3]:
                lines.append(f"- {insight.description}")
        elif failure_patterns:
            for pattern in failure_patterns[:2]:
                if " | Actions: " in pattern:
                    actions = pattern.split(" | Actions: ")[1]
                    lines.append(f"- Avoid pattern: {actions}")
                else:
                    lines.append(f"- {pattern[:100]}")
        else:
            lines.append("- (No specific failure patterns identified)")
        lines.append("")

        # Key insight section
        key_insights = [i for i in contrastive_insights
                       if i.pattern_type in ["key_insight", "divergence_point"]]
        if key_insights:
            lines.append(f"**KEY INSIGHT:** {key_insights[0].description}")

        return "\n".join(lines)

    def summarize_step_guidance(
        self,
        task: str,
        current_state_description: str,
        action_history: List[str],
        step_num: int,
        relevant_insights: List['ContrastiveInsight']
    ) -> str:
        """
        Generate per-step guidance summary (~200 tokens full guidance).

        This is injected into the user message at each step.

        Args:
            task: Current task description
            current_state_description: Description of current page state
            action_history: List of previous actions taken
            step_num: Current step number (0-indexed)
            relevant_insights: Insights relevant to current situation

        Returns:
            Text summary to inject into agent prompt
        """
        lines = [f"**Step {step_num + 1} Guidance:**\n"]

        # Current progress
        lines.append(f"Progress: {len(action_history)} actions taken")
        if action_history:
            recent_actions = action_history[-3:] if len(action_history) > 3 else action_history
            lines.append(f"Recent actions: {' -> '.join(recent_actions)}")
        lines.append("")

        # Success guidance
        success_insights = [i for i in relevant_insights if i.pattern_type == "success_pattern"]
        if success_insights:
            lines.append("**At this stage, successful agents:**")
            for insight in success_insights[:2]:
                lines.append(f"- {insight.description}")
            lines.append("")

        # Failure warnings
        failure_insights = [i for i in relevant_insights if i.pattern_type == "failure_pattern"]
        if failure_insights:
            lines.append("**Avoid these common mistakes:**")
            for insight in failure_insights[:2]:
                lines.append(f"- {insight.description}")
            lines.append("")

        # Key insight reminder
        key_insights = [i for i in relevant_insights
                       if i.pattern_type in ["key_insight", "divergence_point"]]
        if key_insights:
            lines.append(f"**Remember:** {key_insights[0].description}")

        # Step-specific warnings
        if step_num > 10:
            lines.append("")
            lines.append("**Warning:** Many steps taken. Consider if task is complete or if approach needs change.")

        if step_num > 5 and action_history:
            # Check for repeated actions (potential loop)
            if len(action_history) >= 3:
                last_three = action_history[-3:]
                if len(set(last_three)) == 1:
                    lines.append("")
                    lines.append("**Warning:** Repeating same action. Try a different approach.")

        return "\n".join(lines)

    def summarize_success_only_guidance(
        self,
        task: str,
        success_trajectories: List['StoredTrajectory'],
        step_num: int = 0
    ) -> str:
        """
        Generate guidance when only success trajectories are available (fallback mode).

        Args:
            task: Current task description
            success_trajectories: List of successful trajectories
            step_num: Current step number

        Returns:
            Text summary based on success patterns only
        """
        lines = ["**World Model Guidance (Success Patterns):**\n"]
        lines.append("Based on similar successful tasks:\n")

        if success_trajectories:
            lines.append("**SUCCESSFUL APPROACHES:**")
            for i, traj in enumerate(success_trajectories[:3]):
                actions = traj.action_sequence[:5] if traj.action_sequence else []
                if actions:
                    lines.append(f"- Pattern {i+1}: {' -> '.join(actions)}")
                else:
                    lines.append(f"- Completed in {traj.total_rounds} steps")
            lines.append("")

            # Calculate average steps
            avg_steps = sum(t.total_rounds for t in success_trajectories) / len(success_trajectories)
            lines.append(f"**Typical completion:** {avg_steps:.0f} steps")
        else:
            lines.append("(No similar successful tasks found - proceed with caution)")

        return "\n".join(lines)

    def summarize_action_prediction(
        self,
        proposed_action: Dict,
        prediction: 'StatePrediction',
        similar_outcomes: List[Dict] = None
    ) -> str:
        """
        Summarize prediction for a proposed action.

        Args:
            proposed_action: The action being evaluated
            prediction: Predicted outcome
            similar_outcomes: Historical similar action outcomes

        Returns:
            Brief text summary of predicted outcome
        """
        action_type = proposed_action.get('name', 'unknown')

        lines = [f"**Prediction for '{action_type}':**"]
        lines.append(f"Success probability: {prediction.success_probability:.0%}")

        if prediction.expected_changes:
            lines.append(f"Expected: {', '.join(prediction.expected_changes[:2])}")

        if prediction.potential_issues:
            lines.append(f"Watch for: {', '.join(prediction.potential_issues[:2])}")

        return "\n".join(lines)

    def format_compact_guidance(
        self,
        insights: List['ContrastiveInsight'],
        max_tokens: int = 100
    ) -> str:
        """
        Format insights into a compact string for constrained contexts.

        Args:
            insights: List of insights to format
            max_tokens: Approximate token limit

        Returns:
            Compact guidance string
        """
        parts = []

        success = [i for i in insights if i.pattern_type == "success_pattern"]
        failure = [i for i in insights if i.pattern_type == "failure_pattern"]
        key = [i for i in insights if i.pattern_type == "key_insight"]

        if success:
            parts.append(f"DO: {success[0].description[:50]}")
        if failure:
            parts.append(f"AVOID: {failure[0].description[:50]}")
        if key:
            parts.append(f"KEY: {key[0].description[:50]}")

        return " | ".join(parts)
