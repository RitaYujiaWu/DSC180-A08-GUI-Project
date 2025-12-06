"""
TrajectoryAnalyzer: Analyze and compare success vs failure trajectories.

Uses LLM to identify contrastive patterns that distinguish success from failure.
"""

import os
import sys
import json
import logging
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveInsight:
    """A single insight from contrastive analysis."""
    pattern_type: str  # "success_pattern", "failure_pattern", "divergence_point", "key_insight"
    description: str
    evidence_success: List[str]
    evidence_failure: List[str]
    confidence: float


class TrajectoryAnalyzer:
    """
    Analyzes trajectories to identify patterns that distinguish success from failure.

    Uses LLM-based contrastive analysis to extract actionable insights.
    """

    def __init__(self, trajectory_store: 'TrajectoryStore', tool_llm: 'DirectVLLMModel'):
        """
        Initialize the TrajectoryAnalyzer.

        Args:
            trajectory_store: TrajectoryStore for loading full trajectory data
            tool_llm: LLM for generating analysis
        """
        self.trajectory_store = trajectory_store
        self.tool_llm = tool_llm

    def analyze_contrastive_pairs(
        self,
        success_trajectories: List['StoredTrajectory'],
        failure_trajectories: List['StoredTrajectory'],
        task: str
    ) -> List[ContrastiveInsight]:
        """
        Analyze success vs failure trajectories using LLM to identify patterns.

        Args:
            success_trajectories: List of successful trajectories
            failure_trajectories: List of failed trajectories
            task: Current task description

        Returns:
            List of ContrastiveInsight objects
        """
        if not success_trajectories and not failure_trajectories:
            return []

        # Build analysis prompt
        prompt = self._build_analysis_prompt(success_trajectories, failure_trajectories, task)

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]

        try:
            response, _, _ = self.tool_llm.chat(messages=messages, stream=False)
            content = response.content if hasattr(response, 'content') else str(response)
            insights = self._parse_analysis_response(content)
            return insights
        except Exception as e:
            logger.error(f"Error in contrastive analysis: {e}")
            # Fall back to rule-based analysis
            return self._fallback_analysis(success_trajectories, failure_trajectories)

    def _get_system_prompt(self) -> str:
        """Get system prompt for contrastive analysis."""
        return """You are an expert at analyzing GUI automation trajectories.
Your task is to identify patterns that distinguish successful task completions from failures.
Focus on actionable insights that can guide an agent to succeed.
Be concise and specific."""

    def _build_analysis_prompt(
        self,
        success_trajs: List['StoredTrajectory'],
        failure_trajs: List['StoredTrajectory'],
        task: str
    ) -> str:
        """Build prompt for contrastive analysis."""
        prompt = f"**Current Task:** {task}\n\n"

        prompt += "## Successful Trajectories:\n"
        if success_trajs:
            for i, traj in enumerate(success_trajs[:3]):
                actions = traj.action_sequence if traj.action_sequence else self._load_action_sequence(traj)
                prompt += f"{i+1}. Task: {traj.task_description[:100]}\n"
                prompt += f"   Steps: {traj.total_rounds}\n"
                prompt += f"   Actions: {' -> '.join(actions[:8])}\n"
                if traj.final_answer:
                    prompt += f"   Result: {traj.final_answer[:50]}\n"
                prompt += "\n"
        else:
            prompt += "(No successful trajectories available)\n\n"

        prompt += "## Failed Trajectories:\n"
        if failure_trajs:
            for i, traj in enumerate(failure_trajs[:3]):
                actions = traj.action_sequence if traj.action_sequence else self._load_action_sequence(traj)
                prompt += f"{i+1}. Task: {traj.task_description[:100]}\n"
                prompt += f"   Steps: {traj.total_rounds}\n"
                prompt += f"   Actions: {' -> '.join(actions[:8])}\n"
                if traj.final_answer:
                    prompt += f"   Result: {traj.final_answer[:50]}\n"
                prompt += "\n"
        else:
            prompt += "(No failed trajectories available)\n\n"

        prompt += """
Analyze these trajectories and identify:
1. 2-3 key patterns that led to SUCCESS
2. 2-3 common mistakes that led to FAILURE (if available)
3. The critical divergence point where success and failure paths differ

Output ONLY valid JSON (no markdown, no explanation):
{
  "success_patterns": ["pattern1", "pattern2"],
  "failure_patterns": ["pattern1", "pattern2"],
  "divergence_point": "description of where paths diverge",
  "key_insight": "the most important takeaway for the current task"
}
"""
        return prompt

    def _load_action_sequence(self, traj: 'StoredTrajectory') -> List[str]:
        """Load action sequence from trajectory file."""
        try:
            data = traj.load_full_data()
            if data is None:
                return []

            actions = []
            for round_data in data.get('rounds', []):
                response = round_data.get('response', '')
                if isinstance(response, list):
                    response = response[0] if response else ''
                if isinstance(response, dict):
                    response = response.get('content', str(response))

                action_name = self._extract_action_name(response)
                if action_name:
                    actions.append(action_name)
            return actions
        except Exception as e:
            logger.warning(f"Error loading action sequence: {e}")
            return []

    def _extract_action_name(self, response: str) -> Optional[str]:
        """Extract action name from response."""
        try:
            patterns = [
                r'"name"\s*:\s*"([^"]+)"',
                r'"action"\s*:\s*"([^"]+)"',
                r'"action_type"\s*:\s*"([^"]+)"'
            ]
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group(1)
        except:
            pass
        return None

    def _parse_analysis_response(self, response: str) -> List[ContrastiveInsight]:
        """Parse LLM response into ContrastiveInsight objects."""
        insights = []

        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                # Parse success patterns
                for pattern in data.get('success_patterns', []):
                    if pattern:
                        insights.append(ContrastiveInsight(
                            pattern_type="success_pattern",
                            description=pattern,
                            evidence_success=[],
                            evidence_failure=[],
                            confidence=0.8
                        ))

                # Parse failure patterns
                for pattern in data.get('failure_patterns', []):
                    if pattern:
                        insights.append(ContrastiveInsight(
                            pattern_type="failure_pattern",
                            description=pattern,
                            evidence_success=[],
                            evidence_failure=[],
                            confidence=0.8
                        ))

                # Parse divergence point
                if data.get('divergence_point'):
                    insights.append(ContrastiveInsight(
                        pattern_type="divergence_point",
                        description=data['divergence_point'],
                        evidence_success=[],
                        evidence_failure=[],
                        confidence=0.9
                    ))

                # Parse key insight
                if data.get('key_insight'):
                    insights.append(ContrastiveInsight(
                        pattern_type="key_insight",
                        description=data['key_insight'],
                        evidence_success=[],
                        evidence_failure=[],
                        confidence=0.9
                    ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
        except Exception as e:
            logger.warning(f"Error parsing analysis response: {e}")

        return insights

    def _fallback_analysis(
        self,
        success_trajs: List['StoredTrajectory'],
        failure_trajs: List['StoredTrajectory']
    ) -> List[ContrastiveInsight]:
        """Provide rule-based fallback analysis when LLM fails."""
        insights = []

        # Analyze step counts
        if success_trajs:
            avg_success_steps = sum(t.total_rounds for t in success_trajs) / len(success_trajs)
            insights.append(ContrastiveInsight(
                pattern_type="success_pattern",
                description=f"Successful completions average {avg_success_steps:.1f} steps",
                evidence_success=[],
                evidence_failure=[],
                confidence=0.6
            ))

        if failure_trajs:
            avg_failure_steps = sum(t.total_rounds for t in failure_trajs) / len(failure_trajs)
            insights.append(ContrastiveInsight(
                pattern_type="failure_pattern",
                description=f"Failed attempts average {avg_failure_steps:.1f} steps - may indicate getting stuck",
                evidence_success=[],
                evidence_failure=[],
                confidence=0.6
            ))

        # Analyze common actions
        if success_trajs:
            success_actions = []
            for t in success_trajs[:3]:
                actions = t.action_sequence if t.action_sequence else []
                if actions:
                    success_actions.append(actions[0])

            if success_actions:
                most_common = max(set(success_actions), key=success_actions.count)
                insights.append(ContrastiveInsight(
                    pattern_type="success_pattern",
                    description=f"Successful trajectories often start with '{most_common}' action",
                    evidence_success=[],
                    evidence_failure=[],
                    confidence=0.5
                ))

        return insights

    def extract_success_patterns(self, trajectories: List['StoredTrajectory']) -> List[str]:
        """Extract common patterns from successful trajectories."""
        patterns = []
        for traj in trajectories[:3]:
            actions = traj.action_sequence if traj.action_sequence else self._load_action_sequence(traj)
            if actions:
                action_str = ' -> '.join(actions[:5])
                patterns.append(f"Task: {traj.task_description[:50]}... | Actions: {action_str}")
        return patterns

    def extract_failure_patterns(self, trajectories: List['StoredTrajectory']) -> List[str]:
        """Extract common mistake patterns from failed trajectories."""
        patterns = []
        for traj in trajectories[:3]:
            actions = traj.action_sequence if traj.action_sequence else self._load_action_sequence(traj)
            if actions:
                action_str = ' -> '.join(actions[:5])
                patterns.append(f"Task: {traj.task_description[:50]}... | Actions: {action_str}")
        return patterns

    def get_step_specific_insights(
        self,
        insights: List[ContrastiveInsight],
        step_num: int,
        action_history: List[str]
    ) -> List[ContrastiveInsight]:
        """Filter insights relevant to current step."""
        # For now, return all insights
        # Could be enhanced to filter based on step patterns
        return insights
