"""
StatePredictor: Predict next state given current state and proposed action.

Provides action outcome prediction based on historical transitions.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StatePrediction:
    """Prediction about the next state after an action."""
    expected_changes: List[str] = field(default_factory=list)
    expected_url_change: bool = False
    expected_visual_change: str = ""
    success_probability: float = 0.7
    potential_issues: List[str] = field(default_factory=list)
    similar_transitions: List[Dict] = field(default_factory=list)


class StatePredictor:
    """
    Predicts the next state given current state and proposed action.

    Uses action-type heuristics and historical transition patterns
    to predict likely outcomes of actions.
    """

    def __init__(self, tool_llm: 'DirectVLLMModel', prompts_dir: str = None):
        """
        Initialize the StatePredictor.

        Args:
            tool_llm: LLM for more sophisticated predictions (optional)
            prompts_dir: Directory containing prompt templates
        """
        self.tool_llm = tool_llm
        self.prompts_dir = prompts_dir

        # Action type prediction rules
        self._action_predictions = {
            'click': {
                'expected_changes': [
                    "Element may be highlighted/activated",
                    "Page may navigate to new URL",
                    "Modal or dropdown may open"
                ],
                'url_change': True,
                'visual_change': "Click target may change state or trigger navigation",
                'success_prob': 0.75,
                'issues': ["Wrong element clicked", "Element not clickable", "Unexpected popup"]
            },
            'type': {
                'expected_changes': [
                    "Text appears in input field",
                    "Autocomplete suggestions may appear",
                    "Form validation may trigger"
                ],
                'url_change': False,
                'visual_change': "Text input field will show typed content",
                'success_prob': 0.85,
                'issues': ["Wrong field selected", "Input not accepted", "Character encoding issues"]
            },
            'scroll': {
                'expected_changes': [
                    "Page viewport moves",
                    "New content becomes visible",
                    "Lazy-loaded content may appear"
                ],
                'url_change': False,
                'visual_change': "Page scrolls to reveal different content",
                'success_prob': 0.9,
                'issues': ["Already at scroll limit", "Dynamic content not loaded"]
            },
            'select': {
                'expected_changes': [
                    "Dropdown option selected",
                    "Form field value changes",
                    "Dependent fields may update"
                ],
                'url_change': False,
                'visual_change': "Dropdown shows selected value",
                'success_prob': 0.8,
                'issues': ["Option not found", "Dropdown not accessible"]
            },
            'press_key': {
                'expected_changes': [
                    "Key action performed",
                    "Form may submit (Enter)",
                    "Navigation may occur (Tab)"
                ],
                'url_change': True,
                'visual_change': "Depends on key pressed",
                'success_prob': 0.8,
                'issues': ["Unexpected form submission", "Focus not on expected element"]
            },
            'goto': {
                'expected_changes': [
                    "Browser navigates to new URL",
                    "Page fully reloads",
                    "New page content appears"
                ],
                'url_change': True,
                'visual_change': "Entire page changes to new URL",
                'success_prob': 0.85,
                'issues': ["URL not found", "Redirect occurs", "Page load timeout"]
            },
            'go_back': {
                'expected_changes': [
                    "Browser goes to previous page",
                    "Previous page state restored"
                ],
                'url_change': True,
                'visual_change': "Returns to previously visited page",
                'success_prob': 0.9,
                'issues': ["No history to go back to", "Previous page expired"]
            },
            'wait': {
                'expected_changes': [
                    "Pauses execution",
                    "Allows page to load/update"
                ],
                'url_change': False,
                'visual_change': "No change, but page may finish loading",
                'success_prob': 0.95,
                'issues': ["Wait time too short/long"]
            },
            'stop': {
                'expected_changes': [
                    "Task completion signaled",
                    "Answer submitted"
                ],
                'url_change': False,
                'visual_change': "No change - task ends",
                'success_prob': 0.9,
                'issues': ["Premature stop", "Incorrect answer"]
            }
        }

    def predict_next_state(
        self,
        current_state: str,
        action: Optional[Dict],
        task: str = None
    ) -> StatePrediction:
        """
        Predict what the next state will look like after the action.

        Args:
            current_state: Base64-encoded current screenshot
            action: Proposed action dict with 'name' and 'arguments'
            task: Current task description for context

        Returns:
            StatePrediction with expected outcomes
        """
        if action is None:
            return StatePrediction(
                expected_changes=["No action specified"],
                success_probability=0.0
            )

        action_type = action.get('name', '').lower()

        # Get prediction rules for this action type
        rules = self._action_predictions.get(action_type, {})

        if not rules:
            # Unknown action type - return generic prediction
            return StatePrediction(
                expected_changes=["Action outcome uncertain"],
                expected_url_change=False,
                expected_visual_change="Page may change",
                success_probability=0.5,
                potential_issues=["Unknown action type"]
            )

        prediction = StatePrediction(
            expected_changes=rules.get('expected_changes', []),
            expected_url_change=rules.get('url_change', False),
            expected_visual_change=rules.get('visual_change', ''),
            success_probability=rules.get('success_prob', 0.7),
            potential_issues=rules.get('issues', [])
        )

        # Adjust prediction based on action arguments
        prediction = self._adjust_for_arguments(prediction, action)

        return prediction

    def _adjust_for_arguments(
        self,
        prediction: StatePrediction,
        action: Dict
    ) -> StatePrediction:
        """Adjust prediction based on specific action arguments."""
        action_type = action.get('name', '').lower()
        args = action.get('arguments', {})

        if action_type == 'click':
            description = args.get('description', '').lower()
            reasoning = args.get('reasoning', '').lower()

            # Adjust for button-like clicks
            if any(word in description for word in ['button', 'submit', 'search', 'login']):
                prediction.success_probability = 0.8
                prediction.expected_url_change = True

            # Adjust for link clicks
            if any(word in description for word in ['link', 'href', 'navigate']):
                prediction.expected_url_change = True

            # Adjust for input field clicks
            if any(word in description for word in ['input', 'field', 'textbox']):
                prediction.expected_url_change = False
                prediction.expected_changes = ["Input field focused", "Ready for text input"]

        elif action_type == 'type':
            text = args.get('text', '')
            if len(text) > 100:
                prediction.potential_issues.append("Long text input may be truncated")

        elif action_type == 'scroll':
            direction = args.get('direction', 'down').lower()
            prediction.expected_visual_change = f"Page scrolls {direction}"

        elif action_type == 'stop':
            answer = args.get('answer', '')
            if not answer:
                prediction.success_probability = 0.5
                prediction.potential_issues.append("No answer provided")

        return prediction

    def find_similar_transitions(
        self,
        current_state: str,
        action: Optional[Dict],
        trajectory_store: 'TrajectoryStore' = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find historical state-action-state transitions similar to current situation.

        Args:
            current_state: Base64-encoded current screenshot
            action: Proposed action
            trajectory_store: Store to search for similar transitions
            top_k: Number of similar transitions to return

        Returns:
            List of similar historical transitions
        """
        # This could be enhanced with visual similarity search
        # For now, return empty - action-based matching is handled by predict_next_state
        return []

    def estimate_steps_remaining(
        self,
        task: str,
        steps_taken: int,
        success_trajectories: List['StoredTrajectory']
    ) -> Dict:
        """
        Estimate how many steps remain based on similar successful tasks.

        Args:
            task: Current task description
            steps_taken: Number of steps already taken
            success_trajectories: Similar successful trajectories

        Returns:
            Dict with estimated_remaining, min, max, confidence
        """
        if not success_trajectories:
            return {
                'estimated_remaining': 5,
                'min': 2,
                'max': 15,
                'confidence': 0.3
            }

        total_steps = [t.total_rounds for t in success_trajectories]
        avg_total = sum(total_steps) / len(total_steps)
        min_total = min(total_steps)
        max_total = max(total_steps)

        estimated_remaining = max(0, avg_total - steps_taken)

        return {
            'estimated_remaining': int(estimated_remaining),
            'min': max(0, min_total - steps_taken),
            'max': max(0, max_total - steps_taken),
            'confidence': 0.6 if len(success_trajectories) >= 3 else 0.4
        }

    def is_likely_stuck(
        self,
        action_history: List[str],
        step_num: int
    ) -> Dict:
        """
        Detect if the agent appears to be stuck in a loop.

        Args:
            action_history: List of action names taken
            step_num: Current step number

        Returns:
            Dict with is_stuck, reason, suggestion
        """
        if len(action_history) < 3:
            return {'is_stuck': False}

        # Check for immediate repetition
        if len(action_history) >= 3:
            last_three = action_history[-3:]
            if len(set(last_three)) == 1:
                return {
                    'is_stuck': True,
                    'reason': f"Repeating '{last_three[0]}' action",
                    'suggestion': "Try a different action or target"
                }

        # Check for longer cycle
        if len(action_history) >= 6:
            last_six = action_history[-6:]
            if last_six[:3] == last_six[3:]:
                return {
                    'is_stuck': True,
                    'reason': "Cycling through same sequence",
                    'suggestion': "Break the cycle with a different approach"
                }

        # Check for too many steps
        if step_num > 15:
            return {
                'is_stuck': True,
                'reason': "Excessive steps taken",
                'suggestion': "Consider if task is achievable or already complete"
            }

        return {'is_stuck': False}
