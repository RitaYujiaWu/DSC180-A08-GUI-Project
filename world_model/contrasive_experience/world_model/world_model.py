"""
WorldModel: Main interface for state prediction, action guidance, and contrastive learning.

Orchestrates TrajectoryStore, TrajectoryAnalyzer, ContrastiveSummarizer, and StatePredictor
to provide comprehensive guidance to the GUI agent at every step.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_model.trajectory_store import TrajectoryStore, StoredTrajectory
from world_model.trajectory_analyzer import TrajectoryAnalyzer, ContrastiveInsight
from world_model.contrastive_summarizer import ContrastiveSummarizer
from world_model.state_predictor import StatePredictor, StatePrediction

logger = logging.getLogger(__name__)


class WorldModel:
    """
    Main interface for the World Model system.

    Provides:
    - Contrastive learning from success/failure trajectories
    - State prediction for proposed actions
    - Text summaries injected into agent prompts
    - Guidance at task start and every step

    Usage:
        world_model = WorldModel(args, trajectory_store, tool_llm)

        # At task start
        initial_guidance = world_model.get_initial_guidance(task, screenshot, domain)

        # At each step
        step_guidance = world_model.get_step_guidance(task, screenshot, action_history, step_num)

        # Optionally predict action outcome
        prediction = world_model.predict_action_outcome(screenshot, action)
    """

    def __init__(
        self,
        args: argparse.Namespace,
        trajectory_store: TrajectoryStore,
        tool_llm: 'DirectVLLMModel',
        prompts_dir: str = None
    ):
        """
        Initialize the World Model.

        Args:
            args: Configuration arguments (use_world_model, world_model_top_k, etc.)
            trajectory_store: FAISS-indexed trajectory storage
            tool_llm: LLM for generating summaries and analysis
            prompts_dir: Directory containing prompt templates
        """
        self.args = args
        self.trajectory_store = trajectory_store
        self.tool_llm = tool_llm
        self.prompts_dir = prompts_dir or os.path.join(
            os.path.dirname(__file__), 'prompts'
        )

        # Initialize sub-components
        self.analyzer = TrajectoryAnalyzer(trajectory_store, tool_llm)
        self.predictor = StatePredictor(tool_llm, self.prompts_dir)
        self.summarizer = ContrastiveSummarizer(tool_llm, self.prompts_dir)

        # Cache for current task
        self._cached_task: Optional[str] = None
        self._cached_domain: Optional[str] = None
        self._cached_initial_guidance: Optional[str] = None
        self._retrieved_trajectories: Optional[Dict] = None
        self._cached_insights: List[ContrastiveInsight] = []

        # Configuration
        self._top_k = getattr(args, 'world_model_top_k', 3)
        self._step_guidance_enabled = getattr(args, 'world_model_step_guidance', True)

        logger.info("WorldModel initialized successfully")
        logger.info(f"  - Trajectory store: {trajectory_store.get_stats()}")
        logger.info(f"  - Top-k retrieval: {self._top_k}")
        logger.info(f"  - Step guidance: {self._step_guidance_enabled}")

    def get_initial_guidance(
        self,
        task: str,
        initial_screenshot: str = None,
        domain: str = None,
        dataset: str = None
    ) -> str:
        """
        Generate guidance before the first action.

        Called once at the start of a task to:
        1. Retrieve similar success/failure trajectories
        2. Analyze contrastive patterns
        3. Generate initial guidance summary

        Args:
            task: Task description
            initial_screenshot: Base64-encoded initial screenshot
            domain: Optional domain filter (e.g., "shopping", "wikipedia")
            dataset: Optional dataset filter (e.g., "mmina", "webvoyager")

        Returns:
            Text summary (~200 tokens) to inject into agent system prompt
        """
        logger.info(f"Generating initial guidance for task: {task[:50]}...")

        # Retrieve contrastive pairs using prefixed query format
        success_trajs, failure_trajs = self.trajectory_store.retrieve_contrastive_pairs(
            query_task=task,
            query_image=initial_screenshot,
            domain=domain,
            dataset=dataset,
            top_k=self._top_k
        )

        logger.info(f"Retrieved {len(success_trajs)} success, {len(failure_trajs)} failure trajectories")

        # Handle success-only mode (fallback when no failures available)
        if not failure_trajs and success_trajs:
            logger.info("No failure trajectories available - using success-only mode")
            guidance = self.summarizer.summarize_success_only_guidance(
                task=task,
                success_trajectories=success_trajs,
                step_num=0
            )
            self._cache_results(task, domain, success_trajs, [], [], guidance)
            return guidance

        # Handle no data case
        if not success_trajs and not failure_trajs:
            logger.warning("No trajectories found - returning empty guidance")
            self._cache_results(task, domain, [], [], [], "")
            return ""

        # Analyze contrastive pairs
        insights = self.analyzer.analyze_contrastive_pairs(
            success_trajectories=success_trajs,
            failure_trajectories=failure_trajs,
            task=task
        )

        logger.info(f"Generated {len(insights)} contrastive insights")

        # Extract patterns
        success_patterns = self.analyzer.extract_success_patterns(success_trajs)
        failure_patterns = self.analyzer.extract_failure_patterns(failure_trajs)

        # Generate summary
        guidance = self.summarizer.summarize_initial_guidance(
            task=task,
            contrastive_insights=insights,
            success_patterns=success_patterns,
            failure_patterns=failure_patterns
        )

        # Cache results for step guidance
        self._cache_results(task, domain, success_trajs, failure_trajs, insights, guidance)

        return guidance

    def _cache_results(
        self,
        task: str,
        domain: str,
        success_trajs: List[StoredTrajectory],
        failure_trajs: List[StoredTrajectory],
        insights: List[ContrastiveInsight],
        guidance: str
    ):
        """Cache results for use in step guidance."""
        self._cached_task = task
        self._cached_domain = domain
        self._cached_initial_guidance = guidance
        self._cached_insights = insights
        self._retrieved_trajectories = {
            'success': success_trajs,
            'failure': failure_trajs
        }

    def get_step_guidance(
        self,
        task: str,
        current_state: str,
        action_history: List[str],
        step_num: int,
        domain: str = None,
        dataset: str = None
    ) -> str:
        """
        Generate guidance at each step with DYNAMIC retrieval based on current screenshot.

        Unlike initial guidance which uses the first screenshot, step guidance
        re-retrieves similar trajectories using the CURRENT screenshot to find
        examples relevant to the current page state.

        Args:
            task: Task description
            current_state: Base64-encoded CURRENT screenshot (changes each step)
            action_history: List of previous action names
            step_num: Current step number (0-indexed)
            domain: Domain filter (e.g., "shopping", "Amazon")
            dataset: Dataset filter (e.g., "mmina", "webvoyager")

        Returns:
            Text summary to inject into agent user message
        """
        if not self._step_guidance_enabled:
            return ""

        logger.info(f"Generating step {step_num} guidance with current screenshot...")

        # DYNAMIC RETRIEVAL: Re-retrieve using current screenshot (Option B)
        # This finds trajectories similar to the CURRENT state, not initial state
        success_trajs, failure_trajs = self.trajectory_store.retrieve_contrastive_pairs(
            query_task=task,
            query_image=current_state,  # Current screenshot, not initial!
            domain=domain or self._cached_domain,
            dataset=dataset,
            top_k=self._top_k
        )

        logger.info(f"Step {step_num}: Retrieved {len(success_trajs)} success, {len(failure_trajs)} failure trajectories")

        # Check for stuck state
        stuck_info = self.predictor.is_likely_stuck(action_history, step_num)

        # Analyze contrastive pairs for current state
        if success_trajs or failure_trajs:
            insights = self.analyzer.analyze_contrastive_pairs(
                success_trajectories=success_trajs,
                failure_trajectories=failure_trajs,
                task=task
            )
        else:
            # Fall back to cached insights if no new retrieval
            insights = self._cached_insights

        # Generate step guidance with fresh insights
        guidance = self.summarizer.summarize_step_guidance(
            task=task,
            current_state_description="Current page",
            action_history=action_history,
            step_num=step_num,
            relevant_insights=insights
        )

        # Add stuck warning if needed
        if stuck_info.get('is_stuck', False):
            guidance += f"\n\n**ALERT:** {stuck_info.get('reason')}. {stuck_info.get('suggestion', '')}"

        return guidance

    def predict_action_outcome(
        self,
        current_state: str,
        proposed_action: Dict
    ) -> StatePrediction:
        """
        Predict the outcome of a proposed action.

        Args:
            current_state: Base64-encoded current screenshot
            proposed_action: Action dict with 'name' and 'arguments'

        Returns:
            StatePrediction with expected_changes, success_probability, etc.
        """
        prediction = self.predictor.predict_next_state(
            current_state=current_state,
            action=proposed_action,
            task=self._cached_task
        )

        return prediction

    def get_action_prediction_text(
        self,
        current_state: str,
        proposed_action: Dict
    ) -> str:
        """
        Get text summary of action prediction for prompt injection.

        Args:
            current_state: Base64-encoded current screenshot
            proposed_action: Action to evaluate

        Returns:
            Brief text summary of predicted outcome
        """
        prediction = self.predict_action_outcome(current_state, proposed_action)
        return self.summarizer.summarize_action_prediction(
            proposed_action=proposed_action,
            prediction=prediction
        )

    def estimate_completion(
        self,
        task: str,
        steps_taken: int
    ) -> Dict:
        """
        Estimate how close the task is to completion.

        Args:
            task: Task description
            steps_taken: Number of steps already taken

        Returns:
            Dict with estimated_remaining, confidence, etc.
        """
        success_trajs = self._retrieved_trajectories.get('success', []) if self._retrieved_trajectories else []
        return self.predictor.estimate_steps_remaining(task, steps_taken, success_trajs)

    def reset(self):
        """Reset cached data for a new task."""
        logger.info("Resetting WorldModel cache")
        self._cached_task = None
        self._cached_domain = None
        self._cached_initial_guidance = None
        self._retrieved_trajectories = None
        self._cached_insights = []

    def get_compact_guidance(self, max_tokens: int = 100) -> str:
        """
        Get compact guidance for constrained contexts.

        Args:
            max_tokens: Approximate token limit

        Returns:
            Compact guidance string
        """
        if not self._cached_insights:
            return ""
        return self.summarizer.format_compact_guidance(self._cached_insights, max_tokens)

    def get_stats(self) -> Dict:
        """Get statistics about current world model state."""
        return {
            'trajectory_store': self.trajectory_store.get_stats(),
            'cached_task': self._cached_task[:50] if self._cached_task else None,
            'cached_domain': self._cached_domain,
            'num_insights': len(self._cached_insights),
            'retrieved_success': len(self._retrieved_trajectories.get('success', [])) if self._retrieved_trajectories else 0,
            'retrieved_failure': len(self._retrieved_trajectories.get('failure', [])) if self._retrieved_trajectories else 0
        }


def create_world_model(args: argparse.Namespace, tool_llm: 'DirectVLLMModel') -> WorldModel:
    """
    Factory function to create a WorldModel instance.

    Args:
        args: Configuration with world_model_data_path, world_model_index_path, etc.
        tool_llm: LLM for analysis and summarization

    Returns:
        Configured WorldModel instance
    """
    trajectory_store = TrajectoryStore(
        training_data_path=getattr(args, 'world_model_data_path', 'training_data'),
        faiss_index_path=getattr(args, 'world_model_index_path', None),
        multimodal=getattr(args, 'world_model_multimodal', True)
    )

    return WorldModel(
        args=args,
        trajectory_store=trajectory_store,
        tool_llm=tool_llm
    )
