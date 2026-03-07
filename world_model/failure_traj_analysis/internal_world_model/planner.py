from __future__ import annotations

from typing import Any, Dict, List, Optional

from .candidate_generator import CandidateGenerator
from .candidate_scorer import CandidateScorer
from .contrastive_retriever import ContrastiveRetriever
from .schemas import PlannerOutput


class InternalWorldModelPlanner:
    """
    Main orchestration module for think-before-acting.

    Pipeline:
    1. Generate candidate next actions
    2. Retrieve contrastive success/failure memories
    3. Score candidates
    4. Select the best candidate
    """

    def __init__(
        self,
        agent: Any,
        reasoning_bank: Any = None,
        args: Optional[Any] = None,
    ) -> None:
        self.agent = agent
        self.reasoning_bank = reasoning_bank
        self.args = args

        k_candidates = getattr(args, "iwm_k_candidates", 3) if args is not None else 3
        topk_success = getattr(args, "iwm_topk_success", 3) if args is not None else 3
        topk_failure = getattr(args, "iwm_topk_failure", 3) if args is not None else 3
        use_multimodal = getattr(args, "iwm_use_multimodal", True) if args is not None else True
        score_method = getattr(args, "iwm_score_method", "heuristic") if args is not None else "heuristic"
        alpha = getattr(args, "iwm_alpha", 1.0) if args is not None else 1.0
        beta = getattr(args, "iwm_beta", 1.0) if args is not None else 1.0

        llm = self._resolve_llm(agent)

        self.retriever = ContrastiveRetriever(
            reasoning_bank=reasoning_bank,
            topk_success=topk_success,
            topk_failure=topk_failure,
            use_multimodal=use_multimodal,
        )
        self.generator = CandidateGenerator(
            llm=llm,
            k_candidates=k_candidates,
        )
        self.scorer = CandidateScorer(
            llm=llm,
            score_method=score_method,
            alpha=alpha,
            beta=beta,
        )

    def select_action(
        self,
        messages: Optional[List[Dict[str, Any]]],
        trajectory: Optional[Any],
        intent: str,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> PlannerOutput:
        meta_data = meta_data or {}

        try:
            current_image = self._extract_current_image(meta_data=meta_data, trajectory=trajectory)
            current_url = meta_data.get("url") or meta_data.get("current_url")
            observation_summary = self._extract_observation_summary(messages)

            candidates = self.generator.generate(
                intent=intent,
                trajectory=trajectory,
                meta_data=meta_data,
                messages=messages,
            )

            if not candidates:
                return PlannerOutput(
                    selected_action_text=None,
                    debug_info={"error": "No candidate actions generated."},
                )

            success_cases, failure_cases = self.retriever.retrieve(
                intent=intent,
                trajectory=trajectory,
                meta_data=meta_data,
                current_image=current_image,
            )

            candidate_scores = self.scorer.score_candidates(
                candidates=candidates,
                success_cases=success_cases,
                failure_cases=failure_cases,
                intent=intent,
                current_url=current_url,
                observation_summary=observation_summary,
            )

            best_score = self.scorer.select_best(candidate_scores)
            selected_action_text = best_score.candidate.text if best_score is not None else None

            debug_info = {
                "candidates": [c.text for c in candidates],
                "selected_action": selected_action_text,
                "retrieved_success_ids": [c.trajectory_id for c in success_cases],
                "retrieved_failure_ids": [c.trajectory_id for c in failure_cases],
                "candidate_scores": [
                    {
                        "candidate": cs.candidate.text,
                        "success_score": cs.success_score,
                        "failure_score": cs.failure_score,
                        "final_score": cs.final_score,
                        "reasoning": cs.reasoning,
                    }
                    for cs in candidate_scores
                ],
            }

            return PlannerOutput(
                selected_action_text=selected_action_text,
                candidate_scores=candidate_scores,
                retrieved_successes=success_cases,
                retrieved_failures=failure_cases,
                debug_info=debug_info,
            )

        except Exception as exc:
            return PlannerOutput(
                selected_action_text=None,
                debug_info={"error": f"Internal world model planner failure: {exc}"},
            )

    def _resolve_llm(self, agent: Any) -> Any:
        """
        Best-effort resolution of the existing acting LLM from the agent.
        Adjust later if your repo uses a more specific field.
        """
        for attr in ("llm", "model", "policy_llm", "vlm", "engine"):
            if hasattr(agent, attr):
                return getattr(agent, attr)
        return None

    def _extract_current_image(
        self,
        meta_data: Optional[Dict[str, Any]] = None,
        trajectory: Optional[Any] = None,
    ) -> Optional[Any]:
        meta_data = meta_data or {}

        # best-effort image lookup from common field names
        for key in ("current_image", "image", "screenshot", "observation_image"):
            if key in meta_data and meta_data[key] is not None:
                return meta_data[key]

        if isinstance(trajectory, dict):
            for key in ("current_image", "image", "screenshot"):
                if key in trajectory and trajectory[key] is not None:
                    return trajectory[key]

        if hasattr(trajectory, "current_image"):
            return getattr(trajectory, "current_image")

        return None

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