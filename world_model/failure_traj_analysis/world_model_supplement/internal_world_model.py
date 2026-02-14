"""
Implementation:
- Contrastive memory: retrieve similar SUCCESS and FAILURE trajectories (FAISS over embeddings)
- Contrastive analysis: generate structured guidance using provided prompt templates
- Step guidance: re-retrieve per step using current visual state and action history
- Optional integration hooks for existing offline error analyzers in this folder

Design goals:
- Pluggable LLM interface and embedding backend
- Robust to missing optional dependencies (faiss/torch/transformers)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import json
import time
import hashlib

from .llm_utils import LLMClient, ensure_llm_client
from .prompt_templates import PromptTemplates
from .contrastive_memory import ContrastiveTrajectoryStore, RetrievedTrajectory
from .contrastive_analyzer import ContrastiveAnalyzer, ContrastiveInsight
from .guidance_generator import GuidanceGenerator, GuidancePacket
from .utils import hash_bytes, safe_b64_to_pil, coerce_action_history_text


@dataclass
class WorldModelConfig:
    """
    training_data_path layout expected:
      training_data/{dataset}/{domain}/success/*.jsonl
      training_data/{dataset}/{domain}/failure/*.jsonl
    """
    training_data_path: str

    # retrieval
    top_k: int = 3
    multimodal: bool = True  # use screenshot embeddings if available
    use_step_guidance: bool = True

    # embedding
    embed_backend: str = "clip"  # "clip" (default) or "latent" (stub)
    device: str = "cpu"         # "cpu" or "cuda" if available

    # gating (reduce distracting guidance when retrieval is low-confidence)
    min_similarity_for_full_guidance: float = 0.22  # cosine/IP after normalization
    low_confidence_max_words: int = 60

    # caching (step guidance can be expensive)
    enable_cache: bool = True
    cache_max_items: int = 256

    # prompts (optional override)
    step_guidance_template_path: Optional[str] = None
    contrastive_summary_template_path: Optional[str] = None


class InternalWorldModel:
    """
    Orchestrates:
      retrieve -> analyze -> summarize guidance
    """
    def __init__(
        self,
        config: WorldModelConfig,
        tool_llm: Any,
        templates: Optional[PromptTemplates] = None,
    ) -> None:
        self.cfg = config
        self.llm: LLMClient = ensure_llm_client(tool_llm)

        self.templates = templates or PromptTemplates.from_files_or_defaults(
            step_path=self.cfg.step_guidance_template_path,
            contrastive_path=self.cfg.contrastive_summary_template_path,
        )

        self.store = ContrastiveTrajectoryStore(
            training_data_path=self.cfg.training_data_path,
            top_k=self.cfg.top_k,
            multimodal=self.cfg.multimodal,
            embed_backend=self.cfg.embed_backend,
            device=self.cfg.device,
        )

        from .action_ranker import ActionRanker
        self.action_ranker = ActionRanker(
            store=self.store,
            top_k=self.cfg.top_k,
            agg="max",  # you can change to "mean"
            min_sim=self.cfg.min_similarity_for_full_guidance,
            min_margin=float(getattr(self.cfg, "min_action_margin", 0.05)),
            ambiguous_margin=float(getattr(self.cfg, "ambiguous_action_margin", 0.02)),
        )


        self.analyzer = ContrastiveAnalyzer(llm=self.llm, templates=self.templates)
        self.guidance = GuidanceGenerator(
            templates=self.templates,
            min_sim=self.cfg.min_similarity_for_full_guidance,
            low_confidence_max_words=self.cfg.low_confidence_max_words,
        )

        self._cache: Dict[str, GuidancePacket] = {}
        self._cache_order: List[str] = []

        self.stats: Dict[str, Any] = {
            "retrieval_calls": 0,
            "analysis_calls": 0,
            "cache_hits": 0,
            "last_runtime_sec": None,
        }

    # -------------------------
    # Public API (Step 0)
    # -------------------------
    def get_initial_guidance(
        self,
        task: str,
        initial_screenshot_b64: Optional[str],
        domain: str,
        dataset: str,
    ) -> GuidancePacket:
        """
        Retrieve from success+failure memories using (task + initial screenshot),
        then generate a contrastive summary (<= ~200 words by prompt contract).
        """
        start = time.time()

        img = safe_b64_to_pil(initial_screenshot_b64) if initial_screenshot_b64 else None
        succ, fail, min_sim = self.store.retrieve(
            task=task,
            domain=domain,
            dataset=dataset,
            query_image=img,
        )
        self.stats["retrieval_calls"] += 1

        insights = self.analyzer.contrastive_summary(
            task=task,
            domain=domain,
            success_trajs=succ,
            failure_trajs=fail,
        )
        self.stats["analysis_calls"] += 1

        packet = self.guidance.format_initial(
            task=task,
            domain=domain,
            success_trajs=succ,
            failure_trajs=fail,
            insights=insights,
            min_similarity=min_sim,
        )

        self.stats["last_runtime_sec"] = round(time.time() - start, 4)
        return packet

    # -------------------------
    # Public API (Step N)
    # -------------------------
    def get_step_guidance(
        self,
        task: str,
        current_screenshot_b64: Optional[str],
        action_history: Union[List[Any], str, None],
        step_num: int,
        domain: str,
        dataset: str,
        state_description: Optional[str] = None,
        avg_steps: Optional[int] = None,
    ) -> GuidancePacket:
        """
        Dynamic retrieval per step using current screenshot.
        Produces <= ~150 words guidance by prompt contract.
        """
        if not self.cfg.use_step_guidance:
            return GuidancePacket(
                guidance_text="",
                meta={"disabled": True, "mode": "step"},
            )

        start = time.time()
        img = safe_b64_to_pil(current_screenshot_b64) if current_screenshot_b64 else None
        action_text = coerce_action_history_text(action_history)

        cache_key = None
        if self.cfg.enable_cache:
            cache_key = self._make_cache_key(
                task=task,
                dataset=dataset,
                domain=domain,
                step_num=step_num,
                screenshot_b64=current_screenshot_b64,
                action_text=action_text,
            )
            if cache_key in self._cache:
                self.stats["cache_hits"] += 1
                self.stats["last_runtime_sec"] = round(time.time() - start, 4)
                return self._cache[cache_key]

        succ, fail, min_sim = self.store.retrieve(
            task=task,
            domain=domain,
            dataset=dataset,
            query_image=img,
        )
        self.stats["retrieval_calls"] += 1

        # For step guidance, we don’t need full LLM contrastive JSON every time.
        # We use a lightweight summary-driven step prompt; optional analysis can be enabled later.
        packet = self.guidance.format_step(
            task=task,
            step_num=step_num,
            action_history_text=action_text,
            state_description=state_description or "",
            success_trajs=succ,
            failure_trajs=fail,
            min_similarity=min_sim,
            avg_steps=avg_steps,
            llm=self.llm,
        )

        self.stats["analysis_calls"] += 1

        if self.cfg.enable_cache and cache_key:
            self._cache_put(cache_key, packet)

        self.stats["last_runtime_sec"] = round(time.time() - start, 4)
        return packet

    # -------------------------
    # Optional integration hooks
    # -------------------------
    def run_offline_error_analysis(
        self,
        html_path: str,
        output_dir: str,
        llm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_steps: int = 30,
        env: str = "webvoyager",
    ) -> Dict[str, Any]:
        
        os.makedirs(output_dir, exist_ok=True)

        from ..fine_grained_analysis import ErrorTypeDetector
        from ..critical_error_detection import CriticalErrorAnalyzer

        detector = ErrorTypeDetector(model_name=llm_model_name)
        phase1 = detector.process_trajectory(
            html_path=html_path,
            output_dir=output_dir,
            max_steps=max_steps,
            env=env,
        )

        analyzer = CriticalErrorAnalyzer(model_name=llm_model_name)
        critical = analyzer.process_trajectory(
            phase1_file=phase1["output_file"],
            original_trajectory_file=html_path,
            output_dir=output_dir,
        )

        return {"phase1": phase1, "critical": critical}

    # -------------------------
    # Cache helpers
    # -------------------------
    def _make_cache_key(
        self,
        task: str,
        dataset: str,
        domain: str,
        step_num: int,
        screenshot_b64: Optional[str],
        action_text: str,
    ) -> str:
        h = hashlib.sha256()
        h.update(task.encode("utf-8"))
        h.update(dataset.encode("utf-8"))
        h.update(domain.encode("utf-8"))
        h.update(str(step_num).encode("utf-8"))
        if screenshot_b64:
            h.update(hash_bytes(screenshot_b64.encode("utf-8")).encode("utf-8"))
        h.update(hash_bytes(action_text.encode("utf-8")).encode("utf-8"))
        return h.hexdigest()

    def _cache_put(self, key: str, value: GuidancePacket) -> None:
        if key in self._cache:
            return
        self._cache[key] = value
        self._cache_order.append(key)
        if len(self._cache_order) > self.cfg.cache_max_items:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)

    def reset(self) -> None:
        self._cache.clear()
        self._cache_order.clear()
        self.stats.update({"retrieval_calls": 0, "analysis_calls": 0, "cache_hits": 0})

    def rank_candidate_actions(
        self,
        *,
        task: str,
        domain: str,
        dataset: str,
        candidate_actions: List[Any],
        current_screenshot_b64: Optional[str] = None,
        action_history: Any = None,
        state_description: str = "",
        use_image: bool = True,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Mentor-style reranking:
          prefer actions with strong success evidence and weak failure evidence.
        Uses a margin score: success_evidence - failure_evidence.
        Applies confidence gating when evidence is weak/ambiguous.
        """
        return self.action_ranker.rank_actions(
            task=task,
            dataset=dataset,
            domain=domain,
            candidate_actions=candidate_actions,
            current_screenshot_b64=current_screenshot_b64,
            action_history=action_history,
            state_description=state_description,
            use_image=use_image,
        )



def create_internal_world_model(args: Any, tool_llm: Any) -> InternalWorldModel:
    """
    Factory that plays nicely with argparse-like args.
    Only reads attributes if present; uses defaults otherwise.
    """
    cfg = WorldModelConfig(
        training_data_path=getattr(args, "world_model_data_path", getattr(args, "training_data_path", "training_data")),
        top_k=int(getattr(args, "world_model_top_k", 3)),
        multimodal=bool(getattr(args, "world_model_multimodal", True)),
        use_step_guidance=bool(getattr(args, "world_model_step_guidance", True)),
        embed_backend=str(getattr(args, "world_model_embed_backend", "clip")),
        device=str(getattr(args, "world_model_device", "cpu")),
        min_similarity_for_full_guidance=float(getattr(args, "world_model_min_sim", 0.22)),
        low_confidence_max_words=int(getattr(args, "world_model_low_conf_words", 60)),
        enable_cache=bool(getattr(args, "world_model_cache", True)),
        cache_max_items=int(getattr(args, "world_model_cache_max", 256)),
        step_guidance_template_path=getattr(args, "world_model_step_prompt", None),
        contrastive_summary_template_path=getattr(args, "world_model_contrastive_prompt", None),
    )
    return InternalWorldModel(config=cfg, tool_llm=tool_llm)

    
