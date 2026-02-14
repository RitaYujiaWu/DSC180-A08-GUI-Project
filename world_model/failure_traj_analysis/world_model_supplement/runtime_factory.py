from __future__ import annotations

from typing import Any, Optional
from .vllm_openai_client import VLLMOpenAIClient
from .internal_world_model import InternalWorldModel, WorldModelConfig


def create_world_model_from_vllm(
    training_data_path: str,
    *,
    # choose which server to use for world-model text generation
    llm_base_url: str = "http://localhost:9100/v1",
    llm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    # if you want to use the grounding model instead:
    # llm_base_url="http://localhost:9101/v1",
    # llm_model="Tongyi-MiA/UI-Ins-7B",
    top_k: int = 3,
    multimodal: bool = True,
    use_step_guidance: bool = True,
    # IMPORTANT: GPUs 0-3 are used by vLLM in your setup; keep embed device on CPU unless you have spare GPU.
    embed_backend: str = "clip",
    device: str = "cpu",
    min_sim: float = 0.22,
    low_conf_words: int = 60,
    cache: bool = True,
    cache_max: int = 256,
    step_prompt_path: Optional[str] = None,
    contrastive_prompt_path: Optional[str] = None,
) -> InternalWorldModel:
    llm = VLLMOpenAIClient(base_url=llm_base_url, model=llm_model)

    cfg = WorldModelConfig(
        training_data_path=training_data_path,
        top_k=top_k,
        multimodal=multimodal,
        use_step_guidance=use_step_guidance,
        embed_backend=embed_backend,
        device=device,
        min_similarity_for_full_guidance=min_sim,
        low_confidence_max_words=low_conf_words,
        enable_cache=cache,
        cache_max_items=cache_max,
        step_guidance_template_path=step_prompt_path,
        contrastive_summary_template_path=contrastive_prompt_path,
    )

    return InternalWorldModel(config=cfg, tool_llm=llm)
