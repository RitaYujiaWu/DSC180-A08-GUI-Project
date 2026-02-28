from dataclasses import dataclass, field
from typing import List, Optional

import torch

from src.policy_hf import HFPolicy, PolicyConfig


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
    freeze_vision_encoder: bool = True
    vision_param_keywords: List[str] = field(
        default_factory=lambda: [
            "vision",
            "visual",
            "vision_tower",
            "image_tower",
            "vit",
        ]
    )


class HFPolicyLoRA(HFPolicy):
    """
    LoRA-enabled policy with the same public API as HFPolicy.
    Keeps train.py changes minimal by reusing act/evaluate/update/save logic.
    """

    def __init__(self, cfg: PolicyConfig, lr: float, history_n: int = 4, lora_cfg: Optional[LoRAConfig] = None):
        super().__init__(cfg=cfg, lr=lr, history_n=history_n)
        self._apply_lora(lora_cfg or LoRAConfig())
        self.opt = torch.optim.AdamW(
            [p for p in self.net.parameters() if p.requires_grad],
            lr=lr,
        )

    def _apply_lora(self, lora_cfg: LoRAConfig) -> None:
        try:
            from peft import LoraConfig as PeftLoraConfig
            from peft import TaskType, get_peft_model
        except Exception as e:
            raise RuntimeError(
                "LoRA training requires `peft`. Please install it, e.g. `pip install peft`."
            ) from e

        net_unwrapped = self._unwrap_net()

        peft_cfg = PeftLoraConfig(
            r=int(lora_cfg.r),
            lora_alpha=int(lora_cfg.alpha),
            lora_dropout=float(lora_cfg.dropout),
            target_modules=list(lora_cfg.target_modules),
            bias=str(lora_cfg.bias),
            task_type=TaskType.CAUSAL_LM,
        )

        net_unwrapped.model = get_peft_model(net_unwrapped.model, peft_cfg)

        # PEFT wrapping can affect gradient checkpointing; re-apply if enabled.
        try:
            if bool(getattr(self.cfg, "gradient_checkpointing", True)):
                if hasattr(net_unwrapped.model, "gradient_checkpointing_enable"):
                    net_unwrapped.model.gradient_checkpointing_enable()
                if hasattr(net_unwrapped.model, "enable_input_require_grads"):
                    net_unwrapped.model.enable_input_require_grads()
        except Exception:
            pass

        if bool(lora_cfg.freeze_vision_encoder):
            self._freeze_vision_encoder_params(
                net_unwrapped,
                vision_param_keywords=list(lora_cfg.vision_param_keywords),
            )

        if hasattr(net_unwrapped.model, "print_trainable_parameters"):
            net_unwrapped.model.print_trainable_parameters()

    def _freeze_vision_encoder_params(self, net_unwrapped, vision_param_keywords: List[str]) -> None:
        keywords = [k.lower() for k in vision_param_keywords if isinstance(k, str) and k.strip()]
        if not keywords:
            return

        for name, p in net_unwrapped.model.named_parameters():
            lname = name.lower()
            if any(k in lname for k in keywords):
                p.requires_grad = False
