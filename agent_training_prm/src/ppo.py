# src/ppo.py
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
import torch


@dataclass
class PPOConfig:
    gamma: float
    gae_lambda: float
    clip_ratio: float
    vf_coef: float
    ent_coef: float
    train_epochs: int
    minibatch_size: int
    value_clip: bool = True  # extra stabilizer


def compute_gae(rewards, values, dones, gamma, lam):
    """
    rewards, values, dones are 1D arrays length T
    values are V(s_t); dones indicates termination at t
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = 0.0 if (t == T - 1) else values[t + 1]
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    ret = adv + values
    return adv, ret


def ppo_update(
    policy,
    ppo_cfg: PPOConfig,
    batch: Dict[str, Any],
    grad_clip_norm: float,
) -> Dict[str, float]:
    """
    Expects batch to contain:
      - old_logprob: Tensor [B]
      - old_value:   Tensor [B]
      - advantage:   Tensor [B]
      - returns:     Tensor [B]
      - prompts:     List[HF chat messages] length B
      - actions:     List[str] length B
      - screenshots: Optional[List[Optional[bytes]]] length B (fallback only)
      - images_lists: Optional[List[List[PIL.Image]]] length B (PREFERRED for placeholder-image prompts)
    """
    old_logprob: torch.Tensor = batch["old_logprob"]
    old_value: torch.Tensor = batch["old_value"]
    adv: torch.Tensor = batch["advantage"]
    ret: torch.Tensor = batch["returns"]

    prompts = batch["prompts"]
    actions: List[str] = batch["actions"]

    screenshots: Optional[List[Optional[bytes]]] = batch.get("screenshots", None)
    images_lists = batch.get("images_lists", None)  # List[List[PIL.Image]] or None

    assert len(prompts) == len(actions), "prompts/actions length mismatch"
    B = int(old_logprob.shape[0])
    assert len(prompts) == B, f"prompts length {len(prompts)} != batch size {B}"

    if screenshots is not None:
        assert len(screenshots) == B, f"screenshots length {len(screenshots)} != batch size {B}"
    if images_lists is not None:
        assert len(images_lists) == B, f"images_lists length {len(images_lists)} != batch size {B}"

    device = old_logprob.device

    # Normalize advantages
    adv = adv.float()
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    adv = adv.detach()

    old_logprob = old_logprob.float()
    old_value = old_value.float()
    ret = ret.float()

    last_stats: Dict[str, float] = {}

    train_epochs = max(1, int(ppo_cfg.train_epochs))
    mb = max(1, int(ppo_cfg.minibatch_size))

    for _epoch in range(train_epochs):
        idx = torch.randperm(B, device=device)

        for start in range(0, B, mb):
            mb_idx = idx[start : start + mb]
            mb_idx_list = mb_idx.tolist()

            mb_prompts = [prompts[i] for i in mb_idx_list]
            mb_actions = [actions[i] for i in mb_idx_list]

            if screenshots is not None:
                mb_screens = [screenshots[i] for i in mb_idx_list]
            else:
                mb_screens = [None] * len(mb_idx_list)

            mb_images_lists = None
            if images_lists is not None:
                mb_images_lists = [images_lists[i] for i in mb_idx_list]

            mb_old_logprob = old_logprob[mb_idx]
            mb_old_value = old_value[mb_idx]
            mb_adv = adv[mb_idx]
            mb_ret = ret[mb_idx]

            # Evaluate current policy on minibatch
            # IMPORTANT: for placeholder-image prompts, pass images_list_per_prompt.
            new_logprob, new_value, entropy = policy.evaluate(
                prompts=mb_prompts,
                actions=mb_actions,
                screenshots=mb_screens,
                images_list_per_prompt=mb_images_lists,
            )

            new_logprob = new_logprob.float()
            new_value = new_value.float()
            entropy = entropy.float()

            # PPO ratio
            ratio = torch.exp(new_logprob - mb_old_logprob)

            unclipped = ratio * mb_adv
            clipped = torch.clamp(
                ratio,
                1.0 - float(ppo_cfg.clip_ratio),
                1.0 + float(ppo_cfg.clip_ratio),
            ) * mb_adv
            policy_loss = -torch.mean(torch.min(unclipped, clipped))

            # Value loss
            if ppo_cfg.value_clip:
                v_clipped = mb_old_value + torch.clamp(
                    new_value - mb_old_value,
                    -float(ppo_cfg.clip_ratio),
                    float(ppo_cfg.clip_ratio),
                )
                v_loss_1 = (new_value - mb_ret) ** 2
                v_loss_2 = (v_clipped - mb_ret) ** 2
                value_loss = 0.5 * torch.mean(torch.max(v_loss_1, v_loss_2))
            else:
                value_loss = 0.5 * torch.mean((new_value - mb_ret) ** 2)

            # Entropy bonus (maximize entropy => subtract negative mean entropy)
            ent_loss = -torch.mean(entropy)

            loss = policy_loss + float(ppo_cfg.vf_coef) * value_loss + float(ppo_cfg.ent_coef) * ent_loss

            upd = policy.update(loss, grad_clip_norm=grad_clip_norm)

            with torch.no_grad():
                last_stats = {
                    **upd,
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.mean().item()),
                    "ratio_mean": float(ratio.mean().item()),
                    "ratio_min": float(ratio.min().item()),
                    "ratio_max": float(ratio.max().item()),
                    "loss_total": float(loss.item()),
                }

    return last_stats