# src/train.py
import os
import sys
import argparse
from typing import Any, Dict, List, Optional
import time
from PIL import Image

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator

# Make sure "src" is importable even when running as a script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import set_seed, load_yaml, JSONLLogger, now_ts
from src.osworld_tasks import iter_tasks
from src.osworld_env import OSWorldEnv, EnvConfig
from src.prm import PRMClient, PRMConfig, StepHistory
from src.policy_hf import HFPolicy, PolicyConfig
from src.ppo import PPOConfig, compute_gae, ppo_update


def _obs_for_history(obs: Any, max_chars: int = 800) -> str:
    """Keep obs compact (do not inline PNG bytes)."""
    if isinstance(obs, dict):
        o = dict(obs)
        ss = o.get("screenshot", None)
        if isinstance(ss, (bytes, bytearray)):
            o["screenshot"] = f"<bytes:{len(ss)}>"
        return str(o)[:max_chars]
    return str(obs)[:max_chars]


def _get_screenshot_bytes(obs: Any) -> Optional[bytes]:
    """Extract screenshot bytes from OSWorld obs dict."""
    if isinstance(obs, dict):
        ss = obs.get("screenshot", None)
        if isinstance(ss, (bytes, bytearray)):
            return bytes(ss)
    return None


def _save_screenshot_png(ss: Optional[bytes], path: str) -> Optional[str]:
    if not ss:
        return None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(ss)
        return path
    except Exception:
        return None


def history_to_text(hist: StepHistory, k: int) -> str:
    """Text view used by policy prompt; show last k steps in text form."""
    w = hist.window(k)
    lines = []
    n = len(w["obs"])
    for i in range(n):
        lines.append(f"- obs: {w['obs'][i]}")
        lines.append(f"  act: {w['act'][i]}")
        lines.append(f"  res: {w['res'][i]}")
    return "\n".join(lines[-12:])  # cap verbosity


def _broadcast_obj(accelerator: Accelerator, obj: Any, src: int = 0) -> Any:
    """
    Broadcast a Python object from `src` rank to all ranks.

    IMPORTANT:
    - Your Accelerate version does NOT have Accelerator.broadcast_object_list().
    - Use torch.distributed.broadcast_object_list instead.
    """
    if accelerator.num_processes == 1:
        return obj

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized but accelerator.num_processes > 1. "
            "Make sure you launched via `accelerate launch ...`."
        )

    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def main(cfg: Dict[str, Any]) -> None:
    # ----------------------------
    # Accelerator (DDP multi-GPU)
    # ----------------------------
    mp = cfg.get("accelerate", {}).get("mixed_precision", None)
    accelerator = Accelerator(mixed_precision=mp) if mp is not None else Accelerator()

    # IMPORTANT: same seed on all ranks so sampling stays in sync
    set_seed(int(cfg["seed"]))

    out_dir = cfg["logging"]["out_dir"]
    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    logger = JSONLLogger(out_dir=out_dir, name=f"train_{now_ts()}") if accelerator.is_main_process else None

    tasks = iter_tasks(
        test_config_base_dir=cfg["paths"]["test_config_base_dir"],
        meta_path=cfg["paths"]["test_all_meta_path"],
    )
    if len(tasks) == 0:
        raise RuntimeError("No tasks loaded. Check test_config_base_dir/meta_path.")

    # ----------------------------
    # Env + PRM only on rank0
    # ----------------------------
    e = cfg["env"]
    env = None
    prm = None

    if accelerator.is_main_process:
        env = OSWorldEnv(
            EnvConfig(
                provider_name=e["provider_name"],
                os_type=e["os_type"],
                path_to_vm=e.get("path_to_vm", None),
                headless=bool(e["headless"]),
                action_space=e["action_space"],
                observation_type=e["observation_type"],
                screen_width=int(e["screen_width"]),
                screen_height=int(e["screen_height"]),
                max_steps=int(e["max_steps"]),
                sleep_after_execution_s=float(e.get("post_action_sleep_s", 0.5)),
                cdp_wait_timeout_s=int(e.get("cdp_wait_timeout_s", 45)),
                cdp_wait_interval_s=float(e.get("cdp_wait_interval_s", 1.0)),
            )
        )

        p = cfg["prm"]
        prm = PRMClient(
            PRMConfig(
                base_url=p["base_url"],
                model=p["model"],
                timeout_s=int(p["timeout_s"]),
                max_retries=int(p["max_retries"]),
                window_steps=int(p["window_steps"]),
                reward_scale=float(p["reward_scale"]),
                reward_clip=(float(p["reward_clip"][0]), float(p["reward_clip"][1])),
                debug=bool(p.get("debug", False)),
                debug_every=int(p.get("debug_every", 1)),
                debug_dir=str(p.get("debug_dir", "./runs/prm_debug")),
                return_rationale=bool(p.get("return_rationale", False)),
                rationale_max_chars=int(p.get("rationale_max_chars", 300)),
                send_screenshot=bool(p.get("send_screenshot", True)),
                send_screenshot_every=int(p.get("send_screenshot_every", 1)),
                save_debug_screenshot=bool(p.get("save_debug_screenshot", True)),
            )
        )

    # ----------------------------
    # Policy
    # ----------------------------
    pol_cfg = cfg["policy"]
    policy = HFPolicy(
        PolicyConfig(
            model_name_or_path=pol_cfg["model_name_or_path"],
            device=str(accelerator.device),  # per-rank device
            max_new_tokens=int(pol_cfg["max_new_tokens"]),
            temperature=float(pol_cfg["temperature"]),
            top_p=float(pol_cfg["top_p"]),
        ),
        lr=float(cfg["train"]["lr"]),
        history_n=int(pol_cfg.get("history_n", 4)),
    )
    policy.set_accelerator(accelerator)

    # Wrap net/opt with accelerate (DDP)
    policy.net, policy.opt = accelerator.prepare(policy.net, policy.opt)

    # ----------------------------
    # PPO config
    # ----------------------------
    tr = cfg["train"]
    ppo_cfg = PPOConfig(
        gamma=float(tr["gamma"]),
        gae_lambda=float(tr["gae_lambda"]),
        clip_ratio=float(tr["clip_ratio"]),
        vf_coef=float(tr["vf_coef"]),
        ent_coef=float(tr["ent_coef"]),
        train_epochs=int(tr["train_epochs"]),
        minibatch_size=int(tr["minibatch_size"]),
        value_clip=bool(tr.get("value_clip", True)),
    )

    device = accelerator.device
    total_episodes = int(tr["total_episodes"])
    save_every = int(cfg["logging"]["save_every"])
    log_every = int(cfg["logging"]["log_every"])
    post_action_sleep_s = float(e.get("post_action_sleep_s", 0.5))

    # rank0 controls task selection, then broadcast
    task_idx = 0

    for ep in range(1, total_episodes + 1):
        # Reset policy history on all ranks to keep prompts aligned
        policy.reset_episode()

        # Pick task on rank0 and broadcast to all
        task = tasks[task_idx % len(tasks)] if accelerator.is_main_process else None
        task = _broadcast_obj(accelerator, task)
        task_idx += 1

        task_text = task.get("instruction", task.get("goal", ""))

        # Reset env on rank0, broadcast initial obs
        obs = None
        if accelerator.is_main_process:
            assert env is not None
            obs = env.reset(task)
        obs = _broadcast_obj(accelerator, obs)

        hist = StepHistory()

        rewards: List[float] = []
        values: List[float] = []
        logps: List[float] = []
        dones: List[bool] = []

        prompts: List[Any] = []
        actions: List[str] = []
        screenshots: List[Optional[bytes]] = []
        prm_lat: List[float] = []

        ep_return = 0.0
        done = False

        # only main process writes screenshots
        ep_ss_dir = os.path.join(out_dir, "screenshots", f"ep{ep:04d}")
        if accelerator.is_main_process:
            os.makedirs(ep_ss_dir, exist_ok=True)

        for t in range(int(e["max_steps"])):
            # Make sampling deterministic across ranks for this step
            step_seed = int(cfg["seed"]) + ep * 100000 + t
            torch.manual_seed(step_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(step_seed)

            htxt = history_to_text(hist, k=int(cfg["prm"]["window_steps"]))

            ss_before = _get_screenshot_bytes(obs)
            if accelerator.is_main_process:
                _save_screenshot_png(ss_before, os.path.join(ep_ss_dir, f"t{t:03d}_before.png"))

            # IMPORTANT: all ranks call policy.act() to keep DDP in sync
            action, logp, value, _ss_bytes = policy.act(obs=obs, task_text=task_text, history_text=htxt)

            # prompts for PPO replay (identical across ranks)
            prompt_obj, images_list = policy._build_messages(obs=obs, task_text=task_text, history_text=htxt)
            images_lists: List[List[Image.Image]] = []

            images_lists.append(images_list)
            prompts.append(prompt_obj)
            actions.append(action)
            screenshots.append(ss_before)

            # Step env only on rank0, broadcast next_obs/done/info/env_r
            step_out = None
            if accelerator.is_main_process:
                assert env is not None
                next_obs, env_r, done_flag, info = env.step(action)
                if post_action_sleep_s > 0:
                    time.sleep(post_action_sleep_s)
                step_out = (next_obs, float(env_r), bool(done_flag), info)

            step_out = _broadcast_obj(accelerator, step_out)

            next_obs, env_r, done_flag, info = step_out
            done = bool(done_flag)

            ss_after = _get_screenshot_bytes(next_obs)
            if accelerator.is_main_process:
                _save_screenshot_png(ss_after, os.path.join(ep_ss_dir, f"t{t:03d}_after.png"))

            result_str = f"env_reward={env_r}, done={done}, info={str(info)[:400]}"
            result_str = _broadcast_obj(accelerator, result_str)

            # Update history on all ranks (keeps next-step prompting aligned)
            hist.append(
                obs=_obs_for_history(obs),
                act=action,
                res=result_str,
                screenshot=ss_after,
            )

            # PRM only on rank0, broadcast reward
            prm_r = None
            lat = 0.0
            if accelerator.is_main_process:
                assert prm is not None
                t0 = time.time()
                prm_r = prm.score_last_step(
                    task_text=task_text,
                    hist=hist,
                    task_id=task.get("_id", task.get("id", None)),
                )
                lat = time.time() - t0

            prm_r = _broadcast_obj(accelerator, float(prm_r) if prm_r is not None else 0.0)
            lat = _broadcast_obj(accelerator, float(lat))

            prm_lat.append(lat)
            rewards.append(float(prm_r))
            values.append(float(value))
            logps.append(float(logp))
            dones.append(bool(done))

            ep_return += float(prm_r)
            obs = next_obs

            if done:
                break

        # Compute advantages (identical on all ranks)
        adv, ret = compute_gae(
            rewards=np.array(rewards, dtype=np.float32),
            values=np.array(values, dtype=np.float32),
            dones=np.array(dones, dtype=np.bool_),
            gamma=ppo_cfg.gamma,
            lam=ppo_cfg.gae_lambda,
        )

        batch = {
            "old_logprob": torch.tensor(logps, dtype=torch.float32, device=device),
            "old_value": torch.tensor(values, dtype=torch.float32, device=device),
            "advantage": torch.tensor(adv, dtype=torch.float32, device=device),
            "returns": torch.tensor(ret, dtype=torch.float32, device=device),
            "prompts": prompts,
            "actions": actions,
            "screenshots": screenshots,
            "images_lists": images_lists,
        }

        upd = ppo_update(
            policy=policy,
            ppo_cfg=ppo_cfg,
            batch=batch,
            grad_clip_norm=float(tr["grad_clip_norm"]),
        )

        # Log/save only on main process
        if accelerator.is_main_process and (ep % log_every == 0):
            assert logger is not None
            logger.log({
                "episode": ep,
                "task_id": task.get("_id", task.get("id", "")),
                "domain": task.get("_domain", ""),
                "steps": len(rewards),
                "ep_return": float(ep_return),
                "reward/pos_rate": float(np.mean([1.0 if r > 0 else 0.0 for r in rewards])) if rewards else 0.0,
                "prm_window_steps": int(cfg["prm"]["window_steps"]),
                "prm/avg_latency_s": float(np.mean(prm_lat)) if prm_lat else 0.0,
                "prm/max_latency_s": float(np.max(prm_lat)) if prm_lat else 0.0,
                **upd,
            })

        if accelerator.is_main_process and (ep % save_every == 0):
            ckpt_dir = os.path.join(out_dir, f"ckpt_ep{ep}")
            policy.save(ckpt_dir)

        accelerator.wait_for_everyone()

    # Clean NCCL shutdown (removes the ProcessGroupNCCL warning)
    accelerator.wait_for_everyone()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if accelerator.is_main_process:
        print(f"Done. Logs in: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    main(cfg)