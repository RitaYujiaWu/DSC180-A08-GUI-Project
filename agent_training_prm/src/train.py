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


def _parse_every(raw: Any, *, epoch_len: int, field_name: str) -> int:
    if isinstance(raw, str):
        if raw.strip().lower() == "epoch":
            every = int(epoch_len)
        else:
            raise ValueError(f"{field_name} must be an int or 'epoch'. Got: {raw!r}")
    else:
        every = int(raw)
    if every <= 0:
        raise ValueError(f"{field_name} must be > 0. Got: {every}")
    return every


def _extract_success(info: Dict[str, Any]) -> Optional[bool]:
    if not isinstance(info, dict):
        return None
    for k in ["success", "task_success", "is_success", "completed", "goal_achieved"]:
        if k in info and isinstance(info[k], (bool, int, float)):
            return bool(info[k])
    status = info.get("status", None)
    if isinstance(status, str):
        s = status.strip().lower()
        if s in ["success", "succeeded", "passed"]:
            return True
        if s in ["failure", "failed", "error"]:
            return False
    return None


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
    if accelerator.is_main_process and logger is not None:
        logger.log({"run/config_path": cfg.get("_config_path", None)})

    # ----------------------------
    # Optional: Weights & Biases
    # ----------------------------
    wandb_run = None
    wandb_cfg = (cfg.get("logging", {}) or {}).get("wandb", {}) or {}
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    if wandb_enabled and accelerator.is_main_process:
        try:
            import wandb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "W&B logging is enabled but `wandb` is not installed. Install it with `pip install wandb`."
            ) from e

        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "osworld_prm_ppo")),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("name", None),
            group=wandb_cfg.get("group", None),
            tags=list(wandb_cfg.get("tags", [])) if isinstance(wandb_cfg.get("tags", []), list) else None,
            dir=str(wandb_cfg.get("dir", out_dir)),
            mode=str(wandb_cfg.get("mode", "online")),
            config=cfg,
        )

        # Put code + logs under run directory
        if logger is not None and hasattr(logger, "path"):
            wandb_run.log({"logging/jsonl_path": str(logger.path)})

    tasks = iter_tasks(
        test_config_base_dir=cfg["paths"]["test_config_base_dir"],
        meta_path=cfg["paths"]["test_all_meta_path"],
    )
    if len(tasks) == 0:
        raise RuntimeError("No tasks loaded. Check test_config_base_dir/meta_path.")

    def _allreduce_mean_dict(d: Dict[str, float]) -> Dict[str, float]:
        """DDP-safe mean aggregation for scalar metrics."""
        if accelerator.num_processes == 1:
            return d
        if not dist.is_available() or not dist.is_initialized():
            return d

        out: Dict[str, float] = {}
        world = float(dist.get_world_size())
        for k, v in d.items():
            try:
                t = torch.tensor([float(v)], device=accelerator.device, dtype=torch.float32)
            except Exception:
                t = torch.tensor([0.0], device=accelerator.device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            out[k] = float((t / world).item())
        return out

    # ----------------------------
    # Env + PRM
    # ----------------------------
    e = cfg["env"]
    env = None
    prm = None

    tr = cfg["train"]
    rollout_per_rank = bool(tr.get("rollout_per_rank", False))

    # If rollout_per_rank=True, every rank runs its own env and PRM client.
    # This avoids duplicating policy.act() across GPUs on identical observations.
    if rollout_per_rank or accelerator.is_main_process:
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

        # Record which OSWorld DesktopEnv implementation we are using (rank0 only).
        if accelerator.is_main_process:
            impl_info = getattr(env, "impl_info", {}) or {}
            if logger is not None:
                logger.log({"osworld/impl": impl_info})
            print(f"[OSWorld] DesktopEnv from: {impl_info.get('desktop_env_file', '')}")
            if wandb_enabled and wandb_run is not None:
                try:
                    import wandb  # type: ignore
                    wandb.log({"osworld/desktop_env_file": impl_info.get("desktop_env_file", "")})
                    wandb.log({"osworld/desktop_env_module": impl_info.get("desktop_env_module", "")})
                except Exception:
                    pass

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
                fail_open=bool(p.get("fail_open", True)),
                fail_open_reward=float(p.get("fail_open_reward", 0.0)),
            )
        )

    # ----------------------------
    # Policy
    # ----------------------------
    pol_cfg = cfg["policy"]
    policy_cfg = PolicyConfig(
        model_name_or_path=pol_cfg["model_name_or_path"],
        device=str(accelerator.device),  # per-rank device
        max_new_tokens=int(pol_cfg["max_new_tokens"]),
        temperature=float(pol_cfg["temperature"]),
        top_p=float(pol_cfg["top_p"]),
        max_input_tokens=(int(pol_cfg["max_input_tokens"]) if pol_cfg.get("max_input_tokens", None) is not None else None),
        action_token_reserve=int(pol_cfg.get("action_token_reserve", 64)),
        truncation_side=str(pol_cfg.get("truncation_side", "left")),
        max_total_tokens=(int(pol_cfg["max_total_tokens"]) if pol_cfg.get("max_total_tokens", None) is not None else None),
        min_text_tokens=int(pol_cfg.get("min_text_tokens", 256)),
        gradient_checkpointing=bool(pol_cfg.get("gradient_checkpointing", True)),
        max_image_pixels=int(pol_cfg.get("max_image_pixels", 1600 * 1600)),
    )

    lora_cfg = pol_cfg.get("lora", {})
    use_lora = bool(lora_cfg.get("enabled", False))
    if use_lora:
        from src.policy_hf_lora import HFPolicyLoRA, LoRAConfig

        policy = HFPolicyLoRA(
            cfg=policy_cfg,
            lr=float(cfg["train"]["lr"]),
            history_n=int(pol_cfg.get("history_n", 4)),
            lora_cfg=LoRAConfig(
                r=int(lora_cfg.get("r", 16)),
                alpha=int(lora_cfg.get("alpha", 32)),
                dropout=float(lora_cfg.get("dropout", 0.05)),
                target_modules=list(
                    lora_cfg.get(
                        "target_modules",
                        [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                    )
                ),
                bias=str(lora_cfg.get("bias", "none")),
                freeze_vision_encoder=bool(lora_cfg.get("freeze_vision_encoder", True)),
                vision_param_keywords=list(
                    lora_cfg.get(
                        "vision_param_keywords",
                        ["vision", "visual", "vision_tower", "image_tower", "vit"],
                    )
                ),
            ),
        )
    else:
        policy = HFPolicy(
            policy_cfg,
            lr=float(cfg["train"]["lr"]),
            history_n=int(pol_cfg.get("history_n", 4)),
        )
        if logger is not None:
            logger.log(
                {
                    "effective/env_max_steps": int(cfg["env"]["max_steps"]),
                    "effective/policy_history_n": int(pol_cfg.get("history_n", 4)),
                }
            )
    policy.set_accelerator(accelerator)

    # Wrap net/opt with accelerate (DDP)
    policy.net, policy.opt = accelerator.prepare(policy.net, policy.opt)

    # ----------------------------
    # PPO config
    # ----------------------------
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

    save_every_raw = cfg["logging"]["save_every"]
    if isinstance(save_every_raw, str):
        if save_every_raw.strip().lower() == "epoch":
            save_every = len(tasks)
        else:
            raise ValueError(
                "logging.save_every must be an int episode interval or the string 'epoch'. "
                f"Got: {save_every_raw!r}"
            )
    else:
        save_every = int(save_every_raw)

    if save_every <= 0:
        raise ValueError(f"logging.save_every must be > 0. Got: {save_every}")
    log_every = int(cfg["logging"]["log_every"])
    save_screenshots = bool((cfg.get("logging", {}) or {}).get("save_screenshots", True))

    # Task selection:
    # - rollout_per_rank=False: rank0 controls tasks/obs; other ranks follow.
    # - rollout_per_rank=True: each rank runs different tasks for throughput.
    task_idx = accelerator.process_index if rollout_per_rank else 0

    # ----------------------------
    # Optional: validation tasks
    # ----------------------------
    eval_cfg = cfg.get("eval", {}) or {}
    eval_enabled = bool(eval_cfg.get("enabled", False))
    eval_tasks: List[Dict[str, Any]] = []
    eval_every = None
    eval_max_episodes = None
    eval_do_sample = bool(eval_cfg.get("do_sample", False))
    if eval_enabled:
        eval_meta_path = str(eval_cfg.get("meta_path", ""))
        if not eval_meta_path:
            raise ValueError("eval.enabled is true but eval.meta_path is empty")
        eval_tasks = iter_tasks(
            test_config_base_dir=cfg["paths"]["test_config_base_dir"],
            meta_path=eval_meta_path,
        )
        if len(eval_tasks) == 0:
            raise RuntimeError(f"No eval tasks loaded from: {eval_meta_path}")

        eval_every = _parse_every(
            eval_cfg.get("every", "epoch"),
            epoch_len=len(tasks),
            field_name="eval.every",
        )
        eval_max_raw = eval_cfg.get("max_episodes", None)
        if eval_max_raw is None:
            eval_max_episodes = len(eval_tasks)
        else:
            eval_max_episodes = int(eval_max_raw)
        if eval_max_episodes <= 0:
            raise ValueError(f"eval.max_episodes must be > 0 or null. Got: {eval_max_raw!r}")

    did_eval = False

    for ep in range(1, total_episodes + 1):
        # Reset policy history on all ranks to keep prompts aligned
        policy.reset_episode()

        if rollout_per_rank:
            task = tasks[task_idx % len(tasks)]
            task_idx += int(accelerator.num_processes)
        else:
            # Pick task on rank0 and broadcast to all
            task = tasks[task_idx % len(tasks)] if accelerator.is_main_process else None
            task = _broadcast_obj(accelerator, task)
            task_idx += 1

        task_text = task.get("instruction", task.get("goal", ""))

        # Reset env
        obs = None
        if rollout_per_rank:
            assert env is not None
            obs = env.reset(task)
        else:
            # Reset env on rank0, broadcast initial obs
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
        images_lists: List[List[Image.Image]] = []
        prm_lat: List[float] = []

        ep_return = 0.0
        done = False

        # only main process writes screenshots (optional)
        ep_ss_dir = None
        if save_screenshots:
            ep_ss_dir = os.path.join(out_dir, "screenshots", f"ep{ep:04d}")
            if accelerator.is_main_process:
                os.makedirs(ep_ss_dir, exist_ok=True)

        max_steps = int(e["max_steps"])
        for t in range(max_steps):
            # Make sampling deterministic across ranks for this step
            step_seed = int(cfg["seed"]) + ep * 100000 + t
            torch.manual_seed(step_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(step_seed)

            htxt = history_to_text(hist, k=int(cfg["prm"]["window_steps"]))

            ss_before = _get_screenshot_bytes(obs)
            if save_screenshots and accelerator.is_main_process and ep_ss_dir is not None:
                _save_screenshot_png(ss_before, os.path.join(ep_ss_dir, f"t{t:03d}_before.png"))

            # IMPORTANT:
            # - rollout_per_rank=False: all ranks call act() on identical obs to keep DDP in sync.
            # - rollout_per_rank=True: each rank calls act() on its own obs; DDP sync happens during backward.
            action, logp, value, _ss_bytes = policy.act(obs=obs, task_text=task_text, history_text=htxt)

            # prompts for PPO replay (identical across ranks)
            prompt_obj, images_list = policy._build_messages(obs=obs, task_text=task_text, history_text=htxt)
            images_lists.append(images_list)
            prompts.append(prompt_obj)
            actions.append(action)
            screenshots.append(ss_before)

            # Step env
            if rollout_per_rank:
                assert env is not None
                if not done:
                    next_obs, env_r, done_flag, info = env.step(action)
                    done = bool(done_flag)
                else:
                    # Pad after done: keep rollout length fixed across ranks.
                    next_obs, env_r, info = obs, 0.0, {"padded": True}
            else:
                # Step env only on rank0, broadcast next_obs/done/info/env_r
                step_out = None
                if accelerator.is_main_process:
                    assert env is not None
                    next_obs, env_r, done_flag, info = env.step(action)
                    step_out = (next_obs, float(env_r), bool(done_flag), info)

                step_out = _broadcast_obj(accelerator, step_out)

                next_obs, env_r, done_flag, info = step_out
                done = bool(done_flag)

            ss_after = _get_screenshot_bytes(next_obs)
            if save_screenshots and accelerator.is_main_process and ep_ss_dir is not None:
                _save_screenshot_png(ss_after, os.path.join(ep_ss_dir, f"t{t:03d}_after.png"))

            result_str = f"env_reward={env_r}, done={done}, info={str(info)[:400]}"
            if not rollout_per_rank:
                result_str = _broadcast_obj(accelerator, result_str)

            # Update history on all ranks (keeps next-step prompting aligned)
            hist.append(
                obs=_obs_for_history(obs),
                act=action,
                res=result_str,
                screenshot=ss_after,
            )

            if rollout_per_rank:
                # PRM on each rank. For padded steps, skip remote call.
                prm_r = 0.0
                lat = 0.0
                if not bool(info.get("padded", False)):
                    assert prm is not None
                    t0 = time.time()
                    prm_r = float(
                        prm.score_last_step(
                            task_text=task_text,
                            hist=hist,
                            task_id=task.get("_id", task.get("id", None)),
                        )
                        or 0.0
                    )
                    lat = float(time.time() - t0)
            else:
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

            # NOTE: don't break early; keep per-episode rollout length fixed for DDP.

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

        # Aggregate scalar metrics across ranks when doing per-rank rollouts.
        if rollout_per_rank:
            upd = _allreduce_mean_dict({k: float(v) for k, v in upd.items()})

        # Log/save only on main process
        if accelerator.is_main_process and (ep % log_every == 0):
            assert logger is not None
            train_metrics = {
                "episode": ep,
                "task_id": task.get("_id", task.get("id", "")),
                "domain": task.get("_domain", ""),
                "steps": len(rewards),
                "ep_return": float(ep_return),
                "reward/pos_rate": float(np.mean([1.0 if r > 0 else 0.0 for r in rewards])) if rewards else 0.0,
                "prm_window_steps": int(cfg["prm"]["window_steps"]),
                # Always log PRM step-wise signals here (NOT gated by prm.debug).
                "prm/step_rewards": [float(r) for r in rewards],
                "prm/step_latencies_s": [float(x) for x in prm_lat],
                "prm/avg_reward": float(np.mean(rewards)) if rewards else 0.0,
                "prm/last_reward": float(rewards[-1]) if rewards else 0.0,
                "prm/avg_latency_s": float(np.mean(prm_lat)) if prm_lat else 0.0,
                "prm/max_latency_s": float(np.max(prm_lat)) if prm_lat else 0.0,
                **upd,
            }
            logger.log(train_metrics)

            if wandb_enabled and wandb_run is not None:
                import wandb  # type: ignore
                wandb.log(train_metrics, step=ep)

        if accelerator.is_main_process and (ep % save_every == 0):
            ckpt_dir = os.path.join(out_dir, f"ckpt_ep{ep}")
            policy.save(ckpt_dir)

            if wandb_enabled and wandb_run is not None and bool(wandb_cfg.get("upload_checkpoints", True)):
                import wandb  # type: ignore
                art = wandb.Artifact(name=f"ckpt_ep{ep}", type="model")
                art.add_dir(ckpt_dir)
                wandb_run.log_artifact(art)

            accelerator.wait_for_everyone()

        # Optional validation pass (DDP-safe: rank0 steps env, others stay in sync)
        scheduled_eval = bool(eval_enabled and eval_every is not None and (ep % int(eval_every) == 0))
        force_final_eval = bool(eval_enabled and (ep == total_episodes) and (not did_eval))
        do_eval = bool(scheduled_eval or force_final_eval)
        do_eval = _broadcast_obj(accelerator, do_eval)
        if do_eval:
            did_eval = True
            eval_n = int(min(int(eval_max_episodes or len(eval_tasks)), len(eval_tasks)))
            eval_osworld_scores: List[float] = []
            eval_steps: List[int] = []

            # Keep prompts aligned across ranks
            for i in range(eval_n):
                policy.reset_episode()

                eval_task = eval_tasks[i] if accelerator.is_main_process else None
                eval_task = _broadcast_obj(accelerator, eval_task)
                eval_task_text = eval_task.get("instruction", eval_task.get("goal", ""))

                eval_obs = None
                if accelerator.is_main_process:
                    assert env is not None
                    eval_obs = env.reset(eval_task)
                eval_obs = _broadcast_obj(accelerator, eval_obs)

                hist = StepHistory()
                done = False
                osworld_score: Optional[float] = None

                for t in range(int(e["max_steps"])):
                    htxt = history_to_text(hist, k=int(cfg["prm"]["window_steps"]))
                    action, _logp, _value, _ss_bytes = policy.act(
                        obs=eval_obs,
                        task_text=eval_task_text,
                        history_text=htxt,
                        do_sample=bool(eval_do_sample),
                    )

                    step_out = None
                    if accelerator.is_main_process:
                        assert env is not None
                        next_obs, env_r, done_flag, info = env.step(action)
                        step_out = (next_obs, float(env_r), bool(done_flag), info)
                    step_out = _broadcast_obj(accelerator, step_out)

                    next_obs, env_r, done_flag, info = step_out
                    done = bool(done_flag)

                    result_str = f"env_reward={env_r}, done={done}, info={str(info)[:400]}"
                    result_str = _broadcast_obj(accelerator, result_str)
                    ss_after = _get_screenshot_bytes(next_obs)

                    hist.append(
                        obs=_obs_for_history(eval_obs),
                        act=action,
                        res=result_str,
                        screenshot=ss_after,
                    )

                    eval_obs = next_obs
                    if done:
                        break

                # OSWorld official scoring (DesktopEnv.evaluate)
                if accelerator.is_main_process:
                    assert env is not None
                    try:
                        osworld_score = float(env.env.evaluate())
                    except Exception:
                        osworld_score = 0.0
                osworld_score = _broadcast_obj(accelerator, osworld_score)

                if accelerator.is_main_process:
                    eval_osworld_scores.append(float(osworld_score or 0.0))
                    eval_steps.append(int(t + 1))

                accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                val_metrics = {
                    "episode": ep,
                    "val/num_episodes": int(eval_n),
                    "val/osworld_score_avg": float(np.mean(eval_osworld_scores)) if eval_osworld_scores else 0.0,
                    "val/osworld_score_std": float(np.std(eval_osworld_scores)) if eval_osworld_scores else 0.0,
                    "val/success_rate": float(np.mean([1.0 if s >= 0.5 else 0.0 for s in eval_osworld_scores])) if eval_osworld_scores else 0.0,
                    "val/avg_steps": float(np.mean(eval_steps)) if eval_steps else 0.0,
                }
                assert logger is not None
                logger.log(val_metrics)

                if wandb_enabled and wandb_run is not None:
                    import wandb  # type: ignore
                    wandb.log(val_metrics, step=ep)

                    # Per-task table for quick inspection
                    if bool(eval_cfg.get("log_table", True)):
                        table = wandb.Table(columns=["task_id", "domain", "osworld_score", "steps"]) 
                        for i in range(eval_n):
                            tsk = eval_tasks[i]
                            table.add_data(
                                str(tsk.get("_id", tsk.get("id", ""))),
                                str(tsk.get("_domain", "")),
                                float(eval_osworld_scores[i]),
                                int(eval_steps[i]),
                            )
                        wandb.log({"val/table": table}, step=ep)

        accelerator.wait_for_everyone()

    # Clean NCCL shutdown (removes the ProcessGroupNCCL warning)
    accelerator.wait_for_everyone()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if accelerator.is_main_process:
        print(f"Done. Logs in: {out_dir}")

    if wandb_enabled and accelerator.is_main_process and wandb_run is not None:
        try:
            import wandb  # type: ignore
            # Upload JSONL at the end too (handy if resuming)
            if logger is not None and bool(wandb_cfg.get("upload_logs", True)):
                art = wandb.Artifact(name="train_logs", type="logs")
                art.add_file(logger.path)
                wandb_run.log_artifact(art)
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    cfg["_config_path"] = args.config
    main(cfg)