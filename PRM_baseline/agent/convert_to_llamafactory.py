#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import ast
import json as std_json  # avoid shadowing


STEP_RE = re.compile(r"^step_(\d+)_.*\.png$")


def load_labels(all_result_path: Path) -> Dict[str, Dict[str, float]]:
    with all_result_path.open("r", encoding="utf-8") as f:
        raw = f.read()
    # Try strict JSON first; if it fails (e.g., single-quoted Python dict), fall back to ast.literal_eval
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = ast.literal_eval(raw)
    # data structure: { app_name: { run_id: score_float, ... }, ... }
    return data


def collect_run_images(run_dir: Path, max_images: int | None) -> List[str]:
    images_with_idx: List[Tuple[int, Path]] = []
    for child in run_dir.iterdir():
        if not child.is_file():
            continue
        m = STEP_RE.match(child.name)
        if m:
            idx = int(m.group(1))
            images_with_idx.append((idx, child))
    images_with_idx.sort(key=lambda x: x[0])
    if max_images is not None:
        images_with_idx = images_with_idx[:max_images]
    return [str(p.resolve()) for _, p in images_with_idx]


def _extract_goal_from_runtime_log(runtime_log: Path) -> Optional[str]:
    try:
        with runtime_log.open("r", encoding="utf-8", errors="ignore") as f:
            text = f.read(200_000)
    except Exception:
        return None
    patterns = [
        r"Task\s*Goal\s*[:\-]\s*(.+)",
        r"Goal\s*[:\-]\s*(.+)",
        r"Instruction\s*[:\-]\s*(.+)",
        r"Task\s*[:\-]\s*(.+)",
        r"Objective\s*[:\-]\s*(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _extract_goal_from_traj(traj_path: Path) -> Optional[str]:
    try:
        with traj_path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(200):  # scan first 200 lines
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = std_json.loads(line)
                except Exception:
                    continue
                # Common candidate keys
                for key in ["goal", "instruction", "task", "objective", "prompt", "question", "user_instruction"]:
                    val = obj.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
                # Nested 'info' dict
                info = obj.get("info")
                if isinstance(info, dict):
                    for key in ["goal", "instruction", "task", "objective", "prompt", "question"]:
                        val = info.get(key)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
    except Exception:
        return None
    return None


def _extract_goal_from_example(example_path: Path) -> Optional[str]:
    try:
        with example_path.open("r", encoding="utf-8") as f:
            obj = std_json.load(f)
    except Exception:
        return None
    for key in ["goal", "instruction", "task", "objective", "prompt", "question", "task_goal"]:
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # Sometimes nested under 'meta' or 'info'
    for container_key in ["meta", "info", "task_spec"]:
        sub = obj.get(container_key)
        if isinstance(sub, dict):
            for key in ["goal", "instruction", "task", "objective", "prompt", "question", "task_goal", "description"]:
                val = sub.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return None


def extract_task_goal(run_dir: Path, app_name: str, run_id: str, examples_dir: Optional[Path]) -> Optional[str]:
    if examples_dir is not None:
        example_path = examples_dir / app_name / f"{run_id}.json"
        if example_path.is_file():
            goal = _extract_goal_from_example(example_path)
            if goal:
                return goal
    runtime_log = run_dir / "runtime.log"
    if runtime_log.is_file():
        goal = _extract_goal_from_runtime_log(runtime_log)
        if goal:
            return goal
    traj_path = run_dir / "traj.jsonl"
    if traj_path.is_file():
        goal = _extract_goal_from_traj(traj_path)
        if goal:
            return goal
    # Fallback: None
    return None


def _build_window_prompt(n: int, task_goal: Optional[str]) -> str:
    goal_line = f"Task goal: {task_goal}\n" if task_goal else ""
    body = (
        f"You will see {n} screenshots in chronological order (s1..s{n}) representing a sliding window.\n"
        f"- Consider ONLY these {n} steps in this window.\n"
        f"- Decide whether ONLY the last step (s{n}) contributes to the goal, using s1..s{n-1} as context.\n"
        "Output JSON ONLY (no extra text):\n"
        f"{{\"s{n}_contributes\": true|false, \"reason\": \"...\"}}\n"
        f"- The reason should briefly justify your decision for s{n}."
    )
    return goal_line + body


def build_window_sample(
    images: List[str],
    app_name: str,
    task_goal: Optional[str],
    reward: Optional[float],
    last_step_reason: Optional[str] = None,
) -> Dict:
    # LLaMA-Factory multimodal: include "images": [paths] and repeat "<image>" tokens in user content
    image_tokens = "\n".join("<image>" for _ in images) if images else ""
    prompt = _build_window_prompt(len(images), task_goal)
    user_content = f"{image_tokens}\n{prompt}" if image_tokens else prompt
    label = None
    if reward is not None:
        label = "SUCCESS" if reward >= 0.5 else "FAIL"
    # If we have gold label/reason, emit as assistant JSON for SFT
    assistant_content = ""
    if reward is not None and last_step_reason:
        contributes = "true" if reward >= 0.5 else "false"
        assistant_content = std_json.dumps(
            {f"s{len(images)}_contributes": (reward >= 0.5), "reason": last_step_reason},
            ensure_ascii=False,
        )
    return {
        "images": images,
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "meta": {"app": app_name, "label": label, "task_goal": task_goal, "reward": reward},
    }


def _is_image_path(x: Any) -> bool:
    if not isinstance(x, str):
        return False
    lx = x.lower()
    return lx.endswith(".png") or lx.endswith(".jpg") or lx.endswith(".jpeg") or lx.endswith(".webp")


def _extract_images_from_obj(obj: dict) -> list[str]:
    """
    Try multiple common layouts to extract an ordered list of image paths.
    """
    # direct string lists
    for key in ["images", "image_paths", "screenshots", "frames"]:
        v = obj.get(key)
        if isinstance(v, list) and all(_is_image_path(it) for it in v):
            return [str(p) for p in v]

    # lists of dicts with image-ish keys
    candidate_list_keys = ["steps", "trajectory", "actions", "events", "frames", "screens"]
    image_field_keys = [
        "image",
        "img",
        "screenshot",
        "screenshot_file",
        "screenshot_path",
        "path",
        "filepath",
        "file",
    ]
    for list_key in candidate_list_keys:
        lst = obj.get(list_key)
        if isinstance(lst, list) and len(lst) > 0 and isinstance(lst[0], dict):
            out: list[str] = []
            for item in lst:
                path = None
                for fkey in image_field_keys:
                    val = item.get(fkey)
                    if isinstance(val, str) and _is_image_path(val):
                        path = val
                        break
                if path is not None:
                    out.append(str(path))
            if out:
                return out

    # generic: scan all values for first list-of-strings with image suffix
    for k, v in obj.items():
        if isinstance(v, list) and v and all(_is_image_path(it) for it in v):
            return [str(p) for p in v]
        if isinstance(v, list) and v and isinstance(v[0], dict):
            out: list[str] = []
            for item in v:
                if not isinstance(item, dict):
                    continue
                for fkey in image_field_keys:
                    val = item.get(fkey)
                    if isinstance(val, str) and _is_image_path(val):
                        out.append(str(val))
                        break
            if out:
                return out

    return []


def _extract_images_and_rewards_from_annot(obj: dict) -> tuple[list[str], list[Optional[float]], list[Optional[str]]]:
    """
    For annotated trajectories: build absolute paths using trajectory_dir + step['screenshot_file'].
    Also collect per-step rewards if present.
    """
    images: list[str] = []
    rewards: list[Optional[float]] = []
    reasons: list[Optional[str]] = []
    traj_dir = obj.get("trajectory_dir")
    steps = obj.get("steps")
    if isinstance(traj_dir, str) and isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            fname = step.get("screenshot_file")
            if isinstance(fname, str) and _is_image_path(fname):
                abs_path = os.path.join(traj_dir, fname)
                images.append(abs_path)
                r = step.get("reward")
                rewards.append(float(r) if isinstance(r, (int, float)) else None)
                reasons.append(step.get("reason") if isinstance(step.get("reason"), str) else None)
    return images, rewards, reasons


def convert(
    base_dir: Path,
    all_result_path: Path,
    output_path: Path,
    max_images: int | None,
    examples_dir: Optional[Path],
    window_size: int,
) -> int:
    labels = load_labels(all_result_path)
    total = 0
    written = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        # Iterate apps
        for app_name, runs in labels.items():
            app_dir = base_dir / app_name
            if not app_dir.is_dir():
                continue
            # Iterate runs by label map (ensures we have a label)
            for run_id, score in runs.items():
                total += 1
                run_dir = app_dir / run_id
                if not run_dir.is_dir():
                    continue
                images = collect_run_images(run_dir, max_images)
                if not images:
                    # Skip runs with no screenshots
                    continue
                task_goal = extract_task_goal(run_dir, app_name, run_id, examples_dir)
                # sliding windows
                for start in range(0, len(images), 1):
                    end = start + window_size
                    if end > len(images):
                        break
                    window_imgs = images[start:end]
                    reward = float(score) if score is not None else None
                    sample = build_window_sample(window_imgs, app_name, task_goal, reward)
                    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    written += 1
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Convert GUI trajectories into LLaMA-Factory multimodal JSONL with sliding windows and goal-aware prompts."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("YOUR_ROOT/o3_15steps"),
        help="Root directory containing app subdirectories (default: YOUR_ROOT/o3_15steps)",
    )
    parser.add_argument(
        "--annotated-jsonl",
        type=Path,
        default=None,
        help="Optional: path to annotated trajectories JSONL (containing images and reward fields). If provided, this will be used instead of --base-dir/--all-result.",
    )
    parser.add_argument(
        "--all-result",
        type=Path,
        default=Path("YOUR_ROOT/o3_15steps/all_result.json"),
        help="Path to all_result.json containing success/fail per run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("YOUR_ROOT/annot_windows.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=15,
        help="Max screenshots per run (chronological). Use -1 for no limit.",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=None,
        help="Path to OSWorld evaluation examples root (containing per-app subfolders with <run_id>.json).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Sliding window size.",
    )
    args = parser.parse_args()

    max_images = None if args.max_images == -1 else args.max_images
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.annotated_jsonl is not None:
        # Read annotated jsonl and write windows
        written = 0
        with args.output.open("w", encoding="utf-8") as out_f, args.annotated_jsonl.open("r", encoding="utf-8") as in_f:
            skipped_no_images = 0
            short_last_used = 0
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = std_json.loads(line)
                except Exception:
                    continue
                # Prefer annotated structure (absolute paths + per-step rewards)
                images, per_step_rewards, per_step_reasons = _extract_images_and_rewards_from_annot(obj)
                if not images:
                    images = _extract_images_from_obj(obj)
                    per_step_rewards = [None] * len(images)
                    per_step_reasons = [None] * len(images)
                if not images:
                    skipped_no_images += 1
                    continue
                images = [str(p) for p in images]  # keep full absolute paths as-is
                task_goal = (
                    obj.get("task_goal")
                    or obj.get("goal")
                    or obj.get("instruction")
                    or obj.get("task_instruction")
                    or None
                )
                app_name = obj.get("app") or obj.get("application") or "Qwen3-VL-4B-Instruct"
                if max_images is not None:
                    images = images[:max_images]
                    per_step_rewards = per_step_rewards[: len(images)]
                    per_step_reasons = per_step_reasons[: len(images)]
                # Emit windows; if shorter than window_size, emit a single window with all images
                if len(images) < args.window_size:
                    # Use as much context as available (ending at last image)
                    start_idx = max(0, len(images) - args.window_size)
                    window_imgs = images[start_idx:]
                    window_rs = per_step_rewards[start_idx:]
                    window_reasons = per_step_reasons[start_idx:]
                    # last-step reward/reason decide the label and assistant JSON
                    last_r = window_rs[-1] if window_rs else None
                    last_reason = window_reasons[-1] if window_reasons else None
                    sample = build_window_sample(
                        window_imgs,
                        app_name,
                        task_goal,
                        float(last_r) if isinstance(last_r, (int, float)) else None,
                        last_reason,
                    )
                    if isinstance(last_r, (int, float)):
                        sample["meta"]["label"] = "SUCCESS" if last_r >= 0.5 else "FAIL"
                    sample["meta"]["step_rewards"] = window_rs
                    sample["meta"]["step_reasons"] = window_reasons
                    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    written += 1
                    short_last_used += 1
                else:
                    for start in range(0, len(images), 1):
                        end = start + args.window_size
                        if end > len(images):
                            break
                        window_imgs = images[start:end]
                        window_rs = per_step_rewards[start:end]
                        window_reasons = per_step_reasons[start:end]
                        # last-step reward/reason decide the target for sN
                        last_r = window_rs[-1] if window_rs else None
                        last_reason = window_reasons[-1] if window_reasons else None
                        sample = build_window_sample(
                            window_imgs,
                            app_name,
                            task_goal,
                            float(last_r) if isinstance(last_r, (int, float)) else None,
                            last_reason,
                        )
                        if isinstance(last_r, (int, float)):
                            sample["meta"]["label"] = "SUCCESS" if last_r >= 0.5 else "FAIL"
                        sample["meta"]["step_rewards"] = window_rs
                        sample["meta"]["step_reasons"] = window_reasons
                        out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        written += 1
        print(f"Wrote {written} samples to {args.output} from {args.annotated_jsonl}")
        if skipped_no_images:
            print(f"Skipped {skipped_no_images} records with no images.")
        if short_last_used:
            print(f"Used last image for {short_last_used} records shorter than window.")
    else:
        written = convert(
            args.base_dir, args.all_result, args.output, max_images, args.examples_dir, args.window_size
        )
        print(f"Wrote {written} samples to {args.output}")


if __name__ == "__main__":
    main()


