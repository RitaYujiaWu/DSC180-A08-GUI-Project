import json
import os
from typing import Any, Dict, List, Tuple


def load_meta(meta_path: str) -> Dict[str, List[str]]:
    """
    OSWorld meta example (from issue): {"examples_windows/excel": ["uuid", ...]}
    Supports .json or .jsonl containing a single JSON object.
    """
    if meta_path.endswith(".json"):
        with open(meta_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("meta json must be a dict: domain -> list[uuid]")
        return {k: list(v) for k, v in obj.items()}

    if meta_path.endswith(".jsonl"):
        with open(meta_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) != 1:
            raise ValueError("meta jsonl expected to contain exactly one JSON object")
        obj = json.loads(lines[0])
        if not isinstance(obj, dict):
            raise ValueError("meta jsonl must be a dict: domain -> list[uuid]")
        return {k: list(v) for k, v in obj.items()}

    raise ValueError(f"Unsupported meta file: {meta_path}")


def iter_tasks(test_config_base_dir: str, meta_path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of task_config dicts.
    Each task file is usually located at:
      {test_config_base_dir}/{domain}/{uuid}.json
    where domain looks like "examples_windows/excel".
    """
    meta = load_meta(meta_path)
    tasks: List[Dict[str, Any]] = []

    for domain, ids in meta.items():
        for ex_id in ids:
            task_path = os.path.join(test_config_base_dir, domain, f"{ex_id}.json")
            if not os.path.exists(task_path):
                raise FileNotFoundError(f"Task json not found: {task_path}")
            with open(task_path, "r", encoding="utf-8") as f:
                task = json.load(f)
            # Ensure we have consistent fields
            task["_domain"] = domain
            task["_id"] = task.get("id", ex_id)
            tasks.append(task)

    return tasks