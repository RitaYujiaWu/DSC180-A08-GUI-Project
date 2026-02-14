from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union
import os
import json
import base64
import io
import re
import hashlib

def list_jsonl_files(root: str) -> List[str]:
    if not root or not os.path.exists(root):
        return []
    out: List[str] = []
    for name in os.listdir(root):
        if name.endswith(".jsonl"):
            out.append(os.path.join(root, name))
    return sorted(out)

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def safe_b64_to_pil(b64: Optional[str]):
    if not b64:
        return None
    try:
        from PIL import Image
    except Exception as e:
        raise ImportError("PIL is required for image decoding. Install pillow.") from e

    try:
        if "," in b64:
            b64 = b64.split(",", 1)[-1]
        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None

def safe_get_first_screenshot(traj_obj: Dict[str, Any]):
    """
    Best-effort: find first screenshot in known WebVoyager/MMInA formats.
    Returns PIL.Image or None.
    """
    rounds = traj_obj.get("rounds") or []
    if not rounds:
        return None
    # try common keys inside messages
    for r in rounds[:2]:
        msgs = r.get("messages") or []
        for m in msgs:
            if isinstance(m, dict):
                # common: {"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}, ...]}
                content = m.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") in ("image_url", "image"):
                            url = (c.get("image_url") or {}).get("url") if c.get("type") == "image_url" else c.get("data")
                            if isinstance(url, str) and "base64" in url:
                                return safe_b64_to_pil(url)
                # sometimes screenshot stored directly
                for k in ("screenshot", "image", "img", "base64"):
                    if isinstance(m.get(k), str):
                        return safe_b64_to_pil(m.get(k))
    return None

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract first {...} JSON object from text, robustly.
    """
    if not text:
        return None
    text = text.strip()
    # If direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    snippet = m.group(0)
    try:
        return json.loads(snippet)
    except Exception:
        return None

def clip_words(text: str, max_words: int) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()

def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def coerce_action_history_text(action_history: Union[List[Any], str, None]) -> str:
    """
    Your agent loop may pass action history as:
      - list[dict] action JSONs
      - list[str]
      - raw string
    We normalize to a compact string.
    """
    if action_history is None:
        return "(none)"
    if isinstance(action_history, str):
        return action_history.strip() if action_history.strip() else "(none)"
    if isinstance(action_history, list):
        parts: List[str] = []
        for a in action_history[-12:]:
            if isinstance(a, str):
                parts.append(a.strip())
            elif isinstance(a, dict):
                name = a.get("name") or a.get("action") or "action"
                args = a.get("arguments") or a.get("args") or {}
                if isinstance(args, dict):
                    # short preview
                    kv = ", ".join([f"{k}={str(v)[:20]}" for k, v in list(args.items())[:3]])
                    parts.append(f"{name}({kv})" if kv else f"{name}()")
                else:
                    parts.append(f"{name}({str(args)[:40]})")
            else:
                parts.append(str(a)[:80])
        return " -> ".join([p for p in parts if p]) if parts else "(none)"
    return str(action_history)[:200]
