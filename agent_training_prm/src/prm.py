# src/prm.py
import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class PRMConfig:
    base_url: str
    model: str
    timeout_s: int
    max_retries: int
    window_steps: int
    reward_scale: float
    reward_clip: Tuple[float, float]

    # debug / inspection
    debug: bool = False
    debug_every: int = 1
    debug_dir: str = "./runs/prm_debug"

    # ask PRM for short explanation
    return_rationale: bool = False
    rationale_max_chars: int = 300

    # send screenshot to PRM as an image (OpenAI multimodal schema)
    send_screenshot: bool = True
    send_screenshot_every: int = 1  # attach image every N calls
    save_debug_screenshot: bool = True  # also write PNG files under debug_dir


class StepHistory:
    def __init__(self):
        self.obs: List[str] = []
        self.act: List[str] = []
        self.res: List[str] = []
        self.screenshots: List[Optional[bytes]] = []

    def append(self, obs: str, act: str, res: str, screenshot: Optional[bytes] = None) -> None:
        self.obs.append(obs)
        self.act.append(act)
        self.res.append(res)
        self.screenshots.append(screenshot)

    def window(self, k: int) -> Dict[str, List[Any]]:
        k = max(1, int(k))
        return {
            "obs": self.obs[-k:],
            "act": self.act[-k:],
            "res": self.res[-k:],
            "screenshots": self.screenshots[-k:],
        }


class PRMClient:
    def __init__(self, cfg: PRMConfig, api_key: Optional[str] = None):
        self.cfg = cfg
        self.session = requests.Session()
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self._call_idx = 0
        if self.cfg.debug:
            os.makedirs(self.cfg.debug_dir, exist_ok=True)

    def score_last_step(self, task_text: str, hist: StepHistory, task_id: Optional[str] = None) -> float:
        self._call_idx += 1

        w = hist.window(self.cfg.window_steps)
        last_obs = w["obs"][-1] if w.get("obs") else None
        last_act = w["act"][-1] if w.get("act") else None
        last_res = w["res"][-1] if w.get("res") else None
        last_ss = w.get("screenshots", [None])[-1]

        prompt_text = self._build_prompt(task_text, w)
        url = f"{self.cfg.base_url}/chat/completions"

        if self.cfg.return_rationale:
            sys = (
                "You are a process reward model. "
                "Return JSON only in ONE line with keys: reward, label, rationale. "
                f"rationale must be <= {int(self.cfg.rationale_max_chars)} characters."
            )
        else:
            sys = "You are a process reward model. Return JSON only in ONE line."

        # --- build messages (text + optional image) ---
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]

        attach_img = (
            self.cfg.send_screenshot
            and (self._call_idx % max(1, int(self.cfg.send_screenshot_every)) == 0)
            and isinstance(last_ss, (bytes, bytearray))
            and len(last_ss) > 0
        )
        if attach_img:
            b64 = base64.b64encode(bytes(last_ss)).decode("utf-8")
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
        }

        t0 = time.time()
        text = self._post_with_retries(url, payload)
        latency_s = time.time() - t0

        r01, parsed_obj = self._parse_reward(text)

        self._maybe_debug_dump(
            prompt=prompt_text,
            raw=text,
            parsed_reward=r01,
            parsed_obj=parsed_obj,
            latency_s=latency_s,
            task_id=task_id,
            last_obs=last_obs,
            last_action=last_act,
            last_result=last_res,
            last_screenshot=bytes(last_ss) if isinstance(last_ss, (bytes, bytearray)) else None,
        )

        r = float(r01) * float(self.cfg.reward_scale)
        lo, hi = self.cfg.reward_clip
        r = max(lo, min(hi, r))
        return float(r)

    def _post_with_retries(self, url: str, payload: Dict[str, Any]) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self.session.post(url, headers=self.headers, json=payload, timeout=self.cfg.timeout_s)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"PRM request failed after retries: {last_err}")

    def _build_prompt(self, task_text: str, w: Dict[str, List[Any]]) -> str:
        lines: List[str] = []
        if self.cfg.return_rationale:
            lines.append('Return JSON only: {"reward": 0 or 1, "label": "good" or "bad", "rationale": "..." }')
            lines.append(f'Keep "rationale" <= {int(self.cfg.rationale_max_chars)} chars, max 2 sentences.')
        else:
            lines.append('Return JSON only: {"reward": 0 or 1, "label": "good" or "bad"}')

        lines.append("If the last step makes progress toward completing the task, reward=1, else reward=0.")
        lines.append("")
        lines.append(f"Task:\n{task_text}\n")
        lines.append("Trajectory window (oldest -> newest):")

        n = len(w["obs"])
        for i in range(n):
            lines.append(f"{i+1}) Obs: {w['obs'][i]}")
            lines.append(f"   Act: {w['act'][i]}")
            lines.append(f"   Result: {w['res'][i]}")
        lines.append("\nScore ONLY the LAST step.")
        return "\n".join(lines)

    def _parse_reward(self, text: str) -> Tuple[int, Dict[str, Any]]:
        t = text.strip()
        start = t.find("{")
        end = t.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"PRM did not return JSON: {t[:200]}")
        obj = json.loads(t[start:end + 1])

        if "reward" in obj:
            try:
                r = int(obj["reward"])
            except Exception:
                r = 0
            return (1 if r != 0 else 0), obj

        lab = str(obj.get("label", "")).lower()
        return (1 if lab == "good" else 0), obj

    def _maybe_debug_dump(
        self,
        prompt: str,
        raw: str,
        parsed_reward: int,
        parsed_obj: Dict[str, Any],
        latency_s: float,
        task_id: Optional[str],
        last_obs: Optional[str],
        last_action: Optional[str],
        last_result: Optional[str],
        last_screenshot: Optional[bytes],
    ) -> None:
        if not self.cfg.debug:
            return
        every = max(1, int(self.cfg.debug_every))
        if (self._call_idx % every) != 0:
            return

        ss_path = None
        if self.cfg.save_debug_screenshot and isinstance(last_screenshot, (bytes, bytearray)) and len(last_screenshot) > 0:
            ss_path = os.path.join(self.cfg.debug_dir, f"prm_call_{self._call_idx:06d}.png")
            try:
                with open(ss_path, "wb") as f:
                    f.write(last_screenshot)
            except Exception:
                ss_path = None

        rec = {
            "idx": int(self._call_idx),
            "time": float(time.time()),
            "task_id": task_id,
            "window_steps": int(self.cfg.window_steps),
            "latency_s": float(latency_s),
            "last_obs": last_obs,
            "last_action": last_action,
            "last_result": last_result,
            "last_screenshot_path": ss_path,
            "prompt": prompt,
            "raw_response": raw,
            "parsed_reward": int(parsed_reward),
            "parsed_label": parsed_obj.get("label", None),
            "parsed_rationale": parsed_obj.get("rationale", None),
        }

        path = os.path.join(self.cfg.debug_dir, f"prm_call_{self._call_idx:06d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)