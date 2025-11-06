import time
from io import BytesIO
from typing import Any, Dict, Optional, Union

import numpy as np
import requests
from PIL import Image

# Valid action labels exposed by the upstream Android World API
ANDROID_WORLD_ACTION_TYPES = {
    "answer",
    "click",
    "double_tap",
    "input_text",
    "keyboard_enter",
    "long_press",
    "navigate_back",
    "navigate_home",
    "open_app",
    "scroll",
    "status",
    "swipe",
    "unknown",
    "wait",
}

def _pixels_to_png(pixels: list[Any]) -> bytes:
    arr = np.array(pixels, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

class RemoteAndroidWorldEnv:
    def __init__(
        self,
        base_url: str,
        env_port: int = 5000,
        connect_max_try: int = 5,
        suite_family: str = "android_world",
        task_type: Optional[str] = None,
        task_idx: int = 0,
        wait_to_stabilize: bool = True,
        llm_evaluator=None,
        test_task_llm_eval: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.env_port = env_port
        self.connect_max_try = connect_max_try
        self.wait_to_stabilize = wait_to_stabilize
        self.llm_evaluator = llm_evaluator
        self.test_task_llm_eval = test_task_llm_eval
        self.task_type = task_type
        self.task_idx = task_idx
        self.session = requests.Session()

        self._request("/health", method="GET")

        if self.task_type is not None:
            self._request(
                "/suite/reinitialize",
                method="GET",
                params={
                    "task_family": suite_family,
                    "n_task_combinations": 2,
                    "seed": 42,
                },
            )
            self._request(
                "/task/initialize",
                data={"task_type": self.task_type, "task_idx": self.task_idx},
            )

    def _url(self, path: str) -> str:
        return f"{self.base_url}:{self.env_port}{path}"

    def _request(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        method: str = "POST",
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 360,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for _ in range(self.connect_max_try):
            try:
                response = self.session.request(
                    method=method,
                    url=self._url(path),
                    json=data,
                    params=params,
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                time.sleep(1)
        raise Exception(
            f"Request to {path} failed after {self.connect_max_try} attempts"
        ) from last_error

    def _decode_obs(self) -> Dict[str, Any]:
        payload = self._request(
            "/screenshot",
            method="GET",
            params={"wait_to_stabilize": self.wait_to_stabilize},
        )
        screenshot_bytes = _pixels_to_png(payload["pixels"])
        return {
            "screenshot": screenshot_bytes,
            "ui_elements": payload.get("ui_elements"),
        }

    def reset(self, go_home: bool = True) -> Dict[str, Any]:
        self._request("/reset", params={"go_home": go_home})
        return self._decode_obs()

    @staticmethod
    def _prepare_action(action: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        if isinstance(action, dict):
            return dict(action)
        if isinstance(action, str):
            stripped = action.strip()
            if stripped.startswith("finish("):
                return {"action_type": "status", "goal_status": "success"}
            raise ValueError(
                "String-based actions are no longer supported. Pass a dict with "
                "Android World action fields instead."
            )
        raise TypeError(
            "Action must be provided as a dict containing Android World fields."
        )

    def _normalize_action(self, action: Union[Dict[str, Any], str], screen: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._prepare_action(action)
        if "action_type" not in payload:
            raise ValueError("Action payload must include an 'action_type' field.")
        if payload["action_type"] not in ANDROID_WORLD_ACTION_TYPES:
            raise ValueError(
                f"Unsupported action_type '{payload['action_type']}'. Allowed values: {sorted(ANDROID_WORLD_ACTION_TYPES)}"
            )
        arr = Image.open(BytesIO(screen["screenshot"]))
        width, height = arr.size
        if "x_norm" in payload:
            payload["x"] = int(payload.pop("x_norm") * width)
        if "y_norm" in payload:
            payload["y"] = int(payload.pop("y_norm") * height)
        return payload

    def step(self, action: Union[Dict[str, Any], str], pause: float = 2.0):
        obs = self._decode_obs()  # get current screenshot for scaling
        payload = self._normalize_action(action, obs)
        self._request("/execute_action", data=payload)
        next_obs = self._decode_obs()
        reward, done = 0.0, False
        if self.task_type is not None:
            score = self._request(
                "/task/score",
                method="GET",
                params={
                    "task_type": self.task_type,
                    "task_idx": self.task_idx,
                },
            )["score"]
            reward = float(score)
            done = score >= 1.0 or payload.get("goal_status") == "success"
        info = {}
        return next_obs, reward, done, info

    def evaluate(self, task_config=None, trajectory=None):
        if self.test_task_llm_eval and self.llm_evaluator:
            return self.llm_evaluator.evaluate_task(task_config, trajectory)
        if self.task_type is None:
            return {"reward": -1}
        score = self._request(
            "/task/score",
            method="GET",
            params={"task_type": self.task_type, "task_idx": self.task_idx},
        )["score"]
        return {"reward": score}

    def close(self):
        self._request("/close")