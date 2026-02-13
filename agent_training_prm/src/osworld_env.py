# src/osworld_env.py
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import time
import json
import urllib.request
import urllib.error

from desktop_env.desktop_env import DesktopEnv


@dataclass
class EnvConfig:
    provider_name: str
    os_type: str
    path_to_vm: Optional[str]
    headless: bool
    action_space: str
    observation_type: str
    screen_width: int
    screen_height: int
    max_steps: int

    # CDP wait (chrome/web)
    cdp_wait_timeout_s: int = 45
    cdp_wait_interval_s: float = 1.0

    # NEW: give UI time after each action
    sleep_after_execution_s: float = 0.5


def _wait_for_cdp(host: str, port: int, timeout_s: int, interval_s: float) -> bool:
    url = f"http://{host}:{port}/json/version"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status != 200:
                    time.sleep(interval_s)
                    continue
                obj = json.loads(resp.read().decode("utf-8", errors="ignore"))
                if isinstance(obj, dict) and obj.get("webSocketDebuggerUrl"):
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
            pass
        time.sleep(interval_s)
    return False


class OSWorldEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        require_a11y_tree = cfg.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"]

        self.env = DesktopEnv(
            path_to_vm=cfg.path_to_vm,
            action_space=cfg.action_space,
            screen_size=(cfg.screen_width, cfg.screen_height),
            headless=cfg.headless,
            os_type=cfg.os_type,
            provider_name=cfg.provider_name,
            require_a11y_tree=require_a11y_tree,
        )
        self.steps = 0

    def reset(self, task_config: Dict[str, Any]) -> Any:
        self.steps = 0
        obs = self.env.reset(task_config=task_config)

        host = getattr(self.env, "vm_ip", "localhost")
        port = int(getattr(self.env, "chromium_port", 9222))

        ok = _wait_for_cdp(
            host=host,
            port=port,
            timeout_s=int(self.cfg.cdp_wait_timeout_s),
            interval_s=float(self.cfg.cdp_wait_interval_s),
        )
        if not ok:
            print(f"[WARN] CDP not ready at http://{host}:{port} after {self.cfg.cdp_wait_timeout_s}s")

        return obs

    def step(self, action: str) -> Tuple[Any, float, bool, Dict[str, Any]]:
        self.steps += 1

        # IMPORTANT: step executes action, then we sleep so next obs is “settled”
        obs, env_reward, done, info = self.env.step(action)
        time.sleep(float(self.cfg.sleep_after_execution_s))

        if self.steps >= self.cfg.max_steps:
            done = True

        if not isinstance(info, dict):
            info = {"info": info}
        return obs, float(env_reward), bool(done), info