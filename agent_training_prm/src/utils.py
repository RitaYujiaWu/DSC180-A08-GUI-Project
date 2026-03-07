import os
import json
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


@dataclass
class JSONLLogger:
    out_dir: str
    name: str = "metrics"

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.path = os.path.join(self.out_dir, f"{self.name}.jsonl")

    def log(self, obj: Dict[str, Any]) -> None:
        obj = dict(obj)
        obj["_time"] = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)