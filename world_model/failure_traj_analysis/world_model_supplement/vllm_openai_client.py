from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import requests


@dataclass
class VLLMOpenAIClient:
    """

    Example base_url:
      - http://localhost:9100/v1  (Qwen2.5-VL-7B-Instruct)
      - http://localhost:9101/v1  (Tongyi-MiA/UI-Ins-7B)

    Usage:
        llm = VLLMOpenAIClient(base_url="http://localhost:9100/v1", model="Qwen/Qwen2.5-VL-7B-Instruct")
        text = llm.generate("hello")
    """
    base_url: str
    model: str
    api_key: str = "EMPTY"   # vLLM ignores key by default, but keep header for compatibility
    temperature: float = 0.2
    max_tokens: int = 500
    timeout: int = 120

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"

        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        # OpenAI format: choices[0].message.content
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            # fallback: dump something readable
            return str(data)
