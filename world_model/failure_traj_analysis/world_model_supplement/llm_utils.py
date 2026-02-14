from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class SimpleCallableLLM:
    """Adapter for a callable LLM: llm(prompt)->str"""
    def __init__(self, fn):
        self.fn = fn

    def generate(self, prompt: str) -> str:
        out = self.fn(prompt)
        return out if isinstance(out, str) else str(out)


class AttrLLM:
    """
    Adapter for objects that expose one of:
      - .generate(prompt)->str
      - .chat(prompt)->str
      - .invoke(prompt)->str
      - .__call__(prompt)->str
    """
    def __init__(self, obj: Any):
        self.obj = obj

    def generate(self, prompt: str) -> str:
        for name in ("generate", "chat", "invoke"):
            if hasattr(self.obj, name):
                fn = getattr(self.obj, name)
                out = fn(prompt)
                return out if isinstance(out, str) else str(out)
        if callable(self.obj):
            out = self.obj(prompt)
            return out if isinstance(out, str) else str(out)
        raise TypeError("Unsupported LLM object: expected generate/chat/invoke/callable.")


def ensure_llm_client(tool_llm: Any) -> LLMClient:
    if tool_llm is None:
        raise ValueError("tool_llm is required (must be callable or expose generate/chat/invoke).")
    # If it already looks like an LLM client, return it wrapped
    if hasattr(tool_llm, "generate") or hasattr(tool_llm, "chat") or hasattr(tool_llm, "invoke"):
        return AttrLLM(tool_llm)
    if callable(tool_llm):
        return SimpleCallableLLM(tool_llm)
    return AttrLLM(tool_llm)
