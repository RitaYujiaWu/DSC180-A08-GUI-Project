import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def parse_markup_tag(text: str, tag: str) -> list[str]:
    """Extract content from XML-like tags in text."""
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def extract_yaml_block(text: str) -> str | None:
    """Extract YAML block from markdown code fence."""
    pattern = r"```yaml\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    match = match.group(1) if match else None
    if match:
        match = match.strip().strip('```yaml').strip('```')
    return match


def read_json(path: Path | str) -> Any:
    """Read JSON file."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / path
    with open(path, 'r') as f:
        return json.load(f)


def write_json(data: Any, path: Path | str, indent: bool = False) -> None:
    """Write data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2 if indent else None)


def read_yaml(path: Path | str) -> Any:
    """Read YAML file."""
    path = Path(path)
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def write_yaml(data: Any, path: Path | str) -> None:
    """Write data to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def run_llm_job(
    prompts: list[str],
    metadata: list[str],
    llm_client
) -> list[list[str]]:
    """Run LLM job with prompts and return completions."""
    # This is a simplified version - you'll need to implement the actual LLM calling logic
    # based on your LLM client setup
    completions = []
    for prompt, meta in zip(prompts, metadata):
        try:
            # Replace this with actual LLM call
            response, _, _ = llm_client.chat(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],stream=False)
            completions.append([response.content])
        except Exception as e:
            logger.error(f"Error generating completion for {meta}: {e}")
            completions.append([])
    
    return completions

