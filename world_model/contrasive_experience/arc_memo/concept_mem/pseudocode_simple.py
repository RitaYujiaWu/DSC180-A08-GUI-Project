#!/usr/bin/env python3
"""
Simple GUI agent pseudocode generation without Hydra dependency.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, Any, Optional
import yaml
import sys
import argparse
import json

sys.path.append(str(Path(__file__).parent.parent.parent))
from agent.llm_config import DirectVLLMModel
from constants import DATA_DIR, REPO_ROOT
from utils import (
    parse_markup_tag,
    read_json,
    read_yaml,
    run_llm_job,
    write_json
)

# Configure logging
def setup_logging(output_dir: Path):
    """Setup logging to both console and file"""
    log_file = output_dir / "logs" / "pseudocode_generation.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = logging.getLogger(__name__)

PSEUDOCODE_INSTR_PATH = DATA_DIR / "abstract_anno/gui/pseudocode_instr.txt"
PSEUDOCODE_INSTR = PSEUDOCODE_INSTR_PATH.read_text()


PSEUDOCODE_GEN_EX_TEMPLATE = """\
{header}
Task: {task_description}
Agent Trajectory:
```python
{trajectory}
```
Annotation:
<pseudocode>
{pseudocode}
</pseudocode>
<summary>
{summary}
</summary>"""


def format_pseudocode_examples(
    task_solutions: dict[str, str],
    annotations: dict[str, dict],
    header_template: str = "## Example {example_number}",
    transform_trajectory: Callable[[str], str] | None = None,
    delimiter: str = "\n\n",
) -> str:
    """Format ICL examples for trajectory -> pseudocode task"""
    example_strings = []
    for i, (task_id, annotation) in enumerate(annotations.items(), start=1):
        if task_id not in task_solutions:
            logger.info(f"Missing trajectory for ICL example task {task_id}")
            continue
        trajectory = task_solutions[task_id]
        if "pseudocode" not in annotation or "summary" not in annotation:
            logger.info(f"ICL example {task_id} missing pseudocode or summary")
            continue
        if transform_trajectory:
            trajectory = transform_trajectory(trajectory)
        formatted_pseudocode = yaml.dump(annotation["pseudocode"], sort_keys=False)
        task_description = annotation.get("task_description", "Unknown task")
        example = PSEUDOCODE_GEN_EX_TEMPLATE.format(
            header=header_template.format(example_number=i),
            task_description=task_description,
            trajectory=trajectory,
            pseudocode=formatted_pseudocode.strip(),
            summary=annotation["summary"],
        )
        example_strings.append(example)
    return delimiter.join(example_strings)


def parse_model_output(
    model_output: str,
) -> tuple[str, str]:
    """Parse LLM output to extract pseudocode and summary"""
    # returns (pseudocode, summary)
    code_results = parse_markup_tag(model_output, "pseudocode")
    if len(code_results) != 1:
        logger.info(
            f"parse error: expected 1 pseudocode block, got {len(code_results)}"
        )
        pseudocode = ""
    else:
        pseudocode = code_results[0].strip()
    summary_results = parse_markup_tag(model_output, "summary")
    if len(summary_results) != 1:
        logger.info(
            f"parse error: expected 1 summary block, got {len(summary_results)}"
        )
        summary = ""
    else:
        summary = summary_results[0].strip()
    return pseudocode, summary


def generate_pseudocode(
    tasks: list[str],
    trajectories: dict[str, str],
    example_trajectories: dict[str, str],
    task_descriptions: dict[str, str] | None,
    example_annotations: dict[str, dict],
    example_concepts: str,
    llm_client: DirectVLLMModel,
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Generate pseudocode from GUI agent trajectories"""
    # prepare ICL demo string
    formatted_examples = format_pseudocode_examples(
        task_solutions=example_trajectories,
        annotations=example_annotations,
    )
    # prepare prompts
    task_ids = []
    prompts = []
    model_output = []
    for task_id in tasks:
        if task_id not in trajectories:
            logger.warning(f"Missing trajectory for task {task_id}, skipping")
            continue
        task_description = task_descriptions.get(task_id, "Unknown task") if task_descriptions else "Unknown task"
        logger.info(f"Task description: {task_description}")
        prompt = PSEUDOCODE_INSTR.format(
            examples=formatted_examples,
            concepts=example_concepts,
            task_description=task_description,
            trajectory=trajectories[task_id],
        ).strip()
        logger.info(f"prompt: {prompt}")
        task_ids.append(task_id)
        prompts.append(prompt)
        response, _, _ = llm_client.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],stream=False)
        logger.info(f"Model output: {response.content}")
        model_output.append(response.content)

    # parse results
    results = {}
    for task_id, completion in zip(task_ids, model_output):
        if not completion:
            logger.warning(f"No completions for task {task_id}")
            continue
        try:
            pseudocode, summary = parse_model_output(completion)
            results[task_id] = {
                "pseudocode": pseudocode,
                "summary": summary,
                "trajectory": trajectories[task_id],
            }
        except Exception as e:
            logger.error(f"Error parsing output for task {task_id}: {e}")
            continue

    # save to output directory if specified
    if output_dir:
        output_file = output_dir / "initial_analysis.json"
        write_json(results, output_file, indent=True)
    return results

def _load_trajectories(trajectory_file: str) -> dict[str, str]:
    """Load GUI agent trajectories and convert to string format"""
    trajectory_data = read_json(trajectory_file)
    trajectory_strings = {}
    
    for task_id, actions in trajectory_data.items():
        trajectory_strings[task_id] = '\n'.join([f'Action {i+1}: {action}' for i, action in enumerate(actions)])
    
    return trajectory_strings

def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main function to run the GUI agent pseudocode generation process"""
    parser = argparse.ArgumentParser(description="Generate pseudocode from GUI agent trajectories")
    parser.add_argument("--config", type=str, default="/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/arc_memo/configs/config_simple.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {output_dir / 'pseudocode_generation.log'}")
    
    # Load instructions and ICL demos
    hand_annotation_path = Path(config["hand_annotations_file"])
    if not hand_annotation_path.is_absolute():
        hand_annotation_path = REPO_ROOT / hand_annotation_path
    hand_annotations = read_yaml(hand_annotation_path)
    
    example_trajectory_path = Path(config["example_trajectories"])
    if not example_trajectory_path.is_absolute():
        example_trajectory_path = REPO_ROOT / example_trajectory_path
    example_trajectories = read_json(example_trajectory_path)
    
    # Load trajectories
    if config["trajectories"]:
        trajectory_path = Path(config["trajectories"])
        if not trajectory_path.is_absolute():
            trajectory_path = REPO_ROOT / trajectory_path
        trajectories = _load_trajectories(str(trajectory_path))
    else:
        trajectories = {}
    
    # Set up target tasks
    limit = config["limit_tasks"]
    target_tasks = []
    
    if config.get("task_ids") is None:
        for task_id in trajectories.keys():
            if task_id in hand_annotations or (limit and len(target_tasks) >= limit):
                continue
            target_tasks.append(task_id)
    else:
        task_ids = read_json(config["task_ids"])
        if limit:
            target_tasks = task_ids[:limit]
        else:
            target_tasks = task_ids
    
    # Load example concepts
    ex_con_path = Path(config["example_concepts"])
    if not ex_con_path.is_absolute():
        ex_con_path = REPO_ROOT / ex_con_path
    example_concepts = ex_con_path.read_text().strip()
    
    # Load task descriptions
    task_descriptions = None
    if config["task_descriptions"]:
        task_desc_path = Path(config["task_descriptions"])
        if not task_desc_path.is_absolute():
            task_desc_path = REPO_ROOT / task_desc_path
        task_descriptions = read_json(str(task_desc_path))
    
    # Model setup
    llm_client = DirectVLLMModel(
        model_name=config["model"]["name"],
        server_url=config["model"]["server_url"],
        api_key=config["model"]["api_key"],
    )
    
    # Run pseudocode generation
    generate_pseudocode(
        tasks=target_tasks,
        trajectories=trajectories,
        example_trajectories=example_trajectories,
        task_descriptions=task_descriptions,
        example_annotations=hand_annotations,
        example_concepts=example_concepts,
        llm_client=llm_client,
        output_dir=output_dir,
    )
    logger.info(f"Wrote to output directory: {output_dir}")


if __name__ == "__main__":
    main()
