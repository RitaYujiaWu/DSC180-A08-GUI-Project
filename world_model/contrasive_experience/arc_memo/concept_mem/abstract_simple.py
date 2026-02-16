#!/usr/bin/env python3
"""
GUI Agent Concept Abstraction Script

This script abstracts concepts from GUI agent pseudocode solutions.
It takes the output from pseudocode_simple.py and generates reusable concepts.
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

import yaml

from constants import ABSTRACTION_INSTR_PATH, REPO_ROOT
from utils import (
    extract_yaml_block,
    read_json,
    read_yaml,
    write_json,
)
from concept import Concept, ConceptMemory
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from agent.llm_config import DirectVLLMModel

# Setup logging
logger = logging.getLogger(__name__)

# Load abstraction instructions
ABSTRACTION_INSTR = ABSTRACTION_INSTR_PATH.read_text()

CONCEPT_GEN_EX_TEMPLATE = """\
{header}
Task Solution:
```
{summary}
{solution}
```
Annotation:
```yaml
{annotation}
```"""

CONCEPT_MERGE_PROMPT = """\
You are given multiple implementations of the same GUI concept "{concept_name}" from different trajectories.
Your task is to merge them into 1-3 unified, high-quality concepts that capture the essential patterns.

Existing Concept Implementations:
{implementations}

Guidelines:
1. If the implementations are very similar, merge them into ONE concept with a refined implementation.
2. If there are distinct approaches (e.g., different workflows, distinct patterns), create 2-3 separate concepts with clear names (e.g., "concept_name_v1", "concept_name_v2").
3. Remove redundancy and keep only the most important steps.
4. Preserve meaningful cues that help identify when to use this concept.
5. Use the same type as the input concepts.

Output Format (YAML):
```yaml
- concept: <unified_or_refined_name>
  type: <same_type>
  cues:
    - <key identifying cue 1>
    - <key identifying cue 2>
  implementation:
    - <refined step 1>
    - <refined step 2>
```

Generate the merged concept(s) now:
"""


def merge_concepts_with_llm(
    concept_name: str,
    concept_type: str,
    implementations: List[Dict],
    llm_client: DirectVLLMModel,
) -> List[Dict]:
    """
    Use LLM to merge multiple implementations of the same concept into 1-3 unified concepts.
    
    Args:
        concept_name: Name of the concept to merge
        concept_type: Type of the concept
        implementations: List of dicts containing 'cues', 'implementation', 'used_in'
        llm_client: LLM client for generation
        
    Returns:
        List of merged concept dictionaries
    """
    if len(implementations) == 1:
        # No need to merge if there's only one implementation
        return [implementations[0]]
    
    # Format implementations for the prompt
    impl_text = []
    for i, impl in enumerate(implementations, 1):
        impl_text.append(f"Implementation {i} (from tasks: {', '.join(impl.get('used_in', []))}):")
        impl_text.append(f"  Cues:")
        for cue in impl.get('cues', []):
            impl_text.append(f"    - {cue}")
        impl_text.append(f"  Implementation:")
        for step in impl.get('implementation', []):
            impl_text.append(f"    - {step}")
        impl_text.append("")
    
    prompt = CONCEPT_MERGE_PROMPT.format(
        concept_name=concept_name,
        implementations="\n".join(impl_text)
    )
    
    try:
        response, _, _ = llm_client.chat(messages=[{"role": "user", "content": prompt}], stream=False)
        response = response.content        
        
        # Parse YAML output
        yaml_block = extract_yaml_block(response)
        merged_concepts = yaml.safe_load(yaml_block)
        
        if not isinstance(merged_concepts, list):
            logger.warning(f"Expected list of concepts for {concept_name}, got {type(merged_concepts)}")
            # Fallback: keep the first implementation
            return [implementations[0]]
        
        # Validate and add metadata
        for concept in merged_concepts:
            if 'concept' not in concept or 'type' not in concept:
                logger.warning(f"Merged concept missing required fields: {concept}")
                continue
            # Inherit type if not specified
            if not concept.get('type'):
                concept['type'] = concept_type
            # Aggregate used_in from all implementations
            all_used_in = []
            for impl in implementations:
                all_used_in.extend(impl.get('used_in', []))
            concept['used_in'] = list(set(all_used_in))
        
        logger.info(f"Merged {len(implementations)} implementations of '{concept_name}' into {len(merged_concepts)} concept(s)")
        return merged_concepts
        
    except Exception as e:
        logger.error(f"Error merging concepts for {concept_name}: {e}")
        # Fallback: keep the first implementation
        return [implementations[0]]


def format_concept_examples(
    task_solutions: dict[str, str],
    annotations: dict[str, list[dict]],
    header_template: str = "## Example {example_number}",
    delimiter: str = "\n\n",
    skip_tasks: list[str] | None = None,
) -> str:
    """Format ICL examples for task solution -> concept abstraction task"""
    example_strings = []
    for i, (task_id, annotation) in enumerate(annotations.items(), start=1):
        if skip_tasks and task_id in skip_tasks:
            continue
        if task_id not in task_solutions:
            if "pseudocode" in annotation:
                solution = f'\n'.join([f' {pseudocode}' for pseudocode in annotation["pseudocode"]])
            else:
                logger.info(f"Missing solution for ICL example task {task_id}")
                continue
        else:
            solution = task_solutions[task_id]
        summary = annotation.get("summary", "")
        if summary:
            summary = f"# task summary: {summary.strip()}\n"
        formatted_annotation = yaml.dump(annotation["concepts"], sort_keys=False)
        example = CONCEPT_GEN_EX_TEMPLATE.format(
            header=header_template.format(example_number=i),
            summary=summary,
            solution=solution,
            annotation=formatted_annotation.strip(),
        )
        example_strings.append(example)
    return delimiter.join(example_strings)


def parse_concept_model_output(
    model_output: str,
) -> list[dict]:
    """Parse LLM output to extract concept annotations"""
    # returns list[concept annotations in dict form]
    # first get yaml block
    yaml_string = extract_yaml_block(model_output)
    logger.info(f"YAML string: {yaml_string}")
    if not yaml_string:
        logger.info("No YAML block found in model output")
        return []

    # parse the yaml string
    try:
        yaml_data = yaml.safe_load(yaml_string)
        assert isinstance(yaml_data, list), "expected concept list"
        return yaml_data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error parsing concept YAML: {e}")
        return []


def generate_concepts_batch(
    tasks: list[str],
    solutions: dict[str, str],
    summaries: dict[str, str] | None,
    examples: dict[str, dict],
    concept_mem: ConceptMemory,
    llm_client,
    output_dir: Path | None = None,
) -> dict[str, list[dict]]:
    """Generate concepts for a batch of tasks"""
    # prepare ICL demo string
    formatted_examples = format_concept_examples(
        task_solutions=solutions,
        annotations=examples,
    )
    
    # prepare prompts
    task_ids = []
    prompts = []
    model_output = []
    
    for task_id in tasks:
        if task_id not in solutions:
            logger.warning(f"Missing solution for task {task_id}, skipping")
            continue
        summary = summaries.get(task_id, "") if summaries else ""
        if summary:
            summary = f"# task summary: {summary.strip()}\n"
        prompt = ABSTRACTION_INSTR.format(
            examples=formatted_examples,
            concept_list=concept_mem.to_string(),
            summary=summary,
            pseudocode=solutions[task_id],
        ).strip()
        task_ids.append(task_id)
        prompts.append(prompt)
        
        # Call LLM
        response, _, _ = llm_client.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        model_output.append(response.content)

    # parse results
    results = {}
    for task_id, completion in zip(task_ids, model_output):
        if not completion:
            logger.warning(f"No completions for task {task_id}")
            continue
        try:
            concept_list = parse_concept_model_output(completion)
            results[task_id] = concept_list
        except Exception as e:
            logger.error(f"Error parsing output for task {task_id}: {e}")
            continue

    # save to output directory if specified
    if output_dir:
        output_file = output_dir / "concept_lists.json"
        write_json(results, output_file, indent=True)

    return results


def generate_concepts(
    tasks: list[str],
    solutions: dict[str, str],
    summaries: dict[str, str] | None,
    examples: dict[str, dict],
    concept_mem: ConceptMemory,
    llm_client,
    output_dir: Path | None = None,
) -> None:
    """Generate concepts for all tasks and update concept memory"""
    logger.info(f"Generating concepts for {len(tasks)} tasks")
    
    if not os.path.exists(output_dir/"concept_lists.json"):
        print("Generating concept lists...")
        concept_batch = generate_concepts_batch(
            tasks=tasks,
            solutions=solutions,
            summaries=summaries,
            examples=examples,
            concept_mem=concept_mem,
            llm_client=llm_client,
            output_dir=output_dir,
        )
    else:
        print("Loading concept lists from file...")
        concept_batch = read_json(output_dir/"concept_lists.json")
    
    # First, collect all concepts by name
    concepts_by_name: Dict[str, List[Dict]] = {}
    for task_id, concept_list in concept_batch.items():
        for concept in concept_list:
            concept_name = concept.get("concept") or concept.get("name")
            if not concept_name:
                continue
            
            if concept_name not in concepts_by_name:
                concepts_by_name[concept_name] = []
            
            # Store concept with task_id
            concept_with_task = concept.copy()
            concept_with_task['used_in'] = [task_id]
            concepts_by_name[concept_name].append(concept_with_task)
    
    logger.info(f"Collected {len(concepts_by_name)} unique concept names")
    
    # Merge concepts with same name using LLM
    for concept_name, implementations in concepts_by_name.items():
        if len(implementations) == 1:
            # No merging needed, add directly
            task_id = implementations[0]['used_in'][0]
            concept_mem.write_concept(task_id, implementations[0])
        else:
            logger.info(f"Merging {len(implementations)} implementations of '{concept_name}'")
            concept_type = implementations[0].get('type') or implementations[0].get('kind', 'unknown')
            
            # Use LLM to merge
            merged_concepts = merge_concepts_with_llm(
                concept_name=concept_name,
                concept_type=concept_type,
                implementations=implementations,
                llm_client=llm_client,
            )
            
            # Add merged concepts to memory
            for merged_concept in merged_concepts:
                # Use the first task_id from used_in list
                task_ids = merged_concept.get('used_in', [implementations[0]['used_in'][0]])
                primary_task_id = task_ids[0] if task_ids else "merged"
                concept_mem.write_concept(primary_task_id, merged_concept)
    
    # Save final memory
    if output_dir:
        mem_file = output_dir / "memory.json"
        concept_mem.save_to_file(mem_file)
        logger.info(f"Saved concept memory to {mem_file}")


def setup_logging(output_dir: Path) -> None:
    """Setup logging to both console and file"""
    # Create output directory if it doesn't exist
    logs_dir = output_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(logs_dir / 'concept_abstraction.log')  # File output
        ]
    )


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file"""
    config_path = Path(config_file)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="GUI Agent Concept Abstraction")
    parser.add_argument("--config", type=str, help="Path to config file", default="/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/arc_memo/configs/config_abstract.yaml")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup output directory
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {output_dir / 'logs' / 'concept_abstraction.log'}")
    
    # Load hand annotations for ICL examples
    hand_annotation_path = Path(config["hand_annotations_file"])
    if not hand_annotation_path.is_absolute():
        hand_annotation_path = REPO_ROOT / hand_annotation_path
    hand_annotations = read_yaml(hand_annotation_path)
    
    # Load pseudocode from previous step
    pseudocode_path = Path(config["pseudocode_file"])
    if not pseudocode_path.is_absolute():
        pseudocode_path = REPO_ROOT / pseudocode_path
    pseudocode_data = read_json(str(pseudocode_path))
    
    # Extract pseudocode and summaries
    pseudocode = {}
    summaries = {}
    for task_id, entry in pseudocode_data.items():
        if isinstance(entry, str):
            pseudocode[task_id] = entry
        else:
            pseudocode[task_id] = entry["pseudocode"]
            summaries[task_id] = entry.get("summary", "")
    
    # Setup target tasks
    limit = config["limit_tasks"]
    target_tasks = []
    
    for task_id in pseudocode.keys():
        if task_id in hand_annotations or (limit and len(target_tasks) >= limit):
            continue
        target_tasks.append(task_id)
    
    logger.info(f"Processing {len(target_tasks)} tasks for concept abstraction")
    
    # Initialize LLM client
    llm_client = DirectVLLMModel(
        server_url=config["model"]["server_url"],
        model_name=config["model"]["name"],
        api_key=config["model"]["api_key"]
    )
    
    # Initialize concept memory
    concept_mem = ConceptMemory()
    
    # Load example concepts
    ex_con_path = Path(config["example_concepts"])
    if not ex_con_path.is_absolute():
        ex_con_path = REPO_ROOT / ex_con_path
    example_concepts = read_yaml(ex_con_path)
    
    # Initialize concept memory with example concepts
    if "concepts" in example_concepts:
        for concept in example_concepts["concepts"]:
            concept_mem.write_concept("example", concept)
    
    # Initialize from hand annotations
    concept_mem.initialize_from_annotations(hand_annotations)
    
    # Simple one-shot run (no resume / no periodic checkpoints)
    generate_concepts(
        tasks=target_tasks,
        solutions=pseudocode,
        summaries=summaries,
        examples=hand_annotations,
        concept_mem=concept_mem,
        llm_client=llm_client,
        output_dir=output_dir,
    )
    
    logger.info(f"Concept abstraction completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
