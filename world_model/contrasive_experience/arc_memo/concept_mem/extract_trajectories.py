#!/usr/bin/env python3
"""
Extract trajectories from success folders in webvoyager_memory.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from actions.help_functions import parse_action_json, parse_natural_language_with_llm
from agent.llm_config import DirectVLLMModel
from arc_memo.concept_mem.constants import DATA_DIR


def extract_action_from_response(response: str, tool_llm: DirectVLLMModel) -> Optional[Dict]:
    """
    Extract action from a single response using the same logic as agent.py
    """
    try:
        # Parse the response content to extract function call
        parsed_response = parse_action_json(response)
        
        if isinstance(parsed_response, dict) and 'function_call' in parsed_response:
            action_data = str(parsed_response['function_call'])
            return action_data
        else:
            action_data = parse_natural_language_with_llm(response, tool_llm)
            return action_data
    except Exception as e:
        print(f"Error extracting action from response: {e}")
        return None


def is_high_quality_trajectory(responses: List[str]) -> bool:
    """
    Filter out high quality trajectories using negative phrases check
    """
    if len(responses) >= 10:
        return False
    if not responses:
        return False
    # Get the last response
    last_response = responses[-1]
    # Check for negative phrases
    negative_phrases = ['cannot', 'not found', 'not available', "can't"]
    last_response_lower = last_response.lower()
    
    for phrase in negative_phrases:
        if phrase in last_response_lower:
            return False
    return True


def extract_trajectory_from_file(file_path: Path, tool_llm: DirectVLLMModel) -> Optional[Dict]:
    """
    Extract trajectory from a single success file
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        round_num = data['total_rounds']
        conversations = data['rounds']
        responses = [conv['response'] for conv in conversations]
        task_description = data.get('task_description', 'Unknown task')
        
        # Filter for high quality trajectories
        if not is_high_quality_trajectory(responses):
            return None
            
        # Extract actions from responses
        actions = []
        for response in responses:
            action = extract_action_from_response(response, tool_llm)
            if action:
                actions.append(action)
        
        if not actions or len(actions) < 3:
            return None
            
        return {
            'task_description': task_description,
            'actions': actions,
            'total_rounds': round_num,
            'conversation_id': data.get('conversation_id', 'unknown')
        }
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def find_success_files(base_dir: Path) -> List[Path]:
    """
    Find all success files in the webvoyager_memory directory
    """
    success_files = []
    
    for root, dirs, files in os.walk(base_dir):
        if 'success' in root:
            for file in files:
                if file.endswith('.jsonl'):
                    success_files.append(Path(root) / file)
    
    return success_files


def main():
    """
    Main function to extract trajectories from all success folders
    """
    # Base directory containing success folders
    base_dir = Path("/home/wenyi/CoMEM-Agent/CoMEM-Agent-Inference/webvoyager_memory")
    tool_llm = DirectVLLMModel(model_name="Qwen/Qwen2.5-VL-7B-Instruct", server_url='http://localhost:8000/v1', api_key='EMPTY')
    
    # Find all success files
    print("Finding success files...")
    success_files = find_success_files(base_dir)
    print(f"Found {len(success_files)} success files")
    
    # Extract trajectories
    trajectories = {}
    processed_count = 0
    high_quality_count = 0
    
    for file_path in success_files:
        print(f"Processing {os.path.basename(file_path)}...")
        trajectory = extract_trajectory_from_file(file_path, tool_llm)
        
        if trajectory:
            # Use conversation_id as the key, or generate one if not available
            key = trajectory['conversation_id']
            trajectories[key] = trajectory['actions']
            high_quality_count += 1
            print(f"  ✓ Extracted {len(trajectory['actions'])} actions")
        else:
            print(f"  ✗ Skipped (low quality or no valid actions)")
        
        processed_count += 1
    
    print(f"\nExtraction complete:")
    print(f"  Total files processed: {processed_count}")
    print(f"  High quality trajectories: {high_quality_count}")
    print(f"  Total actions extracted: {sum(len(actions) for actions in trajectories.values())}")
    
    # Create output directory
    output_dir = DATA_DIR / "webvoyager_memory"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trajectories to JSON file
    output_file = output_dir / "extracted_trajectories.json"
    with open(output_file, 'w') as f:
        json.dump(trajectories, f, indent=2)
    
    print(f"\nTrajectories saved to: {output_file}")
    
    # Also save task descriptions separately
    task_descriptions = {}
    for file_path in success_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            conv_id = data.get('conversation_id', f'unknown_{len(task_descriptions)}')
            task_desc = data.get('task_description', 'Unknown task')
            task_descriptions[conv_id] = task_desc
        except:
            continue
    
    task_desc_file = output_dir / "extracted_task_descriptions.json"
    with open(task_desc_file, 'w') as f:
        json.dump(task_descriptions, f, indent=2)
    
    print(f"Task descriptions saved to: {task_desc_file}")


if __name__ == "__main__":
    main()

