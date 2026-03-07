#!/usr/bin/env python3
"""Test script for multimodal reasoning bank distillation."""
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from memory.reasoning_bank import distill_multimodal_reasoning_items, parse_trajectory_rounds
from agent.llm_config import load_tool_llm
from config.argument_parser import config as load_config
import json

def test_parse_trajectory():
    """Test trajectory parsing."""
    test_file = "/home/sibo/GUI-Agent-Learn-From-Error/CoMEM-Agent-Inference/data/downloaded_datasets/webvoyager_memory/Amazon/qwen2.5-vl-32b/test/success/Amazon_Amazon_2.jsonl"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False
    
    print("Testing trajectory parsing...")
    with open(test_file, 'r') as f:
        obj = json.load(f)
    
    task, steps = parse_trajectory_rounds(obj)
    print(f"Task: {task[:100]}...")
    print(f"Total steps: {len(steps)}")
    
    for i, step in enumerate(steps[:3]):
        print(f"\nStep {i}:")
        print(f"  Has screenshot: {step['screenshot'] is not None}")
        print(f"  Action: {step['action']}")
        print(f"  Response preview: {step['response'][:100]}...")
    
    return True

def test_multimodal_distillation():
    """Test multimodal distillation on one trajectory."""
    test_file = "/home/sibo/GUI-Agent-Learn-From-Error/CoMEM-Agent-Inference/data/downloaded_datasets/webvoyager_memory/Amazon/qwen2.5-vl-32b/test/success/Amazon_Amazon_2.jsonl"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False
    
    print("\n" + "="*60)
    print("Testing multimodal distillation...")
    print("="*60)
    
    # Load LLM
    _argv_backup = sys.argv
    try:
        sys.argv = [_argv_backup[0]]
        parsed_args = load_config()
    finally:
        sys.argv = _argv_backup
    
    tool_llm = load_tool_llm(parsed_args, model_name='qwen2.5-vl')
    prompts_dir = os.path.join(_project_root, "agent", "prompts")
    
    # Load trajectory
    with open(test_file, 'r') as f:
        obj = json.load(f)
    
    # Run distillation
    print("\nRunning multimodal distillation...")
    items = distill_multimodal_reasoning_items(
        tool_llm,
        prompts_dir=prompts_dir,
        trajectory_obj=obj,
        is_success=True,
        dataset="webvoyager",
        domain="Amazon",
        task_id="Amazon_Amazon_2_test",
        source_path=test_file,
        max_items=2
    )
    
    print(f"\nExtracted {len(items)} items:")
    for i, item in enumerate(items):
        print(f"\nItem {i+1}:")
        print(f"  Key takeaway: {item.get('key_takeaway', 'N/A')}")
        print(f"  Step index: {item.get('step_index', 'N/A')}")
        print(f"  Before image: {item.get('before_image_path', 'N/A')}")
        print(f"  After image: {item.get('after_image_path', 'N/A')}")
        print(f"  Pre-state: {item.get('pre_state_hint', 'N/A')}")
        print(f"  Post-state: {item.get('post_state_hint', 'N/A')}")
        
        # Check if images were saved
        if item.get('before_image_path') and os.path.exists(item['before_image_path']):
            print(f"  ✓ Before image saved")
        if item.get('after_image_path') and os.path.exists(item['after_image_path']):
            print(f"  ✓ After image saved")
    
    return len(items) > 0

if __name__ == "__main__":
    print("Multimodal Reasoning Bank Test Suite")
    print("=" * 60)
    
    # Test 1: Parse trajectory
    if not test_parse_trajectory():
        print("\n❌ Trajectory parsing test failed")
        sys.exit(1)
    else:
        print("\n✓ Trajectory parsing test passed")
    
    # Test 2: Multimodal distillation
    if not test_multimodal_distillation():
        print("\n❌ Multimodal distillation test failed")
        sys.exit(1)
    else:
        print("\n✓ Multimodal distillation test passed")
    
    print("\n" + "="*60)
    print("All tests passed!")

