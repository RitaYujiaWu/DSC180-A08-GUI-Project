#!/usr/bin/env python3
"""Test Stage 1 only: VLM identifies key steps from text trajectory."""
import sys
import os
import json

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agent.llm_config import load_tool_llm
from config.argument_parser import config as load_config

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def parse_trajectory_simple(obj):
    """Simple trajectory parser for testing."""
    task = obj.get('task_description', '')
    rounds = obj.get('rounds', [])
    
    steps = []
    for r in rounds:
        response = r.get('response', '')
        # Extract action from response
        response_clean = response.replace('```json', '').replace('```', '')
        import re
        match = re.search(r'\{\s*"name"\s*:', response_clean)
        if match:
            start = match.start()
            brace_count = 0
            end = start
            for i in range(start, len(response_clean)):
                if response_clean[i] == '{':
                    brace_count += 1
                elif response_clean[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            try:
                action = json.loads(response_clean[start:end])
            except:
                action = None
        else:
            action = None
        
        steps.append({
            'action': action,
            'response': response
        })
    
    return task, steps

def test_stage1(trajectory_file, is_success=True):
    """Run Stage 1: text-only key step identification."""
    
    print("="*80)
    print(f"Stage 1 Test: {os.path.basename(trajectory_file)}")
    print("="*80)
    
    # Load LLM
    print("\n1. Loading VLM...")
    _argv_backup = sys.argv
    try:
        sys.argv = [_argv_backup[0]]
        parsed_args = load_config()
    finally:
        sys.argv = _argv_backup
    
    tool_llm = load_tool_llm(parsed_args, model_name='qwen2.5-vl')
    print("   ✓ VLM loaded")
    
    # Load trajectory
    print("\n2. Loading trajectory...")
    with open(trajectory_file, 'r') as f:
        obj = json.load(f)
    
    task, steps = parse_trajectory_simple(obj)
    print(f"   ✓ Task: {task[:80]}...")
    print(f"   ✓ Total steps: {len(steps)}")
    
    # Build text trajectory
    print("\n3. Building text trajectory...")
    traj_lines = []
    for i, step in enumerate(steps):
        action = step['action']
        response = step['response'][:400]  # truncate
        action_str = json.dumps(action) if action else "N/A"
        traj_lines.append(f"Step {i}: Action={action_str} Response={response}")
    
    trajectory_text = "\n".join(traj_lines)
    print(f"   ✓ Text trajectory: {len(trajectory_text)} chars")
    
    # Load prompt
    print("\n4. Loading Stage 1 prompt...")
    prompts_dir = os.path.join(_project_root, "agent", "prompts")
    prompt_path = os.path.join(prompts_dir, "reasoning_bank_mm_identify_steps.md")
    template = load_prompt(prompt_path)
    
    outcome = "success" if is_success else "failure"
    prompt_text = (template
                   .replace("{task}", task)
                   .replace("{outcome}", outcome)
                   .replace("{trajectory_text}", trajectory_text))
    
    print(f"   ✓ Prompt ready: {len(prompt_text)} chars")
    
    # Call VLM
    print("\n5. Calling VLM to identify key steps...")
    print("   (This may take 10-30 seconds...)")
    
    messages = [{"role": "user", "content": prompt_text}]
    
    try:
        resp, _, _ = tool_llm.chat(messages=messages, stream=False)
        text = getattr(resp, "content", "") or str(resp)
        
        print("\n" + "="*80)
        print("VLM Response:")
        print("="*80)
        print(text)
        print("="*80)
        
        # Parse JSON output
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            key_steps = json.loads(json_str)
            
            print("\n✓ Parsed key steps:")
            for item in key_steps:
                step_idx = item.get('step_index', 'N/A')
                reason = item.get('reason', 'N/A')
                print(f"\n  Step {step_idx}:")
                print(f"    Reason: {reason}")
                
                # Show the actual action at this step
                if step_idx != 'N/A' and 0 <= step_idx < len(steps):
                    actual_action = steps[step_idx]['action']
                    if actual_action:
                        print(f"    Action: {actual_action.get('name', 'N/A')}")
                        if 'arguments' in actual_action:
                            args = actual_action['arguments']
                            if 'text' in args:
                                print(f"    Text: {args['text'][:60]}...")
                            if 'aria_label' in args:
                                print(f"    Target: {args['aria_label'][:60]}...")
            
            return key_steps
        else:
            print("\n✗ Failed to parse JSON from response")
            return None
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test file
    test_file = "/home/sibo/GUI-Agent-Learn-From-Error/CoMEM-Agent-Inference/data/downloaded_datasets/webvoyager_memory/Amazon/qwen2.5-vl-32b/test/success/Amazon_Amazon_2.jsonl"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        print("Please update the path in this script.")
        sys.exit(1)
    
    result = test_stage1(test_file, is_success=True)
    
    if result:
        print("\n" + "="*80)
        print("✓ Stage 1 test completed successfully!")
        print(f"✓ Identified {len(result)} key step(s)")
        print("="*80)
    else:
        print("\n✗ Stage 1 test failed")
        sys.exit(1)

