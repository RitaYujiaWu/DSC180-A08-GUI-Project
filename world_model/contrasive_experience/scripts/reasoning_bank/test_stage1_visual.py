#!/usr/bin/env python3
"""Test Stage 1 with VISUAL mode: VLM sees all screenshots to identify key steps."""
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
    """Parse trajectory with screenshots."""
    task = obj.get('task_description', '')
    rounds = obj.get('rounds', [])
    
    steps = []
    for r in rounds:
        # Extract screenshot
        screenshot = None
        for msg in r.get('messages', []):
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        screenshot = item['image_url']['url']
                        break
        
        # Extract action
        response = r.get('response', '')
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
            'screenshot': screenshot,
            'action': action,
            'response': response
        })
    
    return task, steps

def test_stage1_visual(trajectory_file, is_success=True):
    """Run Stage 1 with visual mode: VLM sees all screenshots."""
    
    print("="*80)
    print(f"Stage 1 VISUAL Test: {os.path.basename(trajectory_file)}")
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
    
    screenshots_count = sum(1 for s in steps if s['screenshot'])
    print(f"   ✓ Screenshots: {screenshots_count}/{len(steps)}")
    
    # Build multimodal trajectory
    print("\n3. Building multimodal trajectory...")
    prompts_dir = os.path.join(_project_root, "agent", "prompts")
    prompt_path = os.path.join(prompts_dir, "reasoning_bank_mm_identify_steps_visual.md")
    template = load_prompt(prompt_path)
    
    outcome = "success" if is_success else "failure"
    header = template.replace("{task}", task).replace("{outcome}", outcome).replace("{trajectory_with_images}", "")
    header = header.split("**Trajectory with screenshots:**")[0] + "**Trajectory with screenshots:**\n"
    
    content_items = []
    content_items.append({"type": "text", "text": header})
    
    # Add each step with screenshot
    for i, step in enumerate(steps):
        action = step['action']
        response = step['response'][:300]
        action_str = json.dumps(action) if action else "N/A"
        
        step_text = f"\n**Step {i}:**\nAction: {action_str}\nResponse: {response}\n"
        content_items.append({"type": "text", "text": step_text})
        
        if step['screenshot']:
            content_items.append({
                "type": "image_url",
                "image_url": {"url": step['screenshot']}
            })
    
    closing = f"\nNow analyze the trajectory and identify the 1-2 most critical steps that caused the {outcome}."
    content_items.append({"type": "text", "text": closing})
    
    print(f"   ✓ Multimodal content: {len(content_items)} items ({screenshots_count} images)")
    
    # Estimate tokens
    text_tokens = sum(len(item.get('text', '')) for item in content_items if item['type'] == 'text') // 4
    image_tokens = screenshots_count * 800  # rough estimate
    total_tokens = text_tokens + image_tokens
    print(f"   ✓ Estimated tokens: ~{total_tokens:,} ({text_tokens:,} text + {image_tokens:,} image)")
    
    # Call VLM
    print("\n4. Calling VLM to identify key steps (with all screenshots)...")
    print("   ⚠️  This may take 30-60 seconds due to multiple images...")
    
    messages = [{"role": "user", "content": content_items}]
    
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
                            if 'element_id' in args:
                                print(f"    Element ID: {args['element_id']}")
            
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Stage 1 with visual mode")
    parser.add_argument("--trajectory", type=str, 
                       default="/home/sibo/GUI-Agent-Learn-From-Error/CoMEM-Agent-Inference/data/downloaded_datasets/webvoyager_memory/Amazon/qwen2.5-vl-32b/test/success/Amazon_Amazon_2.jsonl",
                       help="Path to trajectory JSON file")
    parser.add_argument("--success", action="store_true", default=True,
                       help="Is this a success trajectory?")
    parser.add_argument("--failure", dest="success", action="store_false",
                       help="Is this a failure trajectory?")
    args = parser.parse_args()
    
    if not os.path.exists(args.trajectory):
        print(f"✗ Trajectory file not found: {args.trajectory}")
        sys.exit(1)
    
    result = test_stage1_visual(args.trajectory, is_success=args.success)
    
    if result:
        print("\n" + "="*80)
        print("✓ Stage 1 VISUAL test completed successfully!")
        print(f"✓ Identified {len(result)} key step(s)")
        print("="*80)
        print("\nNext: Run Stage 3 to extract key takeaways from these steps")
    else:
        print("\n✗ Stage 1 VISUAL test failed")
        sys.exit(1)

