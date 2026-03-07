#!/usr/bin/env python3
"""Validation script for multimodal reasoning bank implementation."""
import json
import os
import base64
from io import BytesIO
from PIL import Image

def validate_trajectory_structure(traj_file):
    """Validate that trajectory has required structure."""
    print(f"\n{'='*60}")
    print(f"Validating: {os.path.basename(traj_file)}")
    print('='*60)
    
    with open(traj_file, 'r') as f:
        obj = json.load(f)
    
    # Check required fields
    assert 'task_description' in obj, "Missing task_description"
    assert 'rounds' in obj, "Missing rounds"
    
    task = obj['task_description']
    rounds = obj['rounds']
    
    print(f"✓ Task: {task[:80]}...")
    print(f"✓ Total rounds: {len(rounds)}")
    
    # Check round structure
    screenshots_found = 0
    actions_found = 0
    
    for i, r in enumerate(rounds):
        # Check for screenshot
        has_screenshot = False
        for msg in r.get('messages', []):
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        has_screenshot = True
                        screenshots_found += 1
                        break
        
        # Check for action in response using the actual parser
        response = r.get('response', '')
        # Simple check for JSON structure
        import re
        response_clean = response.replace('```json', '').replace('```', '')
        if re.search(r'\{\s*"name"\s*:', response_clean):
            actions_found += 1
    
    print(f"✓ Screenshots found: {screenshots_found}/{len(rounds)}")
    print(f"✓ Actions found: {actions_found}/{len(rounds)}")
    
    return screenshots_found > 0 and actions_found > 0

def validate_image_saving(traj_file, output_dir="/tmp/test_reasoning_bank"):
    """Test image extraction and saving."""
    print(f"\n{'='*60}")
    print("Testing image extraction and saving")
    print('='*60)
    
    with open(traj_file, 'r') as f:
        obj = json.load(f)
    
    rounds = obj['rounds']
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    for i, r in enumerate(rounds[:3]):  # Test first 3
        for msg in r.get('messages', []):
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        screenshot = item['image_url']['url']
                        
                        # Save image
                        if screenshot.startswith('data:image'):
                            base64_str = screenshot.split(',', 1)[1]
                        else:
                            base64_str = screenshot
                        
                        img_data = base64.b64decode(base64_str)
                        img = Image.open(BytesIO(img_data))
                        
                        # Downscale
                        max_width = 768
                        if img.width > max_width:
                            ratio = max_width / img.width
                            new_height = int(img.height * ratio)
                            img = img.resize((max_width, new_height), Image.LANCZOS)
                        
                        # Save as JPEG
                        if img.mode in ('RGBA', 'LA', 'P'):
                            img = img.convert('RGB')
                        
                        output_path = os.path.join(output_dir, f"step_{i:02d}.jpg")
                        img.save(output_path, 'JPEG', quality=80)
                        
                        size_kb = os.path.getsize(output_path) / 1024
                        print(f"✓ Step {i}: saved to {output_path} ({size_kb:.1f} KB)")
                        saved_count += 1
                        break
    
    print(f"\n✓ Saved {saved_count} images")
    return saved_count > 0

def validate_prompts():
    """Check that prompt templates exist."""
    print(f"\n{'='*60}")
    print("Validating prompt templates")
    print('='*60)
    
    prompts_dir = "/home/sibo/GUI-Agent-Learn-From-Error/CoMEM-Agent-Inference/agent/prompts"
    
    required_prompts = [
        "reasoning_bank_mm_identify_steps.md",
        "reasoning_bank_mm_extract.md"
    ]
    
    for prompt_file in required_prompts:
        path = os.path.join(prompts_dir, prompt_file)
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read()
            print(f"✓ {prompt_file} ({len(content)} chars)")
        else:
            print(f"✗ {prompt_file} NOT FOUND")
            return False
    
    return True

def main():
    print("Multimodal Reasoning Bank Validation")
    print("="*60)
    
    # Test trajectory file
    test_file = "/home/sibo/GUI-Agent-Learn-From-Error/CoMEM-Agent-Inference/data/downloaded_datasets/webvoyager_memory/Amazon/qwen2.5-vl-32b/test/success/Amazon_Amazon_2.jsonl"
    
    if not os.path.exists(test_file):
        print(f"✗ Test file not found: {test_file}")
        print("Please update the path in this script.")
        return False
    
    # Run validations
    try:
        if not validate_trajectory_structure(test_file):
            print("\n✗ Trajectory structure validation failed")
            return False
        
        if not validate_image_saving(test_file):
            print("\n✗ Image saving validation failed")
            return False
        
        if not validate_prompts():
            print("\n✗ Prompt validation failed")
            return False
        
        print("\n" + "="*60)
        print("✓ All validations passed!")
        print("="*60)
        print("\nNext steps:")
        print("1. Ensure VLM server is running (e.g., qwen2.5-vl on localhost:8000)")
        print("2. Run: python scripts/build_reasoning_bank.py --input_glob '...' --multimodal")
        print("3. Check output in memory/reasoning_bank_mm.jsonl")
        print("4. Run agent with --reasoning_bank_multimodal True")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

