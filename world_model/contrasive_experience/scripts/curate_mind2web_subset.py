#!/usr/bin/env python3
"""
Mind2Web Curated Subset Generator

This script analyzes Mind2Web baseline results and uses VLM to evaluate
which tasks are executable (not blocked by login, CAPTCHA, etc.).
It creates a curated subset JSON for reliable benchmarking.

Usage:
    python scripts/curate_mind2web_subset.py \
        --result_dir results/mind2web/test_domain_Info/qwen2.5-vl/20251226_175813 \
        --data_dir data/benchmarks/mind2web/test_domain_Info \
        --output curated_subsets/test_domain_Info.json \
        --vlm_url http://localhost:8000/v1

    # Or process multiple domains:
    python scripts/curate_mind2web_subset.py --process_all
"""

import argparse
import base64
import json
import os
import re
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import requests


@dataclass
class TaskResult:
    """Stores the result of a task evaluation."""
    task_id: str  # Task key like "Entertainment_1", "Info_102"
    file_name: str
    start_url: str
    intent: str
    domain: str
    result: str  # PASS, FAIL
    predicted_answer: str
    is_executable: bool
    non_executable_reason: Optional[str] = None


def extract_screenshots_base64(render_html_path: str, max_count: int = 3) -> List[str]:
    """Extract the first N base64 screenshots from render HTML.
    
    Args:
        render_html_path: Path to the render HTML file
        max_count: Maximum number of screenshots to extract (default: 3)
    
    Returns:
        List of base64-encoded screenshot strings
    """
    if not os.path.exists(render_html_path):
        return []
    
    with open(render_html_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find all base64 images
    matches = re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)', content)
    
    # Return up to max_count screenshots
    return matches[:max_count] if matches else []


def call_vlm_for_executability(
    screenshots_base64: List[str],
    intent: str,
    predicted_answer: str,  # Kept for signature compatibility but not used
    vlm_url: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
) -> Tuple[bool, str]:
    """
    Call VLM to evaluate if a task is executable based on screenshots only.
    
    Args:
        screenshots_base64: List of base64-encoded screenshots (up to 3)
        intent: The task intent
        predicted_answer: (unused) The agent's final answer - not relevant for executability
        vlm_url: VLM server URL
        model_name: VLM model name
    
    Returns:
        (is_executable, reason)
    """
    num_images = len(screenshots_base64)
    
    # Describe what each screenshot represents
    if num_images == 1:
        screenshot_desc = "Screenshot 1: Initial page after navigation"
    elif num_images == 2:
        screenshot_desc = "Screenshot 1: Initial page | Screenshot 2: After some actions"
    else:
        screenshot_desc = "Screenshot 1: Initial page | Screenshot 2: Mid-task | Screenshot 3: Later in task"
    
    prompt = f"""You are checking if a WEBSITE IS ACCESSIBLE. Default answer: EXECUTABLE (true).

Task intent: {intent}

I'm showing you {num_images} screenshot(s) from the agent's web browsing session:
{screenshot_desc}

QUESTION: Looking at the screenshots, is this website BLOCKED by a technical barrier?

BLOCKED (answer false) - ONLY these cases:
- CAPTCHA / "verify you are human" 
- Full-page login wall (no content visible at all)
- Error page: 403, 404, 500, "Access Denied"
- Cloudflare / DDoS protection page
- Blank/white page or about:blank
- Geo-restriction message
- Website failed to load

NOT BLOCKED (answer true) - website is accessible:
- Any website content visible (menus, text, images, forms)
- "No results" message (search works, just no matches)
- Cookie banner (can be dismissed)
- Wrong page (agent mistake, not website issue)
- Ads or popups (can be closed)
- "Sign in for more" but content is visible

RULE: If ANY real website content is visible → answer true.

Respond ONLY: {{"is_executable": true/false, "reason": "brief"}}
"""

    # Build message content with text and multiple images
    content_parts = [{"type": "text", "text": prompt}]
    for img_base64 in screenshots_base64:
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}"
            }
        })

    try:
        response = requests.post(
            f"{vlm_url}/chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.1
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Parse JSON response
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed.get('is_executable', False), parsed.get('reason', 'Unknown')
        
        # Fallback: check for keywords
        is_exec = 'true' in content.lower() and 'is_executable' in content.lower()
        return is_exec, content[:100]
        
    except Exception as e:
        print(f"  VLM call failed: {e}")
        # Fallback to keyword-based detection
        return fallback_executability_check(predicted_answer)


def fallback_executability_check(predicted_answer: str) -> Tuple[bool, str]:
    """Fallback executability check based on keywords in predicted answer.
    
    This is conservative - only marks non-executable for CLEAR website blocking indicators.
    Agent failures (max steps, wrong clicks) are considered EXECUTABLE.
    """
    # These indicate the WEBSITE is blocked (not agent failure)
    blocking_keywords = [
        'captcha', 'human verification', 'cloudflare', 
        '403 forbidden', '404 not found', '500 error',
        'access denied', 'about:blank', 'geo-restrict',
        'website is down', 'page is blank', 'completely empty'
    ]
    
    # These are agent failures, NOT blocking - default to executable
    # "max steps", "early stop", "couldn't find", etc. are agent issues
    
    answer_lower = predicted_answer.lower()
    for keyword in blocking_keywords:
        if keyword in answer_lower:
            return False, f"Website blocking detected: {keyword}"
    
    # Default to executable - assume agent failure, not website blocking
    return True, "Website appears accessible (no blocking indicators)"


def parse_log_file(log_path: str) -> Dict[str, Tuple[str, str, int]]:
    """
    Parse log file to extract task results.
    
    Returns:
        Dict mapping task_key (filename without .json) -> (result, predicted_answer, order_index)
        e.g., "Entertainment_1" -> ("PASS", "answer text", 0)
    """
    results = {}
    current_answer = None
    order_index = 0  # Track the order tasks were processed
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Look for predicted answer
            if '[Result] Predicted answer:' in line:
                current_answer = line.split('[Result] Predicted answer:')[1].strip()
            # Look for PASS/FAIL result
            elif '[Result] (PASS)' in line or '[Result] (FAIL)' in line:
                match = re.search(r'\[Result\] \((PASS|FAIL)\) (.+)$', line)
                if match:
                    result = match.group(1)
                    task_file = match.group(2).strip()
                    # Use filename without .json as unique key (e.g., "Entertainment_1", "Info_102")
                    filename = os.path.basename(task_file).replace('.json', '')
                    results[filename] = (result, current_answer or "", order_index)
                    order_index += 1
                    current_answer = None
    
    return results


def load_task_configs(data_dir: str) -> Dict[str, dict]:
    """Load all task configuration files from a domain directory.
    
    Returns:
        Dict mapping task_key (filename without .json) -> config
        e.g., "Entertainment_1" -> {...config...}
    """
    configs = {}
    for json_path in glob.glob(os.path.join(data_dir, "*.json")):
        with open(json_path, 'r') as f:
            config = json.load(f)
            # Use filename without .json as unique key
            filename = os.path.basename(json_path).replace('.json', '')
            config['_file_name'] = os.path.basename(json_path)
            config['_task_key'] = filename
            configs[filename] = config
    return configs


def process_domain(
    result_dir: str,
    data_dir: str,
    output_path: str,
    vlm_url: str,
    model_name: str,
    use_vlm: bool = True
) -> dict:
    """
    Process a single domain's results and create curated subset.
    
    Args:
        result_dir: Path to results directory (e.g., results/mind2web/.../20251226_175813)
        data_dir: Path to data directory (e.g., data/benchmarks/mind2web/test_domain_Info)
        output_path: Path to save the curated subset JSON
        vlm_url: VLM server URL
        model_name: VLM model name
        use_vlm: Whether to use VLM for evaluation (False = keyword-only)
    
    Returns:
        Summary statistics
    """
    domain_name = os.path.basename(data_dir)
    print(f"\n{'='*60}")
    print(f"Processing domain: {domain_name}")
    print(f"{'='*60}")
    
    # Find log file
    log_files = glob.glob(os.path.join(result_dir, "log_*.log"))
    if not log_files:
        raise FileNotFoundError(f"No log file found in {result_dir}")
    log_path = log_files[0]
    print(f"Log file: {log_path}")
    
    # Parse log results
    log_results = parse_log_file(log_path)
    print(f"Found {len(log_results)} task results in log")
    
    # Load task configs
    task_configs = load_task_configs(data_dir)
    print(f"Found {len(task_configs)} task configs in data directory")
    
    # Process each task
    executable_tasks = []
    non_executable_tasks = []
    
    # Sort by order_index to maintain original processing order
    sorted_task_keys = sorted(log_results.keys(), key=lambda k: log_results[k][2])
    
    for task_key in sorted_task_keys:
        if task_key not in task_configs:
            print(f"  Warning: Task {task_key} not found in configs")
            continue
        
        config = task_configs[task_key]
        result, predicted_answer, order_index = log_results[task_key]
        
        print(f"\nTask {task_key}: {config['intent'][:50]}...")
        
        # Render files are named by 1-based order index (render_1.html, render_2.html, ...)
        render_num = order_index + 1
        render_path = os.path.join(result_dir, f"render_{render_num}.html")
        screenshots_base64 = extract_screenshots_base64(render_path, max_count=3)
        
        # Evaluate executability
        if use_vlm and screenshots_base64:
            print(f"  Using {len(screenshots_base64)} screenshot(s) for VLM evaluation")
            is_executable, reason = call_vlm_for_executability(
                screenshots_base64,
                config['intent'],
                predicted_answer,
                vlm_url,
                model_name
            )
        else:
            is_executable, reason = fallback_executability_check(predicted_answer)
        
        task_result = TaskResult(
            task_id=task_key,  # Now using string key like "Entertainment_1"
            file_name=config['_file_name'],
            start_url=config.get('start_url', ''),
            intent=config.get('intent', ''),
            domain=domain_name,
            result=result,
            predicted_answer=predicted_answer,
            is_executable=is_executable,
            non_executable_reason=reason if not is_executable else None
        )
        
        if is_executable:
            executable_tasks.append(task_result)
            print(f"  ✅ EXECUTABLE (result: {result})")
        else:
            non_executable_tasks.append(task_result)
            print(f"  ❌ NOT EXECUTABLE: {reason}")
    
    # Create output
    output = {
        "dataset_name": "mind2web_executable_subset",
        "domain": domain_name,
        "creation_date": datetime.now().isoformat(),
        "source_result_dir": result_dir,
        "source_data_dir": data_dir,
        "statistics": {
            "total_tested": len(log_results),
            "executable_count": len(executable_tasks),
            "non_executable_count": len(non_executable_tasks),
            "executable_ratio": len(executable_tasks) / len(log_results) if log_results else 0
        },
        "executable_tasks": [asdict(t) for t in executable_tasks],
        "non_executable_tasks": [asdict(t) for t in non_executable_tasks]
    }
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY for {domain_name}")
    print(f"{'='*60}")
    print(f"Total tested: {len(log_results)}")
    print(f"Executable: {len(executable_tasks)} ({len(executable_tasks)/len(log_results)*100:.1f}%)")
    print(f"Non-executable: {len(non_executable_tasks)} ({len(non_executable_tasks)/len(log_results)*100:.1f}%)")
    print(f"Output saved to: {output_path}")
    
    return output['statistics']


def create_combined_subset(domain_outputs: List[str], combined_output: str):
    """Combine multiple domain subsets into a single file."""
    combined = {
        "dataset_name": "mind2web_executable_subset_combined",
        "creation_date": datetime.now().isoformat(),
        "domains": {},
        "all_executable_tasks": []
    }
    
    total_tested = 0
    total_executable = 0
    
    for domain_path in domain_outputs:
        if not os.path.exists(domain_path):
            continue
        with open(domain_path, 'r') as f:
            data = json.load(f)
        
        domain = data['domain']
        combined['domains'][domain] = {
            "statistics": data['statistics'],
            "executable_tasks": data['executable_tasks']
        }
        combined['all_executable_tasks'].extend(data['executable_tasks'])
        total_tested += data['statistics']['total_tested']
        total_executable += data['statistics']['executable_count']
    
    combined['total_statistics'] = {
        "total_tested": total_tested,
        "total_executable": total_executable,
        "overall_executable_ratio": total_executable / total_tested if total_tested else 0
    }
    
    with open(combined_output, 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"\nCombined subset saved to: {combined_output}")
    print(f"Total executable tasks: {total_executable}/{total_tested}")


def main():
    parser = argparse.ArgumentParser(description="Curate Mind2Web executable subset")
    parser.add_argument("--result_dir", type=str, help="Path to results directory")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--vlm_url", type=str, default="http://localhost:8000/v1", help="VLM server URL")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="VLM model name")
    parser.add_argument("--no_vlm", action="store_true", help="Use keyword-only detection (no VLM)")
    parser.add_argument("--process_all", action="store_true", help="Process all domains with latest results")
    
    args = parser.parse_args()
    
    if args.process_all:
        # Find latest results for each domain
        # Check both old location (results/mind2web/) and new location (results/mind2web_baseline/mind2web/)
        domains = ["test_domain_Info", "test_domain_Service", "test_website"]
        domain_outputs = []
        
        for domain in domains:
            # Try new location first (mind2web_baseline), then old location
            result_patterns = [
                f"results/mind2web_baseline/mind2web/{domain}/*/[0-9]*_[0-9]*",
                f"results/mind2web/{domain}/*/[0-9]*_[0-9]*"
            ]
            result_dirs = []
            for pattern in result_patterns:
                result_dirs.extend(sorted(glob.glob(pattern)))
            result_dirs = sorted(result_dirs)  # Sort all found directories
            if not result_dirs:
                print(f"No results found for {domain}")
                continue
            
            latest_result = result_dirs[-1]  # Most recent
            data_dir = f"data/benchmarks/mind2web/{domain}"
            output_path = f"curated_subsets/{domain}.json"
            
            process_domain(
                latest_result, data_dir, output_path,
                args.vlm_url, args.model_name, not args.no_vlm
            )
            domain_outputs.append(output_path)
        
        # Create combined subset
        create_combined_subset(domain_outputs, "curated_subsets/mind2web_combined.json")
    else:
        if not args.result_dir or not args.data_dir or not args.output:
            parser.error("--result_dir, --data_dir, and --output are required (or use --process_all)")
        
        process_domain(
            args.result_dir, args.data_dir, args.output,
            args.vlm_url, args.model_name, not args.no_vlm
        )


if __name__ == "__main__":
    main()

