#!/usr/bin/env python3
import os
import dotenv
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from openai import OpenAI

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
# Use OpenRouter if key is provided, otherwise use OpenAI
if OPENROUTER_API_KEY:
    CLIENT = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    print("Using OpenRouter API")
elif OPENAI_API_KEY:
    CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    print("Using OpenAI API")
else:
    CLIENT = OpenAI()
    print("Warning: No API key found, using default OpenAI client")
EXAMPLES_ROOT = os.getenv("EXAMPLES_ROOT", "OSWorld/evaluation_examples/examples")


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def _load_task_instruction_from_examples(
    traj_path: str,
    base_dir: str,
    examples_root: str = EXAMPLES_ROOT
) -> Optional[str]:
    """
    Infer <type> and <id> from the trajectory path and load the task instruction JSON.
    Expected layout:
      - Trajectory: <base_dir>/<type>/<id>/traj.jsonl
      - Instruction JSON: <examples_root>/<type>/<id>.json
    """
    try:
        traj_dir = Path(traj_path).parent
        instance_id = traj_dir.name
        task_type = traj_dir.parent.name
        json_path = Path(examples_root) / task_type / f"{instance_id}.json"
        if not json_path.exists():
            # Fallback: try to derive from base_dir-relative path if above fails
            rel_parts = Path(traj_path).relative_to(base_dir).parts
            if len(rel_parts) >= 2:
                task_type = rel_parts[0]
                instance_id = rel_parts[1]
                json_path = Path(examples_root) / task_type / f"{instance_id}.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
            for key in ["instruction"]:
                if isinstance(data.get(key), str) and data.get(key).strip():
                    return data.get(key).strip()
        else:
            print(f"Warning: Instruction JSON not found for type={task_type}, id={instance_id}: {json_path}")
    except Exception as e:
        print(f"Warning: Could not load task instruction for {traj_path}: {e}")
    return None


def get_reward_from_llm(
    api_key: str,
    model: str,
    before_image_path: Optional[str],
    after_image_path: Optional[str],
    step_num: int,
    action_text: str,
    is_done: bool,
    total_steps: int,
    trajectory_context: str,
    task_instruction: Optional[str]
) -> Tuple[float, str]:
    """
    Call an annotator model via OpenAI to get reasoning and reward score for a step.
    
    Args:
        api_key: OpenAI API key
        model: OpenAI model id (e.g., 'gpt-4o-mini')
        before_image_path: Path to screenshot image BEFORE the action (optional)
        after_image_path: Path to screenshot image AFTER the action
        step_num: Current step number
        action_text: Human-readable action (step['response'])
        is_done: Whether task is completed
        total_steps: Total number of steps in trajectory
        trajectory_context: Full trajectory (human-readable) for context
        task_instruction: Task goal/instruction text loaded from examples JSON
    
    Returns:
        (reward, reason) where reward is in [0.0, 1.0]
    """
    # Encode images if provided
    base64_before = encode_image(before_image_path) if before_image_path and os.path.exists(before_image_path) else None
    base64_after = encode_image(after_image_path) if after_image_path and os.path.exists(after_image_path) else None
    
    # Construct prompt
    prompt = f"""You are evaluating a single step in a GUI automation trajectory.

Context:
- Step {step_num} of {total_steps}
- This is {'the FINAL step' if is_done else 'an intermediate step'}

Task Goal:
{task_instruction or 'Not provided'}

Full Trajectory (human-readable actions for all steps):
{trajectory_context}

Current Step Action (human-readable): {action_text}

Task: Provide a brief reasoning and a step-wise reward score for this step. You should focus on this current step and its effect to the task goal.

Scoring Guidelines:
- Score range: 0.0 to 1.0
- 1.0: Perfect action that directly contributes to task completion or is the successful completion itself
- 0.8-0.9: Very good action that makes clear progress toward the goal
- 0.6-0.7: Reasonable action that may help but is not optimal
- 0.4-0.5: Action that is somewhat relevant but may not be the best choice
- 0.2-0.3: Action that is questionable or may not help much
- 0.0-0.1: Poor action, mistake, or action that hinders progress

For the final step (when done=True):
- If the task appears successfully completed: 1.0
- If the task is incomplete or failed: 0.0-0.3

For intermediate steps:
- Evaluate based on whether the action makes logical progress toward what appears to be the goal
- Consider if the action is appropriate given the screenshot state
- Consider if the coordinates/action make sense for the visible UI elements

Look at the BEFORE and AFTER screenshots carefully and evaluate:
1. Does the action make sense given the current screen state?
2. Does the state transition appears to progress toward a goal?
3. Is the action executed correctly (right element, right location)?

Respond ONLY with a JSON object with two fields in this order:
{{
  "reason": "<brief reasoning, 1-3 sentences>",
  "reward": <float between 0.0 and 1.0>
}}.    

Do not include any other text, explanation, or markdown formatting."""

    # Build messages like gpt4.py: images first, then prompt text
    def _data_url(image_path: str, b64: str) -> str:
        ext = os.path.splitext(image_path)[1].lstrip('.').lower() if image_path else 'png'
        if ext == 'jpg':
            ext = 'jpeg'
        return f"data:image/{ext};base64,{b64}"
    
    image_contents: List[Dict] = []
    if base64_before:
        image_contents.append({"type": "image_url", "image_url": {"url": _data_url(before_image_path, base64_before)}})
    if base64_after:
        image_contents.append({"type": "image_url", "image_url": {"url": _data_url(after_image_path, base64_after)}})
    
    messages = [{
        "role": "user",
        "content": image_contents + [{"type": "text", "text": prompt}]
    }]
    
    # Use OpenAI SDK similar to gpt4.py, avoid response_format to prevent 400s
    sampling_params: Dict = {}
    if any(x in model.split("-") for x in ["o1", "o3"]):
        sampling_params = {}
    else:
        sampling_params = {"max_completion_tokens": 3000} #{"temperature": 0.3, "top_p": 0.9, "max_tokens": 350}
    
    try:
        response = CLIENT.chat.completions.create(
            model=model,
            messages=messages,
            **sampling_params
        )
        content = response.choices[0].message.content.strip() if response and response.choices else ""
        
        # Parse JSON response
        # Handle cases where response might have markdown code blocks
        if content.startswith("```"):
            # Extract JSON from code block
            lines = content.split('\n')
            json_str = '\n'.join([l for l in lines if not l.startswith('```')])
        else:
            json_str = content
        
        # Try to parse as JSON
        try:
            reward_data = json.loads(json_str)
            reason_text = str(reward_data.get('reason', '')).strip()
            reward = float(reward_data.get('reward', 0.0))
            # Clamp to [0, 1]
            reward = max(0.0, min(1.0, reward))
            return reward, reason_text
        except json.JSONDecodeError:
            # Fallback: try to extract number from text
            import re
            numbers = re.findall(r'\d+\.?\d*', content)
            if numbers:
                reward = float(numbers[0])
                reward = max(0.0, min(1.0, reward))
                print(f"Warning: Could not parse JSON, extracted number: {reward}")
                return reward, ""
            else:
                print(f"Warning: Could not parse reward from response: {content}")
                return 0.0, ""
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return 0.0, ""


def process_trajectory(
    traj_path: str,
    api_key: str,
    model: str,
    base_dir: str,
    output_file: str,
    dry_run: bool = False
) -> Optional[Dict]:
    """
    Process a single trajectory file and annotate rewards.
    
    Args:
        traj_path: Path to traj.jsonl file (in --base-dir)
        api_key: OpenAI API key
        base_dir: Base directory for trajectories (to compute relative path)
        output_file: Output file path to append results to
        dry_run: If True, don't actually call API
    
    Returns:
        Dictionary with trajectory data and annotated steps, or None if failed
    """
    traj_dir = os.path.dirname(traj_path)
    
    # Compute relative path from base_dir for identification
    try:
        rel_path = os.path.relpath(traj_path, base_dir)
    except ValueError:
        rel_path = traj_path
    
    # Read all steps
    steps = []
    with open(traj_path, 'r') as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))
    
    if not steps:
        print(f"Warning: No steps found in {traj_path}")
        return None
    
    total_steps = len(steps)
    
    # Build full trajectory context using human-readable actions (response)
    trajectory_context_lines: List[str] = []
    for s in steps:
        s_num = s.get('step_num')
        s_resp = s.get('response', '')
        s_done = s.get('done', False)
        postfix = " [FINAL]" if s_done else ""
        trajectory_context_lines.append(f"Step {s_num}: {s_resp}{postfix}")
    trajectory_context = "\n".join(trajectory_context_lines)
    # Load task instruction once per trajectory
    task_instruction = _load_task_instruction_from_examples(traj_path=traj_path, base_dir=base_dir)
    if task_instruction:
        print(f"Loaded task instruction for trajectory: {task_instruction[:120]}{'...' if len(task_instruction) > 120 else ''}")
    else:
        print("Warning: No task instruction found; proceeding without explicit goal.")
    
    # Process each step
    updated_steps = []
    for idx, step in enumerate(steps):
        step_num = step['step_num']
        # AFTER image for current step
        screenshot_file_after = step.get('screenshot_file', f"step_{step_num}_{step.get('action_timestamp', '')}.png")
        screenshot_path_after = os.path.join(traj_dir, screenshot_file_after)
        # BEFORE image from previous step if available
        screenshot_path_before: Optional[str] = None
        if idx > 0:
            prev = steps[idx - 1]
            prev_step_num = prev.get('step_num')
            prev_file = prev.get('screenshot_file', f"step_{prev_step_num}_{prev.get('action_timestamp', '')}.png")
            candidate_before = os.path.join(traj_dir, prev_file)
            if os.path.exists(candidate_before):
                screenshot_path_before = candidate_before
        
        if not os.path.exists(screenshot_path_after):
            print(f"Warning: AFTER screenshot not found: {screenshot_path_after}, skipping step {step_num}")
            updated_steps.append(step)
            continue
        
        # Get reward from LLM
        if not dry_run:
            reward, reason = get_reward_from_llm(
                api_key=api_key,
                model=model,
                before_image_path=screenshot_path_before,
                after_image_path=screenshot_path_after,
                step_num=step_num,
                action_text=step.get('response', ''),  # human-readable action
                is_done=step.get('done', False),
                total_steps=total_steps,
                trajectory_context=trajectory_context,
                task_instruction=task_instruction
            )
            step['reward'] = reward
            step['reason'] = reason
            print(f"  Step {step_num}: reward = {reward:.3f}")
        else:
            print(f"  Step {step_num}: [DRY RUN - would call API]")
            # Keep original reward in dry run mode
        
        updated_steps.append(step)
    
    # Create trajectory record
    trajectory_data = {
        'trajectory_path': rel_path,
        'trajectory_dir': traj_dir,
        'task_instruction': task_instruction,
        'total_steps': total_steps,
        'steps': updated_steps
    }
    
    return trajectory_data


def _process_trajectory_wrapper(args_tuple):
    """
    Wrapper for multiprocessing that calls process_trajectory and captures errors.
    Returns (traj_path, result_dict_or_None, error_str_or_None).
    """
    traj_path, api_key, model, base_dir, output_file, dry_run = args_tuple
    try:
        result = process_trajectory(
            traj_path=traj_path,
            api_key=api_key,
            model=model,
            base_dir=base_dir,
            output_file=output_file,  # unused inside, kept for signature stability
            dry_run=dry_run
        )
        return traj_path, result, None
    except Exception as e:
        import traceback
        return traj_path, None, f"{e}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(
        description="Annotate step-wise rewards for trajectories using an annotator model via OpenAI"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='openai/gpt-5-mini',  # Default for OpenRouter; use 'gpt-5-mini' for OpenAI
        help='Annotator model ID (must support images). For OpenRouter, use format like "openai/gpt-4o-mini"'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        help='Base directory containing trajectories (read-only, original files)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file path for annotated trajectories'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode: don\'t actually call API'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=16,
        help='Number of parallel workers (trajectories processed with imap_unordered)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of trajectories to process (for testing)'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing output file (default: overwrite)'
    )
    parser.add_argument(
        '--skip',
        type=int,
        default=0,
        help='Skip first N trajectories (useful when resuming processing)'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Initialize or clear output file
    if not args.append and os.path.exists(args.output_file):
        os.remove(args.output_file)
        print(f"Cleared existing output file: {args.output_file}")
    elif args.append:
        print(f"Appending to existing file: {args.output_file}")
    
    # Find all traj.jsonl files
    base_path = Path(args.base_dir)
    traj_files = list(base_path.rglob('traj.jsonl'))
    
    print(f"Found {len(traj_files)} trajectory files in {args.base_dir}")
    print(f"Output will be written to: {args.output_file}")
    
    # Skip first N trajectories if requested
    if args.skip > 0:
        skipped = traj_files[:args.skip]
        traj_files = traj_files[args.skip:]
        print(f"Skipping first {args.skip} trajectories")
        print(f"Remaining trajectories to process: {len(traj_files)}")
    
    if args.limit:
        traj_files = traj_files[:args.limit]
        print(f"Processing first {len(traj_files)} trajectories (limited)")
    
    # Process trajectories (parallel via imap_unordered across trajectory level)
    success_count = 0
    total = len(traj_files)
    if total == 0:
        print("No trajectories to process.")
        return
    # open output file in main process for serialized writes
    with open(args.output_file, 'a') as out_f:
        if args.workers and args.workers > 1:
            print(f"Running in parallel with {args.workers} workers...")
            # Use OpenRouter key if available, otherwise OpenAI key
            api_key = OPENROUTER_API_KEY if OPENROUTER_API_KEY else OPENAI_API_KEY
            work_items = [
                (str(p), api_key, args.model, args.base_dir, args.output_file, args.dry_run)
                for p in traj_files
            ]
            with Pool(processes=args.workers) as pool:
                for traj_path, result, err in tqdm(
                    pool.imap_unordered(_process_trajectory_wrapper, work_items),
                    total=total,
                    desc="Processing trajectories"
                ):
                    if err:
                        print(f"\nError processing {traj_path}: {err}")
                        continue
                    if result:
                        out_f.write(json.dumps(result) + '\n')
                        out_f.flush()
                        success_count += 1
        else:
            print("Running sequentially (workers=1)...")
            for traj_path in tqdm(traj_files, desc="Processing trajectories"):
                traj_path_str = str(traj_path)
                try:
                    # Use OpenRouter key if available, otherwise OpenAI key
                    api_key = OPENROUTER_API_KEY if OPENROUTER_API_KEY else OPENAI_API_KEY
                    result = process_trajectory(
                        traj_path_str, 
                        api_key,
                        args.model,
                        args.base_dir,
                        args.output_file,
                        args.dry_run
                    )
                    if result:
                        out_f.write(json.dumps(result) + '\n')
                        out_f.flush()
                        success_count += 1
                except Exception as e:
                    print(f"Error processing {traj_path_str}: {e}")
                    import traceback
                    traceback.print_exc()
    
    print(f"\nCompleted: {success_count}/{len(traj_files)} trajectories processed successfully")
    print(f"Annotated trajectories saved to: {args.output_file}")


if __name__ == "__main__":
    main()

