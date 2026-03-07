#!/usr/bin/env python3
"""
Compute success rates from evaluation results.
This script parses HTML render files to extract scores and compute aggregate statistics.
"""
import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import glob


def parse_render_file_for_score(render_path: str) -> Tuple[float, str]:
    """
    Parse HTML render file to extract score and task ID.

    Args:
        render_path: Path to render HTML file

    Returns:
        Tuple of (score, task_id)
    """
    try:
        with open(render_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

            # Look for score in HTML div structure: <div class='final_score'><pre>X.X</pre></div>
            # 0.0 = failed, 1.0 = success
            score_match = re.search(r"<div class='final_score'><pre>([0-9.]+)</pre></div>", content)
            if score_match:
                score = float(score_match.group(1))
            else:
                # Cannot determine score
                return None, None

            # Extract task ID from filename
            filename = os.path.basename(render_path)
            # Pattern: render_<task_id>.html or render_<task_id>_attempt<N>.html
            task_id_match = re.search(r'render_(\d+)(?:_attempt\d+)?\.html', filename)
            if task_id_match:
                task_id = task_id_match.group(1)
            else:
                task_id = filename.replace('.html', '').replace('render_', '')

            return score, task_id
    except Exception as e:
        print(f"Error parsing {render_path}: {e}")
        return None, None


def compute_success_rate(result_dir: str, verbose: bool = False) -> Dict:
    """
    Compute success rate from result directory.

    Args:
        result_dir: Path to results directory
        verbose: Print detailed information

    Returns:
        Dictionary with success rate statistics
    """
    result_dir = Path(result_dir)

    # Find all render HTML files (excluding attempt-specific files for final score)
    render_files = list(result_dir.glob("render_*.html"))

    # Filter to get only the final render files (without _attempt suffix)
    final_render_files = [f for f in render_files if '_attempt' not in f.name]

    if not final_render_files:
        # If no final files, use all render files (might be older format)
        final_render_files = render_files

    if not final_render_files:
        print(f"No render files found in {result_dir}")
        return {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "success_rate": 0.0,
            "scores": {}
        }

    scores = {}
    successful = 0
    failed = 0

    for render_file in sorted(final_render_files):
        score, task_id = parse_render_file_for_score(str(render_file))

        if score is not None:
            scores[task_id] = score
            if score >= 0.5:  # Consider success if score >= 0.5
                successful += 1
            else:
                failed += 1

            if verbose:
                status = "PASS" if score >= 0.5 else "FAIL"
                print(f"Task {task_id}: {status} (score: {score})")

    total = successful + failed
    success_rate = (successful / total * 100) if total > 0 else 0.0

    stats = {
        "total_tasks": total,
        "successful_tasks": successful,
        "failed_tasks": failed,
        "success_rate": success_rate,
        "scores": scores
    }

    return stats


def save_summary(stats: Dict, result_dir: str):
    """Save summary statistics to JSON file."""
    result_dir = Path(result_dir)
    summary_file = result_dir / "score_summary.json"

    with open(summary_file, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"\nSummary saved to: {summary_file}")


def print_summary(stats: Dict, result_dir: str):
    """Print formatted summary."""
    print("\n" + "=" * 60)
    print(f"Evaluation Results: {result_dir}")
    print("=" * 60)
    print(f"Total Tasks:       {stats['total_tasks']}")
    print(f"Successful:        {stats['successful_tasks']}")
    print(f"Failed:            {stats['failed_tasks']}")
    print(f"Success Rate:      {stats['success_rate']:.2f}%")
    print("=" * 60)


def find_recent_results(base_dir: str = "results", eval_type: str = None,
                        domain: str = None, model: str = None, n: int = 5) -> List[str]:
    """
    Find recent result directories.

    Args:
        base_dir: Base results directory
        eval_type: Evaluation type filter (mmina, webvoyager, mind2web)
        domain: Domain filter
        model: Model filter
        n: Number of recent results to return

    Returns:
        List of result directory paths
    """
    base_path = Path(base_dir)

    # Build search pattern
    pattern_parts = [str(base_path)]
    if eval_type:
        pattern_parts.append(eval_type)
    else:
        pattern_parts.append("*")

    if domain:
        pattern_parts.append(domain)
    else:
        pattern_parts.append("*")

    if model:
        pattern_parts.append(model)
    else:
        pattern_parts.append("*")

    pattern_parts.append("*")  # datetime folders

    search_pattern = "/".join(pattern_parts)
    all_result_dirs = glob.glob(search_pattern)

    # Filter to only directories that contain render files
    result_dirs_with_files = []
    for result_dir in all_result_dirs:
        if list(Path(result_dir).glob("render_*.html")):
            result_dirs_with_files.append(result_dir)

    # Sort by modification time (most recent first)
    result_dirs_with_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    return result_dirs_with_files[:n]


def main():
    parser = argparse.ArgumentParser(
        description="Compute success rates from evaluation results"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        help="Path to specific result directory"
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        choices=["mmina", "webvoyager", "mind2web"],
        help="Filter by evaluation type"
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Filter by domain"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Filter by model"
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=1,
        help="Process N most recent result directories (default: 1)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save summary to JSON file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed per-task results"
    )

    args = parser.parse_args()

    if args.result_dir:
        # Process specific directory
        result_dirs = [args.result_dir]
    else:
        # Find recent results
        result_dirs = find_recent_results(
            eval_type=args.eval_type,
            domain=args.domain,
            model=args.model,
            n=args.recent
        )

        if not result_dirs:
            print("No result directories found matching the criteria.")
            return

    # Process each result directory
    for result_dir in result_dirs:
        stats = compute_success_rate(result_dir, verbose=args.verbose)

        if stats["total_tasks"] > 0:
            print_summary(stats, result_dir)

            if args.save:
                save_summary(stats, result_dir)
        else:
            print(f"\nNo tasks found in {result_dir}")


if __name__ == "__main__":
    main()
