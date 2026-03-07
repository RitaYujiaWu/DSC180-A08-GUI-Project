#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

# Ensure repository root is on sys.path so hybrid_memory imports resolve
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hybrid_memory.pipeline import build_memory_index
from hybrid_memory.encoder import MemoryEncoder

DEFAULT_PROMPT_DIR = str((ROOT_DIR / "hybrid_memory" / "prompts").resolve())

def main():
    parser = argparse.ArgumentParser(description="Build Hybrid Phase Index")
    parser.add_argument("--input_glob", type=str, required=True, help="Glob for trajectory JSON/JSONL files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write index files")
    # Domain is now inferred from VLM, this arg is removed.
    # parser.add_argument("--domain", type=str, default="shopping") 
    
    # CLIP model name (default matches what analysis_tools.py uses)
    parser.add_argument("--phase_encoder_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt_dir", type=str, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--vlm_base_url", type=str, default="http://localhost:8000/v1",
                        help="VLM base URL (default: http://localhost:8000/v1)")
    parser.add_argument(
        "--vlm_api_key",
        type=str,
        default=None,
        help=(
            "API key for the OpenAI-compatible client. "
            "Required unless OPENAI_API_KEY is set. "
            "For vLLM openai.api_server, any non-empty value works (e.g. 'EMPTY')."
        ),
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help=(
            "Dataset root directory used for per-domain sampling, e.g. "
            "'CoMEM-Agent-Inference/data/downloaded_datasets/webvoyager_memory'. "
            "Required if any --sample_*_per_domain is set."
        ),
    )
    parser.add_argument(
        "--sample_success_per_domain",
        type=int,
        default=None,
        help="Randomly sample this many success trajectories per domain (requires --dataset_root).",
    )
    parser.add_argument(
        "--sample_failure_per_domain",
        type=int,
        default=None,
        help=(
            "Randomly sample this many failure trajectories per domain (requires --dataset_root). "
            "Failure trajectories are represented by positive/negative parts; we sample the positive-part files."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling (only used when sampling is enabled).",
    )
    parser.add_argument(
        "--skip_domains",
        nargs="*",
        default=[],
        help=(
            "Domain folder names to skip during per-domain sampling "
            "(e.g. Booking Google_Flights). Only applies when sampling is enabled."
        ),
    )
    args = parser.parse_args()

    # Support recursive globs like **/*.json
    files: List[str] = sorted(glob.glob(args.input_glob, recursive=True))
    if not files:
        print(f"No files found for glob: {args.input_glob}")
        return

    # Optional: per-domain sampling (domain == top-level folder under dataset_root, e.g. Allrecipes/Amazon/...)
    if args.sample_success_per_domain is not None or args.sample_failure_per_domain is not None:
        if args.sample_success_per_domain is None or args.sample_failure_per_domain is None:
            raise ValueError("Both --sample_success_per_domain and --sample_failure_per_domain must be set together.")
        if args.sample_success_per_domain <= 0 or args.sample_failure_per_domain <= 0:
            raise ValueError("--sample_*_per_domain must be positive integers.")
        if not args.dataset_root:
            raise ValueError("--dataset_root is required when using --sample_*_per_domain.")

        dataset_root = os.path.abspath(args.dataset_root)
        if not os.path.isdir(dataset_root):
            raise NotADirectoryError(f"dataset_root is not a directory: {dataset_root}")

        rng = random.Random(int(args.seed))
        pos_token = f"{os.sep}positive{os.sep}"
        neg_token = f"{os.sep}negative{os.sep}"
        suc_token = f"{os.sep}success{os.sep}"

        # domain -> files
        domain_success: Dict[str, List[str]] = {}
        domain_failure_pos: Dict[str, List[str]] = {}

        seen_failure_pos: set[str] = set()
        skipped_unpaired_failure: List[str] = []
        for fp in files:
            fp_abs = os.path.abspath(fp)
            if not fp_abs.startswith(dataset_root + os.sep):
                raise ValueError(f"File is not under dataset_root: {fp_abs}")
            rel = os.path.relpath(fp_abs, dataset_root)
            domain = rel.split(os.sep, 1)[0]
            if not domain:
                raise ValueError(f"Unable to infer domain folder from path: {fp_abs}")

            if suc_token in fp_abs:
                domain_success.setdefault(domain, []).append(fp_abs)
                continue

            # Failure trajectories are represented by positive/negative parts. Canonicalize
            # to the positive-part file path so we sample each pair once.
            if neg_token in fp_abs:
                fp_abs = fp_abs.replace(neg_token, pos_token)

            if pos_token in fp_abs:
                if fp_abs in seen_failure_pos:
                    continue
                neg_fp = fp_abs.replace(pos_token, neg_token)
                if not os.path.exists(neg_fp):
                    # Some datasets contain rare unpaired positive parts; these cannot be merged
                    # into a complete failure trajectory (positive+negative). Skip them.
                    skipped_unpaired_failure.append(fp_abs)
                    continue
                domain_failure_pos.setdefault(domain, []).append(fp_abs)
                seen_failure_pos.add(fp_abs)
                continue

            raise ValueError(
                "When sampling per-domain, every matched file must be under one of "
                "/success/, /positive/, or /negative/. Got: " + fp_abs
            )

        all_domains = sorted(set(domain_success.keys()) | set(domain_failure_pos.keys()))
        if not all_domains:
            raise ValueError("No domains discovered for sampling (check --input_glob / --dataset_root).")

        skip_set = set(str(d) for d in (args.skip_domains or []))
        unknown_skips = sorted(skip_set - set(all_domains))
        if unknown_skips:
            raise ValueError(f"--skip_domains contains unknown domains: {unknown_skips}")
        all_domains = [d for d in all_domains if d not in skip_set]
        if not all_domains:
            raise ValueError("All discovered domains were skipped; nothing to sample.")

        sampled: List[str] = []
        for dom in all_domains:
            suc = sorted(domain_success.get(dom, []))
            fail = sorted(domain_failure_pos.get(dom, []))
            if len(suc) < args.sample_success_per_domain:
                raise ValueError(
                    f"Domain {dom!r} has only {len(suc)} success trajectories, "
                    f"need {args.sample_success_per_domain}."
                )
            if len(fail) < args.sample_failure_per_domain:
                raise ValueError(
                    f"Domain {dom!r} has only {len(fail)} failure trajectories, "
                    f"need {args.sample_failure_per_domain}."
                )
            sampled.extend(rng.sample(suc, args.sample_success_per_domain))
            sampled.extend(rng.sample(fail, args.sample_failure_per_domain))

        files = sorted(sampled)
        print(
            f"[Sampling] domains={len(all_domains)} | "
            f"per_domain_success={args.sample_success_per_domain} | "
            f"per_domain_failure={args.sample_failure_per_domain} | "
            f"total_files={len(files)}"
        )
        if skipped_unpaired_failure:
            # Keep output short but visible.
            print(f"[Sampling] skipped_unpaired_failure_parts={len(skipped_unpaired_failure)}")

    encoder = MemoryEncoder(model_name=args.phase_encoder_model)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Call pipeline without domain_name
    build_memory_index(
        input_files=files,
        output_dir=args.output_dir,
        encoder=encoder,
        prompt_dir=args.prompt_dir,
        vlm_model=args.vlm_model,
        vlm_base_url=args.vlm_base_url,
        vlm_api_key=args.vlm_api_key,
    )
    print(f"Hybrid index written to: {args.output_dir}")


if __name__ == "__main__":
    main()
