#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure CoMEM-Agent-Inference/ is on sys.path so hybrid_memory imports resolve
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hybrid_memory.encoder import MemoryEncoder
from hybrid_memory.retriever import HybridRetriever


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _infer_encoder_name_from_domains(index_dir: str) -> str:
    domains_path = os.path.join(index_dir, "domains.json")
    if not os.path.exists(domains_path):
        raise FileNotFoundError(f"domains.json not found at: {domains_path}")
    domains = _load_json(domains_path)
    if not isinstance(domains, dict) or not domains:
        raise ValueError(f"domains.json must be a non-empty dict: {domains_path}")
    first = next(iter(domains.values()))
    if not isinstance(first, dict):
        raise ValueError("domains.json values must be objects")
    enc = first.get("encoder_name")
    if not isinstance(enc, str) or not enc.strip():
        raise ValueError("domains.json missing a valid encoder_name")
    return enc.strip()


def _extract_round_images(round_obj: Dict[str, Any]) -> List[str]:
    imgs: List[str] = []
    for msg in round_obj.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url")
                    if isinstance(url, str) and url.startswith("data:image"):
                        imgs.append(url)
    return imgs


def _extract_step_summary(round_obj: Dict[str, Any]) -> str:
    texts: List[str] = []
    for msg in round_obj.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    txt = str(item.get("text", "")).strip()
                    if txt:
                        texts.append(txt)
    if not texts:
        return ""
    return texts[0][:2000]


def _load_and_maybe_merge_failure(file_path: str) -> Dict[str, Any]:
    """
    Load a trajectory JSON file, and if it is a failure split into positive/negative parts,
    merge them into a single trajectory (positive rounds then negative rounds), matching
    the build-time behavior in hybrid_memory/constructor.py.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")

    def load_one(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"Trajectory JSON must be an object: {path}")
        return obj

    data = load_one(file_path)

    pos_token = f"{os.sep}positive{os.sep}"
    neg_token = f"{os.sep}negative{os.sep}"
    if pos_token not in file_path and neg_token not in file_path:
        rounds = data.get("rounds", [])
        if not isinstance(rounds, list) or not rounds:
            raise ValueError(f"Trajectory has no rounds: {file_path}")
        return data

    if neg_token in file_path:
        pos_path = file_path.replace(neg_token, pos_token)
        neg_path = file_path
    else:
        pos_path = file_path
        neg_path = file_path.replace(pos_token, neg_token)

    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"Positive-part file not found: {pos_path}")
    if not os.path.exists(neg_path):
        raise FileNotFoundError(f"Negative-part file not found: {neg_path}")

    pos_data = data if file_path == pos_path else load_one(pos_path)
    neg_data = data if file_path == neg_path else load_one(neg_path)

    pos_cid = pos_data.get("conversation_id")
    neg_cid = neg_data.get("conversation_id")
    if not isinstance(pos_cid, str) or not pos_cid.strip():
        raise ValueError(f"Missing conversation_id in positive-part: {pos_path}")
    if pos_cid != neg_cid:
        raise ValueError(
            "Mismatched conversation_id between positive/negative parts: "
            f"{pos_cid!r} vs {neg_cid!r}"
        )

    if pos_data.get("split_type") != "positive_part":
        raise ValueError(f"Expected split_type='positive_part' in {pos_path}")
    if neg_data.get("split_type") != "negative_part":
        raise ValueError(f"Expected split_type='negative_part' in {neg_path}")

    pos_task = pos_data.get("task_description")
    neg_task = neg_data.get("task_description")
    if pos_task != neg_task:
        raise ValueError("Mismatched task_description between positive/negative parts")

    pos_eval = pos_data.get("evaluation", {}).get("evaluation", {})
    neg_eval = neg_data.get("evaluation", {}).get("evaluation", {})
    if bool(pos_eval.get("Correctness", False)) or bool(neg_eval.get("Correctness", False)):
        raise ValueError("positive/negative parts must be failure trajectories (Correctness=false)")

    pos_rounds = pos_data.get("rounds")
    neg_rounds = neg_data.get("rounds")
    if not isinstance(pos_rounds, list) or not isinstance(neg_rounds, list):
        raise ValueError("Both positive and negative parts must contain a 'rounds' list")

    merged_rounds = pos_rounds + neg_rounds
    if not merged_rounds:
        raise ValueError("Merged failure trajectory has no rounds")

    merged = dict(pos_data)
    merged["split_type"] = "merged_failure"
    merged["rounds"] = merged_rounds
    merged["total_rounds"] = len(merged_rounds)
    merged["conversation_start"] = pos_data.get("conversation_start")
    merged["conversation_end"] = neg_data.get("conversation_end")
    if not merged.get("conversation_start"):
        raise ValueError("Missing conversation_start in merged failure trajectory")
    if not merged.get("conversation_end"):
        raise ValueError("Missing conversation_end in merged failure trajectory")

    return merged


def _mrr(rank_1_index: Optional[int]) -> float:
    if rank_1_index is None:
        return 0.0
    return 1.0 / float(rank_1_index + 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate hybrid hierarchical retrieval vs phase-only retrieval")
    p.add_argument("--index_dir", type=str, required=True, help="Path to hybrid_index/* directory")
    p.add_argument("--k", type=int, default=3, help="Top-K phases to retrieve")
    p.add_argument(
        "--oracle_domain",
        action="store_true",
        default=False,
        help="If set, pass the ground-truth domain label into hierarchical retriever (disables domain inference).",
    )
    p.add_argument(
        "--max_queries",
        type=int,
        default=200,
        help="Max number of trajectory-step queries to evaluate (random sample).",
    )
    p.add_argument(
        "--queries_per_trajectory",
        type=int,
        default=2,
        help="How many random step queries to sample per trajectory (until max_queries is reached).",
    )
    p.add_argument(
        "--query_step",
        type=int,
        default=None,
        help=(
            "If set, evaluate queries at a fixed step index within each trajectory "
            "(e.g., 0 for the first step). When provided, only ONE query is evaluated "
            "per trajectory (queries_per_trajectory is ignored for that trajectory). "
            "If omitted, a random step is sampled per query."
        ),
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling queries")
    p.add_argument(
        "--encoder_model",
        type=str,
        default=None,
        help="CLIP model name for query encoding. If omitted, inferred from domains.json encoder_name.",
    )
    p.add_argument("--out_json", type=str, default=None, help="Optional path to write detailed results as JSON")
    args = p.parse_args()

    index_dir = os.path.abspath(args.index_dir)
    if not os.path.isdir(index_dir):
        raise NotADirectoryError(f"index_dir is not a directory: {index_dir}")
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.max_queries <= 0:
        raise ValueError("--max_queries must be positive")
    if args.queries_per_trajectory <= 0:
        raise ValueError("--queries_per_trajectory must be positive")
    if args.query_step is not None and int(args.query_step) < 0:
        raise ValueError("--query_step must be >= 0")

    encoder_name = args.encoder_model or _infer_encoder_name_from_domains(index_dir)
    encoder = MemoryEncoder(model_name=encoder_name)
    retriever = HybridRetriever(index_dir=index_dir, encoder=encoder)
    if retriever.phase_index is None:
        raise ValueError("Phase index not found/loaded (phases/vectors.* missing?)")

    trajectories_path = os.path.join(index_dir, "trajectories.json")
    trajectories = _load_json(trajectories_path)
    if not isinstance(trajectories, dict) or not trajectories:
        raise ValueError(f"trajectories.json must be a non-empty dict: {trajectories_path}")

    # Phase metadata is used to compute domain/trajectory purity of retrieved phases.
    phase_meta = retriever.phase_index.meta
    if not isinstance(phase_meta, dict) or not phase_meta:
        raise ValueError("phase_index.meta is empty; cannot evaluate")

    rng = random.Random(int(args.seed))
    traj_ids = sorted(trajectories.keys())
    rng.shuffle(traj_ids)

    # Cache merged trajectory data to avoid repeatedly loading JSON from disk
    traj_data_cache: Dict[str, Dict[str, Any]] = {}

    # Aggregate metrics
    metrics = {
        "phase_only": {
            "domain_hit_rate": 0.0,
            "traj_hit_rate": 0.0,
            "domain_purity": 0.0,
            "traj_purity": 0.0,
            "mean_topk_score": 0.0,
            "mean_top1_score": 0.0,
        },
        "hierarchical": {
            "domain_hit_rate": 0.0,
            "traj_hit_rate": 0.0,
            "domain_purity": 0.0,
            "traj_purity": 0.0,
            "mean_topk_score": 0.0,
            "mean_top1_score": 0.0,
        },
        # Only meaningful when oracle_domain is False (i.e., domain inference is used).
        "domain_inference": {
            "accuracy": 0.0,
        },
    }
    details: List[Dict[str, Any]] = []
    domain_infer_total = 0
    domain_infer_correct = 0

    for traj_id in traj_ids:
        traj = trajectories.get(traj_id)
        if not isinstance(traj, dict):
            raise ValueError(f"Trajectory meta must be an object for trajectory_id={traj_id}")

        task = traj.get("task_description")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Missing task_description for trajectory_id={traj_id}")
        task = task.strip()

        gt_domain = traj.get("domain")
        if not isinstance(gt_domain, str) or not gt_domain.strip():
            raise ValueError(f"Missing domain for trajectory_id={traj_id}")
        gt_domain = gt_domain.strip()

        steps_pointer = traj.get("steps_pointer")
        if not isinstance(steps_pointer, str) or not steps_pointer.strip():
            raise ValueError(f"Missing steps_pointer for trajectory_id={traj_id}")
        steps_pointer = steps_pointer.strip()

        if steps_pointer not in traj_data_cache:
            traj_data_cache[steps_pointer] = _load_and_maybe_merge_failure(steps_pointer)
        raw = traj_data_cache[steps_pointer]
        rounds = raw.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            raise ValueError(f"Trajectory rounds missing/empty for steps_pointer={steps_pointer}")

        for qi in range(int(args.queries_per_trajectory)):
            if len(details) >= int(args.max_queries):
                break
            if args.query_step is not None:
                # Fixed-step evaluation (e.g., step 0 to align with trajectory start keyframe).
                # Only evaluate once per trajectory to avoid duplicate queries.
                if qi > 0:
                    break
                step_idx = int(args.query_step)
                if step_idx >= len(rounds):
                    raise ValueError(
                        f"--query_step={step_idx} out of range for trajectory_id={traj_id} "
                        f"(len(rounds)={len(rounds)})"
                    )
            else:
                step_idx = rng.randrange(len(rounds))
            round_obj = rounds[step_idx]
            if not isinstance(round_obj, dict):
                raise ValueError(f"Round must be an object at step={step_idx} for trajectory_id={traj_id}")

            imgs = _extract_round_images(round_obj)
            if not imgs:
                raise ValueError(f"No screenshot found for trajectory_id={traj_id} step={step_idx}")
            image_b64 = imgs[0]
            page_desc = _extract_step_summary(round_obj)

            # Domain inference accuracy (only when we are not in oracle mode).
            pred_domain_label: Optional[str] = None
            if not bool(args.oracle_domain):
                pred_domain_id = retriever._select_best_domain_id(  # type: ignore[attr-defined]
                    intent=task,
                    page_description=page_desc or "",
                )
                if pred_domain_id is None:
                    raise ValueError("Domain inference returned None (no domain embeddings?)")
                pred_meta = retriever.domains.get(pred_domain_id)
                if not isinstance(pred_meta, dict):
                    raise ValueError(f"Predicted domain id not found in domains.json: {pred_domain_id!r}")
                pred_label = pred_meta.get("domain")
                if not isinstance(pred_label, str) or not pred_label.strip():
                    raise ValueError(f"Predicted domain missing label for domain_id={pred_domain_id!r}")
                pred_domain_label = pred_label.strip()
                domain_infer_total += 1
                if pred_domain_label.lower() == gt_domain.lower():
                    domain_infer_correct += 1

            # ----------------------------
            # Phase-only retrieval
            # ----------------------------
            q_vec = encoder.encode_phase_query(
                intent_text=task,
                page_description=page_desc,
                image_b64=image_b64,
            )
            phase_only_hits = retriever.phase_index.search(q_vec, k=int(args.k))
            phase_only_ids = [x[0] for x in phase_only_hits]
            phase_only_scores = [float(x[1]) for x in phase_only_hits]
            if len(phase_only_scores) != len(phase_only_ids) or len(phase_only_scores) == 0:
                raise ValueError("phase_only retrieval returned empty/invalid scores")

            # ----------------------------
            # Hierarchical retrieval
            # ----------------------------
            domain_arg = gt_domain if bool(args.oracle_domain) else None
            hierarchical = retriever.retrieve(
                intent=task,
                image_b64=image_b64,
                domain=domain_arg,
                k=int(args.k),
                page_description=page_desc,
            )
            hierarchical_ids = [ex["phase_id"] for ex in hierarchical]
            hierarchical_scores = [float(ex["score"]) for ex in hierarchical]
            if len(hierarchical_scores) != len(hierarchical_ids) or len(hierarchical_scores) == 0:
                raise ValueError("hierarchical retrieval returned empty/invalid scores")

            def purity(ids: List[str]) -> Tuple[float, float, bool, bool]:
                if not ids:
                    raise ValueError("retrieval returned empty list")
                dom_ok = 0
                traj_ok = 0
                for rid in ids:
                    rmeta = phase_meta.get(rid, {})
                    rdom = rmeta.get("domain")
                    rtid = rmeta.get("trajectory_id")
                    if rdom == gt_domain:
                        dom_ok += 1
                    if rtid == traj_id:
                        traj_ok += 1
                dom_hit = dom_ok > 0
                traj_hit = traj_ok > 0
                return dom_ok / float(len(ids)), traj_ok / float(len(ids)), dom_hit, traj_hit

            po_dom_purity, po_traj_purity, po_dom_hit, po_traj_hit = purity(phase_only_ids)
            hi_dom_purity, hi_traj_purity, hi_dom_hit, hi_traj_hit = purity(hierarchical_ids)
            po_mean_score = float(sum(phase_only_scores) / float(len(phase_only_scores)))
            hi_mean_score = float(sum(hierarchical_scores) / float(len(hierarchical_scores)))
            po_top1 = float(phase_only_scores[0])
            hi_top1 = float(hierarchical_scores[0])

            details.append(
                {
                    "trajectory_id": traj_id,
                    "domain": gt_domain,
                    "step_idx": step_idx,
                    "pred_domain": pred_domain_label,
                    "phase_only": {
                        "ids": phase_only_ids,
                        "scores": phase_only_scores,
                        "domain_hit": po_dom_hit,
                        "traj_hit": po_traj_hit,
                        "domain_purity": po_dom_purity,
                        "traj_purity": po_traj_purity,
                        "mean_topk_score": po_mean_score,
                        "top1_score": po_top1,
                    },
                    "hierarchical": {
                        "ids": hierarchical_ids,
                        "scores": hierarchical_scores,
                        "domain_hit": hi_dom_hit,
                        "traj_hit": hi_traj_hit,
                        "domain_purity": hi_dom_purity,
                        "traj_purity": hi_traj_purity,
                        "mean_topk_score": hi_mean_score,
                        "top1_score": hi_top1,
                    },
                }
            )

        if len(details) >= int(args.max_queries):
            break

    # Aggregate
    n = float(len(details))
    if n == 0:
        raise ValueError("No queries evaluated")

    for d in details:
        for method in ("phase_only", "hierarchical"):
            metrics[method]["domain_hit_rate"] += 1.0 if bool(d[method]["domain_hit"]) else 0.0
            metrics[method]["traj_hit_rate"] += 1.0 if bool(d[method]["traj_hit"]) else 0.0
            metrics[method]["domain_purity"] += float(d[method]["domain_purity"])
            metrics[method]["traj_purity"] += float(d[method]["traj_purity"])
            metrics[method]["mean_topk_score"] += float(d[method]["mean_topk_score"])
            metrics[method]["mean_top1_score"] += float(d[method]["top1_score"])

    for method in ("phase_only", "hierarchical"):
        for k in list(metrics[method].keys()):
            metrics[method][k] = float(metrics[method][k]) / n

    if bool(args.oracle_domain):
        metrics["domain_inference"]["accuracy"] = None  # type: ignore[assignment]
    else:
        if domain_infer_total == 0:
            raise ValueError("domain_infer_total == 0 (unexpected)")
        metrics["domain_inference"]["accuracy"] = float(domain_infer_correct) / float(domain_infer_total)

    summary = {
        "index_dir": index_dir,
        "k": int(args.k),
        "max_queries": int(args.max_queries),
        "evaluated_queries": int(len(details)),
        "oracle_domain": bool(args.oracle_domain),
        "encoder_model": encoder_name,
        "metrics": metrics,
    }

    print(json.dumps(summary, indent=2))

    if args.out_json:
        out = os.path.abspath(args.out_json)
        payload = {"summary": summary, "details": details}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()


