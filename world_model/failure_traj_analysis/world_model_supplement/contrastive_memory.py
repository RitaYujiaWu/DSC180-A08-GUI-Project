from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import json

from .embedding_backend import Embedder, create_embedder
from .utils import list_jsonl_files, read_jsonl, safe_get_first_screenshot


@dataclass
class RetrievedTrajectory:
    traj_id: str
    task: str
    outcome: str  # "success" or "failure"
    domain: str
    dataset: str
    score: float
    steps: List[Dict[str, Any]]
    total_rounds: Optional[int] = None


class ContrastiveTrajectoryStore:
    """
    Dual-index trajectory store: success index + failure index (FAISS inner product over L2-normalized vectors).

    Expects directory format:
      training_data/{dataset}/{domain}/success/*.jsonl
      training_data/{dataset}/{domain}/failure/*.jsonl
    """
    def __init__(
        self,
        training_data_path: str,
        top_k: int = 3,
        multimodal: bool = True,
        embed_backend: str = "clip",
        device: str = "cpu",
    ) -> None:
        self.training_data_path = training_data_path
        self.top_k = top_k
        self.multimodal = multimodal
        self.embedder: Embedder = create_embedder(backend=embed_backend, device=device)

        # Lazy-built per (dataset, domain)
        self._indices: Dict[str, Any] = {}     # key -> {"success": (index, meta), "failure": (index, meta)}
        self._dims: Optional[int] = None

    def retrieve(
        self,
        task: str,
        domain: str,
        dataset: str,
        query_image=None,
    ) -> Tuple[List[RetrievedTrajectory], List[RetrievedTrajectory], float]:
        """
        Returns: (success_list, failure_list, min_similarity_of_returned)
        """
        key = f"{dataset}::{domain}"
        if key not in self._indices:
            self._indices[key] = self._build_indices(dataset=dataset, domain=domain)

        query_text = f"{dataset}_{domain}: {task}"
        qvec = self.embedder.embed_query(text=query_text, image=query_image if self.multimodal else None)

        succ = self._search(key, "success", qvec, dataset, domain)
        fail = self._search(key, "failure", qvec, dataset, domain)

        sims = [t.score for t in succ + fail]
        min_sim = min(sims) if sims else 0.0
        return succ, fail, float(min_sim)
    
    def score_text_query(
        self,
        *,
        query_text: str,
        domain: str,
        dataset: str,
        query_image=None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Score a custom query against BOTH indices and return raw similarity scores:
          (success_scores, failure_scores)

        This is used for candidate-action ranking.
        """
        key = f"{dataset}::{domain}"
        if key not in self._indices:
            self._indices[key] = self._build_indices(dataset=dataset, domain=domain)

        k = int(top_k) if top_k is not None else self.top_k
        qvec = self.embedder.embed_query(text=query_text, image=query_image if self.multimodal else None)

        succ_scores, _ = self.embedder.search(self._indices[key]["success"][0], qvec, k) if self._indices[key]["success"][0] else ([], [])
        fail_scores, _ = self.embedder.search(self._indices[key]["failure"][0], qvec, k) if self._indices[key]["failure"][0] else ([], [])

        # Ensure lists (faiss returns fixed length sometimes)
        succ_scores = [float(s) for s in succ_scores if s is not None]
        fail_scores = [float(s) for s in fail_scores if s is not None]
        return succ_scores, fail_scores


    # -----------------------
    # internals
    # -----------------------
    def _build_indices(self, dataset: str, domain: str) -> Dict[str, Any]:
        succ_dir = os.path.join(self.training_data_path, dataset, domain, "success")
        fail_dir = os.path.join(self.training_data_path, dataset, domain, "failure")

        succ_files = list_jsonl_files(succ_dir)
        fail_files = list_jsonl_files(fail_dir)

        succ_meta, succ_vecs = self._load_and_embed(succ_files, dataset, domain, outcome="success")
        fail_meta, fail_vecs = self._load_and_embed(fail_files, dataset, domain, outcome="failure")

        succ_index = self.embedder.build_faiss_index(succ_vecs)
        fail_index = self.embedder.build_faiss_index(fail_vecs)

        return {
            "success": (succ_index, succ_meta),
            "failure": (fail_index, fail_meta),
        }

    def _load_and_embed(
        self,
        files: List[str],
        dataset: str,
        domain: str,
        outcome: str,
    ) -> Tuple[List[Dict[str, Any]], "Any"]:
        """
        meta item structure:
          {"traj_id","task","steps","total_rounds","dataset","domain","outcome"}
        """
        meta: List[Dict[str, Any]] = []
        vecs = []

        for path in files:
            for obj in read_jsonl(path):
                task = obj.get("task_description") or obj.get("task") or obj.get("instruction") or ""
                rounds = obj.get("rounds") or []
                total_rounds = obj.get("total_rounds", len(rounds))
                traj_id = obj.get("task_id") or os.path.basename(path)

                # Try to get a first screenshot (if present) for multimodal embedding
                first_img = safe_get_first_screenshot(obj) if self.multimodal else None
                text = f"{dataset}_{domain}: {task}"

                v = self.embedder.embed_trajectory_key(text=text, image=first_img)
                vecs.append(v)

                meta.append({
                    "traj_id": str(traj_id),
                    "task": task,
                    "steps": rounds,
                    "total_rounds": total_rounds,
                    "dataset": dataset,
                    "domain": domain,
                    "outcome": outcome,
                })

        return meta, self.embedder.stack(vecs)

    def _search(
        self,
        key: str,
        outcome: str,
        qvec,
        dataset: str,
        domain: str,
    ) -> List[RetrievedTrajectory]:
        index, meta = self._indices[key][outcome]
        if index is None or len(meta) == 0:
            return []

        scores, idxs = self.embedder.search(index, qvec, self.top_k)
        results: List[RetrievedTrajectory] = []

        for score, i in zip(scores, idxs):
            if i < 0 or i >= len(meta):
                continue
            m = meta[int(i)]
            results.append(
                RetrievedTrajectory(
                    traj_id=m["traj_id"],
                    task=m["task"],
                    outcome=outcome,
                    domain=domain,
                    dataset=dataset,
                    score=float(score),
                    steps=m["steps"],
                    total_rounds=m.get("total_rounds"),
                )
            )
        return results
