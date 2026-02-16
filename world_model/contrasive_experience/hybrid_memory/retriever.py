from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .encoder import MemoryEncoder
from .store import VectorIndex, DiskKV


class HybridRetriever:
    """
    Load hierarchical indices and retrieve top-K phase exemplars for the current state.
    
    Retrieval is hierarchical but keeps scoring simple:
      Domain (VLM/text-inferred) -> Trajectory (similarity over trajectory embeddings) -> Phase (similarity over phase embeddings).
    """
    def __init__(self, index_dir: str, encoder: Optional[MemoryEncoder] = None):
        """
        Args:
            index_dir: Root directory of a hybrid index built by `build_memory_index`,
                       e.g. `hybrid_index/webvoyager`.
        """
        # Root dirs
        self.root_dir = index_dir
        self.phases_dir = os.path.join(index_dir, "phases")
        self.traj_index_dir = os.path.join(index_dir, "trajectories_index")
        # Encoders
        self.encoder = encoder or MemoryEncoder()
        # Indices
        self.phase_index: Optional[VectorIndex] = None
        self.traj_index: Optional[VectorIndex] = None
        # Metadata
        self.trajectories: Dict[str, Dict] = {}
        # Mapping from domain-id -> list of trajectory ids.
        # Domain *labels* (e.g. "shopping") are kept in metadata, but
        # retrieval and gating should prefer the stable id.
        self.domain_id_to_traj_ids: Dict[str, List[str]] = {}
        self.domains: Dict[str, Dict] = {}
        self.domain_embeddings: Dict[str, np.ndarray] = {}
        # Mapping from domain label (lowercased) -> domain id
        self.domain_label_to_id: Dict[str, str] = {}
        # Latent packs are stored under phases/latents
        self.latent_kv = DiskKV(os.path.join(self.phases_dir, "latents"))
        self._load_indices()

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def _load_indices(self) -> None:
        """Load phase / trajectory indices and trajectory metadata."""
        # Phase index: prefer phases/ subdir, fall back to root for legacy layouts
        phase_dir = self.phases_dir
        if not os.path.exists(os.path.join(phase_dir, "vectors.ids.json")):
            # Legacy: index might be directly under root_dir
            if os.path.exists(os.path.join(self.root_dir, "vectors.ids.json")):
                phase_dir = self.root_dir
        if os.path.exists(os.path.join(phase_dir, "vectors.ids.json")):
            self.phase_index = VectorIndex.load(phase_dir)
        # Trajectory index (optional)
        if os.path.exists(os.path.join(self.traj_index_dir, "vectors.ids.json")):
            self.traj_index = VectorIndex.load(self.traj_index_dir)
        # Trajectory metadata for domain/trajectory mapping (optional but useful)
        traj_meta_path = os.path.join(self.root_dir, "trajectories.json")
        if os.path.exists(traj_meta_path):
            try:
                import json

                with open(traj_meta_path, "r") as f:
                    self.trajectories = json.load(f)
            except Exception:
                self.trajectories = {}
        # Build domain-id -> trajectory ids map (simple gating by domain group)
        from collections import defaultdict

        dom_id_to_traj: Dict[str, List[str]] = defaultdict(list)
        for tid, meta in self.trajectories.items():
            # Prefer the explicit domain_id if present; fall back to the
            # human-readable label for legacy indices.
            dom_id = meta.get("domain_id") or meta.get("domain")
            if dom_id:
                dom_id_to_traj[dom_id].append(tid)
        self.domain_id_to_traj_ids = dict(dom_id_to_traj)

        # Domain metadata and embeddings (for domain-level retrieval)
        dom_meta_path = os.path.join(self.root_dir, "domains.json")
        if os.path.exists(dom_meta_path):
            try:
                import json

                with open(dom_meta_path, "r") as f:
                    self.domains = json.load(f)
            except Exception:
                self.domains = {}
        # Build label -> id map for callers that pass a domain label
        self.domain_label_to_id = {}
        for dom_id, meta in self.domains.items():
            label = meta.get("domain")
            if isinstance(label, str) and label.strip():
                self.domain_label_to_id[label.strip().lower()] = dom_id
        # Precompute domain embeddings as numpy arrays
        self.domain_embeddings = {}
        for dom_id, meta in self.domains.items():
            emb = meta.get("embedding")
            if isinstance(emb, list) and emb:
                try:
                    self.domain_embeddings[dom_id] = np.asarray(emb, dtype=np.float32)
                except Exception:
                    continue

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _select_best_domain_id(
        self,
        intent: str,
        page_description: str,
    ) -> Optional[str]:
        """
        Select the most relevant domain name using domain embeddings.
        
        Returns
        -------
        Optional[str]
            The best matching *domain id* (key from domains.json), or None if
            no domain embeddings are available.
        """
        if not self.domain_embeddings:
            return None
        # Build a simple text query describing the task and current page
        query_text = f"Domain: {intent} | {page_description}"
        # Use the domain encoder's text encoder for compatibility with stored embeddings
        q_vec = self.encoder.domain_encoder.encode_text(query_text)
        if q_vec is None:
            return None
        # q_vec is already normalized by CLIP encoder, but double check just in case
        q_vec = q_vec.astype(np.float32)
        q_norm = np.linalg.norm(q_vec) + 1e-8
        best_dom_id: Optional[str] = None
        best_score = -1e9
        for dom_id, emb in self.domain_embeddings.items():
            if emb is None or emb.size == 0:
                continue
            # emb is expected to be normalized from index, but let's be safe
            d_norm = np.linalg.norm(emb) + 1e-8
            score = float(np.dot(emb, q_vec) / (d_norm * q_norm))
            if score > best_score:
                best_score = score
                best_dom_id = dom_id
        return best_dom_id

    def _select_candidate_trajectories(
        self,
        intent: str,
        domain_id: Optional[str],
        max_traj: int = 32,
        image_b64: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        Select top trajectory ids for a given intent/domain using simple similarity.
        If no trajectory index is available, returns None.
        """
        if self.traj_index is None or not self.traj_index.ids:
            return None
        # Encode trajectory-level query (pure intent + image, no domain text injection)
        q_vec = self.encoder.encode_trajectory_query(
            task_description=intent,
            image_b64=image_b64,
        )
        # We over-sample and then filter by domain to keep it simple but robust
        k = min(max_traj * 4, len(self.traj_index.ids))
        raw_results: List[Tuple[str, float]] = self.traj_index.search(q_vec, k=k)
        # If we have a resolved domain-id, restrict to its trajectories.
        # For legacy indices where the mapping may be missing, this will
        # simply result in no gating (empty set).
        allowed_traj_ids = set(self.domain_id_to_traj_ids.get(domain_id, [])) if domain_id else set()
        selected: List[str] = []
        for traj_id, _score in raw_results:
            if allowed_traj_ids and traj_id not in allowed_traj_ids:
                continue
            selected.append(traj_id)
            if len(selected) >= max_traj:
                break
        # Fallback: if domain filter removed everything, just take top-k globally
        if not selected:
            # Helpful debug signal so we can see when domain gating is too strict
            # or when a resolved domain_id has no associated trajectories.
            if domain_id:
                print(
                    f"[HybridRetriever] Domain gating produced no trajectory "
                    f"candidates for domain_id={domain_id!r}; "
                    f"falling back to top-{max_traj} global trajectories."
                )
            selected = [tid for tid, _ in raw_results[:max_traj]]
        return selected

    def retrieve(
        self,
        intent: str,
        image_b64: str,
        domain: Optional[str] = None,
        k: int = 3,
        page_description: Optional[str] = None,
    ) -> List[Dict]:
        """
        Returns list of exemplars:
          [{ phase_id, summary, role, priors, success_stats, latent_pack_path }, ...]
        
        Retrieval steps (all similarity-based, no hand-tuned weights):
          1) Domain: if `domain` is None, infer it via domain embeddings; otherwise use the provided string.
          2) Trajectory: (optional) select top trajectories for (intent, domain).
          3) Phase: search in phase space and keep phases whose trajectory_id is in the top trajectories.
        """
        if self.phase_index is None or len(self.phase_index.ids) == 0:
            return []

        # 0) Resolve domain-id for gating.
        #
        # If the caller passed a `domain` string, it may be either:
        #   - a domain id (key in domains.json), or
        #   - a human domain label (meta["domain"])
        # Otherwise, we infer the best domain-id from domain embeddings using
        # (intent, page_description).
        resolved_domain_id: Optional[str] = None
        if domain:
            dom = domain.strip()
            if dom in self.domains:
                resolved_domain_id = dom
            else:
                dom_label = dom.lower()
                if dom_label not in self.domain_label_to_id:
                    raise ValueError(f"Unknown domain label: {domain!r}")
                resolved_domain_id = self.domain_label_to_id[dom_label]
        else:
            resolved_domain_id = self._select_best_domain_id(
                intent=intent,
                page_description=page_description or "",
            )

        # 1) Trajectory candidates (may be None if no traj index)
        candidate_traj_ids = self._select_candidate_trajectories(
            intent=intent,
            domain_id=resolved_domain_id,
            max_traj=32,
            image_b64=image_b64, # UPDATED: Pass screenshot
        )
        candidate_traj_ids_set = set(candidate_traj_ids) if candidate_traj_ids else None

        # 2) Phase-level query encoding
        q_vec = self.encoder.encode_phase_query(
            intent_text=intent,
            page_description=page_description or "",
            image_b64=image_b64,
        )
        # Over-sample to allow filtering by trajectories
        phase_k = min(max(k * 10, 50), len(self.phase_index.ids))
        phase_results: List[Tuple[str, float]] = self.phase_index.search(q_vec, k=phase_k)

        exemplars: List[Dict] = []
        # 3) Filter phases by candidate trajectories (if available)
        for pid, score in phase_results:
            meta = self.phase_index.meta.get(pid, {})
            traj_id = meta.get("trajectory_id")
            if candidate_traj_ids_set is not None and traj_id not in candidate_traj_ids_set:
                continue
            exemplars.append({
                "phase_id": pid,
                "summary": meta.get("summary", ""),
                "role": meta.get("phase_label", ""),
                "latent_pack_path": self.latent_kv.get(pid),
                "score": score,
            })
            if len(exemplars) >= k:
                break

        # Fallback: if trajectory gating removed everything, just return top-k globally
        if not exemplars:
            for pid, score in phase_results[:k]:
                meta = self.phase_index.meta.get(pid, {})
                exemplars.append({
                    "phase_id": pid,
                    "summary": meta.get("summary", ""),
                    "role": meta.get("phase_label", ""),
                    "latent_pack_path": self.latent_kv.get(pid),
                    "score": score,
                })

        return exemplars
