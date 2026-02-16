"""
Pipeline module: Orchestrate memory construction, encoding, and storage.

Responsibilities:
- Iterate over trajectory files
- Call constructor to build Domain/Trajectory/PhaseNote objects
- Call encoder to generate search vectors and latent packs
- Call store to persist indices and metadata

This is a thin orchestrator that delegates to:
- constructor.py: Build memory objects
- encoder.py: Encode to vectors/latents
- store.py: Persist to disk
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from .schema import Domain, Trajectory, PhaseNote
from .constructor import MemoryConstructor
from .encoder import MemoryEncoder
from .store import VectorIndex, DiskKV


# =============================================================================
# Hierarchical Memory Store
# =============================================================================

class HierarchicalMemoryStore:
    """
    Store for hierarchical memory indices (Domain -> Trajectory -> Phase).
    
    Directory structure:
        output_dir/
        ├── domains.json           # Domain metadata
        ├── trajectories.json      # Trajectory metadata
        ├── phases/
        │   ├── vectors.faiss      # Phase search vectors
        │   ├── vectors.ids.json
        │   ├── vectors.meta.json
        │   └── latents/           # Phase latent packs (.npz)
        │       └── kv.json
        ├── trajectories_index/    # Optional: Trajectory search vectors
        │   ├── vectors.faiss
        │   ├── vectors.ids.json
        │   └── vectors.meta.json
        └── keyframes/             # Saved keyframe images
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.phases_dir = os.path.join(output_dir, "phases")
        self.traj_index_dir = os.path.join(output_dir, "trajectories_index")
        self.latents_dir = os.path.join(self.phases_dir, "latents")
        
        os.makedirs(self.phases_dir, exist_ok=True)
        os.makedirs(self.traj_index_dir, exist_ok=True)
        os.makedirs(self.latents_dir, exist_ok=True)
        
        # Initialize stores
        self.phase_index: Optional[VectorIndex] = None
        self.traj_index: Optional[VectorIndex] = None
        self.latent_kv = DiskKV(self.latents_dir)
        
        # Metadata stores
        self.domains: Dict[str, Dict] = {}
        self.trajectories: Dict[str, Dict] = {}
    
    # -------------------------------------------------------------------------
    # Domain Storage
    # -------------------------------------------------------------------------
    
    def add_domain(self, domain: Domain) -> None:
        """Add a domain to the store."""
        self.domains[domain.id] = domain.to_dict()
    
    def save_domains(self) -> None:
        """Persist domains to disk."""
        path = os.path.join(self.output_dir, "domains.json")
        with open(path, "w") as f:
            json.dump(self.domains, f, indent=2)
    
    # -------------------------------------------------------------------------
    # Trajectory Storage
    # -------------------------------------------------------------------------
    
    def add_trajectory(
        self,
        trajectory: Trajectory,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Add a trajectory to the store with optional embedding.

        Notes
        -----
        We persist BOTH the human‑readable domain label (`domain`) and the
        stable identifier (`domain_id`) so that retrieval can gate by
        domain‑id while still exposing a friendly label in metadata.
        """
        traj_dict = trajectory.to_dict()
        # Keep trajectories.json lightweight and avoid duplication:
        # - retrieval vectors live in trajectories_index/vectors.faiss
        traj_dict.pop("embedding")
        
        # Store embedding if provided
        if embedding is not None:
            # Add to trajectory index
            if self.traj_index is None:
                self.traj_index = VectorIndex(dim=int(embedding.shape[0]))
            self.traj_index.add(
                item_id=trajectory.id,
                vec=embedding,
                meta={
                    "task_description": trajectory.task_description,
                    "outcome": trajectory.outcome,
                    # Human‑oriented label (e.g. "shopping")
                    "domain": trajectory.domain,
                    # Stable id used for gating. For existing indices where
                    # domain_id was not persisted, this will be None and we
                    # should gracefully fall back to the label.
                    "domain_id": trajectory.domain_id,
                    "phase_count": len(trajectory.phase_note_ids),
                },
            )
        
        self.trajectories[trajectory.id] = traj_dict
    
    def save_trajectories(self) -> None:
        """Persist trajectories to disk."""
        path = os.path.join(self.output_dir, "trajectories.json")
        with open(path, "w") as f:
            json.dump(self.trajectories, f, indent=2)
        
        # Save trajectory index if exists
        if self.traj_index is not None:
            self.traj_index.save(self.traj_index_dir)
    
    # -------------------------------------------------------------------------
    # Phase Storage
    # -------------------------------------------------------------------------
    
    def add_phase(
        self,
        phase: PhaseNote,
        retrieval_vec: np.ndarray,
        latent_pack_path: str,
    ) -> None:
        """Add a phase to the index with its vectors."""
        # Initialize index if needed
        if self.phase_index is None:
            self.phase_index = VectorIndex(dim=int(retrieval_vec.shape[0]))
        
        # Build metadata for search
        meta = {
            "summary": phase.summary,
            "phase_label": phase.phase_label,
            "keyframe_paths": phase.keyframe_paths,
            "trajectory_id": phase.trajectory_id,
            "domain": phase.domain,
            "start_step": phase.start_step,
            "end_step": phase.end_step,
        }
        
        # Add to index
        self.phase_index.add(phase.id, retrieval_vec, meta=meta)
        
        # Store latent pack path
        self.latent_kv.set(phase.id, latent_pack_path)
    
    def save_phases(self) -> None:
        """Persist phase index to disk."""
        if self.phase_index is not None:
            self.phase_index.save(self.phases_dir)
    
    # -------------------------------------------------------------------------
    # Save All
    # -------------------------------------------------------------------------
    
    def save_all(self) -> None:
        """Persist all indices and metadata to disk."""
        self.save_domains()
        self.save_trajectories()
        self.save_phases()


# =============================================================================
# Main Pipeline
# =============================================================================

def build_memory_index(
    input_files: List[str],
    output_dir: str,
    encoder: Optional[MemoryEncoder] = None,
    prompt_dir: str = os.path.join(os.path.dirname(__file__), "prompts"),
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    vlm_base_url: Optional[str] = "http://localhost:8000/v1",
    vlm_api_key: Optional[str] = None,
    compute_trajectory_embeddings: bool = True,
    compute_domain_embeddings: bool = True,
) -> HierarchicalMemoryStore:
    """
    Build a hierarchical memory index from trajectory files.
    
    Pipeline:
        1. For each file: construct Domain/Trajectory/PhaseNote objects
        2. Encode phases -> search vectors + latent packs
        3. Aggregate phase vectors -> trajectory embedding
        4. Encode domains -> domain embedding
        5. Store everything to disk
    
    Args:
        input_files: List of trajectory JSON/JSONL file paths
        output_dir: Output directory for index
        site_host: (legacy) ignored
        encoder: MemoryEncoder instance (created if None)
        prompt_dir: Directory containing VLM prompts
        vlm_model: VLM model name for segmentation
        vlm_base_url: VLM API base URL
        vlm_api_key: VLM API key
        compute_trajectory_embeddings: Whether to compute trajectory-level embeddings
        compute_domain_embeddings: Whether to compute domain-level embeddings
    
    Returns:
        HierarchicalMemoryStore with all indices
    """
    # Initialize components
    encoder = encoder or MemoryEncoder()
    store = HierarchicalMemoryStore(output_dir)
    constructor = MemoryConstructor(
        output_dir=output_dir,
        prompt_dir=prompt_dir,
        vlm_model=vlm_model,
        vlm_base_url=vlm_base_url,
        vlm_api_key=vlm_api_key,
    )

    # Canonicalize input files:
    # - success trajectories are single files under .../success/...
    # - failure trajectories are represented as (.../positive/..., .../negative/...)
    #   and must be merged (positive first, then negative) before VLM segmentation.
    pos_token = f"{os.sep}positive{os.sep}"
    neg_token = f"{os.sep}negative{os.sep}"
    canonical_files: List[str] = []
    seen: set[str] = set()
    for fp in input_files:
        canon = fp
        if neg_token in fp:
            canon = fp.replace(neg_token, pos_token)
        if not os.path.exists(canon):
            raise FileNotFoundError(f"Input file not found: {canon}")
        # If this is a positive-part file, require the matching negative-part file.
        if pos_token in canon:
            neg_fp = canon.replace(pos_token, neg_token)
            if not os.path.exists(neg_fp):
                raise FileNotFoundError(
                    f"Missing negative-part trajectory file for: {canon}\n"
                    f"Expected: {neg_fp}"
                )
        if canon not in seen:
            canonical_files.append(canon)
            seen.add(canon)
    input_files = canonical_files
    
    # Process each trajectory file
    for file_path in input_files:
        # Step 1: Construct memory objects
        trajectory, phases = constructor.process_trajectory_file(
            file_path=file_path,
        )
        
        # Step 2: Encode phases and store
        phase_vectors: List[np.ndarray] = []
        phase_credits: List[float] = []
        
        for phase in phases:
            # Encode phase -> search vector + latent pack
            retrieval_vec, latent_path = encoder.encode_phase_from_parts(
                phase_id=phase.id,
                task_description=trajectory.task_description,
                summary_text=phase.summary,
                keyframe_paths=phase.keyframe_paths or [],
                output_dir=os.path.join(output_dir, "phases", "latents"),
                meta={
                    "trajectory_id": phase.trajectory_id,
                    "label": phase.phase_label,
                    "start": phase.start_step,
                    "end": phase.end_step,
                    "domain": phase.domain,
                },
            )
            
            # Update phase with encoding info
            phase.embedding = retrieval_vec.tolist()
            phase.latent_pack_path = latent_path
            phase.encoder_name = encoder.model_name
            
            # Store phase
            store.add_phase(phase, retrieval_vec, latent_path)
            
            # Collect for trajectory aggregation
            phase_vectors.append(retrieval_vec)
            phase_credits.append(1.0)
        
        # Step 3: Compute trajectory embedding using TrajectoryEncoder
        traj_embedding = None
        if compute_trajectory_embeddings and trajectory.keyframe_paths:
            first_frame = [trajectory.keyframe_paths[0]]
            traj_embedding = encoder.encode_trajectory_embedding(
                trajectory=trajectory,
                keyframe_paths=first_frame,
            )
            trajectory.embedding = traj_embedding.tolist()
        
        # Store trajectory
        store.add_trajectory(trajectory, traj_embedding)
    
    # Step 4: Encode and store domains
    for domain in constructor.get_domains():
        if compute_domain_embeddings:
            domain_vec = encoder.encode_domain(domain)
            domain.embedding = domain_vec.tolist()
            domain.encoder_name = encoder.model_name
            domain.embedding_dim = int(domain_vec.shape[0])
        store.add_domain(domain)
    
    # Persist everything
    store.save_all()
    
    print(f"[Pipeline] Built memory index at {output_dir}")
    print(f"  - Domains: {len(store.domains)}")
    print(f"  - Trajectories: {len(store.trajectories)}")
    print(f"  - Phases: {len(store.phase_index.ids) if store.phase_index else 0}")
    
    return store
