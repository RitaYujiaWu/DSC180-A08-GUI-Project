from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import hashlib


def compute_file_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 of a file; returns empty string if file not found."""
    try:
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha.update(chunk)
        return sha.hexdigest()
    except Exception:
        return ""


@dataclass
class Domain:
    id: str
    category: str = "domain"
    domain: str = ""                      # e.g., "shopping", "maps"
    
    # Embedding and encoder metadata are required for domain-level retrieval
    embedding: List[float] = field(default_factory=list)
    encoder_name: str = ""
    encoder_version: str = ""
    embedding_dim: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    id: str
    category: str = "trajectory"
    domain: str = ""
    domain_id: Optional[str] = None              # domain linkage (useful for gating/filtering)
    task_description: str = ""  # DISCRETE text descriptor for the whole task
    outcome: Optional[str] = None            # "success" | "failure" | "partial"
    summary: Optional[str] = None            # DISCRETE short text summary
    phase_note_ids: List[str] = field(default_factory=list)  # Child PhaseNote IDs (scope phases under this trajectory)
    embedding: Optional[List[float]] = None  # RETRIEVAL: 1D trajectory search vector (e.g., success‑weighted mean of child phase embeddings)
    
    # Multimodal artifacts
    # NOTE: For trajectories we now only persist the *first* frame image.
    # This is stored as a list of length 0 or 1 for consistency with
    # phase-level keyframe handling.
    keyframe_paths: Optional[List[str]] = None

    # Storage/pointers
    steps_pointer: Optional[str] = None      # DISCRETE source: path to raw JSON/JSONL with full steps (heavy, not for search)
    source_hash: Optional[str] = None        # provenance for steps_pointer
    traj_latent_pack_path: Optional[str] = None  # CONTINUOUS: optional trajectory‑level latent pack (.npz with (N,D) tensor) for model injection

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def set_source_hash_from_file(self) -> None:
        if self.steps_pointer:
            self.source_hash = compute_file_sha256(self.steps_pointer)


@dataclass
class PhaseNeighbor:
    to_phase_id: str
    relation: str = "next"  # next|alt|backtrack
    count: int = 0
    success_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhaseNote:
    id: str
    category: str = "phase"
    domain: str = ""
    trajectory_id: str = ""
    phase_label: str = ""                  # e.g., search|filter|purchase
    start_step: int = 0
    end_step: int = 0                      # inclusive
    summary: str = ""                      # discrete summary text
    embedding: Optional[List[float]] = None  # 1D Retrieval vector (e.g. CLIP/MiniLM)

    # Multimodal artifacts
    keyframe_indices: Optional[List[int]] = None
    keyframe_paths: Optional[List[str]] = None
    
    # Continuous Memory Latents (for Model Injection)
    latent_pack_path: Optional[str] = None  # Path to .npz with (N, D) tensor
    qformer_path: Optional[str] = None # Path to Q-Former weights (provenance)

    # Encoder provenance
    encoder_name: Optional[str] = None
    encoder_version: Optional[str] = None
    source_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
