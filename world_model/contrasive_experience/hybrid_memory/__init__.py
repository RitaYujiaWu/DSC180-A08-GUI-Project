"""
Hybrid Memory package for hierarchical, multimodal memory retrieval.

Architecture:
- Domain -> Trajectory -> Phase (hierarchical tree)
- Search vectors for fast retrieval (FAISS)
- Latent packs for continuous model injection (Q-Former)

Modules:
- schema: Data classes (Domain, Trajectory, PhaseNote)
- constructor: Build memory objects from raw trajectories
- encoder: Generate search vectors and latent packs
- store: FAISS/numpy index + disk KV storage
- pipeline: Orchestrate construction, encoding, and storage
- retriever: Query and return exemplars with latent paths
"""

from .schema import Domain, Trajectory, PhaseNote, PhaseNeighbor
from .encoder import MemoryEncoder, PhaseEncoder, TrajectoryEncoder, DomainEncoder
from .store import VectorIndex, DiskKV
from .retriever import HybridRetriever
from .pipeline import build_memory_index, HierarchicalMemoryStore
from .constructor import MemoryConstructor

__all__ = [
    # Schema
    "Domain",
    "Trajectory",
    "PhaseNote",
    "PhaseNeighbor",
    # Encoder
    "MemoryEncoder",
    "PhaseEncoder",
    "TrajectoryEncoder",
    "DomainEncoder",
    # Store
    "VectorIndex",
    "DiskKV",
    # Retriever
    "HybridRetriever",
    # Pipeline
    "build_memory_index",
    "HierarchicalMemoryStore",
    # Constructor
    "MemoryConstructor",
]
