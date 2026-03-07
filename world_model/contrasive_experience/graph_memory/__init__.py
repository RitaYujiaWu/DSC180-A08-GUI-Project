"""
Graph Memory Module for Tagged Trajectory Management.

This module provides a graph-based memory system that:
1. Tags trajectories with action/strategy keywords
2. Connects trajectories by shared tags
3. Uses FAISS + graph expansion for diverse retrieval

Building the graph (comprehensive - all domains):
    python -m graph_memory.build_graph_from_trajectories \\
        --memory_data_dir /path/to/data/trajectories \\
        --output_path graph_index/all_domains \\
        --model qwen2.5-vl

Building the graph (single domain):
    python -m graph_memory.build_graph_from_trajectories \\
        --memory_data_dir /path/to/data/trajectories \\
        --output_path graph_index/Amazon \\
        --domain Amazon \\
        --model qwen2.5-vl

Note: Only successful trajectories (from 'success/' folders) are used.
"""

from .tagged_trajectory import TaggedTrajectory
from .tag_extractor import TagExtractor
from .graph_builder import GraphBuilder
from .retriever import GraphMemoryRetriever

__all__ = [
    'TaggedTrajectory',
    'TagExtractor', 
    'GraphBuilder',
    'GraphMemoryRetriever',
]

