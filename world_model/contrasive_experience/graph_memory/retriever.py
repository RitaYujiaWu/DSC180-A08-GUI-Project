"""
Hybrid Retriever: FAISS + Graph Expansion.

Combines embedding-based similarity search with graph-based expansion
for diverse trajectory retrieval.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any

import numpy as np
import faiss

from .tagged_trajectory import TaggedTrajectory
from .graph_builder import GraphBuilder


logger = logging.getLogger(__name__)


class GraphMemoryRetriever:
    """
    Retriever that combines:
    1. FAISS similarity search (multimodal: text + image)
    2. Graph expansion (find neighbors via shared tags)
    3. Re-rank by embedding similarity
    """
    
    def __init__(self,
                 graph_builder: GraphBuilder,
                 embedding_model: Any = None,
                 expand_hops: int = 1,
                 initial_seeds: int = None):
        """
        Initialize the retriever.
        
        Args:
            graph_builder: GraphBuilder instance with trajectories
            embedding_model: Model for computing query embeddings (must have get_multimodal_embeddings)
            expand_hops: Number of hops to expand in graph
            initial_seeds: Number of seeds to retrieve from FAISS in Phase 1.
                          If None, defaults to k//2. Set to 1 to test expansion-heavy retrieval.
        """
        self.graph_builder = graph_builder
        self.embedding_model = embedding_model
        self.expand_hops = expand_hops
        self.initial_seeds = initial_seeds
        
        # FAISS index
        self.faiss_index: Optional[faiss.Index] = None
        self.id_to_faiss_idx: Dict[str, int] = {}
        self.faiss_idx_to_id: Dict[int, str] = {}
        
        # Build FAISS index if trajectories have embeddings
        self._build_faiss_index()
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index from trajectory embeddings."""
        trajectories = self.graph_builder.trajectories
        
        if not trajectories:
            logger.warning("[GraphMemoryRetriever] No trajectories to index")
            return
        
        # Collect embeddings
        embeddings = []
        self.id_to_faiss_idx = {}
        self.faiss_idx_to_id = {}
        
        for traj_id, traj in trajectories.items():
            if traj.embedding is not None:
                idx = len(embeddings)
                embeddings.append(traj.embedding)
                self.id_to_faiss_idx[traj_id] = idx
                self.faiss_idx_to_id[idx] = traj_id
        
        if not embeddings:
            logger.warning("[GraphMemoryRetriever] No embeddings found in trajectories")
            return
        
        # Build index
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)
        
        logger.info(f"[GraphMemoryRetriever] Built FAISS index with {len(embeddings)} vectors")
    
    def add_trajectory(self, trajectory: TaggedTrajectory) -> None:
        """Add a trajectory to both graph and FAISS index."""
        self.graph_builder.add_trajectory(trajectory)
        
        # Add to FAISS if has embedding
        if trajectory.embedding is not None:
            if self.faiss_index is None:
                # Create new index
                dim = trajectory.embedding.shape[0]
                self.faiss_index = faiss.IndexFlatIP(dim)
            
            # Add to index
            idx = self.faiss_index.ntotal
            embedding = trajectory.embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(embedding)
            self.faiss_index.add(embedding)
            
            self.id_to_faiss_idx[trajectory.id] = idx
            self.faiss_idx_to_id[idx] = trajectory.id
    
    def retrieve(self,
                 query_embedding: Optional[np.ndarray] = None,
                 query_text: Optional[str] = None,
                 query_image: Optional[str] = None,
                 query_tags: Optional[Set[str]] = None,
                 k: int = 5) -> List[TaggedTrajectory]:
        """
        Retrieve trajectories using multimodal retrieval:
        - Query = (text intent + current screenshot)
        - Index = (task_description + first screenshot) embeddings
        
        Flow:
        1. FAISS top-k*2 seeds (multimodal similarity)
        2. Expand 1 hop via shared tags
        3. Re-rank all candidates by similarity → return top-k
        
        Args:
            query_embedding: Pre-computed query vector (optional)
            query_text: Query text (used with query_image to compute embedding)
            query_image: Base64 screenshot for multimodal embedding
            query_tags: Optional set of tags for tag-based filtering (currently unused)
            k: Number of results to return
        Returns:
            List of TaggedTrajectory objects
        """
        # Note: query_tags is accepted but not currently used
        # Future enhancement: use tags for additional filtering/boosting
        # Build query embedding (multimodal)
        if query_embedding is None:
            if not query_text or not isinstance(query_text, str) or not query_text.strip():
                raise ValueError("query_text must be a non-empty string when query_embedding is None")
            if query_image is None or not isinstance(query_image, str) or not query_image.strip():
                raise ValueError("query_image must be a non-empty base64 string when query_embedding is None")
            if self.embedding_model is None:
                raise ValueError("embedding_model must not be None when query_embedding is None")
            if not hasattr(self.embedding_model, "get_multimodal_embeddings"):
                raise TypeError("embedding_model must implement get_multimodal_embeddings for multimodal retrieval")
            
            q = self.embedding_model.get_multimodal_embeddings([query_text], [query_image])
            if not isinstance(q, np.ndarray):
                raise TypeError(f"Expected numpy.ndarray from get_multimodal_embeddings, got {type(q)}")
            if q.ndim != 2 or q.shape[0] != 1:
                raise ValueError(f"Expected query embedding shape (1, D), got {q.shape}")
            query_embedding = q[0]
        

        # Phase 1: Get seeds from FAISS
        # Use initial_seeds if specified, otherwise default to k//2
        if self.initial_seeds is not None:
            seed_count = max(self.initial_seeds, 1)
        else:
            seed_count = max(k // 2, 1)  # Default: k/2 seeds
        if self.faiss_index:
            seed_count = min(seed_count, self.faiss_index.ntotal)
        faiss_results = self._faiss_search(query_embedding, seed_count)
        faiss_scores = {traj_id: score for traj_id, score in faiss_results}
        seed_ids = set(faiss_scores.keys())
        logger.info(f"[GraphMemoryRetriever] Phase 1: {len(seed_ids)} seeds from FAISS (initial_seeds={self.initial_seeds})")
        
        # Phase 2: Expand 1 hop from each seed
        candidates = set(seed_ids)
        for seed_id in seed_ids:
            neighbors = self.graph_builder.get_neighbors(seed_id, max_hops=self.expand_hops)
            for neighbor_id, distance, shared_tags in neighbors:
                candidates.add(neighbor_id)
        logger.info(f"[GraphMemoryRetriever] Phase 2: {len(candidates)} candidates after expansion")
        
        # Phase 3: Re-rank all candidates by embedding similarity, return top-k
        scored_candidates = []
        for traj_id in candidates:
            traj = self.graph_builder.get_trajectory(traj_id)
            if traj is None or traj.embedding is None:
                continue
            # Compute cosine similarity
            traj_emb = traj.embedding.astype('float32')
            traj_emb = traj_emb / (np.linalg.norm(traj_emb) + 1e-9)
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
            sim = float(np.dot(query_norm, traj_emb))
            scored_candidates.append((traj_id, sim, traj))
        
        # Sort by similarity descending, take top-k
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        out = [traj for _, _, traj in scored_candidates[:k]]
        logger.info(f"[GraphMemoryRetriever] Phase 3: returning top {len(out)} results")
        return out
    
    def _faiss_search(self, 
                      query_embedding: Optional[np.ndarray],
                      k: int) -> List[Tuple[str, float]]:
        """
        Phase 1: FAISS similarity search.
        
        Returns list of (trajectory_id, score) tuples.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if query_embedding is None:
            raise ValueError("query_embedding must not be None for FAISS search")
        if not isinstance(query_embedding, np.ndarray):
            raise TypeError(f"query_embedding must be a numpy.ndarray, got {type(query_embedding)}")
        if query_embedding.ndim != 1:
            raise ValueError(f"query_embedding must be 1D, got shape {query_embedding.shape}")
        if self.faiss_index is None:
            raise RuntimeError("FAISS index is not built. Ensure trajectories include embeddings before retrieval.")
        if query_embedding.shape[0] != self.faiss_index.d:
            raise ValueError(
                f"query_embedding dim mismatch: got {query_embedding.shape[0]}, expected {self.faiss_index.d}"
            )
        
        # Normalize query
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        # Search
        n_search = min(k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query, n_search)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            
            traj_id = self.faiss_idx_to_id.get(idx)
            if traj_id is None:
                continue
            
            results.append((traj_id, float(score)))
        
        return results
    
    def format_results(self, 
                       results: List[TaggedTrajectory],
                       include_tags: bool = True) -> str:
        """
        Format retrieval results as a string for prompt injection.
        """
        if not results:
            return "[No relevant experiences found]"
        
        lines = ["[Experience Guidance from Graph Memory]"]
        
        # Collect all covered tags for summary
        all_tags = set()
        for traj in results:
            all_tags.update(traj.tags)
        
        lines.append(f"Strategies from {len(results)} relevant experiences:")
        
        for i, traj in enumerate(results, 1):
            lines.append(f"\n{i}. {traj.takeaway}")
            if include_tags:
                tag_str = ", ".join(sorted(traj.tags)[:5])
                lines.append(f"   Tags: {tag_str}")
        
        lines.append(f"\nCovered strategies: {', '.join(sorted(all_tags)[:10])}")
        
        return "\n".join(lines)
    
    def save_index(self, filepath: str) -> None:
        """Save FAISS index to disk."""
        if self.faiss_index is None:
            raise ValueError("Cannot save FAISS index: faiss_index is None")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.faiss_index, str(filepath))
        
        # Save ID mappings
        import json
        mappings = {
            'id_to_idx': self.id_to_faiss_idx,
            'idx_to_id': {str(k): v for k, v in self.faiss_idx_to_id.items()}
        }
        with open(f"{filepath}.mappings.json", 'w') as f:
            json.dump(mappings, f)
        
        logger.info(f"[GraphMemoryRetriever] Saved FAISS index to {filepath}")
    
    def load_index(self, filepath: str) -> None:
        """Load FAISS index from disk."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"FAISS index file not found: {filepath}")
        
        self.faiss_index = faiss.read_index(str(filepath))
        
        # Load ID mappings
        import json
        mappings_path = Path(f"{filepath}.mappings.json")
        if mappings_path.exists():
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
            self.id_to_faiss_idx = mappings.get('id_to_idx', {})
            self.faiss_idx_to_id = {int(k): v for k, v in mappings.get('idx_to_id', {}).items()}
        
        logger.info(f"[GraphMemoryRetriever] Loaded FAISS index from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            'graph_trajectories': len(self.graph_builder),
            'graph_edges': self.graph_builder.graph.number_of_edges(),
            'graph_tags': len(self.graph_builder.tag_to_trajectories),
            'faiss_vectors': self.faiss_index.ntotal if self.faiss_index else 0,
            'expand_hops': self.expand_hops,
        }

