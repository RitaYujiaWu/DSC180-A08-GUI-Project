"""
Graph Builder for Tagged Trajectories.

Builds and manages a graph where:
- Nodes = Trajectories
- Edges = Shared tags (weighted by number of shared tags)

Supports VLM-based deduplication during graph construction:
- Uses FAISS to find similar trajectories
- VLM decides: UPDATE (enrich old), REPLACE (swap), or ADD (new node)
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict

import networkx as nx
import numpy as np
import faiss

from .tagged_trajectory import TaggedTrajectory
from .tag_extractor import TagExtractor


logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds and manages a trajectory graph.
    
    Trajectories are connected if they share tags.
    Edge weight = number of shared tags.
    
    Supports VLM-based deduplication:
    - SIMILARITY_THRESHOLD: Minimum similarity to trigger VLM comparison
    - TOP_K_COMPARE: Number of similar trajectories to compare against
    """
    
    # Deduplication constants
    SIMILARITY_THRESHOLD = 0.92
    TOP_K_COMPARE = 5
    
    def __init__(self, 
                 tag_extractor: Optional[TagExtractor] = None,
                 min_shared_tags: int = 1,
                 llm: Optional[Any] = None,
                 embedding_model: Optional[Any] = None):
        """
        Initialize the graph builder.
        
        Args:
            tag_extractor: TagExtractor instance for extracting tags
            min_shared_tags: Minimum shared tags required to create an edge
            llm: LLM for VLM-based deduplication decisions (optional)
            embedding_model: Model for computing embeddings (optional)
        """
        self.graph = nx.Graph()
        self.tag_extractor = tag_extractor
        self.min_shared_tags = min_shared_tags
        self.llm = llm
        self.embedding_model = embedding_model
        
        # Trajectories stored by ID
        self.trajectories: Dict[str, TaggedTrajectory] = {}
        
        # Tag to trajectory mapping for fast lookup
        self.tag_to_trajectories: Dict[str, Set[str]] = defaultdict(set)
        
        # FAISS index for deduplication retrieval
        self._dedup_faiss_index: Optional[faiss.Index] = None
        self._dedup_id_to_idx: Dict[str, int] = {}
        self._dedup_idx_to_id: Dict[int, str] = {}
        
        # Statistics
        self.stats = {
            'total_trajectories': 0,
            'total_edges': 0,
            'total_tags': 0,
            # Deduplication stats
            'dedup_added': 0,
            'dedup_updated': 0,
            'dedup_replaced': 0,
        }
    
    def add_trajectory(self, trajectory: TaggedTrajectory) -> None:
        """
        Add a trajectory to the graph.
        
        1. Add as node
        2. Update tag mappings
        3. Connect to existing trajectories with shared tags
        
        Args:
            trajectory: TaggedTrajectory to add
        """
        traj_id = trajectory.id
        
        # Skip if already exists
        if traj_id in self.trajectories:
            logger.debug(f"[GraphBuilder] Trajectory {traj_id} already exists, skipping")
            return
        
        # Store trajectory
        self.trajectories[traj_id] = trajectory
        
        # Add node to graph
        self.graph.add_node(
            traj_id,
            tags=list(trajectory.tags),
            takeaway=trajectory.takeaway,
            domain=trajectory.domain,
        )
        
        # Update tag mappings
        for tag in trajectory.tags:
            self.tag_to_trajectories[tag].add(traj_id)
        
        # Connect to existing trajectories with shared tags
        for other_id, other_traj in self.trajectories.items():
            if other_id == traj_id:
                continue
            
            shared_tags = trajectory.shares_tags_with(other_traj)
            
            if len(shared_tags) >= self.min_shared_tags:
                # Add weighted edge
                self.graph.add_edge(
                    traj_id,
                    other_id,
                    weight=len(shared_tags),
                    shared_tags=list(shared_tags),
                )
        
        # Update stats
        self.stats['total_trajectories'] = len(self.trajectories)
        self.stats['total_edges'] = self.graph.number_of_edges()
        self.stats['total_tags'] = len(self.tag_to_trajectories)
        
        logger.debug(f"[GraphBuilder] Added trajectory {traj_id} with {len(trajectory.tags)} tags, "
                    f"{self.graph.degree(traj_id)} connections")
    
    def add_trajectory_from_data(self,
                                  traj_id: str,
                                  takeaway: str,
                                  domain: str = "",
                                  embedding: Optional[np.ndarray] = None,
                                  full_data: Optional[Dict] = None) -> TaggedTrajectory:
        """
        Create and add a trajectory from raw data.
        
        Uses the tag extractor to extract tags from takeaway.
        
        Args:
            traj_id: Trajectory identifier
            takeaway: Summary text
            domain: Domain name
            embedding: Optional embedding vector
            full_data: Optional full trajectory data
        
        Returns:
            Created TaggedTrajectory
        """
        # Extract tags
        if self.tag_extractor is None:
            raise ValueError("GraphBuilder.add_trajectory_from_data requires tag_extractor (got None)")
            tags = self.tag_extractor.extract_tags(takeaway, domain, traj_id)
        
        # Create trajectory
        trajectory = TaggedTrajectory(
            id=traj_id,
            takeaway=takeaway,
            tags=tags,
            embedding=embedding,
            domain=domain,
            full_data=full_data or {},
        )
        
        # Add to graph
        self.add_trajectory(trajectory)
        
        return trajectory
    
    def get_neighbors(self, traj_id: str, 
                      max_hops: int = 1) -> List[Tuple[str, int, Set[str]]]:
        """
        Get neighboring trajectories up to max_hops away.
        
        Args:
            traj_id: Starting trajectory ID
            max_hops: Maximum number of hops
        
        Returns:
            List of (trajectory_id, distance, shared_tags) tuples
        """
        if traj_id not in self.graph:
            return []
        
        neighbors = []
        visited = {traj_id}
        current_frontier = {traj_id}
        
        for hop in range(1, max_hops + 1):
            next_frontier = set()
            
            for node in current_frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
                        
                        # Get shared tags with original node
                        if traj_id in self.trajectories and neighbor in self.trajectories:
                            shared = self.trajectories[traj_id].shares_tags_with(
                                self.trajectories[neighbor]
                            )
                        else:
                            shared = set()
                        
                        neighbors.append((neighbor, hop, shared))
            
            current_frontier = next_frontier
        
        return neighbors
    
    def get_trajectories_by_tag(self, tag: str) -> List[TaggedTrajectory]:
        """Get all trajectories with a specific tag."""
        tag = tag.lower().strip()
        if not tag.startswith('#'):
            tag = f'#{tag}'
        
        traj_ids = self.tag_to_trajectories.get(tag, set())
        return [self.trajectories[tid] for tid in traj_ids if tid in self.trajectories]
    
    def get_trajectory(self, traj_id: str) -> Optional[TaggedTrajectory]:
        """Get a trajectory by ID."""
        return self.trajectories.get(traj_id)
    
    def get_all_tags(self) -> Dict[str, int]:
        """Get all tags with their trajectory counts."""
        return {tag: len(trajs) for tag, trajs in self.tag_to_trajectories.items()}
    
    def get_tag_clusters(self) -> List[Set[str]]:
        """
        Get clusters of related tags based on co-occurrence.
        
        Tags that frequently appear together are clustered.
        """
        # Build tag co-occurrence graph
        tag_graph = nx.Graph()
        
        for traj in self.trajectories.values():
            tags = list(traj.tags)
            for i, tag1 in enumerate(tags):
                for tag2 in tags[i+1:]:
                    if tag_graph.has_edge(tag1, tag2):
                        tag_graph[tag1][tag2]['weight'] += 1
                    else:
                        tag_graph.add_edge(tag1, tag2, weight=1)
        
        # Find connected components
        clusters = list(nx.connected_components(tag_graph))
        return sorted(clusters, key=len, reverse=True)
    
    def save(self, filepath: str) -> None:
        """
        Save graph to disk.
        
        Saves:
        - Graph structure (JSON)
        - Trajectory metadata (JSON)
        - Embeddings (NPY)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save graph structure
        # Note: edges="links" preserves current behavior and matches load() call
        graph_data = nx.node_link_data(self.graph, edges="links")
        with open(f"{filepath}_graph.json", 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Save trajectory metadata
        traj_data = {
            traj_id: traj.to_dict() 
            for traj_id, traj in self.trajectories.items()
        }
        with open(f"{filepath}_trajectories.json", 'w') as f:
            json.dump(traj_data, f, indent=2, ensure_ascii=False)
        
        # Save embeddings
        embeddings = {}
        for traj_id, traj in self.trajectories.items():
            if traj.embedding is not None:
                embeddings[traj_id] = traj.embedding
        
        if embeddings:
            # Save as structured array
            ids = list(embeddings.keys())
            emb_array = np.array([embeddings[i] for i in ids])
            np.savez(f"{filepath}_embeddings.npz", 
                    ids=np.array(ids, dtype=object),
                    embeddings=emb_array)
        
        # Save tag mappings
        tag_data = {tag: list(trajs) for tag, trajs in self.tag_to_trajectories.items()}
        with open(f"{filepath}_tags.json", 'w') as f:
            json.dump(tag_data, f, indent=2)
        
        logger.info(f"[GraphBuilder] Saved graph to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load graph from disk.
        """
        filepath = Path(filepath)
        
        # Load graph structure
        with open(f"{filepath}_graph.json", 'r') as f:
            graph_data = json.load(f)
        # Note: edges="links" is the default in networkx 3.x, explicit param added in 3.4+
        self.graph = nx.node_link_graph(graph_data, edges="links")
        
        # Load trajectory metadata
        with open(f"{filepath}_trajectories.json", 'r') as f:
            traj_data = json.load(f)
        
        # Load embeddings if available
        embeddings = {}
        emb_path = Path(f"{filepath}_embeddings.npz")
        if emb_path.exists():
            data = np.load(emb_path, allow_pickle=True)
            ids = data['ids']
            embs = data['embeddings']
            embeddings = {str(ids[i]): embs[i] for i in range(len(ids))}
        
        # Reconstruct trajectories
        self.trajectories = {}
        for traj_id, data in traj_data.items():
            embedding = embeddings.get(traj_id)
            traj = TaggedTrajectory.from_dict(data, embedding=embedding)
            self.trajectories[traj_id] = traj
        
        # Load tag mappings
        with open(f"{filepath}_tags.json", 'r') as f:
            tag_data = json.load(f)
        self.tag_to_trajectories = defaultdict(set)
        for tag, trajs in tag_data.items():
            self.tag_to_trajectories[tag] = set(trajs)
        
        # Consistency checks across persisted artifacts
        graph_nodes = set(self.graph.nodes())
        traj_ids = set(self.trajectories.keys())
        if graph_nodes != traj_ids:
            only_in_graph = sorted(graph_nodes - traj_ids)
            only_in_trajs = sorted(traj_ids - graph_nodes)
            raise ValueError(
                f"Inconsistent loaded graph artifacts for base path {filepath}: "
                f"nodes_only_in_graph={only_in_graph[:10]} "
                f"nodes_only_in_trajectories={only_in_trajs[:10]}"
            )
        for tag, ids in self.tag_to_trajectories.items():
            for tid in ids:
                if tid not in self.trajectories:
                    raise ValueError(
                        f"Inconsistent tag mapping for base path {filepath}: "
                        f"tag {tag!r} references unknown trajectory id {tid!r}"
                    )
        
        # Update stats
        self.stats['total_trajectories'] = len(self.trajectories)
        self.stats['total_edges'] = self.graph.number_of_edges()
        self.stats['total_tags'] = len(self.tag_to_trajectories)
        
        logger.info(f"[GraphBuilder] Loaded graph from {filepath}: "
                   f"{self.stats['total_trajectories']} trajectories, "
                   f"{self.stats['total_edges']} edges, "
                   f"{self.stats['total_tags']} tags")
    
    def visualize(self) -> str:
        """Generate text visualization of the graph."""
        lines = ["=" * 60]
        lines.append("TRAJECTORY GRAPH SUMMARY")
        lines.append("=" * 60)
        
        lines.append(f"\nTrajectories: {self.stats['total_trajectories']}")
        lines.append(f"Edges: {self.stats['total_edges']}")
        lines.append(f"Unique Tags: {self.stats['total_tags']}")
        
        # Top tags
        lines.append("\nTop Tags:")
        tag_counts = sorted(
            [(tag, len(trajs)) for tag, trajs in self.tag_to_trajectories.items()],
            key=lambda x: -x[1]
        )
        for tag, count in tag_counts[:10]:
            lines.append(f"  {tag}: {count} trajectories")
        
        # Most connected trajectories
        lines.append("\nMost Connected Trajectories:")
        degrees = sorted(self.graph.degree(), key=lambda x: -x[1])
        for traj_id, degree in degrees[:5]:
            traj = self.trajectories.get(traj_id)
            takeaway = traj.takeaway[:50] + "..." if traj else "N/A"
            lines.append(f"  {traj_id} ({degree} connections): {takeaway}")
        
        # Deduplication stats (if any dedup occurred)
        dedup_total = self.stats['dedup_added'] + self.stats['dedup_updated'] + self.stats['dedup_replaced']
        if dedup_total > 0:
            lines.append("\n" + "-" * 40)
            lines.append("DEDUPLICATION STATS:")
            lines.append("-" * 40)
            lines.append(f"  Added (new):     {self.stats['dedup_added']}")
            lines.append(f"  Updated:         {self.stats['dedup_updated']}")
            lines.append(f"  Replaced:        {self.stats['dedup_replaced']}")
            lines.append(f"  Total processed: {dedup_total}")
            
            # Calculate reduction
            original_count = self.stats['dedup_added'] + self.stats['dedup_updated'] + self.stats['dedup_replaced']
            final_count = len(self.trajectories)
            if original_count > 0:
                reduction = original_count - final_count
                reduction_pct = 100 * reduction / original_count if original_count > 0 else 0
                lines.append(f"  Reduction:       {reduction} nodes (-{reduction_pct:.1f}%)")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    # =========================================================================
    # Deduplication Methods
    # =========================================================================
    
    def _build_dedup_faiss_index(self) -> None:
        """
        Build FAISS index for deduplication retrieval.
        
        Uses existing trajectory embeddings to build an inner-product index.
        Should be called after loading trajectories or when index needs refresh.
        """
        embeddings = []
        self._dedup_id_to_idx = {}
        self._dedup_idx_to_id = {}
        
        for traj_id, traj in self.trajectories.items():
            if traj.embedding is not None:
                idx = len(embeddings)
                embeddings.append(traj.embedding)
                self._dedup_id_to_idx[traj_id] = idx
                self._dedup_idx_to_id[idx] = traj_id
        
        if not embeddings:
            logger.warning("[GraphBuilder] No embeddings found for FAISS index")
            self._dedup_faiss_index = None
            return
        
        # Build normalized inner-product index
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        
        dim = embeddings_array.shape[1]
        self._dedup_faiss_index = faiss.IndexFlatIP(dim)
        self._dedup_faiss_index.add(embeddings_array)
        
        logger.info(f"[GraphBuilder] Built FAISS dedup index with {len(embeddings)} vectors")
    
    def _compute_dedup_embedding(self, task_description: str, takeaway: str) -> np.ndarray:
        """
        Compute embedding for deduplication by concatenating task + takeaway.
        
        Args:
            task_description: The task description text
            takeaway: The takeaway/summary text
            
        Returns:
            Embedding vector as numpy array
        """
        if self.embedding_model is None:
            raise ValueError("embedding_model is required for dedup embedding computation")
        
        # Concatenate task and takeaway for comprehensive embedding
        text = f"{task_description}\n{takeaway}"
        
        # Use the embedding model's method
        if hasattr(self.embedding_model, 'get_embeddings'):
            embedding = self.embedding_model.get_embeddings([text])
        elif hasattr(self.embedding_model, 'get_multimodal_embeddings'):
            # For multimodal models, use text-only mode
            embedding = self.embedding_model.get_multimodal_embeddings([text], [None])
        else:
            raise ValueError("embedding_model must have get_embeddings or get_multimodal_embeddings method")
        
        if len(embedding.shape) > 1:
            embedding = embedding[0]
        
        return embedding.astype(np.float32)
    
    def _retrieve_similar(
        self,
        embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve similar trajectories using FAISS.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to retrieve
            threshold: Minimum similarity threshold (default: SIMILARITY_THRESHOLD)
            
        Returns:
            List of (trajectory_id, similarity_score) tuples above threshold
        """
        if threshold is None:
            threshold = self.SIMILARITY_THRESHOLD
        
        if self._dedup_faiss_index is None or self._dedup_faiss_index.ntotal == 0:
            return []
        
        # Normalize query
        query = embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        # Search
        n_search = min(k, self._dedup_faiss_index.ntotal)
        scores, indices = self._dedup_faiss_index.search(query, n_search)
        
        # Filter by threshold
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if score < threshold:
                continue
            
            traj_id = self._dedup_idx_to_id.get(int(idx))
            if traj_id is None:
                continue
            
            results.append((traj_id, float(score)))
        
        return results
    
    def _format_image_url(self, image_b64: Optional[str]) -> Optional[str]:
        """Format base64 image as proper data URL."""
        if image_b64 is None:
            return None
        if not image_b64.startswith("data:image"):
            return f"data:image/png;base64,{image_b64}"
        return image_b64
    
    def _vlm_decide(
        self,
        new_traj: TaggedTrajectory,
        similar_nodes: List[Tuple[str, float]],
    ) -> Dict[str, Any]:
        """
        Use VLM to decide: UPDATE, REPLACE, or ADD.
        
        Includes first screenshot from each trajectory for visual context.
        
        Args:
            new_traj: The new trajectory being added
            similar_nodes: List of (traj_id, similarity) for similar existing nodes
            
        Returns:
            Decision dict with keys:
            - action: "update" | "replace" | "add"
            - reasoning: explanation
            - target_id: ID to update (for update)
            - updated_takeaway: new takeaway (for update)
            - updated_tags: new tags list (for update)
            - old_id: ID to replace (for replace)
        """
        if self.llm is None:
            raise ValueError("llm is required for VLM-based deduplication")
        
        # Get new trajectory info
        new_task = new_traj.full_data.get('task_description', '')
        new_takeaway = new_traj.takeaway
        new_tags = sorted(list(new_traj.tags))
        new_domain = new_traj.domain
        new_images = new_traj.full_data.get('images', [])
        new_first_image = new_images[0] if new_images else None
        
        # Build text prompt
        prompt_text = f"""You are managing a knowledge base of GUI agent strategies for deduplication.

NEW TRAJECTORY (see Screenshot 1):
- task_description: {new_task}
- takeaway: {new_takeaway}
- tags: {new_tags}
- domain: {new_domain}

EXISTING SIMILAR TRAJECTORIES:
"""
        # Collect images for multimodal message
        images_for_message = []
        image_labels = []
        
        # Add new trajectory's first image
        if new_first_image:
            images_for_message.append(self._format_image_url(new_first_image))
            image_labels.append("Screenshot 1: NEW trajectory")
        
        for i, (traj_id, sim_score) in enumerate(similar_nodes, 1):
            existing = self.trajectories.get(traj_id)
            if existing is None:
                continue
            
            existing_task = existing.full_data.get('task_description', '')
            existing_takeaway = existing.takeaway
            existing_tags = sorted(list(existing.tags))
            existing_domain = existing.domain
            existing_images = existing.full_data.get('images', [])
            existing_first_image = existing_images[0] if existing_images else None
            
            screenshot_ref = f"Screenshot {len(images_for_message) + 1}" if existing_first_image else "no screenshot"
            
            prompt_text += f"""
{i}. ID: {traj_id} (similarity: {sim_score:.3f}, see {screenshot_ref})
   task_description: {existing_task}
   takeaway: {existing_takeaway}
   tags: {existing_tags}
   domain: {existing_domain}
"""
            # Add existing trajectory's first image
            if existing_first_image:
                images_for_message.append(self._format_image_url(existing_first_image))
                image_labels.append(f"Screenshot {len(images_for_message)}: EXISTING trajectory {traj_id}")
        
        prompt_text += """
DECIDE one of:

1. **UPDATE**: NEW describes the SAME or very similar strategy as an existing trajectory.
   → Keep the existing node but IMPROVE its takeaway by incorporating insights from NEW.
   → Provide the improved takeaway (must start with "takeaway:").
   → Provide the complete updated tags list (can add, remove, or keep tags).

2. **REPLACE**: NEW describes the SAME strategy as an existing trajectory, but NEW is STRICTLY BETTER (more specific, more actionable, more complete trajectory).
   → Remove the existing node, add NEW as replacement.

3. **ADD**: NEW describes a GENUINELY DIFFERENT strategy not covered by existing trajectories.
   → Add NEW as a new node.

Use the screenshots to understand the visual context and UI state of each trajectory.

OUTPUT (JSON only, no markdown):
{
  "action": "update" | "replace" | "add",
  "reasoning": "one sentence explanation",
  "target_id": "ID to update (required for update)",
  "updated_takeaway": "takeaway: improved takeaway text (required for update)",
  "updated_tags": ["#tag1", "#tag2", ...] (required for update, complete new set),
  "old_id": "ID to replace (required for replace)"
}
"""
        
        # Build multimodal message with images
        if images_for_message:
            # Create multimodal content with images first, then text
            user_content = []
            
            # Add image labels as context
            if image_labels:
                user_content.append({
                    "type": "text",
                    "text": "Screenshots reference:\n" + "\n".join(f"- {label}" for label in image_labels) + "\n\n"
                })
            
            # Add images
            for img_url in images_for_message:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url}
                })
            
            # Add main prompt
            user_content.append({
                "type": "text",
                "text": prompt_text
            })
            
            messages = [
                {"role": "system", "content": "You deduplicate GUI agent strategy takeaways using both text and visual context. Output valid JSON only, no markdown code blocks."},
                {"role": "user", "content": user_content}
            ]
        else:
            # No images available - cannot make informed decision
            raise ValueError(
                f"VLM deduplication requires screenshots. "
                f"New trajectory '{new_traj.id}' and similar nodes have no images."
            )
        
        response, _, _ = self.llm.chat(messages=messages, stream=False, temperature=0.0, max_tokens=500)
        
        if not hasattr(response, 'content'):
            raise ValueError("LLM response missing content")
        
        response_text = response.content.strip()
        
        # Try to extract JSON from response
        # Handle potential markdown code blocks
        if '```json' in response_text:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if match:
                response_text = match.group(1)
        elif '```' in response_text:
            match = re.search(r'```\s*([\s\S]*?)\s*```', response_text)
            if match:
                response_text = match.group(1)
        
        # Parse JSON
        decision = json.loads(response_text)
        
        # Validate required fields
        action = decision.get('action', 'add')
        if action not in ('update', 'replace', 'add'):
            logger.warning(f"[GraphBuilder] Invalid VLM action '{action}', defaulting to 'add'")
            decision['action'] = 'add'
        
        return decision
    
    def _update_trajectory(
        self,
        traj_id: str,
        updated_takeaway: str,
        updated_tags: List[str],
    ) -> None:
        """
        Update an existing trajectory's takeaway and tags.
        
        Preserves the original actions/images for continuous memory.
        Recomputes embedding with the updated takeaway.
        
        Args:
            traj_id: ID of trajectory to update
            updated_takeaway: New takeaway text
            updated_tags: Complete new tags list
        """
        if traj_id not in self.trajectories:
            raise ValueError(f"Trajectory {traj_id} not found")
        
        traj = self.trajectories[traj_id]
        old_takeaway = traj.takeaway
        old_tags = set(traj.tags)
        
        # Update takeaway
        traj.takeaway = updated_takeaway
        
        # Update graph node attribute
        if traj_id in self.graph:
            self.graph.nodes[traj_id]['takeaway'] = updated_takeaway
        
        # Remove old tag mappings
        for tag in old_tags:
            if tag in self.tag_to_trajectories:
                self.tag_to_trajectories[tag].discard(traj_id)
                # Clean up empty tag sets
                if not self.tag_to_trajectories[tag]:
                    del self.tag_to_trajectories[tag]
        
        # Set new tags
        new_tags = set()
        for tag in updated_tags:
            tag = tag.lower().strip()
            if not tag.startswith('#'):
                tag = f'#{tag}'
            new_tags.add(tag)
        traj.tags = new_tags
        
        # Add new tag mappings
        for tag in new_tags:
            self.tag_to_trajectories[tag].add(traj_id)
        
        # Update graph node tags
        if traj_id in self.graph:
            self.graph.nodes[traj_id]['tags'] = list(new_tags)
        
        # Recompute embedding with updated takeaway
        if self.embedding_model is not None:
            task_desc = traj.full_data.get('task_description', '')
            new_embedding = self._compute_dedup_embedding(task_desc, updated_takeaway)
            traj.embedding = new_embedding
            
            # Update FAISS index if it exists
            if traj_id in self._dedup_id_to_idx:
                # Rebuild index (FAISS doesn't support in-place updates easily)
                self._build_dedup_faiss_index()
        
        # Update edge weights based on new tags
        self._update_edges_for_trajectory(traj_id)
        
        logger.info(f"[GraphBuilder] Updated {traj_id}:")
        logger.info(f"  Old takeaway: {old_takeaway[:60]}...")
        logger.info(f"  New takeaway: {updated_takeaway[:60]}...")
        logger.info(f"  Tags changed: {old_tags} -> {new_tags}")
    
    def _update_edges_for_trajectory(self, traj_id: str) -> None:
        """Update edges for a trajectory based on its current tags."""
        if traj_id not in self.trajectories:
            return
        
        traj = self.trajectories[traj_id]
        
        # Remove existing edges
        if traj_id in self.graph:
            edges_to_remove = list(self.graph.edges(traj_id))
            self.graph.remove_edges_from(edges_to_remove)
        
        # Add new edges based on shared tags
        for other_id, other_traj in self.trajectories.items():
            if other_id == traj_id:
                continue
            
            shared_tags = traj.shares_tags_with(other_traj)
            
            if len(shared_tags) >= self.min_shared_tags:
                self.graph.add_edge(
                    traj_id,
                    other_id,
                    weight=len(shared_tags),
                    shared_tags=list(shared_tags),
                )
        
        self.stats['total_edges'] = self.graph.number_of_edges()
    
    def _remove_trajectory(self, traj_id: str) -> None:
        """
        Remove a trajectory from the graph and all indexes.
        
        Args:
            traj_id: ID of trajectory to remove
        """
        if traj_id not in self.trajectories:
            logger.warning(f"[GraphBuilder] Trajectory {traj_id} not found for removal")
            return
        
        traj = self.trajectories[traj_id]
        
        # Remove from tag mappings
        for tag in traj.tags:
            if tag in self.tag_to_trajectories:
                self.tag_to_trajectories[tag].discard(traj_id)
                if not self.tag_to_trajectories[tag]:
                    del self.tag_to_trajectories[tag]
        
        # Remove from graph
        if traj_id in self.graph:
            self.graph.remove_node(traj_id)
        
        # Remove from trajectories dict
        del self.trajectories[traj_id]
        
        # Rebuild FAISS index
        if self._dedup_faiss_index is not None:
            self._build_dedup_faiss_index()
        
        # Update stats
        self.stats['total_trajectories'] = len(self.trajectories)
        self.stats['total_edges'] = self.graph.number_of_edges()
        self.stats['total_tags'] = len(self.tag_to_trajectories)
        
        logger.info(f"[GraphBuilder] Removed trajectory {traj_id}")
    
    def add_trajectory_with_dedup(
        self,
        trajectory: TaggedTrajectory,
        threshold: Optional[float] = None,
    ) -> str:
        """
        Add a trajectory with VLM-based deduplication.
        
        Flow:
        1. Compute embedding for new trajectory (task + takeaway)
        2. FAISS top-K retrieval of similar trajectories
        3. Filter by similarity threshold
        4. If no similar found: ADD as new
        5. If similar found: VLM decides UPDATE/REPLACE/ADD
        
        Args:
            trajectory: TaggedTrajectory to add
            threshold: Similarity threshold (default: SIMILARITY_THRESHOLD)
            
        Returns:
            Result string: "added" | "updated:{id}" | "replaced:{id}"
        """
        if threshold is None:
            threshold = self.SIMILARITY_THRESHOLD
        
        traj_id = trajectory.id
        
        # NOTE: Removed ID-based skip for global evolving
        # VLM should always decide whether to add/update/replace/skip
        # if traj_id in self.trajectories:
        #     logger.debug(f"[GraphBuilder] Trajectory {traj_id} already exists, skipping")
        #     return "skipped:exists"
        
        # Compute embedding if not present
        if trajectory.embedding is None and self.embedding_model is not None:
            task_desc = trajectory.full_data.get('task_description', '')
            trajectory.embedding = self._compute_dedup_embedding(task_desc, trajectory.takeaway)
        
        # Build FAISS index if not exists
        if self._dedup_faiss_index is None and len(self.trajectories) > 0:
            self._build_dedup_faiss_index()
        
        # Step 1: Retrieve similar trajectories
        similar_nodes = []
        if trajectory.embedding is not None and self._dedup_faiss_index is not None:
            similar_nodes = self._retrieve_similar(
                embedding=trajectory.embedding,
                k=self.TOP_K_COMPARE,
                threshold=threshold,
            )
        
        # Step 2: If no similar found, add directly
        if not similar_nodes:
            self.add_trajectory(trajectory)
            self._add_to_faiss_index(trajectory)
            self.stats['dedup_added'] += 1
            logger.info(f"[GraphBuilder] Added new trajectory {traj_id} (no similar found)")
            return "added"
        
        # Step 3: VLM decides
        if self.llm is None:
            # No LLM available, add directly
            logger.warning(f"[GraphBuilder] No LLM for dedup, adding {traj_id} directly")
            self.add_trajectory(trajectory)
            self._add_to_faiss_index(trajectory)
            self.stats['dedup_added'] += 1
            return "added"
        
        try:
            decision = self._vlm_decide(trajectory, similar_nodes)
        except Exception as e:
            logger.error(f"[GraphBuilder] VLM decision failed for {traj_id}: {e}")
            # Fallback: add directly
            self.add_trajectory(trajectory)
            self._add_to_faiss_index(trajectory)
            self.stats['dedup_added'] += 1
            return "added"
        
        action = decision.get('action', 'add')
        reasoning = decision.get('reasoning', '')
        
        if action == 'update':
            target_id = decision.get('target_id')
            updated_takeaway = decision.get('updated_takeaway', '')
            updated_tags = decision.get('updated_tags', [])
            
            if target_id and target_id in self.trajectories and updated_takeaway:
                self._update_trajectory(target_id, updated_takeaway, updated_tags)
                self.stats['dedup_updated'] += 1
                logger.info(f"[GraphBuilder] Updated {target_id} with insights from {traj_id}: {reasoning}")
                return f"updated:{target_id}"
            else:
                # Invalid update, fall back to add
                logger.warning(f"[GraphBuilder] Invalid update decision, adding {traj_id}")
                self.add_trajectory(trajectory)
                self._add_to_faiss_index(trajectory)
                self.stats['dedup_added'] += 1
                return "added"
        
        elif action == 'replace':
            old_id = decision.get('old_id')
            
            if old_id and old_id in self.trajectories:
                self._remove_trajectory(old_id)
                self.add_trajectory(trajectory)
                self._add_to_faiss_index(trajectory)
                self.stats['dedup_replaced'] += 1
                logger.info(f"[GraphBuilder] Replaced {old_id} with {traj_id}: {reasoning}")
                return f"replaced:{old_id}"
            else:
                # Invalid replace, fall back to add
                logger.warning(f"[GraphBuilder] Invalid replace decision, adding {traj_id}")
                self.add_trajectory(trajectory)
                self._add_to_faiss_index(trajectory)
                self.stats['dedup_added'] += 1
                return "added"
        
        else:  # action == 'add'
            self.add_trajectory(trajectory)
            self._add_to_faiss_index(trajectory)
            self.stats['dedup_added'] += 1
            logger.info(f"[GraphBuilder] Added {traj_id} (different strategy): {reasoning}")
            return "added"
    
    def _add_to_faiss_index(self, trajectory: TaggedTrajectory) -> None:
        """Add a single trajectory to the FAISS index."""
        if trajectory.embedding is None:
            return
        
        if self._dedup_faiss_index is None:
            # Create new index
            dim = trajectory.embedding.shape[0]
            self._dedup_faiss_index = faiss.IndexFlatIP(dim)
        
        # Add to index
        idx = self._dedup_faiss_index.ntotal
        embedding = trajectory.embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(embedding)
        self._dedup_faiss_index.add(embedding)
        
        self._dedup_id_to_idx[trajectory.id] = idx
        self._dedup_idx_to_id[idx] = trajectory.id
    
    def get_dedup_stats(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        return {
            'total_trajectories': self.stats['total_trajectories'],
            'dedup_added': self.stats['dedup_added'],
            'dedup_updated': self.stats['dedup_updated'],
            'dedup_replaced': self.stats['dedup_replaced'],
        }
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __contains__(self, traj_id: str) -> bool:
        return traj_id in self.trajectories

