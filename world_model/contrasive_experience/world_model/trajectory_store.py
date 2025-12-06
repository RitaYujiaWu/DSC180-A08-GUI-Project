"""
TrajectoryStore: FAISS-indexed storage for trajectories with separate success/failure indices.

Enables contrastive pair retrieval for the World Model.
"""

import os
import sys
import json
import numpy as np
import faiss
import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from glob import glob

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.help_functions import CLIPTextSimilarity, CLIPMultimodalSimilarity

logger = logging.getLogger(__name__)


@dataclass
class StoredTrajectory:
    """A trajectory stored in the trajectory store."""
    file_path: str
    task_description: str
    prefixed_query: str  # Format: "{dataset}_{domain}: {task_description}"
    domain: str
    dataset: str
    is_success: bool
    total_rounds: int
    final_answer: Optional[str] = None
    first_screenshot: Optional[str] = None  # Base64 for multimodal retrieval
    action_sequence: List[str] = field(default_factory=list)

    def load_full_data(self) -> Optional[Dict]:
        """Load full trajectory data from file."""
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading trajectory {self.file_path}: {e}")
            return None


class TrajectoryStore:
    """
    FAISS-indexed storage for trajectory retrieval with separate success/failure indices.

    Key features:
    - Maintains SEPARATE indices for success and failure trajectories
    - Enables contrastive pair retrieval for learning what distinguishes success from failure
    - Supports both text-only and multimodal (text + image) similarity search
    """

    def __init__(
        self,
        training_data_path: str = "training_data",
        faiss_index_path: Optional[str] = None,
        multimodal: bool = True,
        clip_model_name: str = "openai/clip-vit-base-patch32"
    ):
        """
        Initialize the TrajectoryStore.

        Args:
            training_data_path: Path to training data directory
            faiss_index_path: Path to pre-built FAISS index (optional)
            multimodal: Use multimodal embeddings (text + image)
            clip_model_name: CLIP model for embeddings
        """
        self.training_data_path = training_data_path
        self.multimodal = multimodal

        # Initialize CLIP similarity
        if multimodal:
            self.clip_similarity = CLIPMultimodalSimilarity(clip_model_name)
        else:
            self.clip_similarity = CLIPTextSimilarity(clip_model_name)

        # Separate storage for success/failure trajectories
        self.success_trajectories: List[StoredTrajectory] = []
        self.failure_trajectories: List[StoredTrajectory] = []

        # Separate FAISS indices
        self.success_index: Optional[faiss.IndexFlatIP] = None
        self.failure_index: Optional[faiss.IndexFlatIP] = None
        self.success_embeddings: Optional[np.ndarray] = None
        self.failure_embeddings: Optional[np.ndarray] = None

        # Load or build indices
        if faiss_index_path and os.path.exists(f"{faiss_index_path}_success.faiss"):
            logger.info(f"Loading trajectory store from {faiss_index_path}")
            self.load_index(faiss_index_path)
        else:
            logger.info("Building new trajectory store...")
            self._load_trajectories()
            self._build_indices()
            if faiss_index_path:
                self.save_index(faiss_index_path)

    def _load_trajectories(self):
        """
        Load trajectories from training_data directory structure.

        Expected structure:
            training_data/{dataset}/{domain}/success/*.jsonl
            training_data/{dataset}/{domain}/failure/*.jsonl (or negative/)
        """
        logger.info(f"Loading trajectories from: {self.training_data_path}")

        success_count = 0
        failure_count = 0

        for root, dirs, files in os.walk(self.training_data_path):
            for file in files:
                if not file.endswith('.jsonl'):
                    continue

                file_path = os.path.join(root, file)

                # Determine success/failure from path
                is_success = 'success' in root.lower() or 'positive' in root.lower()
                is_failure = 'failure' in root.lower() or 'negative' in root.lower()

                if not (is_success or is_failure):
                    continue  # Skip files not in success/failure dirs

                try:
                    traj = self._parse_trajectory_file(file_path, root, is_success)
                    if traj is None:
                        continue

                    if is_success:
                        self.success_trajectories.append(traj)
                        success_count += 1
                    else:
                        self.failure_trajectories.append(traj)
                        failure_count += 1

                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
                    continue

        logger.info(f"Loaded {success_count} success trajectories")
        logger.info(f"Loaded {failure_count} failure trajectories")

    def _parse_trajectory_file(
        self,
        file_path: str,
        root: str,
        is_success: bool
    ) -> Optional[StoredTrajectory]:
        """Parse a single trajectory file into StoredTrajectory."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        task_desc = data.get('task_description', '')
        if not task_desc:
            return None

        total_rounds = data.get('total_rounds', len(data.get('rounds', [])))

        # Skip trajectories that are too short or too long
        if total_rounds < 2 or total_rounds > 20:
            return None

        # Parse path for domain/dataset
        domain, dataset = self._extract_domain_dataset(root)

        # Extract first screenshot for multimodal
        first_screenshot = self._extract_first_screenshot(data)

        # Extract action sequence
        action_sequence = self._extract_action_sequence(data)

        # Extract final answer
        final_answer = self._extract_final_answer(data)

        # Create prefixed query like original memory format
        prefixed_query = f"{dataset}_{domain}: {task_desc}" if dataset and domain else task_desc

        return StoredTrajectory(
            file_path=file_path,
            task_description=task_desc,
            prefixed_query=prefixed_query,
            domain=domain,
            dataset=dataset,
            is_success=is_success,
            total_rounds=total_rounds,
            final_answer=final_answer,
            first_screenshot=first_screenshot,
            action_sequence=action_sequence
        )

    def _extract_domain_dataset(self, root: str) -> Tuple[str, str]:
        """Extract domain and dataset from file path."""
        path_parts = root.split(os.sep)
        domain = ''
        dataset = ''

        known_datasets = ['mmina', 'webvoyager', 'mind2web']

        for i, part in enumerate(path_parts):
            if part.lower() in known_datasets:
                dataset = part
                if i + 1 < len(path_parts) and path_parts[i + 1] not in ['success', 'failure', 'positive', 'negative']:
                    domain = path_parts[i + 1]
                break

        return domain, dataset

    def _extract_first_screenshot(self, data: Dict) -> Optional[str]:
        """Extract first screenshot from trajectory for multimodal embedding."""
        rounds = data.get('rounds', [])
        if not rounds:
            return None

        first_round = rounds[0]
        for msg in first_round.get('messages', []):
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    if item.get('type') == 'image_url':
                        return item['image_url']['url']
        return None

    def _extract_action_sequence(self, data: Dict) -> List[str]:
        """Extract action names from trajectory."""
        actions = []
        rounds = data.get('rounds', [])

        for round_data in rounds:
            response = round_data.get('response', '')
            if isinstance(response, list):
                response = response[0] if response else ''
            if isinstance(response, dict):
                response = response.get('content', str(response))

            action_name = self._parse_action_name(response)
            if action_name:
                actions.append(action_name)

        return actions

    def _parse_action_name(self, response: str) -> Optional[str]:
        """Parse action name from response."""
        try:
            # Try various patterns
            patterns = [
                r'"name"\s*:\s*"([^"]+)"',
                r'"action"\s*:\s*"([^"]+)"',
                r'"action_type"\s*:\s*"([^"]+)"'
            ]

            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group(1)
        except:
            pass
        return None

    def _extract_final_answer(self, data: Dict) -> Optional[str]:
        """Extract final answer from trajectory."""
        rounds = data.get('rounds', [])
        if not rounds:
            return None

        last_response = rounds[-1].get('response', '')
        if isinstance(last_response, list):
            last_response = last_response[0] if last_response else ''
        if isinstance(last_response, dict):
            last_response = last_response.get('content', str(last_response))

        # Try to parse stop action answer
        try:
            match = re.search(r'"answer"\s*:\s*"([^"]*)"', last_response)
            if match:
                return match.group(1)
        except:
            pass

        return last_response[:200] if last_response else None

    def _build_indices(self):
        """Build separate FAISS indices for success and failure trajectories."""
        logger.info("Building FAISS indices...")

        # Build success index
        if self.success_trajectories:
            self.success_embeddings = self._compute_embeddings(self.success_trajectories)
            self.success_index = self._create_faiss_index(self.success_embeddings)
            logger.info(f"Built success index with {self.success_index.ntotal} vectors")

        # Build failure index
        if self.failure_trajectories:
            self.failure_embeddings = self._compute_embeddings(self.failure_trajectories)
            self.failure_index = self._create_faiss_index(self.failure_embeddings)
            logger.info(f"Built failure index with {self.failure_index.ntotal} vectors")

    def _compute_embeddings(self, trajectories: List[StoredTrajectory]) -> np.ndarray:
        """Compute embeddings for a list of trajectories using prefixed queries."""
        # Use prefixed query format like original memory: "{dataset}_{domain}: {task}"
        texts = [t.prefixed_query for t in trajectories]

        if self.multimodal:
            images = [t.first_screenshot for t in trajectories]
            embeddings = self.clip_similarity.get_multimodal_embeddings(texts, images)
        else:
            embeddings = self.clip_similarity.get_text_embeddings(texts)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        return embeddings

    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Create FAISS index from embeddings."""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype('float32'))
        return index

    def _get_query_embedding(
        self,
        query_task: str,
        query_image: Optional[str] = None
    ) -> np.ndarray:
        """Get embedding for a query."""
        if self.multimodal:
            if query_image is not None:
                embedding = self.clip_similarity.get_multimodal_embeddings([query_task], [query_image])
            else:
                # Create multimodal embedding with zero image part
                text_emb = self.clip_similarity.get_text_embeddings([query_task])
                zero_img = np.zeros_like(text_emb)
                embedding = np.concatenate([text_emb, zero_img], axis=1)
        else:
            embedding = self.clip_similarity.get_text_embeddings([query_task])

        faiss.normalize_L2(embedding)
        return embedding

    def retrieve_contrastive_pairs(
        self,
        query_task: str,
        query_image: Optional[str] = None,
        domain: Optional[str] = None,
        dataset: Optional[str] = None,
        top_k: int = 3
    ) -> Tuple[List[StoredTrajectory], List[StoredTrajectory]]:
        """
        Retrieve similar success and failure trajectories for contrastive learning.

        Args:
            query_task: Task description to query
            query_image: Optional base64 screenshot
            domain: Optional domain filter
            dataset: Optional dataset filter (e.g., 'mmina', 'webvoyager')
            top_k: Number of trajectories per category

        Returns:
            Tuple of (success_trajectories, failure_trajectories)
        """
        # Format query with prefix like original memory: "{dataset}_{domain}: {task}"
        if dataset and domain:
            prefixed_query = f"{dataset}_{domain}: {query_task}"
        else:
            prefixed_query = query_task

        query_embedding = self._get_query_embedding(prefixed_query, query_image)

        success_results = self._search_index(
            query_embedding,
            self.success_index,
            self.success_trajectories,
            domain,
            top_k
        )

        failure_results = self._search_index(
            query_embedding,
            self.failure_index,
            self.failure_trajectories,
            domain,
            top_k
        )

        return success_results, failure_results

    def _search_index(
        self,
        query_embedding: np.ndarray,
        index: Optional[faiss.IndexFlatIP],
        trajectories: List[StoredTrajectory],
        domain: Optional[str],
        top_k: int
    ) -> List[StoredTrajectory]:
        """Search a single FAISS index with optional domain filtering."""
        if index is None or not trajectories:
            return []

        # Search more than needed to allow for filtering
        search_k = top_k * 3
        scores, indices = index.search(query_embedding.astype('float32'), search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            traj = trajectories[idx]

            # Apply domain filter
            if domain and traj.domain.lower() != domain.lower():
                continue

            results.append(traj)

            if len(results) >= top_k:
                break

        return results

    def retrieve_success_only(
        self,
        query_task: str,
        query_image: Optional[str] = None,
        domain: Optional[str] = None,
        dataset: Optional[str] = None,
        top_k: int = 3
    ) -> List[StoredTrajectory]:
        """Retrieve only success trajectories (for fallback mode)."""
        # Format query with prefix like original memory
        if dataset and domain:
            prefixed_query = f"{dataset}_{domain}: {query_task}"
        else:
            prefixed_query = query_task

        query_embedding = self._get_query_embedding(prefixed_query, query_image)
        return self._search_index(
            query_embedding,
            self.success_index,
            self.success_trajectories,
            domain,
            top_k
        )

    def save_index(self, path: str):
        """Save indices and metadata to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Save success index
        if self.success_index:
            faiss.write_index(self.success_index, f"{path}_success.faiss")
            np.save(f"{path}_success_emb.npy", self.success_embeddings)

        # Save failure index
        if self.failure_index:
            faiss.write_index(self.failure_index, f"{path}_failure.faiss")
            np.save(f"{path}_failure_emb.npy", self.failure_embeddings)

        # Save trajectory metadata
        metadata = {
            'success': [
                {
                    'file_path': t.file_path,
                    'task_description': t.task_description,
                    'prefixed_query': t.prefixed_query,
                    'domain': t.domain,
                    'dataset': t.dataset,
                    'total_rounds': t.total_rounds,
                    'final_answer': t.final_answer,
                    'action_sequence': t.action_sequence
                }
                for t in self.success_trajectories
            ],
            'failure': [
                {
                    'file_path': t.file_path,
                    'task_description': t.task_description,
                    'prefixed_query': t.prefixed_query,
                    'domain': t.domain,
                    'dataset': t.dataset,
                    'total_rounds': t.total_rounds,
                    'final_answer': t.final_answer,
                    'action_sequence': t.action_sequence
                }
                for t in self.failure_trajectories
            ],
            'multimodal': self.multimodal
        }

        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved trajectory store to {path}")

    def load_index(self, path: str):
        """Load indices and metadata from disk."""
        # Load metadata
        with open(f"{path}_metadata.json", 'r') as f:
            metadata = json.load(f)

        # Reconstruct success trajectories
        for item in metadata.get('success', []):
            # Reconstruct prefixed_query if not present (backward compatibility)
            prefixed_query = item.get('prefixed_query')
            if not prefixed_query:
                dataset = item.get('dataset', '')
                domain = item.get('domain', '')
                task = item['task_description']
                prefixed_query = f"{dataset}_{domain}: {task}" if dataset and domain else task

            self.success_trajectories.append(StoredTrajectory(
                file_path=item['file_path'],
                task_description=item['task_description'],
                prefixed_query=prefixed_query,
                domain=item['domain'],
                dataset=item['dataset'],
                is_success=True,
                total_rounds=item.get('total_rounds', 0),
                final_answer=item.get('final_answer'),
                action_sequence=item.get('action_sequence', [])
            ))

        # Reconstruct failure trajectories
        for item in metadata.get('failure', []):
            # Reconstruct prefixed_query if not present (backward compatibility)
            prefixed_query = item.get('prefixed_query')
            if not prefixed_query:
                dataset = item.get('dataset', '')
                domain = item.get('domain', '')
                task = item['task_description']
                prefixed_query = f"{dataset}_{domain}: {task}" if dataset and domain else task

            self.failure_trajectories.append(StoredTrajectory(
                file_path=item['file_path'],
                task_description=item['task_description'],
                prefixed_query=prefixed_query,
                domain=item['domain'],
                dataset=item['dataset'],
                is_success=False,
                total_rounds=item.get('total_rounds', 0),
                final_answer=item.get('final_answer'),
                action_sequence=item.get('action_sequence', [])
            ))

        # Load success index
        if os.path.exists(f"{path}_success.faiss"):
            self.success_index = faiss.read_index(f"{path}_success.faiss")
            self.success_embeddings = np.load(f"{path}_success_emb.npy")

        # Load failure index
        if os.path.exists(f"{path}_failure.faiss"):
            self.failure_index = faiss.read_index(f"{path}_failure.faiss")
            self.failure_embeddings = np.load(f"{path}_failure_emb.npy")

        logger.info(f"Loaded {len(self.success_trajectories)} success, {len(self.failure_trajectories)} failure trajectories")

    def get_stats(self) -> Dict:
        """Get statistics about the trajectory store."""
        return {
            'success_count': len(self.success_trajectories),
            'failure_count': len(self.failure_trajectories),
            'total_count': len(self.success_trajectories) + len(self.failure_trajectories),
            'multimodal': self.multimodal,
            'domains': list(set(
                [t.domain for t in self.success_trajectories] +
                [t.domain for t in self.failure_trajectories]
            )),
            'datasets': list(set(
                [t.dataset for t in self.success_trajectories] +
                [t.dataset for t in self.failure_trajectories]
            ))
        }
