from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


class VectorIndex:
    """
    Generic FAISS-backed index for dense retrieval vectors.

    Used for both phase-level and trajectory-level embeddings. The index
    itself is agnostic to what each vector represents; callers associate
    semantic meaning via `ids` and `meta`.

    This implementation now *requires* FAISS. If FAISS is not installed,
    constructing a VectorIndex will raise an ImportError instead of silently
    falling back to another implementation.

    Persists:
      - vectors.faiss
      - vectors.ids.json
      - vectors.meta.json
    """

    def __init__(self, dim: int):
        if not _HAS_FAISS:
            raise ImportError(
                "FAISS is required for VectorIndex but is not installed. "
                "Please install faiss (e.g., `pip install faiss-cpu` or "
                "`pip install faiss-gpu`) and rebuild the index."
            )
        self.dim = dim
        self.ids: List[str] = []
        self.meta: Dict[str, Dict] = {}
        # We use inner-product search with normalized embeddings, which is
        # equivalent to cosine similarity.
        self.index = faiss.IndexFlatIP(dim)

    def add(self, item_id: str, vec: np.ndarray, meta: Optional[Dict] = None) -> None:
        """Add a single vector and its metadata to the index."""
        vec = vec.astype(np.float32).reshape(1, -1)
        self.index.add(vec)
        self.ids.append(item_id)
        if meta:
            self.meta[item_id] = meta

    def save(self, out_dir: str) -> None:
        """Persist the FAISS index, ids, and metadata to disk."""
        os.makedirs(out_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(out_dir, "vectors.faiss"))
        with open(os.path.join(out_dir, "vectors.ids.json"), "w") as f:
            json.dump(self.ids, f)
        with open(os.path.join(out_dir, "vectors.meta.json"), "w") as f:
            json.dump(self.meta, f)

    @classmethod
    def load(cls, in_dir: str) -> "VectorIndex":
        """Load an index that was previously saved via `save`."""
        if not _HAS_FAISS:
            raise ImportError(
                "FAISS is required for VectorIndex but is not installed. "
                "Please install faiss (e.g., `pip install faiss-cpu` or "
                "`pip install faiss-gpu`) and rebuild/load the index."
            )
        with open(os.path.join(in_dir, "vectors.ids.json"), "r") as f:
            ids = json.load(f)
        with open(os.path.join(in_dir, "vectors.meta.json"), "r") as f:
            meta = json.load(f)

        index_path = os.path.join(in_dir, "vectors.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Existing non-FAISS indices are no longer supported; "
                "please rebuild the memory index with FAISS enabled."
            )
        index = faiss.read_index(index_path)
        dim = index.d
        obj = cls(dim)
        obj.index = index
        obj.ids = ids
        obj.meta = meta
        return obj

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if len(self.ids) == 0:
            return []
        q = query_vec.astype(np.float32).reshape(1, -1)
        scores, idxs = self.index.search(q, min(k, len(self.ids)))
        idxs = idxs[0]
        scores = scores[0]
        results: List[Tuple[str, float]] = []
        for i, s in zip(idxs, scores):
            if i < 0 or i >= len(self.ids):
                continue
            results.append((self.ids[i], float(s)))
        return results


class DiskKV:
    """
    Simple disk-backed KV for latent_pack paths or auxiliary metadata.
    Persists to kv.json under the directory.
    """
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.file_path = os.path.join(dir_path, "kv.json")
        os.makedirs(dir_path, exist_ok=True)
        self.kv: Dict[str, str] = {}
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    self.kv = json.load(f)
            except Exception:
                self.kv = {}

    def set(self, key: str, value: str) -> None:
        self.kv[key] = value
        with open(self.file_path, "w") as f:
            json.dump(self.kv, f)

    def get(self, key: str) -> Optional[str]:
        return self.kv.get(key)


