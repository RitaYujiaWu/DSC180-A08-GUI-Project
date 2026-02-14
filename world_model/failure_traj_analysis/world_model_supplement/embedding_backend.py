from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import numpy as np

# Optional deps:
# - faiss
# - torch
# - transformers


class Embedder:
    """
    Abstract embedder interface.
    Returns L2-normalized float32 vectors for FAISS inner product search.
    """

    def embed_query(self, text: str, image=None) -> Any:
        raise NotImplementedError

    def embed_trajectory_key(self, text: str, image=None) -> Any:
        return self.embed_query(text=text, image=image)

    def build_faiss_index(self, vectors: Any) -> Any:
        raise NotImplementedError

    def search(self, index: Any, qvec: Any, top_k: int) -> Tuple[List[float], List[int]]:
        raise NotImplementedError

    def stack(self, vecs: List[Any]) -> Any:
        return np.stack(vecs, axis=0).astype("float32")


@dataclass
class ClipEmbedder(Embedder):
    device: str = "cpu"
    model_name: str = "openai/clip-vit-base-patch32"

    def __post_init__(self) -> None:
        try:
            import torch  # noqa
            from transformers import CLIPModel, CLIPProcessor  # noqa
        except Exception as e:
            raise ImportError(
                "CLIP embedder requires torch + transformers. "
                "Install them (and faiss) to enable retrieval."
            ) from e

        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def _l2norm(self, x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
        return (x / n).astype("float32")

    def embed_query(self, text: str, image=None) -> np.ndarray:
        torch = self.torch

        inputs = self.processor(
            text=[text],
            images=[image] if image is not None else None,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_feat = self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            text_vec = text_feat.detach().cpu().numpy()

            if image is not None and "pixel_values" in inputs:
                img_feat = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                img_vec = img_feat.detach().cpu().numpy()
                vec = np.concatenate([text_vec, img_vec], axis=-1)
            else:
                vec = text_vec

        return self._l2norm(vec)[0]

    def build_faiss_index(self, vectors: np.ndarray) -> Any:
        try:
            import faiss  
        except Exception as e:
            raise ImportError("faiss is required for retrieval. Install faiss-cpu or faiss-gpu.") from e

        import faiss
        vectors = vectors.astype("float32")
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return index

    def search(self, index: Any, qvec: np.ndarray, top_k: int) -> Tuple[List[float], List[int]]:
        q = qvec.reshape(1, -1).astype("float32")
        scores, idxs = index.search(q, top_k)
        return scores[0].tolist(), idxs[0].tolist()


@dataclass
class LatentEmbedderStub(Embedder):
    """
    Placeholder for Q2: learned contrastive encoder.
    Keeps the same interface so you can swap in later.

    For now, it falls back to a deterministic hashing-based embedding
    (NOT semantically meaningful) to keep code runnable in minimal envs.
    """
    dim: int = 512

    def _l2norm(self, x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
        return (x / n).astype("float32")

    def _hash_embed(self, s: str) -> np.ndarray:
        import hashlib
        h = hashlib.sha256(s.encode("utf-8")).digest()
        # expand deterministically
        rng = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        reps = int(np.ceil(self.dim / rng.shape[0]))
        v = np.tile(rng, reps)[: self.dim]
        return self._l2norm(v)

    def embed_query(self, text: str, image=None) -> np.ndarray:
        return self._hash_embed(text)

    def build_faiss_index(self, vectors: Any) -> Any:
        try:
            import faiss  
        except Exception as e:
            raise ImportError("faiss is required for retrieval. Install faiss-cpu or faiss-gpu.") from e
        import faiss
        vectors = np.asarray(vectors, dtype="float32")
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return index

    def search(self, index: Any, qvec: Any, top_k: int) -> Tuple[List[float], List[int]]:
        q = np.asarray(qvec, dtype="float32").reshape(1, -1)
        scores, idxs = index.search(q, top_k)
        return scores[0].tolist(), idxs[0].tolist()


def create_embedder(backend: str = "clip", device: str = "cpu") -> Embedder:
    backend = (backend or "clip").lower().strip()
    if backend == "clip":
        return ClipEmbedder(device=device)
    if backend == "latent":
        return LatentEmbedderStub()
    raise ValueError(f"Unknown embed backend: {backend}. Use 'clip' or 'latent'.")
