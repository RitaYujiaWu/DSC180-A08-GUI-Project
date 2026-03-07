"""
Encoder module: Generate search vectors and latent packs for all memory levels.

Supports:
- Phase encoding: text summary + keyframe images → search vector + latent pack
- Trajectory encoding: task description + phase aggregation → search vector + optional latent pack
- Domain encoding: domain descriptor → search vector (for routing/gating)

Search vectors are used for fast FAISS retrieval.
Latent packs store continuous embeddings for model injection.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

from .schema import Domain, PhaseNote, Trajectory

from transformers import CLIPModel, CLIPProcessor


# =============================================================================
# Base Encoder (Text + Image utilities)
# =============================================================================

class BaseEncoder:
    """Base class with shared text/image encoding utilities using CLIP."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.model = None
        self.processor = None

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()
    
    # -------------------------------------------------------------------------
    # Image I/O
    # -------------------------------------------------------------------------
    
    @staticmethod
    def read_image_from_path(path: str) -> Optional[Image.Image]:
        """Load image from file path."""
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    
    @staticmethod
    def read_image_from_b64(b64: str) -> Optional[Image.Image]:
        """Decode base64 string to PIL Image."""
        try:
            if b64.startswith("data:image"):
                b64 = b64.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            return None
    
    # -------------------------------------------------------------------------
    # Encoding
    # -------------------------------------------------------------------------
    
    def _get_device(self):
        return self.model.device if self.model else "cpu"

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to vector using CLIP."""
        if self.model is not None and self.processor is not None:
            try:
                inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self._get_device())
                with torch.no_grad():
                    text_features = self.model.get_text_features(**inputs)
                # Normalize
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                return text_features.cpu().numpy()[0].astype(np.float32)
            except Exception as e:
                print(f"Error encoding text with CLIP: {e}")
                pass
        
        # Deterministic fallback: hash to fixed random vector
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.normal(0, 1, size=(512,)).astype(np.float32) # CLIP base is 512
    
    def encode_image(
        self, 
        img: Image.Image, 
        size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Encode image to vector using CLIP.
        """
        if self.model is not None and self.processor is not None:
            try:
                inputs = self.processor(images=img, return_tensors="pt").to(self._get_device())
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                # Normalize
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                return image_features.cpu().numpy()[0].astype(np.float32)
            except Exception as e:
                print(f"Error encoding image with CLIP: {e}")
                pass

        # Fallback
        return np.zeros((512,), dtype=np.float32)
    
    # -------------------------------------------------------------------------
    # Fusion
    # -------------------------------------------------------------------------
    
    def fuse_text_and_images(
        self,
        text_vec: np.ndarray,
        image_vecs: List[np.ndarray],
    ) -> np.ndarray:
        """
        Fuse text and image vectors into a single retrieval vector via concatenation.
        
        Current strategy (aligned with trajectory/phase encoders):
        - If at least one image vector is provided, use ONLY the first one.
        - If no images are provided, use a zero vector for the image part.
        - Concatenate text_vec and image_vec, then L2-normalize.
        """
        if len(image_vecs) == 0:
            # If no images, use zero vector of same dim
            img_vec = np.zeros_like(text_vec)
        else:
            # Use only the first image vector. We keep the list interface for
            # flexibility, but upstream code (trajectory/phase encoders,
            # query encoders) now pass at most one image.
            img_vec = image_vecs[0].astype(np.float32)
            # Ensure it is normalized.
            norm = np.linalg.norm(img_vec) + 1e-8
            img_vec = img_vec / norm
        
        # Concatenate
        fused = np.concatenate([text_vec, img_vec], axis=0)
        
        # L2-normalize final vector
        norm = np.linalg.norm(fused) + 1e-8
        fused = fused / norm
        
        return fused.astype(np.float32)
    
    @staticmethod
    def normalize(vec: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        norm = np.linalg.norm(vec) + 1e-8
        return (vec / norm).astype(np.float32)


# =============================================================================
# Phase Encoder
# =============================================================================

class PhaseEncoder(BaseEncoder):
    """
    Encode PhaseNote objects.
    
    Produces:
    - Search vector: For FAISS retrieval (Concatenated Text+Image)
    - Latent pack: For model injection (saved as .npz)
    """
    
    def encode(
        self,
        phase: PhaseNote,
        output_dir: str,
    ) -> Tuple[np.ndarray, str]:
        """
        Encode a PhaseNote to search vector and latent pack.
        
        Args:
            phase: PhaseNote object
            output_dir: Directory to save latent pack
        
        Returns:
            (retrieval_vec, latent_pack_path)
        """
        return self.encode_from_parts(
            phase_id=phase.id,
            task_description=None,
            summary_text=phase.summary,
            keyframe_paths=phase.keyframe_paths or [],
            output_dir=output_dir,
            meta={
                "trajectory_id": phase.trajectory_id,
                "label": phase.phase_label,
                "start": phase.start_step,
                "end": phase.end_step,
                "domain": phase.domain,
            },
        )
    
    def encode_from_parts(
        self,
        phase_id: str,
        task_description: Optional[str],
        summary_text: str,
        keyframe_paths: List[str],
        output_dir: str,
        meta: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Encode phase from individual components.
        
        Args:
            phase_id: Unique phase identifier
            task_description: The parent trajectory's task description (intent)
            summary_text: Phase summary text
            keyframe_paths: Paths to keyframe images
            output_dir: Directory to save latent pack
            meta: Optional metadata
        
        Returns:
            (retrieval_vec, latent_pack_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Encode text (task intent + phase summary)
        #
        # We include the parent task_description to better align phase vectors
        # with retrieval queries, which are constructed from (intent + page).
        text_parts: List[str] = []
        if task_description:
            task = task_description.strip()
            if task:
                text_parts.append(f"Task: {task}")
        if summary_text:
            summ = summary_text.strip()
            if summ:
                label = None
                if isinstance(meta, dict):
                    label = meta.get("label")
                if label:
                    text_parts.append(f"Phase ({label}): {summ}")
                else:
                    text_parts.append(f"Phase: {summ}")
        text = "\n".join(text_parts)
        text_vec = self.encode_text(text)
        
        # Encode images (phases now use only the first keyframe image,
        # aligned with trajectory embeddings which use the first frame).
        image_vecs: List[np.ndarray] = []
        if keyframe_paths:
            img = self.read_image_from_path(keyframe_paths[0])
            if img is not None:
                image_vecs.append(self.encode_image(img))
        
        # Fuse
        retrieval_vec = self.fuse_text_and_images(text_vec, image_vecs)
        
        # Build latent pack
        latent_pack = {
            "phase_id": phase_id,
            "task_description": task_description,
            "summary_text": summary_text,
            "text_for_embedding": text,
            "keyframe_paths": keyframe_paths[:2],
            "model_name": self.model_name,
            "vec_dim": int(retrieval_vec.shape[0]),
            "meta": meta or {},
        }
        
        latent_pack_path = os.path.join(output_dir, f"{phase_id}.npz")
        np.savez_compressed(
            latent_pack_path,
            retrieval_vec=retrieval_vec.astype(np.float16),
            meta_json=np.frombuffer(json.dumps(latent_pack).encode("utf-8"), dtype=np.uint8),
        )
        
        return retrieval_vec, latent_pack_path
    
    def encode_query(
        self,
        intent_text: str,
        page_description: str,
        image_b64: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode a query for phase retrieval.
        
        Args:
            intent_text: User intent/task description
            page_description: Current page description
            image_b64: Optional screenshot as base64
            image_path: Optional screenshot file path
        
        Returns:
            Query vector for similarity search
        """
        text_parts: List[str] = []
        intent = (intent_text or "").strip()
        if intent:
            text_parts.append(f"Task: {intent}")
        page = (page_description or "").strip()
        if page:
            text_parts.append(f"Page: {page}")
        text = "\n".join(text_parts)
        text_vec = self.encode_text(text)
        
        image_vecs: List[np.ndarray] = []
        if image_b64:
            img = self.read_image_from_b64(image_b64)
            if img is not None:
                image_vecs.append(self.encode_image(img))
        if image_path and len(image_vecs) == 0:
            img = self.read_image_from_path(image_path)
            if img is not None:
                image_vecs.append(self.encode_image(img))
        
        return self.fuse_text_and_images(text_vec, image_vecs)


# =============================================================================
# Trajectory Encoder
# =============================================================================

class TrajectoryEncoder(BaseEncoder):
    """
    Encode Trajectory objects.

    This produces a single retrieval vector for trajectory-level similarity search.
    """
    
    def encode_embedding(
        self,
        trajectory: Trajectory,
        keyframe_paths: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Encode a trajectory embedding from task description + optional summary and keyframe(s).
        
        Args:
            trajectory: Trajectory object
            keyframe_paths: Optional list of image paths (e.g. start and end screenshots)
        
        Returns:
            Trajectory search vector
        """
        # Combine task description and summary
        text_parts = []
        if trajectory.task_description:
            text_parts.append(f"Task: {trajectory.task_description}")
        if trajectory.summary:
            text_parts.append(f"Summary: {trajectory.summary}")
        
        text = "\n".join(text_parts) if text_parts else "Unknown trajectory"
        text_vec = self.encode_text(text)
        
        # Encode images if available (start/end frames)
        image_vecs: List[np.ndarray] = []
        if keyframe_paths:
            for p in keyframe_paths:
                img = self.read_image_from_path(p)
                if img is not None:
                    image_vecs.append(self.encode_image(img))
        
        # Fuse text + images
        # If no images, fuse_text_and_images handles zero-padding
        return self.fuse_text_and_images(text_vec, image_vecs)
    
    def encode_query(
        self,
        task_description: str,
        domain: Optional[str] = None,
        image_b64: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode a query for trajectory retrieval.
        
        Args:
            task_description: User's task description
            domain: Optional domain for context
            image_b64: Optional current screenshot (representing start state match)
        
        Returns:
            Query vector for trajectory search
        """
        text_parts: List[str] = []
        task = (task_description or "").strip()
        if task:
            text_parts.append(f"Task: {task}")
        if domain:
            dom = domain.strip()
            if dom:
                text_parts.append(f"Domain: {dom}")
        text = "\n".join(text_parts)
        
        text_vec = self.encode_text(text)
        
        image_vecs: List[np.ndarray] = []
        if image_b64:
            img = self.read_image_from_b64(image_b64)
            if img is not None:
                image_vecs.append(self.encode_image(img))
        
        return self.fuse_text_and_images(text_vec, image_vecs)


# =============================================================================
# Domain Encoder
# =============================================================================

class DomainEncoder(BaseEncoder):
    """
    Encode Domain objects for routing/gating.
    
    Domain vectors are used for:
    - Filtering/partitioning the search space
    - Soft gating (similarity-based domain selection)
    """
    
    def encode(self, domain: Domain) -> np.ndarray:
        """
        Encode a domain to a search vector.
        
        Args:
            domain: Domain object
        
        Returns:
            Domain search vector
        """
        # Build simple domain descriptor (we keep domain-level modeling minimal for now)
        text = f"Domain: {domain.domain}" if domain.domain else "Unknown domain"
        text_vec = self.encode_text(text)
        # Pad for dimension consistency (if we ever search domain vs trajectory)
        # Usually domain search is its own space, but keeping it consistent doesn't hurt.
        # Actually, for domain-level search, we might just want text similarity.
        # But BaseEncoder is built on CLIP text features.
        return self.normalize(text_vec)
    
    def encode_query(
        self,
        domain_name: Optional[str] = None,
        site_host: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode a query for domain matching.
        
        Args:
            domain_name: Domain name (e.g., "shopping")
            site_host: (ignored) kept for backward compatibility
        
        Returns:
            Query vector for domain matching
        """
        # For now we keep domain queries simple and ignore site_host
        text = f"Domain: {domain_name}" if domain_name else "general"
        text_vec = self.encode_text(text)
        
        return self.normalize(text_vec)


# =============================================================================
# Unified Memory Encoder
# =============================================================================

class MemoryEncoder:
    """
    Unified encoder for all memory levels using CLIP-based embeddings.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        # Shared base encoder instance to avoid loading model multiple times?
        # Actually BaseEncoder loads it. To share, we can pass it or let them load individually (wasteful).
        # Let's initialize one BaseEncoder core logic or let them inherit.
        # Since they inherit, they each init their own model. This is bad for VRAM.
        # Better: Initialize one set of model/processor and share.
        
        self.model = None
        self.processor = None
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

        self.phase_encoder = PhaseEncoder(model_name)
        self.trajectory_encoder = TrajectoryEncoder(model_name)
        self.domain_encoder = DomainEncoder(model_name)
        
        # Share the loaded model to save memory
        if self.model:
            self.phase_encoder.model = self.model
            self.phase_encoder.processor = self.processor
            self.trajectory_encoder.model = self.model
            self.trajectory_encoder.processor = self.processor
            self.domain_encoder.model = self.model
            self.domain_encoder.processor = self.processor
    
    # -------------------------------------------------------------------------
    # Phase
    # -------------------------------------------------------------------------
    
    def encode_phase(
        self,
        phase: PhaseNote,
        output_dir: str,
    ) -> Tuple[np.ndarray, str]:
        """Encode a PhaseNote."""
        return self.phase_encoder.encode(phase, output_dir)
    
    def encode_phase_from_parts(
        self,
        phase_id: str,
        task_description: Optional[str],
        summary_text: str,
        keyframe_paths: List[str],
        output_dir: str,
        meta: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, str]:
        """Encode phase from components."""
        return self.phase_encoder.encode_from_parts(
            phase_id, task_description, summary_text, keyframe_paths, output_dir, meta
        )
    
    def encode_phase_query(
        self,
        intent_text: str,
        page_description: str,
        image_b64: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> np.ndarray:
        """Encode query for phase retrieval."""
        return self.phase_encoder.encode_query(
            intent_text, page_description, image_b64, image_path
        )
    
    # -------------------------------------------------------------------------
    # Trajectory
    # -------------------------------------------------------------------------
    
    def encode_trajectory_embedding(
        self, 
        trajectory: Trajectory,
        keyframe_paths: Optional[List[str]] = None
    ) -> np.ndarray:
        """Encode a trajectory embedding from text + optional keyframes."""
        return self.trajectory_encoder.encode_embedding(trajectory, keyframe_paths)
    
    def encode_trajectory_query(
        self,
        task_description: str,
        domain: Optional[str] = None,
        image_b64: Optional[str] = None,
    ) -> np.ndarray:
        """Encode query for trajectory retrieval."""
        return self.trajectory_encoder.encode_query(task_description, domain, image_b64)
    
    # -------------------------------------------------------------------------
    # Domain
    # -------------------------------------------------------------------------
    
    def encode_domain(self, domain: Domain) -> np.ndarray:
        """Encode a domain."""
        return self.domain_encoder.encode(domain)
    
    def encode_domain_query(
        self,
        domain_name: Optional[str] = None,
        site_host: Optional[str] = None,
    ) -> np.ndarray:
        """Encode query for domain matching."""
        return self.domain_encoder.encode_query(domain_name, site_host)