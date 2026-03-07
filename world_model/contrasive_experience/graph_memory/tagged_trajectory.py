"""
Tagged Trajectory Data Structure.

A trajectory with extracted tags for graph-based memory organization.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Any
import numpy as np


@dataclass
class TaggedTrajectory:
    """
    A trajectory with associated tags for graph-based memory.
    
    Attributes:
        id: Unique identifier (e.g., "Coursera--18")
        takeaway: Summary text extracted from the trajectory
        tags: Set of tags (e.g., {"#filter", "#categories", "#beginner"})
        embedding: Vector embedding for FAISS similarity search
        domain: Domain/website name (e.g., "Coursera")
        full_data: Original trajectory data (actions, images, etc.)
        usage_count: How many times this trajectory was retrieved
        success_rate: Success rate when this trajectory's strategy was used
    """
    id: str
    takeaway: str
    tags: Set[str]
    embedding: Optional[np.ndarray] = None
    domain: str = ""
    full_data: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    success_rate: float = 0.0
    
    def __post_init__(self):
        """Validate and normalize tags."""
        # Ensure tags are a set
        if not isinstance(self.tags, set):
            self.tags = set(self.tags) if self.tags else set()
        
        # Normalize tags: lowercase and ensure # prefix
        normalized_tags = set()
        for tag in self.tags:
            tag = tag.lower().strip()
            if not tag.startswith('#'):
                tag = f'#{tag}'
            normalized_tags.add(tag)
        self.tags = normalized_tags
    
    def shares_tags_with(self, other: 'TaggedTrajectory') -> Set[str]:
        """Return the set of tags shared with another trajectory."""
        return self.tags & other.tags
    
    def num_shared_tags(self, other: 'TaggedTrajectory') -> int:
        """Return the number of tags shared with another trajectory."""
        return len(self.shares_tags_with(other))
    
    def has_tag(self, tag: str) -> bool:
        """Check if this trajectory has a specific tag."""
        tag = tag.lower().strip()
        if not tag.startswith('#'):
            tag = f'#{tag}'
        return tag in self.tags
    
    def to_dict(self, include_full_data: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Args:
            include_full_data: If True, includes full_data (actions, images, etc.)
                              for continuous memory support. Default True.
        """
        result = {
            'id': self.id,
            'takeaway': self.takeaway,
            'tags': list(self.tags),
            'domain': self.domain,
            'usage_count': self.usage_count,
            'success_rate': self.success_rate,
        }
        
        # Include full_data for continuous memory (Q-Former) support
        # Note: images are base64 strings, so they can be serialized
        if include_full_data and self.full_data:
            # Serialize full_data but exclude very large fields if needed
            serializable_full_data = {}
            for key, value in self.full_data.items():
                if key == 'images':
                    # Store image count but not actual images (too large for JSON)
                    # Images should be stored separately or loaded from file_path
                    serializable_full_data['_image_count'] = len(value) if value else 0
                    serializable_full_data['_has_images'] = bool(value)
                else:
                    serializable_full_data[key] = value
            result['full_data'] = serializable_full_data
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], 
                  embedding: Optional[np.ndarray] = None,
                  full_data: Optional[Dict] = None) -> 'TaggedTrajectory':
        """Create from dictionary."""
        # Prefer explicit full_data override; otherwise use serialized full_data (if present).
        fd = full_data if full_data is not None else data.get('full_data', {})
        if fd is None:
            fd = {}
        if not isinstance(fd, dict):
            raise ValueError(f"full_data must be a dict, got {type(fd)}")
        return cls(
            id=data['id'],
            takeaway=data['takeaway'],
            tags=set(data.get('tags', [])),
            embedding=embedding,
            domain=data.get('domain', ''),
            full_data=fd,
            usage_count=data.get('usage_count', 0),
            success_rate=data.get('success_rate', 0.0),
        )
    
    def __hash__(self):
        """Hash based on id for use in sets/dicts."""
        return hash(self.id)
    
    def __eq__(self, other):
        """Equality based on id."""
        if isinstance(other, TaggedTrajectory):
            return self.id == other.id
        return False
    
    def __repr__(self):
        return f"TaggedTrajectory(id={self.id!r}, tags={self.tags}, domain={self.domain!r})"

