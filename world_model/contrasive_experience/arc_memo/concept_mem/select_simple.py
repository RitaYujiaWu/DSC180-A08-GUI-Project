"""
Simplified concept selection using CLIP embeddings for semantic similarity.

Usage:
    python select_simple.py --task_id Amazon--1 --reasoning_plan "Search for products with filters"
    python select_simple.py --task_id test_task --reasoning_plan "Login and navigate to dashboard"
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from memory.help_functions import CLIPTextSimilarity
from arc_memo.concept_mem.concept import ConceptMemory
from arc_memo.concept_mem.utils import read_json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MEMORY_PATH = Path(__file__).parent.parent / "output" / "memory.json"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "output"


class ConceptSelector:
    """Select concepts based on CLIP semantic similarity between reasoning plan and cues."""
    
    def __init__(self, clip_model: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP-based concept selector.
        
        Args:
            clip_model: HuggingFace CLIP model identifier
        """
        logger.info(f"Initializing CLIP model: {clip_model}")
        self.clip = CLIPTextSimilarity(model_name=clip_model)
        
        # Cache for cue embeddings
        self.cue_embeddings = None
        self.cue_list = []  # Ordered list of cues matching embeddings
        self.cue_to_concepts = {}  # cue text -> list of concept names
        
    def precompute_cue_embeddings(self, memory: ConceptMemory):
        """
        Precompute embeddings for all cues in the concept memory.
        
        Args:
            memory: ConceptMemory instance with concepts
        """
        logger.info("Precomputing embeddings for all cues...")
        
        self.cue_list = []
        self.cue_to_concepts = {}
        
        for concept_name, concept in memory.concepts.items():
            if not concept.cues:
                continue
            for cue in concept.cues:
                if cue not in self.cue_to_concepts:
                    self.cue_to_concepts[cue] = []
                    self.cue_list.append(cue)
                self.cue_to_concepts[cue].append(concept_name)
        
        if not self.cue_list:
            logger.warning("No cues found in concept memory!")
            return
        
        # Batch encode all cues using existing CLIP class
        self.cue_embeddings = self.clip.get_text_embeddings(self.cue_list)
        
        # Normalize embeddings for cosine similarity
        self.cue_embeddings = self.cue_embeddings / np.linalg.norm(
            self.cue_embeddings, axis=1, keepdims=True
        )
        
        logger.info(f"Precomputed {len(self.cue_list)} cue embeddings for {len(memory.concepts)} concepts")
    
    def select_concepts(
        self, 
        reasoning_plan: str, 
        memory: ConceptMemory,
        top_k: int = 10
    ) -> List[str]:
        """
        Select top-k concepts based on similarity between reasoning plan and cues.
        
        Args:
            reasoning_plan: The LLM-generated reasoning plan (string)
            memory: ConceptMemory instance
            top_k: Number of concepts to select
            
        Returns:
            List of selected concept names
        """
        if self.cue_embeddings is None:
            logger.info("Cue embeddings not precomputed, computing now...")
            self.precompute_cue_embeddings(memory)
        
        if self.cue_embeddings is None or len(self.cue_list) == 0:
            logger.warning("No cue embeddings available, returning empty selection")
            return []
        
        # Encode reasoning plan
        plan_embedding = self.clip.get_text_embeddings([reasoning_plan])
        
        # Normalize
        plan_embedding = plan_embedding / np.linalg.norm(plan_embedding, axis=1, keepdims=True)
        
        # Calculate similarity scores for all cues (cosine similarity)
        similarities = np.dot(plan_embedding, self.cue_embeddings.T)[0]
        
        # Create (cue, score) pairs and sort
        cue_scores: List[Tuple[str, float]] = [
            (cue, float(score)) for cue, score in zip(self.cue_list, similarities)
        ]
        cue_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top cues and their associated concepts
        selected_concepts = []
        seen_concepts = set()
        
        for cue, score in cue_scores:
            for concept_name in self.cue_to_concepts[cue]:
                if concept_name not in seen_concepts:
                    selected_concepts.append(concept_name)
                    seen_concepts.add(concept_name)
                    logger.debug(f"Selected '{concept_name}' via cue '{cue}' (score: {score:.3f})")
                    
                if len(selected_concepts) >= top_k:
                    break
            
            if len(selected_concepts) >= top_k:
                break
        
        logger.info(f"Selected {len(selected_concepts)} concepts from top cues")
        return selected_concepts[:top_k]


def render_concepts_as_suggestions(memory: ConceptMemory, concept_names: List[str]) -> str:
    """
    Render selected concepts as a polished, well-structured prompt with implementation suggestions.
    
    Args:
        memory: ConceptMemory instance
        concept_names: List of concept names to render
        
    Returns:
        Formatted string with implementation suggestions as a cohesive prompt
    """
    if not concept_names:
        return "No relevant concepts found."
    
    # Group concepts by type for better organization
    concepts_by_type: Dict[str, List] = {}
    for name in concept_names:
        if name not in memory.concepts:
            continue
        concept = memory.concepts[name]
        concept_type = concept.type
        if concept_type not in concepts_by_type:
            concepts_by_type[concept_type] = []
        concepts_by_type[concept_type].append(concept)
    
    # Build the prompt
    prompt_parts = []
    
    # Header
    prompt_parts.append("# Implementation Guidance")
    prompt_parts.append("\nBased on similar tasks, here are relevant implementation patterns and best practices:\n")
    
    # Organize by type
    type_order = [
        "ui_navigation", "authentication", "search_filter", "data_entry",
        "form_handling", "selection", "data_extraction", "verification"
    ]
    
    for concept_type in type_order:
        if concept_type not in concepts_by_type:
            continue
        
        concepts = concepts_by_type[concept_type]
        type_label = concept_type.replace('_', ' ').title()
        
        prompt_parts.append(f"## {type_label}")
        
        for concept in concepts:
            # Add concept name as a subsection
            prompt_parts.append(f"\n### {concept.name.replace('_', ' ').title()}")
            
            # Add implementation steps as a numbered list
            if concept.implementation:
                for i, step in enumerate(concept.implementation, 1):
                    # Clean up the step text
                    step = step.strip()
                    if not step.endswith('.'):
                        step += '.'
                    prompt_parts.append(f"{i}. {step}")
            else:
                prompt_parts.append("*(No specific implementation notes available)*")
            
            prompt_parts.append("")  # Empty line between concepts
    
    # Add any remaining types not in the standard order
    for concept_type, concepts in concepts_by_type.items():
        if concept_type in type_order:
            continue
        
        type_label = concept_type.replace('_', ' ').title()
        prompt_parts.append(f"## {type_label}")
        
        for concept in concepts:
            prompt_parts.append(f"\n### {concept.name.replace('_', ' ').title()}")
            if concept.implementation:
                for i, step in enumerate(concept.implementation, 1):
                    step = step.strip()
                    if not step.endswith('.'):
                        step += '.'
                    prompt_parts.append(f"{i}. {step}")
            else:
                prompt_parts.append("*(No specific implementation notes available)*")
            prompt_parts.append("")
    
    # Footer
    prompt_parts.append("\n---")
    prompt_parts.append("\n**Note:** These are general implementation patterns. Adapt them to your specific task requirements and UI structure.")
    
    return "\n".join(prompt_parts)


def get_ps_memory(
    reasoning_plan: str,
    memory_path: str | Path = DEFAULT_MEMORY_PATH,
    top_k: int = 10,
    clip_model: str = "openai/clip-vit-base-patch32",
) -> str:
    """
    Main function: Select problem-specific memory based on reasoning plan.
    
    Args:
        reasoning_plan: LLM-generated reasoning/planning text
        memory_path: Path to memory.json
        top_k: Number of concepts to select
        clip_model: CLIP model name
        
    Returns:
        Formatted string with implementation suggestions
    """
    memory_path = Path(memory_path)
    
    # Load concept memory
    logger.info(f"Loading concept memory from {memory_path}")
    cm = ConceptMemory()
    cm.load_from_file(memory_path)
    logger.info(f"Loaded {len(cm.concepts)} concepts")
    
    # Initialize selector and precompute embeddings
    selector = ConceptSelector(clip_model=clip_model)
    selector.precompute_cue_embeddings(cm)
    
    # Select concepts based on reasoning plan
    selected_concepts = selector.select_concepts(reasoning_plan, cm, top_k=top_k)
    
    # Render as implementation suggestions
    suggestions = render_concepts_as_suggestions(cm, selected_concepts)
    
    return suggestions


def save_output(
    suggestions: str,
    selected_concepts: List[str],
    reasoning_plan: str,
    task_id: str,
    output_dir: Path
):
    """Save selection results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save suggestions as text
    suggestions_file = output_dir / "ps_memory_suggestions.txt"
    with open(suggestions_file, 'w') as f:
        f.write(suggestions)
    logger.info(f"Saved suggestions to {suggestions_file}")
    
    # Save metadata as JSON
    metadata = {
        "task_id": task_id,
        "reasoning_plan": reasoning_plan,
        "selected_concepts": selected_concepts,
        "num_selected": len(selected_concepts)
    }
    metadata_file = output_dir / "selection_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Select problem-specific memory using CLIP embeddings"
    )
    parser.add_argument(
        "--task_id",
        type=str,
        default="test_task",
        help="Task identifier (for logging/output naming)"
    )
    parser.add_argument(
        "--reasoning_plan",
        type=str,
        default="Search for products with filters",
        help="LLM-generated reasoning plan (string describing the task approach)"
    )
    parser.add_argument(
        "--memory_path",
        type=str,
        default=str(DEFAULT_MEMORY_PATH),
        help="Path to memory.json"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of concepts to select"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP model name from HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save output files"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Task ID: {args.task_id}")
    logger.info(f"Reasoning plan: {args.reasoning_plan[:100]}...")
    
    # Load memory
    cm = ConceptMemory()
    cm.load_from_file(Path(args.memory_path))
    
    # Initialize selector
    selector = ConceptSelector(clip_model=args.clip_model)
    selector.precompute_cue_embeddings(cm)
    
    # Select concepts
    selected_concepts = selector.select_concepts(args.reasoning_plan, cm, top_k=args.top_k)
    
    # Render suggestions
    suggestions = render_concepts_as_suggestions(cm, selected_concepts)
    
    # # Save outputs
    # save_output(
    #     suggestions=suggestions,
    #     selected_concepts=selected_concepts,
    #     reasoning_plan=args.reasoning_plan,
    #     task_id=args.task_id,
    #     output_dir=Path(args.output_dir)
    # )
    
    # Print to console
    print("\n" + "="*80)
    print("SELECTED IMPLEMENTATION SUGGESTIONS")
    print("="*80)
    print(suggestions)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
