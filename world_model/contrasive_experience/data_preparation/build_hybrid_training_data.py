#!/usr/bin/env python3
"""
Build Hybrid Training Data for CoMEM Agent.

This script preprocesses GT trajectories offline to create augmented training samples
that include:
1. Digested guidance from Graph Memory (discrete memory)
2. Retrieved takeaways with file paths (for continuous memory)

Usage:
    python build_hybrid_training_data.py \
        --gt_trajectory_dirs /path/to/mind2web /path/to/guiact_converted \
        --graph_index_path /path/to/graph_index/all_domains \
        --output_dir /path/to/output \
        --vllm_server http://localhost:8000/v1

Output format (for each sample):
{
    "task_id": "uid_record_XXXXX",
    "task_description": "Find a hotel in Paris under $100",
    "digested_guidance": "Focus on using price filters...",
    "retrieved_takeaways": [
        {
            "trajectory_id": "Amazon_Amazon_45",
            "takeaway": "Use filters to narrow down options",
            "file_path": "/absolute/path/to/trajectory.jsonl"
        },
        ...
    ]
}
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import logging
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_memory import GraphBuilder, GraphMemoryRetriever
from memory.help_functions import CLIPMultimodalSimilarity
from openai import OpenAI


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VLMDigester:
    """VLM-based digester for graph memory takeaways."""
    
    def __init__(self, server_url: str, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.client = OpenAI(base_url=server_url, api_key="EMPTY")
        self.model_name = model_name
        
    def digest(self, current_task: str, current_image: str, takeaways: List[str]) -> str:
        """
        Digest retrieved takeaways into task-specific guidance.
        
        Args:
            current_task: The current task description
            current_image: Base64 encoded first screenshot (REQUIRED)
            takeaways: List of takeaway strings from retrieved trajectories
            
        Returns:
            Digested guidance as a single paragraph
        """
        if not current_image:
            raise ValueError("current_image is required for VLM digestion")
        
        summaries_text = "\n".join(f"- {s}" for s in takeaways)
        
        system = (
            "You are an expert at analyzing past GUI agent experiences to help with a new task.\n"
            "Given the current task, current screenshot, and retrieved experience takeaways,\n"
            "synthesize them into focused, actionable guidance.\n\n"
            "Output format: ONE concise paragraph (2-3 sentences) that answers:\n"
            "1. Which strategies from past experiences are MOST relevant to this specific task?\n"
            "2. What key actions or filters should be prioritized?\n\n"
            "IMPORTANT RULES:\n"
            "- Focus ONLY on navigation/search strategies, NOT on when to stop.\n"
            "- Do NOT mention stopping, completing, or finishing the task.\n"
            "- Do NOT give instructions about providing answers or explanations.\n"
            "- Be specific to the current task. Do NOT just repeat the summaries.\n"
            "- Do NOT use bullet points. Write as a coherent paragraph."
        )
        
        user_text = f"Current Task: {current_task}\n\nRetrieved Experience Takeaways:\n{summaries_text}"
        
        # Handle both raw base64 and data URL formats
        if current_image.startswith("data:image"):
            image_url = current_image
        else:
            image_url = f"data:image/png;base64,{current_image}"
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )
        
        return response.choices[0].message.content.strip()


def load_gt_trajectories(trajectory_dir: str) -> List[Dict]:
    """
    Load GT trajectories from directory.
    
    Args:
        trajectory_dir: Directory containing trajectory files (can be nested)
        
    Returns:
        List of trajectory dicts with file paths
    """
    trajectories = []
    trajectory_dir = Path(trajectory_dir)
    
    # Find all JSON/JSONL files recursively
    for json_file in trajectory_dir.rglob("*.json*"):
        # Skip directories and non-files
        if not json_file.is_file():
            continue
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Validate required fields
            if all(k in data for k in ['conversation_id', 'task_description', 'rounds']):
                data['_file_path'] = str(json_file.resolve())  # Absolute path
                trajectories.append(data)
        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(trajectories)} GT trajectories from {trajectory_dir}")
    return trajectories


def extract_first_image(trajectory: Dict) -> Optional[str]:
    """Extract the first screenshot from a trajectory."""
    if not trajectory.get('rounds'):
        return None
    
    first_round = trajectory['rounds'][0]
    messages = first_round.get('messages', [])
    
    for msg in messages:
        if msg.get('role') == 'user':
            content = msg.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'image_url':
                        return item.get('image_url', {}).get('url')
    return None


def process_trajectories(
    gt_trajectories: List[Dict],
    retriever: GraphMemoryRetriever,
    digester: VLMDigester,
    embedding_model: CLIPMultimodalSimilarity,
    output_dir: str,
    k: int = 10,
    batch_size: int = 100,
):
    """
    Process GT trajectories and build augmented training samples.
    
    Args:
        gt_trajectories: List of GT trajectory dicts
        retriever: GraphMemoryRetriever instance
        digester: VLMDigester instance
        embedding_model: Embedding model for queries
        output_dir: Output directory for augmented samples
        k: Number of similar trajectories to retrieve (default 10)
        batch_size: Number of samples per output file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    stats = {
        'total_trajectories': 0,
        'successful_samples': 0,
        'failed_no_image': 0,
        'failed_retrieval': 0,
        'failed_digestion': 0,
    }
    
    for traj in tqdm(gt_trajectories, desc="Processing trajectories"):
        stats['total_trajectories'] += 1
        task_id = traj['conversation_id']
        task_description = traj['task_description']
        first_image = extract_first_image(traj)
        
        if not first_image:
            logger.warning(f"No first image found for {task_id}")
            stats['failed_no_image'] += 1
            continue
        
        # Retrieve similar trajectories from graph memory
        try:
            query_embedding = embedding_model.get_multimodal_embeddings(
                [task_description], 
                [first_image]
            )[0]
            
            retrieved = retriever.retrieve(
                query_embedding=query_embedding,
                k=k
            )
            
            if not retrieved:
                logger.warning(f"No similar trajectories found for {task_id}")
                stats['failed_retrieval'] += 1
                continue
                
        except Exception as e:
            logger.warning(f"Retrieval failed for {task_id}: {e}")
            stats['failed_retrieval'] += 1
            continue
        
        # Get takeaways for digestion
        takeaways = [t.takeaway for t in retrieved]
        
        # Digest takeaways into guidance
        try:
            digested_guidance = digester.digest(
                current_task=task_description,
                current_image=first_image,
                takeaways=takeaways
            )
        except Exception as e:
            logger.warning(f"Digestion failed for {task_id}: {e}")
            stats['failed_digestion'] += 1
            continue
        
        # Build retrieved_takeaways with file paths
        retrieved_takeaways = []
        for t in retrieved:
            takeaway_info = {
                "trajectory_id": t.id,
                "takeaway": t.takeaway,
                "file_path": t.full_data.get('file_path', '') if t.full_data else ''
            }
            retrieved_takeaways.append(takeaway_info)
        
        # Build the output sample
        sample = {
            "task_id": task_id,
            "task_description": task_description,
            "digested_guidance": digested_guidance,
            "retrieved_takeaways": retrieved_takeaways,
        }
        
        all_samples.append(sample)
        stats['successful_samples'] += 1
        
        if stats['successful_samples'] % 50 == 0:
            logger.info(f"Processed {stats['successful_samples']} samples...")
    
    # Save samples in batches
    for i in range(0, len(all_samples), batch_size):
        batch = all_samples[i:i + batch_size]
        batch_file = output_dir / f"hybrid_training_batch_{i // batch_size:05d}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved batch {i // batch_size} with {len(batch)} samples to {batch_file}")
    
    # Also save all samples in a single file
    all_samples_file = output_dir / "all_samples.json"
    with open(all_samples_file, 'w') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved all {len(all_samples)} samples to {all_samples_file}")
    
    # Save stats
    stats_file = output_dir / "preprocessing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total trajectories processed: {stats['total_trajectories']}")
    logger.info(f"Successful samples: {stats['successful_samples']}")
    logger.info(f"Failed (no image): {stats['failed_no_image']}")
    logger.info(f"Failed (retrieval): {stats['failed_retrieval']}")
    logger.info(f"Failed (digestion): {stats['failed_digestion']}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}")
    
    return all_samples, stats


def main():
    parser = argparse.ArgumentParser(description="Build hybrid training data for CoMEM Agent")
    
    parser.add_argument(
        "--gt_trajectory_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing GT trajectory files"
    )
    parser.add_argument(
        "--graph_index_path",
        type=str,
        required=True,
        help="Path to graph index (without file extension, e.g., graph_index/all_domains)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for augmented training samples"
    )
    parser.add_argument(
        "--vllm_server",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server URL for VLM digestion"
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name for vLLM server"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of similar trajectories to retrieve (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of samples per output file"
    )
    
    args = parser.parse_args()
    
    # Initialize components
    logger.info("Initializing embedding model...")
    embedding_model = CLIPMultimodalSimilarity()
    
    logger.info("Loading graph index...")
    graph_builder = GraphBuilder()
    graph_builder.load(args.graph_index_path)
    logger.info(f"Loaded graph with {len(graph_builder.trajectories)} trajectories")
    
    logger.info("Initializing retriever...")
    retriever = GraphMemoryRetriever(
        graph_builder=graph_builder,
        embedding_model=embedding_model,
        expand_hops=1
    )
    
    logger.info("Initializing VLM digester...")
    digester = VLMDigester(
        server_url=args.vllm_server,
        model_name=args.vllm_model
    )
    
    # Load GT trajectories from all directories
    logger.info("Loading GT trajectories...")
    gt_trajectories = []
    for traj_dir in args.gt_trajectory_dirs:
        logger.info(f"Loading from: {traj_dir}")
        gt_trajectories.extend(load_gt_trajectories(trajectory_dir=traj_dir))
    
    logger.info(f"Total GT trajectories loaded: {len(gt_trajectories)}")
    
    if not gt_trajectories:
        logger.error("No GT trajectories found!")
        return
    
    # Process trajectories
    logger.info("Processing trajectories...")
    process_trajectories(
        gt_trajectories=gt_trajectories,
        retriever=retriever,
        digester=digester,
        embedding_model=embedding_model,
        output_dir=args.output_dir,
        k=args.k,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
