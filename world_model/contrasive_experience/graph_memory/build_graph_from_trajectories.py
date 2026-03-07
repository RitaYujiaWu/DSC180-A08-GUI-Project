#!/usr/bin/env python3
"""
Build Graph Memory from Trajectory Files.

This script processes SUCCESSFUL trajectory JSONL files (same format as discrete memory),
generates VLM-based takeaway summaries for each, extracts tags, and builds
a graph where trajectories are connected by shared tags.

Supports VLM-based deduplication (enabled by default):
- Uses FAISS to find similar trajectories (threshold: 0.92)
- VLM decides: UPDATE (enrich old), REPLACE (swap), or ADD (new node)
- Embeddings use concatenated task_description + takeaway for comprehensive similarity

Usage (single domain with dedup):
    python -m graph_memory.build_graph_from_trajectories \
        --memory_data_dir /path/to/data/trajectories \
        --output_path graph_index/Amazon \
        --domain Amazon \
        --model qwen2.5-vl

Usage (comprehensive - all domains with dedup):
    python -m graph_memory.build_graph_from_trajectories \
        --memory_data_dir /path/to/data/trajectories \
        --output_path graph_index/all_domains \
        --model qwen2.5-vl

Usage (disable dedup):
    python -m graph_memory.build_graph_from_trajectories \
        --memory_data_dir /path/to/data/trajectories \
        --output_path graph_index/all_domains \
        --model qwen2.5-vl \
        --no_dedup

Usage (custom threshold):
    python -m graph_memory.build_graph_from_trajectories \
        --memory_data_dir /path/to/data/trajectories \
        --output_path graph_index/all_domains \
        --model qwen2.5-vl \
        --similarity_threshold 0.95

Note: Only trajectories from 'success/' folders are used (no negative/positive).

The output can then be loaded by GraphBuilder.load() for retrieval.
"""

import argparse
import json
import logging
import os
import re
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_memory import TagExtractor, GraphBuilder, TaggedTrajectory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_action_from_response(response: str) -> Optional[Dict]:
    """Parse action JSON from a response string."""
    if not isinstance(response, str):
        return None
    
    # Try ```json block first
    if "```json" in response:
        match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    
    # Try Action: prefix
    match = re.search(r'Action:\s*(\{.*\})', response)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try direct JSON
    try:
        obj = json.loads(response)
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            return obj
        # Handle double-encoded JSON (guiact_converted format)
        # First json.loads returns a string, second returns the dict
        if isinstance(obj, str):
            inner_obj = json.loads(obj)
            if isinstance(inner_obj, dict) and "name" in inner_obj and "arguments" in inner_obj:
                return inner_obj
    except json.JSONDecodeError:
        pass
    
    return None


def extract_base64_image(round_data: Dict) -> Optional[str]:
    """Extract base64 image from a round's messages."""
    if 'messages' not in round_data:
        return None
    for msg in round_data['messages']:
        content = msg.get('content')
        if isinstance(content, list):
            for item in content:
                if item.get('type') == 'image_url':
                    return item['image_url']['url']
    return None


def format_actions_for_summary(actions: List[Dict], max_actions: int = 8) -> str:
    """Format actions into a string for VLM summarization."""
    lines = []
    for a in actions[:max_actions]:
        name = a.get("name", "unknown")
        args = a.get("arguments", {})
        reasoning = args.get("reasoning", "")
        lines.append(f"- {name}: {reasoning}")
    return "\n".join(lines)


def summarize_trajectory_with_vlm(
    llm,
    task: str,
    actions_text: str,
    image_b64: Optional[str] = None,
) -> str:
    """
    Generate a one-sentence takeaway from a trajectory using VLM.
    
    Same prompt as agent.py _summarize_trajectory_with_vlm.
    """
    system = (
        "You extract actionable heuristics from SUCCESSFUL GUI agent trajectories.\n"
        "Return EXACTLY 1 sentence starting with 'takeaway:' in this format:\n"
        "takeaway: <ONE concise actionable heuristic>\n"
        "Constraints:\n"
        "- Focus on WHAT strategy worked (not what the agent did).\n"
        "- Must start with 'takeaway:' (exact substring).\n"
        "- Keep it under 20 words.\n"
        "- No quotes, no bullet points, no step-by-step narration, no coordinates/IDs."
    )
    user_text = f"Task: {task}\nActions:\n{actions_text}"
    
    if image_b64 is None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]
    else:
        # Ensure proper data URL format
        if not image_b64.startswith("data:image"):
            image_b64 = f"data:image/png;base64,{image_b64}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ]},
        ]
    
    resp, _, _ = llm.chat(messages=messages, stream=False, temperature=0.0, max_tokens=128)
    
    if not hasattr(resp, "content"):
        raise ValueError("LLM response missing content")
    
    summary = resp.content.strip()
    
    # Validate format
    if not summary:
        raise ValueError("Empty summary returned by LLM")
    
    # Extract just the takeaway line if there's extra text
    for line in summary.split('\n'):
        line = line.strip()
        if line.lower().startswith("takeaway:"):
            return line
    
    # If no proper takeaway line found, use first line
    first_line = summary.split('\n')[0].strip()
    if not first_line.lower().startswith("takeaway:"):
        first_line = f"takeaway: {first_line}"
    
    return first_line


def load_trajectory_file(filepath: str) -> Optional[Dict]:
    """Load and validate a trajectory JSONL file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        task_description = data.get('task_description', '')
        if not task_description:
            logger.warning(f"Skipping {filepath}: empty task_description")
            return None
        
        rounds = data.get('rounds', [])
        total_rounds = data.get('total_rounds', len(rounds))
        
        if total_rounds >= 15:
            logger.info(f"Skipping {filepath}: {total_rounds} rounds (need < 15)")
            return None
        
        return data
    except Exception as e:
        logger.warning(f"Error loading {filepath}: {e}")
        return None


def process_trajectory(
    filepath: str,
    data: Dict,
    llm,
    domain: str,
    dataset: str,
    clip_model,
    summary_cache: Dict[str, str],
) -> Optional[Tuple[str, str, str, np.ndarray, str, List[Dict], List[str]]]:
    """
    Process a single trajectory file and return node data.
    
    Returns:
        (traj_id, takeaway, task_description, embedding, first_image, actions, images) or None
        
    Note: actions and images are needed for CONTINUOUS MEMORY (Q-Former) integration.
    """
    traj_id = Path(filepath).stem  # e.g., "Amazon_Amazon_2"
    task_description = data['task_description']
    rounds = data['rounds']
    
    # Extract actions and images from ALL rounds (needed for continuous memory)
    all_actions = []
    all_images = []
    for r in rounds:
        response = r.get('response', '')
        if isinstance(response, list):
            response = response[0] if response else ''
        if isinstance(response, dict) and 'content' in response:
            response = response['content']
        
        action = parse_action_from_response(response)
        image = extract_base64_image(r)
        
        if action and 'name' in action:
            all_actions.append(action)
            all_images.append(image)  # May be None, but keep alignment
        
    if len(all_actions) < 2:
        logger.warning(f"Skipping {traj_id}: only {len(all_actions)} valid actions")
        return None
    
    # Get first screenshot (needed for both cache hit and miss)
    first_image = all_images[0] if all_images else None
        
    # Check cache first for takeaway
    if traj_id in summary_cache:
        takeaway = summary_cache[traj_id]
        logger.info(f"[Cache hit] {traj_id}: {takeaway}")
    else:
        # Format actions for summarization
        actions_text = format_actions_for_summary(all_actions, max_actions=8)
        
        # Generate takeaway with VLM
        try:
            takeaway = summarize_trajectory_with_vlm(
                llm=llm,
                task=task_description,
                actions_text=actions_text,
                image_b64=first_image,
            )
            logger.info(f"[Generated] {traj_id}: {takeaway}")
            summary_cache[traj_id] = takeaway
        except Exception as e:
            logger.error(f"Error summarizing {traj_id}: {e}")
            return None
    
    # Compute embedding for the takeaway
    # Use the SAME multimodal retrieval representation as discrete memory:
    # query/key = "<dataset>_<domain>: <task_description>" + first screenshot
    prefixed_query = f"{dataset}_{domain}: {task_description}"
    embedding = clip_model.get_multimodal_embeddings([prefixed_query], [first_image])
    if len(embedding.shape) > 1:
        embedding = embedding[0]
    
    # Get first image for reference
    first_image = all_images[0] if all_images else None
    
    # Filter out actions with None images for continuous memory
    # Keep only (action, image) pairs where image is not None
    filtered_actions = []
    filtered_images = []
    for action, img in zip(all_actions, all_images):
        if img is not None:
            filtered_actions.append(action)
            filtered_images.append(img)
    
    return traj_id, takeaway, task_description, embedding.astype(np.float32), first_image, filtered_actions, filtered_images


def extract_domain_from_path(filepath: str) -> str:
    """Extract domain name from trajectory file path.
    
    Expected path structure: .../trajectories/<dataset>/<domain>/<model>/test/success/<file>.jsonl
    or: .../trajectories/webvoyager/<domain>/<model>/test/success/<file>.jsonl
    """
    parts = Path(filepath).parts
    # Look for 'success' and go up to find domain
    for i, part in enumerate(parts):
        if part == 'success' and i >= 4:
            # domain is typically 4 levels up from the file
            # .../domain/model/test/success/file.jsonl
            return parts[i - 3]  # domain
    # Fallback: try to extract from filename (e.g., "Amazon_Amazon_2.jsonl")
    filename = Path(filepath).stem
    if '_' in filename:
        return filename.split('_')[0]
    return "unknown"


def extract_dataset_from_path(filepath: str) -> str:
    """Extract dataset name from trajectory file path.
    
    Expected path structure: .../trajectories/<dataset>/<domain>/<model>/test/success/<file>.jsonl
    """
    parts = Path(filepath).parts
    for i, part in enumerate(parts):
        if part == 'success' and i >= 5:
            # .../<dataset>/<domain>/<model>/test/success/<file>
            return parts[i - 4]
    return "unknown"


def build_graph(
    memory_data_dir: str,
    output_path: str,
    domain: Optional[str],
    llm,
    tag_extractor: TagExtractor,
    clip_model,
    summary_cache_path: Optional[str] = None,
    use_dedup: bool = True,
    similarity_threshold: float = 0.92,
    load_from: Optional[str] = None,
    max_trajectories: Optional[int] = None,
):
    """
    Build graph memory from SUCCESSFUL trajectory files.
    
    Args:
        memory_data_dir: Root directory containing trajectory files
        output_path: Base path for saving graph index
        domain: Optional domain filter (e.g., "Amazon"). If None, builds comprehensive graph.
        llm: LLM for summarization
        tag_extractor: TagExtractor for extracting tags
        clip_model: CLIP model for embeddings
        summary_cache_path: Optional path to cache summaries
        use_dedup: Whether to use VLM-based deduplication (default: True)
        similarity_threshold: Similarity threshold for dedup retrieval (default: 0.92)
        load_from: Optional path to existing graph to append to (enables incremental building)
        max_trajectories: Optional maximum number of trajectories to process (random sample)
    """
    # Load summary cache
    summary_cache: Dict[str, str] = {}
    if summary_cache_path and Path(summary_cache_path).exists():
        try:
            with open(summary_cache_path, 'r') as f:
                cache_data = json.load(f)
            summary_cache = cache_data.get('summaries', {})
            logger.info(f"Loaded {len(summary_cache)} cached summaries")
        except Exception as e:
            logger.warning(f"Error loading summary cache: {e}")
    
    # Find all SUCCESS trajectory folders (only success, not negative/positive)
    success_folders = []
    for root, dirs, _ in os.walk(memory_data_dir, followlinks=True):
        if 'success' in dirs:
            success_path = os.path.join(root, 'success')
            # Domain filter (optional)
            if domain:
                # Check if domain is in the path
                if domain.lower() not in root.lower():
                    continue
            success_folders.append(success_path)
            logger.info(f"Found success folder: {success_path}")
    
    if not success_folders:
        raise ValueError(f"No success folders found in {memory_data_dir}")
    
    # Collect all trajectory files
    trajectory_files = []
    for folder in success_folders:
        files = glob(os.path.join(folder, '*.jsonl'))
        trajectory_files.extend(files)
    
    logger.info(f"Found {len(trajectory_files)} trajectory files")
    
    # Sample trajectories if max_trajectories is specified
    if max_trajectories is not None and len(trajectory_files) > max_trajectories:
        import random
        random.seed(42)  # For reproducibility
        original_count = len(trajectory_files)
        trajectory_files = random.sample(trajectory_files, max_trajectories)
        logger.info(f"Sampled {max_trajectories} trajectories (from {original_count} total)")
    
    # Initialize graph builder with dedup support
    graph_builder = GraphBuilder(
        tag_extractor=tag_extractor,
        llm=llm if use_dedup else None,
        embedding_model=clip_model if use_dedup else None,
    )
    
    # Load existing graph if specified (append mode)
    if load_from:
        logger.info(f"Loading existing graph from: {load_from}")
        graph_builder.load(load_from)
        logger.info(f"Loaded {len(graph_builder.trajectories)} existing trajectories")
    
    # Set custom threshold if provided
    if use_dedup and similarity_threshold != GraphBuilder.SIMILARITY_THRESHOLD:
        logger.info(f"Using custom similarity threshold: {similarity_threshold}")
        # Override class constant for this instance
        graph_builder.SIMILARITY_THRESHOLD = similarity_threshold
    
    if use_dedup:
        logger.info(f"Deduplication ENABLED (threshold={similarity_threshold}, top_k={GraphBuilder.TOP_K_COMPARE})")
    else:
        logger.info("Deduplication DISABLED")
    
    # Process each trajectory
    processed = 0
    domain_counts: Dict[str, int] = {}
    dedup_results: Dict[str, int] = {'added': 0, 'updated': 0, 'replaced': 0, 'skipped': 0}
    
    for filepath in trajectory_files:
        data = load_trajectory_file(filepath)
        if data is None:
            continue
        
        # Extract dataset/domain from path if not specified (comprehensive mode)
        traj_domain = domain if domain else extract_domain_from_path(filepath)
        traj_dataset = extract_dataset_from_path(filepath)
        
        result = process_trajectory(
            filepath=filepath,
            data=data,
            llm=llm,
            domain=traj_domain,
            dataset=traj_dataset,
            clip_model=clip_model,
            summary_cache=summary_cache,
        )
        
        if result is None:
            continue
        
        traj_id, takeaway, task_description, embedding, first_image, actions, images = result
        
        # Extract tags from takeaway
        try:
            tags = tag_extractor.extract_tags(
                takeaway=takeaway,
                domain=traj_domain,
                trajectory_id=traj_id,
            )
        except Exception as e:
            logger.warning(f"Error extracting tags for {traj_id}: {e}")
            tags = {f"#{traj_domain.lower()}"} if traj_domain else set()
        
        # Create trajectory node with FULL DATA for continuous memory
        # Storing actions and images enables hybrid memory (discrete + continuous)
        trajectory = TaggedTrajectory(
            id=traj_id,
            takeaway=takeaway,
            tags=tags,
            embedding=embedding,
            domain=traj_domain,
            full_data={
                'task_description': task_description,
                'file_path': filepath,
                # NEW: Store actions and images for continuous memory (Q-Former)
                'actions': actions,  # List[Dict] with name, arguments
                'images': images,    # List[str] base64 encoded images
            },
        )
        
        # Add trajectory with or without deduplication
        if use_dedup:
            result = graph_builder.add_trajectory_with_dedup(
                trajectory=trajectory,
                threshold=similarity_threshold,
            )
            # Track dedup results
            if result.startswith('added'):
                dedup_results['added'] += 1
            elif result.startswith('updated'):
                dedup_results['updated'] += 1
            elif result.startswith('replaced'):
                dedup_results['replaced'] += 1
            elif result.startswith('skipped'):
                dedup_results['skipped'] += 1
        else:
            graph_builder.add_trajectory(trajectory)
        
        processed += 1
        domain_counts[traj_domain] = domain_counts.get(traj_domain, 0) + 1
        
        if processed % 10 == 0:
            logger.info(f"Processed {processed} trajectories...")
    
    # Log domain distribution
    logger.info(f"Domain distribution: {domain_counts}")
    
    # Log dedup statistics if enabled
    if use_dedup:
        logger.info(f"Deduplication results: {dedup_results}")
        dedup_stats = graph_builder.get_dedup_stats()
        logger.info(f"  Added: {dedup_stats['dedup_added']}, "
                   f"Updated: {dedup_stats['dedup_updated']}, "
                   f"Replaced: {dedup_stats['dedup_replaced']}")
    
    logger.info(f"Built graph with {len(graph_builder)} nodes, "
                f"{graph_builder.graph.number_of_edges()} edges, "
                f"{len(graph_builder.tag_to_trajectories)} unique tags")
    
    # Save graph
    graph_builder.save(output_path)
    logger.info(f"Saved graph to {output_path}")
    
    # Save summary cache
    if summary_cache_path:
        Path(summary_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_cache_path, 'w') as f:
            json.dump({'summaries': summary_cache}, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(summary_cache)} summaries to {summary_cache_path}")
    
    # Print dedup summary prominently if enabled
    if use_dedup:
        print("\n" + "=" * 60)
        print("DEDUPLICATION SUMMARY")
        print("=" * 60)
        print(f"  Trajectories processed: {processed}")
        print(f"  ├─ Added (new nodes):   {dedup_results['added']}")
        print(f"  ├─ Updated (enriched):  {dedup_results['updated']}")
        print(f"  ├─ Replaced (swapped):  {dedup_results['replaced']}")
        print(f"  └─ Skipped (existing):  {dedup_results['skipped']}")
        reduction = processed - len(graph_builder)
        if processed > 0:
            reduction_pct = 100 * reduction / processed
            print(f"\n  Final nodes: {len(graph_builder)} (reduced by {reduction}, -{reduction_pct:.1f}%)")
        print("=" * 60)
    
    # Print graph summary
    print("\n" + "=" * 60)
    print(graph_builder.visualize())
    
    return graph_builder


def main():
    parser = argparse.ArgumentParser(description="Build Graph Memory from Trajectory Files")
    parser.add_argument("--memory_data_dir", type=str, required=True,
                        help="Root directory containing trajectory files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Base path for saving graph index (without extension)")
    parser.add_argument("--domain", type=str, default=None,
                        help="Optional domain filter (e.g., Amazon)")
    parser.add_argument("--tag_cache_path", type=str, default="graph_memory_cache/tags.json",
                        help="Path to cache extracted tags")
    parser.add_argument("--summary_cache_path", type=str, default="graph_memory_cache/summaries.json",
                        help="Path to cache VLM-generated summaries")
    parser.add_argument("--model", type=str, default="qwen2.5-vl",
                        help="Model name for VLM summarization")
    # Deduplication arguments
    parser.add_argument("--use_dedup", action="store_true", default=True,
                        help="Enable VLM-based deduplication (default: True)")
    parser.add_argument("--no_dedup", action="store_true",
                        help="Disable VLM-based deduplication")
    parser.add_argument("--similarity_threshold", type=float, default=0.92,
                        help="Similarity threshold for dedup retrieval (default: 0.92)")
    # Append mode
    parser.add_argument("--load_from", type=str, default=None,
                        help="Path to existing graph to append to (enables incremental building)")
    # Sampling
    parser.add_argument("--max_trajectories", type=int, default=None,
                        help="Maximum number of trajectories to process (random sample if exceeded)")
    args = parser.parse_args()
    
    # Handle dedup flag
    use_dedup = args.use_dedup and not args.no_dedup
    
    # Import LLM and CLIP
    from agent.llm_config import load_tool_llm
    from memory.help_functions import CLIPTextSimilarity
    
    # Create a minimal args namespace for load_tool_llm
    class LLMArgs:
        def __init__(self):
            self.tool_model_name = args.model
            self.provider = "custom"
    
    llm_args = LLMArgs()
    llm = load_tool_llm(llm_args)
    
    # Initialize tag extractor
    tag_extractor = TagExtractor(llm=llm, cache_path=args.tag_cache_path)
    
    # Initialize CLIP (multimodal) to match discrete-memory retrieval representation
    from memory.help_functions import CLIPMultimodalSimilarity
    clip_model = CLIPMultimodalSimilarity()
    
    # Build graph
    build_graph(
        memory_data_dir=args.memory_data_dir,
        output_path=args.output_path,
        domain=args.domain,
        llm=llm,
        tag_extractor=tag_extractor,
        clip_model=clip_model,
        summary_cache_path=args.summary_cache_path,
        use_dedup=use_dedup,
        similarity_threshold=args.similarity_threshold,
        load_from=args.load_from,
        max_trajectories=args.max_trajectories,
    )


if __name__ == "__main__":
    main()

