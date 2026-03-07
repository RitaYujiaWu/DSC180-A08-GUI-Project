#!/usr/bin/env python3
"""
Create a subset of graph memory index with a specified number of trajectories.

Usage:
    python create_memory_subset.py --size 500
    python create_memory_subset.py --size 100 250 500 1000
    python create_memory_subset.py --size 500 --base_path graph_index/all_domains --output_dir graph_index
"""

import argparse
import json
import random
import numpy as np
from pathlib import Path


def create_subset(size: int, base_path: str, output_dir: str, seed: int = 42) -> str:
    """
    Create a subset of the graph memory index.
    
    Args:
        size: Number of trajectories to include
        base_path: Path to full graph index (without extension)
        output_dir: Directory to save subset indices
        seed: Random seed for reproducibility
        
    Returns:
        Path to the created subset index (without extension)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    output_path = f"{output_dir}/ablation_size_{size}"
    
    # Check if already exists
    if Path(f"{output_path}_trajectories.json").exists():
        print(f"[INFO] Subset for size {size} already exists at {output_path}")
        return output_path
    
    print(f"[INFO] Creating subset with {size} trajectories...")
    print(f"  Base: {base_path}")
    print(f"  Output: {output_path}")
    
    # Load full graph
    with open(f"{base_path}_trajectories.json", 'r') as f:
        all_trajectories = json.load(f)
    
    with open(f"{base_path}_tags.json", 'r') as f:
        all_tags = json.load(f)
    
    with open(f"{base_path}_graph.json", 'r') as f:
        full_graph = json.load(f)
    
    embeddings_data = np.load(f"{base_path}_embeddings.npz", allow_pickle=True)
    all_ids = list(embeddings_data['ids'])
    all_embeddings = embeddings_data['embeddings']
    
    print(f"  Full graph: {len(all_trajectories)} trajectories")
    
    # Sample trajectories
    traj_ids = list(all_trajectories.keys())
    if size >= len(traj_ids):
        selected_ids = set(traj_ids)
        print(f"  Using all {len(traj_ids)} trajectories")
    else:
        selected_ids = set(random.sample(traj_ids, size))
        print(f"  Sampled {size} trajectories")
    
    # Filter trajectories
    subset_trajectories = {k: v for k, v in all_trajectories.items() if k in selected_ids}
    
    # Filter tags
    subset_tags = {}
    for tag, traj_list in all_tags.items():
        filtered = [t for t in traj_list if t in selected_ids]
        if filtered:
            subset_tags[tag] = filtered
    
    # Filter graph
    subset_nodes = [n for n in full_graph['nodes'] if n['id'] in selected_ids]
    subset_links = [l for l in full_graph['links'] 
                   if l['source'] in selected_ids and l['target'] in selected_ids]
    
    subset_graph = {
        'directed': full_graph.get('directed', False),
        'multigraph': full_graph.get('multigraph', False),
        'graph': full_graph.get('graph', {}),
        'nodes': subset_nodes,
        'links': subset_links
    }
    
    # Filter embeddings
    id_to_idx = {str(id_): i for i, id_ in enumerate(all_ids)}
    subset_indices = [id_to_idx[str(tid)] for tid in selected_ids if str(tid) in id_to_idx]
    subset_emb_ids = [all_ids[i] for i in subset_indices]
    subset_embeddings = all_embeddings[subset_indices]
    
    # Save subset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(f"{output_path}_trajectories.json", 'w') as f:
        json.dump(subset_trajectories, f)
    
    with open(f"{output_path}_tags.json", 'w') as f:
        json.dump(subset_tags, f)
    
    with open(f"{output_path}_graph.json", 'w') as f:
        json.dump(subset_graph, f)
    
    np.savez(f"{output_path}_embeddings.npz", 
             ids=np.array(subset_emb_ids), 
             embeddings=subset_embeddings)
    
    print(f"[DONE] Saved subset to {output_path}")
    print(f"  - Trajectories: {len(subset_trajectories)}")
    print(f"  - Tags: {len(subset_tags)}")
    print(f"  - Nodes: {len(subset_nodes)}")
    print(f"  - Edges: {len(subset_links)}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create graph memory subsets for ablation")
    parser.add_argument("--size", type=int, nargs='+', required=True,
                       help="Number of trajectories (can specify multiple)")
    parser.add_argument("--base_path", type=str, 
                       default="graph_index/all_domains",
                       help="Path to full graph index (without extension)")
    parser.add_argument("--output_dir", type=str,
                       default="graph_index",
                       help="Directory to save subset indices")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    for size in args.size:
        create_subset(size, args.base_path, args.output_dir, args.seed)


if __name__ == "__main__":
    main()

