# Graph Memory Module

A graph-based memory system for GUI agents that organizes trajectories by semantic tags and uses graph expansion for diverse retrieval.

## Overview

The Graph Memory module addresses the **diversity problem** in memory retrieval: when using only similarity-based search (FAISS), retrieved memories tend to be redundant - covering the same strategies repeatedly. This module solves this by:

1. **Summarizing trajectories** into one-sentence takeaways via VLM (same approach as discrete memory)
2. **Tagging takeaways** with semantic keywords extracted via LLM
3. **Building a graph** where trajectories are connected by shared tags
4. **Hybrid retrieval** combining FAISS similarity with graph expansion for diversity
5. **Digesting** retrieved takeaways into task-specific guidance at inference time

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Graph Memory System                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  BUILD PHASE (offline):                                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐                      │
│  │ Trajectory  │───►│ VLM Summary │───►│ TagExtractor │───►┌──────────────┐  │
│  │   Files     │    │ (takeaway)  │    │   (LLM)      │    │ GraphBuilder │  │
│  └─────────────┘    └─────────────┘    └──────────────┘    │  (NetworkX)  │  │
│                                                             └──────────────┘  │
│                                                                               │
│  RETRIEVAL PHASE (online):                                                    │
│  ┌─────────────┐    ┌───────────────┐    ┌─────────────┐                     │
│  │   Query     │───►│   Retriever   │───►│   Digest    │───► Guidance       │
│  │ (task+img)  │    │ (FAISS+Graph) │    │   (VLM)     │                     │
│  └─────────────┘    └───────────────┘    └─────────────┘                     │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. TaggedTrajectory (`tagged_trajectory.py`)

Data structure representing a trajectory with tags:

```python
@dataclass
class TaggedTrajectory:
    id: str              # e.g., "Coursera--18"
    takeaway: str        # Summary text
    tags: Set[str]       # e.g., {"#filter", "#categories", "#beginner"}
    embedding: np.ndarray
    domain: str          # e.g., "Coursera"
    full_data: Dict      # Original trajectory data
```

### 2. TagExtractor (`tag_extractor.py`)

Extracts semantic tags from trajectory takeaways using LLM:

```python
extractor = TagExtractor(llm=tool_llm, cache_path="cache/tags.json")
tags = extractor.extract_tags(
    takeaway="Filter by specific categories like 'For Individuals'...",
    domain="Coursera"
)
# Returns: {"#filter", "#categories", "#coursera"}
```

**Tag Categories:**
- **Actions**: `#search`, `#filter`, `#click`, `#scroll`, `#navigate`
- **UI Elements**: `#search_bar`, `#dropdown`, `#button`, `#menu`
- **Strategies**: `#free`, `#beginner`, `#categories`, `#sort`

### 3. GraphBuilder (`graph_builder.py`)

Builds and manages the trajectory graph:

```python
builder = GraphBuilder(tag_extractor=extractor)

# Add trajectories
builder.add_trajectory_from_data(
    traj_id="Coursera--18",
    takeaway="Filter by categories...",
    domain="Coursera",
    embedding=clip_embedding
)

# Get neighbors (trajectories with shared tags)
neighbors = builder.get_neighbors("Coursera--18", max_hops=1)

# Save/load graph
builder.save("graph_index/coursera")
builder.load("graph_index/coursera")
```

**Graph Structure:**
- **Nodes**: Trajectories
- **Edges**: Shared tags (weighted by number of shared tags)

### 4. GraphMemoryRetriever (`retriever.py`)

Hybrid retrieval combining FAISS and graph expansion:

```python
retriever = GraphMemoryRetriever(
    graph_builder=builder,
    embedding_model=clip_model,
    expand_hops=1,
    diversity_weight=0.3
)

# Retrieve diverse trajectories
results = retriever.retrieve(
    query_embedding=query_vec,
    query_text="Find free beginner courses",
    k=5,
    domain="Coursera"
)
```

**Retrieval Pipeline:**

1. **FAISS Phase**: Find top-K similar trajectories by embedding
2. **Graph Expansion**: Expand to 1-hop neighbors via shared tags  
3. **Diversity Selection**: Greedy selection maximizing tag coverage
4. **Digestion**: Pass all retrieved takeaways to VLM for task-specific guidance

## Usage

### Step 1: Build Graph Index from Trajectories

First, build the graph index from your **successful** trajectory files (same format as discrete memory).

**Option A: Comprehensive graph (all domains) - Recommended**

```bash
cd CoMEM-Agent-Inference

python -m graph_memory.build_graph_from_trajectories \
    --memory_data_dir "data/trajectories" \
    --output_path "graph_index/all_domains" \
    --model qwen2.5-vl \
    --tag_cache_path "graph_memory_cache/tags.json" \
    --summary_cache_path "graph_memory_cache/summaries.json"
```

**Option B: Single domain graph**

```bash
python -m graph_memory.build_graph_from_trajectories \
    --memory_data_dir "data/trajectories" \
    --output_path "graph_index/Amazon" \
    --domain Amazon \
    --model qwen2.5-vl
```

This will:
1. Load all `*.jsonl` trajectory files from `success/` folders only (no negative/positive)
2. Generate VLM takeaways for each trajectory (cached)
3. Extract semantic tags from each takeaway (cached)
4. Build a graph where nodes are trajectories and edges connect shared tags
5. Save the graph index to the specified output path

### Step 2: Enable Graph Memory in Agent

```bash
./scripts/runners/run_agent.sh \
    --eval_type webvoyager \
    --domain Amazon \
    --model qwen2.5-vl \
    --use_graph_memory \
    --graph_memory_index_path graph_index/all_domains \
    --graph_tag_cache_path graph_memory_cache/tags.json \
    --graph_expand_hops 1 \
    --graph_diversity_weight 0.3 \
    --graph_similar_num 5
```

Note: When using a comprehensive graph, the `--domain` flag filters retrieval to that domain's trajectories.

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_graph_memory` | False | Enable graph memory |
| `--graph_memory_index_path` | None | Path to saved graph index |
| `--graph_tag_cache_path` | `graph_memory_cache/tags.json` | Cache for extracted tags |
| `--graph_expand_hops` | 1 | Hops to expand in graph |
| `--graph_diversity_weight` | 0.3 | Diversity vs similarity weight |
| `--graph_similar_num` | 5 | Number of trajectories to retrieve |

### Programmatic Usage

```python
from graph_memory import TagExtractor, GraphBuilder, GraphMemoryRetriever

# Initialize components
extractor = TagExtractor(llm=tool_llm)
builder = GraphBuilder(tag_extractor=extractor)

# Build graph from existing trajectories
for traj in trajectories:
    builder.add_trajectory_from_data(
        traj_id=traj['id'],
        takeaway=traj['takeaway'],
        domain=traj['domain'],
        embedding=compute_embedding(traj['takeaway'])
    )

# Initialize retriever
retriever = GraphMemoryRetriever(
    graph_builder=builder,
    embedding_model=clip_model
)

# Retrieve for a new task
results = retriever.retrieve(
    query_text="Find undergraduate computer science programs",
    k=5
)

# Format for prompt injection
guidance = retriever.format_results(results)
```

## How It Works

### Example: Coursera Domain

Given 10 trajectories about finding courses on Coursera:

**Without Graph Memory (FAISS only):**
```
1. Filter by categories → uses #filter, #categories
2. Filter by level → uses #filter, #beginner  
3. Filter by subjects → uses #filter, #categories
4. Filter by type → uses #filter, #categories
5. Filter by price → uses #filter, #free
```
All 5 results focus on filtering! 😕

**With Graph Memory (FAISS + Graph Expansion):**
```
1. Filter by categories → #filter, #categories
2. Use search bar → #search, #search_bar (expanded via graph)
3. Check ratings → #rating, #sort (expanded for diversity)
4. Browse providers → #provider, #browse (new tag coverage)
5. Filter by level → #filter, #beginner
```
Diverse strategies! 🎉

### Diversity Selection Algorithm

```python
selected = []
covered_tags = set()

while len(selected) < k:
    best_score = -inf
    for candidate in candidates:
        # Relevance from FAISS
        faiss_score = faiss_scores[candidate.id]
        
        # Bonus for new tags not yet covered
        new_tags = candidate.tags - covered_tags
        diversity_bonus = len(new_tags) * diversity_weight
        
        score = faiss_score + diversity_bonus
        
    selected.append(best_candidate)
    covered_tags |= best_candidate.tags
```

## File Structure

```
graph_memory/
├── __init__.py                      # Module exports
├── tagged_trajectory.py             # TaggedTrajectory dataclass
├── tag_extractor.py                 # LLM-based tag extraction
├── graph_builder.py                 # NetworkX graph management
├── retriever.py                     # FAISS + graph expansion retrieval
├── build_graph_from_trajectories.py # Script to build graph from trajectory files
└── README.md                        # This file
```

## Integration with Existing Memory

Graph Memory can be used alongside discrete memory:

```bash
# Both enabled - graph memory appends to discrete memory guidance
--use_discrete_memory \
--use_graph_memory
```

The agent will:
1. Build discrete memory guidance (summarized takeaways)
2. Build graph memory guidance (diverse strategies)
3. Combine both in the system prompt

## Performance Considerations

- **Tag Extraction**: LLM calls are cached to avoid repeated extraction
- **Graph Updates**: O(n) for adding a trajectory (checking all existing nodes)
- **Retrieval**: O(k × n) for FAISS + O(k × degree) for graph expansion
- **Storage**: Graph stored as JSON + FAISS index as binary

## Future Improvements

1. **Hierarchical Tags**: Domain → Task Type → Strategy → Action
2. **Tag Clustering**: Auto-discover related tags
3. **Temporal Decay**: Prioritize recent successful strategies
4. **Cross-Domain Transfer**: Share strategies across similar domains

