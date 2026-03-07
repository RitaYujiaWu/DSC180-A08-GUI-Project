# Action Scaling System

A trajectory-based learning system that helps GUI agents learn from past experiences by matching similar states and evaluating action outcomes.

## Overview

This system builds a searchable memory of **state-action-state** transitions from historical trajectories. When the agent encounters a new state, it retrieves similar past states and analyzes what actions succeeded or failed in those contexts.

## Core Concept

```
Historical Trajectory: State1 --[Action]--> State2
                         ↓
                    Extract & Store
                         ↓
Current State ----[CLIP Match]----> Similar State1 from history
                                           ↓
                                    [LLM Evaluates]
                                           ↓
                               "This action succeeded/failed because..."
                                           ↓
                                  Action Recommendations
```

## Components

### 1. `trajectory_analyzer.py`

Main system class that manages trajectory memory and retrieval.

**Key Features:**
- **Load Trajectories**: Parse JSONL files into state-action-state tuples
- **Embed States**: Generate CLIP embeddings for all state screenshots
- **Find Similar States**: CLIP-based similarity search (cosine similarity)
- **Evaluate Actions**: LLM analyzes whether actions succeeded or failed
- **Generate Recommendations**: Aggregated guidance on what to do/avoid

**Usage:**
```python
from trajectory_analyzer import TrajectoryAnalyzer

# Initialize analyzer
analyzer = TrajectoryAnalyzer(
    trajectory_dir="/path/to/trajectories",
    state_embedding_path="state_embeddings/webvoyager_memory.npy",
    tool_llm=llm_client,
    model_name="openai/clip-vit-base-patch32"
)

# Load and embed trajectories
analyzer.load_trajectories()
analyzer.embed_states()

# Get recommendations for current state
recommendations = analyzer.get_action_recommendations(current_state_image)
```

### 2. `help_functions.py`

HTML generation utilities for visualizing similar states and evaluations.

**Function:**
- `generate_html_content()`: Creates interactive HTML reports showing:
  - Current state screenshot
  - Top-k similar historical states (with similarity scores)
  - Before/after screenshots for each similar state
  - LLM evaluation for each action (success/risk/recommendation)

### 3. `prompts/evaluate_single_action.txt`

LLM prompt template for evaluating individual actions.

**Output Format:**
```json
{
  "action_evaluation": {
    "success": true/false,
    "evaluation_confidence": 0.0-1.0,
    "action_quality": "excellent/good/fair/poor",
    "reasoning": "Why this action succeeded/failed",
    "risk_level": "low/medium/high",
    "applicability_to_current_state": "high/medium/low",
    "lessons_learned": ["Key lesson 1", "Key lesson 2"],
    "recommendation": "repeat/adapt/avoid",
    "adaptation_suggestions": ["How to adapt", "Modifications"]
  }
}
```

## Directory Structure

```
action_scaling/
├── trajectory_analyzer.py          # Main analyzer class
├── help_functions.py                # HTML generation utilities
├── prompts/
│   └── evaluate_single_action.txt  # LLM evaluation prompt
├── state_embeddings/                # Cached embeddings
│   ├── webvoyager_memory.json      # State metadata
│   └── webvoyager_memory.npy       # CLIP embeddings (numpy)
└── check_similar_states/            # HTML visualization outputs
    └── similar_states_*.html        # Timestamped reports
```

## Workflow

### Step 1: Build Trajectory Memory (Offline)

```python
analyzer = TrajectoryAnalyzer(trajectory_dir="webvoyager_results")

# Extract state-action-state tuples from all trajectories
analyzer.load_trajectories()  
# Output: List of (state1_img, action, state2_img) tuples

# Generate CLIP embeddings for all state1 images
analyzer.embed_states()
# Output: state_embeddings/webvoyager_memory.{json,npy}
```

### Step 2: Query at Test Time (Online)

```python
# Agent encounters new state
current_state_img = "data:image/png;base64,..."

# Find top-10 similar historical states
similar_states = analyzer.find_similar_states(
    current_state_img, 
    top_k=10
)
# Returns: [(StateActionState, similarity_score), ...]

# Get LLM analysis and recommendations
recommendations = analyzer.get_action_recommendations(current_state_img)
# Returns: Structured JSON with action guidance
```

### Step 3: Visualize Results

HTML reports are automatically saved to `check_similar_states/`:

```html
📊 Current State
   └── [Screenshot]

📋 Similar State 1 (similarity: 0.89)
   ├── Before: [Screenshot]
   ├── Action: "CLICK element_123"
   ├── After: [Screenshot]
   └── Evaluation:
       ├── Success: ✓
       ├── Quality: excellent
       ├── Risk: low
       ├── Reasoning: "Action successfully navigated to target page"
       └── Recommendation: repeat
```

## State Representation

**StateActionState Dataclass:**
```python
@dataclass
class StateActionState:
    state1_image: str        # Base64 screenshot before action
    action: str              # Action taken (JSON or text)
    state2_image: str        # Base64 screenshot after action
    trajectory_path: str     # Source trajectory file
    round_index: int         # Position in trajectory
```

## CLIP Embedding Cache

**Format:** `state_embeddings/{name}.{json,npy}`

- **`.json`**: Metadata for each state
  ```json
  [
    {
      "trajectory_path": "/path/to/task.jsonl",
      "round_index": 5
    }
  ]
  ```

- **`.npy`**: NumPy array of CLIP embeddings (shape: `[N, embedding_dim]`)

**Benefits:**
- Fast retrieval (cosine similarity on precomputed embeddings)
- Memory efficient (only metadata in JSON, embeddings in binary)
- Resumable (skip re-embedding on subsequent runs)

## Key Methods

### `TrajectoryAnalyzer.load_trajectories()`
Parses JSONL trajectory files and extracts state-action-state tuples.

### `TrajectoryAnalyzer.embed_states()`
Generates CLIP embeddings for all state1 images. Saves to disk for reuse.

### `TrajectoryAnalyzer.find_similar_states(current_state, top_k)`
Returns top-k most similar historical states using CLIP cosine similarity.

### `TrajectoryAnalyzer.get_action_recommendations(current_state)`
End-to-end pipeline:
1. Find similar states (CLIP)
2. Evaluate each action individually (LLM)
3. Aggregate into structured recommendations
4. Save HTML visualization

### `TrajectoryAnalyzer._evaluate_single_action(sas, current_state)`
LLM analyzes a single state-action-state tuple:
- Compares current state with historical state1
- Examines state1 → state2 transition
- Determines success/failure and why
- Assesses applicability to current situation

## Output Example

**Recommendations JSON:**
```json
{
  "current_situation": "User is on product listing page with filters visible",
  "historical_insights": [
    "Similar states often succeed with filter application",
    "Clicking 'Apply Filter' typically leads to refined results"
  ],
  "action_guidance": {
    "recommended_actions": [
      "Apply price filter with clear range",
      "Verify filter options before clicking"
    ],
    "risky_actions": [
      "Clicking without setting filter values",
      "Rapid successive filter changes"
    ]
  },
  "practical_suggestions": [
    "Set price range first, then apply",
    "Wait for page load after filter application"
  ]
}
```

## Integration with Agent

```python
# In agent decision loop
def select_action(self, current_state_image):
    # Get recommendations from trajectory memory
    recommendations = self.analyzer.get_action_recommendations(
        current_state_image
    )
    
    # Use recommendations to:
    # 1. Filter candidate actions (avoid risky ones)
    # 2. Prioritize successful patterns
    # 3. Guide LLM reasoning with historical insights
    
    prompt = f"""
    Current State: [image]
    
    Historical Experience:
    {recommendations['historical_insights']}
    
    Recommended Actions:
    {recommendations['action_guidance']['recommended_actions']}
    
    Actions to Avoid:
    {recommendations['action_guidance']['risky_actions']}
    
    What action should I take?
    """
    
    return llm.generate(prompt)
```


## Notes

- **Embedding Cache**: Always check if embeddings exist before regenerating
- **Trajectory Format**: Expects JSONL with `rounds` structure containing `messages` with image URLs
- **HTML Reports**: Useful for debugging and understanding retrieval quality
- **Similarity Threshold**: Adjust `top_k` based on memory size and desired precision

