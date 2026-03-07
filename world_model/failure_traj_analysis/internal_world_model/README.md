# Internal World Model (A supplement for the contrasive_experience folder and share most of files)

## Overview

The **Internal World Model (IWM)** enables the GUI agent to **think before acting** by evaluating multiple possible actions before executing one.

Instead of immediately predicting the next action from the policy model, the internal world model introduces an intermediate reasoning step that:

1. Generates multiple candidate actions
2. Retrieves similar **successful** and **failed** trajectories from memory
3. Evaluates candidates using this contrastive evidence
4. Selects the best action

This allows the agent to leverage past experiences and avoid repeating known mistakes.

Importantly, the internal world model operates entirely at **inference time** and does **not require additional training**.

---

# High-Level Architecture

The reasoning pipeline is:

```
Current State
     │
     ▼
Candidate Action Generation
     │
     ▼
Contrastive Memory Retrieval
     │
     ▼
Candidate Evaluation
     │
     ▼
Best Action Selection
     │
     ▼
Agent Executes Action
```

This introduces a **planning layer between perception and action**.

---

# Reasoning Process

## 1. Candidate Action Generation

The system first generates **multiple possible next actions** using the language model.

Instead of committing to a single action, the agent proposes several hypotheses.

Example:

```
1. Click the search bar
2. Scroll down the page
3. Open the navigation menu
```

These candidate actions are later evaluated before execution.

---

## 2. Contrastive Memory Retrieval

The system retrieves examples from the **Reasoning Bank**, including:

- **Successful trajectories**
- **Failure trajectories**

Retrieval is conditioned on:

- task intent
- domain / website
- recent actions
- page URL
- optionally the current screenshot (multimodal retrieval)

This provides **experience-based evidence** for evaluating candidate actions.

---

## 3. Candidate Evaluation

Each candidate action is evaluated using the retrieved memories.

Two scoring strategies are supported.

### Heuristic Scoring

Measures textual similarity between:

- candidate action
- retrieved success cases
- retrieved failure cases

Score formula:

```
final_score = alpha * success_score - beta * failure_score
```

### LLM-Based Scoring

The language model explicitly evaluates the candidate action.

Example structured output:

```
SUCCESS_ALIGNMENT: 7
FAILURE_RISK: 2
FINAL_SCORE: 5
REASONING: Clicking the search bar matches successful patterns and avoids navigation failures.
```

---

## 4. Action Selection

The candidate with the **highest score** is selected and returned to the agent.

The agent then executes the action through the normal execution pipeline.

---

# Folder Structure

```
internal_world_model/
│
├── __init__.py
├── planner.py
├── schemas.py
├── contrastive_retriever.py
├── candidate_generator.py
├── candidate_scorer.py
├── prompt_builder.py
└── README.md
```

---

# File Descriptions

## planner.py

The **main controller** of the internal world model.

Responsibilities:

- Generate candidate actions
- Retrieve contrastive memories
- Score candidates
- Select the best action

Main class:

```
InternalWorldModelPlanner
```

Main method:

```
select_action(messages, trajectory, intent, meta_data)
```

Returns a `PlannerOutput` containing:

- selected action
- candidate scores
- retrieved cases
- debugging information

---

## contrastive_retriever.py

Handles **contrastive retrieval** from the Reasoning Bank.

Retrieves:

- top-k **success cases**
- top-k **failure cases**

based on:

- task intent
- domain
- recent actions
- page URL
- screenshot (optional)

Main class:

```
ContrastiveRetriever
```

Main method:

```
retrieve(intent, trajectory, meta_data, current_image)
```

Returns:

```
(success_cases, failure_cases)
```

Each retrieved item is converted into a `RetrievedCase` object.

---

## candidate_generator.py

Generates **multiple candidate actions** before execution.

Main class:

```
CandidateGenerator
```

Main method:

```
generate(intent, trajectory, meta_data, messages)
```

Steps:

1. Build candidate-generation prompt
2. Query the language model
3. Parse numbered responses
4. Deduplicate candidates

Output:

```
List[CandidateAction]
```

---

## candidate_scorer.py

Scores candidate actions using **contrastive evidence**.

Main class:

```
CandidateScorer
```

Main methods:

```
score_candidates(...)
select_best(...)
```

Supports two scoring modes:

- **heuristic scoring**
- **LLM-based scoring**

---

## prompt_builder.py

Centralizes prompt construction for:

- candidate generation
- candidate scoring

Functions:

```
build_candidate_generation_prompt(...)
build_candidate_scoring_prompt(...)
format_retrieved_cases(...)
```

Keeping prompts centralized makes the system easier to modify and maintain.

---

## schemas.py

Defines shared data structures used across the internal world model.

Key dataclasses include:

### RetrievedCase

Represents a retrieved trajectory example.

Fields include:

- trajectory id
- label (success or failure)
- task
- summary
- similarity score

### CandidateAction

Represents a generated action candidate.

### CandidateScore

Stores evaluation results for a candidate action.

### PlannerOutput

The final output of the planner containing:

- selected action
- candidate scores
- retrieved cases
- debug information

---

# Integration with the Agent

The internal world model is integrated inside:

```
agent/agent.py
```

Execution flow inside the agent:

```
_prepare_messages()
        │
        ▼
InternalWorldModelPlanner.select_action()
        │
        ▼
Selected action returned
        │
        ▼
_process_response()
        │
        ▼
Environment execution
```

The planner runs **before the action is executed**, allowing the agent to reason about multiple options.

---

# Configuration

The internal world model is controlled by CLI flags defined in `argument_parser.py`.

| Argument | Description |
|--------|-------------|
| `--use_internal_world_model` | Enable the internal world model |
| `--iwm_k_candidates` | Number of candidate actions |
| `--iwm_topk_success` | Number of retrieved successful cases |
| `--iwm_topk_failure` | Number of retrieved failure cases |
| `--iwm_score_method` | Scoring method (`heuristic` or `llm`) |
| `--iwm_alpha` | Success weight |
| `--iwm_beta` | Failure penalty weight |
| `--iwm_use_multimodal` | Enable screenshot-aware retrieval |

Example run:

```
python run.py \
    --benchmark webvoyager \
    --use_reasoning_bank True \
    --use_internal_world_model True \
    --iwm_k_candidates 3 \
    --iwm_score_method heuristic
```

---

# Debugging Information

The planner returns detailed debugging metadata:

```
{
  "candidates": [...],
  "selected_action": "...",
  "retrieved_success_ids": [...],
  "retrieved_failure_ids": [...],
  "candidate_scores": [...]
}
```

This information is stored in:

```
meta_data["iwm_debug"]
```

and can be used for debugging and analysis.

---

# Future Improvements

Possible extensions include:

### Learned World Models
Train a model to predict success probabilities instead of heuristic scoring.

### Transition-Based Reasoning
Use `(state, action, next_state)` trajectories to evaluate action outcomes.

### Multi-Step Planning
Generate deeper action sequences rather than single-step candidates.

### Improved Multimodal Retrieval
Use stronger vision-language embeddings to improve screenshot-based retrieval.

---

# Summary

The Internal World Model introduces a **reasoning layer between perception and action**.

Instead of acting immediately, the agent follows the loop:

```
observe → hypothesize → compare with memory → evaluate → act
```

This improves decision-making by leveraging past successes and avoiding known failures.
