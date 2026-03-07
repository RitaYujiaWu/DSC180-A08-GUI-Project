# Failure Trajectory Analysis

## Overview

This module contains the **failure trajectory analysis pipeline** used to understand why a GUI-agent episode failed.

Its purpose is not to execute tasks, but to **analyze completed trajectories after the fact** and extract useful failure signals that can support contrastive memory, error diagnosis, and future correction.

At a high level, the pipeline does four things:

1. Detect errors at the **step level**
2. Identify the **critical failure point**
3. Evaluate the final task outcome
4. Mine failure-aware experience and stopping signals

These components are useful for building a **contrastive system** because successful trajectories tell the agent what to do, while failed trajectories help explain **what not to do**, **where things went wrong**, and **how failure cascaded across steps**.

---

# What This Module Is For

Failure analysis is a core part of a contrastive memory system.

A successful-only memory bank can tell the agent what worked before, but it cannot explain:

- which earlier decisions caused a failed trajectory
- whether an error came from memory, reflection, planning, action, or the environment
- what correction would have changed the trajectory
- when the agent should stop instead of continuing unproductively

This module addresses those gaps by providing tools for:

- **fine-grained step analysis**
- **critical error identification**
- **final result evaluation**
- **experience memory extraction**
- **early stop detection**

Together, these components make failure trajectories useful rather than just discarded.

---

# High-Level Pipeline

The failure analysis workflow can be understood as the following pipeline:

```
Failed Trajectory
│
▼
Step-Level Error Detection
│
▼
Critical Failure Point Identification
│
▼
Final Outcome / Error Evaluation
│
▼
Failure-Aware Memory and Query Mining
│
▼
Signals for Contrastive Reasoning
```

Each stage contributes a different kind of supervision.

---

# Folder Contents

This extracted module currently contains:

```
failure_traj_analysis/
├── internal_world_model
├── critical_error_detection.py
├── early_stop.py
├── evaluator.py
├── experience_memory.py
└── fine_grained_analysis.py

```
---

# File-by-File Description

## 1. `fine_grained_analysis.py`

This file implements **Phase 1: step-level error type detection**.

Its main role is to analyze a trajectory step by step and determine whether each step contains an error in one or more internal modules, such as:

- memory
- reflection
- planning
- action
- system
- others

### Main purpose

For each step in a failed trajectory, the analyzer tries to answer:

- Did the agent make an error at this step?
- Which internal module was responsible?
- What type of error was it?
- What evidence supports that conclusion?

### Key design

The analysis is **module-aware**. Instead of only looking at the final failure, it breaks a step into internal reasoning components such as:

- `<memory>`
- `<reflection>`
- `<plan>`
- `<action>`

and evaluates each one independently.

It also uses the **current screenshot** and the **next screenshot** to check whether the agent’s output actually led to the expected transition.

### Main outputs

The output is a structured step-by-step analysis containing:

- step number
- detected errors for each module
- evidence
- reasoning
- step summary

This becomes the input to the next phase: **critical error identification**.

### Why it matters

This file provides the **fine-grained supervision layer** of the failure-analysis system. Instead of saying “the task failed,” it says:

- **where**
- **how**
- **in which module**
- **with what evidence**

the failure emerged.

---

## 2. `critical_error_detection.py`

This file implements **Phase 2: critical failure point identification**.

After Phase 1 has produced step-by-step module-level errors, this file tries to find the **single earliest critical error** that caused the overall task to fail.

### Main purpose

Not every detected error is equally important.

A failed trajectory may contain many downstream mistakes, but this module asks:

> What was the **first truly decisive error** that set the agent on an unrecoverable path to failure?

### Core idea

The analysis is explicitly **causal** and **trajectory-level**.

Rather than choosing the most severe-looking error locally, it tries to identify:

- the **root cause**
- the **earliest critical step**
- the **module responsible**
- the **error type**
- the **cascading effects** on later steps
- the **correction guidance** that could have changed the outcome

### Key design choices

This module includes several important safeguards:

- It skips critical-error analysis for successful tasks.
- It retries analysis if the model produces an invalid result such as a **step-1 memory or reflection error**, because step 1 has no prior history to remember or reflect on.
- It validates the returned error type against allowed module definitions.
- It uses robust JSON parsing with fallback repair logic and regex recovery when the model output is malformed.

### Main outputs

The main result is a `CriticalError` object containing fields such as:

- `critical_step`
- `critical_module`
- `error_type`
- `root_cause`
- `evidence`
- `correction_guidance`
- `cascading_effects`
- `confidence`

### Why it matters

This file provides the **root-cause signal** for contrastive learning and memory construction.

Instead of only knowing that a trajectory failed, we can identify:

- the **decisive failure point**
- how that failure propagated
- what correction would likely have changed the outcome

That makes failure trajectories much more useful for future reasoning.

---

## 3. `evaluator.py`

This file implements a **trajectory evaluation and failure interpretation system**.

It is used to determine whether the final task result was successful and, if not, to analyze the likely cause of failure from screenshots, actions, and rendered HTML.

### Main purpose

This file supports several evaluation tasks:

- judging whether a completed task was **successful**
- extracting the final answer from rendered trajectories
- analyzing the likely **failure reason**
- locating the likely **error step**
- generating **related queries** for training or data augmentation

### Main components

### `Evaluator`

A base evaluation class that provides utilities for working with trajectories, such as fetching the last action or last state.

### `LLMEvaluator`

A higher-level evaluator that uses a vision-language model to:

- compare the task instruction against the final screenshots and textual answer
- return a success verdict
- diagnose likely reasons for failure
- locate critical error steps
- generate new related queries based on failure types

### Key capabilities

#### Final task evaluation

The evaluator checks whether the final rendered trajectory actually satisfies the original task, using screenshots and the extracted final answer.

#### Failure reasoning

It can analyze a failed trajectory and generate structured explanations such as:

- failure tag
- failure reason
- supporting evidence
- correct action

#### Error-step localization

It can locate the step where failure most likely occurred and suggest what the correct action should have been instead.

#### Query generation

For a failed action, it can generate related or grounded task queries that preserve the same interaction type, such as:

- click failure
- type failure
- selection failure
- popup handling failure
- sort failure
- filter failure

### Why it matters

This file turns failed trajectories into **interpretable supervision**.

It helps answer:

- Did the task actually fail?
- Why did it fail?
- At which step?
- What should have happened instead?
- What related examples could help train the agent?

This is highly useful for building a richer failure memory bank.

---

## 4. `experience_memory.py`

This file implements **experience memory retrieval** using a pooled memory bank and FAISS-based similarity search.

Although it is broader than failure-only analysis, it is important because it provides the retrieval machinery that can later be extended into a **contrastive success/failure memory system**.

### Main purpose

The memory system loads trajectory data from training data folders, embeds them using CLIP-based similarity models, and retrieves similar past examples for the current task.

### Key functions

### Loading memory

The file scans the training data directory, collects trajectories from `success` folders, and stores them in a unified memory pool.

Each memory includes metadata such as:

- file path
- task description
- dataset
- domain
- prefixed query
- optional base64 screenshot

### Embedding and indexing

It builds FAISS indices using:

- text embeddings
- or multimodal embeddings

This enables efficient nearest-neighbor retrieval.

### Similar conversation retrieval

Given a current query and optional screenshot, it retrieves similar past trajectories.

### Action parsing

It also contains robust logic to parse action outputs from response text or JSON-like responses, with fallback to LLM-based parsing.

### Experience memory construction

It can construct a formatted experience-memory string summarizing example tasks and action reasoning from retrieved conversations.

### Why it matters for failure analysis

This file provides the **retrieval substrate** needed for contrastive systems.

Even though the current implementation loads successful memories, the same structure is highly relevant for failure-aware extensions because it already supports:

- indexed retrieval
- multimodal search
- action extraction
- trajectory summarization

This makes it a natural foundation for building **contrastive memory between successful and failed trajectories**.

---

## 5. `early_stop.py`

This file implements **early stopping logic** for the agent.

### Main purpose

It decides whether the current trajectory should stop early instead of continuing.

### What it checks

The current implementation checks two main cases:

1. **Maximum step limit reached**
2. **Repeated parsing failures**

If the trajectory exceeds the maximum number of steps, it should stop.

If the last `k` actions are parsing failures or null actions, it should also stop.

### Why it matters for failure analysis

Early stopping is important because not all failures come from reasoning errors. Some trajectories fail because the agent gets stuck in an unproductive loop, for example:

- repeated unparsable actions
- repeated null actions
- step budget exhaustion

These are meaningful failure signals in their own right.

In a contrastive framework, early-stop conditions help distinguish between:

- recoverable reasoning mistakes
- irrecoverable dead-end trajectories

---

# How These Files Work Together

These files form a layered failure-analysis stack.

## Step 1: Fine-grained error detection

`fine_grained_analysis.py` analyzes each step and detects module-level errors.

Output:
- per-step module errors
- evidence
- reasoning

## Step 2: Critical error identification

`critical_error_detection.py` takes Phase 1 results and identifies the **single earliest root-cause error**.

Output:
- critical step
- critical module
- root cause
- correction guidance
- cascading effects

## Step 3: Outcome and failure interpretation

`evaluator.py` determines whether the task actually succeeded and can produce:

- failure tags
- failure reasons
- failure step localization
- correction suggestions
- related training queries

## Step 4: Memory and retrieval support

`experience_memory.py` provides retrieval and structured memory extraction, which can be used to connect failure analysis back into the agent’s memory system.

## Step 5: Stop-signal supervision

`early_stop.py` provides signals for trajectories that should terminate early due to repeated unproductive behavior.

---

# Why This Matters for a Contrastive System

A contrastive system depends on both sides of the contrast:

- **successful trajectories** show what to imitate
- **failed trajectories** show what to avoid

This extracted failure-analysis code is valuable because it transforms failures into **structured, reusable knowledge**.

Instead of storing a failed trajectory as raw data, the system can extract:

- error type
- responsible module
- critical step
- root cause
- supporting evidence
- correction guidance
- related failure queries
- early stop signals

That information can later support:

- contrastive retrieval
- failure-aware planning
- candidate-action rejection
- root-cause memory
- task-specific correction prompts

In other words, this module helps turn failures into **actionable negative supervision**.

---

# Typical Usage Flow

A typical usage pattern looks like this:

## 1. Run step-level analysis

Use `fine_grained_analysis.py` to process a failed rendered trajectory and produce module-level error annotations.

## 2. Run critical error detection

Use `critical_error_detection.py` on the Phase 1 output and original trajectory to identify the earliest decisive failure.

## 3. Evaluate the outcome

Use `evaluator.py` to verify whether the trajectory truly failed and to produce a higher-level failure explanation.

## 4. Mine memory or corrective signals

Use:
- `experience_memory.py` for retrieval and memory construction
- `early_stop.py` for dead-end stopping signals

## 5. Feed the results into a contrastive framework

The extracted failure signals can then be integrated into:

- contrastive memory banks
- internal world models
- correction prompts
- failure-aware candidate scoring

---

# Strengths of This Design

This failure-analysis stack has several strong properties.

## 1. It is structured

The system does not treat failure as a single label. It decomposes failure into:

- step
- module
- type
- cause
- effect

## 2. It is causal

The critical-error phase focuses on the **earliest decisive root cause**, not just the final symptom.

## 3. It is multimodal

The analysis can use both:

- text/action traces
- screenshots before and after steps

## 4. It is reusable

The outputs are structured JSON-like artifacts that can be reused for:

- memory
- evaluation
- training data generation
- error correction

## 5. It aligns naturally with internal world models

The extracted signals are exactly the kind of evidence needed for “think-before-acting” systems:

- what tends to succeed
- what tends to fail
- how failure begins
- which corrections matter most

---

# Limitations and Notes

A few practical limitations are worth noting.

## LLM dependence

Several components rely on a vision-language model for:

- error detection
- critical error identification
- evaluation
- failure explanation

This makes the analysis flexible, but also means outputs can vary depending on prompting and model quality.

## Parsing robustness is necessary

Because model outputs are not always clean JSON, the code includes repair and fallback parsing logic. This is important for reliability.

## Current memory implementation is not fully contrastive yet

`experience_memory.py` mainly loads successful trajectories, so it is a retrieval foundation rather than a full success/failure contrastive bank.

## Some components are more analysis-oriented than runtime-oriented

Files like `critical_error_detection.py` and `fine_grained_analysis.py` are primarily offline analysis tools rather than direct inference-time agent modules.

---

# Summary

This failure trajectory analysis module provides the **negative side of contrastive learning for GUI agents**.

It helps transform raw failed trajectories into structured signals such as:

- module-level errors
- critical root causes
- failure tags
- corrective guidance
- memory retrieval cues
- early-stop conditions

Together, these tools make failure trajectories useful for:

- contrastive memory
- failure-aware planning
- internal world models
- post-hoc diagnosis
- future correction

In short, this module answers the question:

> not just **that** the agent failed, but **where**, **why**, and **how that failure could have been avoided**.

