# Hybrid Memory (Hierarchical Graph‑Latent Phases)

Purpose: phase‑level multimodal retrieval for GUI agents. Each Phase note encodes text + first keyframe into a single retrieval vector and a latent pack for reuse. Retrieval guides the agent with phase exemplars and optional "experience" inputs for continuous memory. Domains are inferred per trajectory by a VLM rather than hard‑coded from dataset folders.

Components
- schema.py: dataclasses for Domain / Trajectory / PhaseNote
- encoder.py: encode phase/trajectory/domain text+images → retrieval_vec + latent_pack (.npz)
- store.py: FAISS index (VectorIndex) + disk KV
- retriever.py: top‑K exemplar retrieval for current intent+page
- constructor.py: build Domain/Trajectory/PhaseNote objects from raw trajectories (VLM segmentation)
- pipeline.py: orchestrate construction, encoding, and storage
- prompts/: VLM system prompts for success/failure phase segmentation

Build index (example)
```bash
python scripts/hybrid/build_hybrid_index.py \
  --input_glob "data/downloaded_datasets/webvoyager_memory/Amazon/**/success/*.jsonl" \
  --output_dir hybrid_index/amazon \
  --vlm_base_url http://localhost:8000/v1 \
  --vlm_model "Qwen/Qwen2.5-VL-32B-Instruct"
```

Use in agent
- Flags: `--use_hybrid_memory --hybrid_index_dir hybrid_index/amazon --hybrid_k 3`
- Agent injects a compact "[Phase Exemplars]" block and optionally feeds exemplars as experience when continuous memory is enabled but no external memory is present.
- At retrieval time, the retriever can infer the most relevant semantic domain from the current intent + page description using stored domain embeddings.

Notes
- Latent packs store summary + keyframe paths and the retrieval vector (fp16), ready for later upgrades to unified VLM token latents.
- FAISS is required; there is no fallback.
- Trajectory and phase embeddings use only the first keyframe image (aligned for consistency).

Directory structure
```
hybrid_memory/
  README.md
  __init__.py
  schema.py
  encoder.py
  store.py
  retriever.py
  constructor.py
  pipeline.py
  prompts/
    success_phase_segmentation.txt
    failure_phase_segmentation.txt

scripts/hybrid/
  build_hybrid_index.py
  run_hybrid_eval.sh
  run_hybrid_ablate.sh
```

Functions by module
- schema.py
  - Domain (dataclass): id, category, domain, embedding, encoder_name/version, embedding_dim; `to_dict()`
  - Trajectory (dataclass): fields for pointers, summary and provenance; `to_dict()`, `set_source_hash_from_file()`
  - PhaseNeighbor (dataclass): to_phase_id, relation, count, success_count; `to_dict()`
  - PhaseNote (dataclass): phase metadata (label, span, summary, keyframes) + latent paths; `to_dict()`

- encoder.py
  - Base / Phase / Trajectory / Domain encoders plus `MemoryEncoder`:
    - PhaseEncoder: encode phase text+first keyframe → retrieval_vec + latent pack (`encode_phase_from_parts`, `encode_query`).
    - TrajectoryEncoder: encode trajectory text+first keyframe → retrieval_vec (`encode_direct`, `encode_query`).
    - DomainEncoder: encode `Domain` objects and domain queries (text only).
    - MemoryEncoder: unified API (`encode_phase_*`, `encode_trajectory_*`, `encode_domain*`).

- store.py
  - class VectorIndex(dim: int)
    - `add(item_id, vec, meta=None)`
    - `save(out_dir)`
    - `load(in_dir) -> VectorIndex` (classmethod)
    - `search(query_vec, k=5) -> List[(item_id, score)]`
  - class DiskKV(dir_path)
    - `set(key, value)`
    - `get(key) -> str | None`

- retriever.py
  - class HybridRetriever(index_dir, encoder=None)
    - Loads phase / trajectory indices, domains.json, and latent KV store.
    - Hierarchical retrieval:
      - (Optional) infer best domain via domain embeddings from intent + page description.
      - Retrieve top trajectories for (intent, domain) using trajectory index.
      - Retrieve top‑K phases for (intent, page, screenshot) restricted to those trajectories.
    - `retrieve(intent, image_b64, domain=None, k=3, page_description=None) -> List[exemplar_dict]`

- constructor.py
  - DomainConstructor: create/manage Domain objects
  - TrajectoryConstructor: build Trajectory from raw JSON + extract first keyframe
  - PhaseNoteConstructor: build PhaseNote objects from VLM segmentation output
  - MemoryConstructor: unified constructor that orchestrates VLM segmentation and object creation

- pipeline.py
  - HierarchicalMemoryStore: store for domains, trajectories, and phases with VectorIndex
  - `build_memory_index(...)`: orchestrates VLM segmentation, encoding, and storage

Prompts
- prompts/success_phase_segmentation.txt: system instructions for segmenting successful trajectories into phases (strict JSON).
- prompts/failure_phase_segmentation.txt: system instructions for segmenting failed trajectories and annotating failure analysis (strict JSON).
