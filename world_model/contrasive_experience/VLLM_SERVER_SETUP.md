# vLLM Server Setup Guide

## Currently Running Servers

| Model | GPUs | Port | Status |
|-------|------|------|--------|
| Qwen2.5-VL-7B | 0-1 | 8000 | ✓ Running |
| UI-Ins-7B | 2-3 | 8006 | ✓ Running |

## Start Your Own Models (Matching run_agent.sh - Ports 8010/8011)

### Main Agent Model (Port 8010)

```bash
Terminal 1: Main Agent Model (Port 8010)

  conda activate gui-agent
  CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-VL-7B-Instruct \
      --port 8010 \
      --tensor-parallel-size 2 \
      --max-model-len 81920 \
      --gpu-memory-utilization 0.9


```

### Grounding Model (Port 8011)

```bash
  Terminal 2: Grounding Model (Port 8011) ← This is what you need to fix!

  conda activate gui-agent
  CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server \
      --model Tongyi-MiA/UI-Ins-7B \
      --port 8011 \
      --tensor-parallel-size 2 \
      --max-model-len 81920 \
      --gpu-memory-utilization 0.9
```

## Verify Servers

```bash
# Check main agent (port 8010)
curl http://localhost:8010/v1/models

# Check grounding (port 8011)
curl http://localhost:8011/v1/models

# Check GPUs
nvidia-smi
```

## Quick Reference

- **Ports**: 8010 (main agent), 8011 (grounding)
- **Max length**: 81920 (80K tokens)
- **GPUs**: 4-5 for main agent, 6-7 for grounding
- **Config file**: `agent/llm_config.py` (already updated to use 8010/8011)
