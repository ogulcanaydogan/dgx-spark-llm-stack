# Visibility & Community Launch Strategy (Stars KPI)

This document defines the go-to-market plan for `dgx-spark-llm-stack` after roadmap closure.

## Current Positioning (Truth-Only)

Single narrative:

> Production-ready DGX Spark (`GB10`, `sm_121`) LLM stack with pre-built wheels, validated benchmarks, operational runbooks, and explicit workaround guidance for known gaps.

Current status references (as of `2026-04-03`):
- Roadmap: `43/43` complete
- TensorRT-LLM attention-sinks: legacy fail + stable pass validated (`artifacts/benchmarks/tensorrt-llm-attention-sinks-2026-04-03.json`)

## Product Claims We Can Safely Make

- PyTorch + CUDA stack is reproducibly installable on DGX Spark via release wheels.
- Inference/training/quality benchmarks are published with methodology and artifacts.
- vLLM serving path is documented and operationally smoke-tested.
- Known incompatibilities are documented with deterministic fallbacks:
  - Triton: `TRITON_PTXAS_PATH` env fix
  - flash-attention: SDPA fallback
  - TransformerEngine FP8: BF16 fallback
  - TensorRT-LLM attention-sinks: avoid legacy tag, use validated stable path

## 7-Day Launch Sequence (Primary KPI: GitHub Stars)

### Day 1
- X thread + GitHub announcement post
- CTA: `./install.sh` and share results/issues

### Day 2
- Reddit `r/LocalLLaMA`
- Focus: practical setup + benchmark links + compatibility matrix

### Day 3
- Reddit `r/nvidia`
- Focus: DGX Spark hardware reality + reproducible stack

### Day 4
- Show HN
- Focus: what changed from “unsupported” to “usable in practice”

### Day 5-7
- NVIDIA Developer Forum replies on DGX Spark/GB10 threads
- Hugging Face model discussions (Qwen/Llama/Gemma) where DGX Spark support is asked

## CTA Standard

Use the same closing CTA on every channel:

- "Start with `./install.sh`, run `python scripts/verify_install.py`, and share your output."
- "If you use DGX Spark / GB10 (`sm_121`), report your model + config + results in Issues."

## Measurement Plan

Primary KPI:
- GitHub stars delta (daily + 7-day cumulative)

Secondary KPIs:
- Release download trend (wheel assets)
- GitHub traffic/views and unique visitors

Suggested checks:

```bash
gh api repos/ogulcanaydogan/dgx-spark-llm-stack/traffic/views
gh release view --json assets
```

## Messaging Guardrails

- Do not claim universal compatibility.
- Do not claim flash-attention or TransformerEngine FP8 works on `sm_121`.
- Always link proof artifacts for benchmark/performance statements.
- Keep “works with caveats” explicit where applicable.
