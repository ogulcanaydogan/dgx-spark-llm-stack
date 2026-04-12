# DGX Spark LLM Stack

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-ARM64-green.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://python.org)
[![GPU](https://img.shields.io/badge/GPU-GB10_(sm__121)-76B900.svg)]()
[![Benchmark Baseline](https://img.shields.io/badge/Baseline-2026--03--13-0A66C2.svg)]()
[![Inference 14B NF4](https://img.shields.io/badge/Inference_14B_NF4-18.91_tok%2Fs-0A7D00.svg)]()
[![Inference 32B FP4](https://img.shields.io/badge/Inference_32B_FP4-9.79_tok%2Fs-006D77.svg)]()
[![Inference 72B NF4](https://img.shields.io/badge/Inference_72B_NF4-3.80_tok%2Fs-8A2BE2.svg)]()
[![Training 7B LoRA](https://img.shields.io/badge/Training_7B_LoRA-2.58_samples%2Fs-0057B8.svg)]()

**PyTorch, Triton, flash-attention, BitsAndBytes** — pre-built wheels and reproducible build scripts for **NVIDIA DGX Spark** (GB10, sm_121, Blackwell, CUDA 13.0, Python 3.12, ARM64).

> Can't `pip install torch` on your DGX Spark? You're in the right place.

## The Problem

DGX Spark ships with GB10 — a Blackwell GPU with compute capability `sm_121`. Most ML frameworks don't officially support this architecture yet:

- **PyTorch**: Official wheels max out at `sm_120`, emit warnings on `sm_121`
- **Triton**: bundled `ptxas` may fail on `sm_121a`; use CUDA 13.0 `ptxas` via env override
- **flash-attention**: No `sm_121` kernels, compilation fails
- **vLLM**: Requires Docker or source builds for Blackwell
- **TransformerEngine**: MXFP8 broken on this arch

This repo provides **build scripts, pre-built wheels, compatibility info, and benchmarks** so you can run a full LLM stack on your DGX Spark without fighting the toolchain.

## Quick Start

```bash
git clone https://github.com/ogulcanaydogan/dgx-spark-llm-stack.git
cd dgx-spark-llm-stack
./install.sh
```

This downloads pre-built wheels from GitHub Releases and installs the full stack. After installation, it runs verification automatically.

## Compatibility Matrix

| Library | Version | sm_121 Status | Notes |
|---------|---------|---------------|-------|
| PyTorch | 2.9.1 | ⚠️ Warning, works | Official max sm_120; our wheel targets sm_121 |
| Triton | 3.5.1 | ⚠️ Works with env fix | Set `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` on DGX Spark |
| flash-attention | 2.7+ | ❌ Not supported | Use SDPA fallback (see docs) |
| BitsAndBytes | 0.49+ | ✅ Works | FP4/NF4 quantization tested |
| vLLM | 0.8+ | ⚠️ Docker only | Build from source or use NGC container |
| llama.cpp | Latest | ✅ Works well | Best option for inference |
| TensorRT-LLM | 0.9+ | ⚠️ Partial | Legacy `1.1.0rc1` fails with SM90-only assertion; stable `1.2.0` pass validated (2026-04-03) |
| TransformerEngine | - | ❌ Broken | MXFP8 training unsupported |
| Unsloth | Latest | ✅ Works | Recommended for fine-tuning |
| transformers | 4.48+ | ✅ Works | Standard HF stack |
| PEFT / LoRA | Latest | ✅ Works | QLoRA with BitsAndBytes OK |
| TRL | Latest | ✅ Works | SFT, DPO, ORPO all work |

Full details: [COMPATIBILITY.md](COMPATIBILITY.md)

## Pre-built Wheels

Check [GitHub Releases](https://github.com/ogulcanaydogan/dgx-spark-llm-stack/releases) for pre-built wheels:

- `torch-2.9.1+cu130-cp312-cp312-linux_aarch64.whl`
- `bitsandbytes-0.49.0+cu130-cp312-cp312-linux_aarch64.whl`
- `SHA256SUMS` (checksum manifest for release artifacts)

These are built on DGX Spark with CUDA 13.0, Python 3.12, GCC 13.3.
`install.sh` verifies release wheel checksums before installation.

## Build from Source

If you prefer to build everything yourself:

```bash
# Set up environment
source configs/env.sh

# Build all components (~6 hours total)
./build/build_all.sh

# Or build individually
./build/build_pytorch.sh      # ~4 hours
./build/build_triton.sh       # ~30 min
./build/build_flash_attn.sh   # ~20 min
./build/build_bitsandbytes.sh # ~10 min
```

## vLLM Container (Phase 4)

Deterministic DGX Spark container flow using NGC PyTorch base + multi-stage vLLM source build.

Build image:

```bash
docker build \
  --build-arg VLLM_REF=v0.18.0 \
  -f docker/vllm/Dockerfile \
  -t dgx-spark-vllm:0.18.0 \
  .
```
Note: first build can take significantly longer because `xformers` is compiled from source on ARM64.

Run OpenAI-compatible server:

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -e VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
  -e VLLM_USE_V1=1 \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  dgx-spark-vllm:0.18.0
```

Smoke test (build + /health + /v1/models):

```bash
./scripts/smoke_vllm_container.sh
```

Optional smoke build overrides:

```bash
XFORMERS_DISABLE_FLASH_ATTN=1 XFORMERS_BUILD_JOBS=2 ./scripts/smoke_vllm_container.sh
```

Key runtime env vars:

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model id |
| `VLLM_HOST` | `0.0.0.0` | Bind host |
| `VLLM_PORT` | `8000` | API port inside container |
| `VLLM_DTYPE` | `bfloat16` | vLLM dtype |
| `VLLM_MAX_MODEL_LEN` | `4096` | Max model length |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory target ratio |
| `VLLM_USE_V1` | `1` | Use vLLM V1 engine path |
| `VLLM_ATTENTION_BACKEND` | `FLASH_ATTN` | Attention backend for V1 |
| `VLLM_ENABLE_CUSTOM_OPS` | `0` | Keep custom C++ ops disabled by default on DGX Spark |

## Verification

```bash
python scripts/verify_install.py
```

Sample output:
```
GPU: NVIDIA GB10 (128 GB) — Compute Capability: 12.1
CUDA: 13.0 — Driver: 570.x
PyTorch: 2.9.1+cu130 — CUDA available: ✓
Libraries: transformers ✓ | peft ✓ | trl ✓ | bitsandbytes ✓
MatMul test (4096×4096): PASSED — 2.3 TFLOPS
```

## Benchmarks

```bash
python scripts/benchmark_inference.py   # Token generation speed
python scripts/benchmark_training.py    # Fine-tuning throughput
python scripts/evaluate_perplexity.py   # FP16/NF4/FP4 quality (perplexity)
```

Phase 3 benchmark results (Qwen 7B/14B/32B/72B inference, FP16/NF4/FP4 quality, LoRA/QLoRA training):
- [docs/benchmarks.md](docs/benchmarks.md)
- `artifacts/benchmarks/phase3-baseline-2026-03-13.json`
- `artifacts/benchmarks/inference-extended-32b-72b-2026-03-13.json`
- `artifacts/benchmarks/inference-fp4-2026-03-13.json`
- `artifacts/benchmarks/quality-ppl-fp16-nf4-fp4-2026-03-13.json`

## Documentation

- [Quick Start Guide](docs/quickstart.md) — Get running in 5 minutes
- [Training Guide](docs/training_guide.md) — Fine-tune LLMs on DGX Spark
- [Troubleshooting](docs/troubleshooting.md) — Known issues and solutions
- [Reproducible Builds](docs/reproducible-builds.md) — Deterministic wheel build and release flow
- [Benchmarks](docs/benchmarks.md) — Phase 3 baseline numbers and methodology
- [Ollama Integration Guide](docs/ollama_guide.md) — Model import, quantization, and API usage on DGX Spark
- [llama.cpp Build Guide](docs/llama_cpp_guide.md) — sm_121 CUDA build and GGUF inference on DGX Spark
- [NGC Container Recipe](docs/ngc_recipe.md) — Pinned NGC PyTorch workflow for DGX Spark LLM workloads
- [Docker Compose vLLM Stack](docs/compose_vllm.md) — vLLM + OpenAI-compatible API via Docker Compose
- [Example Notebooks](docs/notebooks.md) — Inference, fine-tuning, and evaluation notebooks with Spark smoke flow
- [FP8 Workaround](docs/fp8_workaround.md) — TransformerEngine FP8 fail + BF16 fallback validation on DGX Spark
- [Continuous Batching (vLLM)](docs/continuous_batching_vllm.md) — Serial vs concurrent OpenAI API smoke on DGX Spark
- [Power/Thermal Profiling](docs/power_thermal_profiling.md) — Operational power and temperature profile during continuous batching smoke
- [Multi-GPU Guide](docs/multi_gpu_guide.md) — Distributed preflight and torchrun cluster recipe for DGX Spark environments
- [Speculative Decoding](docs/speculative_decoding.md) — Baseline vs assistant-model benchmark artifact on DGX Spark
- [KV Cache Optimization](docs/kv_cache_optimization.md) — vLLM max-model-len and memory-utilization sweep with recommended profile
- [TensorRT-LLM Attention Sinks](docs/tensorrt_llm_attention_sinks.md) — Deterministic fail (legacy) + pass (stable) validation on `sm_121`
- [Visibility Strategy](docs/visibility.md) — Community distribution, KPI tracking, and messaging guardrails
- [Community Launch Pack](docs/launch_pack.md) — Copy-ready X/Reddit/HN/forum posts with CTA and link set
- [Day 1 Launch Log](docs/day1_launch_2026-04-05.md) — X + GitHub announcement execution record and KPI deltas
- [Day 2 Launch Log](docs/day2_launch_2026-04-12.md) — Reddit `r/LocalLLaMA` post draft, baseline metrics, and follow-up checklist

## Roadmap

This project is under active development. Here's what's next:

| Phase | Focus | Status |
|-------|-------|--------|
| 1. Foundation | Repo structure, build scripts, docs, compat matrix | ✅ Done |
| 2. Pre-built Wheels | Compile and publish wheels to GitHub Releases | ✅ Done |
| 3. Benchmarks | Inference tok/s, training throughput, model-specific tables | ✅ Done |
| 4. Community | vLLM Dockerfile, Ollama, llama.cpp guide, NGC recipe | ✅ Done |
| 5. Upstream | PyTorch sm_121 PR, Triton fix, flash-attention issue | ✅ Done |
| 6. Advanced | Multi-GPU, TensorRT-LLM, FP8 workaround | ✅ Done |

Full details with task checklists: **[ROADMAP.md](ROADMAP.md)**

## Contributing

Read the contribution guide first: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

Contributions are welcome, especially roadmap-aligned docs, build fixes, benchmarks, and integration improvements.

## System Specs (DGX Spark)

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GB10 (Blackwell, sm_121) |
| VRAM | 128 GB unified memory |
| CPU | 20-core ARM64 (Grace) |
| RAM | 121 GB |
| Storage | 1.9 TB NVMe |
| CUDA | 13.0 |
| GCC | 13.3 |
| CMake | 3.28 |
| Python | 3.12 |

## Acknowledgments

- [Emre Yüz](https://github.com/emreyuz) and his [pytorch-gb10](https://github.com/emreyuz/pytorch-gb10) repo for pioneering PyTorch on GB10
- NVIDIA for DGX Spark developer documentation

## License

[Apache License 2.0](LICENSE)
