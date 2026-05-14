# Roadmap

This document tracks the development roadmap for DGX Spark LLM Stack. Each phase builds on the previous one.

## Phase 1 — Foundation ✅

> Repo structure, build scripts, docs, compatibility matrix.

- [x] Repository structure and organization
- [x] Build scripts for PyTorch, Triton, flash-attention, BitsAndBytes
- [x] Automated installer (`install.sh`)
- [x] Compatibility matrix with status for all major ML libraries
- [x] Quick start guide, training guide, troubleshooting docs
- [x] Verification script (`verify_install.py`)
- [x] Benchmark scripts (inference + training)
- [x] Environment config (`configs/env.sh`)
- [x] Apache 2.0 license

## Phase 2 — Pre-built Wheels

> Compile wheels on DGX Spark, publish to GitHub Releases, add CI testing.

- [x] Build PyTorch 2.9.1 wheel on DGX Spark (sm_121, CUDA 13.0, ARM64)
- [x] Build BitsAndBytes wheel with CUDA 13.0 support
- [x] Upload wheels to GitHub Releases with checksums (SHA256)
- [x] Add `install.sh` auto-download from Releases
- [x] CI workflow: test wheel installation on fresh DGX Spark
- [x] CI workflow: run `verify_install.py` after install
- [x] Document wheel build reproducibility steps

## Phase 3 — Benchmarks & Validation

> Real benchmark results with numbers — inference speed, training throughput, model-specific tables.
> Baseline + FP4/quality + cross-hardware snapshot published on 2026-03-13 for Qwen 7B/14B/32B/72B.

- [x] Inference benchmarks: tok/s across model sizes (7B, 14B, 32B, 72B)
- [x] Training benchmarks: SFT throughput (samples/sec) for LoRA and QLoRA
- [x] Memory usage profiling per model size
- [x] Comparison table: DGX Spark vs. RTX 4090 vs. A100 (where applicable)
- [x] Quantization benchmarks: FP16 vs. NF4 vs. FP4 quality and speed
- [x] Publish results in `docs/benchmarks.md` with methodology
- [x] Add benchmark badges to README

## Phase 4 — Community & Ecosystem

> Integrations with popular inference and serving frameworks.

- [x] vLLM Dockerfile optimized for DGX Spark
- [x] Ollama integration guide (model import, quantization)
- [x] llama.cpp build guide with sm_121 optimizations
- [x] NGC container recipe for DGX Spark LLM workloads
- [x] Docker Compose stack: vLLM + OpenAI-compatible API
- [x] Example notebooks: inference, fine-tuning, evaluation
- [x] Contributing guide (`CONTRIBUTING.md`)

## Phase 5 — Upstream Contributions

> Push fixes and support upstream to reduce the need for custom builds.

- [x] PyTorch: PR to add `sm_121` to supported architectures
- [x] Triton: Fix `ptxas` sm_121a recognition issue
- [x] flash-attention: Open issue / PR for sm_121 kernel support
- [x] BitsAndBytes: Ensure ARM64 + CUDA 13.0 in CI matrix
- [x] vLLM: DGX Spark support in official Docker images
- [x] Track upstream issue links in `docs/upstream_status.md`

## Phase 6 — Advanced

> Multi-GPU, TensorRT-LLM, and advanced optimizations.

- [x] Multi-GPU guide (DGX Spark cluster with NVLink)
- [x] TensorRT-LLM: fix attention sinks on sm_121
- [x] FP8 workaround for TransformerEngine on Blackwell
- [x] Speculative decoding benchmarks
- [x] KV cache optimization for 128 GB unified memory
- [x] Continuous batching setup with vLLM
- [x] Power consumption and thermal profiling

---

## v0.2.0 — Multi-hardware Support

> Target: 2026-Q3

Extends the stack beyond DGX Spark to a second hardware family. Profile-based configuration allows users to select their hardware at `source configs/env.sh` time without modifying build scripts.

- [x] Hardware-profile abstraction (`configs/profiles/dgx-spark.env`, `configs/profiles/h100.env`)
- [x] Profile-aware `configs/env.sh` (selectable via `HW_PROFILE` env, default `dgx-spark`)
- [x] Profile-aware `scripts/check_compatibility.py --profile h100|dgx-spark`
- [x] H100 (Hopper, sm_90) documented in `COMPATIBILITY.md`
- [x] H100 install path in `install.sh` (skip custom-wheel build, use upstream pip)
- [ ] H100 Docker variant in `docker/vllm/Dockerfile`
- [ ] H100 smoke tests (`scripts/smoke_*.sh` profile-aware)
- [ ] CI matrix: add `h100` profile axis (requires self-hosted x86_64 runner)
- [ ] Live H100 benchmark numbers in `docs/benchmarks.md`

---

## Status Legend

- ✅ Phase complete
- 🔄 In progress
- ⬚ Not started

**Current Phase: 6 — Advanced**

## How to Contribute

Pick any unchecked item above, open an issue or PR. See the [Contributing](#) section in README for guidelines.
