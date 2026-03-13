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
> Baseline + inference extension published on 2026-03-13 for Qwen 7B/14B/32B/72B (FP16 + NF4 where applicable). FP4 and cross-hardware comparison are pending.

- [x] Inference benchmarks: tok/s across model sizes (7B, 14B, 32B, 72B)
- [x] Training benchmarks: SFT throughput (samples/sec) for LoRA and QLoRA
- [x] Memory usage profiling per model size
- [ ] Comparison table: DGX Spark vs. RTX 4090 vs. A100 (where applicable)
- [ ] Quantization benchmarks: FP16 vs. NF4 vs. FP4 quality and speed
- [x] Publish results in `docs/benchmarks.md` with methodology
- [x] Add benchmark badges to README

## Phase 4 — Community & Ecosystem

> Integrations with popular inference and serving frameworks.

- [ ] vLLM Dockerfile optimized for DGX Spark
- [ ] Ollama integration guide (model import, quantization)
- [ ] llama.cpp build guide with sm_121 optimizations
- [ ] NGC container recipe for DGX Spark LLM workloads
- [ ] Docker Compose stack: vLLM + OpenAI-compatible API
- [ ] Example notebooks: inference, fine-tuning, evaluation
- [ ] Contributing guide (`CONTRIBUTING.md`)

## Phase 5 — Upstream Contributions

> Push fixes and support upstream to reduce the need for custom builds.

- [ ] PyTorch: PR to add `sm_121` to supported architectures
- [ ] Triton: Fix `ptxas` sm_121a recognition issue
- [ ] flash-attention: Open issue / PR for sm_121 kernel support
- [ ] BitsAndBytes: Ensure ARM64 + CUDA 13.0 in CI matrix
- [ ] vLLM: DGX Spark support in official Docker images
- [ ] Track upstream issue links in `docs/upstream_status.md`

## Phase 6 — Advanced

> Multi-GPU, TensorRT-LLM, and advanced optimizations.

- [ ] Multi-GPU guide (DGX Spark cluster with NVLink)
- [ ] TensorRT-LLM: fix attention sinks on sm_121
- [ ] FP8 workaround for TransformerEngine on Blackwell
- [ ] Speculative decoding benchmarks
- [ ] KV cache optimization for 128 GB unified memory
- [ ] Continuous batching setup with vLLM
- [ ] Power consumption and thermal profiling

---

## Status Legend

- ✅ Phase complete
- 🔄 In progress
- ⬚ Not started

**Current Phase: 3 — Benchmarks & Validation**

## How to Contribute

Pick any unchecked item above, open an issue or PR. See the [Contributing](#) section in README for guidelines.
