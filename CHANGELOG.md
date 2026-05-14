# Changelog

All notable changes to the DGX Spark LLM Stack will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **v0.2.0 Hardware-profile abstraction**
  - `configs/profiles/dgx-spark.env` — extracted GB10/sm_121/aarch64/CUDA-13 vars
  - `configs/profiles/h100.env` — Hopper/sm_90/x86_64/CUDA-12.4 profile with `INSTALL_STRATEGY=upstream-wheels`
  - `configs/env.sh` now sources `configs/profiles/${HW_PROFILE}.env` (default: `dgx-spark`) for backward-compatible hardware selection
  - `scripts/check_compatibility.py --profile h100|dgx-spark` — profile-aware compatibility report; H100 branch marks flash-attention and TransformerEngine as fully supported
  - `COMPATIBILITY.md` H100 section covering Hopper sm_90 framework status
- **v0.2.0 H100 install path** — profile-aware `install.sh` with `upstream-wheels` branch for H100; installs PyTorch from `download.pytorch.org/whl/cu124` and flash-attention from pip (sm_90 kernels available upstream); DGX Spark custom-wheels flow preserved as default

## [0.1.0] — 2026-03-13

### Added

- Build scripts for PyTorch, Triton, flash-attention, BitsAndBytes on DGX Spark (GB10/sm_121)
- Automated installer (`install.sh`) with wheel download from GitHub Releases
- Compatibility matrix for 20+ ML libraries
- Benchmark results: inference tok/s for Qwen 7B/14B/32B/72B; training SFT throughput
- vLLM Docker stack with OpenAI-compatible API
- Multi-GPU guide, TensorRT-LLM smoke tests, FP8 workaround docs
- Verification scripts (`verify_install.py`, `check_compatibility.py`)
- NGC container recipe

[Unreleased]: https://github.com/ogulcanaydogan/dgx-spark-llm-stack/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ogulcanaydogan/dgx-spark-llm-stack/releases/tag/v0.1.0
