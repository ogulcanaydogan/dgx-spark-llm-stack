# Upstream Status Tracker (Phase 5)

This file tracks upstream issue/PR links for DGX Spark (`sm_121`, ARM64, CUDA 13.0) compatibility work.

Last global check date: **2026-03-31**

| Component | Upstream Repo | Issue / PR Links | Status | Last Checked | Local Evidence Pointer |
|---|---|---|---|---|---|
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | [#178891 add sm_121 Blackwell support in aarch64 CUDA13 binary arch selection](https://github.com/pytorch/pytorch/pull/178891), [#172629 DGX Spark GB10 warning](https://github.com/pytorch/pytorch/issues/172629) | opened | 2026-03-31 | `docs/troubleshooting.md` (PyTorch section), `COMPATIBILITY.md` |
| Triton | [triton-lang/triton](https://github.com/triton-lang/triton) | [#9181 ptxas `sm_121a` not defined](https://github.com/triton-lang/triton/issues/9181) | existing | 2026-03-31 | `docs/troubleshooting.md` (Triton section), `COMPATIBILITY.md` |
| flash-attention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | [#1969 support `sm121` on GB10](https://github.com/Dao-AILab/flash-attention/issues/1969) | existing | 2026-03-31 | `docs/troubleshooting.md` (flash-attention section), `COMPATIBILITY.md` |
| BitsAndBytes | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | [#1779 CUDA 13 architecture selection build issue](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1779), [#1218 aarch64 CUDA wheel request](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1218) | existing | 2026-03-31 | `docs/troubleshooting.md` (BitsAndBytes section), `build/build_bitsandbytes.sh` |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | [#36821 no `sm_121` support on aarch64 DGX Spark](https://github.com/vllm-project/vllm/issues/36821) | existing | 2026-03-31 | `docs/troubleshooting.md` (vLLM section), `docker/vllm/Dockerfile` |

## Notes

- Strategy in this phase is tracker-first: use existing upstream records where available and keep links current.
- If a critical gap has no suitable open upstream record in future checks, open a new issue/PR and replace status with `opened`.
- `gh auth status` currently reports an invalid local token in this environment; issue discovery above used existing public upstream records.
