# Upstream Status Tracker (Phase 5)

This file tracks upstream issue/PR links for DGX Spark (`sm_121`, ARM64, CUDA 13.0) compatibility work.

Last global check date: **2026-03-31**

| Component | Upstream Repo | Issue / PR Links | Status | Last Checked | Local Evidence Pointer |
|---|---|---|---|---|---|
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | [#178891 add sm_121 Blackwell support in aarch64 CUDA13 binary arch selection](https://github.com/pytorch/pytorch/pull/178891), [#172629 DGX Spark GB10 warning](https://github.com/pytorch/pytorch/issues/172629) | opened | 2026-03-31 | `docs/troubleshooting.md` (PyTorch section), `COMPATIBILITY.md` |
| Triton | [triton-lang/triton](https://github.com/triton-lang/triton) | [#8498 enable TMA gather4 on sm_120/sm_121 (merged)](https://github.com/triton-lang/triton/pull/8498), [#8539 `sm_121a` ptxas fatal](https://github.com/triton-lang/triton/issues/8539), [#9181 `sm_121a` ptxas fatal](https://github.com/triton-lang/triton/issues/9181) | resolved-upstream | 2026-03-31 | `docs/troubleshooting.md` (Triton section), `build/build_triton.sh` |
| flash-attention | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | [#1969 support `sm121` on GB10](https://github.com/Dao-AILab/flash-attention/issues/1969), [DGX Spark evidence comment](https://github.com/Dao-AILab/flash-attention/issues/1969#issuecomment-4164154447) | pending-review | 2026-03-31 | `docs/troubleshooting.md` (flash-attention section), `COMPATIBILITY.md` |
| BitsAndBytes | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | [#1779 CUDA 13 architecture selection build issue](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1779), [#1218 aarch64 CUDA wheel request](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1218), [CI matrix workflow (`python-package.yml`)](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/.github/workflows/python-package.yml), [ARM64 + CUDA 13.0.2 successful job](https://github.com/bitsandbytes-foundation/bitsandbytes/actions/runs/23774264108/job/69272466447) | resolved-upstream | 2026-03-31 | `docs/troubleshooting.md` (BitsAndBytes section), `build/build_bitsandbytes.sh` |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | [#36821 no `sm_121` support on aarch64 DGX Spark](https://github.com/vllm-project/vllm/issues/36821) | existing | 2026-03-31 | `docs/troubleshooting.md` (vLLM section), `docker/vllm/Dockerfile` |

## Notes

- Strategy in this phase is tracker-first: use existing upstream records where available and keep links current.
- If a critical gap has no suitable open upstream record in future checks, open a new issue/PR and replace status with `opened`.
- `gh auth status` is valid in this environment as of 2026-03-31.
- Spark validation (2026-03-31): default Triton ptxas (12.8) reproduces `sm_121a` failure; `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` (13.0) passes Triton JIT smoke.
- BitsAndBytes verification (2026-03-31): CI matrix build coverage is confirmed; CUDA runtime nightly coverage remains x64-heavy.
