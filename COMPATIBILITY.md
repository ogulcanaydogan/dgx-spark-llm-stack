# Compatibility Matrix — DGX Spark (GB10, sm_121)

> Last updated: 2026-03-31
> System: DGX Spark, CUDA 13.0, GCC 13.3, Python 3.12, ARM64

## Overview

The NVIDIA GB10 GPU in DGX Spark uses the Blackwell architecture with compute capability `sm_121`. This is newer than what most ML frameworks officially support, leading to various compatibility issues.

## Core ML Frameworks

| Library | Tested Version | Status | Install Method | Notes |
|---------|---------------|--------|----------------|-------|
| **PyTorch** | 2.9.1 | ⚠️ Works with warnings | Pre-built wheel | Official wheels target max sm_120. Emits "unsupported gpu architecture" warning during compilation but runs correctly. Our wheel is built with `TORCH_CUDA_ARCH_LIST=12.1`. |
| **Triton** | 3.5.1 | ⚠️ Works with env fix | Source build or installed wheel | Default bundled `ptxas` (12.8) can fail on `sm_121a`. Use `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` (13.0) on DGX Spark. |
| **flash-attention** | 2.7+ | ❌ Not supported | Skip (use SDPA) | No sm_121 CUDA kernels. Compilation fails. Use PyTorch's built-in `F.scaled_dot_product_attention()` as fallback — performance is comparable on Blackwell. |
| **TransformerEngine** | Latest | ❌ Broken | N/A | MXFP8 format not supported on sm_121. FP8 training will not work. Use BF16 instead. |

## Quantization & Optimization

| Library | Tested Version | Status | Install Method | Notes |
|---------|---------------|--------|----------------|-------|
| **BitsAndBytes** | 0.49+ | ✅ Works | Pre-built wheel | FP4 and NF4 quantization tested and working. Requires building from source with CUDA 13.0 for ARM64. |
| **GPTQ** | Latest | ⚠️ Partial | pip | Quantization works, but some kernels fall back to slower paths. |
| **AWQ** | Latest | ⚠️ Partial | pip | Similar to GPTQ — works but not fully optimized for sm_121. |
| **llama.cpp** | Latest | ✅ Works well | Build from source | Best inference option. CUDA backend works natively. Supports Q4_K_M, Q5_K_M, Q8_0, and FP16. |

## Inference Servers

| Library | Tested Version | Status | Install Method | Notes |
|---------|---------------|--------|----------------|-------|
| **vLLM** | 0.8+ | ⚠️ Docker only | NGC container or source build | Does not pip-install cleanly due to Triton dependency. Use NGC PyTorch container as base. |
| **TensorRT-LLM** | 0.9+ | ⚠️ Partial | NGC container | Basic inference works. Attention sinks and some advanced features broken. |
| **Ollama** | Latest | ✅ Works | Binary install | Uses llama.cpp backend. Simple and effective for local inference. |
| **text-generation-inference** | Latest | ⚠️ Untested | Docker | Should work via Docker with NGC base image. Community testing welcome. |

## Training & Fine-tuning

| Library | Tested Version | Status | Install Method | Notes |
|---------|---------------|--------|----------------|-------|
| **transformers** | 4.48+ | ✅ Works | pip | Standard Hugging Face stack. No issues. |
| **PEFT** | Latest | ✅ Works | pip | LoRA, QLoRA all functional. |
| **TRL** | Latest | ✅ Works | pip | SFT, DPO, ORPO trainers all work. |
| **Unsloth** | Latest | ✅ Works | pip | Recommended for efficient fine-tuning. 2x faster than vanilla HF training. |
| **DeepSpeed** | Latest | ⚠️ Partial | pip | ZeRO-1/2 work. ZeRO-3 untested on single GPU. |
| **Axolotl** | Latest | ⚠️ Partial | pip | Works if you disable flash-attention (use SDPA). |

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ Works | Tested and functional on DGX Spark |
| ⚠️ Partial / Warning | Works with caveats or workarounds needed |
| ❌ Broken | Does not work; alternative recommended |

## Known Issues

### PyTorch sm_121 Warning
```
UserWarning: CUDA with SM 12.1 is not natively supported. Falling back to SM 12.0 kernels.
```
This warning is harmless. All operations work correctly — PyTorch uses sm_120 binary-compatible kernels.

### Triton ptxas Error
```
ptxas fatal : Unsupported GPU architecture 'sm_121a'
```
This is reproducible with Triton 3.5.1 when it uses bundled `ptxas` 12.8.

**Fix**:
```bash
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

With CUDA 13.0 `ptxas`, Triton JIT compile works on GB10 (`sm_121a`). If compile still fails, disable `torch.compile()` for that workload.

### flash-attention Compilation Failure
```
fatal error: no kernel image is available for execution on the device
```
flash-attention does not include sm_121 kernels. The recommended workaround is PyTorch's native SDPA, which uses efficient attention on Blackwell without custom kernels.

### TransformerEngine MXFP8
```
RuntimeError: MXFP8 is not supported on compute capability 12.1
```
FP8 mixed-precision training via TransformerEngine is not available. Use BF16 training instead — GB10 has excellent BF16 throughput.

## Recommended Stack

For the smoothest experience on DGX Spark:

**Training:**
```
PyTorch 2.9.1 (custom wheel) + transformers + PEFT + TRL + Unsloth + BitsAndBytes
```

**Inference:**
```
llama.cpp (GGUF models) or Ollama for simple serving
```

**Precision:** BF16 for training, Q4_K_M or Q8_0 for inference
