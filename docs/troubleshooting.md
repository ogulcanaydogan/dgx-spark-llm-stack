# Troubleshooting — DGX Spark LLM Stack

Common issues and solutions for running ML workloads on DGX Spark (GB10, sm_121).

## PyTorch

### "Unsupported GPU architecture sm_121" warning

```
UserWarning: CUDA with SM 12.1 is not natively supported.
```

**Status**: Harmless warning. PyTorch uses sm_120 binary-compatible kernels.

**Fix**: Suppress the warning if it bothers you:
```python
import warnings
warnings.filterwarnings("ignore", message=".*SM 12.1.*")
```

### PyTorch CUDA not available

```python
>>> torch.cuda.is_available()
False
```

**Check**:
1. Is the NVIDIA driver loaded? `nvidia-smi`
2. Was PyTorch built with CUDA? `python -c "import torch; print(torch.version.cuda)"`
3. Are you using the correct wheel? Must be `cu130` for CUDA 13.0

**Fix**: Install our pre-built wheel:
```bash
./install.sh
```

## Triton

### "ptxas fatal: Unsupported GPU architecture sm_121a"

```
ptxas fatal : Unsupported GPU architecture 'sm_121a'
```

**Status**: Reproducible when Triton uses bundled `ptxas` 12.8 on DGX Spark.

**Fix**: Force Triton to use system CUDA 13.0 `ptxas`:
```bash
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

**Fallback**: Disable Triton backend for `torch.compile()` if the workload still fails:
```python
# Use inductor without Triton
import torch
model = torch.compile(model, backend="eager")
# Or just don't use torch.compile()
```

### torch.compile() fails

If `torch.compile()` fails with Triton errors, disable it:
```python
# Instead of:
model = torch.compile(model)

# Just use the model directly:
model = model  # no compilation
```

Most HF training pipelines work fine without `torch.compile()`.

## flash-attention

### "No kernel image available for execution on the device"

```
RuntimeError: no kernel image is available for execution on the device
```

**Status**: flash-attention has no sm_121 kernels.

**Fix**: Use SDPA instead:
```python
# In model loading:
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="sdpa",
)

# Or set environment variable:
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
```

### ImportError: flash_attn

```
ImportError: No module named 'flash_attn'
```

This is fine. flash-attention is not required. HF transformers will automatically fall back to SDPA.

## BitsAndBytes

### "CUDA Setup failed" on import

```
RuntimeError: CUDA Setup failed despite CUDA being available.
```

**Fix**: Build from source with CUDA 13.0:
```bash
./build/build_bitsandbytes.sh
pip install dist/bitsandbytes-*.whl
```

### Slow 4-bit inference

4-bit quantized models may be slower than expected if BitsAndBytes CUDA kernels aren't compiled for sm_121.

**Fix**: Verify the installation:
```python
import bitsandbytes as bnb
print(bnb.cuda_setup.evaluate_cuda_setup())
```

## TransformerEngine

### "MXFP8 is not supported"

```
RuntimeError: MXFP8 is not supported on compute capability 12.1
```

**Status**: Known issue. TransformerEngine's FP8 format is not supported on sm_121.

**Fix**: Use BF16 instead of FP8:
```python
# Don't use TransformerEngine on DGX Spark
# Use standard BF16 training:
training_args = TrainingArguments(bf16=True)
```

## vLLM

### vLLM fails to install via pip

vLLM depends on Triton; on DGX Spark you may need `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` for Triton JIT paths.

**Options**:
1. Use NGC PyTorch container as base
2. Use `llama.cpp` / Ollama for inference instead
3. Build vLLM from source after building Triton:
```bash
./build/build_triton.sh
pip install vllm --no-build-isolation
```

## General

### Out of Memory (OOM)

Despite 128 GB unified memory, OOM can happen with large models.

**Fixes**:
- Use QLoRA (4-bit) instead of LoRA (BF16)
- Reduce batch size
- Enable gradient checkpointing: `model.gradient_checkpointing_enable()`
- Reduce sequence length
- Use `torch.cuda.empty_cache()` between experiments

### Slow training compared to A100

The GB10 has lower raw compute TFLOPS than A100. Expected performance:

| Operation | GB10 vs A100-80GB |
|-----------|-------------------|
| FP32 matmul | ~0.3x |
| BF16 matmul | ~0.4x |
| 4-bit inference | ~0.5x |
| Memory capacity | 1.6x (128 vs 80 GB) |

The GB10's advantage is memory capacity, not raw speed. Use it for:
- Large model inference that doesn't fit on A100-40GB
- QLoRA fine-tuning of 70B+ models
- Development and prototyping

### How to check GPU utilization

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# In Python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.1f} GB")
print(f"Max:       {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
```

## Still stuck?

1. Run the compatibility check: `python scripts/check_compatibility.py`
2. Open an issue on [GitHub](https://github.com/ogulcanaydogan/dgx-spark-llm-stack/issues)
3. Check NVIDIA DGX Spark documentation
