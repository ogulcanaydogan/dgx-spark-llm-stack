# Quick Start — DGX Spark LLM Stack

Get a working LLM training & inference setup on your DGX Spark in 5 minutes.

## Prerequisites

- NVIDIA DGX Spark with GB10 GPU
- CUDA 13.0 installed (default on DGX Spark)
- Python 3.12 (default on DGX Spark)
- `pip` and `git` available

## Option 1: One-Command Install (Recommended)

```bash
git clone https://github.com/ogulcanaydogan/dgx-spark-llm-stack.git
cd dgx-spark-llm-stack
./install.sh
```

This downloads pre-built wheels and installs everything.

## Option 2: Manual Install

If you prefer to install manually:

```bash
# 1. Clone the repo
git clone https://github.com/ogulcanaydogan/dgx-spark-llm-stack.git
cd dgx-spark-llm-stack

# 2. Download pre-built PyTorch wheel from Releases
gh release download latest --pattern "torch-*.whl" --dir /tmp/wheels

# 3. Install PyTorch
pip install /tmp/wheels/torch-*.whl

# 4. Install ML stack
pip install transformers accelerate peft trl datasets \
    bitsandbytes sentencepiece safetensors rich wandb

# 5. Verify
python scripts/verify_install.py
```

## Option 3: Build from Source

For full control over build options:

```bash
git clone https://github.com/ogulcanaydogan/dgx-spark-llm-stack.git
cd dgx-spark-llm-stack
./install.sh --from-source
```

This builds PyTorch, Triton, and BitsAndBytes from source (~5 hours).

## Verify Installation

```bash
python scripts/verify_install.py
```

Expected output:
```
GPU: NVIDIA GB10 (128 GB) — Compute Capability: 12.1
CUDA: 13.0
PyTorch: 2.9.1+cu130 — CUDA available: ✓
Libraries: transformers ✓ | peft ✓ | trl ✓ | bitsandbytes ✓
MatMul test (4096×4096): PASSED
```

## Quick Test: Run Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

prompt = "The best thing about DGX Spark is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Next Steps

- [Training Guide](training_guide.md) — Fine-tune LLMs on DGX Spark
- [Troubleshooting](troubleshooting.md) — Common issues and fixes
- [Compatibility Matrix](../COMPATIBILITY.md) — What works and what doesn't
