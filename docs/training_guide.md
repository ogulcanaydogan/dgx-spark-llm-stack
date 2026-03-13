# Training Guide — Fine-tune LLMs on DGX Spark

This guide covers fine-tuning language models on the DGX Spark (GB10, 128 GB unified memory).

## What Works

| Method | Supported | Max Model Size (approx) |
|--------|-----------|------------------------|
| LoRA (BF16) | ✅ | ~30B parameters |
| QLoRA (4-bit) | ✅ | ~70B parameters |
| Full fine-tuning | ⚠️ | ~7B parameters |
| FSDP | ❌ | Single GPU only |
| FP8 training | ❌ | TransformerEngine broken |

## Recommended: QLoRA with Unsloth

Unsloth gives ~2x speedup over vanilla HF training on Blackwell.

```python
from unsloth import FastLanguageModel
import torch

# Load model in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-bnb-4bit",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)
```

## Standard HF Training with TRL

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training
training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    max_seq_length=2048,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # your dataset here
    peft_config=lora_config,
    tokenizer=tokenizer,
)

trainer.train()
```

## DPO Training

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    output_dir="./dpo-output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    bf16=True,
    max_length=1024,
    max_prompt_length=512,
    beta=0.1,
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dpo_dataset,  # needs chosen/rejected columns
    peft_config=lora_config,
    tokenizer=tokenizer,
)

trainer.train()
```

## Important: Disable flash-attention

flash-attention doesn't work on sm_121. Set the `attn_implementation` to `sdpa`:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="sdpa",  # Use PyTorch SDPA instead of flash-attn
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

Or set the environment variable:
```bash
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
```

## Memory Tips

The GB10 has 128 GB unified memory — generous for a single GPU.

| Model | Method | Approx VRAM | Batch Size |
|-------|--------|-------------|------------|
| 3B | LoRA BF16 | ~8 GB | 4-8 |
| 7B | QLoRA 4-bit | ~6 GB | 4-8 |
| 7B | LoRA BF16 | ~16 GB | 2-4 |
| 14B | QLoRA 4-bit | ~10 GB | 2-4 |
| 70B | QLoRA 4-bit | ~42 GB | 1 |

Tips:
- Use `gradient_checkpointing=True` for large models
- Use `gradient_accumulation_steps` to simulate larger batches
- Monitor with `nvidia-smi` or `torch.cuda.memory_summary()`

## Benchmarking

```bash
# Quick benchmark
python scripts/benchmark_training.py --model microsoft/phi-2 --method qlora

# Full suite
python scripts/benchmark_training.py --suite
```
