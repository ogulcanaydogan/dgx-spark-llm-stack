#!/usr/bin/env python3
"""Benchmark LLM fine-tuning on DGX Spark (GB10, sm_121).

Tests training throughput with LoRA/QLoRA configurations.
"""

import time
import argparse
import sys

def check_deps():
    try:
        import torch
        import transformers
        import peft
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            sys.exit(1)
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install torch transformers peft datasets")
        sys.exit(1)


def benchmark_training(model_name: str, method: str = "lora",
                       num_steps: int = 20, batch_size: int = 1,
                       seq_length: int = 512):
    """Benchmark fine-tuning throughput."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Method: {method}")
    print(f"Steps: {num_steps}, Batch: {batch_size}, SeqLen: {seq_length}")
    print(f"{'='*60}")

    # Load model
    load_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}

    if method == "qlora":
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            print("WARNING: bitsandbytes not available, falling back to LoRA")
            method = "lora"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if method == "qlora":
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    mem_before = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory (model): {mem_before:.1f} GB")

    # Create dummy dataset
    dummy_text = "This is a benchmark training sample for DGX Spark. " * 50
    encodings = tokenizer(
        [dummy_text] * (num_steps * batch_size),
        truncation=True,
        max_length=seq_length,
        padding="max_length",
        return_tensors="pt",
    )

    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()

    print(f"\nRunning {num_steps} training steps...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    step = 0
    total_tokens = 0
    for batch in dataloader:
        if step >= num_steps:
            break

        input_ids = batch[0].to(model.device)
        attention_mask = batch[1].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_tokens += input_ids.numel()
        step += 1

        if step % 5 == 0:
            print(f"  Step {step}/{num_steps} — Loss: {loss.item():.4f}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mem_peak = torch.cuda.max_memory_allocated() / 1e9

    # Results
    tokens_per_sec = total_tokens / elapsed
    samples_per_sec = (num_steps * batch_size) / elapsed

    print(f"\nResults:")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  Tokens/sec:       {tokens_per_sec:.0f}")
    print(f"  Samples/sec:      {samples_per_sec:.2f}")
    print(f"  Peak GPU memory:  {mem_peak:.1f} GB")
    print(f"  Final loss:       {loss.item():.4f}")

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "method": method,
        "tokens_per_sec": round(tokens_per_sec),
        "samples_per_sec": round(samples_per_sec, 2),
        "peak_memory_gb": round(mem_peak, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM training on DGX Spark")
    parser.add_argument("--model", type=str, default="microsoft/phi-2",
                        help="Model name or path")
    parser.add_argument("--method", choices=["lora", "qlora"],
                        default="lora", help="Training method")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--seq-length", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--suite", action="store_true",
                        help="Run full benchmark suite")
    args = parser.parse_args()

    check_deps()

    print("╔══════════════════════════════════════════════════╗")
    print("║     DGX Spark — Training Benchmark               ║")
    print("╚══════════════════════════════════════════════════╝")

    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    if args.suite:
        configs = [
            ("microsoft/phi-2", "lora"),
            ("microsoft/phi-2", "qlora"),
            ("meta-llama/Llama-3.2-3B", "lora"),
            ("meta-llama/Llama-3.2-3B", "qlora"),
        ]
        results = []
        for model, method in configs:
            try:
                r = benchmark_training(model, method, args.steps, args.batch_size, args.seq_length)
                results.append(r)
            except Exception as e:
                print(f"FAILED: {model} ({method}): {e}")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<35} {'Method':<8} {'tok/s':<10} {'VRAM':<8}")
        print("-" * 60)
        for r in results:
            print(f"{r['model']:<35} {r['method']:<8} {r['tokens_per_sec']:<10} {r['peak_memory_gb']:.1f} GB")
    else:
        benchmark_training(args.model, args.method, args.steps, args.batch_size, args.seq_length)


if __name__ == "__main__":
    main()
