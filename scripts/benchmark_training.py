#!/usr/bin/env python3
"""Benchmark LLM fine-tuning on DGX Spark (GB10, sm_121)."""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

BASELINE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]
BASELINE_METHODS = ["lora", "qlora"]


def check_deps():
    try:
        import torch
        import transformers  # noqa: F401
        import peft  # noqa: F401

        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            sys.exit(1)
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        print("Install with: pip install torch transformers peft datasets")
        sys.exit(1)


def parse_csv_list(raw_value: str):
    if not raw_value:
        return []
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def normalize_method(value: str):
    normalized = value.strip().lower()
    if normalized not in {"lora", "qlora"}:
        raise ValueError(f"Unsupported training method: {value}")
    return normalized


def get_env_info():
    import torch

    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(total_mem / 1e9, 1),
    }


def benchmark_training(model_name: str, method: str = "lora", num_steps: int = 20, batch_size: int = 1, seq_length: int = 512):
    """Benchmark fine-tuning throughput for one model+method pair."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    normalized_method = normalize_method(method)

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Method: {normalized_method}")
    print(f"Steps: {num_steps}, Batch: {batch_size}, SeqLen: {seq_length}")
    print(f"{'=' * 60}")

    load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

    if normalized_method == "qlora":
        try:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            print("WARNING: bitsandbytes not available, falling back to LoRA")
            normalized_method = "lora"

    print("Loading model...")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    if normalized_method == "qlora":
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    mem_before = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory (model): {mem_before:.1f} GB")
    torch.cuda.reset_peak_memory_stats()

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()

    print(f"\nRunning {num_steps} training steps...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    step = 0
    total_tokens = 0
    loss = None
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
            print(f"  Step {step}/{num_steps} - Loss: {loss.item():.4f}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mem_peak = torch.cuda.max_memory_allocated() / 1e9
    tokens_per_sec = total_tokens / elapsed
    samples_per_sec = (num_steps * batch_size) / elapsed

    print("\nResults:")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  Tokens/sec:       {tokens_per_sec:.0f}")
    print(f"  Samples/sec:      {samples_per_sec:.2f}")
    print(f"  Peak GPU memory:  {mem_peak:.1f} GB")
    print(f"  Final loss:       {loss.item():.4f}")

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "method": normalized_method,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "samples_per_sec": round(samples_per_sec, 4),
        "peak_memory_gb": round(mem_peak, 2),
        "model_memory_gb": round(mem_before, 2),
        "model_load_time_s": round(load_time, 3),
        "final_loss": round(loss.item(), 6),
    }


def write_json_output(path: str, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
    print(f"\nWrote JSON results to: {path}")


def print_summary(results):
    print(f"\n{'=' * 86}")
    print("SUMMARY")
    print(f"{'=' * 86}")
    print(f"{'Model':<34} {'Method':<8} {'tok/s':<11} {'samples/s':<11} {'Peak VRAM':<10} {'Load(s)':<8}")
    print("-" * 86)
    for result in results:
        print(
            f"{result['model']:<34} {result['method']:<8} {result['tokens_per_sec']:<11} "
            f"{result['samples_per_sec']:<11} {result['peak_memory_gb']:<10.2f} {result['model_load_time_s']:<8.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM training on DGX Spark")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="Model name or path for single-run mode")
    parser.add_argument("--models", type=str, default="", help="Comma-separated model list for matrix runs")
    parser.add_argument("--method", choices=["lora", "qlora"], default="lora", help="Training method for single-run mode")
    parser.add_argument("--methods", type=str, default="", help="Comma-separated method list (lora,qlora)")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument("--suite", action="store_true", help="Run legacy benchmark suite")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run Phase 3 baseline matrix (Qwen2.5 7B/14B x LoRA/QLoRA)",
    )
    parser.add_argument("--output-json", type=str, default="", help="Optional path to write machine-readable JSON results")
    args = parser.parse_args()

    check_deps()

    print("╔══════════════════════════════════════════════════╗")
    print("║     DGX Spark - Training Benchmark               ║")
    print("╚══════════════════════════════════════════════════╝")

    env_info = get_env_info()
    print(f"GPU: {env_info['gpu_name']}")
    print(f"PyTorch: {env_info['torch_version']}")

    results = []
    failures = []

    if args.baseline:
        models = BASELINE_MODELS
        methods = BASELINE_METHODS
    elif args.suite:
        configs = [
            ("microsoft/phi-2", "lora"),
            ("microsoft/phi-2", "qlora"),
            ("meta-llama/Llama-3.2-3B", "lora"),
            ("meta-llama/Llama-3.2-3B", "qlora"),
        ]
        for model_name, method in configs:
            try:
                result = benchmark_training(model_name, method, args.steps, args.batch_size, args.seq_length)
                results.append(result)
            except Exception as exc:
                print(f"FAILED: {model_name} ({method}): {exc}")
                failures.append({"model": model_name, "method": method, "error": str(exc)})
        print_summary(results)
        payload = {
            "benchmark_type": "training",
            "mode": "suite",
            "environment": env_info,
            "config": {"steps": args.steps, "batch_size": args.batch_size, "seq_length": args.seq_length},
            "results": results,
            "failures": failures,
        }
        if args.output_json:
            write_json_output(args.output_json, payload)
        if failures:
            sys.exit(1)
        return
    else:
        models = parse_csv_list(args.models) or [args.model]
        methods = parse_csv_list(args.methods) or [args.method]

    for model_name in models:
        for method in methods:
            try:
                result = benchmark_training(model_name, method, args.steps, args.batch_size, args.seq_length)
                result.update(
                    {
                        "gpu_name": env_info["gpu_name"],
                        "torch_version": env_info["torch_version"],
                        "cuda_version": env_info["cuda_version"],
                    }
                )
                results.append(result)
            except Exception as exc:
                print(f"FAILED: {model_name} ({method}): {exc}")
                failures.append({"model": model_name, "method": method, "error": str(exc)})

    print_summary(results)

    payload = {
        "benchmark_type": "training",
        "mode": "baseline" if args.baseline else "matrix",
        "environment": env_info,
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "models": models,
            "methods": methods,
        },
        "results": results,
        "failures": failures,
    }
    if args.output_json:
        write_json_output(args.output_json, payload)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
