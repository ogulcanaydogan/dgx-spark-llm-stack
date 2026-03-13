#!/usr/bin/env python3
"""Evaluate perplexity for FP16/NF4/FP4 quantization comparisons."""

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone

BASELINE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]
BASELINE_QUANTIZATIONS = ["fp16", "nf4", "fp4"]


def check_deps():
    try:
        import torch
        import datasets  # noqa: F401
        import transformers  # noqa: F401

        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            sys.exit(1)
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        print("Install with: pip install torch transformers datasets")
        sys.exit(1)


def parse_csv_list(raw_value: str):
    if not raw_value:
        return []
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def normalize_quantization(value: str):
    normalized = value.strip().lower()
    if normalized in {"none", "fp16"}:
        return "fp16"
    if normalized in {"4bit", "nf4"}:
        return "nf4"
    if normalized in {"fp4", "4bit-fp4"}:
        return "fp4"
    if normalized in {"8bit", "int8"}:
        return "int8"
    raise ValueError(f"Unsupported quantization: {value}")


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_wikitext_subset(dataset_name: str, dataset_config: str, split: str, subset_size: int):
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    texts = []
    for row in dataset:
        text = row.get("text", "")
        if not text:
            continue
        text = text.strip()
        if not text:
            continue
        texts.append(text)
        if len(texts) >= subset_size:
            break

    if len(texts) < subset_size:
        print(f"WARNING: Requested subset_size={subset_size}, only found {len(texts)} non-empty rows")

    if not texts:
        raise RuntimeError("No non-empty text rows found in dataset split")

    return texts


def evaluate_perplexity(
    model_name: str,
    quantization: str,
    texts,
    max_length: int,
    batch_size: int,
    seed: int,
    device_map_mode: str,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant_mode = normalize_quantization(quantization)

    print(f"\n{'=' * 72}")
    print(f"Model:        {model_name}")
    print(f"Quantization: {quant_mode}")
    print(f"Max length:   {max_length}")
    print(f"Batch size:   {batch_size}")
    print(f"Samples:      {len(texts)}")
    print(f"{'=' * 72}")

    if device_map_mode == "cuda":
        selected_device_map = {"": 0}
    else:
        selected_device_map = "auto"

    load_kwargs = {"device_map": selected_device_map, "torch_dtype": torch.float16}

    if quant_mode in {"nf4", "fp4"}:
        try:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=quant_mode,
            )
        except ImportError:
            print("WARNING: bitsandbytes not available, using FP16")
            quant_mode = "fp16"
    elif quant_mode == "int8":
        try:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            print("WARNING: bitsandbytes not available, using FP16")
            quant_mode = "fp16"

    set_seed(seed)

    print("Loading model...")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    mem_loaded = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after load: {mem_loaded:.2f} GB")

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    valid_rows = attention_mask.sum(dim=1) >= 2
    input_ids = input_ids[valid_rows]
    attention_mask = attention_mask[valid_rows]

    if input_ids.shape[0] == 0:
        raise RuntimeError("No valid tokenized rows after filtering")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    total_nll = 0.0
    total_tokens = 0

    start = time.perf_counter()
    for offset in range(0, input_ids.shape[0], batch_size):
        batch_ids = input_ids[offset : offset + batch_size].to(model.device)
        batch_mask = attention_mask[offset : offset + batch_size].to(model.device)
        labels = batch_ids.clone()
        labels[batch_mask == 0] = -100

        with torch.no_grad():
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=labels)

        n_tokens = int(batch_mask.sum().item())
        total_nll += float(outputs.loss.item()) * n_tokens
        total_tokens += n_tokens

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_nll = total_nll / max(total_tokens, 1)
    perplexity = float(math.exp(avg_nll))
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

    print("\nQuality results:")
    print(f"  Perplexity:      {perplexity:.4f}")
    print(f"  Eval tokens:     {total_tokens}")
    print(f"  Eval time:       {elapsed:.2f}s")
    print(f"  Eval tokens/sec: {tokens_per_sec:.1f}")
    print(f"  Peak GPU memory: {peak_mem:.2f} GB")

    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "quantization": quant_mode,
        "dataset": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "dataset_split": "validation",
        "perplexity": round(perplexity, 6),
        "avg_nll": round(avg_nll, 6),
        "eval_samples": int(input_ids.shape[0]),
        "eval_tokens": int(total_tokens),
        "max_length": max_length,
        "batch_size": batch_size,
        "seed": seed,
        "eval_time_s": round(elapsed, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "gpu_memory_gb": round(mem_loaded, 2),
        "peak_memory_gb": round(peak_mem, 2),
        "model_load_time_s": round(load_time, 3),
        "device_map_mode": device_map_mode,
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
    print("PERPLEXITY SUMMARY")
    print(f"{'=' * 86}")
    print(f"{'Model':<34} {'Quant':<8} {'PPL':<12} {'tok/s':<12} {'Peak VRAM':<10} {'Load(s)':<8}")
    print("-" * 86)
    for result in results:
        print(
            f"{result['model']:<34} {result['quantization']:<8} {result['perplexity']:<12} "
            f"{result['tokens_per_sec']:<12} {result['peak_memory_gb']:<10.2f} {result['model_load_time_s']:<8.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity for quantization comparison")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name for single-run mode")
    parser.add_argument("--models", type=str, default="", help="Comma-separated model list for matrix runs")
    parser.add_argument(
        "--quantization",
        choices=["none", "fp16", "4bit", "nf4", "fp4", "8bit", "int8"],
        default="fp16",
        help="Quantization for single-run mode",
    )
    parser.add_argument(
        "--quantizations",
        type=str,
        default="",
        help="Comma-separated quantization list (none/fp16, 4bit/nf4/fp4, 8bit/int8)",
    )
    parser.add_argument("--dataset", type=str, default="wikitext", help="HuggingFace dataset name (default: wikitext)")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset config name (default: wikitext-2-raw-v1)",
    )
    parser.add_argument("--split", type=str, default="validation", help="Dataset split (default: validation)")
    parser.add_argument("--subset-size", type=int, default=256, help="Number of non-empty rows to evaluate")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length per sample")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic random seed")
    parser.add_argument(
        "--device-map",
        choices=["auto", "cuda"],
        default="cuda",
        help="Model placement strategy (default: cuda)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run Phase 3 FP4 quality matrix (Qwen2.5 7B/14B/32B x FP16/NF4/FP4)",
    )
    parser.add_argument("--output-json", type=str, default="", help="Optional path to write machine-readable JSON results")
    args = parser.parse_args()

    check_deps()

    print("╔══════════════════════════════════════════════════╗")
    print("║     DGX Spark — Quantization Quality (PPL)      ║")
    print("╚══════════════════════════════════════════════════╝")

    env_info = get_env_info()
    print(f"GPU: {env_info['gpu_name']}")
    print(f"VRAM: {env_info['gpu_memory_gb']:.0f} GB")
    print(f"PyTorch: {env_info['torch_version']}")

    if args.baseline:
        models = BASELINE_MODELS
        quantizations = BASELINE_QUANTIZATIONS
    else:
        models = parse_csv_list(args.models) or [args.model]
        quantizations = parse_csv_list(args.quantizations) or [args.quantization]

    texts = load_wikitext_subset(args.dataset, args.dataset_config, args.split, args.subset_size)

    results = []
    failures = []
    for model_name in models:
        for quantization in quantizations:
            try:
                result = evaluate_perplexity(
                    model_name=model_name,
                    quantization=quantization,
                    texts=texts,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    device_map_mode=args.device_map,
                )
                result.update(
                    {
                        "gpu_name": env_info["gpu_name"],
                        "torch_version": env_info["torch_version"],
                        "cuda_version": env_info["cuda_version"],
                    }
                )
                results.append(result)
            except Exception as exc:
                print(f"FAILED: {model_name} ({quantization}): {exc}")
                failures.append({"model": model_name, "quantization": quantization, "error": str(exc)})

    print_summary(results)

    payload = {
        "benchmark_type": "quality_perplexity",
        "mode": "baseline" if args.baseline else "matrix",
        "environment": env_info,
        "config": {
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "subset_size": args.subset_size,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device_map": args.device_map,
            "models": models,
            "quantizations": quantizations,
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
