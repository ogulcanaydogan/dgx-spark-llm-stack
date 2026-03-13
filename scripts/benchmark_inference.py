#!/usr/bin/env python3
"""Benchmark LLM inference on DGX Spark (GB10, sm_121)."""

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
BASELINE_QUANTIZATIONS = ["fp16", "nf4"]


def check_deps():
    try:
        import torch

        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            sys.exit(1)
    except ImportError:
        print("ERROR: PyTorch not installed")
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
    if normalized in {"8bit", "int8"}:
        return "int8"
    raise ValueError(f"Unsupported quantization: {value}")


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


def benchmark_generation(model_name: str, max_new_tokens: int = 128, quantization: str = "none", num_runs: int = 3):
    """Benchmark text generation for one model+quantization pair."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant_mode = normalize_quantization(quantization)

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Quantization: {quant_mode}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"{'=' * 60}")

    load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

    if quant_mode == "nf4":
        try:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
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

    print("Loading model...")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    mem_allocated = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory: {mem_allocated:.1f} GB")

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize()

    times = []
    tokens_generated = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        n_tokens = output.shape[1] - inputs["input_ids"].shape[1]
        times.append(elapsed)
        tokens_generated.append(n_tokens)

    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    tokens_per_sec = avg_tokens / avg_time
    ttft_ms_rough = times[0] / tokens_generated[0] * 1000

    print(f"\nResults ({num_runs} runs):")
    print(f"  Avg time:       {avg_time:.2f}s")
    print(f"  Avg tokens:     {avg_tokens:.0f}")
    print(f"  Tokens/sec:     {tokens_per_sec:.1f}")
    print(f"  Time to first:  ~{ttft_ms_rough:.0f}ms (rough)")

    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "quantization": quant_mode,
        "max_new_tokens": max_new_tokens,
        "num_runs": num_runs,
        "avg_time_s": round(avg_time, 3),
        "avg_tokens": round(avg_tokens, 1),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "ttft_ms_rough": round(ttft_ms_rough, 2),
        "gpu_memory_gb": round(mem_allocated, 2),
        "model_load_time_s": round(load_time, 3),
    }


def write_json_output(path: str, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
    print(f"\nWrote JSON results to: {path}")


def print_summary(results):
    print(f"\n{'=' * 74}")
    print("SUMMARY")
    print(f"{'=' * 74}")
    print(f"{'Model':<34} {'Quant':<8} {'tok/s':<10} {'VRAM':<10} {'Load(s)':<8}")
    print("-" * 74)
    for result in results:
        print(
            f"{result['model']:<34} {result['quantization']:<8} "
            f"{result['tokens_per_sec']:<10} {result['gpu_memory_gb']:<10.2f} {result['model_load_time_s']:<8.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM inference on DGX Spark")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="Model name or path for single-run mode (default: microsoft/phi-2)",
    )
    parser.add_argument("--models", type=str, default="", help="Comma-separated model list for matrix runs")
    parser.add_argument("--tokens", type=int, default=128, help="Max new tokens to generate (default: 128)")
    parser.add_argument(
        "--quantization",
        choices=["none", "fp16", "4bit", "nf4", "8bit", "int8"],
        default="none",
        help="Quantization for single-run mode",
    )
    parser.add_argument(
        "--quantizations",
        type=str,
        default="",
        help="Comma-separated quantization list (none/fp16, 4bit/nf4, 8bit/int8)",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs (default: 3)")
    parser.add_argument("--suite", action="store_true", help="Run legacy benchmark suite with multiple models")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run Phase 3 baseline matrix (Qwen2.5 7B/14B x FP16/NF4)",
    )
    parser.add_argument("--output-json", type=str, default="", help="Optional path to write machine-readable JSON results")
    args = parser.parse_args()

    check_deps()

    print("╔══════════════════════════════════════════════════╗")
    print("║     DGX Spark — Inference Benchmark              ║")
    print("╚══════════════════════════════════════════════════╝")

    env_info = get_env_info()
    print(f"GPU: {env_info['gpu_name']}")
    print(f"VRAM: {env_info['gpu_memory_gb']:.0f} GB")
    print(f"PyTorch: {env_info['torch_version']}")

    failures = []
    results = []

    if args.baseline:
        models = BASELINE_MODELS
        quantizations = BASELINE_QUANTIZATIONS
    elif args.suite:
        matrix = [
            ("microsoft/phi-2", "none"),
            ("microsoft/phi-2", "4bit"),
            ("meta-llama/Llama-3.2-3B", "none"),
            ("meta-llama/Llama-3.2-3B", "4bit"),
        ]
        models = []
        quantizations = []
        for model_name, quantization in matrix:
            models.append(model_name)
            quantizations.append(quantization)
        for model_name, quantization in matrix:
            try:
                results.append(benchmark_generation(model_name, args.tokens, quantization, args.runs))
            except Exception as exc:
                print(f"FAILED: {model_name} ({quantization}): {exc}")
                failures.append({"model": model_name, "quantization": quantization, "error": str(exc)})
        print_summary(results)
        payload = {
            "benchmark_type": "inference",
            "mode": "suite",
            "environment": env_info,
            "config": {"tokens": args.tokens, "runs": args.runs},
            "results": results,
            "failures": failures,
        }
        if args.output_json:
            write_json_output(args.output_json, payload)
        return
    else:
        models = parse_csv_list(args.models) or [args.model]
        quantizations = parse_csv_list(args.quantizations) or [args.quantization]

    for model_name in models:
        for quantization in quantizations:
            try:
                result = benchmark_generation(model_name, args.tokens, quantization, args.runs)
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
        "benchmark_type": "inference",
        "mode": "baseline" if args.baseline else "matrix",
        "environment": env_info,
        "config": {
            "tokens": args.tokens,
            "runs": args.runs,
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
