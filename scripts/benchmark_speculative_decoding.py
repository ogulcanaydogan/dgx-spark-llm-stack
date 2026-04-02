#!/usr/bin/env python3
"""Benchmark speculative decoding on DGX Spark using transformers assistant_model."""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone


def check_deps():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            sys.exit(1)
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        print("Install with: pip install torch transformers")
        sys.exit(1)


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


def write_json_output(path: str, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
    print(f"\nWrote JSON results to: {path}")


def run_mode(
    model,
    tokenizer,
    prompt,
    runs,
    max_new_tokens,
    assistant_model=None,
    assistant_tokenizer=None,
):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if assistant_model is not None:
        generate_kwargs["assistant_model"] = assistant_model
        generate_kwargs["tokenizer"] = tokenizer
        generate_kwargs["assistant_tokenizer"] = assistant_tokenizer

    # Warmup
    warmup_kwargs = dict(generate_kwargs)
    warmup_kwargs["max_new_tokens"] = min(16, max_new_tokens)
    with torch.no_grad():
        model.generate(**inputs, **warmup_kwargs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    output_tokens = []
    for index in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        times.append(elapsed)
        output_tokens.append(new_tokens)
        print(f"  Run {index + 1}/{runs}: {elapsed:.3f}s ({new_tokens} tokens)")

    avg_time = sum(times) / len(times)
    avg_tokens = sum(output_tokens) / len(output_tokens)
    tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0.0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    return {
        "avg_time_s": round(avg_time, 4),
        "tokens_per_sec": round(tokens_per_sec, 4),
        "avg_output_tokens": round(avg_tokens, 2),
        "gpu_memory_gb": round(peak_mem_gb, 3),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding on DGX Spark")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Target model ID")
    parser.add_argument("--draft-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Draft/assistant model ID")
    parser.add_argument("--tokens", type=int, default=256, help="Max new tokens per generation")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs per mode")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Summarize why speculative decoding can improve LLM serving throughput in one paragraph.",
        help="Prompt used for baseline and speculative runs",
    )
    parser.add_argument("--output-json", type=str, default="", help="Path to write JSON benchmark output")
    args = parser.parse_args()

    check_deps()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("╔══════════════════════════════════════════════════╗")
    print("║  DGX Spark — Speculative Decoding Benchmark      ║")
    print("╚══════════════════════════════════════════════════╝")

    env_info = get_env_info()
    print(f"GPU: {env_info['gpu_name']}")
    print(f"PyTorch: {env_info['torch_version']}")

    device_map = {"": 0}
    dtype = torch.float16

    print(f"\nLoading target model: {args.target_model}")
    target_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        device_map=device_map,
        torch_dtype=dtype,
    )
    target_load_time = time.perf_counter() - target_load_start
    print(f"Target model loaded in {target_load_time:.2f}s")

    print("\n[baseline] Running normal decode")
    baseline = run_mode(
        model=target_model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        runs=args.runs,
        max_new_tokens=args.tokens,
        assistant_model=None,
    )
    baseline["model_load_time_s"] = round(target_load_time, 4)
    print(
        f"[specdec] baseline_run complete avg_time_s={baseline['avg_time_s']} "
        f"tokens_per_sec={baseline['tokens_per_sec']}"
    )

    print(f"\nLoading draft model: {args.draft_model}")
    draft_load_start = time.perf_counter()
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        device_map=device_map,
        torch_dtype=dtype,
    )
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
    if draft_tokenizer.pad_token is None:
        draft_tokenizer.pad_token = draft_tokenizer.eos_token
    draft_load_time = time.perf_counter() - draft_load_start
    print(f"Draft model loaded in {draft_load_time:.2f}s")

    print("\n[speculative] Running assistant_model decode")
    speculative = run_mode(
        model=target_model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        runs=args.runs,
        max_new_tokens=args.tokens,
        assistant_model=draft_model,
        assistant_tokenizer=draft_tokenizer,
    )
    speculative["model_load_time_s"] = round(draft_load_time, 4)
    print(
        f"[specdec] speculative_run complete avg_time_s={speculative['avg_time_s']} "
        f"tokens_per_sec={speculative['tokens_per_sec']}"
    )

    baseline_tps = baseline["tokens_per_sec"]
    speculative_tps = speculative["tokens_per_sec"]
    speedup = (speculative_tps / baseline_tps) if baseline_tps > 0 else None

    payload = {
        "benchmark_type": "speculative_decoding",
        "environment": env_info,
        "config": {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "tokens": args.tokens,
            "runs": args.runs,
            "prompt": args.prompt,
            "dtype": "float16",
            "device_map": "cuda:0",
        },
        "baseline": baseline,
        "speculative": speculative,
        "comparison": {
            "speedup_ratio_vs_baseline": round(speedup, 4) if speedup is not None else None,
        },
    }

    print(
        f"\n[specdec] speedup_ratio_vs_baseline={payload['comparison']['speedup_ratio_vs_baseline']}"
    )

    output_path = args.output_json or f"artifacts/benchmarks/speculative-decoding-{datetime.now(timezone.utc).date().isoformat()}.json"
    write_json_output(output_path, payload)
    print("[specdec] PASS: speculative decoding benchmark completed")


if __name__ == "__main__":
    main()
