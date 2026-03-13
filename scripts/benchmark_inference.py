#!/usr/bin/env python3
"""Benchmark LLM inference on DGX Spark (GB10, sm_121).

Tests token generation speed with various model sizes and quantization levels.
"""

import time
import argparse
import sys

def check_deps():
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            sys.exit(1)
    except ImportError:
        print("ERROR: PyTorch not installed")
        sys.exit(1)

def benchmark_generation(model_name: str, max_new_tokens: int = 128,
                         quantization: str = "none", num_runs: int = 3):
    """Benchmark text generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Quantization: {quantization}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"{'='*60}")

    # Load model
    load_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}

    if quantization == "4bit":
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            print("WARNING: bitsandbytes not available, using FP16")
    elif quantization == "8bit":
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            print("WARNING: bitsandbytes not available, using FP16")

    print("Loading model...")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    # Memory usage
    import torch
    mem_allocated = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory: {mem_allocated:.1f} GB")

    # Benchmark
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=16, do_sample=False)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    tokens_generated = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        n_tokens = output.shape[1] - inputs["input_ids"].shape[1]
        times.append(elapsed)
        tokens_generated.append(n_tokens)

    # Results
    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    tokens_per_sec = avg_tokens / avg_time

    print(f"\nResults ({num_runs} runs):")
    print(f"  Avg time:       {avg_time:.2f}s")
    print(f"  Avg tokens:     {avg_tokens:.0f}")
    print(f"  Tokens/sec:     {tokens_per_sec:.1f}")
    print(f"  Time to first:  ~{times[0]/tokens_generated[0]*1000:.0f}ms (rough)")

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "quantization": quantization,
        "avg_time_s": round(avg_time, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "gpu_memory_gb": round(mem_allocated, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM inference on DGX Spark")
    parser.add_argument("--model", type=str, default="microsoft/phi-2",
                        help="Model name or path (default: microsoft/phi-2)")
    parser.add_argument("--tokens", type=int, default=128,
                        help="Max new tokens to generate (default: 128)")
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"],
                        default="none", help="Quantization method")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of benchmark runs (default: 3)")
    parser.add_argument("--suite", action="store_true",
                        help="Run full benchmark suite with multiple models")
    args = parser.parse_args()

    check_deps()

    print("╔══════════════════════════════════════════════════╗")
    print("║     DGX Spark — Inference Benchmark              ║")
    print("╚══════════════════════════════════════════════════╝")

    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    total_mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f"VRAM: {total_mem / 1e9:.0f} GB")
    print(f"PyTorch: {torch.__version__}")

    if args.suite:
        models = [
            ("microsoft/phi-2", "none"),
            ("microsoft/phi-2", "4bit"),
            ("meta-llama/Llama-3.2-3B", "none"),
            ("meta-llama/Llama-3.2-3B", "4bit"),
        ]
        results = []
        for model, quant in models:
            try:
                r = benchmark_generation(model, args.tokens, quant, args.runs)
                results.append(r)
            except Exception as e:
                print(f"FAILED: {model} ({quant}): {e}")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<35} {'Quant':<8} {'tok/s':<10} {'VRAM':<8}")
        print("-" * 60)
        for r in results:
            print(f"{r['model']:<35} {r['quantization']:<8} {r['tokens_per_sec']:<10} {r['gpu_memory_gb']:.1f} GB")
    else:
        benchmark_generation(args.model, args.tokens, args.quantization, args.runs)

if __name__ == "__main__":
    main()
