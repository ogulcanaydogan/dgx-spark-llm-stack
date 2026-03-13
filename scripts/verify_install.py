#!/usr/bin/env python3
"""Verify DGX Spark LLM stack installation.

Checks GPU, CUDA, PyTorch, and all ML libraries.
Runs a simple matmul benchmark as a smoke test.
"""

import sys
import subprocess

def has_rich():
    try:
        from rich.console import Console
        return True
    except ImportError:
        return False

if has_rich():
    from rich.console import Console
    from rich.table import Table
    console = Console()
    def ok(msg):    console.print(f"  [green]✓[/green] {msg}")
    def warn(msg):  console.print(f"  [yellow]⚠[/yellow] {msg}")
    def fail(msg):  console.print(f"  [red]✗[/red] {msg}")
    def header(msg): console.print(f"\n[bold cyan]{msg}[/bold cyan]")
else:
    def ok(msg):    print(f"  ✓ {msg}")
    def warn(msg):  print(f"  ⚠ {msg}")
    def fail(msg):  print(f"  ✗ {msg}")
    def header(msg): print(f"\n{msg}")


def check_gpu():
    """Check GPU availability and specs."""
    header("GPU Information")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                ok(f"GPU: {parts[0]}")
                ok(f"VRAM: {parts[1]}")
                ok(f"Compute Capability: {parts[2]}")
            else:
                ok(f"GPU: {result.stdout.strip()}")
        else:
            fail("nvidia-smi failed")
    except FileNotFoundError:
        fail("nvidia-smi not found")
    except subprocess.TimeoutExpired:
        fail("nvidia-smi timed out")


def check_cuda():
    """Check CUDA version."""
    header("CUDA")
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line:
                    ok(f"CUDA: {line.strip()}")
                    break
        else:
            fail("nvcc failed")
    except FileNotFoundError:
        fail("nvcc not found — is CUDA installed?")


def check_pytorch():
    """Check PyTorch and CUDA support."""
    header("PyTorch")
    try:
        import torch
        ok(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            ok(f"CUDA available: Yes")
            ok(f"CUDA version (torch): {torch.version.cuda}")
            ok(f"Device: {torch.cuda.get_device_name(0)}")
            cap = torch.cuda.get_device_capability(0)
            ok(f"Compute capability: {cap[0]}.{cap[1]}")
        else:
            fail("CUDA not available in PyTorch")
    except ImportError:
        fail("PyTorch not installed")


def check_library(name, import_name=None):
    """Check if a library is importable and get its version."""
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        ok(f"{name}: {version}")
        return True
    except ImportError:
        fail(f"{name}: not installed")
        return False
    except Exception as e:
        warn(f"{name}: import error — {e}")
        return False


def check_libraries():
    """Check all ML libraries."""
    header("ML Libraries")
    libs = [
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("datasets", "datasets"),
        ("bitsandbytes", "bitsandbytes"),
        ("triton", "triton"),
        ("flash_attn", "flash_attn"),
        ("sentencepiece", "sentencepiece"),
        ("safetensors", "safetensors"),
    ]
    results = {}
    for name, imp in libs:
        results[name] = check_library(name, imp)
    return results


def run_matmul_test():
    """Run a simple GPU matmul as a smoke test."""
    header("GPU Smoke Test (MatMul 4096×4096)")
    try:
        import torch
        if not torch.cuda.is_available():
            fail("Skipping — CUDA not available")
            return

        import time
        device = torch.device("cuda")
        size = 4096

        # Warmup
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        b = torch.randn(size, size, device=device, dtype=torch.float32)
        for _ in range(3):
            torch.mm(a, b)
        torch.cuda.synchronize()

        # Benchmark
        iterations = 10
        start = time.perf_counter()
        for _ in range(iterations):
            torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_time = elapsed / iterations
        # FLOPS for matmul: 2 * N^3
        flops = 2 * (size ** 3) / avg_time
        tflops = flops / 1e12

        ok(f"MatMul: {avg_time*1000:.1f} ms avg ({tflops:.2f} TFLOPS FP32)")
        ok("PASSED")

    except Exception as e:
        fail(f"MatMul test failed: {e}")


def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║     DGX Spark LLM Stack — Installation Check    ║")
    print("╚══════════════════════════════════════════════════╝")

    check_gpu()
    check_cuda()
    check_pytorch()
    check_libraries()
    run_matmul_test()
    print()


if __name__ == "__main__":
    main()
