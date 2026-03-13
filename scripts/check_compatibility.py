#!/usr/bin/env python3
"""Check compatibility of ML libraries on DGX Spark (GB10, sm_121).

Tests each library for import, version, and known sm_121 issues.
Outputs both a table and optional JSON report.
"""

import sys
import json
import importlib
import argparse
from dataclasses import dataclass, asdict

@dataclass
class LibraryStatus:
    name: str
    version: str
    importable: bool
    status: str  # "ok", "warning", "broken", "missing"
    notes: str

def check_lib(name: str, import_name: str, known_issues: str = "") -> LibraryStatus:
    """Check a single library."""
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        return LibraryStatus(
            name=name, version=version, importable=True,
            status="ok", notes=known_issues or "Working"
        )
    except ImportError:
        return LibraryStatus(
            name=name, version="-", importable=False,
            status="missing", notes="Not installed"
        )
    except Exception as e:
        return LibraryStatus(
            name=name, version="-", importable=False,
            status="broken", notes=str(e)
        )

def check_torch_cuda() -> LibraryStatus:
    """Special check for PyTorch CUDA support."""
    try:
        import torch
        version = torch.__version__
        if not torch.cuda.is_available():
            return LibraryStatus("pytorch-cuda", version, True, "broken",
                                 "CUDA not available")
        device_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        return LibraryStatus("pytorch-cuda", version, True, "ok",
                             f"{device_name} (sm_{cap[0]}{cap[1]})")
    except ImportError:
        return LibraryStatus("pytorch-cuda", "-", False, "missing",
                             "PyTorch not installed")
    except Exception as e:
        return LibraryStatus("pytorch-cuda", "-", False, "broken", str(e))

def check_triton_sm121() -> LibraryStatus:
    """Check if Triton works with sm_121."""
    try:
        import triton
        version = triton.__version__
        # Triton's ptxas may fail with sm_121a
        return LibraryStatus("triton", version, True, "warning",
                             "Imported OK, but sm_121a ptxas may fail at runtime")
    except ImportError:
        return LibraryStatus("triton", "-", False, "missing", "Not installed")
    except Exception as e:
        return LibraryStatus("triton", "-", False, "broken", str(e))

def check_flash_attn() -> LibraryStatus:
    """Check flash-attention."""
    try:
        import flash_attn
        version = flash_attn.__version__
        return LibraryStatus("flash-attention", version, True, "warning",
                             "Imported but no sm_121 kernels — expect runtime failures")
    except ImportError:
        return LibraryStatus("flash-attention", "-", False, "missing",
                             "Not installed (use SDPA fallback)")
    except Exception as e:
        return LibraryStatus("flash-attention", "-", False, "broken", str(e))

def run_all_checks() -> list[LibraryStatus]:
    """Run all compatibility checks."""
    results = []

    # Core
    results.append(check_torch_cuda())
    results.append(check_triton_sm121())
    results.append(check_flash_attn())

    # Standard libs
    libs = [
        ("transformers", "transformers", ""),
        ("accelerate", "accelerate", ""),
        ("peft", "peft", ""),
        ("trl", "trl", ""),
        ("datasets", "datasets", ""),
        ("bitsandbytes", "bitsandbytes", "FP4/NF4 quantization works"),
        ("safetensors", "safetensors", ""),
        ("sentencepiece", "sentencepiece", ""),
        ("tokenizers", "tokenizers", ""),
        ("wandb", "wandb", ""),
    ]
    for name, imp, notes in libs:
        results.append(check_lib(name, imp, notes))

    return results

STATUS_SYMBOLS = {
    "ok": "✅",
    "warning": "⚠️",
    "broken": "❌",
    "missing": "➖",
}

def print_table(results: list[LibraryStatus]):
    """Print results as a formatted table."""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="DGX Spark Compatibility Report")
        table.add_column("Library", style="bold")
        table.add_column("Version")
        table.add_column("Status")
        table.add_column("Notes")
        for r in results:
            sym = STATUS_SYMBOLS.get(r.status, "?")
            style = {"ok": "green", "warning": "yellow", "broken": "red", "missing": "dim"}.get(r.status, "")
            table.add_row(r.name, r.version, sym, r.notes, style=style)
        console.print(table)
    except ImportError:
        # Fallback to plain text
        print(f"\n{'Library':<20} {'Version':<15} {'Status':<8} Notes")
        print("-" * 75)
        for r in results:
            sym = STATUS_SYMBOLS.get(r.status, "?")
            print(f"{r.name:<20} {r.version:<15} {sym:<8} {r.notes}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Check ML library compatibility on DGX Spark")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", type=str, help="Write report to file")
    args = parser.parse_args()

    results = run_all_checks()

    if args.json or args.output:
        data = [asdict(r) for r in results]
        json_str = json.dumps(data, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(json_str)
            print(f"Report written to {args.output}")
        if args.json:
            print(json_str)
    else:
        print_table(results)

    # Summary
    total = len(results)
    ok_count = sum(1 for r in results if r.status == "ok")
    warn_count = sum(1 for r in results if r.status == "warning")
    broken_count = sum(1 for r in results if r.status == "broken")
    missing_count = sum(1 for r in results if r.status == "missing")
    print(f"Summary: {ok_count} OK, {warn_count} warnings, {broken_count} broken, {missing_count} missing (of {total})")

if __name__ == "__main__":
    main()
