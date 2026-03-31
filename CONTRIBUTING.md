# Contributing

Thanks for helping improve DGX Spark LLM Stack.

## What To Contribute

This repo is focused on DGX Spark (GB10, `sm_121`) enablement. Good contributions include:

- Documentation improvements (guides, troubleshooting, examples)
- Build/install script fixes
- Benchmark and validation updates
- Container/runtime integration updates
- Upstream tracking items listed in `ROADMAP.md`

## Quick Workflow

1. Fork and clone the repository.
2. Create a focused branch from `main`.
3. Keep scope small (one feature/fix/docs topic per PR).
4. Run relevant validation commands before opening a PR.
5. Open a PR with evidence/logs and roadmap context.

## Validation Commands

Run checks based on your change type:

- General install/stack changes:
```bash
python3 scripts/verify_install.py
```

- Notebook/docs smoke flow changes:
```bash
./scripts/smoke_notebooks.sh
```

- vLLM container changes:
```bash
./scripts/smoke_vllm_container.sh
```

- Benchmark script changes (quick sanity):
```bash
python3 scripts/benchmark_inference.py --model Qwen/Qwen2.5-0.5B-Instruct --tokens 64 --runs 1 --device-map cuda
python3 scripts/benchmark_training.py --model Qwen/Qwen2.5-0.5B-Instruct --method lora --steps 2 --batch-size 1 --seq-length 256
python3 scripts/evaluate_perplexity.py --model Qwen/Qwen2.5-0.5B-Instruct --quantization fp16 --subset-size 8 --max-length 128 --batch-size 1 --device-map cuda
```

## Commit and PR Rules

- Use clear, scoped commit messages.
- Keep PRs single-purpose.
- Reference the relevant roadmap line(s) in the PR description.
- Do not mark roadmap items complete without execution evidence.

## PR Checklist

Before requesting review, include:

- What changed and why
- Exact commands executed
- Key output/log snippets
- Risk/rollback notes
- Any follow-up work left open

## DGX Spark Evidence Requirement

For GPU/benchmark/container changes, include command evidence from a real Spark run:

- Environment basics (`python3 --version`, `nvidia-smi`)
- The exact command used
- PASS/FAIL result and important metrics or endpoint checks

Without this evidence, roadmap closure is considered incomplete.

## Issue Template (Recommended)

Use this structure when opening issues:

```md
### Environment
- Host: DGX Spark
- GPU: NVIDIA GB10 (`sm_121`)
- Python:
- CUDA / Driver:

### Steps to Reproduce
1.
2.
3.

### Expected Behavior

### Actual Behavior

### Logs / Command Output

### Notes
```
