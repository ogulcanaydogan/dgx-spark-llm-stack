# Speculative Decoding Benchmarks (DGX Spark)

This guide runs a reproducible speculative decoding benchmark using `transformers` with `assistant_model`.

## Why This Matters

Speculative decoding can increase serving throughput by proposing draft tokens from a smaller model and verifying them with a larger target model.
This run is a measurement artifact, not a guarantee that speedup is always positive.

## One-Command Benchmark

From repository root:

```bash
python3 scripts/benchmark_speculative_decoding.py \
  --target-model Qwen/Qwen2.5-7B-Instruct \
  --draft-model Qwen/Qwen2.5-0.5B-Instruct \
  --tokens 256 \
  --runs 5 \
  --output-json artifacts/benchmarks/speculative-decoding-2026-04-02.json
```

## Output Artifact

Artifact path:

- `artifacts/benchmarks/speculative-decoding-<date>.json`

Required fields:

- `baseline.avg_time_s`
- `baseline.tokens_per_sec`
- `baseline.avg_output_tokens`
- `baseline.gpu_memory_gb`
- `baseline.model_load_time_s`
- `speculative.avg_time_s`
- `speculative.tokens_per_sec`
- `speculative.avg_output_tokens`
- `speculative.gpu_memory_gb`
- `speculative.model_load_time_s`
- `comparison.speedup_ratio_vs_baseline`

## PASS Signatures

Expected console markers:

- `[specdec] baseline_run complete ...`
- `[specdec] speculative_run complete ...`
- `[specdec] speedup_ratio_vs_baseline=...`
- `[specdec] PASS: speculative decoding benchmark completed`

## Tunable Parameters

- `--target-model`: target model id
- `--draft-model`: assistant model id
- `--tokens`: max generated tokens
- `--runs`: timed run count per mode
- `--prompt`: shared prompt for both modes
- `--output-json`: explicit artifact path

## Spark Troubleshooting

1. CUDA unavailable
- Verify host GPU health with `nvidia-smi`.
- Confirm PyTorch CUDA visibility:
  - `python3 -c "import torch; print(torch.cuda.is_available())"`

2. Model download/auth issues
- Export HF token if required:
  - `export HF_TOKEN=hf_xxx`

3. OOM during load/generation
- Reduce workload:
  - lower `--tokens`
  - lower `--runs`
- Retry after ensuring no stale heavy process occupies GPU memory.

4. Slow first run
- Initial model pulls and cache warmup can dominate first execution time; compare with subsequent runs for stable behavior.
