# Benchmarks (Phase 3 Baseline)

This document captures the first **Phase 3 baseline** benchmark run for DGX Spark.

## Baseline Snapshot

- Date: `2026-03-13`
- Host: DGX Spark (`NVIDIA GB10`, ARM64)
- PyTorch: `2.9.1a0+gitd38164a` (custom wheel)
- CUDA runtime: `13.0`
- Models: `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`

## Methodology

Inference setup:
- Matrix: `7B, 14B` x `FP16, NF4`
- Prompt: fixed single prompt in script
- Generation length: `256` new tokens
- Timed runs per row: `5`

Training setup:
- Matrix: `7B, 14B` x `LoRA, QLoRA`
- Steps: `100`
- Batch size: `1`
- Sequence length: `512`
- Dataset: deterministic synthetic text batch generated in script

Commands used:

```bash
python scripts/benchmark_inference.py \
  --baseline \
  --tokens 256 \
  --runs 5 \
  --output-json artifacts/benchmarks/inference-baseline-2026-03-13.json

python scripts/benchmark_training.py \
  --baseline \
  --steps 100 \
  --batch-size 1 \
  --seq-length 512 \
  --output-json artifacts/benchmarks/training-baseline-2026-03-13.json
```

Combined artifact:

`artifacts/benchmarks/phase3-baseline-2026-03-13.json`

## Results

### Inference Throughput

| Model | Quantization | Tokens/sec | GPU Memory (GB) | Model Load Time (s) |
|---|---|---:|---:|---:|
| Qwen2.5-7B-Instruct | FP16 | 12.96 | 15.23 | 630.80 |
| Qwen2.5-7B-Instruct | NF4 | 35.92 | 5.87 | 5.78 |
| Qwen2.5-14B-Instruct | FP16 | 7.49 | 33.22 | 1284.27 |
| Qwen2.5-14B-Instruct | NF4 | 18.53 | 14.27 | 8.13 |

### Training Throughput

| Model | Method | Samples/sec | Tokens/sec | Peak GPU Memory (GB) | Model Load Time (s) |
|---|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | LoRA | 2.5751 | 1318.44 | 19.83 | 5.38 |
| Qwen2.5-7B-Instruct | QLoRA | 0.8899 | 455.64 | 9.64 | 4.43 |
| Qwen2.5-14B-Instruct | LoRA | 1.1710 | 599.56 | 40.87 | 7.69 |
| Qwen2.5-14B-Instruct | QLoRA | 0.4065 | 208.13 | 15.78 | 6.94 |

## Notes and Limits

- This baseline does **not** include `32B`/`72B` model runs yet.
- Quantization comparison currently covers `FP16` vs `NF4`; `FP4` is still pending.
- First-time model downloads dominate load time. Throughput numbers are from timed generation/training loops, not download duration.

## Next Run Targets

- Extend inference matrix to include `32B` and `72B`.
- Add `FP4` quantization row and quality/performance comparison.
- Add cross-hardware comparison table (DGX Spark vs RTX 4090 vs A100 where available).
