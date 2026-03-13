# Benchmarks (Phase 3)

This document captures the Phase 3 benchmark runs for DGX Spark.

## Baseline Snapshot

- Date: `2026-03-13`
- Host: DGX Spark (`NVIDIA GB10`, ARM64)
- PyTorch: `2.9.1a0+gitd38164a` (custom wheel)
- CUDA runtime: `13.0`
- Models: `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, `Qwen/Qwen2.5-72B-Instruct`

## Methodology

Inference setup:
- Matrix: `7B, 14B` x `FP16, NF4` plus `32B` x `FP16, NF4` and `72B` x `NF4`
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

python scripts/benchmark_inference.py \
  --models Qwen/Qwen2.5-32B-Instruct \
  --quantizations fp16,nf4 \
  --tokens 256 \
  --runs 5 \
  --device-map cuda \
  --output-json artifacts/benchmarks/inference-32b-2026-03-13.json

python scripts/benchmark_inference.py \
  --models Qwen/Qwen2.5-72B-Instruct \
  --quantizations nf4 \
  --tokens 256 \
  --runs 5 \
  --device-map cuda \
  --output-json artifacts/benchmarks/inference-72b-nf4-2026-03-13.json
```

Combined artifact:

`artifacts/benchmarks/phase3-baseline-2026-03-13.json`

Extended inference artifact:

`artifacts/benchmarks/inference-extended-32b-72b-2026-03-13.json`

## Results

### Inference Throughput

| Model | Quantization | Tokens/sec | GPU Memory (GB) | Model Load Time (s) |
|---|---|---:|---:|---:|
| Qwen2.5-7B-Instruct | FP16 | 12.96 | 15.23 | 630.80 |
| Qwen2.5-7B-Instruct | NF4 | 35.92 | 5.87 | 5.78 |
| Qwen2.5-14B-Instruct | FP16 | 7.49 | 33.22 | 1284.27 |
| Qwen2.5-14B-Instruct | NF4 | 18.53 | 14.27 | 8.13 |
| Qwen2.5-32B-Instruct | FP16 | 3.46 | 65.53 | 31.71 |
| Qwen2.5-32B-Instruct | NF4 | 9.73 | 20.71 | 22.14 |
| Qwen2.5-72B-Instruct | NF4 | 3.80 | 44.51 | 2750.27 |

### Training Throughput

| Model | Method | Samples/sec | Tokens/sec | Peak GPU Memory (GB) | Model Load Time (s) |
|---|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | LoRA | 2.5751 | 1318.44 | 19.83 | 5.38 |
| Qwen2.5-7B-Instruct | QLoRA | 0.8899 | 455.64 | 9.64 | 4.43 |
| Qwen2.5-14B-Instruct | LoRA | 1.1710 | 599.56 | 40.87 | 7.69 |
| Qwen2.5-14B-Instruct | QLoRA | 0.4065 | 208.13 | 15.78 | 6.94 |

## Notes and Limits

- Quantization comparison currently covers `FP16` vs `NF4`; `FP4` is still pending.
- First-time model downloads dominate load time. Throughput numbers are from timed generation/training loops, not download duration.

## Next Run Targets

- Add `FP4` quantization row and quality/performance comparison.
- Add cross-hardware comparison table (DGX Spark vs RTX 4090 vs A100 where available).
