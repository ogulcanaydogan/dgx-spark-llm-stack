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
- Matrix: `7B, 14B, 32B` x `FP16, NF4, FP4`
- Extension kept from previous run: `72B` x `NF4`
- Prompt: fixed single prompt in script
- Generation length: `256` new tokens
- Timed runs per row: `5`
- Device map: `cuda`

Quality setup:
- Metric: perplexity (`exp(avg_nll)`)
- Dataset: WikiText-2 validation (`wikitext-2-raw-v1`)
- Deterministic subset: first `32` non-empty rows
- Max length: `512`
- Seed: `1337`
- Batch size: `1`
- Matrix: `7B, 14B, 32B` x `FP16, NF4, FP4`

Training setup:
- Matrix: `7B, 14B` x `LoRA, QLoRA`
- Steps: `100`
- Batch size: `1`
- Sequence length: `512`
- Dataset: deterministic synthetic text batch generated in script

Commands used:

```bash
python scripts/benchmark_inference.py \
  --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct,Qwen/Qwen2.5-32B-Instruct \
  --quantizations fp16,nf4,fp4 \
  --tokens 256 \
  --runs 5 \
  --device-map cuda \
  --output-json artifacts/benchmarks/inference-fp4-2026-03-13.json

python scripts/evaluate_perplexity.py \
  --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-14B-Instruct,Qwen/Qwen2.5-32B-Instruct \
  --quantizations fp16,nf4,fp4 \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split validation \
  --subset-size 32 \
  --max-length 512 \
  --batch-size 1 \
  --seed 1337 \
  --device-map cuda \
  --output-json artifacts/benchmarks/quality-ppl-fp16-nf4-fp4-2026-03-13.json
```

Artifacts:

- `artifacts/benchmarks/phase3-baseline-2026-03-13.json` (combined inference + training + quality)
- `artifacts/benchmarks/inference-fp4-2026-03-13.json`
- `artifacts/benchmarks/quality-ppl-fp16-nf4-fp4-2026-03-13.json`
- `artifacts/benchmarks/inference-extended-32b-72b-2026-03-13.json`

## Results

### Inference Throughput (All Sizes + FP4)

| Model | Quantization | Tokens/sec | GPU Memory (GB) | Model Load Time (s) |
|---|---|---:|---:|---:|
| Qwen2.5-7B-Instruct | FP16 | 12.61 | 15.23 | 4.816 |
| Qwen2.5-7B-Instruct | NF4 | 35.52 | 5.87 | 5.238 |
| Qwen2.5-7B-Instruct | FP4 | 35.44 | 9.54 | 4.033 |
| Qwen2.5-14B-Instruct | FP16 | 7.58 | 36.90 | 10.627 |
| Qwen2.5-14B-Instruct | NF4 | 18.91 | 17.95 | 7.647 |
| Qwen2.5-14B-Instruct | FP4 | 18.98 | 25.42 | 6.724 |
| Qwen2.5-32B-Instruct | FP16 | 3.45 | 65.54 | 44.117 |
| Qwen2.5-32B-Instruct | NF4 | 9.74 | 20.71 | 16.627 |
| Qwen2.5-32B-Instruct | FP4 | 9.79 | 38.29 | 14.204 |
| Qwen2.5-72B-Instruct | NF4 | 3.80 | 44.51 | 2750.272 |

### Quantization Quality (Perplexity)

| Model | Quantization | Perplexity | Eval Tokens/sec | Peak GPU Memory (GB) |
|---|---|---:|---:|---:|
| Qwen2.5-7B-Instruct | FP16 | 817596.794711 | 36.89 | 16.24 |
| Qwen2.5-7B-Instruct | NF4 | 352028.654832 | 35.92 | 6.86 |
| Qwen2.5-7B-Instruct | FP4 | 238966.601718 | 35.73 | 10.54 |
| Qwen2.5-14B-Instruct | FP16 | 135374.019262 | 14.12 | 38.04 |
| Qwen2.5-14B-Instruct | NF4 | 54627.458419 | 13.69 | 19.10 |
| Qwen2.5-14B-Instruct | FP4 | 119353.758199 | 13.71 | 26.56 |
| Qwen2.5-32B-Instruct | FP16 | 17315.029933 | 10.24 | 66.75 |
| Qwen2.5-32B-Instruct | NF4 | 49750.648188 | 9.81 | 21.93 |
| Qwen2.5-32B-Instruct | FP4 | 23262.711831 | 9.79 | 39.52 |

### Training Throughput

| Model | Method | Samples/sec | Tokens/sec | Peak GPU Memory (GB) | Model Load Time (s) |
|---|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | LoRA | 2.5751 | 1318.44 | 19.83 | 5.383 |
| Qwen2.5-7B-Instruct | QLoRA | 0.8899 | 455.64 | 9.64 | 4.433 |
| Qwen2.5-14B-Instruct | LoRA | 1.1710 | 599.56 | 40.87 | 7.694 |
| Qwen2.5-14B-Instruct | QLoRA | 0.4065 | 208.13 | 15.78 | 6.941 |

### Cross-Hardware (Strict-Match Rule)

Strict-match rule: a value is shown only if all of these match exactly: model, quantization method, token setting (`256` output tokens), and comparable runtime setup. Otherwise the cell is `N/A`.

| Model | Quantization | Tokens | DGX Spark (tok/s) | RTX 4090 | A100 | Match Note |
|---|---|---:|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | FP16 | 256 | 12.61 | N/A | N/A | No strict-match external row found |
| Qwen2.5-7B-Instruct | NF4 | 256 | 35.52 | N/A | N/A | No strict-match external row found |
| Qwen2.5-7B-Instruct | FP4 | 256 | 35.44 | N/A | N/A | No strict-match external row found |
| Qwen2.5-14B-Instruct | FP16 | 256 | 7.58 | N/A | N/A | No strict-match external row found |
| Qwen2.5-14B-Instruct | NF4 | 256 | 18.91 | N/A | N/A | No strict-match external row found |
| Qwen2.5-14B-Instruct | FP4 | 256 | 18.98 | N/A | N/A | No strict-match external row found |
| Qwen2.5-32B-Instruct | FP16 | 256 | 3.45 | N/A | N/A | No strict-match external row found |
| Qwen2.5-32B-Instruct | NF4 | 256 | 9.74 | N/A | N/A | No strict-match external row found |
| Qwen2.5-32B-Instruct | FP4 | 256 | 9.79 | N/A | N/A | No strict-match external row found |

### External Reference Numbers (Non-Strict, Cited)

| Hardware | Model / Setup | Throughput | Source | Retrieved (UTC) | Config Note |
|---|---|---:|---|---|---|
| A100-80GB | Qwen2.5-7B-Instruct (BF16, vLLM, input length 1) | 84.28 tok/s | [Qwen Speed Benchmark](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html) | 2026-03-13 | Not strict-match: quant/runtime/token config differ |
| A100-80GB | Qwen2.5-14B-Instruct (BF16, transformers, input length 1) | 31.63 tok/s | [Qwen Speed Benchmark](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html) | 2026-03-13 | Not strict-match: quant/runtime/token config differ |
| A100-80GB | Qwen2.5-32B-Instruct (AWQ, vLLM, input length 1) | 66.33 tok/s | [Qwen Speed Benchmark](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html) | 2026-03-13 | Not strict-match: quant/runtime/token config differ |
| 2x RTX 4090 | DeepSeek-R1-Distill-Qwen-14B (Q6_K, 32k context) | 155.72 tok/s | [IOG Qwen Inference Benchmark](https://github.com/International-Open-Source-Guild/InterOp/tree/main/inference_benchmark) | 2026-03-13 | Not strict-match: model/GPU count/quant/context differ |

## Notes and Limits

- Perplexity values are produced by the project’s deterministic fixed-subset protocol and are intended for within-run relative comparison across FP16/NF4/FP4.
- First-time model downloads can dominate load time. Throughput numbers are from timed inference/evaluation loops.
- Strict-match cross-hardware cells remain `N/A` until equivalent public measurements exist for the exact model + quant + token setup.

## Next Run Targets

- Add strict-match external rows when reproducible 4090/A100 measurements with `tokens=256` become available.
- Expand perplexity protocol with a larger fixed subset and additional quality metrics in the next Phase 3 iteration.
