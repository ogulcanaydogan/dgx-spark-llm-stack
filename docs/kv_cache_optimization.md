# KV Cache Optimization (128 GB Unified Memory)

This guide runs a deterministic vLLM parameter sweep to select a stable KV-cache profile on DGX Spark.

## Why This Matters

`max-model-len` and `gpu-memory-utilization` directly change KV cache allocation behavior in vLLM.
On 128 GB unified memory systems, tuning these values can improve stable concurrent throughput.

## One-Command Sweep

From repository root:

```bash
./scripts/benchmark_kv_cache_optimization.sh
```

Default sweep matrix:

- `VLLM_MAX_MODEL_LEN`: `4096,8192,16384`
- `VLLM_GPU_MEMORY_UTILIZATION`: `0.85,0.90,0.95`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Workload: existing continuous batching smoke (`4` serial + `16` concurrent, `8` workers, `48` max tokens)
- HF cache mount: enabled by default (`$HOME/.cache/huggingface` -> `/root/.cache/huggingface`)

## Output Artifact

Summary artifact:

- `artifacts/benchmarks/kv-cache-optimization-<date>.json`

Important fields:

- `model`
- `sweep_config`
- `results` (one row per parameter combination)
- `recommended_profile`
- `selection_rule`
- `generated_at_utc`

Per-result fields:

- `max_model_len`
- `gpu_memory_utilization`
- `pass`
- `serial_summary`
- `concurrent_summary`
- `error` (for failed runs)

## Deterministic Winner Selection

`recommended_profile` is chosen by:

1. Highest `concurrent_summary.throughput_req_per_s` among PASS rows
2. Tie-break: lower `gpu_memory_utilization`
3. Tie-break: lower `max_model_len`

## PASS Signatures

Expected console markers:

- `[kv-cache-opt] PASS combo ...` (at least one)
- `[kv-cache-opt] winner max_model_len=... gpu_memory_utilization=...`
- `[kv-cache-opt] PASS: KV cache optimization benchmark completed`

## Tunable Parameters

You can override defaults:

- `VLLM_MODEL`
- `MAX_MODEL_LEN_LIST` (CSV)
- `GPU_MEMORY_UTILIZATION_LIST` (CSV)
- `BASELINE_REQUESTS`
- `CONCURRENT_REQUESTS`
- `CONCURRENCY`
- `MAX_TOKENS`
- `RESULT_JSON_PATH`
- `RUN_ROOT`
- `READY_TIMEOUT_SECS`
- `REQUEST_TIMEOUT_SECS`
- `ENABLE_HF_CACHE_MOUNT`
- `HF_CACHE_HOST_DIR`

Example:

```bash
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
MAX_MODEL_LEN_LIST=4096,8192 \
GPU_MEMORY_UTILIZATION_LIST=0.85,0.90 \
RESULT_JSON_PATH=artifacts/benchmarks/kv-cache-optimization-2026-04-02.json \
./scripts/benchmark_kv_cache_optimization.sh
```

## Spark Troubleshooting

1. OOM or container crash
- Reduce sweep space first (`MAX_MODEL_LEN_LIST` or utilization list).
- Inspect per-run logs under `RUN_ROOT` (printed by script).

2. Timeout while waiting for API readiness
- Increase `READY_TIMEOUT_SECS` via environment passed through to the smoke script.

3. Port conflict
- Existing smoke flow already auto-switches host port when `8000` is busy.

4. Model download/auth problems
- Export token when needed:
  - `export HF_TOKEN=hf_xxx`

5. No PASS rows in summary
- Treat as non-closure condition; do not close roadmap item until at least one passing profile exists.
