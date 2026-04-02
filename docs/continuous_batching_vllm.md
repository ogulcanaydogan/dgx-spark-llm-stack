# Continuous Batching with vLLM (DGX Spark)

This guide validates continuous batching behavior on DGX Spark using the OpenAI-compatible vLLM server.

## Why This Matters

Continuous batching lets vLLM schedule and interleave incoming requests, improving throughput under concurrent traffic.
This smoke is an operational check, not a full performance benchmark study.

## One-Command Smoke

From repo root:

```bash
./scripts/smoke_vllm_continuous_batching.sh
```

The script will:

- build the pinned vLLM image (`v0.18.0`) from `docker/vllm/Dockerfile`
- start OpenAI API server with model `Qwen/Qwen2.5-0.5B-Instruct`
- validate `/health` and `/v1/models`
- run serial requests (baseline) and concurrent requests (continuous batching smoke)
- fail on any non-200 response or empty model output

## Tunable Parameters

You can override defaults via env vars:

| Variable | Default | Description |
|---|---|---|
| `BASELINE_REQUESTS` | `4` | Serial request count before concurrent run |
| `CONCURRENT_REQUESTS` | `16` | Total requests in concurrent run |
| `CONCURRENCY` | `8` | Parallel workers for concurrent run |
| `REQUEST_TIMEOUT_SECS` | `120` | Per-request timeout |
| `MAX_TOKENS` | `48` | Tokens per generation |
| `VLLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Target model |
| `VLLM_REF` | `v0.18.0` | vLLM ref for image build |
| `VLLM_HOST_PORT` | `8000` | Host port (auto-fallback if busy) |
| `RESULT_JSON_PATH` | unset | Optional JSON summary output path |

Example:

```bash
CONCURRENCY=12 CONCURRENT_REQUESTS=24 RESULT_JSON_PATH=/tmp/cbatch.json \
  ./scripts/smoke_vllm_continuous_batching.sh
```

## Expected PASS Signatures

```text
[vllm-cbatch] /health OK
[vllm-cbatch] /v1/models OK (model=...)
serial_run completed total=... success=... errors=0 ...
concurrent_run completed total=... success=... errors=0 ...
non_empty_outputs serial=... concurrent=...
continuous_batching_result pass
[vllm-cbatch] PASS: continuous batching smoke completed
```

## Spark Troubleshooting

### Port conflict

If `8000` is occupied, the script auto-selects a free port. You can also pin one:

```bash
VLLM_HOST_PORT=18000 ./scripts/smoke_vllm_continuous_batching.sh
```

### Model download/auth issues

```bash
export HF_TOKEN=hf_xxx
./scripts/smoke_vllm_continuous_batching.sh
```

### Timeout or startup failure

Increase readiness timeout and inspect container logs (the script prints tail on failure):

```bash
READY_TIMEOUT_SECS=1200 ./scripts/smoke_vllm_continuous_batching.sh
```
