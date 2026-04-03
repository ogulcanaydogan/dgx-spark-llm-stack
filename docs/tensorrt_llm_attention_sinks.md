# TensorRT-LLM Attention Sinks Validation (`sm_121`)

This guide defines deterministic fail+pass validation for the DGX Spark TensorRT-LLM attention-sinks roadmap item.

## Why This Exists

On legacy TensorRT-LLM tags, GPT-OSS style workloads on Blackwell (`sm_121`) can fail with:

`The attention sinks is only supported on SM90`

This smoke flow proves:

1. Legacy tag reproduces the assertion (expected fail).
2. Stable tag runs the same benchmark path without that assertion and writes a report JSON (expected pass).

## One-Command Smoke

From repository root:

```bash
./scripts/smoke_tensorrt_llm_attention_sinks.sh
```

Defaults:

- fail image: `nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc1`
- pass image: `nvcr.io/nvidia/tensorrt-llm/release:1.2.0`
- model: `openai/gpt-oss-20b`
- benchmark: `trtllm-bench ... throughput --backend pytorch --streaming --concurrency 8`

## Output Artifact

Artifact path:

- `artifacts/benchmarks/tensorrt-llm-attention-sinks-<date>.json`

Latest validated artifact:

- `artifacts/benchmarks/tensorrt-llm-attention-sinks-2026-04-03.json`

Required closure fields:

- `fail_case.has_sm90_assertion = true`
- `pass_case.has_sm90_assertion = false`
- `pass_case.report_json_exists = true`
- `closure_ready = true`

## Tunables

- `TRTLLM_FAIL_IMAGE`
- `TRTLLM_PASS_IMAGE`
- `MODEL_ID`
- `TP_SIZE`
- `CONCURRENCY`
- `ENABLE_HF_CACHE_MOUNT`
- `HF_CACHE_HOST_DIR`
- `ARTIFACT_PATH`
- `RUN_ROOT`

Example:

```bash
TRTLLM_FAIL_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc1 \
TRTLLM_PASS_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.2.0 \
MODEL_ID=openai/gpt-oss-20b \
./scripts/smoke_tensorrt_llm_attention_sinks.sh
```

## Spark Troubleshooting

1. NGC pull/auth errors
- Ensure Docker can pull `nvcr.io/nvidia/tensorrt-llm/release:*`.
- If private model access is needed, export `HF_TOKEN`.

2. Timeout or long first run
- First run is slower due to image/model download and runtime compilation.
- Re-run with cache mount enabled (default on).

3. OOM / runtime errors
- Reduce `CONCURRENCY`.
- Verify GPU state with `nvidia-smi`.

4. Closure does not pass
- Check `fail_case.log` and `pass_case.log` paths in artifact.
- Do not close roadmap item unless `closure_ready` is `true`.
