# Docker Compose Stack: vLLM + OpenAI API (DGX Spark)

This guide runs the existing DGX Spark vLLM image flow via Docker Compose and validates OpenAI-compatible endpoints.

## Scope and Defaults

- Compose file: `docker/compose/vllm.yml`
- Default vLLM ref: `v0.18.0`
- Default API port: `8000`
- Default model: `Qwen/Qwen2.5-0.5B-Instruct`
- Base image pin: `nvcr.io/nvidia/pytorch:25.11-py3`

## Prerequisites

- Docker + Compose plugin available (`docker compose version`)
- NVIDIA runtime healthy (`nvidia-smi`, `docker info`)
- Hugging Face access for model download (public model works without token)

Checks:

```bash
docker compose version
docker info
nvidia-smi
```

## Bring Up the Stack

From repo root:

```bash
mkdir -p "$HOME/.cache/huggingface"

docker compose -f docker/compose/vllm.yml config
docker compose -f docker/compose/vllm.yml up -d --build
```

Check service status and logs:

```bash
docker compose -f docker/compose/vllm.yml ps
docker compose -f docker/compose/vllm.yml logs -f vllm
```

## Smoke Validation

### Health endpoint

```bash
curl -sf http://127.0.0.1:8000/health
```

### OpenAI models endpoint

```bash
curl -sf http://127.0.0.1:8000/v1/models | jq '.data | length'
```

Acceptance:

- `/health` succeeds.
- `/v1/models` returns valid JSON and `data` length is greater than `0`.

## Optional: HF Token

For gated/private models:

```bash
export HF_TOKEN=hf_xxx
docker compose -f docker/compose/vllm.yml up -d --build
```

`HF_TOKEN` is forwarded as both `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`.

## Shutdown and Cleanup

```bash
docker compose -f docker/compose/vllm.yml down
docker compose -f docker/compose/vllm.yml ps
```

After `down`, Compose-managed service containers should be removed.

## Reproducibility Notes

- Keep pinned versions stable (`VLLM_REF`, `NGC_BASE`).
- Keep runtime flags stable (`--ipc=host`, `ulimits`, GPU access).
- Reuse host HF cache mount (`$HOME/.cache/huggingface`) to avoid redownload.

## Spark Command Evidence (2026-03-30)

Validated live on Spark host:

```text
docker compose config: success
docker compose up -d --build: success
/health: success
/v1/models: valid JSON with non-empty data
docker compose down: success
```

## Troubleshooting (Spark)

### Port 8000 already in use

```bash
ss -lntp | grep ':8000' || true
VLLM_HOST_PORT=18000 docker compose -f docker/compose/vllm.yml up -d --build
```

### GPU not visible in container

```bash
docker info | grep -i runtime
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 nvidia-smi
```

### Model download/auth failures

```bash
export HF_TOKEN=hf_xxx
docker compose -f docker/compose/vllm.yml up -d --build
docker compose -f docker/compose/vllm.yml logs --tail 200 vllm
```

### Container exits before ready

```bash
docker compose -f docker/compose/vllm.yml ps
docker compose -f docker/compose/vllm.yml logs --tail 300 vllm
```
