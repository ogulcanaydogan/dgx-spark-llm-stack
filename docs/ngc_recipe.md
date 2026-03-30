# NGC Container Recipe (DGX Spark LLM Workloads)

This guide provides a reproducible NGC-based container workflow for DGX Spark using `nvcr.io/nvidia/pytorch:25.11-py3`.

## Scope and Defaults

- Scope: Phase 4 NGC recipe item only.
- Base image pin: `nvcr.io/nvidia/pytorch:25.11-py3`.
- Default Hugging Face cache mount:
  - Host: `$HOME/.cache/huggingface`
  - Container: `/root/.cache/huggingface`

## Spark Prerequisites

- Docker is installed and usable by your user.
- NVIDIA runtime is available in Docker.
- GPU is visible on host (`nvidia-smi`).
- Internet access exists for `docker pull` and model/tokenizer downloads.

Checks:

```bash
docker --version
docker info
nvidia-smi
```

## Pull Pinned NGC Base

```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

Expected:

- Pull succeeds.
- Digest is shown.

## Profile 1: Base Smoke (GPU + Torch)

### GPU visibility inside container

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/pytorch:25.11-py3 \
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

### Python/Torch CUDA check inside container

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/pytorch:25.11-py3 \
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

## Profile 2: LLM Runtime Prep (HF cache mount + runtime flags)

Use Spark-friendly runtime flags for memory/shared IPC:

```bash
mkdir -p "$HOME/.cache/huggingface"

docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  nvcr.io/nvidia/pytorch:25.11-py3 \
  bash -lc 'python -m pip install --quiet "transformers>=4.48.2,<5" && python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained(\"Qwen/Qwen2.5-0.5B-Instruct\"); print(\"tokenizer\", t.__class__.__name__)"'
```

Expected:

- Command exits `0`.
- Non-empty tokenizer output (for example `tokenizer Qwen2TokenizerFast`).

## Reproducibility Notes

- Always pin image tags (`25.11-py3`) instead of floating tags.
- Keep runtime flags stable across runs:
  - `--gpus all`
  - `--ipc=host`
  - `--ulimit memlock=-1`
  - `--ulimit stack=67108864`
- Keep Hugging Face cache mount path stable to avoid repeated downloads.
- Record key runtime versions with each run:
  - `docker --version`
  - `nvidia-smi`
  - `python -c "import torch; print(torch.__version__)"`

## Spark Command Evidence (2026-03-30)

Validated on Spark:

```text
docker version: 29.1.3
docker pull: nvcr.io/nvidia/pytorch:25.11-py3 (digest sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21)
container nvidia-smi: NVIDIA GB10, 580.142
container torch check: torch 2.10.0a0+b558c986e8.nv25.11 cuda True
LLM prep with HF cache mount: tokenizer Qwen2TokenizerFast
```

## Troubleshooting (Spark)

### NGC pull/auth issues

Symptoms:

- `docker pull nvcr.io/nvidia/pytorch:25.11-py3` fails with auth/network errors.

Actions:

```bash
docker logout nvcr.io || true
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

If your environment requires credentials, authenticate to NGC first and retry.

### NVIDIA runtime not found

Symptoms:

- `--gpus all` fails or GPU devices are missing in container.

Actions:

```bash
docker info | grep -i runtime
nvidia-smi
```

Ensure Docker sees `nvidia` runtime and host GPU is healthy.

### GPU not visible in container

Symptoms:

- Host `nvidia-smi` works but container `nvidia-smi` fails.

Actions:

```bash
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 nvidia-smi
```

If this fails, fix NVIDIA Container Toolkit/runtime on host before LLM steps.

### Disk/cache pressure

Symptoms:

- Slow pulls/downloads, partial model artifacts, or write errors.

Actions:

```bash
df -h
du -sh "$HOME/.cache/huggingface" || true
```

Clean old cache entries if disk is low and re-run the LLM prep command.
