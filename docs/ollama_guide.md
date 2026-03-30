# Ollama Integration Guide (DGX Spark)

This guide covers Ollama setup and model import on DGX Spark, including quantization verification and OpenAI-compatible API usage.

## Scope and Default Profile

- Scope: Phase 4 Ollama integration only.
- Default model profile in this guide: `qwen2.5:7b-instruct-q4_K_M` (`Q4_K_M` quantization).

## Prerequisites (Spark)

- ARM64 Linux host (DGX Spark).
- NVIDIA driver and runtime are working (`nvidia-smi` should succeed).
- Sufficient free memory and disk:
  - Recommended: at least `30 GB` free disk for model cache growth.
  - Recommended: at least `16 GB` free RAM for stable local serving.
- Internet access for model download.

Quick checks:

```bash
uname -m
nvidia-smi
df -h
free -h
```

## Install Ollama

### Option A (system-wide, requires sudo)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

### Option B (user-local, no sudo)

Use this path when `sudo` is unavailable.

```bash
mkdir -p ~/.local/ollama-install
cd ~/.local/ollama-install
curl -fsSL https://ollama.com/download/ollama-linux-arm64.tar.zst -o ollama-linux-arm64.tar.zst
tar -I zstd -xf ollama-linux-arm64.tar.zst
~/.local/ollama-install/bin/ollama --version
```

## Start Service and Check Health

Run Ollama in the foreground:

```bash
~/.local/ollama-install/bin/ollama serve
```

Or detached (recommended for SSH sessions):

```bash
nohup env OLLAMA_HOST=127.0.0.1:11434 \
  ~/.local/ollama-install/bin/ollama serve \
  >/tmp/ollama.log 2>&1 &
echo $! > /tmp/ollama.pid
```

Health/control check:

```bash
curl -sf http://127.0.0.1:11434/api/tags | jq '.models | length'
```

## Model Import (Default + Deterministic Fallback)

### Primary profile (default)

```bash
~/.local/ollama-install/bin/ollama pull qwen2.5:7b-instruct-q4_K_M
```

### Deterministic fallback if tag mismatch

Some versions may not support `ollama search` (for example `0.19.0` on Spark). Use deterministic fallback tags:

```bash
for tag in \
  qwen2.5:7b-instruct-q4_K_M \
  qwen2.5:7b-q4_K_M \
  qwen2.5:7b-instruct \
  qwen2.5:7b
do
  echo "Trying $tag"
  ~/.local/ollama-install/bin/ollama pull "$tag" && break
done
```

Check the imported model list:

```bash
curl -sf http://127.0.0.1:11434/api/tags | jq -r '.models[].name'
```

## Inference Usage

### CLI prompt

```bash
~/.local/ollama-install/bin/ollama run qwen2.5:7b-instruct-q4_K_M "Write one short sentence about DGX Spark."
```

### OpenAI-compatible API (`/v1/chat/completions`)

```bash
curl -sf http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b-instruct-q4_K_M",
    "messages": [{"role":"user","content":"Reply with exactly: oai-ok"}],
    "temperature": 0
  }' | jq -r '.choices[0].message.content'
```

Native Ollama API example (`/api/generate`):

```bash
curl -sf http://127.0.0.1:11434/api/generate \
  -d '{
    "model":"qwen2.5:7b-instruct-q4_K_M",
    "prompt":"Reply with exactly: http-ok",
    "stream": false
  }' | jq -r '.response'
```

## Quantization Selection and Verification

This guide uses `Q4_K_M` because it provides a practical memory/performance tradeoff for 7B on Spark.

Verify model metadata:

```bash
curl -sf http://127.0.0.1:11434/api/show \
  -d '{"model":"qwen2.5:7b-instruct-q4_K_M"}' \
  | jq '.details | {family, parameter_size, quantization_level, format}'
```

Expected key field:

- `quantization_level`: `Q4_K_M`

## Spark Command Evidence (2026-03-30)

Validated on Spark host with live commands:

```text
ollama version is 0.19.0
models_count=5
model present: qwen2.5:7b-instruct-q4_K_M
openai_compatible_response: oai-ok
native_api_response: http-ok
quantization_level: Q4_K_M
```

## Spark Troubleshooting

### Port conflict on `11434`

Symptoms:
- `ollama serve` fails to bind.

Checks/fix:

```bash
ss -lntp | grep 11434 || true
pkill -f "ollama serve" || true
nohup env OLLAMA_HOST=127.0.0.1:11434 ~/.local/ollama-install/bin/ollama serve >/tmp/ollama.log 2>&1 &
```

### Interrupted model download

Symptoms:
- pull fails mid-download due to network interruption.

Checks/fix:

```bash
~/.local/ollama-install/bin/ollama pull qwen2.5:7b-instruct-q4_K_M
~/.local/ollama-install/bin/ollama list
```

Re-run `pull`; Ollama resumes/reuses downloaded layers when possible.

### GPU not visible / unexpectedly high RAM usage

Symptoms:
- slow inference, high CPU/RAM usage, low GPU utilization.

Checks/fix:

```bash
nvidia-smi
docker info | grep -i runtime
```

- Confirm NVIDIA driver/runtime is healthy before starting Ollama workloads.
- Reduce concurrent model sessions.
- Use lighter quant/profile when memory pressure is high (for example `Q4_K_M` over higher-memory variants).
