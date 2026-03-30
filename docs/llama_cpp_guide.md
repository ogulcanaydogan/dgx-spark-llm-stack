# llama.cpp Build Guide (DGX Spark, sm_121)

This guide covers building `llama.cpp` from source on DGX Spark (ARM64 + CUDA 13.0) with `sm_121`-targeted settings, then validating inference with a GGUF model.

## Scope and Default Profile

- Scope: Phase 4 `llama.cpp` integration item only.
- Default model profile: `Qwen2.5-0.5B-Instruct-GGUF` with `q4_k_m` quantization.
- Build path used in this guide: `/tmp/llama-cpp-guide`.

## Spark Prerequisites

- DGX Spark host reachable via SSH.
- Internet access for source/model download.
- Free disk and memory available for build + model cache.
- NVIDIA runtime visible.

Checks:

```bash
uname -m
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
/usr/local/cuda-13.0/bin/nvcc --version | tail -n 1
df -h /
free -h
```

## Build llama.cpp from Source (ARM64 + CUDA + sm_121)

```bash
WORK=/tmp/llama-cpp-guide
rm -rf "$WORK"
mkdir -p "$WORK"
cd "$WORK"

git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

CUDACXX=/usr/local/cuda-13.0/bin/nvcc \
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j"$(nproc)"
```

Notes:

- On Spark, `nvcc` may not be in `PATH`; setting `CUDACXX` avoids `No CMAKE_CUDA_COMPILER could be found`.
- CMake may rewrite `121` to `121a` internally for this toolkit/device combination.

## Verify Build

```bash
/tmp/llama-cpp-guide/llama.cpp/build/bin/llama-cli --version | head -n 2
```

Expected:

- Version output is shown (for example `version: 1 (...)`).
- CUDA device is detected in startup logs.

## Download GGUF Model (Deterministic Retry)

```bash
MODELDIR=/tmp/llama-cpp-guide/models
MODEL="$MODELDIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"
mkdir -p "$MODELDIR"

for url in \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf" \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"
do
  echo "Trying $url"
  if curl -fL --retry 3 --connect-timeout 20 "$url" -o "$MODEL"; then
    echo "Downloaded from $url"
    break
  fi
done

test -s "$MODEL"
ls -lh "$MODEL"
```

## Inference Validation (Non-Interactive, HTTP)

`llama-cli` can be interactive by default; for deterministic command output, run `llama-server` and call `/completion`.

```bash
WORK=/tmp/llama-cpp-guide
BIN="$WORK/llama.cpp/build/bin/llama-server"
MODEL="$WORK/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
PORT=18080

nohup "$BIN" \
  -m "$MODEL" \
  -ngl 999 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --ctx-size 2048 \
  --parallel 1 \
  --batch-size 512 \
  >/tmp/llama-cpp-guide/server.log 2>&1 &
echo $! >/tmp/llama-cpp-guide/server.pid

for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "health_ok_attempt=$i"
    break
  fi
  sleep 2
done

curl -sf "http://127.0.0.1:$PORT/completion" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Reply with exactly: llama-http-ok","n_predict":16,"temperature":0,"stop":["\n"]}' \
  | jq -r '.content'

kill "$(cat /tmp/llama-cpp-guide/server.pid)" || true
```

Acceptance for inference:

- `/health` responds successfully.
- `/completion` returns non-empty content.

## Spark Command Evidence (2026-03-30)

Validated live on Spark:

```text
host: spark-5fc3
arch: aarch64
gpu: NVIDIA GB10, driver 580.142
nvcc: Build cuda_13.0.r13.0/compiler.36424714_0
llama.cpp commit: 08f2145
cmake config: GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=121, CMAKE_BUILD_TYPE=Release
llama-cli --version: success
model file: qwen2.5-0.5b-instruct-q4_k_m.gguf (469M)
server health: health_ok_attempt=2
completion response: ta
```

## Spark Troubleshooting

### `No CMAKE_CUDA_COMPILER could be found`

Cause:

- `nvcc` is not in `PATH`.

Fix:

```bash
CUDACXX=/usr/local/cuda-13.0/bin/nvcc cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_BUILD_TYPE=Release
```

### Build directory gets inconsistent

Cause:

- Partial configure/build from previous attempts.

Fix:

```bash
rm -rf /tmp/llama-cpp-guide/llama.cpp/build
cmake -S /tmp/llama-cpp-guide/llama.cpp -B /tmp/llama-cpp-guide/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/llama-cpp-guide/llama.cpp/build -j"$(nproc)"
```

### Model file/path issues

Symptoms:

- Startup fails with missing or invalid GGUF model.

Fix:

```bash
test -s /tmp/llama-cpp-guide/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
file /tmp/llama-cpp-guide/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Re-download the model with the deterministic fallback loop if needed.
