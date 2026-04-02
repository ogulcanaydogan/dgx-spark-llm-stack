#!/usr/bin/env bash
# Validate vLLM continuous batching on DGX Spark with serial + concurrent OpenAI API requests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VLLM_IMAGE_TAG_PREFIX="${VLLM_IMAGE_TAG_PREFIX:-dgx-spark-vllm-cbatch}"
NGC_BASE="${NGC_BASE:-nvcr.io/nvidia/pytorch:25.11-py3}"
VLLM_REF="${VLLM_REF:-v0.18.0}"
XFORMERS_VERSION="${XFORMERS_VERSION:-0.0.31}"
XFORMERS_DISABLE_FLASH_ATTN="${XFORMERS_DISABLE_FLASH_ATTN:-1}"
XFORMERS_BUILD_JOBS="${XFORMERS_BUILD_JOBS:-2}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1}"

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
VLLM_ENABLE_CUSTOM_OPS="${VLLM_ENABLE_CUSTOM_OPS:-0}"

VLLM_HOST_PORT="${VLLM_HOST_PORT:-8000}"
READY_TIMEOUT_SECS="${READY_TIMEOUT_SECS:-900}"
REQUEST_TIMEOUT_SECS="${REQUEST_TIMEOUT_SECS:-120}"
BASELINE_REQUESTS="${BASELINE_REQUESTS:-4}"
CONCURRENT_REQUESTS="${CONCURRENT_REQUESTS:-16}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-48}"
PROMPT_PREFIX="${PROMPT_PREFIX:-DGX Spark continuous batching smoke}"
RESULT_JSON_PATH="${RESULT_JSON_PATH:-}"

CONTAINER_NAME=""
RUN_HOST_PORT=""

log() {
  echo "[vllm-cbatch] $*" >&2
}

cleanup() {
  if [[ -n "${CONTAINER_NAME}" ]]; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
}

fail_with_logs() {
  log "ERROR: $*"
  if [[ -n "${CONTAINER_NAME}" ]]; then
    docker logs --tail 220 "${CONTAINER_NAME}" || true
  fi
  exit 1
}

port_is_available() {
  local port="$1"
  python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
for family, addr in ((socket.AF_INET6, ("::", port)), (socket.AF_INET, ("0.0.0.0", port))):
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(addr)
        sock.close()
        print("1")
        raise SystemExit(0)
    except OSError:
        continue
print("0")
PY
}

find_free_port() {
  python3 <<'PY'
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
}

sanitize_tag() {
  echo "$1" | tr '/:+@' '----' | tr -c 'A-Za-z0-9._-' '-'
}

build_image() {
  local image_tag="${VLLM_IMAGE_TAG_PREFIX}:$(sanitize_tag "${VLLM_REF}")"
  log "Building image ${image_tag} ..."
  docker build \
    --build-arg "NGC_BASE=${NGC_BASE}" \
    --build-arg "VLLM_REF=${VLLM_REF}" \
    --build-arg "XFORMERS_VERSION=${XFORMERS_VERSION}" \
    --build-arg "XFORMERS_DISABLE_FLASH_ATTN=${XFORMERS_DISABLE_FLASH_ATTN}" \
    --build-arg "XFORMERS_BUILD_JOBS=${XFORMERS_BUILD_JOBS}" \
    --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
    --file "${REPO_ROOT}/docker/vllm/Dockerfile" \
    --tag "${image_tag}" \
    "${REPO_ROOT}"
  echo "${image_tag}"
}

start_container() {
  local image_tag="$1"
  RUN_HOST_PORT="${VLLM_HOST_PORT}"
  if [[ "$(port_is_available "${RUN_HOST_PORT}")" != "1" ]]; then
    RUN_HOST_PORT="$(find_free_port)"
    log "Port ${VLLM_HOST_PORT} is busy, switching to ${RUN_HOST_PORT}"
  fi

  CONTAINER_NAME="vllm-cbatch-$(date +%s)"
  local run_cmd="python -m vllm.entrypoints.openai.api_server --model \"${VLLM_MODEL}\" --host 0.0.0.0 --port 8000 --dtype \"${VLLM_DTYPE}\" --max-model-len \"${VLLM_MAX_MODEL_LEN}\" --gpu-memory-utilization \"${VLLM_GPU_MEMORY_UTILIZATION}\""

  docker run --detach \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --publish "${RUN_HOST_PORT}:8000" \
    --env "VLLM_MODEL=${VLLM_MODEL}" \
    --env "VLLM_DTYPE=${VLLM_DTYPE}" \
    --env "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}" \
    --env "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}" \
    --env "VLLM_USE_V1=${VLLM_USE_V1}" \
    --env "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}" \
    --env "VLLM_ENABLE_CUSTOM_OPS=${VLLM_ENABLE_CUSTOM_OPS}" \
    "${image_tag}" \
    bash -lc "${run_cmd}" >/dev/null
}

wait_until_ready() {
  local base_url="http://127.0.0.1:${RUN_HOST_PORT}"
  local deadline=$((SECONDS + READY_TIMEOUT_SECS))

  until curl -fsS "${base_url}/health" >/dev/null 2>&1; do
    if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || echo false)" != "true" ]]; then
      fail_with_logs "container exited before /health became ready"
    fi
    if (( SECONDS >= deadline )); then
      fail_with_logs "timed out waiting for /health (${READY_TIMEOUT_SECS}s)"
    fi
    sleep 5
  done
  log "/health OK"
}

verify_models_endpoint() {
  local base_url="http://127.0.0.1:${RUN_HOST_PORT}"
  local models_json
  models_json="$(mktemp)"
  curl -fsS "${base_url}/v1/models" >"${models_json}" || fail_with_logs "/v1/models request failed"

  local model_id
  model_id="$(python3 - "${models_json}" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
data = payload.get("data")
if not isinstance(data, list) or len(data) == 0:
    raise SystemExit(1)
model = data[0].get("id")
if not model:
    raise SystemExit(1)
print(model)
PY
  )" || fail_with_logs "/v1/models returned invalid or empty payload"
  rm -f "${models_json}"

  log "/v1/models OK (model=${model_id})"
  echo "${model_id}"
}

run_load_test() {
  local model_id="$1"
  local base_url="http://127.0.0.1:${RUN_HOST_PORT}"
  local output_json
  output_json="$(mktemp)"

  set +e
  python3 - "${base_url}" "${model_id}" "${BASELINE_REQUESTS}" "${CONCURRENT_REQUESTS}" "${CONCURRENCY}" "${REQUEST_TIMEOUT_SECS}" "${MAX_TOKENS}" "${PROMPT_PREFIX}" "${output_json}" <<'PY'
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request

base_url, model_id = sys.argv[1], sys.argv[2]
baseline_requests = int(sys.argv[3])
concurrent_requests = int(sys.argv[4])
concurrency = int(sys.argv[5])
request_timeout = int(sys.argv[6])
max_tokens = int(sys.argv[7])
prompt_prefix = sys.argv[8]
output_json_path = sys.argv[9]


def extract_text(payload):
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    text = first.get("text")
    if isinstance(text, str):
        return text
    msg = first.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            return content
    return ""


def send_once(idx, endpoint):
    prompt = f"{prompt_prefix} #{idx}. Return one short sentence."
    if endpoint == "completions":
        payload = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    else:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }

    url = f"{base_url}/v1/{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=request_timeout) as resp:
            body = resp.read()
            status = resp.getcode()
        elapsed = time.perf_counter() - started
        parsed = json.loads(body.decode("utf-8"))
        text = extract_text(parsed).strip()
        ok = status == 200 and len(text) > 0
        return {"ok": ok, "status": status, "elapsed_s": elapsed, "error": "", "output_len": len(text)}
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        return {"ok": False, "status": 0, "elapsed_s": elapsed, "error": f"{type(exc).__name__}: {exc}", "output_len": 0}


def select_endpoint():
    for endpoint in ("completions", "chat/completions"):
        trial = send_once(0, endpoint)
        if trial["ok"]:
            return endpoint
    raise SystemExit("No working OpenAI endpoint found for load test")


def summarize(label, results, elapsed):
    total = len(results)
    success = sum(1 for r in results if r["ok"])
    errors = total - success
    non_empty = sum(1 for r in results if r["output_len"] > 0)
    avg_latency = sum(r["elapsed_s"] for r in results) / total if total else 0.0
    throughput = success / elapsed if elapsed > 0 else 0.0
    summary = {
        "label": label,
        "total": total,
        "success": success,
        "errors": errors,
        "non_empty_outputs": non_empty,
        "elapsed_s": elapsed,
        "avg_latency_s": avg_latency,
        "throughput_req_per_s": throughput,
    }
    return summary


endpoint = select_endpoint()
print(f"endpoint_mode={endpoint}")

serial_start = time.perf_counter()
serial_results = [send_once(i + 1, endpoint) for i in range(baseline_requests)]
serial_elapsed = time.perf_counter() - serial_start
serial_summary = summarize("serial", serial_results, serial_elapsed)
print(
    "serial_run completed "
    f"total={serial_summary['total']} success={serial_summary['success']} "
    f"errors={serial_summary['errors']} elapsed_s={serial_summary['elapsed_s']:.3f}"
)

conc_start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
    futures = [pool.submit(send_once, baseline_requests + i + 1, endpoint) for i in range(concurrent_requests)]
    concurrent_results = [f.result() for f in futures]
conc_elapsed = time.perf_counter() - conc_start
concurrent_summary = summarize("concurrent", concurrent_results, conc_elapsed)
print(
    "concurrent_run completed "
    f"total={concurrent_summary['total']} success={concurrent_summary['success']} "
    f"errors={concurrent_summary['errors']} elapsed_s={concurrent_summary['elapsed_s']:.3f} "
    f"workers={concurrency}"
)

print(
    "non_empty_outputs "
    f"serial={serial_summary['non_empty_outputs']} "
    f"concurrent={concurrent_summary['non_empty_outputs']}"
)

payload = {
    "base_url": base_url,
    "model_id": model_id,
    "endpoint_mode": endpoint,
    "serial_summary": serial_summary,
    "concurrent_summary": concurrent_summary,
}
with open(output_json_path, "w", encoding="utf-8") as fp:
    json.dump(payload, fp, indent=2, sort_keys=True)

if serial_summary["errors"] > 0 or concurrent_summary["errors"] > 0:
    raise SystemExit(2)
if serial_summary["non_empty_outputs"] != serial_summary["total"]:
    raise SystemExit(3)
if concurrent_summary["non_empty_outputs"] != concurrent_summary["total"]:
    raise SystemExit(4)

print("continuous_batching_result pass")
PY

  local status=$?
  set -e
  cat "${output_json}"
  if [[ -n "${RESULT_JSON_PATH}" ]]; then
    mkdir -p "$(dirname "${RESULT_JSON_PATH}")"
    cp "${output_json}" "${RESULT_JSON_PATH}"
    log "Saved load-test JSON to ${RESULT_JSON_PATH}"
  fi
  rm -f "${output_json}"

  if [[ ${status} -ne 0 ]]; then
    fail_with_logs "continuous batching load test failed"
  fi
}

main() {
  command -v docker >/dev/null 2>&1 || fail_with_logs "docker not found"
  command -v python3 >/dev/null 2>&1 || fail_with_logs "python3 not found"
  command -v curl >/dev/null 2>&1 || fail_with_logs "curl not found"

  trap cleanup EXIT

  local image_tag
  image_tag="$(build_image)"
  start_container "${image_tag}"
  wait_until_ready
  local model_id
  model_id="$(verify_models_endpoint)"
  run_load_test "${model_id}"

  log "PASS: continuous batching smoke completed"
}

main "$@"
