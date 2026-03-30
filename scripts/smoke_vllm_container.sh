#!/usr/bin/env bash
# Build and smoke-test the DGX Spark vLLM container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VLLM_IMAGE_TAG_PREFIX="${VLLM_IMAGE_TAG_PREFIX:-dgx-spark-vllm}"
VLLM_REF="${VLLM_REF:-}"
VLLM_REF_CANDIDATES="${VLLM_REF_CANDIDATES:-v0.18.0,v0.17.1,v0.16.0}"
VLLM_RUNTIME_PROFILES="${VLLM_RUNTIME_PROFILES:-v1_flashattn,v1_auto,v0_xformers}"

NGC_BASE="${NGC_BASE:-nvcr.io/nvidia/pytorch:25.11-py3}"
XFORMERS_VERSION="${XFORMERS_VERSION:-0.0.31}"
XFORMERS_DISABLE_FLASH_ATTN="${XFORMERS_DISABLE_FLASH_ATTN:-1}"
XFORMERS_BUILD_JOBS="${XFORMERS_BUILD_JOBS:-2}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1}"

HOST_PORT="${VLLM_HOST_PORT:-8000}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_ENABLE_CUSTOM_OPS="${VLLM_ENABLE_CUSTOM_OPS:-0}"
SMOKE_TIMEOUT_SECS="${SMOKE_TIMEOUT_SECS:-900}"

log() {
  echo "[vLLM-smoke] $*" >&2
}

cleanup_container() {
  local name="$1"
  docker rm -f "$name" >/dev/null 2>&1 || true
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
        sys.exit(0)
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

split_csv() {
  local raw="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "$raw"
}

build_image() {
  local ref="$1"
  local tag_suffix
  tag_suffix="$(sanitize_tag "$ref")"
  local image_tag="${VLLM_IMAGE_TAG_PREFIX}:${tag_suffix}"

  log "Building image ${image_tag} (vLLM ${ref})..."
  docker build \
    --build-arg "NGC_BASE=${NGC_BASE}" \
    --build-arg "VLLM_REF=${ref}" \
    --build-arg "XFORMERS_VERSION=${XFORMERS_VERSION}" \
    --build-arg "XFORMERS_DISABLE_FLASH_ATTN=${XFORMERS_DISABLE_FLASH_ATTN}" \
    --build-arg "XFORMERS_BUILD_JOBS=${XFORMERS_BUILD_JOBS}" \
    --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
    --file "${REPO_ROOT}/docker/vllm/Dockerfile" \
    --tag "${image_tag}" \
    "${REPO_ROOT}"
}

start_container() {
  local image_tag="$1"
  local profile="$2"
  local ref="$3"
  local container_name="vllm-smoke-${ref//[^a-zA-Z0-9]/-}-${profile}-$(date +%s)"
  local run_host_port="$HOST_PORT"

  if [[ -z "${VLLM_HOST_PORT:-}" ]] && [[ "$(port_is_available "$run_host_port")" != "1" ]]; then
    run_host_port="$(find_free_port)"
    log "Port ${HOST_PORT} is busy, switching to host port ${run_host_port}"
  fi

  local -a docker_run_args=(
    --detach
    --name "$container_name"
    --gpus all
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --publish "${run_host_port}:8000"
    --env "VLLM_MODEL=${VLLM_MODEL}"
    --env "VLLM_DTYPE=${VLLM_DTYPE}"
    --env "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"
    --env "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
    --env "VLLM_ENABLE_CUSTOM_OPS=${VLLM_ENABLE_CUSTOM_OPS}"
  )

  if [[ -n "${HF_TOKEN:-}" ]]; then
    docker_run_args+=(--env "HF_TOKEN=${HF_TOKEN}" --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
  fi

  local cmd_prefix=""
  case "$profile" in
    v1_flashattn)
      docker_run_args+=(--env "VLLM_USE_V1=1" --env "VLLM_ATTENTION_BACKEND=FLASH_ATTN")
      ;;
    v1_auto)
      docker_run_args+=(--env "VLLM_USE_V1=1")
      cmd_prefix='unset VLLM_ATTENTION_BACKEND; '
      ;;
    v0_xformers)
      docker_run_args+=(--env "VLLM_USE_V1=0" --env "VLLM_ATTENTION_BACKEND=XFORMERS")
      ;;
    *)
      log "ERROR: unsupported runtime profile '${profile}'"
      return 2
      ;;
  esac

  local start_err
  start_err="$(mktemp)"
  local run_cmd="${cmd_prefix}python -m vllm.entrypoints.openai.api_server --model \"${VLLM_MODEL}\" --host 0.0.0.0 --port 8000 --dtype \"${VLLM_DTYPE}\" --max-model-len \"${VLLM_MAX_MODEL_LEN}\" --gpu-memory-utilization \"${VLLM_GPU_MEMORY_UTILIZATION}\""

  if ! docker run "${docker_run_args[@]}" "$image_tag" bash -lc "$run_cmd" >/dev/null 2>"$start_err"; then
    if [[ -z "${VLLM_HOST_PORT:-}" ]] && grep -q "port is already allocated" "$start_err"; then
      run_host_port="$(find_free_port)"
      log "Port race detected, retrying on host port ${run_host_port}"
      docker_run_args=("${docker_run_args[@]/${HOST_PORT}:8000/${run_host_port}:8000}")
      docker run "${docker_run_args[@]}" "$image_tag" bash -lc "$run_cmd" >/dev/null
    else
      log "ERROR: failed to start container for profile ${profile}"
      cat "$start_err" >&2
      rm -f "$start_err"
      return 1
    fi
  fi
  rm -f "$start_err"

  echo "${container_name},${run_host_port}"
}

run_smoke_for_profile() {
  local image_tag="$1"
  local ref="$2"
  local profile="$3"

  local start_info
  start_info="$(start_container "$image_tag" "$profile" "$ref")" || return 1

  local container_name="${start_info%,*}"
  local run_host_port="${start_info##*,}"
  local base_url="http://127.0.0.1:${run_host_port}"
  local deadline=$((SECONDS + SMOKE_TIMEOUT_SECS))

  log "Trying ref=${ref} profile=${profile} on ${base_url}"

  until curl -fsS "${base_url}/health" >/dev/null 2>&1; do
    if [[ "$(docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null || echo false)" != "true" ]]; then
      log "ERROR: container exited before /health became ready (ref=${ref}, profile=${profile})"
      docker logs --tail 220 "$container_name" || true
      cleanup_container "$container_name"
      return 1
    fi
    if (( SECONDS >= deadline )); then
      log "ERROR: Timed out waiting for /health (${SMOKE_TIMEOUT_SECS}s) (ref=${ref}, profile=${profile})"
      docker logs --tail 220 "$container_name" || true
      cleanup_container "$container_name"
      return 1
    fi
    sleep 5
  done
  log "/health OK (ref=${ref}, profile=${profile})"

  local models_json
  models_json="$(mktemp)"
  if ! curl -fsS "${base_url}/v1/models" >"$models_json"; then
    log "ERROR: /v1/models request failed (ref=${ref}, profile=${profile})"
    docker logs --tail 220 "$container_name" || true
    rm -f "$models_json"
    cleanup_container "$container_name"
    return 1
  fi

  if ! python3 - "$models_json" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding='utf-8').read())
data = payload.get('data')
if not isinstance(data, list) or len(data) == 0:
    raise SystemExit(1)
print('models_count', len(data))
PY
  then
    log "ERROR: /v1/models returned invalid or empty payload (ref=${ref}, profile=${profile})"
    docker logs --tail 220 "$container_name" || true
    rm -f "$models_json"
    cleanup_container "$container_name"
    return 1
  fi

  rm -f "$models_json"
  cleanup_container "$container_name"

  echo "WINNER_REF=${ref}"
  echo "WINNER_PROFILE=${profile}"
  echo "WINNER_IMAGE_TAG=${image_tag}"
  log "PASS: vLLM container smoke test completed"
  return 0
}

declare -a ref_candidates
if [[ -n "$VLLM_REF" ]]; then
  ref_candidates=("$VLLM_REF")
else
  split_csv "$VLLM_REF_CANDIDATES" ref_candidates
fi

if [[ ${#ref_candidates[@]} -eq 0 ]]; then
  log "ERROR: no VLLM ref candidates provided"
  exit 1
fi

declare -a runtime_profiles
split_csv "$VLLM_RUNTIME_PROFILES" runtime_profiles
if [[ ${#runtime_profiles[@]} -eq 0 ]]; then
  log "ERROR: no runtime profiles provided"
  exit 1
fi

winner_ref=""
winner_profile=""
winner_image_tag=""

for ref in "${ref_candidates[@]}"; do
  if ! build_image "$ref"; then
    log "Build failed for ref=${ref}, trying next candidate"
    continue
  fi
  image_tag="${VLLM_IMAGE_TAG_PREFIX}:$(sanitize_tag "$ref")"
  for profile in "${runtime_profiles[@]}"; do
    if output="$(run_smoke_for_profile "$image_tag" "$ref" "$profile")"; then
      echo "$output"
      winner_ref="$ref"
      winner_profile="$profile"
      winner_image_tag="$image_tag"
      break 2
    fi
  done
done

if [[ -z "$winner_ref" ]]; then
  log "ERROR: no ref/profile combination passed smoke"
  exit 1
fi

log "Winner selected: ref=${winner_ref}, profile=${winner_profile}, image=${winner_image_tag}"
