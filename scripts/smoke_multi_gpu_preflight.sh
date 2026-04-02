#!/usr/bin/env bash
# Validate DGX Spark distributed preflight for future multi-GPU/multi-node runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BENCH_DATE="${BENCH_DATE:-$(date +%F)}"
RESULT_JSON_PATH="${RESULT_JSON_PATH:-${REPO_ROOT}/artifacts/benchmarks/multi-gpu-preflight-${BENCH_DATE}.json}"
TORCHRUN_LOG_PATH="${TORCHRUN_LOG_PATH:-/tmp/multi_gpu_preflight_torchrun.log}"
NGC_BASE="${NGC_BASE:-nvcr.io/nvidia/pytorch:25.11-py3}"

PRECHECK_JSON="$(mktemp /tmp/multi_gpu_precheck.XXXXXX.json)"

log() {
  echo "[multi-gpu-preflight] $*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

cleanup() {
  rm -f "${PRECHECK_JSON}"
}

run_host_preflight() {
  export PATH="$HOME/.local/bin:${PATH}"

  command -v python3 >/dev/null 2>&1 || fail "python3 not found"
  command -v torchrun >/dev/null 2>&1 || fail "torchrun not found"
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found"
  command -v docker >/dev/null 2>&1 || fail "docker not found"

  local gpu_line
  gpu_line="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
  [[ -n "${gpu_line}" ]] || fail "nvidia-smi returned no GPU rows"
}

run_runtime_precheck() {
  local container_output
  local precheck_json_line
  container_output="$(docker run --rm -i \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${NGC_BASE}" \
    python - <<'PY'
import json
import socket
from datetime import datetime, timezone

import torch
import torch.distributed as dist


def detect_nccl():
    if hasattr(dist, "is_nccl_available"):
        return bool(dist.is_nccl_available())
    try:
        import torch.distributed.distributed_c10d as c10d

        if hasattr(c10d, "is_nccl_available"):
            return bool(c10d.is_nccl_available())
    except Exception:  # noqa: BLE001
        pass
    backend = getattr(dist, "Backend", None)
    return bool(backend is not None and hasattr(backend, "NCCL"))


payload = {
    "hostname": socket.gethostname(),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()),
    "nccl_available": detect_nccl(),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}

if not payload["cuda_available"]:
    raise SystemExit("torch.cuda.is_available() is False")
if payload["cuda_device_count"] < 1:
    raise SystemExit("torch.cuda.device_count() < 1")
if not payload["nccl_available"]:
    raise SystemExit("NCCL is not available in runtime")

print("PRECHECK_JSON::" + json.dumps(payload, sort_keys=True))
PY
)"

  precheck_json_line="$(printf '%s\n' "${container_output}" | grep 'PRECHECK_JSON::' | tail -n 1 | sed 's/^PRECHECK_JSON:://')"
  [[ -n "${precheck_json_line}" ]] || fail "failed to capture precheck JSON from runtime"

  python3 - "${precheck_json_line}" "${PRECHECK_JSON}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
PY
}

run_torchrun_smoke() {
  rm -f "${TORCHRUN_LOG_PATH}"
  set +e
  docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${NGC_BASE}" \
    bash -lc "cat >/tmp/multi_gpu_torchrun.py <<'PY'
import os
import sys

import torch
import torch.distributed as dist

backend = 'nccl'
rank = int(os.environ.get('RANK', '0'))
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
world_size = int(os.environ.get('WORLD_SIZE', '1'))

try:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method='env://')
    tensor = torch.tensor([1.0], device=f'cuda:{local_rank}')
    dist.all_reduce(tensor)
    dist.barrier()
    if rank == 0:
        print(f'torchrun_nccl_ok rank={rank} local_rank={local_rank} world_size={world_size} all_reduce={tensor.item():.1f}')
except Exception as exc:  # noqa: BLE001
    print(f'torchrun_nccl_fail {type(exc).__name__}: {exc}', file=sys.stderr)
    raise
finally:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
PY
torchrun --standalone --nnodes=1 --nproc_per_node=1 /tmp/multi_gpu_torchrun.py" >"${TORCHRUN_LOG_PATH}" 2>&1
  local exit_code=$?
  set -e

  if [[ "${exit_code}" -ne 0 ]]; then
    tail -n 200 "${TORCHRUN_LOG_PATH}" || true
    fail "torchrun single-rank NCCL smoke failed (exit=${exit_code})"
  fi

  if ! grep -q "torchrun_nccl_ok" "${TORCHRUN_LOG_PATH}"; then
    tail -n 200 "${TORCHRUN_LOG_PATH}" || true
    fail "torchrun log does not contain success marker"
  fi

  log "torchrun single-rank NCCL smoke PASS"
}

write_result_json() {
  mkdir -p "$(dirname "${RESULT_JSON_PATH}")"

  python3 - "${PRECHECK_JSON}" "${RESULT_JSON_PATH}" <<'PY'
import json
import sys
from datetime import datetime, timezone

precheck = json.load(open(sys.argv[1], encoding="utf-8"))

payload = {
    "hostname": precheck["hostname"],
    "torch_version": precheck["torch_version"],
    "cuda_version": precheck["cuda_version"],
    "cuda_device_count": precheck["cuda_device_count"],
    "nccl_available": precheck["nccl_available"],
    "torchrun_single_rank_nccl_ok": True,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}

with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)

print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

main() {
  trap cleanup EXIT

  run_host_preflight
  run_runtime_precheck
  run_torchrun_smoke
  write_result_json >/dev/null

  log "Result JSON: ${RESULT_JSON_PATH}"
  log "torchrun log: ${TORCHRUN_LOG_PATH}"
  log "PASS: multi-gpu preflight completed"
}

main "$@"
