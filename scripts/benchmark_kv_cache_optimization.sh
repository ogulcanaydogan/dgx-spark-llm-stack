#!/usr/bin/env bash
# Sweep vLLM KV-cache-related parameters on DGX Spark and produce a single summary artifact.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BENCH_DATE="${BENCH_DATE:-$(date +%F)}"
RESULT_JSON_PATH="${RESULT_JSON_PATH:-${REPO_ROOT}/artifacts/benchmarks/kv-cache-optimization-${BENCH_DATE}.json}"

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_MODEL_LEN_LIST="${MAX_MODEL_LEN_LIST:-4096,8192,16384}"
GPU_MEMORY_UTILIZATION_LIST="${GPU_MEMORY_UTILIZATION_LIST:-0.85,0.90,0.95}"

BASELINE_REQUESTS="${BASELINE_REQUESTS:-4}"
CONCURRENT_REQUESTS="${CONCURRENT_REQUESTS:-16}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-48}"
READY_TIMEOUT_SECS="${READY_TIMEOUT_SECS:-1800}"
REQUEST_TIMEOUT_SECS="${REQUEST_TIMEOUT_SECS:-120}"
ENABLE_HF_CACHE_MOUNT="${ENABLE_HF_CACHE_MOUNT:-1}"
HF_CACHE_HOST_DIR="${HF_CACHE_HOST_DIR:-$HOME/.cache/huggingface}"
HF_CACHE_CONTAINER_DIR="${HF_CACHE_CONTAINER_DIR:-/root/.cache/huggingface}"

SMOKE_SCRIPT="${SMOKE_SCRIPT:-${REPO_ROOT}/scripts/smoke_vllm_continuous_batching.sh}"
RUN_ROOT="${RUN_ROOT:-/tmp/kv_cache_opt_${BENCH_DATE}_$$}"
RESULTS_NDJSON="${RUN_ROOT}/results.ndjson"

log() {
  echo "[kv-cache-opt] $*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

require_bin() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 not found"
}

run_one_combo() {
  local max_model_len="$1"
  local gpu_util="$2"
  local combo_id="len${max_model_len}_util${gpu_util//./}"
  local combo_json="${RUN_ROOT}/${combo_id}.json"
  local combo_log="${RUN_ROOT}/${combo_id}.log"
  local exit_code=0

  log "Running combo max_model_len=${max_model_len}, gpu_memory_utilization=${gpu_util}"

  set +e
  (
    cd "${REPO_ROOT}"
    VLLM_MODEL="${VLLM_MODEL}" \
    VLLM_MAX_MODEL_LEN="${max_model_len}" \
    VLLM_GPU_MEMORY_UTILIZATION="${gpu_util}" \
    BASELINE_REQUESTS="${BASELINE_REQUESTS}" \
    CONCURRENT_REQUESTS="${CONCURRENT_REQUESTS}" \
    CONCURRENCY="${CONCURRENCY}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    READY_TIMEOUT_SECS="${READY_TIMEOUT_SECS}" \
    REQUEST_TIMEOUT_SECS="${REQUEST_TIMEOUT_SECS}" \
    ENABLE_HF_CACHE_MOUNT="${ENABLE_HF_CACHE_MOUNT}" \
    HF_CACHE_HOST_DIR="${HF_CACHE_HOST_DIR}" \
    HF_CACHE_CONTAINER_DIR="${HF_CACHE_CONTAINER_DIR}" \
    RESULT_JSON_PATH="${combo_json}" \
    "${SMOKE_SCRIPT}"
  ) >"${combo_log}" 2>&1
  exit_code=$?
  set -e

  python3 - "${max_model_len}" "${gpu_util}" "${exit_code}" "${combo_json}" "${combo_log}" <<'PY' >>"${RESULTS_NDJSON}"
import json
import os
import sys

max_model_len = int(sys.argv[1])
gpu_utilization = float(sys.argv[2])
exit_code = int(sys.argv[3])
combo_json = sys.argv[4]
combo_log = sys.argv[5]

entry = {
    "max_model_len": max_model_len,
    "gpu_memory_utilization": gpu_utilization,
    "pass": False,
    "exit_code": exit_code,
    "result_json_path": combo_json,
    "run_log_path": combo_log,
    "serial_summary": None,
    "concurrent_summary": None,
    "error": None,
}

def tail_text(path, max_lines=120):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    return "\n".join(lines[-max_lines:])

if exit_code == 0 and os.path.exists(combo_json):
    try:
        payload = json.load(open(combo_json, encoding="utf-8"))
        serial_summary = payload.get("serial_summary")
        concurrent_summary = payload.get("concurrent_summary")
        if not isinstance(serial_summary, dict) or not isinstance(concurrent_summary, dict):
            raise ValueError("missing serial/concurrent summary in combo JSON")
        entry["serial_summary"] = serial_summary
        entry["concurrent_summary"] = concurrent_summary
        entry["pass"] = True
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"json_parse_error: {type(exc).__name__}: {exc}"
else:
    snippet = tail_text(combo_log)
    marker = ""
    for line in reversed(snippet.splitlines()):
        if "ERROR:" in line or "failed" in line.lower() or "timeout" in line.lower():
            marker = line.strip()
            break
    entry["error"] = marker or f"smoke_exit_code={exit_code}"

print(json.dumps(entry, sort_keys=True))
PY

  if [[ "${exit_code}" -eq 0 ]]; then
    log "PASS combo max_model_len=${max_model_len}, gpu_memory_utilization=${gpu_util}"
  else
    log "FAIL combo max_model_len=${max_model_len}, gpu_memory_utilization=${gpu_util} (exit=${exit_code})"
  fi
}

main() {
  require_bin bash
  require_bin python3
  require_bin docker

  [[ -x "${SMOKE_SCRIPT}" ]] || fail "smoke script not executable: ${SMOKE_SCRIPT}"

  mkdir -p "${RUN_ROOT}"
  : >"${RESULTS_NDJSON}"

  IFS=',' read -r -a max_model_lens <<<"${MAX_MODEL_LEN_LIST}"
  IFS=',' read -r -a gpu_utils <<<"${GPU_MEMORY_UTILIZATION_LIST}"

  [[ "${#max_model_lens[@]}" -gt 0 ]] || fail "MAX_MODEL_LEN_LIST is empty"
  [[ "${#gpu_utils[@]}" -gt 0 ]] || fail "GPU_MEMORY_UTILIZATION_LIST is empty"

  for max_model_len in "${max_model_lens[@]}"; do
    for gpu_util in "${gpu_utils[@]}"; do
      run_one_combo "${max_model_len}" "${gpu_util}"
    done
  done

  mkdir -p "$(dirname "${RESULT_JSON_PATH}")"

  python3 - "${RESULTS_NDJSON}" "${RESULT_JSON_PATH}" "${VLLM_MODEL}" "${MAX_MODEL_LEN_LIST}" "${GPU_MEMORY_UTILIZATION_LIST}" "${BASELINE_REQUESTS}" "${CONCURRENT_REQUESTS}" "${CONCURRENCY}" "${MAX_TOKENS}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

results_ndjson = sys.argv[1]
result_json_path = sys.argv[2]
model = sys.argv[3]
max_model_len_list = sys.argv[4]
gpu_util_list = sys.argv[5]
baseline_requests = int(sys.argv[6])
concurrent_requests = int(sys.argv[7])
concurrency = int(sys.argv[8])
max_tokens = int(sys.argv[9])

rows = []
with open(results_ndjson, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

if not rows:
    raise SystemExit("no sweep rows were recorded")

pass_rows = []
for row in rows:
    if row.get("pass"):
        concurrent = row.get("concurrent_summary") or {}
        throughput = concurrent.get("throughput_req_per_s")
        if isinstance(throughput, (int, float)):
            pass_rows.append(row)

if not pass_rows:
    raise SystemExit("no passing rows with numeric concurrent throughput")

winner = sorted(
    pass_rows,
    key=lambda row: (
        -float(row["concurrent_summary"]["throughput_req_per_s"]),
        float(row["gpu_memory_utilization"]),
        int(row["max_model_len"]),
    ),
)[0]

payload = {
    "benchmark_type": "kv_cache_optimization",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": model,
    "sweep_config": {
        "max_model_len_list": [int(x) for x in max_model_len_list.split(",") if x.strip()],
        "gpu_memory_utilization_list": [float(x) for x in gpu_util_list.split(",") if x.strip()],
        "baseline_requests": baseline_requests,
        "concurrent_requests": concurrent_requests,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "ready_timeout_secs": int(os.environ.get("READY_TIMEOUT_SECS", "1800")),
        "request_timeout_secs": int(os.environ.get("REQUEST_TIMEOUT_SECS", "120")),
        "enable_hf_cache_mount": os.environ.get("ENABLE_HF_CACHE_MOUNT", "1"),
    },
    "selection_rule": (
        "Choose highest concurrent throughput among passing profiles; "
        "tie-break by lower gpu_memory_utilization, then lower max_model_len."
    ),
    "results": rows,
    "recommended_profile": {
        "max_model_len": winner["max_model_len"],
        "gpu_memory_utilization": winner["gpu_memory_utilization"],
        "concurrent_throughput_req_per_s": winner["concurrent_summary"]["throughput_req_per_s"],
        "serial_throughput_req_per_s": winner["serial_summary"]["throughput_req_per_s"],
        "result_json_path": winner["result_json_path"],
        "run_log_path": winner["run_log_path"],
    },
}

with open(result_json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)

print(json.dumps(payload["recommended_profile"], sort_keys=True))
PY

  local winner_line
  winner_line="$(python3 - "${RESULT_JSON_PATH}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
r = payload["recommended_profile"]
print(
    "winner max_model_len={m} gpu_memory_utilization={u} concurrent_throughput_req_per_s={t}".format(
        m=r["max_model_len"],
        u=r["gpu_memory_utilization"],
        t=r["concurrent_throughput_req_per_s"],
    )
)
PY
)"

  log "${winner_line}"
  log "Sweep logs/result cache: ${RUN_ROOT}"
  log "Summary JSON: ${RESULT_JSON_PATH}"
  log "PASS: KV cache optimization benchmark completed"
}

main "$@"
