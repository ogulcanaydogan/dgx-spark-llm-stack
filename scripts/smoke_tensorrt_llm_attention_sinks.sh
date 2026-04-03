#!/usr/bin/env bash
# Validate TensorRT-LLM attention-sinks behavior on DGX Spark:
# - legacy image reproduces SM90-only assertion
# - stable image runs benchmark and writes report JSON

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BENCH_DATE="${BENCH_DATE:-$(date +%F)}"

TRTLLM_FAIL_IMAGE="${TRTLLM_FAIL_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc1}"
TRTLLM_PASS_IMAGE="${TRTLLM_PASS_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.0}"
MODEL_ID="${MODEL_ID:-openai/gpt-oss-20b}"
TP_SIZE="${TP_SIZE:-1}"
CONCURRENCY="${CONCURRENCY:-8}"
ENABLE_HF_CACHE_MOUNT="${ENABLE_HF_CACHE_MOUNT:-1}"
HF_CACHE_HOST_DIR="${HF_CACHE_HOST_DIR:-$HOME/.cache/huggingface}"
HF_CACHE_CONTAINER_DIR="${HF_CACHE_CONTAINER_DIR:-/root/.cache/huggingface}"

RUN_ROOT="${RUN_ROOT:-/tmp/trtllm-attention-sinks-${BENCH_DATE}-$$}"
DATASET_PATH="${RUN_ROOT}/dataset.jsonl"
ARTIFACT_PATH="${ARTIFACT_PATH:-${REPO_ROOT}/artifacts/benchmarks/tensorrt-llm-attention-sinks-${BENCH_DATE}.json}"

FAIL_LOG="${RUN_ROOT}/fail-case.log"
PASS_LOG="${RUN_ROOT}/pass-case.log"
FAIL_REPORT="${RUN_ROOT}/fail-report.json"
PASS_REPORT="${RUN_ROOT}/pass-report.json"

log() {
  echo "[trtllm-sinks] $*" >&2
}

die() {
  echo "[trtllm-sinks] ERROR: $*" >&2
  exit 1
}

require_bin() {
  command -v "$1" >/dev/null 2>&1 || die "$1 not found"
}

run_case() {
  local case_name="$1"
  local image="$2"
  local report_path="$3"
  local log_path="$4"
  local exit_code
  local -a volume_args=()
  local -a env_args=()

  if [[ "${ENABLE_HF_CACHE_MOUNT}" == "1" ]]; then
    mkdir -p "${HF_CACHE_HOST_DIR}"
    volume_args+=(--volume "${HF_CACHE_HOST_DIR}:${HF_CACHE_CONTAINER_DIR}")
  fi

  if [[ -n "${HF_TOKEN:-}" ]]; then
    env_args+=(--env "HF_TOKEN=${HF_TOKEN}")
  fi

  log "Pulling image for ${case_name}: ${image}"
  docker pull "${image}" >/dev/null

  log "Running ${case_name} benchmark"
  set +e
  docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --volume "${RUN_ROOT}:/workspace" \
    "${volume_args[@]}" \
    "${env_args[@]}" \
    "${image}" \
    bash -lc "trtllm-bench --model \"${MODEL_ID}\" throughput --dataset /workspace/dataset.jsonl --tp ${TP_SIZE} --backend pytorch --streaming --concurrency ${CONCURRENCY} --report_json /workspace/$(basename "${report_path}")" \
    >"${log_path}" 2>&1
  exit_code=$?
  set -e

  printf "%s" "${exit_code}"
}

main() {
  require_bin docker
  require_bin nvidia-smi
  require_bin python3

  nvidia-smi >/dev/null 2>&1 || die "nvidia-smi failed; GPU runtime unavailable"

  mkdir -p "${RUN_ROOT}"
  mkdir -p "$(dirname "${ARTIFACT_PATH}")"
  rm -f "${FAIL_LOG}" "${PASS_LOG}" "${FAIL_REPORT}" "${PASS_REPORT}" "${DATASET_PATH}"

  cat >"${DATASET_PATH}" <<'EOF'
{"task_id":1,"prompt":"Write one short sentence about DGX Spark reliability.","output_tokens":48}
{"task_id":2,"prompt":"Explain Blackwell sm_121 in one concise sentence.","output_tokens":48}
{"task_id":3,"prompt":"Give one sentence on TensorRT-LLM profiling.","output_tokens":48}
{"task_id":4,"prompt":"Summarize why deterministic smoke tests matter.","output_tokens":48}
EOF

  local fail_exit_code pass_exit_code
  fail_exit_code="$(run_case "fail_case" "${TRTLLM_FAIL_IMAGE}" "${FAIL_REPORT}" "${FAIL_LOG}")"
  pass_exit_code="$(run_case "pass_case" "${TRTLLM_PASS_IMAGE}" "${PASS_REPORT}" "${PASS_LOG}")"

  python3 - "${FAIL_LOG}" "${PASS_LOG}" "${FAIL_REPORT}" "${PASS_REPORT}" "${ARTIFACT_PATH}" "${TRTLLM_FAIL_IMAGE}" "${TRTLLM_PASS_IMAGE}" "${MODEL_ID}" "${fail_exit_code}" "${pass_exit_code}" "${CONCURRENCY}" "${TP_SIZE}" <<'PY'
import json
import pathlib
import re
import sys
from datetime import datetime, timezone

fail_log = pathlib.Path(sys.argv[1])
pass_log = pathlib.Path(sys.argv[2])
fail_report = pathlib.Path(sys.argv[3])
pass_report = pathlib.Path(sys.argv[4])
artifact_path = pathlib.Path(sys.argv[5])
fail_image = sys.argv[6]
pass_image = sys.argv[7]
model_id = sys.argv[8]
fail_exit_raw = sys.argv[9]
pass_exit_raw = sys.argv[10]
concurrency = int(sys.argv[11])
tp_size = int(sys.argv[12])

sm90_pattern = re.compile(r"attention sinks is only supported on SM90", re.IGNORECASE)

def read_text(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")

def last_error_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        l = line.strip()
        if not l:
            continue
        if "ERROR" in l or "Assertion failed" in l or "RuntimeError" in l:
            return l[:400]
    return ""

def parse_exit_code(raw: str) -> int:
    try:
        return int(raw.strip())
    except Exception:
        matches = re.findall(r"-?\d+", raw)
        if not matches:
            return -1
        try:
            return int(matches[-1])
        except Exception:
            return -1

fail_exit = parse_exit_code(fail_exit_raw)
pass_exit = parse_exit_code(pass_exit_raw)

fail_text = read_text(fail_log)
pass_text = read_text(pass_log)

fail_has_sm90 = bool(sm90_pattern.search(fail_text))
pass_has_sm90 = bool(sm90_pattern.search(pass_text))
pass_report_exists = pass_report.exists() and pass_report.stat().st_size > 0
fail_report_exists = fail_report.exists() and fail_report.stat().st_size > 0

closure_ready = (
    fail_has_sm90
    and (not pass_has_sm90)
    and pass_report_exists
)

payload = {
    "benchmark_type": "tensorrt_llm_attention_sinks",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "model_id": model_id,
    "command_config": {
        "tp_size": tp_size,
        "concurrency": concurrency,
        "backend": "pytorch",
        "streaming": True,
    },
    "fail_case": {
        "image": fail_image,
        "exit_code": fail_exit,
        "exit_code_raw": fail_exit_raw,
        "log_path": str(fail_log),
        "report_json_path": str(fail_report),
        "report_json_exists": fail_report_exists,
        "has_sm90_assertion": fail_has_sm90,
        "error_excerpt": last_error_line(fail_text),
    },
    "pass_case": {
        "image": pass_image,
        "exit_code": pass_exit,
        "exit_code_raw": pass_exit_raw,
        "log_path": str(pass_log),
        "report_json_path": str(pass_report),
        "report_json_exists": pass_report_exists,
        "has_sm90_assertion": pass_has_sm90,
        "error_excerpt": last_error_line(pass_text),
    },
    "closure_ready": closure_ready,
}

artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

  if ! python3 - "${ARTIFACT_PATH}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
fail_case = payload["fail_case"]
pass_case = payload["pass_case"]
assert fail_case["has_sm90_assertion"] is True
assert pass_case["has_sm90_assertion"] is False
assert pass_case["report_json_exists"] is True
assert payload["closure_ready"] is True
PY
  then
    log "Fail log tail:"
    tail -n 80 "${FAIL_LOG}" || true
    log "Pass log tail:"
    tail -n 80 "${PASS_LOG}" || true
    die "closure criteria not met"
  fi

  log "Fail case has SM90 assertion: true"
  log "Pass case has SM90 assertion: false"
  log "Pass report exists: true"
  log "Summary JSON: ${ARTIFACT_PATH}"
  log "PASS: TensorRT-LLM attention sinks smoke completed"
}

main "$@"
