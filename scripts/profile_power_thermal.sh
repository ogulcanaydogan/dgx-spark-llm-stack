#!/usr/bin/env bash
# Profile DGX Spark power/thermal behavior while running a workload.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BENCH_DATE="${BENCH_DATE:-$(date +%F)}"
SAMPLE_INTERVAL_MS="${SAMPLE_INTERVAL_MS:-1000}"
RAW_CSV_PATH="${RAW_CSV_PATH:-/tmp/power_thermal_samples.csv}"
WORKLOAD_LOG_PATH="${WORKLOAD_LOG_PATH:-/tmp/power_thermal_workload.log}"
WORKLOAD_CMD="${WORKLOAD_CMD:-./scripts/smoke_vllm_continuous_batching.sh}"
RESULT_JSON_PATH="${RESULT_JSON_PATH:-${REPO_ROOT}/artifacts/benchmarks/power-thermal-continuous-batching-${BENCH_DATE}.json}"

SAMPLER_PID=""

log() {
  echo "[power-thermal] $*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

stop_sampler() {
  if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    kill "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
}

cleanup() {
  stop_sampler
}

run_preflight() {
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found"
  command -v python3 >/dev/null 2>&1 || fail "python3 not found"
  command -v bash >/dev/null 2>&1 || fail "bash not found"

  local probe
  probe="$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  [[ -n "${probe}" ]] || fail "nvidia-smi query returned no data"
  log "Preflight metrics probe: ${probe}"
}

start_sampler() {
  rm -f "${RAW_CSV_PATH}"
  nvidia-smi \
    --query-gpu=timestamp,name,temperature.gpu,power.draw,utilization.gpu \
    --format=csv,noheader,nounits \
    --loop-ms "${SAMPLE_INTERVAL_MS}" >"${RAW_CSV_PATH}" 2>/dev/null &
  SAMPLER_PID="$!"
  sleep 1
  kill -0 "${SAMPLER_PID}" 2>/dev/null || fail "failed to start nvidia-smi sampler"
  log "Sampler started (pid=${SAMPLER_PID}, interval_ms=${SAMPLE_INTERVAL_MS})"
}

run_workload() {
  rm -f "${WORKLOAD_LOG_PATH}"
  local workload_exit
  local start_ts end_ts

  start_ts="$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"

  set +e
  bash -lc "cd \"${REPO_ROOT}\" && ${WORKLOAD_CMD}" >"${WORKLOAD_LOG_PATH}" 2>&1
  workload_exit=$?
  set -e

  end_ts="$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"

  echo "${workload_exit}|${start_ts}|${end_ts}"
}

write_summary_json() {
  local workload_exit="$1"
  local start_ts="$2"
  local end_ts="$3"

  mkdir -p "$(dirname "${RESULT_JSON_PATH}")"

  python3 - "${RAW_CSV_PATH}" "${RESULT_JSON_PATH}" "${WORKLOAD_CMD}" "${WORKLOAD_LOG_PATH}" "${workload_exit}" "${SAMPLE_INTERVAL_MS}" "${start_ts}" "${end_ts}" <<'PY'
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone

raw_csv = sys.argv[1]
result_json = sys.argv[2]
workload_cmd = sys.argv[3]
workload_log = sys.argv[4]
workload_exit = int(sys.argv[5])
sample_interval_ms = int(sys.argv[6])
start_ts = float(sys.argv[7])
end_ts = float(sys.argv[8])


def parse_float(raw):
    value = (raw or "").strip()
    if value in {"", "[N/A]", "N/A", "Not Supported", "Unknown Error"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


rows = []
with open(raw_csv, encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 5:
            continue
        rows.append(
            {
                "timestamp": row[0].strip(),
                "gpu_name": row[1].strip(),
                "temp_c": parse_float(row[2]),
                "power_w": parse_float(row[3]),
                "gpu_util_pct": parse_float(row[4]),
            }
        )

sample_count = len(rows)
if sample_count == 0:
    raise SystemExit("sample_count is 0")

temps = [r["temp_c"] for r in rows if r["temp_c"] is not None]
powers = [r["power_w"] for r in rows if r["power_w"] is not None]
utils = [r["gpu_util_pct"] for r in rows if r["gpu_util_pct"] is not None]

if len(temps) == 0 and len(powers) == 0 and len(utils) == 0:
    raise SystemExit("all primary metrics are unavailable (temp/power/util)")

duration_s = max(0.0, end_ts - start_ts)

def avg_or_none(values):
    return statistics.fmean(values) if values else None

def max_or_none(values):
    return max(values) if values else None

payload = {
    "benchmark_type": "power_thermal_profile",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "workload_cmd": workload_cmd,
    "workload_log_path": workload_log,
    "workload_exit_code": workload_exit,
    "sample_interval_ms": sample_interval_ms,
    "sample_count": sample_count,
    "duration_s": round(duration_s, 3),
    "first_sample_timestamp": rows[0]["timestamp"],
    "last_sample_timestamp": rows[-1]["timestamp"],
    "gpu_name": rows[0]["gpu_name"],
    "metrics": {
        "avg_temp_c": round(avg_or_none(temps), 3) if temps else None,
        "max_temp_c": round(max_or_none(temps), 3) if temps else None,
        "avg_power_w": round(avg_or_none(powers), 3) if powers else None,
        "max_power_w": round(max_or_none(powers), 3) if powers else None,
        "avg_gpu_util_pct": round(avg_or_none(utils), 3) if utils else None,
        "max_gpu_util_pct": round(max_or_none(utils), 3) if utils else None,
    },
    "metric_samples": {
        "temp_c": len(temps),
        "power_w": len(powers),
        "gpu_util_pct": len(utils),
    },
}

with open(result_json, "w", encoding="utf-8") as out:
    json.dump(payload, out, indent=2, sort_keys=True)

print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

main() {
  trap cleanup EXIT

  run_preflight
  start_sampler
  IFS='|' read -r workload_exit start_ts end_ts < <(run_workload)
  stop_sampler
  SAMPLER_PID=""

  if [[ "${workload_exit}" -ne 0 ]]; then
    tail -n 220 "${WORKLOAD_LOG_PATH}" || true
    fail "workload exited with code ${workload_exit}"
  fi

  local summary_json
  if ! summary_json="$(write_summary_json "${workload_exit}" "${start_ts}" "${end_ts}")"; then
    tail -n 120 "${WORKLOAD_LOG_PATH}" || true
    fail "failed to parse sampler output or write JSON summary"
  fi

  local summary_line
  summary_line="$(python3 - "${RESULT_JSON_PATH}" <<'PY'
import json
import sys

p = json.load(open(sys.argv[1], encoding="utf-8"))
m = p["metrics"]
print(
    "samples={sample_count} duration_s={duration_s} workload_exit={workload_exit_code} "
    "avg_temp_c={avg_temp_c} max_temp_c={max_temp_c} "
    "avg_power_w={avg_power_w} max_power_w={max_power_w} "
    "avg_gpu_util_pct={avg_gpu_util_pct} max_gpu_util_pct={max_gpu_util_pct}".format(
        sample_count=p["sample_count"],
        duration_s=p["duration_s"],
        workload_exit_code=p["workload_exit_code"],
        avg_temp_c=m["avg_temp_c"],
        max_temp_c=m["max_temp_c"],
        avg_power_w=m["avg_power_w"],
        max_power_w=m["max_power_w"],
        avg_gpu_util_pct=m["avg_gpu_util_pct"],
        max_gpu_util_pct=m["max_gpu_util_pct"],
    )
)
PY
)"

  log "${summary_line}"
  log "Raw samples: ${RAW_CSV_PATH}"
  log "Workload log: ${WORKLOAD_LOG_PATH}"
  log "Summary JSON: ${RESULT_JSON_PATH}"
  log "PASS: power/thermal profiling completed"
}

main "$@"
