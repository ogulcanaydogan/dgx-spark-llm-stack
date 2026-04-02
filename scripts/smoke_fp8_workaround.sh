#!/usr/bin/env bash
# Validate FP8 failure + BF16 workaround for TransformerEngine on DGX Spark.

set -euo pipefail

NGC_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/pytorch:25.11-py3}"
LOG_FILE="${LOG_FILE:-/tmp/fp8-workaround-smoke.log}"

log() {
  echo "[fp8-smoke] $*"
}

die() {
  echo "[fp8-smoke] ERROR: $*" >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || die "docker not found"

log "Running FP8 fail + BF16 pass checks in ${NGC_IMAGE}"
rm -f "${LOG_FILE}"

docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${NGC_IMAGE}" \
  bash -lc 'python - <<'"'"'PY'"'"'
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling

print("cuda", torch.cuda.is_available())

model = te.Linear(256, 256, params_dtype=torch.bfloat16).cuda()
x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)

recipe = MXFP8BlockScaling()
try:
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        _ = model(x)
    torch.cuda.synchronize()
    print("mxfp8_result unexpected_success")
except Exception as exc:  # expected on sm_121 / Blackwell
    print("mxfp8_result fail")
    print("mxfp8_error_type", type(exc).__name__)
    print("mxfp8_error", str(exc))

try:
    y = model(x)
    torch.cuda.synchronize()
    print("bf16_workaround ok", tuple(y.shape), str(y.dtype))
except Exception as exc:
    print("bf16_workaround fail")
    print("bf16_error_type", type(exc).__name__)
    print("bf16_error", str(exc))
PY' | tee "${LOG_FILE}"

grep -q "^mxfp8_result fail$" "${LOG_FILE}" || die "expected FP8 failure marker not found"
grep -E -q "MXFP8.*not supported|not supported on 12\\.0\\+ architectures|not supported on compute capability" "${LOG_FILE}" \
  || die "expected MXFP8 unsupported message not found"
grep -q "^bf16_workaround ok" "${LOG_FILE}" || die "expected BF16 workaround success marker not found"

log "PASS: FP8 workaround validated on DGX Spark"
