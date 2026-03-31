#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v jupyter >/dev/null 2>&1 && jupyter nbconvert --version >/dev/null 2>&1; then
  NBCONVERT_CMD=(jupyter nbconvert)
elif python3 -m nbconvert --version >/dev/null 2>&1; then
  NBCONVERT_CMD=(python3 -m nbconvert)
else
  echo "ERROR: nbconvert not found"
  echo "Install with: python3 -m pip install --user notebook nbconvert ipykernel"
  exit 1
fi

NOTEBOOKS=(
  "notebooks/01_inference.ipynb"
  "notebooks/02_finetuning_lora.ipynb"
  "notebooks/03_evaluation_perplexity.ipynb"
)

mkdir -p artifacts/notebooks artifacts/notebooks/logs

for notebook in "${NOTEBOOKS[@]}"; do
  base_name="$(basename "$notebook" .ipynb)"
  log_path="artifacts/notebooks/logs/${base_name}.log"

  echo "Running $notebook"
  if ! "${NBCONVERT_CMD[@]}" \
      --to notebook \
      --execute "$notebook" \
      --output "${base_name}.executed.ipynb" \
      --output-dir artifacts/notebooks \
      --ExecutePreprocessor.timeout=0 >"$log_path" 2>&1; then
    echo "ERROR: notebook execution failed: $notebook"
    tail -n 80 "$log_path" || true
    exit 1
  fi

  echo "PASS: $notebook"
done

python3 - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/notebooks")
checks = {
    "inference-smoke.json": ["tokens_per_sec", "gpu_memory_gb"],
    "training-smoke.json": ["samples_per_sec", "peak_memory_gb"],
    "eval-smoke.json": ["perplexity", "tokens_per_sec"],
}

for filename, required_fields in checks.items():
    path = root / filename
    if not path.exists():
        raise SystemExit(f"missing expected output: {path}")
    payload = json.loads(path.read_text())
    results = payload.get("results", [])
    if not results:
        raise SystemExit(f"empty results in: {path}")
    row = results[0]
    for field in required_fields:
        value = row.get(field)
        if value is None:
            raise SystemExit(f"missing field {field} in: {path}")
print("JSON validation passed")
PY

echo "PASS: notebook smoke completed"
