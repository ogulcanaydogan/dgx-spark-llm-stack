#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

shopt -s nullglob
launch_files=(docs/day*_launch_*.md)
shopt -u nullglob

if [ ${#launch_files[@]} -eq 0 ]; then
  echo "No launch runbook files found under docs/day*_launch_*.md"
  exit 1
fi

IFS=$'\n' launch_files=($(printf "%s\n" "${launch_files[@]}" | sort -V))
unset IFS

pending_total=0

echo "Launch URL Evidence Check"
echo "Repository: $ROOT_DIR"
echo

for file in "${launch_files[@]}"; do
  pending_lines="$(rg -n 'URL.*`PENDING' "$file" || true)"
  if [ -z "$pending_lines" ]; then
    echo "[OK] $file"
  else
    echo "[PENDING] $file"
    echo "$pending_lines"
    count="$(printf "%s\n" "$pending_lines" | wc -l | tr -d ' ')"
    pending_total=$((pending_total + count))
  fi
  echo
done

echo "Total pending URL entries: $pending_total"

if [ "$pending_total" -gt 0 ]; then
  exit 1
fi

echo "All launch URL evidence entries are filled."
