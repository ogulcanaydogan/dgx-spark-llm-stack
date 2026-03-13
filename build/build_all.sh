#!/usr/bin/env bash
# Build all components for DGX Spark LLM Stack
# Runs builds in dependency order: PyTorch → Triton → flash-attn → BitsAndBytes
#
# Total build time: ~5 hours on DGX Spark (20 cores)
# Output: dist/*.whl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/env.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[BUILD-ALL]${NC} $*"; }
warn()  { echo -e "${YELLOW}[BUILD-ALL]${NC} $*"; }

TOTAL_START=$(date +%s)

info "Starting full stack build at $(date)"
info "Output directory: ${BUILD_OUTPUT_DIR}"
echo ""

# ── 1. PyTorch (must be first — other builds depend on it) ────────────────────

info "═══ Step 1/4: Building PyTorch ═══"
bash "${SCRIPT_DIR}/build_pytorch.sh"
echo ""

# Install the wheel so subsequent builds can import torch
TORCH_WHEEL=$(find "${BUILD_OUTPUT_DIR}" -name "torch-*.whl" | head -1)
if [[ -n "${TORCH_WHEEL}" ]]; then
    info "Installing PyTorch wheel for subsequent builds..."
    pip install "${TORCH_WHEEL}" --force-reinstall
fi
echo ""

# ── 2. Triton ─────────────────────────────────────────────────────────────────

info "═══ Step 2/4: Building Triton ═══"
bash "${SCRIPT_DIR}/build_triton.sh" || warn "Triton build failed (non-fatal)"
echo ""

# ── 3. flash-attention ────────────────────────────────────────────────────────

info "═══ Step 3/4: Building flash-attention ═══"
bash "${SCRIPT_DIR}/build_flash_attn.sh" || warn "flash-attention build failed (expected — use SDPA)"
echo ""

# ── 4. BitsAndBytes ───────────────────────────────────────────────────────────

info "═══ Step 4/4: Building BitsAndBytes ═══"
bash "${SCRIPT_DIR}/build_bitsandbytes.sh"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_HOURS=$(( TOTAL_ELAPSED / 3600 ))
TOTAL_MINS=$(( (TOTAL_ELAPSED % 3600) / 60 ))

info "═══════════════════════════════════════════"
info "Build complete! Total time: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
info ""
info "Wheels in ${BUILD_OUTPUT_DIR}:"
ls -lh "${BUILD_OUTPUT_DIR}"/*.whl 2>/dev/null || warn "No wheel files found"
info ""
info "Install all: pip install ${BUILD_OUTPUT_DIR}/*.whl"
info "═══════════════════════════════════════════"
