#!/usr/bin/env bash
# Build Triton from source for DGX Spark (GB10, sm_121, CUDA 13.0)
# Triton's ptxas invocation doesn't recognize sm_121a — this build attempts
# to work around that by building from main branch which has partial fixes.
#
# Output: dist/triton-*.whl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/env.sh"

TRITON_BRANCH="${TRITON_BRANCH:-main}"
BUILD_DIR="${BUILD_DIR:-/tmp/triton-build}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[Triton]${NC} $*"; }
warn()  { echo -e "${YELLOW}[Triton]${NC} $*"; }
error() { echo -e "${RED}[Triton]${NC} $*"; exit 1; }

# ── Pre-flight ────────────────────────────────────────────────────────────────

info "Checking requirements..."
command -v nvcc   &>/dev/null || error "nvcc not found"
command -v cmake  &>/dev/null || error "cmake not found"
command -v python &>/dev/null || error "python not found"

PTXAS_VERSION=$(ptxas --version 2>&1 | grep -oP 'release \K[\d.]+' || echo "unknown")
info "ptxas version: ${PTXAS_VERSION}"

# ── Clone ─────────────────────────────────────────────────────────────────────

if [[ -d "${BUILD_DIR}/triton" ]]; then
    info "Using existing Triton source at ${BUILD_DIR}/triton"
    cd "${BUILD_DIR}/triton"
    git fetch origin
else
    info "Cloning Triton (${TRITON_BRANCH})..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    git clone --depth 1 --branch "${TRITON_BRANCH}" \
        https://github.com/triton-lang/triton.git
    cd triton
fi

# ── Build dependencies ────────────────────────────────────────────────────────

info "Installing build dependencies..."
pip install --quiet cmake ninja pybind11 setuptools

# ── Build ─────────────────────────────────────────────────────────────────────

info "Building Triton..."
info "Build started at: $(date)"

cd python
pip wheel --no-deps --wheel-dir="${BUILD_DIR}/dist" . 2>&1 | tee "${BUILD_DIR}/triton-build.log"

# ── Copy output ───────────────────────────────────────────────────────────────

WHEEL_FILE=$(find "${BUILD_DIR}/dist" -name "triton-*.whl" | head -1)
if [[ -z "${WHEEL_FILE}" ]]; then
    error "Build failed — no wheel file produced. Check ${BUILD_DIR}/triton-build.log"
fi

cp "${WHEEL_FILE}" "${BUILD_OUTPUT_DIR}/"
FINAL_WHEEL="${BUILD_OUTPUT_DIR}/$(basename "${WHEEL_FILE}")"

info "Build complete!"
info "Wheel: ${FINAL_WHEEL}"
info "Build finished at: $(date)"
warn ""
warn "NOTE: Triton on sm_121 is experimental. Some kernels may fail at runtime."
warn "torch.compile() with Triton backend may not work for all operations."
