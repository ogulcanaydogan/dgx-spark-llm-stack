#!/usr/bin/env bash
# Build BitsAndBytes from source for DGX Spark (GB10, sm_121, CUDA 13.0, ARM64)
#
# Output: dist/bitsandbytes-*.whl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/env.sh"

BNB_VERSION="${BNB_VERSION:-0.49.0}"
BUILD_DIR="${BUILD_DIR:-/tmp/bnb-build}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[BnB]${NC} $*"; }
warn()  { echo -e "${YELLOW}[BnB]${NC} $*"; }
error() { echo -e "${RED}[BnB]${NC} $*"; exit 1; }

# ── Pre-flight ────────────────────────────────────────────────────────────────

info "Checking requirements..."
command -v nvcc   &>/dev/null || error "nvcc not found"
command -v cmake  &>/dev/null || error "cmake not found"
command -v python &>/dev/null || error "python not found"

# ── Clone ─────────────────────────────────────────────────────────────────────

if [[ -d "${BUILD_DIR}/bitsandbytes" ]]; then
    info "Using existing source at ${BUILD_DIR}/bitsandbytes"
    cd "${BUILD_DIR}/bitsandbytes"
    git fetch --tags
else
    info "Cloning bitsandbytes ${BNB_VERSION}..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    git clone --depth 1 --branch "${BNB_VERSION}" \
        https://github.com/bitsandbytes-foundation/bitsandbytes.git
    cd bitsandbytes
fi

# ── Build CUDA kernels ───────────────────────────────────────────────────────

info "Building CUDA kernels..."

# BitsAndBytes uses cmake for CUDA kernel compilation
cmake -B build \
    -DCOMPUTE_BACKEND=cuda \
    -DCOMPUTE_CAPABILITY="121" \
    -DCMAKE_CUDA_ARCHITECTURES="121" \
    -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j "${MAX_JOBS}"

# ── Build wheel ───────────────────────────────────────────────────────────────

info "Building Python wheel..."
info "Build started at: $(date)"

pip install --quiet setuptools wheel
python setup.py bdist_wheel 2>&1 | tee "${BUILD_DIR}/bnb-build.log"

# ── Copy output ───────────────────────────────────────────────────────────────

WHEEL_FILE=$(find dist/ -name "bitsandbytes-*.whl" | head -1)
if [[ -z "${WHEEL_FILE}" ]]; then
    error "Build failed — no wheel file produced. Check ${BUILD_DIR}/bnb-build.log"
fi

cp "${WHEEL_FILE}" "${BUILD_OUTPUT_DIR}/"
FINAL_WHEEL="${BUILD_OUTPUT_DIR}/$(basename "${WHEEL_FILE}")"

info "Build complete!"
info "Wheel: ${FINAL_WHEEL}"
info "Build finished at: $(date)"
info ""
info "Install with: pip install ${FINAL_WHEEL}"
