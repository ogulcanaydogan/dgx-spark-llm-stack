#!/usr/bin/env bash
# Build PyTorch from source for DGX Spark (GB10, sm_121, CUDA 13.0, Python 3.12)
# Expected build time: ~4 hours on DGX Spark (20 cores)
#
# Output: dist/torch-*.whl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/env.sh"

PYTORCH_VERSION="${PYTORCH_VERSION:-v2.9.1}"
BUILD_DIR="${BUILD_DIR:-/tmp/pytorch-build}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[PyTorch]${NC} $*"; }
warn()  { echo -e "${YELLOW}[PyTorch]${NC} $*"; }
error() { echo -e "${RED}[PyTorch]${NC} $*"; exit 1; }

# ── Pre-flight checks ────────────────────────────────────────────────────────

info "Checking build requirements..."

command -v nvcc   &>/dev/null || error "nvcc not found. Set CUDA_HOME correctly."
command -v git    &>/dev/null || error "git not found."
command -v python &>/dev/null || error "python not found."
command -v cmake  &>/dev/null || error "cmake not found."
command -v ninja  &>/dev/null || warn "ninja not found. Install with: pip install ninja"

PYTHON_VERSION=$(python --version 2>&1 | grep -oP '\d+\.\d+')
info "Python: ${PYTHON_VERSION}"
info "CUDA: $(nvcc --version | grep release | awk '{print $6}' | tr -d ',')"
info "Target arch: ${TORCH_CUDA_ARCH_LIST}"
info "Parallel jobs: ${MAX_JOBS}"

# ── Install build dependencies ────────────────────────────────────────────────

info "Installing build dependencies..."
pip install --quiet cmake ninja pyyaml setuptools typing-extensions

# ── Clone PyTorch ─────────────────────────────────────────────────────────────

if [[ -d "${BUILD_DIR}/pytorch" ]]; then
    info "Using existing PyTorch source at ${BUILD_DIR}/pytorch"
    cd "${BUILD_DIR}/pytorch"
    git fetch --tags
else
    info "Cloning PyTorch ${PYTORCH_VERSION}..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    git clone --recursive --depth 1 --branch "${PYTORCH_VERSION}" \
        https://github.com/pytorch/pytorch.git
    cd pytorch
fi

git checkout "${PYTORCH_VERSION}"
git submodule sync
git submodule update --init --recursive

# ── Build ─────────────────────────────────────────────────────────────────────

info "Starting PyTorch build (this will take ~4 hours)..."
info "Build started at: $(date)"

# Clean previous builds
python setup.py clean 2>/dev/null || true

# Build wheel
python setup.py bdist_wheel 2>&1 | tee "${BUILD_DIR}/pytorch-build.log"

# ── Copy output ───────────────────────────────────────────────────────────────

WHEEL_FILE=$(find dist/ -name "torch-*.whl" | head -1)
if [[ -z "${WHEEL_FILE}" ]]; then
    error "Build failed — no wheel file found. Check ${BUILD_DIR}/pytorch-build.log"
fi

cp "${WHEEL_FILE}" "${BUILD_OUTPUT_DIR}/"
FINAL_WHEEL="${BUILD_OUTPUT_DIR}/$(basename "${WHEEL_FILE}")"

info "Build complete!"
info "Wheel: ${FINAL_WHEEL}"
info "Size: $(du -h "${FINAL_WHEEL}" | cut -f1)"
info "Build finished at: $(date)"
info ""
info "Install with: pip install ${FINAL_WHEEL}"
