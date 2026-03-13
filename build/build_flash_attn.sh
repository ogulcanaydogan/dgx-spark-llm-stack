#!/usr/bin/env bash
# Build flash-attention for DGX Spark (GB10, sm_121)
#
# WARNING: flash-attention does NOT have sm_121 kernels. This script attempts
# an sm_120 binary-compatible build. If it fails, use PyTorch SDPA instead:
#   torch.nn.functional.scaled_dot_product_attention()
#
# Output: dist/flash_attn-*.whl (if build succeeds)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/env.sh"

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-v2.7.4}"
BUILD_DIR="${BUILD_DIR:-/tmp/flash-attn-build}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[FlashAttn]${NC} $*"; }
warn()  { echo -e "${YELLOW}[FlashAttn]${NC} $*"; }
error() { echo -e "${RED}[FlashAttn]${NC} $*"; exit 1; }

# ── Pre-flight ────────────────────────────────────────────────────────────────

info "Checking requirements..."
command -v nvcc   &>/dev/null || error "nvcc not found"
command -v python &>/dev/null || error "python not found"

# Check if PyTorch is installed
python -c "import torch; print(f'PyTorch {torch.__version__}')" || \
    error "PyTorch not found. Build PyTorch first: ./build/build_pytorch.sh"

# ── Clone ─────────────────────────────────────────────────────────────────────

if [[ -d "${BUILD_DIR}/flash-attention" ]]; then
    info "Using existing source at ${BUILD_DIR}/flash-attention"
    cd "${BUILD_DIR}/flash-attention"
else
    info "Cloning flash-attention ${FLASH_ATTN_VERSION}..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    git clone --depth 1 --branch "${FLASH_ATTN_VERSION}" \
        https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
fi

# ── Attempt sm_120 compatible build ───────────────────────────────────────────

info "Attempting build with sm_120 binary compatibility..."
warn "flash-attention has no sm_121 kernels — this is a best-effort build."

# Force sm_120 target (binary compatible with sm_121)
export TORCH_CUDA_ARCH_LIST="12.0"

pip install --quiet packaging ninja setuptools

info "Building flash-attention (this may take ~20 minutes)..."
info "Build started at: $(date)"

python setup.py bdist_wheel 2>&1 | tee "${BUILD_DIR}/flash-attn-build.log"
BUILD_EXIT=$?

# ── Check result ──────────────────────────────────────────────────────────────

# Restore arch list
export TORCH_CUDA_ARCH_LIST="12.1"

WHEEL_FILE=$(find dist/ -name "flash_attn-*.whl" 2>/dev/null | head -1)

if [[ ${BUILD_EXIT} -ne 0 ]] || [[ -z "${WHEEL_FILE}" ]]; then
    warn "============================================"
    warn "flash-attention build FAILED (expected)."
    warn ""
    warn "This is a known issue — sm_121 is not supported."
    warn "Use PyTorch's built-in SDPA instead:"
    warn ""
    warn "  import torch.nn.functional as F"
    warn "  output = F.scaled_dot_product_attention(q, k, v)"
    warn ""
    warn "SDPA performance on Blackwell is comparable to flash-attention."
    warn "============================================"
    exit 0
fi

cp "${WHEEL_FILE}" "${BUILD_OUTPUT_DIR}/"
FINAL_WHEEL="${BUILD_OUTPUT_DIR}/$(basename "${WHEEL_FILE}")"

info "Build succeeded!"
info "Wheel: ${FINAL_WHEEL}"
info "Build finished at: $(date)"
warn "NOTE: This wheel uses sm_120 kernels. Some operations may not work correctly."
