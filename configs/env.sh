#!/usr/bin/env bash
# Build environment loader — delegates hardware-specific vars to a profile.
#
# Usage:
#   source configs/env.sh                    # default: dgx-spark
#   HW_PROFILE=h100 source configs/env.sh   # H100 / Hopper
#
# Available profiles: dgx-spark (default), h100
# Profile files live in configs/profiles/<name>.env

set -euo pipefail

HW_PROFILE="${HW_PROFILE:-dgx-spark}"
PROFILE_FILE="$(dirname "${BASH_SOURCE[0]}")/profiles/${HW_PROFILE}.env"

if [[ ! -f "${PROFILE_FILE}" ]]; then
    echo "ERROR: unknown HW_PROFILE '${HW_PROFILE}' — no profile at ${PROFILE_FILE}" >&2
    echo "Available profiles: $(ls "$(dirname "${BASH_SOURCE[0]}")/profiles/"*.env 2>/dev/null | xargs -n1 basename | sed 's/\.env//' | tr '\n' ' ')" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${PROFILE_FILE}"

# Common vars that apply regardless of hardware profile
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Build parallelism — use all available cores
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export MAKEFLAGS="-j${MAX_JOBS}"

# PyTorch build flags (only relevant for custom-wheels profile)
export USE_CUDA=1
export USE_CUDNN=1
export USE_NCCL=0
export USE_DISTRIBUTED=0
export USE_MKLDNN=1
export BUILD_TEST=0
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"

# Build output directory
export BUILD_OUTPUT_DIR="${BUILD_OUTPUT_DIR:-$(pwd)/dist}"
mkdir -p "${BUILD_OUTPUT_DIR}" 2>/dev/null || true

echo "=== LLM Stack Build Environment ==="
echo "HW_PROFILE:            ${HW_PROFILE}"
echo "CUDA_HOME:             ${CUDA_HOME}"
echo "TORCH_CUDA_ARCH_LIST:  ${TORCH_CUDA_ARCH_LIST}"
echo "INSTALL_STRATEGY:      ${INSTALL_STRATEGY}"
echo "MAX_JOBS:              ${MAX_JOBS}"
echo "CC:                    ${CC}"
echo "CXX:                   ${CXX}"
echo "BUILD_OUTPUT_DIR:      ${BUILD_OUTPUT_DIR}"
echo "Python:                $(python --version 2>&1)"
echo "===================================="
