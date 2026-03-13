#!/usr/bin/env bash
# Environment variables for building ML libraries on DGX Spark (GB10, sm_121)
# Source this before running any build script: source configs/env.sh

set -euo pipefail

# CUDA configuration
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# GPU architecture targeting
export TORCH_CUDA_ARCH_LIST="12.1"
export CUDA_COMPUTE_CAPABILITIES="12.1"

# Build parallelism — use all available cores
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export MAKEFLAGS="-j${MAX_JOBS}"

# PyTorch build flags
export USE_CUDA=1
export USE_CUDNN=1
export USE_NCCL=0
export USE_DISTRIBUTED=0
export USE_MKLDNN=1
export BUILD_TEST=0
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"

# Compiler
export CC="${CC:-gcc-13}"
export CXX="${CXX:-g++-13}"

# Build output directory
export BUILD_OUTPUT_DIR="${BUILD_OUTPUT_DIR:-$(pwd)/dist}"
mkdir -p "${BUILD_OUTPUT_DIR}" 2>/dev/null || true

echo "=== DGX Spark Build Environment ==="
echo "CUDA_HOME:             ${CUDA_HOME}"
echo "TORCH_CUDA_ARCH_LIST:  ${TORCH_CUDA_ARCH_LIST}"
echo "MAX_JOBS:              ${MAX_JOBS}"
echo "CC:                    ${CC}"
echo "CXX:                   ${CXX}"
echo "BUILD_OUTPUT_DIR:      ${BUILD_OUTPUT_DIR}"
echo "Python:                $(python --version 2>&1)"
echo "===================================="
