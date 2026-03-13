#!/usr/bin/env bash
# DGX Spark LLM Stack — One-command installer
# Downloads pre-built wheels from GitHub Releases and installs the full stack.
#
# Usage: ./install.sh [--from-source]

set -euo pipefail

REPO="ogulcanaydogan/dgx-spark-llm-stack"
RELEASE_TAG="latest"
WHEEL_DIR="/tmp/dgx-spark-wheels"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Pre-flight checks ────────────────────────────────────────────────────────

check_system() {
    info "Checking system requirements..."

    # Check we're on Linux ARM64
    if [[ "$(uname -s)" != "Linux" ]]; then
        error "This installer is for Linux only (detected: $(uname -s))"
    fi

    if [[ "$(uname -m)" != "aarch64" ]]; then
        error "This installer is for ARM64 only (detected: $(uname -m))"
    fi

    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
    if [[ "${PYTHON_VERSION}" != "3.12" ]]; then
        warn "Expected Python 3.12, found ${PYTHON_VERSION}. Wheels may not be compatible."
    fi

    # Check CUDA
    if ! command -v nvcc &>/dev/null; then
        error "nvcc not found. Is CUDA installed? Expected CUDA 13.0 at /usr/local/cuda-13.0"
    fi

    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+')
    info "CUDA version: ${CUDA_VERSION}"

    # Check GPU
    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        info "GPU: ${GPU_NAME}"
    fi
}

# ── Install from pre-built wheels ─────────────────────────────────────────────

install_from_wheels() {
    info "Downloading pre-built wheels from GitHub Releases..."
    mkdir -p "${WHEEL_DIR}"

    # Get the latest release asset URLs
    ASSETS=$(gh release view "${RELEASE_TAG}" --repo "${REPO}" --json assets -q '.assets[].url' 2>/dev/null || true)

    if [[ -z "${ASSETS}" ]]; then
        warn "No pre-built wheels found in GitHub Releases."
        warn "Falling back to pip install (without custom wheels)."
        install_pip_packages
        return
    fi

    # Download all wheel files
    while IFS= read -r url; do
        filename=$(basename "${url}")
        if [[ "${filename}" == *.whl ]]; then
            info "Downloading ${filename}..."
            gh release download "${RELEASE_TAG}" --repo "${REPO}" \
                --pattern "${filename}" --dir "${WHEEL_DIR}" --clobber
        fi
    done <<< "${ASSETS}"

    # Install wheels in dependency order
    info "Installing wheels..."

    # PyTorch first
    TORCH_WHEEL=$(find "${WHEEL_DIR}" -name "torch-*.whl" 2>/dev/null | head -1)
    if [[ -n "${TORCH_WHEEL}" ]]; then
        pip install "${TORCH_WHEEL}" --force-reinstall
    fi

    # Then other custom wheels
    for wheel in "${WHEEL_DIR}"/*.whl; do
        [[ "${wheel}" == *"torch-"* ]] && continue
        pip install "${wheel}" --force-reinstall
    done

    # Install remaining packages from PyPI
    install_pip_packages
}

# ── Install pip packages ──────────────────────────────────────────────────────

install_pip_packages() {
    info "Installing ML packages from PyPI..."

    pip install --upgrade pip setuptools wheel

    # Core ML stack
    pip install \
        transformers \
        accelerate \
        datasets \
        tokenizers \
        safetensors \
        peft \
        trl \
        sentencepiece \
        protobuf

    # Quantization
    if ! python3 -c "import bitsandbytes" 2>/dev/null; then
        pip install bitsandbytes || warn "bitsandbytes install failed — build from source with ./build/build_bitsandbytes.sh"
    fi

    # Training utilities
    pip install \
        wandb \
        tensorboard \
        rich

    info "Package installation complete."
}

# ── Build from source ─────────────────────────────────────────────────────────

install_from_source() {
    info "Building all components from source..."
    source "${SCRIPT_DIR}/configs/env.sh"
    bash "${SCRIPT_DIR}/build/build_all.sh"
    install_pip_packages
}

# ── Verify installation ──────────────────────────────────────────────────────

verify() {
    info "Running installation verification..."
    python3 "${SCRIPT_DIR}/scripts/verify_install.py"
}

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
    echo "╔══════════════════════════════════════════════════╗"
    echo "║      DGX Spark LLM Stack — Installer            ║"
    echo "║      GB10 (sm_121) · CUDA 13.0 · Python 3.12    ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""

    check_system

    if [[ "${1:-}" == "--from-source" ]]; then
        install_from_source
    else
        install_from_wheels
    fi

    verify

    echo ""
    info "Installation complete! Run 'python scripts/verify_install.py' anytime to check status."
}

main "$@"
