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

# ── Helpers ───────────────────────────────────────────────────────────────────

command_exists() {
    command -v "$1" &>/dev/null
}

fetch_latest_release_assets() {
    local token
    local tmp_json
    local api_url
    token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
    api_url="https://api.github.com/repos/${REPO}/releases/${RELEASE_TAG}"
    tmp_json="$(mktemp)"

    if ! command_exists curl; then
        warn "curl is not available; cannot fetch release assets."
        rm -f "${tmp_json}"
        return 1
    fi

    if [[ -n "${token}" ]]; then
        curl -fsSL \
            -H "Accept: application/vnd.github+json" \
            -H "Authorization: Bearer ${token}" \
            "${api_url}" -o "${tmp_json}" || {
                rm -f "${tmp_json}"
                return 1
            }
    else
        curl -fsSL \
            -H "Accept: application/vnd.github+json" \
            "${api_url}" -o "${tmp_json}" || {
                rm -f "${tmp_json}"
                return 1
            }
    fi

    python3 - "${tmp_json}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    data = json.load(fh)

for asset in data.get("assets", []):
    name = asset.get("name", "")
    url = asset.get("browser_download_url", "")
    if name and url:
        print(f"{name}|{url}")
PY

    rm -f "${tmp_json}"
}

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
    info "Attempting install from latest GitHub Release assets..."
    mkdir -p "${WHEEL_DIR}"
    rm -f "${WHEEL_DIR}"/*.whl "${WHEEL_DIR}/SHA256SUMS" 2>/dev/null || true

    # Query latest release assets through GitHub API.
    local release_assets
    release_assets="$(fetch_latest_release_assets || true)"

    if [[ -z "${release_assets}" ]]; then
        warn "No pre-built wheels found in GitHub Releases."
        warn "Falling back to pip install (without custom wheels)."
        install_pip_packages
        return
    fi

    # Download wheel assets and checksum manifest.
    local downloaded_any=0
    while IFS='|' read -r filename url; do
        [[ -z "${filename}" ]] && continue
        if [[ "${filename}" == *.whl || "${filename}" == "SHA256SUMS" ]]; then
            downloaded_any=1
            info "Downloading ${filename}..."
            curl -fsSL --retry 3 --retry-delay 2 \
                "${url}" -o "${WHEEL_DIR}/${filename}"
        fi
    done <<< "${release_assets}"

    if [[ "${downloaded_any}" -eq 0 ]]; then
        warn "Latest release has no wheel assets."
        warn "Falling back to pip install (without custom wheels)."
        install_pip_packages
        return
    fi

    shopt -s nullglob
    local wheels=("${WHEEL_DIR}"/*.whl)
    shopt -u nullglob

    if [[ "${#wheels[@]}" -eq 0 ]]; then
        warn "No wheel files were downloaded from latest release."
        warn "Falling back to pip install (without custom wheels)."
        install_pip_packages
        return
    fi

    # Mandatory integrity check when release wheel assets are present.
    if [[ ! -f "${WHEEL_DIR}/SHA256SUMS" ]]; then
        error "Release wheels found but SHA256SUMS is missing. Refusing to install unverified artifacts."
    fi

    info "Verifying wheel checksums..."
    (
        cd "${WHEEL_DIR}"
        sha256sum -c SHA256SUMS
    ) || error "Checksum verification failed for release assets."

    # Install wheels in dependency order
    info "Installing wheels..."

    # PyTorch first
    TORCH_WHEEL=$(find "${WHEEL_DIR}" -maxdepth 1 -name "torch-*.whl" 2>/dev/null | head -1)
    if [[ -n "${TORCH_WHEEL}" ]]; then
        pip install "${TORCH_WHEEL}" --force-reinstall --no-deps
    fi

    # Then other custom wheels
    for wheel in "${wheels[@]}"; do
        [[ "${wheel}" == *"torch-"* ]] && continue
        pip install "${wheel}" --force-reinstall --no-deps
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
