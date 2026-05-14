#!/usr/bin/env bash
# DGX Spark LLM Stack — One-command installer
#
# Usage:
#   ./install.sh              # default: DGX Spark (GB10), custom wheels
#   HW_PROFILE=h100 ./install.sh        # H100 (Hopper), upstream pip wheels
#   ./install.sh --from-source          # build everything from source (DGX Spark only)

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

# ── Load hardware profile ─────────────────────────────────────────────────────

load_profile() {
    local env_sh="${SCRIPT_DIR}/configs/env.sh"
    if [[ -f "${env_sh}" ]]; then
        # shellcheck source=configs/env.sh
        source "${env_sh}"
    else
        warn "configs/env.sh not found; falling back to DGX Spark defaults."
        HW_PROFILE="${HW_PROFILE:-dgx-spark}"
        EXPECTED_ARCH="${EXPECTED_ARCH:-aarch64}"
        INSTALL_STRATEGY="${INSTALL_STRATEGY:-custom-wheels}"
    fi
}

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

    if [[ "$(uname -s)" != "Linux" ]]; then
        error "This installer is for Linux only (detected: $(uname -s))"
    fi

    local actual_arch
    actual_arch="$(uname -m)"
    if [[ "${actual_arch}" != "${EXPECTED_ARCH}" ]]; then
        error "Profile '${HW_PROFILE}' expects ${EXPECTED_ARCH} but found ${actual_arch}. Set HW_PROFILE correctly."
    fi

    PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
    if [[ "${PYTHON_VERSION}" != "3.12" ]]; then
        warn "Expected Python 3.12, found ${PYTHON_VERSION}. Wheels may not be compatible."
    fi

    if ! command -v nvcc &>/dev/null; then
        error "nvcc not found. Is CUDA installed? Expected CUDA at ${CUDA_HOME:-<not set>}"
    fi

    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[\d.]+')
    info "CUDA version: ${CUDA_VERSION}"

    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        info "GPU: ${GPU_NAME}"
    fi
}

# ── H100 upstream-wheels install ──────────────────────────────────────────────

install_upstream_pytorch() {
    info "H100 profile: installing PyTorch from upstream wheels (cu124)..."
    pip install --upgrade pip setuptools wheel

    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

    # flash-attention ships sm_90 kernels upstream
    pip install flash-attn --no-build-isolation || \
        warn "flash-attn install failed; SDPA fallback will be used automatically."

    install_pip_packages
}

# ── DGX Spark custom-wheels install ──────────────────────────────────────────

install_from_wheels() {
    info "DGX Spark profile: attempting install from latest GitHub Release assets..."
    mkdir -p "${WHEEL_DIR}"
    rm -f "${WHEEL_DIR}"/*.whl "${WHEEL_DIR}/SHA256SUMS" 2>/dev/null || true

    local release_assets
    release_assets="$(fetch_latest_release_assets || true)"

    if [[ -z "${release_assets}" ]]; then
        warn "No pre-built wheels found in GitHub Releases."
        warn "Falling back to pip install (without custom wheels)."
        install_pip_packages
        return
    fi

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

    if [[ ! -f "${WHEEL_DIR}/SHA256SUMS" ]]; then
        error "Release wheels found but SHA256SUMS is missing. Refusing to install unverified artifacts."
    fi

    info "Verifying wheel checksums..."
    (
        cd "${WHEEL_DIR}"
        sha256sum -c SHA256SUMS
    ) || error "Checksum verification failed for release assets."

    info "Installing wheels..."
    TORCH_WHEEL=$(find "${WHEEL_DIR}" -maxdepth 1 -name "torch-*.whl" 2>/dev/null | head -1)
    if [[ -n "${TORCH_WHEEL}" ]]; then
        pip install "${TORCH_WHEEL}" --force-reinstall --no-deps
    fi

    for wheel in "${wheels[@]}"; do
        [[ "${wheel}" == *"torch-"* ]] && continue
        pip install "${wheel}" --force-reinstall --no-deps
    done

    install_pip_packages
}

# ── Common pip packages ───────────────────────────────────────────────────────

install_pip_packages() {
    info "Installing ML packages from PyPI..."

    pip install --upgrade pip setuptools wheel

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

    if ! python3 -c "import bitsandbytes" 2>/dev/null; then
        pip install bitsandbytes || warn "bitsandbytes install failed — build from source with ./build/build_bitsandbytes.sh"
    fi

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
    load_profile

    echo "╔══════════════════════════════════════════════════╗"
    printf "║  DGX Spark LLM Stack — Installer                ║\n"
    printf "║  Profile: %-38s║\n" "${HW_PROFILE} (${INSTALL_STRATEGY})"
    printf "║  Arch: %-41s║\n" "${EXPECTED_ARCH:-unknown}"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""

    check_system

    if [[ "${1:-}" == "--from-source" ]]; then
        install_from_source
    elif [[ "${INSTALL_STRATEGY:-custom-wheels}" == "upstream-wheels" ]]; then
        install_upstream_pytorch
    else
        install_from_wheels
    fi

    verify

    echo ""
    info "Installation complete! Run 'python scripts/verify_install.py' anytime to check status."
}

main "$@"
