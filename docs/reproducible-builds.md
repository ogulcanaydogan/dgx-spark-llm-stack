# Reproducible Wheel Builds

This document describes the Phase 2 release process for reproducible `torch` and `bitsandbytes` wheels on DGX Spark.

## Build Environment (required)

- OS: Linux on ARM64 (`aarch64`)
- GPU: NVIDIA GB10 (`sm_121`)
- CUDA: `13.0` (`/usr/local/cuda-13.0`)
- Python: `3.12`
- GCC/G++: `13.x` (defaults in `configs/env.sh`)
- Runner labels (CI): `self-hosted`, `linux`, `arm64`, `dgx-spark`

## Source of Truth

- Release workflow: `.github/workflows/release-wheels.yml`
- Build scripts:
  - `build/build_pytorch.sh`
  - `build/build_bitsandbytes.sh`
- Environment config: `configs/env.sh`

## Reproducible Build Steps (manual)

1. Check out a tagged commit (example):

```bash
git fetch --tags
git checkout v0.1.0
```

2. Prepare environment and clean artifacts:

```bash
source configs/env.sh
rm -rf dist
mkdir -p dist
```

3. Build PyTorch wheel:

```bash
bash build/build_pytorch.sh
```

4. Install the generated PyTorch wheel (required before BitsAndBytes build):

```bash
pip install --force-reinstall dist/torch-*.whl
```

5. Build BitsAndBytes wheel:

```bash
bash build/build_bitsandbytes.sh
```

6. Generate checksums:

```bash
cd dist
sha256sum *.whl > SHA256SUMS
cat SHA256SUMS
```

## Artifact Naming Expectations

- `torch-*.whl` for ARM64 + Python 3.12 + CUDA 13.0 build
- `bitsandbytes-*.whl` for ARM64 + Python 3.12 + CUDA 13.0 build
- `SHA256SUMS` containing checksums for all wheels in `dist/`

## Release Procedure

1. Push a semver tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

2. GitHub Actions runs `.github/workflows/release-wheels.yml` and publishes:
   - `torch-*.whl`
   - `bitsandbytes-*.whl`
   - `SHA256SUMS`

3. GitHub Actions runs `.github/workflows/verify-install.yml` on release publish and validates:
   - `./install.sh` on a fresh environment
   - `python scripts/verify_install.py` output without failed checks

## Installer Integrity Rule

`install.sh` downloads wheel assets from the latest release and requires `SHA256SUMS` when release wheels are present. If checksum verification fails, installation aborts and does not install unverified wheels.
