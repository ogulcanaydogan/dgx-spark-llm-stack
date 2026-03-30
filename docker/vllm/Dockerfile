ARG NGC_BASE=nvcr.io/nvidia/pytorch:25.11-py3
ARG VLLM_REF=v0.18.0
ARG XFORMERS_VERSION=0.0.31
ARG XFORMERS_DISABLE_FLASH_ATTN=1
ARG XFORMERS_BUILD_JOBS=2
ARG TORCH_CUDA_ARCH_LIST=12.1

FROM ${NGC_BASE} AS builder
ARG VLLM_REF
ARG XFORMERS_VERSION
ARG XFORMERS_DISABLE_FLASH_ATTN
ARG XFORMERS_BUILD_JOBS
ARG TORCH_CUDA_ARCH_LIST

WORKDIR /opt/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none https://github.com/vllm-project/vllm.git vllm \
    && git -C vllm checkout --detach "tags/${VLLM_REF}"

WORKDIR /opt/src/vllm

RUN python -m pip install --upgrade pip setuptools wheel "setuptools_scm[toml]" \
    && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${XFORMERS_BUILD_JOBS}" XFORMERS_DISABLE_FLASH_ATTN="${XFORMERS_DISABLE_FLASH_ATTN}" \
       python -m pip wheel "xformers==${XFORMERS_VERSION}" \
      --no-build-isolation --no-deps --wheel-dir /opt/wheels \
    && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" python -m pip wheel . --no-build-isolation --no-deps --wheel-dir /opt/wheels

FROM ${NGC_BASE} AS runtime
ARG VLLM_REF

ENV PIP_NO_CACHE_DIR=1 \
    VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    VLLM_HOST=0.0.0.0 \
    VLLM_PORT=8000 \
    VLLM_DTYPE=bfloat16 \
    VLLM_MAX_MODEL_LEN=4096 \
    VLLM_GPU_MEMORY_UTILIZATION=0.9 \
    VLLM_USE_V1=1 \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    VLLM_ENABLE_CUSTOM_OPS=0 \
    VLLM_PINNED_REF=${VLLM_REF}

COPY --from=builder /opt/wheels /opt/wheels

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-deps /opt/wheels/xformers-*.whl /opt/wheels/vllm-*.whl \
    && python - <<'PY' > /tmp/vllm-runtime-requirements.txt
import email
import glob
import zipfile

from packaging.requirements import Requirement

wheel_path = glob.glob("/opt/wheels/vllm-*.whl")[0]
skip = {"torch", "torchaudio", "torchvision", "triton", "xformers"}

with zipfile.ZipFile(wheel_path) as zf:
    metadata_name = [name for name in zf.namelist()
                     if name.endswith(".dist-info/METADATA")][0]
    metadata = email.message_from_bytes(zf.read(metadata_name))

requirements = []
for raw_req in metadata.get_all("Requires-Dist") or []:
    req = Requirement(raw_req)
    name = req.name.lower()
    if name in skip:
        continue
    if name == "transformers":
        requirements.append("transformers>=4.48.2,<5.0.0")
        continue
    if name == "huggingface-hub":
        requirements.append("huggingface-hub>=0.30.0,<1.0.0")
        continue
    requirements.append(str(req))

print("\n".join(requirements))
PY

RUN python -m pip install --no-cache-dir -r /tmp/vllm-runtime-requirements.txt \
    && rm -rf /opt/wheels /tmp/vllm-runtime-requirements.txt

EXPOSE 8000

CMD ["bash", "-lc", "python -m vllm.entrypoints.openai.api_server --model \"${VLLM_MODEL}\" --host \"${VLLM_HOST}\" --port \"${VLLM_PORT}\" --dtype \"${VLLM_DTYPE}\" --max-model-len \"${VLLM_MAX_MODEL_LEN}\" --gpu-memory-utilization \"${VLLM_GPU_MEMORY_UTILIZATION}\""]
