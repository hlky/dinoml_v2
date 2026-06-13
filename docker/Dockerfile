# syntax=docker/dockerfile:1.7

ARG UBUNTU_IMAGE=ubuntu:22.04

FROM ${UBUNTU_IMAGE} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_DISTRO=ubuntu2204
ARG CUDA_ARCH=x86_64
ARG CUDA_TOOLKIT_VERSION=12-9
ARG CUDA_KEYRING_VERSION=1.1-1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHAUDIO_VERSION=2.10.0
ARG CUDA_BINDINGS_VERSION=12.9.4
ARG TRITON_VERSION=3.6.0
ARG DINOML_V1_GIT_URL=https://github.com/hlky/dinoml.git
ARG DINOML_V1_REF=main
ARG TRANSFORMERS_GIT_URL=https://github.com/huggingface/transformers.git
ARG TRANSFORMERS_REF=main
ARG DIFFUSERS_GIT_URL=https://github.com/huggingface/diffusers.git
ARG DIFFUSERS_REF=main
ARG RUNPODCTL_VERSION=v2.1.6

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SHELL=/bin/bash \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    RP_WORKSPACE=/workspace \
    HF_HOME=/workspace/.cache/huggingface \
    PIP_CACHE_DIR=/workspace/.cache/pip \
    DINOML_V2_SRC=/opt/src/dinoml_v2 \
    DINOML_V1_SRC=/opt/src/dinoml_v1 \
    TRANSFORMERS_SRC=/opt/src/transformers \
    DIFFUSERS_SRC=/opt/src/diffusers \
    DINOML_VENV=/opt/venvs/dinoml \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcusparseLt/12:/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    PATH=/opt/venvs/dinoml/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/bin:${PATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      dpkg \
      git \
      gnupg \
      libopenblas0-pthread \
      nginx \
      ninja-build \
      openssh-server \
      pkg-config \
      python3 \
      python3-dev \
      python3-venv \
      tar \
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/${CUDA_ARCH}/cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb" -O /tmp/cuda-keyring.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    rm -f /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      "cuda-toolkit-${CUDA_TOOLKIT_VERSION}" \
      libcudnn9-dev-cuda-12 \
      libcusparselt0-cuda-12 \
      libnccl2 \
      libnvshmem3-cuda-12 && \
    printf '%s\n' \
      /usr/local/cuda/lib64 \
      /usr/local/cuda/targets/x86_64-linux/lib \
      /usr/lib/x86_64-linux-gnu/libcusparseLt/12 \
      /usr/lib/x86_64-linux-gnu/nvshmem/12 \
      > /etc/ld.so.conf.d/dinoml-cuda.conf && \
    ldconfig && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /var/cache/apt/*.bin

WORKDIR /opt/src

RUN mkdir -p "${RP_WORKSPACE}" "${HF_HOME}" "${PIP_CACHE_DIR}" /var/run/sshd && \
    curl -fsSL "https://github.com/runpod/runpodctl/releases/download/${RUNPODCTL_VERSION}/runpodctl-linux-amd64.tar.gz" | \
      tar -xz -C /usr/local/bin runpodctl

RUN git clone \
      --branch "${DINOML_V1_REF}" \
      --depth 1 \
      --recursive \
      "${DINOML_V1_GIT_URL}" \
      "${DINOML_V1_SRC}" && \
    rm -rf "${DINOML_V1_SRC}/.git" && \
    git clone \
      --branch "${TRANSFORMERS_REF}" \
      --depth 1 \
      "${TRANSFORMERS_GIT_URL}" \
      "${TRANSFORMERS_SRC}" && \
    rm -rf "${TRANSFORMERS_SRC}/.git" && \
    git clone \
      --branch "${DIFFUSERS_REF}" \
      --depth 1 \
      "${DIFFUSERS_GIT_URL}" \
      "${DIFFUSERS_SRC}" && \
    rm -rf "${DIFFUSERS_SRC}/.git"

COPY . ${DINOML_V2_SRC}/

RUN test -d "${DINOML_V2_SRC}/.git" && \
    test -f "${DINOML_V2_SRC}/third_party/cutlass/CMakeLists.txt" && \
    test -f "${DINOML_V2_SRC}/third_party/libgguf/pyproject.toml" && \
    test -f "${DINOML_V2_SRC}/third_party/flash_attn/CMakeLists.txt" && \
    test -f "${DINOML_V2_SRC}/third_party/composable_kernel/CHANGELOG.md" && \
    test -f "${DINOML_V1_SRC}/.gitmodules"

RUN python3 -m venv "${DINOML_VENV}" && \
    "${DINOML_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel && \
    "${DINOML_VENV}/bin/pip" install \
      --index-url "${TORCH_INDEX_URL}" \
      --extra-index-url https://pypi.org/simple \
      --no-deps \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}" && \
    "${DINOML_VENV}/bin/pip" install \
      --extra-index-url "${TORCH_INDEX_URL}" \
      "cuda-bindings==${CUDA_BINDINGS_VERSION}" \
      "cmake>=3.24,<4" \
      filelock \
      fsspec \
      gguf \
      "hf-xet==1.5.1" \
      httpx \
      "huggingface_hub==1.19.0" \
      importlib_metadata \
      jinja2 \
      networkx \
      numpy \
      packaging \
      pillow \
      pybind11 \
      psutil \
      pyyaml \
      pytest \
      regex \
      requests \
      safetensors \
      sympy \
      "tokenizers==0.22.2" \
      "triton==${TRITON_VERSION}" \
      tqdm \
      typer \
      typing-extensions && \
    "${DINOML_VENV}/bin/pip" install click ninja && \
    "${DINOML_VENV}/bin/pip" install --no-deps "accelerate==1.14.0" && \
    "${DINOML_VENV}/bin/pip" install --no-deps -e "${TRANSFORMERS_SRC}" && \
    "${DINOML_VENV}/bin/pip" install --no-deps -e "${DIFFUSERS_SRC}" && \
    "${DINOML_VENV}/bin/pip" install --no-deps -e "${DINOML_V2_SRC}[dev]" && \
    "${DINOML_VENV}/bin/pip" install --no-deps -e "${DINOML_V2_SRC}/third_party/libgguf"

RUN "${DINOML_VENV}/bin/python" - <<'PY'
import dinoml
import diffusers
import libgguf
import torch
import transformers

print("v2", dinoml.__file__)
print("cuda_home", __import__("os").environ.get("CUDA_HOME"))
print("torch", torch.__version__, torch.version.cuda)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("libgguf", libgguf.__file__)
PY

RUN "${DINOML_VENV}/bin/python" -m pybind11 --cmakedir

RUN PYTHONPATH="/opt/src/dinoml_v1/src${PYTHONPATH:+:${PYTHONPATH}}" "${DINOML_VENV}/bin/python" - <<'PY'
import dinoml

print("v1", dinoml.__file__)
PY

FROM ${UBUNTU_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_DISTRO=ubuntu2204
ARG CUDA_ARCH=x86_64
ARG CUDA_TOOLKIT_VERSION=12-9
ARG CUDA_KEYRING_VERSION=1.1-1
ARG RUNPODCTL_VERSION=v2.1.6

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SHELL=/bin/bash \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    RP_WORKSPACE=/workspace \
    HF_HOME=/workspace/.cache/huggingface \
    PIP_CACHE_DIR=/workspace/.cache/pip \
    DINOML_V2_SRC=/opt/src/dinoml_v2 \
    DINOML_V1_SRC=/opt/src/dinoml_v1 \
    TRANSFORMERS_SRC=/opt/src/transformers \
    DIFFUSERS_SRC=/opt/src/diffusers \
    DINOML_VENV=/opt/venvs/dinoml \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcusparseLt/12:/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    PATH=/opt/venvs/dinoml/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/bin:${PATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      dpkg \
      git \
      gnupg \
      libopenblas0-pthread \
      nginx \
      ninja-build \
      openssh-server \
      pkg-config \
      python3 \
      python3-dev \
      python3-venv \
      tar \
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/${CUDA_ARCH}/cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb" -O /tmp/cuda-keyring.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    rm -f /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      "cuda-toolkit-${CUDA_TOOLKIT_VERSION}" \
      libcudnn9-dev-cuda-12 \
      libcusparselt0-cuda-12 \
      libnccl2 \
      libnvshmem3-cuda-12 && \
    find /usr/local/cuda-12.9/targets/x86_64-linux/lib -maxdepth 1 -name '*.a' \
      ! -name 'libcudadevrt.a' \
      ! -name 'libcudart_static.a' \
      ! -name 'libculibos.a' \
      ! -name 'libcublas_static.a' \
      ! -name 'libcublasLt_static.a' \
      ! -name 'libcurand_static.a' \
      -delete && \
    rm -f /lib/x86_64-linux-gnu/libnccl_static.a /usr/lib/x86_64-linux-gnu/libnccl_static.a && \
    rm -rf /usr/local/cuda-12.9/compat && \
    printf '%s\n' \
      /usr/local/cuda/lib64 \
      /usr/local/cuda/targets/x86_64-linux/lib \
      /usr/lib/x86_64-linux-gnu/libcusparseLt/12 \
      /usr/lib/x86_64-linux-gnu/nvshmem/12 \
      > /etc/ld.so.conf.d/dinoml-cuda.conf && \
    ldconfig && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /var/cache/apt/*.bin

WORKDIR /opt/src

RUN mkdir -p "${RP_WORKSPACE}" "${HF_HOME}" "${PIP_CACHE_DIR}" /var/run/sshd && \
    curl -fsSL "https://github.com/runpod/runpodctl/releases/download/${RUNPODCTL_VERSION}/runpodctl-linux-amd64.tar.gz" | \
      tar -xz -C /usr/local/bin runpodctl

COPY --from=builder /opt/venvs/dinoml /opt/venvs/dinoml
COPY --from=builder /opt/src/dinoml_v1 /opt/src/dinoml_v1
COPY --from=builder /opt/src/transformers /opt/src/transformers
COPY --from=builder /opt/src/diffusers /opt/src/diffusers
COPY --from=builder /opt/src/dinoml_v2 /opt/src/dinoml_v2

RUN printf '%s\n' \
      '#!/usr/bin/env bash' \
      'set -euo pipefail' \
      'exec "'"${DINOML_VENV}"'/bin/python" "$@"' \
      > /usr/local/bin/dinoml-v2-python && \
    printf '%s\n' \
      '#!/usr/bin/env bash' \
      'set -euo pipefail' \
      'export PYTHONPATH="/opt/src/dinoml_v1/src${PYTHONPATH:+:${PYTHONPATH}}"' \
      'exec "'"${DINOML_VENV}"'/bin/python" "$@"' \
      > /usr/local/bin/dinoml-v1-python && \
    printf '%s\n' \
      '#!/usr/bin/env bash' \
      'set -euo pipefail' \
      'exec "'"${DINOML_VENV}"'/bin/python" -m dinoml.cli "$@"' \
      > /usr/local/bin/dinoml-v2-cli && \
    chmod +x \
      /usr/local/bin/dinoml-v1-python \
      /usr/local/bin/dinoml-v2-python \
      /usr/local/bin/dinoml-v2-cli

RUN "${DINOML_VENV}/bin/python" - <<'PY'
import dinoml
import diffusers
import libgguf
import torch
import transformers

print("v2", dinoml.__file__)
print("cuda_home", __import__("os").environ.get("CUDA_HOME"))
print("torch", torch.__version__, torch.version.cuda)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("libgguf", libgguf.__file__)
PY

RUN "${DINOML_VENV}/bin/python" -m pybind11 --cmakedir

RUN dinoml-v1-python - <<'PY'
import dinoml

print("v1", dinoml.__file__)
PY

RUN command -v nvcc && nvcc --version

COPY --chmod=755 docker/start.sh /start.sh

RUN rm -f /etc/ssh/ssh_host_*

WORKDIR ${RP_WORKSPACE}

ENTRYPOINT ["/start.sh"]
