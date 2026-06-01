from __future__ import annotations

import os


FLASH_ATTN_CUDA_LIBRARY = "flash_attn_cuda"
FLASH_ATTN_CUDA_SUPPORTED_DTYPES = ("float16", "bfloat16")


def flash_attn_cuda_symbol(dtype: str) -> str:
    normalized = _normalize_flash_attn_dtype(dtype)
    return f"dinoml_flash_attn_cuda_fwd_{normalized}_v1"


def flash_attn_cuda_qkv_symbol(dtype: str) -> str:
    normalized = _normalize_flash_attn_dtype(dtype)
    return f"dinoml_flash_attn_cuda_qkv_fwd_{normalized}_v1"


def flash_attn_cuda_static_kv_cache_symbol(dtype: str) -> str:
    normalized = _normalize_flash_attn_dtype(dtype)
    return f"dinoml_flash_attn_cuda_static_kv_cache_fwd_{normalized}_v1"


def flash_attn_cuda_static_library_name(dtype: str = "float16") -> str:
    _normalize_flash_attn_dtype(dtype)
    stem = "dinoml_flash_attn_cuda"
    return f"{stem}.lib" if os.name == "nt" else f"lib{stem}.a"


def flash_attn_cuda_upstream_static_library_name() -> str:
    stem = "flash_attention"
    return f"{stem}.lib" if os.name == "nt" else f"lib{stem}.a"


def flash_attn_cuda_cmake_target() -> str:
    return "dinoml_flash_attn_cuda"


def flash_attn_cuda_upstream_cmake_target() -> str:
    return "flash_attention"


def _normalize_flash_attn_dtype(dtype: str) -> str:
    normalized = str(dtype)
    if normalized not in FLASH_ATTN_CUDA_SUPPORTED_DTYPES:
        supported = ", ".join(FLASH_ATTN_CUDA_SUPPORTED_DTYPES)
        raise ValueError(f"CUDA FlashAttention supports {supported}, got {dtype!r}")
    return normalized
