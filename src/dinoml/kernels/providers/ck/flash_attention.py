from __future__ import annotations

import os


FLASH_ATTN_CK_LIBRARY = "flash_attn_ck"
FLASH_ATTN_CK_SUPPORTED_DTYPES = ("float16", "bfloat16")


def flash_attn_ck_symbol(dtype: str) -> str:
    normalized = _normalize_flash_attn_dtype(dtype)
    return f"dinoml_flash_attn_ck_fwd_{normalized}_v1"


def flash_attn_ck_qkv_symbol(dtype: str) -> str:
    normalized = _normalize_flash_attn_dtype(dtype)
    return f"dinoml_flash_attn_ck_qkv_fwd_{normalized}_v1"


def flash_attn_ck_static_library_name(dtype: str = "float16") -> str:
    _normalize_flash_attn_dtype(dtype)
    stem = "dinoml_flash_attn_ck"
    return f"{stem}.lib" if os.name == "nt" else f"lib{stem}.a"


def flash_attn_ck_cmake_target() -> str:
    return "dinoml_flash_attn_ck"


def _normalize_flash_attn_dtype(dtype: str) -> str:
    normalized = str(dtype)
    if normalized not in FLASH_ATTN_CK_SUPPORTED_DTYPES:
        supported = ", ".join(FLASH_ATTN_CK_SUPPORTED_DTYPES)
        raise ValueError(f"CK FlashAttention supports {supported}, got {dtype!r}")
    return normalized
