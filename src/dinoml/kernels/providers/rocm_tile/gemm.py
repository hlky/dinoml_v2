from __future__ import annotations

from typing import Any, Mapping

from dinoml.kernels.families.gemm import GEMM_OPS, gemm_op_spec, normalize_gemm_dtype
from dinoml.kernels.providers.rocm_tile.common import rocm_tile_fp32_fallback_required


ROCM_TILE_GEMM_LIBRARY = "rocm_tile_gemm"


def rocm_tile_gemm_symbol(op_name: str, dtype: str) -> str:
    gemm_op_spec(op_name)
    normalized = normalize_gemm_dtype(dtype)
    if normalized != "float32":
        raise ValueError(f"ROCm Tile GEMM fallback only supports float32, got {dtype!r}")
    return f"dinoml_rocm_tile_{op_name}_float32_v1"


def rocm_tile_gemm_supported(op_name: str, dtype: str, target: Mapping[str, Any] | None = None) -> bool:
    return op_name in GEMM_OPS and rocm_tile_fp32_fallback_required(normalize_gemm_dtype(dtype), target)


def rocm_tile_gemm_static_library_name(op_name: str, dtype: str) -> str:
    del op_name, dtype
    return "dinoml_rocm_kernels"
