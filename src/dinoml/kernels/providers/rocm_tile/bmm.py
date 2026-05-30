from __future__ import annotations

from typing import Any, Mapping

from dinoml.kernels.families.bmm import BMM_OPS, bmm_op_spec, normalize_bmm_dtype
from dinoml.kernels.providers.rocm_tile.common import rocm_tile_fp32_fallback_required


ROCM_TILE_BMM_LIBRARY = "rocm_tile_bmm"


def rocm_tile_bmm_symbol(op_name: str, dtype: str) -> str:
    bmm_op_spec(op_name)
    normalized = normalize_bmm_dtype(dtype)
    if normalized != "float32":
        raise ValueError(f"ROCm Tile BMM fallback only supports float32, got {dtype!r}")
    return f"dinoml_rocm_tile_{op_name}_float32_v1"


def rocm_tile_bmm_supported(op_name: str, dtype: str, target: Mapping[str, Any] | None = None) -> bool:
    return op_name in BMM_OPS and rocm_tile_fp32_fallback_required(normalize_bmm_dtype(dtype), target)


def rocm_tile_bmm_static_library_name(op_name: str, dtype: str) -> str:
    del op_name, dtype
    return "dinoml_rocm_kernels"
