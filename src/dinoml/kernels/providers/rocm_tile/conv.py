from __future__ import annotations

from typing import Any, Mapping

from dinoml.kernels.providers.rocm_tile.common import rocm_tile_fp32_fallback_required
from dinoml.ops.conv import CONV2D_BIAS_FAMILY_OPS, CONV2D_BIAS_DTYPES


ROCM_TILE_CONV_LIBRARY = "rocm_tile_conv"


def rocm_tile_conv_symbol(op_name: str, dtype: str) -> str:
    if op_name not in CONV2D_BIAS_FAMILY_OPS:
        raise ValueError(f"Unsupported ROCm Tile Conv op {op_name!r}")
    if dtype not in CONV2D_BIAS_DTYPES:
        raise ValueError(f"Unsupported Conv dtype {dtype!r}")
    if dtype != "float32":
        raise ValueError(f"ROCm Tile Conv fallback only supports float32, got {dtype!r}")
    return f"dinoml_rocm_tile_{op_name}_float32_v1"


def rocm_tile_conv_supported(op_name: str, dtype: str, target: Mapping[str, Any] | None = None) -> bool:
    return (
        op_name in CONV2D_BIAS_FAMILY_OPS
        and dtype == "float32"
        and rocm_tile_fp32_fallback_required(dtype, target)
    )


def rocm_tile_conv_static_library_name(op_name: str, dtype: str) -> str:
    del op_name, dtype
    return "dinoml_rocm_kernels"
