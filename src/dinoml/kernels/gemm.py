from __future__ import annotations

from dinoml.ir import normalize_dtype


GEMM_OPS = ("gemm_rcr", "gemm_rrr")
GEMM_SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")
GEMM_DTYPE_SUFFIXES = {
    "float16": "f16",
    "float32": "f32",
    "bfloat16": "bf16",
}


def cutlass_gemm_symbol(op_name: str, dtype: str) -> str:
    _validate_gemm_op(op_name)
    suffix = gemm_dtype_suffix(dtype)
    return f"dinoml_cutlass_{op_name}_{suffix}"


def cutlass_gemm_profiler_symbol(op_name: str, dtype: str) -> str:
    _validate_gemm_op(op_name)
    suffix = gemm_dtype_suffix(dtype)
    return f"dinoml_profile_cutlass_{op_name}_{suffix}"


def gemm_dtype_suffix(dtype: str) -> str:
    normalized = normalize_dtype(dtype)
    try:
        return GEMM_DTYPE_SUFFIXES[normalized]
    except KeyError as exc:
        supported = ", ".join(GEMM_SUPPORTED_DTYPES)
        raise ValueError(f"Unsupported GEMM dtype {dtype!r}; supported dtypes: {supported}") from exc


def _validate_gemm_op(op_name: str) -> None:
    if op_name not in GEMM_OPS:
        supported = ", ".join(GEMM_OPS)
        raise ValueError(f"Unsupported GEMM op {op_name!r}; supported ops: {supported}")
