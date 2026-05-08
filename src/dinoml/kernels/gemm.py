from __future__ import annotations

import hashlib
from typing import Any

from dinoml.ir import canonical_json, normalize_dtype


GEMM_OPS = ("gemm_rcr", "gemm_rrr")
GEMM_SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")
GEMM_DTYPE_SUFFIXES = {
    "float16": "f16",
    "float32": "f32",
    "bfloat16": "bf16",
}
CUTLASS_DEFAULT_CANDIDATE_ID = "cutlass_default"
CUTLASS_DEFAULT_SYMBOL_ID = "default"
CUTLASS_GEMM_LAUNCH_ABI = "dinoml_cutlass_gemm_v1"
CUTLASS_GEMM_CANDIDATE_SET_SCHEMA_VERSION = 1


def cutlass_gemm_symbol(op_name: str, dtype: str) -> str:
    _validate_gemm_op(op_name)
    suffix = gemm_dtype_suffix(dtype)
    return f"dinoml_cutlass_{op_name}_{suffix}"


def cutlass_gemm_profiler_symbol(op_name: str, dtype: str) -> str:
    _validate_gemm_op(op_name)
    suffix = gemm_dtype_suffix(dtype)
    return f"dinoml_profile_cutlass_{op_name}_{suffix}"


def cutlass_gemm_default_candidate(op_name: str, dtype: str) -> dict[str, Any]:
    _validate_gemm_op(op_name)
    normalized_dtype = normalize_dtype(dtype)
    kernel_symbol = cutlass_gemm_symbol(op_name, normalized_dtype)
    profiler_symbol = cutlass_gemm_profiler_symbol(op_name, normalized_dtype)
    config = {
        "candidate_id": CUTLASS_DEFAULT_CANDIDATE_ID,
        "symbol_id": CUTLASS_DEFAULT_SYMBOL_ID,
        "provider": "cutlass",
        "family": "gemm_universal",
        "op": op_name,
        "dtype": normalized_dtype,
        "layouts": {
            "a": "row",
            "b": "row" if op_name == "gemm_rrr" else "column",
            "c": "row",
        },
        "epilogue": "linear_combination",
        "accumulator_dtype": "float32",
        "launch_abi": CUTLASS_GEMM_LAUNCH_ABI,
        "cutlass": {
            "api": "device_gemm_default",
            "threadblock": None,
            "warp": None,
            "instruction": None,
            "stages": None,
            "align": None,
        },
    }
    candidate = {
        **config,
        "kernel_symbol": kernel_symbol,
        "profiler_symbol": profiler_symbol,
        "candidate_config_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }
    return candidate


def cutlass_gemm_candidates(op_name: str, dtype: str) -> tuple[dict[str, Any], ...]:
    return (cutlass_gemm_default_candidate(op_name, dtype),)


def cutlass_gemm_candidate_set(op_name: str, dtype: str) -> dict[str, Any]:
    _validate_gemm_op(op_name)
    normalized_dtype = normalize_dtype(dtype)
    candidates = cutlass_gemm_candidates(op_name, normalized_dtype)
    config = {
        "schema_version": CUTLASS_GEMM_CANDIDATE_SET_SCHEMA_VERSION,
        "candidate_set_id": cutlass_gemm_candidate_set_id(op_name, normalized_dtype),
        "provider": "cutlass",
        "family": "gemm_universal",
        "op": op_name,
        "dtype": normalized_dtype,
        "layouts": {
            "a": "row",
            "b": "row" if op_name == "gemm_rrr" else "column",
            "c": "row",
        },
        "epilogue": "linear_combination",
        "accumulator_dtype": "float32",
        "launch_abi": CUTLASS_GEMM_LAUNCH_ABI,
        "generator": "static_default_v1",
        "candidate_config_keys": [candidate["candidate_config_key"] for candidate in candidates],
    }
    return {
        **config,
        "candidate_count": len(candidates),
        "candidate_set_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }


def cutlass_gemm_candidate_set_id(op_name: str, dtype: str) -> str:
    _validate_gemm_op(op_name)
    suffix = gemm_dtype_suffix(dtype)
    return f"cutlass_{op_name}_{suffix}_linear_combination_v1"


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
