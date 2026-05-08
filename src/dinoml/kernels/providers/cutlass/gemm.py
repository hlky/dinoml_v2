from __future__ import annotations

import hashlib
from typing import Any

from dinoml.ir import canonical_json
from dinoml.kernels.families.gemm import GEMM_SUPPORTED_DTYPES, gemm_op_spec, normalize_gemm_dtype


GEMM_DTYPE_SUFFIXES = {
    "float16": "f16",
    "float32": "f32",
    "bfloat16": "bf16",
}
CUTLASS_DEFAULT_CANDIDATE_ID = "cutlass_default"
CUTLASS_DEFAULT_SYMBOL_ID = "default"
CUTLASS_GEMM_CANDIDATE_SET_SCHEMA_VERSION = 1


def cutlass_gemm_symbol(op_name: str, dtype: str) -> str:
    gemm_op_spec(op_name)
    suffix = gemm_dtype_suffix(dtype)
    return f"dinoml_cutlass_{op_name}_{suffix}"


def cutlass_gemm_profiler_symbol(op_name: str, dtype: str) -> str:
    gemm_op_spec(op_name)
    suffix = gemm_dtype_suffix(dtype)
    return f"dinoml_profile_cutlass_{op_name}_{suffix}"


def cutlass_gemm_default_candidate(op_name: str, dtype: str) -> dict[str, Any]:
    spec = gemm_op_spec(op_name)
    normalized_dtype = normalize_gemm_dtype(dtype)
    kernel_symbol = cutlass_gemm_symbol(op_name, normalized_dtype)
    profiler_symbol = cutlass_gemm_profiler_symbol(op_name, normalized_dtype)
    epilogue = spec.epilogue.to_json()
    config = {
        "candidate_id": CUTLASS_DEFAULT_CANDIDATE_ID,
        "symbol_id": CUTLASS_DEFAULT_SYMBOL_ID,
        "provider": "cutlass",
        "family": "gemm_universal",
        "op": op_name,
        "dtype": normalized_dtype,
        "layouts": dict(spec.layouts),
        "epilogue": spec.epilogue.name,
        "epilogue_config": epilogue,
        "accumulator_dtype": spec.epilogue.accumulator_dtype,
        "launch_abi": spec.epilogue.launch_abi,
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
    spec = gemm_op_spec(op_name)
    normalized_dtype = normalize_gemm_dtype(dtype)
    candidates = cutlass_gemm_candidates(op_name, normalized_dtype)
    config = {
        "schema_version": CUTLASS_GEMM_CANDIDATE_SET_SCHEMA_VERSION,
        "candidate_set_id": cutlass_gemm_candidate_set_id(op_name, normalized_dtype),
        "provider": "cutlass",
        "family": "gemm_universal",
        "op": op_name,
        "dtype": normalized_dtype,
        "layouts": dict(spec.layouts),
        "epilogue": spec.epilogue.name,
        "epilogue_config": spec.epilogue.to_json(),
        "accumulator_dtype": spec.epilogue.accumulator_dtype,
        "launch_abi": spec.epilogue.launch_abi,
        "generator": "static_default_v1",
        "candidate_config_keys": [candidate["candidate_config_key"] for candidate in candidates],
    }
    return {
        **config,
        "candidate_count": len(candidates),
        "candidate_set_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }


def cutlass_gemm_candidate_set_id(op_name: str, dtype: str) -> str:
    spec = gemm_op_spec(op_name)
    suffix = gemm_dtype_suffix(dtype)
    return f"cutlass_{op_name}_{suffix}_{spec.epilogue.name}_v1"


def gemm_dtype_suffix(dtype: str) -> str:
    normalized = normalize_gemm_dtype(dtype)
    return GEMM_DTYPE_SUFFIXES[normalized]


__all__ = [
    "GEMM_SUPPORTED_DTYPES",
    "CUTLASS_DEFAULT_CANDIDATE_ID",
    "CUTLASS_DEFAULT_SYMBOL_ID",
    "CUTLASS_GEMM_CANDIDATE_SET_SCHEMA_VERSION",
    "cutlass_gemm_candidate_set",
    "cutlass_gemm_candidate_set_id",
    "cutlass_gemm_candidates",
    "cutlass_gemm_default_candidate",
    "cutlass_gemm_profiler_symbol",
    "cutlass_gemm_symbol",
    "gemm_dtype_suffix",
]
