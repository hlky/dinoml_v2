from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Mapping

from dinoml.ir import canonical_json
from dinoml.kernels.families.gemm import GEMM_SUPPORTED_DTYPES, gemm_op_spec, normalize_gemm_dtype


CK_GEMM_CANDIDATE_SET_SCHEMA_VERSION = 1
CK_GEMM_USED_CANDIDATE_PLAN_SCHEMA_VERSION = 1
CK_GEMM_DEFAULT_SYMBOL_ID = "xdl_custom_v1"
CK_GEMM_TUNED_SYMBOL_ID = "xdl_wide_m_v1"
CK_GEMM_DTYPE_SUFFIXES = {
    "float16": "float16",
    "float32": "float32",
    "bfloat16": "bfloat16",
}
CK_GEMM_DEFAULT_WORKSPACE_NBYTES = 0


CK_GEMM_CONFIGS = (
    {
        "name": "baseline",
        "symbol_id": CK_GEMM_DEFAULT_SYMBOL_ID,
        "config_enum": "kBaseline",
        "priority": 0,
        "tile": {"block_size": 256, "m_per_block": 128, "n_per_block": 128},
        "min_problem": {},
    },
    {
        "name": "wide_m",
        "symbol_id": CK_GEMM_TUNED_SYMBOL_ID,
        "config_enum": "kWideM",
        "priority": 10,
        "tile": {"block_size": 256, "m_per_block": 256, "n_per_block": 128},
        "min_problem": {"m": 128, "n": 64, "k": 32},
    },
)


def ck_gemm_symbol(op_name: str, dtype: str, symbol_id: str | None = None) -> str:
    gemm_op_spec(op_name)
    normalized_dtype = normalize_gemm_dtype(dtype)
    candidate_suffix = symbol_id or CK_GEMM_DEFAULT_SYMBOL_ID
    return f"dinoml_ck_{op_name}_{ck_gemm_dtype_suffix(normalized_dtype)}_{candidate_suffix}"


def ck_gemm_profiler_symbol(op_name: str, dtype: str, symbol_id: str | None = None) -> str:
    gemm_op_spec(op_name)
    normalized_dtype = normalize_gemm_dtype(dtype)
    candidate_suffix = symbol_id or CK_GEMM_DEFAULT_SYMBOL_ID
    return f"dinoml_profile_ck_{op_name}_{ck_gemm_dtype_suffix(normalized_dtype)}_{candidate_suffix}"


def ck_gemm_default_candidate(op_name: str, dtype: str, target: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return ck_gemm_candidates(op_name, dtype, target=target)[0]


def ck_gemm_candidates(
    op_name: str,
    dtype: str,
    target: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    del target
    normalized_dtype = normalize_gemm_dtype(dtype)
    return tuple(_ck_gemm_candidate(op_name, normalized_dtype, config) for config in CK_GEMM_CONFIGS)


def ck_gemm_candidate_set(
    op_name: str,
    dtype: str,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del target
    spec = gemm_op_spec(op_name)
    normalized_dtype = normalize_gemm_dtype(dtype)
    candidates = ck_gemm_candidates(op_name, normalized_dtype)
    config = {
        "schema_version": CK_GEMM_CANDIDATE_SET_SCHEMA_VERSION,
        "candidate_set_id": ck_gemm_candidate_set_id(op_name, normalized_dtype),
        "provider": "ck",
        "family": "gemm_universal",
        "op": op_name,
        "dtype": normalized_dtype,
        "layouts": dict(spec.layouts),
        "epilogue": spec.epilogue.name,
        "epilogue_config": spec.epilogue.to_json(),
        "accumulator_dtypes": sorted({str(candidate["accumulator_dtype"]) for candidate in candidates}),
        "target_policy": {"rocm": True},
        "launch_abi": _ck_gemm_launch_abi(op_name),
        "split_k_values": [1],
        "split_k_default": 1,
        "supports_split_k": False,
        "workspace_nbytes": CK_GEMM_DEFAULT_WORKSPACE_NBYTES,
        "generator": "static_ck_gemm_xdl_custom_candidates_v1",
        "candidate_config_keys": [candidate["candidate_config_key"] for candidate in candidates],
    }
    return {
        **config,
        "candidate_count": len(candidates),
        "candidate_set_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }


def ck_gemm_candidate_set_id(op_name: str, dtype: str) -> str:
    spec = gemm_op_spec(op_name)
    return f"ck_{op_name}_{ck_gemm_dtype_suffix(dtype)}_{spec.epilogue.name}_v1"


def ck_gemm_used_candidate_plan(kernel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "ck_gemm":
            continue
        candidates = [dict(candidate) for candidate in item.get("candidates", [])]
        selected = _selected_candidate(item, candidates)
        candidate_set = dict(item.get("candidate_set", {}))
        entry_config = {
            "op": str(item["op"]),
            "dtype": str(selected.get("dtype") or candidate_set.get("dtype") or ""),
            "kernel_symbol": str(item["kernel_symbol"]),
            "profiler_symbol": item.get("profiler_symbol"),
            "candidate_set_id": item.get("candidate_set_id"),
            "candidate_set_key": item.get("candidate_set_key"),
            "candidate_config_keys": [str(candidate["candidate_config_key"]) for candidate in candidates],
            "kernel_symbols": sorted({str(candidate.get("kernel_symbol") or item["kernel_symbol"]) for candidate in candidates}),
            "profiler_symbols": sorted(
                {
                    str(candidate.get("profiler_symbol") or item.get("profiler_symbol"))
                    for candidate in candidates
                    if candidate.get("profiler_symbol") or item.get("profiler_symbol")
                }
            ),
            "selected_candidate_id": item.get("selected_candidate_id"),
            "selected_candidate": selected,
            "candidate_set": candidate_set,
            "candidates": candidates,
        }
        entries.append(entry_config)
    entries = sorted(entries, key=lambda entry: (entry["op"], entry["dtype"], entry["kernel_symbol"]))
    candidate_sets = _unique_by_key((entry["candidate_set"] for entry in entries), "candidate_set_key")
    candidates = _unique_by_key((candidate for entry in entries for candidate in entry["candidates"]), "candidate_config_key")
    config = {
        "schema_version": CK_GEMM_USED_CANDIDATE_PLAN_SCHEMA_VERSION,
        "provider": "ck",
        "library": "ck_gemm",
        "target": dict(kernel_manifest.get("target", {})),
        "kernel_manifest_cache_key": kernel_manifest.get("cache_key"),
        "support_cache_key": kernel_manifest.get("support_cache_key"),
        "entries": entries,
        "candidate_sets": candidate_sets,
        "candidates": candidates,
        "candidate_set_keys": [item["candidate_set_key"] for item in candidate_sets],
        "candidate_config_keys": [item["candidate_config_key"] for item in candidates],
        "kernel_symbols": sorted({symbol for entry in entries for symbol in entry["kernel_symbols"]}),
        "profiler_symbols": sorted({symbol for entry in entries for symbol in entry["profiler_symbols"]}),
    }
    return {
        **config,
        "used_candidate_plan_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }


def render_ck_gemm_source(source: str, used_candidate_plan: Mapping[str, Any]) -> str:
    if "DINOML_CK_GENERATED_EXPORTS" not in source:
        raise ValueError("CK GEMM source does not contain the generated exports marker")
    symbols = {
        *[str(symbol) for symbol in used_candidate_plan.get("kernel_symbols", [])],
        *[str(symbol) for symbol in used_candidate_plan.get("profiler_symbols", [])],
    }
    if not symbols:
        raise ValueError("CK GEMM used candidate plan does not contain any symbols")
    lines = source.rstrip().splitlines()
    marker = next((index for index, line in enumerate(lines) if "DINOML_CK_GENERATED_EXPORTS" in line), -1)
    if marker < 0:
        raise ValueError("CK GEMM source does not contain the generated exports marker")
    export_lines = []
    available: set[str] = set()
    for candidate in used_candidate_plan.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        export_line = _generated_export_line(candidate)
        line_symbols = _generated_export_symbols(export_line)
        if not line_symbols:
            continue
        export_lines.append(export_line)
        available.update(line_symbols)
    missing = sorted(symbols - available)
    if missing:
        raise ValueError(f"CK GEMM source is missing symbols: {', '.join(missing)}")
    selected = []
    seen = set()
    for line in export_lines:
        line_symbols = _generated_export_symbols(line)
        if not (symbols & line_symbols) or line_symbols & seen:
            continue
        selected.append(line)
        seen.update(line_symbols)
    return "\n".join([*lines[: marker + 1], *selected]) + "\n"


def ck_gemm_dtype_suffix(dtype: str) -> str:
    return CK_GEMM_DTYPE_SUFFIXES[normalize_gemm_dtype(dtype)]


def ck_gemm_static_library_name(op_name: str, dtype: str) -> str:
    normalized = normalize_gemm_dtype(dtype)
    stem = f"dinoml_ck_{op_name}_{normalized}"
    return f"{stem}.lib" if os.name == "nt" else f"lib{stem}.a"


def ck_gemm_cmake_target(op_name: str, dtype: str) -> str:
    normalized = normalize_gemm_dtype(dtype)
    return f"dinoml_ck_gemm_{op_name}_{normalized}"


def _ck_gemm_candidate(op_name: str, dtype: str, kernel_config: Mapping[str, Any]) -> dict[str, Any]:
    spec = gemm_op_spec(op_name)
    symbol_id = str(kernel_config["symbol_id"])
    launch_abi = _ck_gemm_launch_abi(op_name)
    vector_width = _ck_gemm_vector_width(dtype, str(kernel_config["name"]))
    cde_vector_width = _ck_gemm_cde_vector_width(dtype, str(kernel_config["name"]))
    b_alignment_key = "k" if spec.base_layout == "rcr" else "n"
    selection_predicate = {
        "priority": int(kernel_config["priority"]),
        "min_problem": dict(kernel_config.get("min_problem", {})),
        "alignment": {
            "a_k": vector_width,
            f"b_{b_alignment_key}": vector_width,
            "output_n": cde_vector_width,
        },
    }
    ck_config = {
        "api": "device_gemm_multiple_d_xdl_cshuffle",
        "symbol_id": symbol_id,
        "source": "kernels/rocm/src/ck_gemm.hip",
        "mode": "custom_ck_xdl_instances",
        "config": {
            "name": str(kernel_config["name"]),
            "config_enum": str(kernel_config["config_enum"]),
            "tile": dict(kernel_config["tile"]),
            "vector_width": vector_width,
            "cde_vector_width": cde_vector_width,
        },
    }
    config = {
        "candidate_id": f"ck_{op_name}_{dtype}_{symbol_id}",
        "symbol_id": symbol_id,
        "provider": "ck",
        "family": "gemm_universal",
        "op": op_name,
        "dtype": dtype,
        "layouts": dict(spec.layouts),
        "epilogue": spec.epilogue.name,
        "epilogue_config": spec.epilogue.to_json(),
        "accumulator_dtype": "float32",
        "optional": False,
        "launch_abi": launch_abi,
        "split_k_values": [1],
        "split_k_default": 1,
        "supports_split_k": False,
        "workspace_nbytes": CK_GEMM_DEFAULT_WORKSPACE_NBYTES,
        "selection_predicate": selection_predicate,
        "ck": ck_config,
    }
    return {
        **config,
        "kernel_symbol": ck_gemm_symbol(op_name, dtype, symbol_id),
        "profiler_symbol": ck_gemm_profiler_symbol(op_name, dtype, symbol_id),
        "candidate_config_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }


def _ck_gemm_launch_abi(op_name: str) -> str:
    spec = gemm_op_spec(op_name)
    if not spec.epilogue.inputs:
        return "dinoml_ck_gemm_v1"
    residual_count = len(spec.epilogue.residual_inputs)
    if residual_count == 0:
        return "dinoml_ck_gemm_bias_v1"
    if residual_count == 1:
        return "dinoml_ck_gemm_bias_residual_v1"
    if residual_count == 2:
        return "dinoml_ck_gemm_bias_residual2_v1"
    raise ValueError(f"Unsupported CK GEMM epilogue input set for {op_name}: {spec.epilogue.inputs!r}")


def _generated_export_line(candidate: Mapping[str, Any]) -> str:
    op_name = str(candidate["op"])
    dtype = str(candidate["dtype"])
    ctype = _ck_export_ctype(dtype)
    layout_b = "kRcr" if gemm_op_spec(op_name).base_layout == "rcr" else "kRrr"
    epilogue = _ck_epilogue_enum(str(candidate["epilogue"]))
    symbol_id = str(candidate["symbol_id"])
    launch_abi = str(candidate["launch_abi"])
    config_enum = str(candidate.get("ck", {}).get("config", {}).get("config_enum", "kBaseline"))
    if launch_abi == "dinoml_ck_gemm_v1":
        return f"DINOML_CK_GEMM_EXPORT({op_name}, {dtype}, {ctype}, {symbol_id}, {layout_b}, {epilogue}, {config_enum})"
    if launch_abi == "dinoml_ck_gemm_bias_v1":
        return f"DINOML_CK_GEMM_BIAS_EXPORT({op_name}, {dtype}, {ctype}, {symbol_id}, {layout_b}, {epilogue}, {config_enum})"
    if launch_abi == "dinoml_ck_gemm_bias_residual_v1":
        return f"DINOML_CK_GEMM_BIAS_RESIDUAL_EXPORT({op_name}, {dtype}, {ctype}, {symbol_id}, {layout_b}, {epilogue}, {config_enum})"
    if launch_abi == "dinoml_ck_gemm_bias_residual2_v1":
        return f"DINOML_CK_GEMM_BIAS_RESIDUAL2_EXPORT({op_name}, {dtype}, {ctype}, {symbol_id}, {layout_b}, {epilogue}, {config_enum})"
    raise ValueError(f"Unsupported CK GEMM launch ABI: {launch_abi!r}")


def _generated_export_symbols(line: str) -> frozenset[str]:
    stripped = line.strip()
    match = re.match(r"(DINOML_CK_GEMM(?:_BIAS(?:_RESIDUAL2|_RESIDUAL)?)?_EXPORT)\((.*)\)\s*$", stripped)
    if match is None:
        return frozenset()
    args = [arg.strip() for arg in match.group(2).split(",")]
    try:
        op_name, dtype_name, symbol_id = args[0], args[1], args[3]
    except IndexError as exc:
        raise ValueError(f"Malformed CK GEMM generated export: {line[:160]!r}") from exc
    return frozenset(
        {
            f"dinoml_ck_{op_name}_{dtype_name}_{symbol_id}",
            f"dinoml_profile_ck_{op_name}_{dtype_name}_{symbol_id}",
        }
    )


def _ck_export_ctype(dtype: str) -> str:
    if dtype == "float32":
        return "float"
    if dtype == "float16":
        return "half"
    if dtype == "bfloat16":
        return "dinoml::bfloat16"
    raise ValueError(f"Unsupported CK GEMM export dtype: {dtype!r}")


def _ck_gemm_vector_width(dtype: str, config_name: str) -> int:
    if config_name == "baseline":
        return 1
    if dtype == "float32":
        return 4
    return 8


def _ck_gemm_cde_vector_width(dtype: str, config_name: str) -> int:
    if config_name == "baseline":
        return 1
    if dtype == "float32":
        return 2
    return 4


def _ck_epilogue_enum(epilogue: str) -> str:
    aliases = {
        "linear_combination": "kLinearCombination",
        "bias": "kBias",
        "bias_relu": "kBiasRelu",
        "bias_gelu": "kBiasGelu",
        "bias_fast_gelu": "kBiasFastGelu",
        "bias_quick_gelu": "kBiasQuickGelu",
        "bias_sigmoid": "kBiasSigmoid",
        "bias_tanh": "kBiasTanh",
        "bias_swish": "kBiasSwish",
        "bias_hardswish": "kBiasHardSwish",
        "bias_elup1": "kBiasElup1",
        "bias_add": "kBiasAdd",
        "bias_add_add": "kBiasAddAdd",
        "bias_mul": "kBiasMul",
        "bias_mul_add": "kBiasMulAdd",
        "bias_add_relu": "kBiasAddRelu",
        "bias_add_add_relu": "kBiasAddAddRelu",
        "bias_sigmoid_mul": "kBiasSigmoidMul",
        "bias_sigmoid_mul_tanh": "kBiasSigmoidMulTanh",
        "bias_mul_tanh": "kBiasMulTanh",
    }
    try:
        return aliases[epilogue]
    except KeyError as exc:
        raise ValueError(f"Unsupported CK GEMM epilogue: {epilogue!r}") from exc


def _selected_candidate(item: Mapping[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {}
    selected_id = item.get("selected_candidate_id")
    for candidate in candidates:
        if candidate.get("candidate_id") == selected_id:
            return dict(candidate)
    return dict(candidates[0])


def _unique_by_key(items: Any, key: str) -> list[dict[str, Any]]:
    result = {}
    for item in items:
        payload = dict(item)
        item_key = payload.get(key)
        if item_key is not None:
            result[str(item_key)] = payload
    return [result[item_key] for item_key in sorted(result)]


__all__ = [
    "CK_GEMM_CANDIDATE_SET_SCHEMA_VERSION",
    "CK_GEMM_DEFAULT_SYMBOL_ID",
    "CK_GEMM_USED_CANDIDATE_PLAN_SCHEMA_VERSION",
    "ck_gemm_candidate_set",
    "ck_gemm_candidate_set_id",
    "ck_gemm_candidates",
    "ck_gemm_cmake_target",
    "ck_gemm_default_candidate",
    "ck_gemm_profiler_symbol",
    "ck_gemm_static_library_name",
    "ck_gemm_symbol",
    "ck_gemm_used_candidate_plan",
    "render_ck_gemm_source",
]
