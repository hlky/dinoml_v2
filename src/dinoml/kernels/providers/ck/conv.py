from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Mapping

from dinoml.ir import canonical_json


CK_CONV_CANDIDATE_SET_SCHEMA_VERSION = 2
CK_CONV_USED_CANDIDATE_PLAN_SCHEMA_VERSION = 1
CK_CONV_DEFAULT_SYMBOL_ID = "xdl_custom_v1"
CK_CONV_TUNED_SYMBOL_ID = "xdl_wide_n_v1"
CK_CONV_WIDE_M_SYMBOL_ID = "xdl_wide_m_v1"
CK_CONV_SQUARE_SYMBOL_ID = "xdl_square_v1"
CK_CONV_SKINNY_M_SYMBOL_ID = "xdl_skinny_m_v1"
CK_CONV_SKINNY_N_SYMBOL_ID = "xdl_skinny_n_v1"
CK_CONV_SMALL_SYMBOL_ID = "xdl_small_v1"
CK_CONV_OPS = ("conv2d_bias",)
CK_CONV_SUPPORTED_DTYPES = ("float16", "float32")
CK_CONV_DEFAULT_WORKSPACE_NBYTES = 0


CK_CONV_CONFIGS = (
    {
        "name": "baseline",
        "symbol_id": CK_CONV_DEFAULT_SYMBOL_ID,
        "config_enum": "kBaseline",
        "priority": 0,
        "tile": {"block_size": 256, "m_per_block": 128, "n_per_block": 128},
        "min_problem": {},
    },
    {
        "name": "wide_n",
        "symbol_id": CK_CONV_TUNED_SYMBOL_ID,
        "config_enum": "kWideN",
        "priority": 60,
        "tile": {"block_size": 256, "m_per_block": 128, "n_per_block": 256},
        "min_problem": {"gemm_m": 128, "out_channels": 64},
    },
    {
        "name": "wide_m",
        "symbol_id": CK_CONV_WIDE_M_SYMBOL_ID,
        "config_enum": "kWideM",
        "priority": 50,
        "tile": {"block_size": 256, "m_per_block": 256, "n_per_block": 128},
        "min_problem": {"gemm_m": 256, "out_channels": 32},
    },
    {
        "name": "square",
        "symbol_id": CK_CONV_SQUARE_SYMBOL_ID,
        "config_enum": "kSquare",
        "priority": 40,
        "tile": {"block_size": 256, "m_per_block": 128, "n_per_block": 128},
        "min_problem": {"gemm_m": 64, "out_channels": 32},
    },
    {
        "name": "skinny_m",
        "symbol_id": CK_CONV_SKINNY_M_SYMBOL_ID,
        "config_enum": "kSkinnyM",
        "priority": 30,
        "tile": {"block_size": 256, "m_per_block": 64, "n_per_block": 128},
        "min_problem": {"gemm_m": 16, "out_channels": 64},
    },
    {
        "name": "skinny_n",
        "symbol_id": CK_CONV_SKINNY_N_SYMBOL_ID,
        "config_enum": "kSkinnyN",
        "priority": 20,
        "tile": {"block_size": 256, "m_per_block": 128, "n_per_block": 64},
        "min_problem": {"gemm_m": 64, "out_channels": 16},
    },
    {
        "name": "small",
        "symbol_id": CK_CONV_SMALL_SYMBOL_ID,
        "config_enum": "kSmall",
        "priority": 10,
        "tile": {"block_size": 256, "m_per_block": 64, "n_per_block": 64},
        "min_problem": {"gemm_m": 16, "out_channels": 16},
    },
)


def ck_conv_symbol(op_name: str, dtype: str, symbol_id: str | None = None) -> str:
    _validate_ck_conv_op(op_name)
    normalized_dtype = _normalize_ck_conv_dtype(dtype)
    candidate_suffix = symbol_id or CK_CONV_DEFAULT_SYMBOL_ID
    return f"dinoml_ck_{op_name}_{normalized_dtype}_{candidate_suffix}"


def ck_conv_profiler_symbol(op_name: str, dtype: str, symbol_id: str | None = None) -> str:
    _validate_ck_conv_op(op_name)
    normalized_dtype = _normalize_ck_conv_dtype(dtype)
    candidate_suffix = symbol_id or CK_CONV_DEFAULT_SYMBOL_ID
    return f"dinoml_profile_ck_{op_name}_{normalized_dtype}_{candidate_suffix}"


def ck_conv_default_candidate(
    op_name: str,
    dtype: str,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return ck_conv_candidates(op_name, dtype, target=target)[0]


def ck_conv_candidates(
    op_name: str,
    dtype: str,
    target: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    del target
    normalized_dtype = _normalize_ck_conv_dtype(dtype)
    return tuple(_ck_conv_candidate(op_name, normalized_dtype, config) for config in CK_CONV_CONFIGS)


def ck_conv_candidate_set(
    op_name: str,
    dtype: str,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del target
    _validate_ck_conv_op(op_name)
    normalized_dtype = _normalize_ck_conv_dtype(dtype)
    candidates = ck_conv_candidates(op_name, normalized_dtype)
    epilogue_config = _ck_conv_epilogue_config(op_name)
    config = {
        "schema_version": CK_CONV_CANDIDATE_SET_SCHEMA_VERSION,
        "candidate_set_id": ck_conv_candidate_set_id(op_name, normalized_dtype),
        "provider": "ck",
        "family": "conv2d_fprop",
        "op": op_name,
        "dtype": normalized_dtype,
        "accumulator_dtype": "float32",
        "epilogue": epilogue_config["name"],
        "epilogue_config": epilogue_config,
        "semantic_layout": {
            "activation": "nchw",
            "weight": "oihw",
            "bias": "o",
            "output": "nchw",
        },
        "provider_layout": {
            "activation": "g_nhw_c_strided",
            "weight": "g_k_yx_c_strided",
            "bias": "g_k",
            "output": "g_nhw_k_strided",
        },
        "supported_groups": [1],
        "supported_dtypes": list(CK_CONV_SUPPORTED_DTYPES),
        "target_policy": {"rocm": True},
        "launch_abi": epilogue_config["launch_abi"],
        "split_k_values": [1],
        "split_k_default": 1,
        "supports_split_k": False,
        "workspace_nbytes": CK_CONV_DEFAULT_WORKSPACE_NBYTES,
        "generator": "static_ck_conv2d_xdl_curated_candidates_v3",
        "candidate_config_keys": [candidate["candidate_config_key"] for candidate in candidates],
    }
    return {
        **config,
        "candidate_count": len(candidates),
        "candidate_set_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }


def ck_conv_candidate_set_id(op_name: str, dtype: str) -> str:
    _validate_ck_conv_op(op_name)
    return f"ck_{op_name}_{_normalize_ck_conv_dtype(dtype)}_bias_v3"


def ck_conv_used_candidate_plan(kernel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "ck_conv":
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
        "schema_version": CK_CONV_USED_CANDIDATE_PLAN_SCHEMA_VERSION,
        "provider": "ck",
        "library": "ck_conv",
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


def render_ck_conv_source(source: str, used_candidate_plan: Mapping[str, Any]) -> str:
    if "DINOML_CK_CONV_GENERATED_EXPORTS" not in source:
        raise ValueError("CK Conv source does not contain the generated exports marker")
    symbols = {
        *[str(symbol) for symbol in used_candidate_plan.get("kernel_symbols", [])],
        *[str(symbol) for symbol in used_candidate_plan.get("profiler_symbols", [])],
    }
    if not symbols:
        raise ValueError("CK Conv used candidate plan does not contain any symbols")
    lines = source.rstrip().splitlines()
    marker = next((index for index, line in enumerate(lines) if "DINOML_CK_CONV_GENERATED_EXPORTS" in line), -1)
    if marker < 0:
        raise ValueError("CK Conv source does not contain the generated exports marker")
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
        raise ValueError(f"CK Conv source is missing symbols: {', '.join(missing)}")
    selected = []
    seen = set()
    for line in export_lines:
        line_symbols = _generated_export_symbols(line)
        if not (symbols & line_symbols) or line_symbols & seen:
            continue
        selected.append(line)
        seen.update(line_symbols)
    return "\n".join([*lines[: marker + 1], *selected]) + "\n"


def ck_conv_static_library_name(op_name: str, dtype: str) -> str:
    _validate_ck_conv_op(op_name)
    normalized = _normalize_ck_conv_dtype(dtype)
    stem = f"dinoml_ck_{op_name}_{normalized}"
    return f"{stem}.lib" if os.name == "nt" else f"lib{stem}.a"


def ck_conv_cmake_target(op_name: str, dtype: str) -> str:
    _validate_ck_conv_op(op_name)
    normalized = _normalize_ck_conv_dtype(dtype)
    return f"dinoml_ck_conv_{op_name}_{normalized}"


def _ck_conv_candidate(op_name: str, dtype: str, kernel_config: Mapping[str, Any]) -> dict[str, Any]:
    _validate_ck_conv_op(op_name)
    symbol_id = str(kernel_config["symbol_id"])
    epilogue_config = _ck_conv_epilogue_config(op_name)
    config_name = str(kernel_config["name"])
    tile = {**dict(kernel_config["tile"]), "k_per_block": _ck_conv_k_per_block(dtype, config_name)}
    vector_width = _ck_conv_vector_width(dtype, config_name)
    cde_vector_width = _ck_conv_cde_vector_width(dtype, config_name)
    selection_predicate = {
        "priority": int(kernel_config["priority"]),
        "exact": {
            "groups": 1,
        },
        "min_problem": dict(kernel_config.get("min_problem", {})),
        "alignment": {
            "in_channels": vector_width,
            "out_channels": cde_vector_width,
        },
    }
    ck_config = {
        "api": "device_grouped_conv_fwd_multiple_abd_xdl_cshuffle",
        "symbol_id": symbol_id,
        "source": "kernels/rocm/src/ck_conv.hip",
        "mode": "custom_ck_xdl_instances",
        "config": {
            "name": str(kernel_config["name"]),
            "config_enum": str(kernel_config["config_enum"]),
            "tile": tile,
            "pipeline": "v1",
            "num_gemm_k_prefetch_stage": 1,
            "vector_width": vector_width,
            "cde_vector_width": cde_vector_width,
            "gemm_specialization": "mnk_padding",
        },
    }
    config = {
        "candidate_id": f"ck_{op_name}_{dtype}_{symbol_id}",
        "symbol_id": symbol_id,
        "provider": "ck",
        "family": "conv2d_fprop",
        "op": op_name,
        "dtype": dtype,
        "accumulator_dtype": "float32",
        "epilogue": epilogue_config["name"],
        "epilogue_config": epilogue_config,
        "semantic_layout": {
            "activation": "nchw",
            "weight": "oihw",
            "bias": "o",
            "output": "nchw",
        },
        "provider_layout": {
            "activation": "g_nhw_c_strided",
            "weight": "g_k_yx_c_strided",
            "bias": "g_k",
            "output": "g_nhw_k_strided",
        },
        "supported_groups": [1],
        "optional": False,
        "launch_abi": epilogue_config["launch_abi"],
        "split_k_values": [1],
        "split_k_default": 1,
        "supports_split_k": False,
        "workspace_nbytes": CK_CONV_DEFAULT_WORKSPACE_NBYTES,
        "selection_predicate": selection_predicate,
        "ck": ck_config,
    }
    return {
        **config,
        "kernel_symbol": ck_conv_symbol(op_name, dtype, symbol_id),
        "profiler_symbol": ck_conv_profiler_symbol(op_name, dtype, symbol_id),
        "candidate_config_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }


def _ck_conv_epilogue_config(op_name: str) -> dict[str, Any]:
    _validate_ck_conv_op(op_name)
    return {"name": "bias", "inputs": ["bias"], "launch_abi": "dinoml_ck_conv2d_bias_v1"}


def _generated_export_line(candidate: Mapping[str, Any]) -> str:
    op_name = str(candidate["op"])
    dtype = str(candidate["dtype"])
    ctype = _ck_export_ctype(dtype)
    symbol_id = str(candidate["symbol_id"])
    launch_abi = str(candidate["launch_abi"])
    config_enum = str(candidate.get("ck", {}).get("config", {}).get("config_enum", "kBaseline"))
    if launch_abi == "dinoml_ck_conv2d_bias_v1":
        return f"DINOML_CK_CONV2D_BIAS_EXPORT({op_name}, {dtype}, {ctype}, {symbol_id}, {config_enum})"
    raise ValueError(f"Unsupported CK Conv launch ABI: {launch_abi!r}")


def _generated_export_symbols(line: str) -> frozenset[str]:
    stripped = line.strip()
    match = re.match(r"DINOML_CK_CONV2D_BIAS_EXPORT\((.*)\)\s*$", stripped)
    if match is None:
        return frozenset()
    args = [arg.strip() for arg in match.group(1).split(",")]
    try:
        op_name, dtype_name, symbol_id = args[0], args[1], args[3]
    except IndexError as exc:
        raise ValueError(f"Malformed CK Conv generated export: {line[:160]!r}") from exc
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
    raise ValueError(f"Unsupported CK Conv export dtype: {dtype!r}")


def _ck_conv_vector_width(dtype: str, config_name: str) -> int:
    if config_name == "baseline":
        return 1
    if config_name == "small":
        return 2 if dtype == "float32" else 4
    if dtype == "float32":
        return 4
    return 8


def _ck_conv_k_per_block(dtype: str, config_name: str) -> int:
    if config_name == "wide_n":
        return 16 if dtype == "float32" else 32
    if config_name in {"square", "skinny_m", "skinny_n"}:
        return 32 if dtype == "float32" else 64
    if config_name == "small":
        return 16 if dtype == "float32" else 32
    return 32


def _ck_conv_cde_vector_width(dtype: str, config_name: str) -> int:
    if config_name == "baseline":
        return 1
    if dtype == "float32":
        return 4
    return 4


def _normalize_ck_conv_dtype(dtype: str) -> str:
    normalized = str(dtype)
    if normalized not in CK_CONV_SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported CK Conv dtype {dtype!r}; supported dtypes: {', '.join(CK_CONV_SUPPORTED_DTYPES)}"
        )
    return normalized


def _validate_ck_conv_op(op_name: str) -> None:
    if op_name not in CK_CONV_OPS:
        raise ValueError(f"Unsupported CK Conv op {op_name!r}")


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
    "CK_CONV_CANDIDATE_SET_SCHEMA_VERSION",
    "CK_CONV_DEFAULT_SYMBOL_ID",
    "CK_CONV_OPS",
    "CK_CONV_SUPPORTED_DTYPES",
    "CK_CONV_USED_CANDIDATE_PLAN_SCHEMA_VERSION",
    "ck_conv_candidate_set",
    "ck_conv_candidate_set_id",
    "ck_conv_candidates",
    "ck_conv_cmake_target",
    "ck_conv_default_candidate",
    "ck_conv_profiler_symbol",
    "ck_conv_static_library_name",
    "ck_conv_symbol",
    "ck_conv_used_candidate_plan",
    "render_ck_conv_source",
]
