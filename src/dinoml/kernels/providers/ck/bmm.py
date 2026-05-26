from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Mapping

from dinoml.ir import canonical_json
from dinoml.kernels.families.bmm import BMM_SUPPORTED_DTYPES, bmm_op_spec, normalize_bmm_dtype


CK_BMM_CANDIDATE_SET_SCHEMA_VERSION = 2
CK_BMM_USED_CANDIDATE_PLAN_SCHEMA_VERSION = 1
CK_BMM_DEFAULT_SYMBOL_ID = "xdl_custom_v1"
CK_BMM_TUNED_SYMBOL_ID = "xdl_wide_m_v1"
CK_BMM_WIDE_N_SYMBOL_ID = "xdl_wide_n_v1"
CK_BMM_SQUARE_SYMBOL_ID = "xdl_square_v1"
CK_BMM_SKINNY_M_SYMBOL_ID = "xdl_skinny_m_v1"
CK_BMM_SKINNY_N_SYMBOL_ID = "xdl_skinny_n_v1"
CK_BMM_SMALL_SYMBOL_ID = "xdl_small_v1"
CK_BMM_DEFAULT_WORKSPACE_NBYTES = 0


CK_BMM_CONFIGS = (
    {
        "name": "baseline",
        "symbol_id": CK_BMM_DEFAULT_SYMBOL_ID,
        "config_enum": "kBaseline",
        "priority": 0,
        "tile": {"block_size": 256, "m_per_block": 128, "n_per_block": 128},
        "min_problem": {},
    },
    {
        "name": "wide_m",
        "symbol_id": CK_BMM_TUNED_SYMBOL_ID,
        "config_enum": "kWideM",
        "priority": 60,
        "tile": {"block_size": 256, "m_per_block": 256, "n_per_block": 128},
        "min_problem": {"m": 128, "n": 64, "k": 32},
    },
    {
        "name": "wide_n",
        "symbol_id": CK_BMM_WIDE_N_SYMBOL_ID,
        "config_enum": "kWideN",
        "priority": 50,
        "tile": {"block_size": 256, "m_per_block": 128, "n_per_block": 256},
        "min_problem": {"m": 64, "n": 128, "k": 32},
    },
    {
        "name": "square",
        "symbol_id": CK_BMM_SQUARE_SYMBOL_ID,
        "config_enum": "kSquare",
        "priority": 40,
        "tile": {"block_size": 256, "m_per_block": 128, "n_per_block": 128},
        "min_problem": {"m": 64, "n": 64, "k": 32},
    },
    {
        "name": "skinny_m",
        "symbol_id": CK_BMM_SKINNY_M_SYMBOL_ID,
        "config_enum": "kSkinnyM",
        "priority": 30,
        "tile": {"block_size": 256, "m_per_block": 64, "n_per_block": 128},
        "min_problem": {"m": 16, "n": 64, "k": 32},
    },
    {
        "name": "skinny_n",
        "symbol_id": CK_BMM_SKINNY_N_SYMBOL_ID,
        "config_enum": "kSkinnyN",
        "priority": 20,
        "tile": {"block_size": 256, "m_per_block": 128, "n_per_block": 64},
        "min_problem": {"m": 64, "n": 16, "k": 32},
    },
    {
        "name": "small",
        "symbol_id": CK_BMM_SMALL_SYMBOL_ID,
        "config_enum": "kSmall",
        "priority": 10,
        "tile": {"block_size": 256, "m_per_block": 64, "n_per_block": 64},
        "min_problem": {"m": 16, "n": 16, "k": 16},
    },
)


def ck_bmm_symbol(op_name: str, dtype: str, symbol_id: str | None = None) -> str:
    bmm_op_spec(op_name)
    normalized_dtype = normalize_bmm_dtype(dtype)
    candidate_suffix = symbol_id or CK_BMM_DEFAULT_SYMBOL_ID
    return f"dinoml_ck_{op_name}_{normalized_dtype}_{candidate_suffix}"


def ck_bmm_profiler_symbol(op_name: str, dtype: str, symbol_id: str | None = None) -> str:
    bmm_op_spec(op_name)
    normalized_dtype = normalize_bmm_dtype(dtype)
    candidate_suffix = symbol_id or CK_BMM_DEFAULT_SYMBOL_ID
    return f"dinoml_profile_ck_{op_name}_{normalized_dtype}_{candidate_suffix}"


def ck_bmm_default_candidate(op_name: str, dtype: str, target: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return ck_bmm_candidates(op_name, dtype, target=target)[0]


def ck_bmm_candidates(
    op_name: str,
    dtype: str,
    target: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    del target
    normalized_dtype = normalize_bmm_dtype(dtype)
    return tuple(_ck_bmm_candidate(op_name, normalized_dtype, config) for config in CK_BMM_CONFIGS)


def ck_bmm_candidate_set(
    op_name: str,
    dtype: str,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del target
    spec = bmm_op_spec(op_name)
    normalized_dtype = normalize_bmm_dtype(dtype)
    candidates = ck_bmm_candidates(op_name, normalized_dtype)
    epilogue_config = _ck_bmm_epilogue_config(spec.epilogue)
    config = {
        "schema_version": CK_BMM_CANDIDATE_SET_SCHEMA_VERSION,
        "candidate_set_id": ck_bmm_candidate_set_id(op_name, normalized_dtype),
        "provider": "ck",
        "family": "bmm_strided",
        "op": op_name,
        "dtype": normalized_dtype,
        "layouts": dict(spec.layouts),
        "epilogue": spec.epilogue,
        "epilogue_config": epilogue_config,
        "accumulator_dtypes": sorted({str(candidate["accumulator_dtype"]) for candidate in candidates}),
        "target_policy": {"rocm": True},
        "launch_abi": epilogue_config["launch_abi"],
        "split_k_values": [1],
        "split_k_default": 1,
        "supports_split_k": False,
        "workspace_nbytes": CK_BMM_DEFAULT_WORKSPACE_NBYTES,
        "generator": "static_ck_bmm_xdl_curated_candidates_v3",
        "candidate_config_keys": [candidate["candidate_config_key"] for candidate in candidates],
    }
    return {
        **config,
        "candidate_count": len(candidates),
        "candidate_set_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }


def ck_bmm_candidate_set_id(op_name: str, dtype: str) -> str:
    spec = bmm_op_spec(op_name)
    epilogue = "linear_combination" if spec.epilogue == "none" else spec.epilogue
    return f"ck_{op_name}_{normalize_bmm_dtype(dtype)}_{epilogue}_v3"


def ck_bmm_used_candidate_plan(kernel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "ck_bmm":
            continue
        all_candidates = [dict(candidate) for candidate in item.get("candidates", [])]
        selected = _selected_candidate(item, all_candidates)
        candidates = _used_candidate_plan_candidates(item, all_candidates, selected)
        candidate_set = dict(item.get("candidate_set", {}))
        execution_plan_selection = item.get("execution_plan_selection")
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
            "pruned_by_execution_plan": isinstance(execution_plan_selection, Mapping),
        }
        if isinstance(execution_plan_selection, Mapping):
            entry_config["execution_plan_selection"] = dict(execution_plan_selection)
        entries.append(entry_config)
    entries = sorted(entries, key=lambda entry: (entry["op"], entry["dtype"], entry["kernel_symbol"]))
    candidate_sets = _unique_by_key((entry["candidate_set"] for entry in entries), "candidate_set_key")
    candidates = _unique_by_key((candidate for entry in entries for candidate in entry["candidates"]), "candidate_config_key")
    config = {
        "schema_version": CK_BMM_USED_CANDIDATE_PLAN_SCHEMA_VERSION,
        "provider": "ck",
        "library": "ck_bmm",
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
        "pruned_by_execution_plan": any(bool(entry.get("pruned_by_execution_plan")) for entry in entries),
    }
    return {
        **config,
        "used_candidate_plan_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }


def _used_candidate_plan_candidates(
    item: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    selected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(item.get("execution_plan_selection"), Mapping):
        return [dict(selected)] if selected else []
    return candidates


def render_ck_bmm_source(source: str, used_candidate_plan: Mapping[str, Any]) -> str:
    if "DINOML_CK_BMM_GENERATED_EXPORTS" not in source:
        raise ValueError("CK BMM source does not contain the generated exports marker")
    symbols = {
        *[str(symbol) for symbol in used_candidate_plan.get("kernel_symbols", [])],
        *[str(symbol) for symbol in used_candidate_plan.get("profiler_symbols", [])],
    }
    if not symbols:
        raise ValueError("CK BMM used candidate plan does not contain any symbols")
    lines = source.rstrip().splitlines()
    marker = next((index for index, line in enumerate(lines) if "DINOML_CK_BMM_GENERATED_EXPORTS" in line), -1)
    if marker < 0:
        raise ValueError("CK BMM source does not contain the generated exports marker")
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
        raise ValueError(f"CK BMM source is missing symbols: {', '.join(missing)}")
    selected = []
    seen = set()
    for line in export_lines:
        line_symbols = _generated_export_symbols(line)
        if not (symbols & line_symbols) or line_symbols & seen:
            continue
        selected.append(line)
        seen.update(line_symbols)
    return "\n".join([*lines[: marker + 1], *selected]) + "\n"


def ck_bmm_static_library_name(op_name: str, dtype: str) -> str:
    normalized = normalize_bmm_dtype(dtype)
    stem = f"dinoml_ck_{op_name}_{normalized}"
    return f"{stem}.lib" if os.name == "nt" else f"lib{stem}.a"


def ck_bmm_cmake_target(op_name: str, dtype: str) -> str:
    normalized = normalize_bmm_dtype(dtype)
    return f"dinoml_ck_bmm_{op_name}_{normalized}"


def _ck_bmm_candidate(op_name: str, dtype: str, kernel_config: Mapping[str, Any]) -> dict[str, Any]:
    spec = bmm_op_spec(op_name)
    symbol_id = str(kernel_config["symbol_id"])
    epilogue_config = _ck_bmm_epilogue_config(spec.epilogue)
    config_name = str(kernel_config["name"])
    tile = {**dict(kernel_config["tile"]), "k_per_block": _ck_bmm_k_per_block(dtype, config_name)}
    vector_width = _ck_bmm_vector_width(dtype, str(kernel_config["name"]))
    cde_vector_width = _ck_bmm_cde_vector_width(dtype, str(kernel_config["name"]))
    a_alignment_key = "k" if spec.a_layout == "r" else "m"
    b_alignment_key = "n" if spec.b_layout == "r" else "k"
    selection_predicate = {
        "priority": int(kernel_config["priority"]),
        "min_problem": dict(kernel_config.get("min_problem", {})),
        "alignment": {
            f"a_{a_alignment_key}": vector_width,
            f"b_{b_alignment_key}": vector_width,
            "output_n": cde_vector_width,
        },
        "padded_block_loop_minimum": {
            "k": {
                "block": int(tile["k_per_block"]),
                "minimum": 3,
            },
        },
    }
    ck_config = {
        "api": "device_batched_gemm_multiple_d_xdl_cshuffle_v3",
        "symbol_id": symbol_id,
        "source": "kernels/rocm/src/ck_bmm.hip",
        "mode": "custom_ck_xdl_instances",
        "config": {
            "name": str(kernel_config["name"]),
            "config_enum": str(kernel_config["config_enum"]),
            "tile": tile,
            "pipeline": "v3",
            "vector_width": vector_width,
            "cde_vector_width": cde_vector_width,
            "gemm_specialization": "mnk_padding",
        },
    }
    config = {
        "candidate_id": f"ck_{op_name}_{dtype}_{symbol_id}",
        "symbol_id": symbol_id,
        "provider": "ck",
        "family": "bmm_strided",
        "op": op_name,
        "dtype": dtype,
        "layouts": dict(spec.layouts),
        "epilogue": spec.epilogue,
        "epilogue_config": epilogue_config,
        "accumulator_dtype": "float32",
        "optional": False,
        "launch_abi": epilogue_config["launch_abi"],
        "split_k_values": [1],
        "split_k_default": 1,
        "supports_split_k": False,
        "workspace_nbytes": CK_BMM_DEFAULT_WORKSPACE_NBYTES,
        "selection_predicate": selection_predicate,
        "ck": ck_config,
    }
    return {
        **config,
        "kernel_symbol": ck_bmm_symbol(op_name, dtype, symbol_id),
        "profiler_symbol": ck_bmm_profiler_symbol(op_name, dtype, symbol_id),
        "candidate_config_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }


def _ck_bmm_epilogue_config(epilogue: str) -> dict[str, Any]:
    if epilogue == "none":
        return {"name": "none", "inputs": [], "launch_abi": "dinoml_ck_bmm_v1"}
    if epilogue == "add":
        return {"name": "add", "inputs": ["d0"], "launch_abi": "dinoml_ck_bmm_add_v1"}
    raise ValueError(f"Unsupported CK BMM epilogue: {epilogue!r}")


def _generated_export_line(candidate: Mapping[str, Any]) -> str:
    op_name = str(candidate["op"])
    dtype = str(candidate["dtype"])
    ctype = _ck_export_ctype(dtype)
    spec = bmm_op_spec(op_name)
    layout = _ck_layout_enum(spec.base_layout)
    epilogue = _ck_epilogue_enum(str(candidate["epilogue"]))
    symbol_id = str(candidate["symbol_id"])
    launch_abi = str(candidate["launch_abi"])
    config_enum = str(candidate.get("ck", {}).get("config", {}).get("config_enum", "kBaseline"))
    if launch_abi == "dinoml_ck_bmm_v1":
        return f"DINOML_CK_BMM_EXPORT({op_name}, {dtype}, {ctype}, {symbol_id}, {layout}, {epilogue}, {config_enum})"
    if launch_abi == "dinoml_ck_bmm_add_v1":
        return f"DINOML_CK_BMM_ADD_EXPORT({op_name}, {dtype}, {ctype}, {symbol_id}, {layout}, {epilogue}, {config_enum})"
    raise ValueError(f"Unsupported CK BMM launch ABI: {launch_abi!r}")


def _generated_export_symbols(line: str) -> frozenset[str]:
    stripped = line.strip()
    match = re.match(r"(DINOML_CK_BMM(?:_ADD)?_EXPORT)\((.*)\)\s*$", stripped)
    if match is None:
        return frozenset()
    args = [arg.strip() for arg in match.group(2).split(",")]
    try:
        op_name, dtype_name, symbol_id = args[0], args[1], args[3]
    except IndexError as exc:
        raise ValueError(f"Malformed CK BMM generated export: {line[:160]!r}") from exc
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
    raise ValueError(f"Unsupported CK BMM export dtype: {dtype!r}")


def _ck_bmm_vector_width(dtype: str, config_name: str) -> int:
    if config_name == "baseline":
        return 1
    if config_name == "small":
        return 2 if dtype == "float32" else 4
    if dtype == "float32":
        return 4
    return 8


def _ck_bmm_k_per_block(dtype: str, config_name: str) -> int:
    if config_name == "baseline":
        return 16 if dtype == "float32" else 32
    if config_name in {"square", "skinny_m", "skinny_n"}:
        return 32 if dtype == "float32" else 64
    if config_name == "small":
        return 16 if dtype == "float32" else 32
    return 32


def _ck_bmm_cde_vector_width(dtype: str, config_name: str) -> int:
    if config_name == "baseline":
        return 1
    if dtype == "float32":
        return 2
    return 4


def _ck_layout_enum(layout: str) -> str:
    supported = {"ccc", "ccr", "crc", "crr", "rcc", "rcr", "rrc", "rrr"}
    if layout not in supported:
        raise ValueError(f"Unsupported CK BMM layout: {layout!r}")
    return "k" + layout.capitalize()


def _ck_epilogue_enum(epilogue: str) -> str:
    if epilogue == "none":
        return "kLinearCombination"
    if epilogue == "add":
        return "kAdd"
    raise ValueError(f"Unsupported CK BMM epilogue: {epilogue!r}")


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
    "BMM_SUPPORTED_DTYPES",
    "CK_BMM_CANDIDATE_SET_SCHEMA_VERSION",
    "CK_BMM_DEFAULT_SYMBOL_ID",
    "CK_BMM_USED_CANDIDATE_PLAN_SCHEMA_VERSION",
    "ck_bmm_candidate_set",
    "ck_bmm_candidate_set_id",
    "ck_bmm_candidates",
    "ck_bmm_cmake_target",
    "ck_bmm_default_candidate",
    "ck_bmm_profiler_symbol",
    "ck_bmm_static_library_name",
    "ck_bmm_symbol",
    "ck_bmm_used_candidate_plan",
    "render_ck_bmm_source",
]
