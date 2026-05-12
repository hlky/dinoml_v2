from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from dinoml.ir import canonical_json
from dinoml.kernels.families.bmm import BMM_SUPPORTED_DTYPES, bmm_op_spec, normalize_bmm_dtype
from dinoml.kernels.providers.cutlass.gemm import (
    CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE,
    _cutlass_candidate_config_buildable_for_layouts,
    cutlass_gemm_target_policy,
    gemm_dtype_suffix,
)


CUTLASS_BMM_CANDIDATE_SET_SCHEMA_VERSION = 1
CUTLASS_BMM_USED_CANDIDATE_PLAN_SCHEMA_VERSION = 1
CUTLASS_BMM_SPLIT_K_VALUES = (1,)
CUTLASS_BMM_DEFAULT_SPLIT_K = 1
CUTLASS_BMM_DEFAULT_WORKSPACE_NBYTES = 0


def cutlass_bmm_symbol(op_name: str, dtype: str) -> str:
    normalized_dtype = normalize_bmm_dtype(dtype)
    default_symbol_id = str(CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE[normalized_dtype][0]["symbol_id"])
    return f"dinoml_cutlass_{op_name}_{normalized_dtype}_{default_symbol_id}"


def cutlass_bmm_profiler_symbol(op_name: str, dtype: str) -> str:
    normalized_dtype = normalize_bmm_dtype(dtype)
    default_symbol_id = str(CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE[normalized_dtype][0]["symbol_id"])
    return f"dinoml_profile_cutlass_{op_name}_{normalized_dtype}_{default_symbol_id}"


def cutlass_bmm_candidates(
    op_name: str,
    dtype: str,
    target: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    normalized_dtype = normalize_bmm_dtype(dtype)
    return tuple(
        _cutlass_bmm_candidate(op_name, normalized_dtype, config)
        for config in _cutlass_candidate_configs_for_target(op_name, normalized_dtype, target)
    )


def cutlass_bmm_candidate_set(
    op_name: str,
    dtype: str,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = bmm_op_spec(op_name)
    normalized_dtype = normalize_bmm_dtype(dtype)
    candidates = cutlass_bmm_candidates(op_name, normalized_dtype, target=target)
    epilogue_config = _cutlass_bmm_epilogue_config(spec.epilogue)
    config = {
        "schema_version": CUTLASS_BMM_CANDIDATE_SET_SCHEMA_VERSION,
        "candidate_set_id": cutlass_bmm_candidate_set_id(op_name, normalized_dtype),
        "provider": "cutlass",
        "family": "bmm_strided",
        "op": op_name,
        "dtype": normalized_dtype,
        "layouts": dict(spec.layouts),
        "epilogue": spec.epilogue,
        "epilogue_config": epilogue_config,
        "accumulator_dtypes": sorted({str(candidate["accumulator_dtype"]) for candidate in candidates}),
        "target_policy": cutlass_gemm_target_policy(target),
        "launch_abi": epilogue_config["launch_abi"],
        "split_k_values": list(CUTLASS_BMM_SPLIT_K_VALUES),
        "split_k_default": CUTLASS_BMM_DEFAULT_SPLIT_K,
        "supports_split_k": False,
        "workspace_nbytes": CUTLASS_BMM_DEFAULT_WORKSPACE_NBYTES,
        "generator": "static_cutlass_bmm_candidates_v1",
        "candidate_config_keys": [candidate["candidate_config_key"] for candidate in candidates],
    }
    return {
        **config,
        "candidate_count": len(candidates),
        "candidate_set_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }


def cutlass_bmm_candidate_set_id(op_name: str, dtype: str) -> str:
    suffix = gemm_dtype_suffix(dtype)
    spec = bmm_op_spec(op_name)
    epilogue = "linear_combination" if spec.epilogue == "none" else spec.epilogue
    return f"cutlass_{op_name}_{suffix}_{epilogue}_v1"


def cutlass_bmm_used_candidate_plan(kernel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "cutlass_bmm":
            continue
        candidates = [dict(candidate) for candidate in item.get("candidates", [])]
        selected = _selected_candidate(item, candidates)
        candidate_set = dict(item.get("candidate_set", {}))
        dtype = str(selected.get("dtype") or candidate_set.get("dtype") or "")
        entry_config = {
            "op": str(item["op"]),
            "dtype": dtype,
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
            "alignment_fallbacks": [
                dict(fallback)
                for fallback in item.get("alignment_fallbacks", [])
                if isinstance(fallback, Mapping)
            ],
            "cutlass_alignment": (
                dict(item["cutlass_alignment"])
                if isinstance(item.get("cutlass_alignment"), Mapping)
                else None
            ),
            "candidate_set": candidate_set,
            "candidates": candidates,
        }
        entries.append(entry_config)
    entries = sorted(entries, key=lambda entry: (entry["op"], entry["dtype"], entry["kernel_symbol"]))
    candidate_sets = _unique_by_key((entry["candidate_set"] for entry in entries), "candidate_set_key")
    candidates = _unique_by_key((candidate for entry in entries for candidate in entry["candidates"]), "candidate_config_key")
    config = {
        "schema_version": CUTLASS_BMM_USED_CANDIDATE_PLAN_SCHEMA_VERSION,
        "provider": "cutlass",
        "library": "cutlass_bmm",
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


def render_cutlass_bmm_source(source: str, used_candidate_plan: Mapping[str, Any]) -> str:
    symbols = {
        *[str(symbol) for symbol in used_candidate_plan.get("kernel_symbols", [])],
        *[str(symbol) for symbol in used_candidate_plan.get("profiler_symbols", [])],
    }
    if not symbols:
        raise ValueError("CUTLASS BMM used candidate plan does not contain any symbols")
    lines = source.rstrip().splitlines()
    first_export = next((index for index, line in enumerate(lines) if line.startswith("DINOML_FORWARD_BMM")), len(lines))
    generated_lines = lines[first_export:]
    available = {symbol for line in generated_lines for symbol in _generated_export_symbols(line)}
    dynamic_policy_aliases = []
    dynamic_export_lines = []
    existing_source = "\n".join(lines)
    for candidate in used_candidate_plan.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        export_line = _generated_export_line(candidate)
        export_symbols = _generated_export_symbols(export_line)
        if not export_symbols:
            continue
        policy_name = str(candidate["cutlass_policy"])
        if policy_name not in existing_source and policy_name not in "\n".join(dynamic_policy_aliases):
            dynamic_policy_aliases.append(_generated_policy_alias(candidate))
        dynamic_export_lines.append(export_line)
        available.update(export_symbols)
    missing = sorted(symbols - available)
    if missing:
        raise ValueError(f"CUTLASS BMM source is missing symbols: {', '.join(missing)}")
    selected = []
    seen = set()
    for line in [*dynamic_export_lines, *generated_lines]:
        line_symbols = _generated_export_symbols(line)
        if not line_symbols or not (symbols & line_symbols) or line_symbols & seen:
            continue
        selected.append(line)
        seen.update(line_symbols)
    return "\n".join([*lines[:first_export], *dynamic_policy_aliases, "", *selected]) + "\n"


def _cutlass_bmm_candidate(
    op_name: str,
    dtype: str,
    candidate_config: Mapping[str, Any],
) -> dict[str, Any]:
    spec = bmm_op_spec(op_name)
    symbol_id = str(candidate_config["symbol_id"])
    kernel_symbol = f"dinoml_cutlass_{op_name}_{dtype}_{symbol_id}"
    profiler_symbol = f"dinoml_profile_cutlass_{op_name}_{dtype}_{symbol_id}"
    cutlass_config = {**dict(candidate_config["cutlass"]), "api": "device_gemm_batched"}
    if spec.epilogue == "add":
        cutlass_config["epilogue_source"] = "d0"
    epilogue_config = _cutlass_bmm_epilogue_config(spec.epilogue)
    config = {
        "candidate_id": f"cutlass_{op_name}_{dtype}_{symbol_id}",
        "symbol_id": symbol_id,
        "provider": "cutlass",
        "family": "bmm_strided",
        "op": op_name,
        "dtype": dtype,
        "layouts": dict(spec.layouts),
        "epilogue": spec.epilogue,
        "epilogue_config": epilogue_config,
        "accumulator_dtype": str(candidate_config["accumulator_dtype"]),
        "cutlass_policy": str(candidate_config["cutlass_policy"]),
        "optional": bool(candidate_config.get("optional", False)),
        "launch_abi": epilogue_config["launch_abi"],
        "split_k_values": list(CUTLASS_BMM_SPLIT_K_VALUES),
        "split_k_default": CUTLASS_BMM_DEFAULT_SPLIT_K,
        "supports_split_k": False,
        "workspace_nbytes": CUTLASS_BMM_DEFAULT_WORKSPACE_NBYTES,
        "cutlass": cutlass_config,
    }
    return {
        **config,
        "kernel_symbol": kernel_symbol,
        "profiler_symbol": profiler_symbol,
        "candidate_config_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }


def _cutlass_candidate_configs_for_target(
    op_name: str,
    dtype: str,
    target: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    spec = bmm_op_spec(op_name)
    configs: tuple[Mapping[str, Any], ...] = tuple(
        config
        for config in CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE[dtype]
        if _cutlass_candidate_config_buildable_for_layouts(
            config,
            a_layout=spec.layouts["a"],
            b_layout=spec.layouts["b"],
        )
    )
    if target is None:
        return configs
    policy = cutlass_gemm_target_policy(target)
    if dtype == "float32" and policy["no_tf32"]:
        configs = tuple(config for config in configs if config["cutlass"]["opclass"] != "tensorop")
    if dtype == "float16":
        accumulator_dtype = "float16" if policy["use_fp16_acc"] else "float32"
        configs = tuple(config for config in configs if config["accumulator_dtype"] == accumulator_dtype)
    return configs


def _generated_policy_alias(candidate: Mapping[str, Any]) -> str:
    cutlass_config = candidate["cutlass"]
    threadblock = [int(dim) for dim in cutlass_config["threadblock"]]
    warp = [int(dim) for dim in cutlass_config["warp"]]
    instruction = [int(dim) for dim in cutlass_config["instruction"]]
    opclass = "cutlass::arch::OpClassSimt" if cutlass_config["opclass"] == "simt" else "cutlass::arch::OpClassTensorOp"
    accumulator = _cutlass_cpp_element(str(candidate["accumulator_dtype"]))
    align = int(cutlass_config["align"])
    math_operator = _cutlass_math_operator_cpp(str(cutlass_config.get("math_operator", "multiply_add")))
    return (
        f"using {candidate['cutlass_policy']} = GemmPolicy<\n"
        f"    {opclass},\n"
        "    cutlass::arch::Sm80,\n"
        f"    cutlass::gemm::GemmShape<{threadblock[0]}, {threadblock[1]}, {threadblock[2]}>,\n"
        f"    cutlass::gemm::GemmShape<{warp[0]}, {warp[1]}, {warp[2]}>,\n"
        f"    cutlass::gemm::GemmShape<{instruction[0]}, {instruction[1]}, {instruction[2]}>,\n"
        f"    {accumulator},\n"
        f"    {int(cutlass_config['stages'])},\n"
        f"    {align},\n"
        f"    {align},\n"
        f"    {math_operator}>;"
    )


def _generated_export_line(candidate: Mapping[str, Any]) -> str:
    op_name = str(candidate["op"])
    dtype = str(candidate["dtype"])
    ctype, element = _cutlass_export_dtype_args(dtype)
    symbol_id = str(candidate["symbol_id"])
    policy = str(candidate["cutlass_policy"])
    spec = bmm_op_spec(op_name)
    layout_a = _cutlass_layout_cpp(spec.a_layout)
    layout_b = _cutlass_layout_cpp(spec.b_layout)
    layout_c = _cutlass_layout_cpp(spec.c_layout)
    if str(candidate.get("epilogue")) == "add":
        return (
            f"DINOML_FORWARD_BMM_ADD_EXPORT({op_name}, {dtype}, {ctype}, {element}, "
            f"{layout_a}, {layout_b}, {layout_c}, {symbol_id}, {policy})"
        )
    return (
        f"DINOML_FORWARD_BMM_EXPORT({op_name}, {dtype}, {ctype}, {element}, "
        f"{layout_a}, {layout_b}, {layout_c}, {symbol_id}, {policy})"
    )


def _generated_export_symbols(line: str) -> frozenset[str]:
    stripped = line.strip()
    if not stripped.startswith("DINOML_FORWARD_BMM"):
        return frozenset()
    match = re.match(r"(DINOML_FORWARD_BMM(?:_ADD)?_EXPORT)\((.*)\)\s*$", stripped)
    if match is None:
        return frozenset()
    args = [arg.strip() for arg in match.group(2).split(",")]
    try:
        op_name, dtype_name, symbol_id = args[0], args[1], args[7]
    except IndexError as exc:
        raise ValueError(f"Malformed CUTLASS BMM generated export: {line[:160]!r}") from exc
    return frozenset(
        {
            f"dinoml_cutlass_{op_name}_{dtype_name}_{symbol_id}",
            f"dinoml_profile_cutlass_{op_name}_{dtype_name}_{symbol_id}",
        }
    )


def _cutlass_bmm_epilogue_config(epilogue: str) -> dict[str, Any]:
    if epilogue == "none":
        return {"name": "none", "inputs": [], "launch_abi": "dinoml_cutlass_bmm_v1"}
    if epilogue == "add":
        return {"name": "add", "inputs": ["d0"], "launch_abi": "dinoml_cutlass_bmm_add_v1"}
    raise ValueError(f"Unsupported CUTLASS BMM epilogue: {epilogue!r}")


def _cutlass_export_dtype_args(dtype: str) -> tuple[str, str]:
    if dtype == "float32":
        return "float", "float"
    if dtype == "float16":
        return "half", "cutlass::half_t"
    if dtype == "bfloat16":
        return "__nv_bfloat16", "cutlass::bfloat16_t"
    raise ValueError(f"Unsupported CUTLASS BMM export dtype: {dtype!r}")


def _cutlass_cpp_element(dtype: str) -> str:
    if dtype == "float32":
        return "float"
    if dtype == "float16":
        return "cutlass::half_t"
    if dtype == "bfloat16":
        return "cutlass::bfloat16_t"
    raise ValueError(f"Unsupported CUTLASS BMM element dtype: {dtype!r}")


def _cutlass_layout_cpp(layout: str) -> str:
    if layout == "r":
        return "cutlass::layout::RowMajor"
    if layout == "c":
        return "cutlass::layout::ColumnMajor"
    raise ValueError(f"Unsupported CUTLASS BMM layout marker: {layout!r}")


def _cutlass_math_operator_cpp(math_operator: str) -> str:
    if math_operator == "multiply_add":
        return "cutlass::arch::OpMultiplyAdd"
    if math_operator == "multiply_add_fast_f16":
        return "cutlass::arch::OpMultiplyAddFastF16"
    if math_operator == "multiply_add_fast_bf16":
        return "cutlass::arch::OpMultiplyAddFastBF16"
    if math_operator == "multiply_add_fast_f32":
        return "cutlass::arch::OpMultiplyAddFastF32"
    raise ValueError(f"Unsupported CUTLASS BMM math operator: {math_operator!r}")


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
    "CUTLASS_BMM_CANDIDATE_SET_SCHEMA_VERSION",
    "CUTLASS_BMM_USED_CANDIDATE_PLAN_SCHEMA_VERSION",
    "cutlass_bmm_candidate_set",
    "cutlass_bmm_candidate_set_id",
    "cutlass_bmm_candidates",
    "cutlass_bmm_profiler_symbol",
    "cutlass_bmm_symbol",
    "cutlass_bmm_used_candidate_plan",
    "render_cutlass_bmm_source",
]
