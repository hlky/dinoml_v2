from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

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
CUTLASS_GEMM_USED_CANDIDATE_PLAN_SCHEMA_VERSION = 1


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


def cutlass_gemm_used_candidate_plan(kernel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "cutlass_gemm":
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
            "candidate_set": candidate_set,
            "candidates": candidates,
        }
        entries.append(entry_config)
    entries = sorted(entries, key=lambda entry: (entry["op"], entry["dtype"], entry["kernel_symbol"]))
    candidate_sets = _unique_by_key((entry["candidate_set"] for entry in entries), "candidate_set_key")
    candidates = _unique_by_key((candidate for entry in entries for candidate in entry["candidates"]), "candidate_config_key")
    config = {
        "schema_version": CUTLASS_GEMM_USED_CANDIDATE_PLAN_SCHEMA_VERSION,
        "provider": "cutlass",
        "library": "cutlass_gemm",
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


def render_cutlass_gemm_source(source: str, used_candidate_plan: Mapping[str, Any]) -> str:
    symbols = {
        *[str(symbol) for symbol in used_candidate_plan.get("kernel_symbols", [])],
        *[str(symbol) for symbol in used_candidate_plan.get("profiler_symbols", [])],
    }
    if not symbols:
        raise ValueError("CUTLASS GEMM used candidate plan does not contain any symbols")
    blocks = _extern_c_blocks(source)
    available = {name for name, _ in blocks}
    missing = sorted(symbols - available)
    if missing:
        raise ValueError(f"CUTLASS GEMM source is missing symbols: {', '.join(missing)}")
    first_extern = source.find('extern "C"')
    if first_extern < 0:
        raise ValueError("CUTLASS GEMM source does not contain extern C exports")
    prefix = source[:first_extern].rstrip()
    selected_blocks = [block.strip() for name, block in blocks if name in symbols]
    return "\n\n".join([prefix, *selected_blocks]) + "\n"


def gemm_dtype_suffix(dtype: str) -> str:
    normalized = normalize_gemm_dtype(dtype)
    return GEMM_DTYPE_SUFFIXES[normalized]


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


def _extern_c_blocks(source: str) -> list[tuple[str, str]]:
    blocks = []
    position = 0
    while True:
        start = source.find('extern "C"', position)
        if start < 0:
            break
        brace = source.find("{", start)
        if brace < 0:
            raise ValueError("Malformed CUTLASS GEMM source: missing function body")
        signature = source[start:brace]
        match = re.search(r'extern\s+"C"\s+(?:int|float)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', signature)
        if match is None:
            raise ValueError(f"Malformed CUTLASS GEMM source signature: {signature[:120]!r}")
        depth = 0
        end = brace
        while end < len(source):
            char = source[end]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1
        if depth != 0:
            raise ValueError("Malformed CUTLASS GEMM source: unterminated function body")
        blocks.append((match.group(1), source[start:end]))
        position = end
    return blocks


__all__ = [
    "GEMM_SUPPORTED_DTYPES",
    "CUTLASS_DEFAULT_CANDIDATE_ID",
    "CUTLASS_DEFAULT_SYMBOL_ID",
    "CUTLASS_GEMM_CANDIDATE_SET_SCHEMA_VERSION",
    "CUTLASS_GEMM_USED_CANDIDATE_PLAN_SCHEMA_VERSION",
    "cutlass_gemm_candidate_set",
    "cutlass_gemm_candidate_set_id",
    "cutlass_gemm_candidates",
    "cutlass_gemm_default_candidate",
    "cutlass_gemm_profiler_symbol",
    "cutlass_gemm_symbol",
    "cutlass_gemm_used_candidate_plan",
    "gemm_dtype_suffix",
    "render_cutlass_gemm_source",
]
