from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from dinoml.ir import canonical_json
from dinoml.kernels.families.gemm import GEMM_SUPPORTED_DTYPES, gemm_op_spec, normalize_gemm_dtype


GEMM_DTYPE_SUFFIXES = {
    "float16": "float16",
    "float32": "float32",
    "bfloat16": "bfloat16",
}
CUTLASS_GEMM_CANDIDATE_SET_SCHEMA_VERSION = 1
CUTLASS_GEMM_USED_CANDIDATE_PLAN_SCHEMA_VERSION = 1
CUTLASS_SM80_TENSOROP_16816_TILES = (
    ((256, 128, 32), 3, (4, 2, 1)),
    ((128, 256, 32), 3, (2, 4, 1)),
    ((256, 64, 32), 3, (4, 1, 1)),
    ((256, 64, 32), 4, (4, 1, 1)),
    ((64, 256, 32), 4, (1, 4, 1)),
    ((128, 128, 32), 3, (2, 2, 1)),
    ((128, 128, 32), 4, (2, 2, 1)),
    ((128, 128, 32), 5, (2, 2, 1)),
    ((128, 64, 32), 6, (2, 2, 1)),
    ((64, 128, 32), 6, (2, 2, 1)),
    ((64, 64, 32), 10, (2, 2, 1)),
    ((256, 128, 64), 3, (4, 2, 1)),
    ((128, 256, 64), 3, (2, 4, 1)),
    ((256, 64, 64), 4, (4, 1, 1)),
    ((64, 256, 64), 4, (1, 4, 1)),
    ((128, 128, 64), 4, (2, 2, 1)),
    ((256, 64, 64), 3, (4, 1, 1)),
    ((64, 256, 64), 3, (1, 4, 1)),
    ((128, 128, 64), 3, (2, 2, 1)),
    ((128, 64, 64), 3, (2, 2, 1)),
    ((64, 128, 64), 3, (2, 2, 1)),
    ((64, 64, 64), 5, (2, 2, 1)),
    ((256, 64, 32), 2, (4, 1, 1)),
    ((64, 256, 32), 2, (1, 4, 1)),
    ((192, 128, 32), 3, (4, 2, 1)),
    ((128, 192, 32), 3, (4, 2, 1)),
    ((192, 128, 32), 4, (4, 2, 1)),
    ((128, 192, 32), 4, (4, 2, 1)),
    ((160, 128, 32), 3, (4, 2, 1)),
    ((128, 160, 32), 3, (4, 2, 1)),
    ((160, 128, 32), 4, (4, 2, 1)),
    ((128, 160, 32), 4, (4, 2, 1)),
    ((224, 128, 32), 3, (4, 2, 1)),
    ((128, 224, 32), 3, (2, 4, 1)),
    ((224, 128, 32), 4, (4, 2, 1)),
    ((128, 224, 32), 4, (2, 4, 1)),
    ((192, 160, 32), 3, (4, 2, 1)),
    ((160, 192, 32), 3, (2, 4, 1)),
    ((192, 160, 32), 4, (4, 2, 1)),
    ((160, 192, 32), 4, (2, 4, 1)),
    ((256, 96, 32), 3, (4, 2, 1)),
    ((96, 256, 32), 3, (2, 4, 1)),
    ((256, 96, 32), 2, (4, 2, 1)),
    ((96, 256, 32), 2, (2, 4, 1)),
    ((192, 96, 32), 3, (4, 2, 1)),
    ((96, 192, 32), 3, (2, 4, 1)),
)
CUTLASS_SM80_TENSOROP_1688_TILES = (
    ((256, 128, 16), 3, (4, 2, 1)),
    ((128, 256, 16), 3, (2, 4, 1)),
    ((256, 64, 16), 4, (4, 1, 1)),
    ((64, 256, 16), 4, (1, 4, 1)),
    ((128, 128, 16), 5, (2, 2, 1)),
    ((128, 128, 16), 4, (2, 2, 1)),
    ((128, 128, 16), 3, (2, 2, 1)),
    ((128, 64, 16), 6, (2, 2, 1)),
    ((64, 128, 16), 6, (2, 2, 1)),
    ((64, 64, 16), 10, (2, 2, 1)),
    ((256, 128, 32), 3, (4, 2, 1)),
    ((128, 256, 32), 3, (2, 4, 1)),
    ((256, 64, 32), 4, (4, 1, 1)),
    ((64, 256, 32), 4, (1, 4, 1)),
    ((128, 128, 32), 4, (2, 2, 1)),
    ((128, 128, 32), 3, (2, 2, 1)),
    ((128, 64, 32), 3, (2, 2, 1)),
    ((64, 128, 32), 3, (2, 2, 1)),
    ((64, 64, 32), 5, (2, 2, 1)),
)
CUTLASS_SM80_TENSOROP_16816_ALIGNMENTS = (8, 4, 2)
CUTLASS_SM80_TENSOROP_TF32_ALIGNMENTS = (4, 2, 1)


def _cutlass_symbol_id(
    threadblock: tuple[int, int, int],
    stages: int,
    warp_count: tuple[int, int, int],
    align: int,
    accumulator_dtype: str,
    math: str,
) -> str:
    tb = "x".join(str(dim) for dim in threadblock)
    wc = "x".join(str(dim) for dim in warp_count)
    accumulator = accumulator_dtype.replace("float", "f").replace("bfloat", "bf")
    return f"tensorop_sm80_{math}_{tb}_s{stages}_w{wc}_{accumulator}_align{align}"


def _cutlass_candidate_config(
    threadblock: tuple[int, int, int],
    stages: int,
    warp_count: tuple[int, int, int],
    align: int,
    *,
    dtype: str,
    accumulator_dtype: str,
    instruction: tuple[int, int, int],
    math: str,
    optional: bool = False,
) -> dict[str, Any]:
    symbol_id = _cutlass_symbol_id(threadblock, stages, warp_count, align, accumulator_dtype, math)
    return {
        "candidate_id": f"cutlass_{symbol_id}",
        "symbol_id": symbol_id,
        "dtype": dtype,
        "accumulator_dtype": accumulator_dtype,
        "optional": optional,
        "cutlass": {
            "api": "device_gemm",
            "opclass": "tensorop",
            "arch": "sm80",
            "math": math,
            "threadblock": list(threadblock),
            "warp_count": list(warp_count),
            "warp": [int(threadblock[index] // warp_count[index]) for index in range(3)],
            "instruction": list(instruction),
            "stages": stages,
            "align": align,
        },
    }


def _cutlass_sm80_tensorop_16816_candidate_configs() -> tuple[dict[str, Any], ...]:
    configs = []
    for threadblock, stages, warp_count in CUTLASS_SM80_TENSOROP_16816_TILES:
        for align in CUTLASS_SM80_TENSOROP_16816_ALIGNMENTS:
            configs.append(
                _cutlass_candidate_config(
                    threadblock,
                    stages,
                    warp_count,
                    align,
                    dtype="float16",
                    accumulator_dtype="float32",
                    instruction=(16, 8, 16),
                    math="16816",
                )
            )
            configs.append(
                _cutlass_candidate_config(
                    threadblock,
                    stages,
                    warp_count,
                    align,
                    dtype="float16",
                    accumulator_dtype="float16",
                    instruction=(16, 8, 16),
                    math="16816",
                )
            )
            configs.append(
                _cutlass_candidate_config(
                    threadblock,
                    stages,
                    warp_count,
                    align,
                    dtype="bfloat16",
                    accumulator_dtype="float32",
                    instruction=(16, 8, 16),
                    math="16816",
                )
            )
    return tuple(configs)


def _cutlass_sm80_tensorop_tf32_candidate_configs() -> tuple[dict[str, Any], ...]:
    return tuple(
        _cutlass_candidate_config(
            threadblock,
            stages,
            warp_count,
            align,
            dtype="float32",
            accumulator_dtype="float32",
            instruction=(16, 8, 8),
            math="tf32",
            optional=True,
        )
        for threadblock, stages, warp_count in CUTLASS_SM80_TENSOROP_1688_TILES
        for align in CUTLASS_SM80_TENSOROP_TF32_ALIGNMENTS
    )


CUTLASS_GEMM_CANDIDATE_CONFIGS = (
    *_cutlass_sm80_tensorop_16816_candidate_configs(),
    *_cutlass_sm80_tensorop_tf32_candidate_configs(),
)
CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE = {
    dtype: tuple(config for config in CUTLASS_GEMM_CANDIDATE_CONFIGS if config["dtype"] == dtype)
    for dtype in GEMM_SUPPORTED_DTYPES
}
CUTLASS_DEFAULT_SYMBOL_ID = str(CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE["float16"][0]["symbol_id"])
CUTLASS_DEFAULT_CANDIDATE_ID = str(CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE["float16"][0]["candidate_id"])


def cutlass_gemm_symbol(op_name: str, dtype: str, symbol_id: str | None = None) -> str:
    gemm_op_spec(op_name)
    normalized_dtype = normalize_gemm_dtype(dtype)
    suffix = gemm_dtype_suffix(normalized_dtype)
    default_symbol_id = str(CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE[normalized_dtype][0]["symbol_id"])
    candidate_suffix = f"_{symbol_id}" if symbol_id else f"_{default_symbol_id}"
    return f"dinoml_cutlass_{op_name}_{suffix}{candidate_suffix}"


def cutlass_gemm_profiler_symbol(op_name: str, dtype: str, symbol_id: str | None = None) -> str:
    gemm_op_spec(op_name)
    normalized_dtype = normalize_gemm_dtype(dtype)
    suffix = gemm_dtype_suffix(normalized_dtype)
    default_symbol_id = str(CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE[normalized_dtype][0]["symbol_id"])
    candidate_suffix = f"_{symbol_id}" if symbol_id else f"_{default_symbol_id}"
    return f"dinoml_profile_cutlass_{op_name}_{suffix}{candidate_suffix}"


def cutlass_gemm_default_candidate(op_name: str, dtype: str, target: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return cutlass_gemm_candidates(op_name, dtype, target=target)[0]


def _cutlass_gemm_candidate(op_name: str, dtype: str, candidate_config: Mapping[str, Any]) -> dict[str, Any]:
    spec = gemm_op_spec(op_name)
    normalized_dtype = normalize_gemm_dtype(dtype)
    symbol_id = str(candidate_config["symbol_id"])
    kernel_symbol = cutlass_gemm_symbol(op_name, normalized_dtype, symbol_id)
    profiler_symbol = cutlass_gemm_profiler_symbol(op_name, normalized_dtype, symbol_id)
    epilogue = spec.epilogue.to_json()
    cutlass_config = dict(candidate_config["cutlass"])
    config = {
        "candidate_id": str(candidate_config["candidate_id"]),
        "symbol_id": str(candidate_config["symbol_id"]),
        "provider": "cutlass",
        "family": "gemm_universal",
        "op": op_name,
        "dtype": normalized_dtype,
        "layouts": dict(spec.layouts),
        "epilogue": spec.epilogue.name,
        "epilogue_config": epilogue,
        "accumulator_dtype": str(candidate_config["accumulator_dtype"]),
        "optional": bool(candidate_config.get("optional", False)),
        "launch_abi": spec.epilogue.launch_abi,
        "cutlass": cutlass_config,
    }
    candidate = {
        **config,
        "kernel_symbol": kernel_symbol,
        "profiler_symbol": profiler_symbol,
        "candidate_config_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }
    return candidate


def cutlass_gemm_candidates(
    op_name: str,
    dtype: str,
    target: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    normalized_dtype = normalize_gemm_dtype(dtype)
    return tuple(
        _cutlass_gemm_candidate(op_name, normalized_dtype, config)
        for config in _cutlass_candidate_configs_for_target(normalized_dtype, target)
    )


def cutlass_gemm_candidate_set(
    op_name: str,
    dtype: str,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = gemm_op_spec(op_name)
    normalized_dtype = normalize_gemm_dtype(dtype)
    candidates = cutlass_gemm_candidates(op_name, normalized_dtype, target=target)
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
        "accumulator_dtypes": sorted({str(candidate["accumulator_dtype"]) for candidate in candidates}),
        "target_policy": cutlass_gemm_target_policy(target),
        "launch_abi": spec.epilogue.launch_abi,
        "generator": "static_cutlass_gemm_candidates_v1",
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


def cutlass_gemm_target_policy(target: Mapping[str, Any] | None) -> dict[str, bool]:
    return {
        "no_tf32": bool((target or {}).get("no_tf32", False)),
        "use_fp16_acc": bool((target or {}).get("use_fp16_acc", False)),
    }


def _cutlass_candidate_configs_for_target(
    dtype: str,
    target: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    configs: tuple[Mapping[str, Any], ...] = CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE[dtype]
    if target is None:
        return configs
    policy = cutlass_gemm_target_policy(target)
    if dtype == "float32" and policy["no_tf32"]:
        configs = tuple(config for config in configs if config["cutlass"]["math"] != "tf32")
        if not configs:
            raise NotImplementedError(
                "CUTLASS GEMM no_tf32=True requires non-TF32 float32 candidates; "
                "DinoML v2 does not generate those candidates yet"
            )
    if dtype == "float16":
        accumulator_dtype = "float16" if policy["use_fp16_acc"] else "float32"
        configs = tuple(config for config in configs if config["accumulator_dtype"] == accumulator_dtype)
    return configs


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
    if "DINOML_CUTLASS_GENERATED_EXPORTS" in source:
        return _render_generated_cutlass_gemm_source(source, used_candidate_plan)
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


def _render_generated_cutlass_gemm_source(source: str, used_candidate_plan: Mapping[str, Any]) -> str:
    symbols = {
        *[str(symbol) for symbol in used_candidate_plan.get("kernel_symbols", [])],
        *[str(symbol) for symbol in used_candidate_plan.get("profiler_symbols", [])],
    }
    if not symbols:
        raise ValueError("CUTLASS GEMM used candidate plan does not contain any symbols")
    lines = source.rstrip().splitlines()
    try:
        first_export = next(index for index, line in enumerate(lines) if line.startswith("DINOML_FORWARD_GEMM"))
    except StopIteration as exc:
        raise ValueError("CUTLASS GEMM generated source does not contain export invocations") from exc
    generated_lines = lines[first_export:]
    available = {symbol for line in generated_lines for symbol in _generated_export_symbols(line)}
    missing = sorted(symbols - available)
    if missing:
        raise ValueError(f"CUTLASS GEMM source is missing symbols: {', '.join(missing)}")
    selected = []
    seen = set()
    for line in generated_lines:
        line_symbols = _generated_export_symbols(line)
        if not line_symbols or not (symbols & line_symbols) or line in seen:
            continue
        selected.append(line)
        seen.add(line)
    return "\n".join([*lines[:first_export], "", *selected]) + "\n"


def _generated_export_symbols(line: str) -> frozenset[str]:
    stripped = line.strip()
    if not stripped.startswith("DINOML_FORWARD_GEMM"):
        return frozenset()
    match = re.match(r"(DINOML_FORWARD_GEMM(?:_BIAS(?:_ACTIVATION)?)?_EXPORT)\((.*)\)\s*$", stripped)
    if match is None:
        return frozenset()
    macro = match.group(1)
    args = [arg.strip() for arg in match.group(2).split(",")]
    try:
        if macro in {"DINOML_FORWARD_GEMM_EXPORT", "DINOML_FORWARD_GEMM_BIAS_EXPORT"}:
            op_name, dtype_name, symbol_id = args[0], args[1], args[5]
        else:
            op_name, dtype_name, symbol_id = args[0], args[1], args[7]
    except IndexError as exc:
        raise ValueError(f"Malformed CUTLASS GEMM generated export: {line[:160]!r}") from exc
    return frozenset(
        {
            f"dinoml_cutlass_{op_name}_{dtype_name}_{symbol_id}",
            f"dinoml_profile_cutlass_{op_name}_{dtype_name}_{symbol_id}",
        }
    )


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
    "CUTLASS_GEMM_CANDIDATE_CONFIGS",
    "CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE",
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
