from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.ir import canonical_json, dtype_nbytes


CUTLASS_CONV_CANDIDATE_SET_SCHEMA_VERSION = 1
CUTLASS_CONV_USED_CANDIDATE_PLAN_SCHEMA_VERSION = 1
CONV_OPS = ("conv2d_bias",)
CONV_SUPPORTED_DTYPES = ("float16", "float32")
_CUTLASS_CONV_DEFAULT_SYMBOL_ID = "scaffold_sm80_nhwc_ohwi_bias"


def cutlass_conv_symbol(op_name: str, dtype: str, symbol_id: str | None = None) -> str:
    normalized_dtype = _normalize_conv_dtype(dtype)
    symbol_suffix = symbol_id or _CUTLASS_CONV_DEFAULT_SYMBOL_ID
    return f"dinoml_cutlass_{op_name}_{normalized_dtype}_{symbol_suffix}"


def cutlass_conv_profiler_symbol(op_name: str, dtype: str, symbol_id: str | None = None) -> str:
    normalized_dtype = _normalize_conv_dtype(dtype)
    symbol_suffix = symbol_id or _CUTLASS_CONV_DEFAULT_SYMBOL_ID
    return f"dinoml_profile_cutlass_{op_name}_{normalized_dtype}_{symbol_suffix}"


def cutlass_conv_candidate_set_id(op_name: str, dtype: str) -> str:
    normalized_dtype = _normalize_conv_dtype(dtype)
    return f"cutlass_conv_{op_name}_{normalized_dtype}_nhwc_ohwi_bias_v1"


def cutlass_conv_candidates(
    op_name: str,
    dtype: str,
    *,
    target: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    _validate_conv_op_name(op_name)
    normalized_dtype = _normalize_conv_dtype(dtype)
    normalized_target = _normalize_target_policy(target)
    accumulator_dtype = "float32"
    symbol_id = _CUTLASS_CONV_DEFAULT_SYMBOL_ID
    candidate = {
        "candidate_id": f"cutlass_{symbol_id}",
        "candidate_config_key": hashlib.sha256(
            canonical_json(
                {
                    "op": op_name,
                    "dtype": normalized_dtype,
                    "symbol_id": symbol_id,
                    "target_policy": normalized_target,
                }
            ).encode("utf-8")
        ).hexdigest(),
        "kernel_symbol": cutlass_conv_symbol(op_name, normalized_dtype, symbol_id),
        "profiler_symbol": cutlass_conv_profiler_symbol(op_name, normalized_dtype, symbol_id),
        "provider": "cutlass",
        "family": "conv2d_fprop",
        "dtype": normalized_dtype,
        "accumulator_dtype": accumulator_dtype,
        "epilogue": "bias",
        "launch_abi": "dinoml_cutlass_conv2d_bias_v1",
        "layouts": {
            "activation_semantic": "nchw",
            "weight_semantic": "oihw",
            "output_semantic": "nchw",
            "activation_provider": "nhwc",
            "weight_provider": "ohwi",
            "output_provider": "nhwc",
        },
        "cutlass": {
            "opclass": "tensorop" if normalized_dtype == "float16" else "simt",
            "arch": "sm80",
            "iterator_algorithm": "analytic",
            "kind": "manifest_scaffold_only",
        },
        "optional": False,
        "target_policy": normalized_target,
        "status": "manifest_scaffold_only",
    }
    return [candidate]


def cutlass_conv_candidate_set(
    op_name: str,
    dtype: str,
    *,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_conv_op_name(op_name)
    normalized_dtype = _normalize_conv_dtype(dtype)
    candidates = cutlass_conv_candidates(op_name, normalized_dtype, target=target)
    normalized_target = _normalize_target_policy(target)
    config = {
        "schema_version": CUTLASS_CONV_CANDIDATE_SET_SCHEMA_VERSION,
        "provider": "cutlass",
        "family": "conv2d_fprop",
        "op": op_name,
        "dtype": normalized_dtype,
        "accumulator_dtype": "float32",
        "epilogue": "bias",
        "launch_abi": "dinoml_cutlass_conv2d_bias_v1",
        "semantic_layout": {"activation": "nchw", "weight": "oihw", "bias": "o", "output": "nchw"},
        "provider_layout": {"activation": "nhwc", "weight": "ohwi", "bias": "o", "output": "nhwc"},
        "supported_groups": [1],
        "supported_dtypes": list(CONV_SUPPORTED_DTYPES),
        "candidate_count": len(candidates),
        "candidate_config_keys": [str(candidate["candidate_config_key"]) for candidate in candidates],
        "target_policy": normalized_target,
        "status": "manifest_scaffold_only",
    }
    return {
        **config,
        "candidate_set_id": cutlass_conv_candidate_set_id(op_name, normalized_dtype),
        "candidate_set_key": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
    }


def cutlass_conv_used_candidate_plan(kernel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "cutlass_conv":
            continue
        candidates = [dict(candidate) for candidate in item.get("candidates", [])]
        candidate_set = dict(item.get("candidate_set", {}))
        conv_plan = item.get("cutlass_conv_plan")
        conv_plan_payload = dict(conv_plan) if isinstance(conv_plan, Mapping) else {}
        conv_plan_key = hashlib.sha256(canonical_json(conv_plan_payload).encode("utf-8")).hexdigest() if conv_plan_payload else ""
        selected_id = str(item.get("selected_candidate_id", ""))
        selected = next((candidate for candidate in candidates if str(candidate.get("candidate_id")) == selected_id), None)
        candidate_config_key = str(selected.get("candidate_config_key") if selected else "")
        entries.append(
            {
                "op": str(item.get("op", "")),
                "candidate_set_id": str(item.get("candidate_set_id", "")),
                "candidate_set_key": str(item.get("candidate_set_key", "")),
                "candidate_set": candidate_set,
                "selected_candidate_id": selected_id,
                "candidate_config_key": candidate_config_key,
                "kernel_symbol": str(item.get("kernel_symbol", "")),
                "profiler_symbol": str(item.get("profiler_symbol", "")),
                "cutlass_conv_plan": conv_plan_payload,
                "cutlass_conv_plan_key": conv_plan_key,
                "candidate_set": candidate_set,
                "candidates": candidates,
            }
        )
    entries = sorted(entries, key=lambda entry: (entry["op"], entry["candidate_set_id"], entry["kernel_symbol"]))
    candidate_sets = _unique_by_key((entry["candidate_set"] for entry in entries), "candidate_set_key")
    candidates = _unique_by_key((candidate for entry in entries for candidate in entry["candidates"]), "candidate_config_key")
    payload = {
        "schema_version": CUTLASS_CONV_USED_CANDIDATE_PLAN_SCHEMA_VERSION,
        "provider": "cutlass",
        "library": "cutlass_conv",
        "library_name": "cutlass_conv",
        "family": "conv2d_fprop",
        "target": dict(kernel_manifest.get("target", {})),
        "kernel_manifest_cache_key": kernel_manifest.get("cache_key"),
        "support_cache_key": kernel_manifest.get("support_cache_key"),
        "entries": entries,
        "candidate_sets": candidate_sets,
        "candidates": candidates,
        "candidate_set_keys": [str(item.get("candidate_set_key", "")) for item in candidate_sets],
        "candidate_config_keys": [str(item.get("candidate_config_key", "")) for item in candidates],
        "kernel_symbols": sorted({entry["kernel_symbol"] for entry in entries if entry["kernel_symbol"]}),
        "profiler_symbols": sorted({entry["profiler_symbol"] for entry in entries if entry["profiler_symbol"]}),
    }
    payload["used_candidate_plan_key"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def cutlass_conv_layout_plan(
    node: Mapping[str, Any],
    *,
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if str(node.get("op", "")) != "conv2d_bias":
        raise ValueError(f"Unsupported CUTLASS conv scaffold op {node.get('op')!r}")
    x_name, weight_name, bias_name = [str(name) for name in node.get("inputs", ())[:3]]
    output_name = str(node.get("outputs", ("",))[0])
    x_shape = [int(dim) for dim in tensor_map[x_name]["shape"]]
    weight_shape = [int(dim) for dim in tensor_map[weight_name]["shape"]]
    bias_shape = [int(dim) for dim in tensor_map[bias_name]["shape"]]
    output_shape = [int(dim) for dim in tensor_map[output_name]["shape"]]
    dtype = str(tensor_map[output_name]["dtype"])
    dtype_size = dtype_nbytes(dtype)
    stride = [int(item) for item in node.get("attrs", {}).get("stride", (1, 1))]
    padding = [int(item) for item in node.get("attrs", {}).get("padding", (0, 0))]
    dilation = [int(item) for item in node.get("attrs", {}).get("dilation", (1, 1))]
    groups = int(node.get("attrs", {}).get("groups", 1))
    temporary_buffers = [
        {"name": "activation_nhwc", "kind": "layout_pack", "layout": "nhwc", "nbytes": _nbytes(x_shape, dtype_size)},
        {"name": "weight_ohwi", "kind": "layout_pack", "layout": "ohwi", "nbytes": _nbytes(weight_shape, dtype_size)},
        {"name": "output_nhwc", "kind": "layout_pack", "layout": "nhwc", "nbytes": _nbytes(output_shape, dtype_size)},
    ]
    return {
        "schema_version": 1,
        "kind": "cutlass_conv2d_bias_manifest_scaffold",
        "status": "manifest_scaffold_only",
        "blocked_reason": "cutlass_conv_runtime_launcher_not_implemented",
        "node_id": str(node.get("id", "")),
        "op_family": "conv2d_bias",
        "dtype": dtype,
        "semantic_layout": {"activation": "nchw", "weight": "oihw", "bias": "o", "output": "nchw"},
        "provider_layout": {"activation": "nhwc", "weight": "ohwi", "bias": "o", "output": "nhwc"},
        "layout_translation": {
            "input_pack": "nchw_to_nhwc_temporary",
            "output_unpack": "nhwc_to_nchw_temporary",
            "bias": "direct_per_output_channel",
            "input_pack_nbytes": _nbytes(x_shape, dtype_size),
            "output_unpack_nbytes": _nbytes(output_shape, dtype_size),
        },
        "weight_transform": {
            "from": "oihw",
            "to": "ohwi",
            "pack": "oihw_to_ohwi_temporary",
            "temporary_nbytes": _nbytes(weight_shape, dtype_size),
            "runtime_persistent": False,
            "channel_pad_multiple": 1,
            "padded_input_channels": int(weight_shape[1]),
            "padded_output_channels": int(weight_shape[0]),
            "padding_fill_value": 0.0,
        },
        "conv_config": {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        },
        "input_shape": x_shape,
        "weight_shape": weight_shape,
        "bias_shape": bias_shape,
        "output_shape": output_shape,
        "workspace_nbytes": 0,
        "temporary_buffers": temporary_buffers,
        "temporary_nbytes": sum(int(buffer["nbytes"]) for buffer in temporary_buffers),
    }


def _nbytes(shape: list[int], dtype_size: int) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * int(dtype_size)


def _normalize_conv_dtype(dtype: str) -> str:
    normalized = str(dtype)
    if normalized not in CONV_SUPPORTED_DTYPES:
        supported = ", ".join(CONV_SUPPORTED_DTYPES)
        raise ValueError(f"Unsupported CUTLASS conv dtype {dtype!r}; supported dtypes: {supported}")
    return normalized


def _normalize_target_policy(target: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized_target = dict(target or {})
    return {
        "no_tf32": bool(normalized_target.get("no_tf32", False)),
        "use_fp16_acc": bool(normalized_target.get("use_fp16_acc", False)),
    }


def _validate_conv_op_name(op_name: str) -> None:
    if op_name not in CONV_OPS:
        supported = ", ".join(CONV_OPS)
        raise ValueError(f"Unsupported CUTLASS conv op {op_name!r}; supported ops: {supported}")


def _unique_by_key(items, key: str) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for item in items:
        payload = dict(item)
        value = str(payload.get(key, ""))
        if value in seen:
            continue
        seen.add(value)
        unique.append(payload)
    return unique


__all__ = [
    "CONV_OPS",
    "CONV_SUPPORTED_DTYPES",
    "CUTLASS_CONV_CANDIDATE_SET_SCHEMA_VERSION",
    "CUTLASS_CONV_USED_CANDIDATE_PLAN_SCHEMA_VERSION",
    "cutlass_conv_candidate_set",
    "cutlass_conv_candidate_set_id",
    "cutlass_conv_candidates",
    "cutlass_conv_layout_plan",
    "cutlass_conv_profiler_symbol",
    "cutlass_conv_symbol",
    "cutlass_conv_used_candidate_plan",
]
