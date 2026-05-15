from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.ir import canonical_json, dtype_nbytes


CUTLASS_CONV_CANDIDATE_SET_SCHEMA_VERSION = 1
CUTLASS_CONV_USED_CANDIDATE_PLAN_SCHEMA_VERSION = 1
CONV_OPS = ("conv2d_bias",)
CONV_SUPPORTED_DTYPES = ("float16", "float32")
_CUTLASS_CONV_SIMT_SYMBOL_ID = "simt_sm80_nhwc_ohwi_bias"
_CUTLASS_CONV_FEW_CHANNELS_SYMBOL_ID = "tensorop_sm80_nhwc_ohwi_bias_few_channels_c3"
_CUTLASS_CONV_FIXED_CHANNELS_SYMBOL_IDS = {
    4: "tensorop_sm80_nhwc_ohwi_bias_fixed_channels_c4",
    8: "tensorop_sm80_nhwc_ohwi_bias_fixed_channels_c8",
}
_CUTLASS_CONV_OPTIMIZED_SYMBOL_ID = "tensorop_sm80_nhwc_ohwi_bias_optimized_align8"
_CUTLASS_CONV_DEFAULT_SYMBOL_ID = _CUTLASS_CONV_SIMT_SYMBOL_ID
_CUTLASS_CONV_SCAFFOLD_KIND = "cutlass_conv2d_bias_manifest_scaffold"
_CUTLASS_CONV_SCAFFOLD_STATUS = "manifest_scaffold_only"
_CUTLASS_CONV_RUNTIME_STATUS = "bounded_runtime"
_CUTLASS_CONV_SCAFFOLD_BLOCKED_REASON = "cutlass_conv_runtime_launcher_not_implemented"
_CUTLASS_CONV_PROFILER_BLOCKED_REASON = "cutlass_conv_profiler_not_implemented"
_CUTLASS_CONV_RUNTIME_PROFILER_STATUS = "bounded_runtime_profiler"
_CUTLASS_CONV_TRANSFORM_ABI = "dinoml_cutlass_layout_transform_v1"
_CUTLASS_CONV_STUB_RETURN_CODE = 901
_CUTLASS_CONV_STUB_PROFILE_MS = -1.0
_CUTLASS_CONV_SCAFFOLD_SEMANTIC_LAYOUT = {
    "activation": "nchw",
    "weight": "oihw",
    "bias": "o",
    "output": "nchw",
}
_CUTLASS_CONV_SCAFFOLD_PROVIDER_LAYOUT = {
    "activation": "nhwc",
    "weight": "ohwi",
    "bias": "o",
    "output": "nhwc",
}


def cutlass_conv_symbol(op_name: str, dtype: str, symbol_id: str | None = None) -> str:
    normalized_dtype = _normalize_conv_dtype(dtype)
    symbol_suffix = symbol_id or _CUTLASS_CONV_DEFAULT_SYMBOL_ID
    return f"dinoml_cutlass_{op_name}_{normalized_dtype}_{symbol_suffix}"


def cutlass_conv_profiler_symbol(op_name: str, dtype: str, symbol_id: str | None = None) -> str:
    normalized_dtype = _normalize_conv_dtype(dtype)
    symbol_suffix = symbol_id or _CUTLASS_CONV_DEFAULT_SYMBOL_ID
    return f"dinoml_profile_cutlass_{op_name}_{normalized_dtype}_{symbol_suffix}"


def cutlass_conv_input_pack_symbol(dtype: str) -> str:
    normalized_dtype = _normalize_conv_dtype(dtype)
    return f"dinoml_cutlass_conv_input_pack_nchw_to_nhwc_{normalized_dtype}_v1"


def cutlass_conv_weight_pack_symbol(dtype: str) -> str:
    normalized_dtype = _normalize_conv_dtype(dtype)
    return f"dinoml_cutlass_conv_weight_pack_oihw_to_ohwi_{normalized_dtype}_v1"


def cutlass_conv_output_unpack_symbol(dtype: str) -> str:
    normalized_dtype = _normalize_conv_dtype(dtype)
    return f"dinoml_cutlass_conv_output_unpack_nhwc_to_nchw_{normalized_dtype}_v1"


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
    status = _conv_candidate_status(normalized_dtype)
    candidates = [
        _cutlass_conv_candidate(
            op_name,
            normalized_dtype,
            symbol_id=_CUTLASS_CONV_SIMT_SYMBOL_ID,
            target_policy=normalized_target,
            status=status,
            accumulator_dtype="float32",
            cutlass={
                "opclass": "simt",
                "arch": "sm80",
                "iterator_algorithm": "analytic",
                "instruction_shape": [1, 1, 1],
                "threadblock": [128, 64, 8],
                "warp": [32, 64, 8],
                "stages": 2,
                "align_a": 1,
                "align_b": 1,
                "kind": "implicit_gemm_runtime_launcher" if status == _CUTLASS_CONV_RUNTIME_STATUS else "manifest_scaffold_only",
            },
            selection_predicate={
                "kind": "fallback",
                "description": "correctness-first SIMT fallback for the bounded Conv2d bias slice",
            },
            optional=False,
        )
    ]
    if normalized_dtype == "float16":
        candidates.append(
            _cutlass_conv_candidate(
                op_name,
                normalized_dtype,
                symbol_id=_CUTLASS_CONV_FEW_CHANNELS_SYMBOL_ID,
                target_policy=normalized_target,
                status=status,
                accumulator_dtype="float32",
                cutlass={
                    "opclass": "tensorop",
                    "arch": "sm80",
                    "iterator_algorithm": "few_channels",
                    "instruction_shape": [16, 8, 16],
                    "threadblock": [128, 128, 64],
                    "warp": [64, 64, 64],
                    "stages": 2,
                    "align_a": 1,
                    "align_b": 1,
                    "math_operator": "multiply_add",
                    "kind": "implicit_gemm_runtime_launcher",
                    "v1_inspiration": {
                        "few_channels": True,
                        "semantic_input_channels": 3,
                        "iterator_algorithm": "FewChannels",
                        "align_a": 1,
                        "align_b": 1,
                        "stages": 2,
                    },
                },
                selection_predicate={
                    "kind": "semantic_input_channels",
                    "input_channels": 3,
                    "dtype": "float16",
                    "groups": 1,
                    "requires_layout_translation": "nchw_oihw_to_nhwc_ohwi",
                    "padding_policy": "none",
                },
                optional=True,
            )
        )
        for channel_count, symbol_id in _CUTLASS_CONV_FIXED_CHANNELS_SYMBOL_IDS.items():
            candidates.append(
                _cutlass_conv_candidate(
                    op_name,
                    normalized_dtype,
                    symbol_id=symbol_id,
                    target_policy=normalized_target,
                    status=status,
                    accumulator_dtype="float32",
                    cutlass={
                        "opclass": "tensorop",
                        "arch": "sm80",
                        "iterator_algorithm": "fixed_channels",
                        "instruction_shape": [16, 8, 16],
                        "threadblock": [128, 64, 32],
                        "warp": [64, 16, 32],
                        "stages": 3,
                        "align_a": channel_count,
                        "align_b": channel_count,
                        "math_operator": "multiply_add",
                        "kind": "implicit_gemm_runtime_launcher",
                        "v1_inspiration": {
                            "few_channels": True,
                            "semantic_input_channels": channel_count,
                            "iterator_algorithm": "FixedChannels",
                            "align_a": channel_count,
                            "align_b": channel_count,
                            "stages": 3,
                        },
                    },
                    selection_predicate={
                        "kind": "semantic_input_channels",
                        "input_channels": channel_count,
                        "dtype": "float16",
                        "groups": 1,
                        "requires_layout_translation": "nchw_oihw_to_nhwc_ohwi",
                        "padding_policy": "none",
                    },
                    optional=True,
                )
            )
        candidates.append(
            _cutlass_conv_candidate(
                op_name,
                normalized_dtype,
                symbol_id=_CUTLASS_CONV_OPTIMIZED_SYMBOL_ID,
                target_policy=normalized_target,
                status=status,
                accumulator_dtype="float32",
                cutlass={
                    "opclass": "tensorop",
                    "arch": "sm80",
                    "iterator_algorithm": "optimized",
                    "instruction_shape": [16, 8, 16],
                    "threadblock": [128, 128, 64],
                    "warp": [64, 64, 64],
                    "stages": 3,
                    "align_a": 8,
                    "align_b": 8,
                    "math_operator": "multiply_add",
                    "kind": "implicit_gemm_runtime_launcher",
                    "v1_inspiration": {
                        "iterator_algorithm": "Optimized",
                        "min_semantic_input_channels": 16,
                        "input_channels_multiple": 8,
                        "output_channels_multiple": 8,
                        "align_a": 8,
                        "align_b": 8,
                        "stages": 3,
                    },
                },
                selection_predicate={
                    "kind": "natural_alignment",
                    "dtype": "float16",
                    "groups": 1,
                    "min_input_channels": 16,
                    "input_channels_multiple": 8,
                    "output_channels_multiple": 8,
                    "requires_layout_translation": "nchw_oihw_to_nhwc_ohwi",
                    "padding_policy": "none",
                },
                optional=True,
            )
        )
    return candidates


def cutlass_conv_candidate_compatible_with_plan(
    candidate: Mapping[str, Any],
    cutlass_conv_plan: Mapping[str, Any],
) -> bool:
    dtype = str(cutlass_conv_plan.get("dtype", ""))
    if str(candidate.get("dtype", "")) != dtype:
        return False
    candidate_layouts = dict(candidate.get("layouts", {}))
    semantic_layout = dict(cutlass_conv_plan.get("semantic_layout", {}))
    provider_layout = dict(cutlass_conv_plan.get("provider_layout", {}))
    expected_layouts = {
        "activation_semantic": semantic_layout.get("activation"),
        "weight_semantic": semantic_layout.get("weight"),
        "output_semantic": semantic_layout.get("output"),
        "activation_provider": provider_layout.get("activation"),
        "weight_provider": provider_layout.get("weight"),
        "output_provider": provider_layout.get("output"),
    }
    if candidate_layouts != expected_layouts:
        return False
    conv_config = dict(cutlass_conv_plan.get("conv_config", {}))
    groups = int(conv_config.get("groups", -1) or -1)
    if groups != 1:
        return False
    weight_shape = cutlass_conv_plan.get("weight_shape", ())
    input_shape = cutlass_conv_plan.get("input_shape", ())
    if not isinstance(weight_shape, (list, tuple)) or len(weight_shape) != 4:
        return False
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 4:
        return False
    input_channels = int(weight_shape[1])
    output_channels = int(weight_shape[0])
    if int(input_shape[1]) != input_channels:
        return False
    predicate = candidate.get("selection_predicate", {})
    if not isinstance(predicate, Mapping):
        return False
    if str(predicate.get("dtype", dtype)) != dtype or int(predicate.get("groups", groups)) != groups:
        return False
    if predicate.get("requires_layout_translation") not in (None, "nchw_oihw_to_nhwc_ohwi"):
        return False
    if str(predicate.get("padding_policy", "none")) == "none":
        weight_transform = dict(cutlass_conv_plan.get("weight_transform", {}))
        if int(weight_transform.get("channel_pad_multiple", -1) or -1) != 1:
            return False
        if int(weight_transform.get("padded_input_channels", -1) or -1) != input_channels:
            return False
        if int(weight_transform.get("padded_output_channels", -1) or -1) != output_channels:
            return False
    exact_runtime_slice = predicate.get("exact_runtime_slice")
    if exact_runtime_slice is not None and not _cutlass_conv_exact_runtime_slice_compatible(
        exact_runtime_slice,
        input_shape=input_shape,
        weight_shape=weight_shape,
        output_shape=cutlass_conv_plan.get("output_shape", ()),
        conv_config=conv_config,
    ):
        return False
    kind = str(predicate.get("kind", ""))
    if kind == "fallback":
        return True
    if kind == "semantic_input_channels":
        return int(predicate.get("input_channels", -1) or -1) == input_channels
    if kind == "natural_alignment":
        min_input_channels = int(predicate.get("min_input_channels", 1) or 1)
        input_multiple = int(predicate.get("input_channels_multiple", 1) or 1)
        output_multiple = int(predicate.get("output_channels_multiple", 1) or 1)
        return (
            input_channels >= min_input_channels
            and input_multiple > 0
            and output_multiple > 0
            and input_channels % input_multiple == 0
            and output_channels % output_multiple == 0
        )
    return False


def _cutlass_conv_candidate(
    op_name: str,
    dtype: str,
    *,
    symbol_id: str,
    target_policy: Mapping[str, Any],
    status: str,
    accumulator_dtype: str,
    cutlass: Mapping[str, Any],
    selection_predicate: Mapping[str, Any],
    optional: bool,
) -> dict[str, Any]:
    config_payload = {
        "op": op_name,
        "dtype": dtype,
        "symbol_id": symbol_id,
        "target_policy": dict(target_policy),
        "cutlass": dict(cutlass),
        "selection_predicate": dict(selection_predicate),
    }
    return {
        "candidate_id": f"cutlass_{symbol_id}",
        "candidate_config_key": hashlib.sha256(canonical_json(config_payload).encode("utf-8")).hexdigest(),
        "symbol_id": symbol_id,
        "kernel_symbol": cutlass_conv_symbol(op_name, dtype, symbol_id),
        "profiler_symbol": cutlass_conv_profiler_symbol(op_name, dtype, symbol_id),
        "provider": "cutlass",
        "family": "conv2d_fprop",
        "dtype": dtype,
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
        "cutlass": dict(cutlass),
        "selection_predicate": dict(selection_predicate),
        "optional": bool(optional),
        "target_policy": dict(target_policy),
        "status": status,
        "profiler_status": (
            _CUTLASS_CONV_RUNTIME_PROFILER_STATUS
            if status == _CUTLASS_CONV_RUNTIME_STATUS
            else "unsupported_stub"
        ),
        **(
            {}
            if status == _CUTLASS_CONV_RUNTIME_STATUS
            else {"profiler_blocked_reason": _CUTLASS_CONV_PROFILER_BLOCKED_REASON}
        ),
    }


def cutlass_conv_candidate_set(
    op_name: str,
    dtype: str,
    *,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_conv_op_name(op_name)
    normalized_dtype = _normalize_conv_dtype(dtype)
    candidates = cutlass_conv_candidates(op_name, normalized_dtype, target=target)
    candidate_statuses = {str(candidate.get("status", "")) for candidate in candidates}
    status = (
        _CUTLASS_CONV_RUNTIME_STATUS
        if candidates and candidate_statuses == {_CUTLASS_CONV_RUNTIME_STATUS}
        else _CUTLASS_CONV_SCAFFOLD_STATUS
    )
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
        "status": status,
        "profiler_status": (
            _CUTLASS_CONV_RUNTIME_PROFILER_STATUS
            if status == _CUTLASS_CONV_RUNTIME_STATUS
            else "unsupported_stub"
        ),
        **(
            {}
            if status == _CUTLASS_CONV_RUNTIME_STATUS
            else {"profiler_blocked_reason": _CUTLASS_CONV_PROFILER_BLOCKED_REASON}
        ),
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
        item_node_id = _optional_str(item.get("node_id"))
        if item_node_id is None and isinstance(conv_plan, Mapping):
            item_node_id = _optional_str(conv_plan.get("node_id"))
        selected_id = str(item.get("selected_candidate_id", ""))
        selected = next((candidate for candidate in candidates if str(candidate.get("candidate_id")) == selected_id), None)
        conv_plan_payload = validate_cutlass_conv_scaffold_plan(
            conv_plan,
            candidate=selected,
            node_id=item_node_id,
        )
        conv_plan_key = hashlib.sha256(canonical_json(conv_plan_payload).encode("utf-8")).hexdigest()
        candidate_config_key = str(selected.get("candidate_config_key") if selected else "")
        entries.append(
            {
                "op": str(item.get("op", "")),
                "candidate_set_id": str(item.get("candidate_set_id", "")),
                "candidate_set_key": str(item.get("candidate_set_key", "")),
                "candidate_set": candidate_set,
                "selected_candidate_id": selected_id,
                "node_id": item_node_id,
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
    transform_helpers = _cutlass_conv_transform_helpers(entries)
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
        "profiler_symbols": sorted(
            {
                str(candidate.get("profiler_symbol", ""))
                for candidate in candidates
                if candidate.get("profiler_symbol")
            }
        ),
        "transform_helpers": transform_helpers,
        "transform_helper_symbols": [str(item["symbol"]) for item in transform_helpers],
    }
    payload["used_candidate_plan_key"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def render_cutlass_conv_scaffold_source(source_text: str, used_candidate_plan: Mapping[str, Any]) -> str:
    exports = _cutlass_conv_stub_source_exports(used_candidate_plan)
    marker = "// DINOML_CUTLASS_CONV_STUB_EXPORTS"
    if marker not in source_text:
        raise ValueError("CUTLASS Conv scaffold source is missing the stub export marker")
    return source_text.replace(marker, exports)


def normalize_cutlass_conv_used_candidate_plan(used_candidate_plan: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(used_candidate_plan, Mapping):
        raise ValueError("CUTLASS Conv support scaffold requires a used candidate plan mapping")
    raw_entries = used_candidate_plan.get("entries", ())
    if not isinstance(raw_entries, (list, tuple)):
        raise ValueError("CUTLASS Conv used candidate plan entries must be a list")
    entries = [
        _normalize_cutlass_conv_used_candidate_entry(entry)
        for entry in raw_entries
    ]
    entries = sorted(entries, key=lambda entry: (entry["op"], entry["candidate_set_id"], entry["kernel_symbol"]))
    candidate_sets = _unique_by_key((entry["candidate_set"] for entry in entries), "candidate_set_key")
    candidates = _unique_by_key((candidate for entry in entries for candidate in entry["candidates"]), "candidate_config_key")
    transform_helpers = _cutlass_conv_transform_helpers(entries)
    payload = {
        "schema_version": CUTLASS_CONV_USED_CANDIDATE_PLAN_SCHEMA_VERSION,
        "provider": "cutlass",
        "library": "cutlass_conv",
        "library_name": "cutlass_conv",
        "family": "conv2d_fprop",
        "target": dict(used_candidate_plan.get("target", {})),
        "kernel_manifest_cache_key": used_candidate_plan.get("kernel_manifest_cache_key"),
        "support_cache_key": used_candidate_plan.get("support_cache_key"),
        "entries": entries,
        "candidate_sets": candidate_sets,
        "candidates": candidates,
        "candidate_set_keys": [str(item.get("candidate_set_key", "")) for item in candidate_sets],
        "candidate_config_keys": [str(item.get("candidate_config_key", "")) for item in candidates],
        "kernel_symbols": sorted({entry["kernel_symbol"] for entry in entries if entry["kernel_symbol"]}),
        "profiler_symbols": sorted(
            {
                str(candidate.get("profiler_symbol", ""))
                for candidate in candidates
                if candidate.get("profiler_symbol")
            }
        ),
        "transform_helpers": transform_helpers,
        "transform_helper_symbols": [str(item["symbol"]) for item in transform_helpers],
    }
    payload["used_candidate_plan_key"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def cutlass_conv_wrapper_stages(kernel_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for item in kernel_manifest.get("required_kernels", ()):
        if not isinstance(item, Mapping) or str(item.get("kernel_library", "")) != "cutlass_conv":
            continue
        stages.extend(_cutlass_conv_item_wrapper_stages(item))
    return stages


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
    conv_config = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
    }
    plan_status = _conv_plan_status(
        dtype,
        input_shape=x_shape,
        weight_shape=weight_shape,
        output_shape=output_shape,
        conv_config=conv_config,
    )
    return validate_cutlass_conv_scaffold_plan(
        {
        "schema_version": 1,
        "kind": _CUTLASS_CONV_SCAFFOLD_KIND,
        "status": plan_status,
        **_conv_plan_status_payload(plan_status),
        "node_id": str(node.get("id", "")),
        "op_family": "conv2d_bias",
        "dtype": dtype,
        "semantic_layout": dict(_CUTLASS_CONV_SCAFFOLD_SEMANTIC_LAYOUT),
        "provider_layout": dict(_CUTLASS_CONV_SCAFFOLD_PROVIDER_LAYOUT),
        "layout_translation": {
            "input_pack": "nchw_to_nhwc_temporary",
            "output_unpack": "nhwc_to_nchw_temporary",
            "bias": "direct_per_output_channel",
            "input_pack_nbytes": _nbytes(x_shape, dtype_size),
            "output_unpack_nbytes": _nbytes(output_shape, dtype_size),
            "input_pack_symbol": cutlass_conv_input_pack_symbol(dtype),
            "output_unpack_symbol": cutlass_conv_output_unpack_symbol(dtype),
        },
        "weight_transform": {
            "from": "oihw",
            "to": "ohwi",
            "pack": "oihw_to_ohwi_temporary",
            "temporary_nbytes": _nbytes(weight_shape, dtype_size),
            "pack_symbol": cutlass_conv_weight_pack_symbol(dtype),
            "runtime_persistent": False,
            "channel_pad_multiple": 1,
            "padded_input_channels": int(weight_shape[1]),
            "padded_output_channels": int(weight_shape[0]),
            "padding_fill_value": 0.0,
        },
        "conv_config": conv_config,
        "input_shape": x_shape,
        "weight_shape": weight_shape,
        "bias_shape": bias_shape,
        "output_shape": output_shape,
        "workspace_nbytes": 0,
        "temporary_buffers": temporary_buffers,
        "temporary_nbytes": sum(int(buffer["nbytes"]) for buffer in temporary_buffers),
        },
        node_id=_optional_str(node.get("id")),
    )


def validate_cutlass_conv_scaffold_plan(
    plan: Mapping[str, Any] | None,
    *,
    candidate: Mapping[str, Any] | None = None,
    node_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise ValueError("CUTLASS Conv scaffold requires cutlass_conv_plan transform metadata")
    payload = dict(plan)
    if str(payload.get("kind")) != _CUTLASS_CONV_SCAFFOLD_KIND:
        raise ValueError(f"Unsupported CUTLASS Conv scaffold kind {payload.get('kind')!r}")
    status = str(payload.get("status"))
    if status not in {_CUTLASS_CONV_SCAFFOLD_STATUS, _CUTLASS_CONV_RUNTIME_STATUS}:
        raise ValueError(f"Unsupported CUTLASS Conv scaffold status {payload.get('status')!r}")
    if str(payload.get("op_family")) != "conv2d_bias":
        raise ValueError(f"Unsupported CUTLASS Conv scaffold op family {payload.get('op_family')!r}")
    if node_id is not None and str(payload.get("node_id", "")) != node_id:
        raise ValueError(
            f"CUTLASS Conv scaffold node_id mismatch: expected {node_id!r}, got {payload.get('node_id')!r}"
        )
    dtype = _normalize_conv_dtype(str(payload.get("dtype")))
    semantic_layout = dict(payload.get("semantic_layout", {}))
    provider_layout = dict(payload.get("provider_layout", {}))
    expected_semantic_layout = dict(_CUTLASS_CONV_SCAFFOLD_SEMANTIC_LAYOUT)
    expected_provider_layout = dict(_CUTLASS_CONV_SCAFFOLD_PROVIDER_LAYOUT)
    if semantic_layout != expected_semantic_layout:
        raise ValueError(
            f"CUTLASS Conv scaffold semantic_layout must be {expected_semantic_layout}, got {semantic_layout!r}"
        )
    if provider_layout != expected_provider_layout:
        raise ValueError(
            f"CUTLASS Conv scaffold provider_layout must be {expected_provider_layout}, got {provider_layout!r}"
        )
    x_shape = _validate_positive_shape(payload.get("input_shape"), rank=4, name="input_shape")
    weight_shape = _validate_positive_shape(payload.get("weight_shape"), rank=4, name="weight_shape")
    bias_shape = _validate_positive_shape(payload.get("bias_shape"), rank=1, name="bias_shape")
    output_shape = _validate_positive_shape(payload.get("output_shape"), rank=4, name="output_shape")
    dtype_size = dtype_nbytes(dtype)
    conv_config = dict(payload.get("conv_config", {}))
    stride = _validate_positive_shape(conv_config.get("stride"), rank=2, name="conv_config.stride")
    padding = _validate_non_negative_shape(conv_config.get("padding"), rank=2, name="conv_config.padding")
    dilation = _validate_positive_shape(conv_config.get("dilation"), rank=2, name="conv_config.dilation")
    groups = conv_config.get("groups")
    if not isinstance(groups, int) or isinstance(groups, bool) or groups != 1:
        raise ValueError(f"CUTLASS Conv scaffold currently requires conv_config.groups == 1, got {groups!r}")
    normalized_conv_config = {"stride": stride, "padding": padding, "dilation": dilation, "groups": 1}
    expected_status = _conv_plan_status(
        dtype,
        input_shape=x_shape,
        weight_shape=weight_shape,
        output_shape=output_shape,
        conv_config=normalized_conv_config,
    )
    if status != expected_status:
        raise ValueError(
            f"CUTLASS Conv scaffold status for dtype/shape slice must be {expected_status!r}, got {status!r}"
        )
    if status == _CUTLASS_CONV_SCAFFOLD_STATUS:
        if str(payload.get("blocked_reason")) != _CUTLASS_CONV_SCAFFOLD_BLOCKED_REASON:
            raise ValueError(
                "CUTLASS Conv scaffold blocked_reason must record "
                f"{_CUTLASS_CONV_SCAFFOLD_BLOCKED_REASON!r}"
            )
    else:
        runtime = payload.get("runtime")
        if not isinstance(runtime, Mapping):
            raise ValueError("CUTLASS Conv bounded runtime plan requires runtime metadata")
        if str(runtime.get("status")) != _CUTLASS_CONV_RUNTIME_STATUS:
            raise ValueError("CUTLASS Conv bounded runtime metadata has an unexpected status")
        if str(payload.get("profiler_status")) != _CUTLASS_CONV_RUNTIME_PROFILER_STATUS:
            raise ValueError(
                "CUTLASS Conv bounded runtime plan must record "
                f"profiler_status={_CUTLASS_CONV_RUNTIME_PROFILER_STATUS!r}"
            )
    if weight_shape[1] != x_shape[1]:
        raise ValueError(
            "CUTLASS Conv scaffold weight/input channel mismatch for groups=1: "
            f"weight I={weight_shape[1]} vs input C={x_shape[1]}"
        )
    if bias_shape[0] != weight_shape[0] or output_shape[1] != weight_shape[0]:
        raise ValueError(
            "CUTLASS Conv scaffold output/bias channels must match weight O: "
            f"bias={bias_shape[0]}, output C={output_shape[1]}, weight O={weight_shape[0]}"
        )
    layout_translation = dict(payload.get("layout_translation", {}))
    expected_input_nbytes = _nbytes(x_shape, dtype_size)
    expected_output_nbytes = _nbytes(output_shape, dtype_size)
    if layout_translation.get("input_pack") != "nchw_to_nhwc_temporary":
        raise ValueError("CUTLASS Conv scaffold layout_translation.input_pack must be 'nchw_to_nhwc_temporary'")
    if layout_translation.get("output_unpack") != "nhwc_to_nchw_temporary":
        raise ValueError("CUTLASS Conv scaffold layout_translation.output_unpack must be 'nhwc_to_nchw_temporary'")
    if layout_translation.get("bias") != "direct_per_output_channel":
        raise ValueError("CUTLASS Conv scaffold layout_translation.bias must be 'direct_per_output_channel'")
    if int(layout_translation.get("input_pack_nbytes", -1)) != expected_input_nbytes:
        raise ValueError(
            "CUTLASS Conv scaffold input_pack_nbytes mismatch: "
            f"expected {expected_input_nbytes}, got {layout_translation.get('input_pack_nbytes')!r}"
        )
    if str(layout_translation.get("input_pack_symbol")) != cutlass_conv_input_pack_symbol(dtype):
        raise ValueError(
            "CUTLASS Conv scaffold layout_translation.input_pack_symbol mismatch: "
            f"expected {cutlass_conv_input_pack_symbol(dtype)!r}, got {layout_translation.get('input_pack_symbol')!r}"
        )
    if int(layout_translation.get("output_unpack_nbytes", -1)) != expected_output_nbytes:
        raise ValueError(
            "CUTLASS Conv scaffold output_unpack_nbytes mismatch: "
            f"expected {expected_output_nbytes}, got {layout_translation.get('output_unpack_nbytes')!r}"
        )
    if str(layout_translation.get("output_unpack_symbol")) != cutlass_conv_output_unpack_symbol(dtype):
        raise ValueError(
            "CUTLASS Conv scaffold layout_translation.output_unpack_symbol mismatch: "
            f"expected {cutlass_conv_output_unpack_symbol(dtype)!r}, got {layout_translation.get('output_unpack_symbol')!r}"
        )
    weight_transform = dict(payload.get("weight_transform", {}))
    expected_weight_nbytes = _nbytes(weight_shape, dtype_size)
    if weight_transform.get("from") != "oihw":
        raise ValueError("CUTLASS Conv scaffold weight_transform.from must be 'oihw'")
    if weight_transform.get("to") != "ohwi":
        raise ValueError("CUTLASS Conv scaffold weight_transform.to must be 'ohwi'")
    if weight_transform.get("pack") != "oihw_to_ohwi_temporary":
        raise ValueError("CUTLASS Conv scaffold weight_transform.pack must be 'oihw_to_ohwi_temporary'")
    if str(weight_transform.get("pack_symbol")) != cutlass_conv_weight_pack_symbol(dtype):
        raise ValueError(
            "CUTLASS Conv scaffold weight_transform.pack_symbol mismatch: "
            f"expected {cutlass_conv_weight_pack_symbol(dtype)!r}, got {weight_transform.get('pack_symbol')!r}"
        )
    if int(weight_transform.get("temporary_nbytes", -1)) != expected_weight_nbytes:
        raise ValueError(
            "CUTLASS Conv scaffold weight temporary_nbytes mismatch: "
            f"expected {expected_weight_nbytes}, got {weight_transform.get('temporary_nbytes')!r}"
        )
    if bool(weight_transform.get("runtime_persistent")):
        raise ValueError("CUTLASS Conv scaffold weight_transform.runtime_persistent must be false")
    channel_pad_multiple = weight_transform.get("channel_pad_multiple")
    if not isinstance(channel_pad_multiple, int) or isinstance(channel_pad_multiple, bool) or channel_pad_multiple <= 0:
        raise ValueError(
            "CUTLASS Conv scaffold weight_transform.channel_pad_multiple must be a positive integer, "
            f"got {channel_pad_multiple!r}"
        )
    expected_padded_input_channels = _round_up(weight_shape[1], channel_pad_multiple)
    expected_padded_output_channels = _round_up(weight_shape[0], channel_pad_multiple)
    if int(weight_transform.get("padded_input_channels", -1)) != expected_padded_input_channels:
        raise ValueError(
            "CUTLASS Conv scaffold padded_input_channels mismatch: "
            f"expected {expected_padded_input_channels}, got {weight_transform.get('padded_input_channels')!r}"
        )
    if int(weight_transform.get("padded_output_channels", -1)) != expected_padded_output_channels:
        raise ValueError(
            "CUTLASS Conv scaffold padded_output_channels mismatch: "
            f"expected {expected_padded_output_channels}, got {weight_transform.get('padded_output_channels')!r}"
        )
    if float(weight_transform.get("padding_fill_value", 0.0)) != 0.0:
        raise ValueError("CUTLASS Conv scaffold padding_fill_value must be 0.0")
    temporary_buffers = payload.get("temporary_buffers", ())
    if not isinstance(temporary_buffers, (list, tuple)):
        raise ValueError("CUTLASS Conv scaffold temporary_buffers must be a list")
    expected_buffers = (
        ("activation_nhwc", "layout_pack", "nhwc", expected_input_nbytes),
        ("weight_ohwi", "layout_pack", "ohwi", expected_weight_nbytes),
        ("output_nhwc", "layout_pack", "nhwc", expected_output_nbytes),
    )
    if len(temporary_buffers) != len(expected_buffers):
        raise ValueError(
            f"CUTLASS Conv scaffold temporary_buffers must contain {len(expected_buffers)} entries, "
            f"got {len(temporary_buffers)}"
        )
    normalized_buffers = []
    for buffer, (expected_name, expected_kind, expected_layout, expected_nbytes) in zip(
        temporary_buffers, expected_buffers, strict=True
    ):
        if not isinstance(buffer, Mapping):
            raise ValueError("CUTLASS Conv scaffold temporary_buffers entries must be objects")
        buffer_payload = dict(buffer)
        if str(buffer_payload.get("name")) != expected_name:
            raise ValueError(
                f"CUTLASS Conv scaffold temporary buffer name mismatch: expected {expected_name!r}, "
                f"got {buffer_payload.get('name')!r}"
            )
        if str(buffer_payload.get("kind")) != expected_kind:
            raise ValueError(
                f"CUTLASS Conv scaffold temporary buffer kind mismatch for {expected_name!r}: "
                f"expected {expected_kind!r}, got {buffer_payload.get('kind')!r}"
            )
        if str(buffer_payload.get("layout")) != expected_layout:
            raise ValueError(
                f"CUTLASS Conv scaffold temporary buffer layout mismatch for {expected_name!r}: "
                f"expected {expected_layout!r}, got {buffer_payload.get('layout')!r}"
            )
        if int(buffer_payload.get("nbytes", -1)) != expected_nbytes:
            raise ValueError(
                f"CUTLASS Conv scaffold temporary buffer nbytes mismatch for {expected_name!r}: "
                f"expected {expected_nbytes}, got {buffer_payload.get('nbytes')!r}"
            )
        normalized_buffers.append(buffer_payload)
    if int(payload.get("workspace_nbytes", -1)) != 0:
        raise ValueError("CUTLASS Conv scaffold workspace_nbytes must be 0 for the current scaffold")
    expected_temporary_nbytes = sum(int(buffer["nbytes"]) for buffer in normalized_buffers)
    if int(payload.get("temporary_nbytes", -1)) != expected_temporary_nbytes:
        raise ValueError(
            "CUTLASS Conv scaffold temporary_nbytes mismatch: "
            f"expected {expected_temporary_nbytes}, got {payload.get('temporary_nbytes')!r}"
        )
    if candidate is not None:
        candidate_layouts = dict(candidate.get("layouts", {}))
        expected_candidate_layouts = {
            "activation_semantic": semantic_layout["activation"],
            "weight_semantic": semantic_layout["weight"],
            "output_semantic": semantic_layout["output"],
            "activation_provider": provider_layout["activation"],
            "weight_provider": provider_layout["weight"],
            "output_provider": provider_layout["output"],
        }
        if candidate_layouts != expected_candidate_layouts:
            raise ValueError(
                "CUTLASS Conv scaffold candidate layouts do not match transform plan: "
                f"expected {expected_candidate_layouts}, got {candidate_layouts!r}"
            )
        if str(candidate.get("dtype")) != dtype:
            raise ValueError(
                f"CUTLASS Conv scaffold candidate dtype mismatch: expected {dtype!r}, got {candidate.get('dtype')!r}"
            )
        selected_candidate = payload.get("selected_candidate")
        if isinstance(selected_candidate, Mapping):
            if str(selected_candidate.get("candidate_id", "")) != str(candidate.get("candidate_id", "")):
                raise ValueError(
                    "CUTLASS Conv scaffold selected_candidate.candidate_id mismatch: "
                    f"expected {candidate.get('candidate_id')!r}, got {selected_candidate.get('candidate_id')!r}"
                )
            if str(selected_candidate.get("kernel_symbol", "")) != str(candidate.get("kernel_symbol", "")):
                raise ValueError(
                    "CUTLASS Conv scaffold selected_candidate.kernel_symbol mismatch: "
                    f"expected {candidate.get('kernel_symbol')!r}, got {selected_candidate.get('kernel_symbol')!r}"
                )
    payload["dtype"] = dtype
    payload["semantic_layout"] = semantic_layout
    payload["provider_layout"] = provider_layout
    payload["layout_translation"] = layout_translation
    payload["weight_transform"] = weight_transform
    payload["conv_config"] = normalized_conv_config
    payload["input_shape"] = x_shape
    payload["weight_shape"] = weight_shape
    payload["bias_shape"] = bias_shape
    payload["output_shape"] = output_shape
    payload["temporary_buffers"] = normalized_buffers
    payload["workspace_nbytes"] = 0
    payload["temporary_nbytes"] = expected_temporary_nbytes
    payload["status"] = status
    if status == _CUTLASS_CONV_RUNTIME_STATUS:
        payload["runtime"] = dict(payload["runtime"])
        payload["profiler_status"] = _CUTLASS_CONV_RUNTIME_PROFILER_STATUS
        payload.pop("profiler_blocked_reason", None)
        payload.pop("blocked_reason", None)
    else:
        payload["blocked_reason"] = _CUTLASS_CONV_SCAFFOLD_BLOCKED_REASON
        payload.pop("runtime", None)
    return payload


def validate_cutlass_conv_plan(
    plan: Mapping[str, Any] | None,
    *,
    candidate: Mapping[str, Any] | None = None,
    node_id: str | None = None,
) -> dict[str, Any]:
    # Compatibility alias for existing callers; this validator remains scoped to
    # the current NHWC/OHWI Conv2d bias provider plan.
    return validate_cutlass_conv_scaffold_plan(plan, candidate=candidate, node_id=node_id)


def _normalize_cutlass_conv_used_candidate_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise ValueError("CUTLASS Conv used candidate plan entries must be objects")
    payload = dict(entry)
    op_name = str(payload.get("op", ""))
    _validate_conv_op_name(op_name)
    selected_candidate_id = str(payload.get("selected_candidate_id", ""))
    if not selected_candidate_id:
        raise ValueError("CUTLASS Conv used candidate plan entry is missing selected_candidate_id")
    raw_candidates = payload.get("candidates", ())
    if not isinstance(raw_candidates, (list, tuple)):
        raise ValueError("CUTLASS Conv used candidate plan entry candidates must be a list")
    candidates = []
    selected_candidate = None
    for candidate in raw_candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("CUTLASS Conv used candidate plan entry candidates must be objects")
        candidate_payload = dict(candidate)
        candidates.append(candidate_payload)
        if str(candidate_payload.get("candidate_id", "")) == selected_candidate_id:
            selected_candidate = candidate_payload
    if selected_candidate is None:
        raise ValueError(
            "CUTLASS Conv used candidate plan entry selected_candidate_id does not match any candidate: "
            f"{selected_candidate_id!r}"
        )
    _validate_cutlass_conv_selected_candidate(op_name, selected_candidate)
    node_id = _optional_str(payload.get("node_id"))
    if node_id is None and isinstance(payload.get("cutlass_conv_plan"), Mapping):
        node_id = _optional_str(payload["cutlass_conv_plan"].get("node_id"))
    conv_plan_payload = validate_cutlass_conv_scaffold_plan(
        payload.get("cutlass_conv_plan"),
        candidate=selected_candidate,
        node_id=node_id,
    )
    conv_plan_key = hashlib.sha256(canonical_json(conv_plan_payload).encode("utf-8")).hexdigest()
    candidate_set = _normalize_cutlass_conv_candidate_set(
        payload.get("candidate_set"),
        op_name=op_name,
        dtype=str(selected_candidate["dtype"]),
        selected_candidate=selected_candidate,
    )
    if payload.get("candidate_set_id") and str(payload.get("candidate_set_id")) != str(candidate_set["candidate_set_id"]):
        raise ValueError(
            "CUTLASS Conv used candidate plan entry candidate_set_id mismatch: "
            f"expected {candidate_set['candidate_set_id']!r}, got {payload.get('candidate_set_id')!r}"
        )
    if payload.get("candidate_set_key") and str(payload.get("candidate_set_key")) != str(candidate_set["candidate_set_key"]):
        raise ValueError(
            "CUTLASS Conv used candidate plan entry candidate_set_key mismatch: "
            f"expected {candidate_set['candidate_set_key']!r}, got {payload.get('candidate_set_key')!r}"
        )
    expected_candidate_config_key = str(selected_candidate.get("candidate_config_key", ""))
    if payload.get("candidate_config_key") and str(payload.get("candidate_config_key")) != expected_candidate_config_key:
        raise ValueError(
            "CUTLASS Conv used candidate plan entry candidate_config_key mismatch: "
            f"expected {expected_candidate_config_key!r}, got {payload.get('candidate_config_key')!r}"
        )
    if payload.get("kernel_symbol") and str(payload.get("kernel_symbol")) != str(selected_candidate.get("kernel_symbol", "")):
        raise ValueError(
            "CUTLASS Conv used candidate plan entry kernel_symbol mismatch: "
            f"expected {selected_candidate.get('kernel_symbol')!r}, got {payload.get('kernel_symbol')!r}"
        )
    if payload.get("profiler_symbol") and str(payload.get("profiler_symbol")) != str(selected_candidate.get("profiler_symbol", "")):
        raise ValueError(
            "CUTLASS Conv used candidate plan entry profiler_symbol mismatch: "
            f"expected {selected_candidate.get('profiler_symbol')!r}, got {payload.get('profiler_symbol')!r}"
        )
    return {
        "op": op_name,
        "candidate_set_id": str(candidate_set["candidate_set_id"]),
        "candidate_set_key": str(candidate_set["candidate_set_key"]),
        "candidate_set": candidate_set,
        "selected_candidate_id": selected_candidate_id,
        "node_id": node_id,
        "candidate_config_key": expected_candidate_config_key,
        "kernel_symbol": str(selected_candidate.get("kernel_symbol", "")),
        "profiler_symbol": str(selected_candidate.get("profiler_symbol", "")),
        "cutlass_conv_plan": conv_plan_payload,
        "cutlass_conv_plan_key": conv_plan_key,
        "candidates": candidates,
    }


def _normalize_cutlass_conv_candidate_set(
    candidate_set: Mapping[str, Any] | None,
    *,
    op_name: str,
    dtype: str,
    selected_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(candidate_set, Mapping):
        raise ValueError("CUTLASS Conv used candidate plan entry is missing candidate_set metadata")
    payload = dict(candidate_set)
    expected = cutlass_conv_candidate_set(op_name, dtype, target=payload.get("target_policy"))
    for field in (
        "candidate_set_id",
        "candidate_set_key",
        "provider",
        "family",
        "op",
        "dtype",
        "accumulator_dtype",
        "epilogue",
        "launch_abi",
        "semantic_layout",
        "provider_layout",
        "supported_groups",
        "supported_dtypes",
        "candidate_count",
        "candidate_config_keys",
        "target_policy",
        "status",
        "profiler_status",
    ):
        if payload.get(field) != expected.get(field):
            raise ValueError(
                f"CUTLASS Conv used candidate plan entry candidate_set.{field} mismatch: "
                f"expected {expected.get(field)!r}, got {payload.get(field)!r}"
            )
    if expected.get("profiler_blocked_reason") is not None and payload.get("profiler_blocked_reason") != expected.get(
        "profiler_blocked_reason"
    ):
        raise ValueError(
            "CUTLASS Conv used candidate plan entry candidate_set.profiler_blocked_reason mismatch: "
            f"expected {expected.get('profiler_blocked_reason')!r}, got {payload.get('profiler_blocked_reason')!r}"
        )
    if str(selected_candidate.get("candidate_config_key", "")) not in payload["candidate_config_keys"]:
        raise ValueError(
            "CUTLASS Conv used candidate plan entry candidate_set does not contain the selected candidate config key"
        )
    return expected


def _validate_cutlass_conv_selected_candidate(op_name: str, candidate: Mapping[str, Any]) -> None:
    dtype = _normalize_conv_dtype(str(candidate.get("dtype")))
    expected_candidates = cutlass_conv_candidates(op_name, dtype, target=candidate.get("target_policy"))
    expected = next(
        (
            item
            for item in expected_candidates
            if str(item.get("candidate_id", "")) == str(candidate.get("candidate_id", ""))
        ),
        None,
    )
    if expected is None:
        raise ValueError(
            "CUTLASS Conv used candidate plan entry selected candidate_id is not emitted by the current scaffold: "
            f"{candidate.get('candidate_id')!r}"
        )
    candidate_payload = dict(candidate)
    for field in (
        "candidate_id",
        "candidate_config_key",
        "symbol_id",
        "kernel_symbol",
        "profiler_symbol",
        "provider",
        "family",
        "dtype",
        "accumulator_dtype",
        "epilogue",
        "launch_abi",
        "layouts",
        "cutlass",
        "selection_predicate",
        "optional",
        "target_policy",
        "status",
        "profiler_status",
    ):
        if candidate_payload.get(field) != expected.get(field):
            raise ValueError(
                f"CUTLASS Conv used candidate plan entry candidate.{field} mismatch: "
                f"expected {expected.get(field)!r}, got {candidate_payload.get(field)!r}"
            )
    if expected.get("profiler_blocked_reason") is not None and candidate_payload.get(
        "profiler_blocked_reason"
    ) != expected.get("profiler_blocked_reason"):
        raise ValueError(
            "CUTLASS Conv used candidate plan entry candidate.profiler_blocked_reason mismatch: "
            f"expected {expected.get('profiler_blocked_reason')!r}, got {candidate_payload.get('profiler_blocked_reason')!r}"
        )


def _cutlass_conv_item_wrapper_stages(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    op_name = str(item.get("op", ""))
    _validate_conv_op_name(op_name)
    selected_candidate = _selected_cutlass_conv_candidate(item)
    node_id = _optional_str(item.get("node_id"))
    if node_id is None and isinstance(item.get("cutlass_conv_plan"), Mapping):
        node_id = _optional_str(item["cutlass_conv_plan"].get("node_id"))
    conv_plan = validate_cutlass_conv_scaffold_plan(
        item.get("cutlass_conv_plan"),
        candidate=selected_candidate,
        node_id=node_id,
    )
    layout_translation = dict(conv_plan["layout_translation"])
    weight_transform = dict(conv_plan["weight_transform"])
    temporary_buffers = {
        str(buffer["name"]): dict(buffer)
        for buffer in conv_plan.get("temporary_buffers", ())
        if isinstance(buffer, Mapping)
    }
    activation_shape = list(conv_plan["input_shape"])
    weight_shape = list(conv_plan["weight_shape"])
    output_shape = list(conv_plan["output_shape"])
    conv_config = dict(conv_plan["conv_config"])
    stage_common = {
        "schema_version": 1,
        "op": op_name,
        "node_id": node_id,
        "kernel_library": "cutlass_conv",
        "dtype": str(conv_plan["dtype"]),
        "status": str(conv_plan["status"]),
        **({"blocked_reason": str(conv_plan["blocked_reason"])} if conv_plan.get("blocked_reason") else {}),
    }
    return [
        {
            **stage_common,
            "stage_index": 0,
            "stage_name": "activation_pack",
            "stage_kind": "transform_helper",
            "symbol": str(layout_translation["input_pack_symbol"]),
            "helper_abi": _CUTLASS_CONV_TRANSFORM_ABI,
            "tensor_role": "activation",
            "layout_from": "nchw",
            "layout_to": "nhwc",
            "source": {"kind": "semantic_tensor", "role": "activation", "layout": "nchw"},
            "destination": _temporary_buffer_descriptor(temporary_buffers["activation_nhwc"]),
            "shape_args": [
                {"name": "n", "placeholder": "activation_n", "value": int(activation_shape[0])},
                {"name": "c", "placeholder": "activation_c", "value": int(activation_shape[1])},
                {"name": "h", "placeholder": "activation_h", "value": int(activation_shape[2])},
                {"name": "w", "placeholder": "activation_w", "value": int(activation_shape[3])},
            ],
        },
        {
            **stage_common,
            "stage_index": 1,
            "stage_name": "weight_pack",
            "stage_kind": "transform_helper",
            "symbol": str(weight_transform["pack_symbol"]),
            "helper_abi": _CUTLASS_CONV_TRANSFORM_ABI,
            "tensor_role": "weight",
            "layout_from": "oihw",
            "layout_to": "ohwi",
            "source": {"kind": "semantic_tensor", "role": "weight", "layout": "oihw"},
            "destination": _temporary_buffer_descriptor(temporary_buffers["weight_ohwi"]),
            "shape_args": [
                {"name": "out_c", "placeholder": "weight_o", "value": int(weight_shape[0])},
                {"name": "in_c", "placeholder": "weight_i", "value": int(weight_shape[1])},
                {"name": "kernel_h", "placeholder": "kernel_h", "value": int(weight_shape[2])},
                {"name": "kernel_w", "placeholder": "kernel_w", "value": int(weight_shape[3])},
            ],
        },
        {
            **stage_common,
            "stage_index": 2,
            "stage_name": "provider_launch",
            "stage_kind": "provider_launcher",
            "symbol": str(selected_candidate["kernel_symbol"]),
            "launch_abi": str(selected_candidate["launch_abi"]),
            "inputs": [
                _temporary_buffer_descriptor(temporary_buffers["activation_nhwc"]),
                _temporary_buffer_descriptor(temporary_buffers["weight_ohwi"]),
                {"kind": "semantic_tensor", "role": "bias", "layout": "o"},
            ],
            "output": _temporary_buffer_descriptor(temporary_buffers["output_nhwc"]),
            "shape_args": [
                {"name": "n", "placeholder": "activation_n", "value": int(activation_shape[0])},
                {"name": "h", "placeholder": "activation_h", "value": int(activation_shape[2])},
                {"name": "w", "placeholder": "activation_w", "value": int(activation_shape[3])},
                {"name": "c", "placeholder": "activation_c", "value": int(activation_shape[1])},
                {"name": "out_h", "placeholder": "output_h", "value": int(output_shape[2])},
                {"name": "out_w", "placeholder": "output_w", "value": int(output_shape[3])},
                {"name": "out_c", "placeholder": "output_c", "value": int(output_shape[1])},
                {"name": "kernel_h", "placeholder": "kernel_h", "value": int(weight_shape[2])},
                {"name": "kernel_w", "placeholder": "kernel_w", "value": int(weight_shape[3])},
                {"name": "stride_h", "placeholder": "stride_h", "value": int(conv_config["stride"][0])},
                {"name": "stride_w", "placeholder": "stride_w", "value": int(conv_config["stride"][1])},
                {"name": "pad_h", "placeholder": "pad_h", "value": int(conv_config["padding"][0])},
                {"name": "pad_w", "placeholder": "pad_w", "value": int(conv_config["padding"][1])},
                {"name": "dilation_h", "placeholder": "dilation_h", "value": int(conv_config["dilation"][0])},
                {"name": "dilation_w", "placeholder": "dilation_w", "value": int(conv_config["dilation"][1])},
            ],
        },
        {
            **stage_common,
            "stage_index": 3,
            "stage_name": "output_unpack",
            "stage_kind": "transform_helper",
            "symbol": str(layout_translation["output_unpack_symbol"]),
            "helper_abi": _CUTLASS_CONV_TRANSFORM_ABI,
            "tensor_role": "output",
            "layout_from": "nhwc",
            "layout_to": "nchw",
            "source": _temporary_buffer_descriptor(temporary_buffers["output_nhwc"]),
            "destination": {"kind": "semantic_tensor", "role": "output", "layout": "nchw"},
            "shape_args": [
                {"name": "n", "placeholder": "output_n", "value": int(output_shape[0])},
                {"name": "c", "placeholder": "output_c", "value": int(output_shape[1])},
                {"name": "h", "placeholder": "output_h", "value": int(output_shape[2])},
                {"name": "w", "placeholder": "output_w", "value": int(output_shape[3])},
            ],
        },
    ]


def _nbytes(shape: list[int], dtype_size: int) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * int(dtype_size)


def _validate_positive_shape(value: Any, *, rank: int, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != rank:
        raise ValueError(f"CUTLASS Conv scaffold {name} must be a rank-{rank} integer shape, got {value!r}")
    dims = []
    for dim in value:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError(f"CUTLASS Conv scaffold {name} must contain positive integers, got {value!r}")
        dims.append(int(dim))
    return dims


def _validate_non_negative_shape(value: Any, *, rank: int, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != rank:
        raise ValueError(f"CUTLASS Conv scaffold {name} must be a rank-{rank} integer shape, got {value!r}")
    dims = []
    for dim in value:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
            raise ValueError(f"CUTLASS Conv scaffold {name} must contain non-negative integers, got {value!r}")
        dims.append(int(dim))
    return dims


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


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


def _selected_cutlass_conv_candidate(item: Mapping[str, Any]) -> Mapping[str, Any]:
    selected_id = str(item.get("selected_candidate_id", ""))
    candidates = item.get("candidates", ())
    if isinstance(candidates, (list, tuple)):
        for candidate in candidates:
            if isinstance(candidate, Mapping) and str(candidate.get("candidate_id", "")) == selected_id:
                return dict(candidate)
    raise ValueError(
        "CUTLASS Conv scaffold wrapper stages require a selected candidate present in the manifest item"
    )


def _temporary_buffer_descriptor(buffer: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "temporary_buffer",
        "name": str(buffer["name"]),
        "layout": str(buffer["layout"]),
        "nbytes": int(buffer["nbytes"]),
    }


def _cutlass_conv_transform_helpers(entries) -> list[dict[str, Any]]:
    helpers_by_symbol: dict[str, dict[str, Any]] = {}
    for entry in entries:
        conv_plan = dict(entry["cutlass_conv_plan"])
        dtype = str(conv_plan["dtype"])
        layout_translation = dict(conv_plan["layout_translation"])
        weight_transform = dict(conv_plan["weight_transform"])
        for helper in (
            {
                "symbol": str(layout_translation["input_pack_symbol"]),
                "dtype": dtype,
                "tensor_role": "activation",
                "transform": str(layout_translation["input_pack"]),
                "layout_from": "nchw",
                "layout_to": "nhwc",
                "shape_order": ["n", "c", "h", "w"],
                "helper_abi": _CUTLASS_CONV_TRANSFORM_ABI,
            },
            {
                "symbol": str(weight_transform["pack_symbol"]),
                "dtype": dtype,
                "tensor_role": "weight",
                "transform": str(weight_transform["pack"]),
                "layout_from": "oihw",
                "layout_to": "ohwi",
                "shape_order": ["o", "i", "h", "w"],
                "helper_abi": _CUTLASS_CONV_TRANSFORM_ABI,
            },
            {
                "symbol": str(layout_translation["output_unpack_symbol"]),
                "dtype": dtype,
                "tensor_role": "output",
                "transform": str(layout_translation["output_unpack"]),
                "layout_from": "nhwc",
                "layout_to": "nchw",
                "shape_order": ["n", "c", "h", "w"],
                "helper_abi": _CUTLASS_CONV_TRANSFORM_ABI,
            },
        ):
            existing = helpers_by_symbol.get(helper["symbol"])
            if existing is not None and existing != helper:
                raise ValueError(
                    "CUTLASS Conv transform helper metadata drifted for symbol "
                    f"{helper['symbol']!r}: expected {existing!r}, got {helper!r}"
                )
            helpers_by_symbol[helper["symbol"]] = helper
    return sorted(helpers_by_symbol.values(), key=lambda item: (item["dtype"], item["tensor_role"], item["symbol"]))


def _cutlass_conv_stub_source_exports(used_candidate_plan: Mapping[str, Any]) -> str:
    lines: list[str] = []
    emitted: set[str] = set()
    transform_helpers = [
        dict(item)
        for item in used_candidate_plan.get("transform_helpers", ())
        if isinstance(item, Mapping)
    ]
    if transform_helpers:
        lines.append(_cutlass_conv_transform_runtime_support_source())
    for helper in transform_helpers:
        symbol_name = str(helper.get("symbol", ""))
        if not symbol_name or symbol_name in emitted:
            continue
        emitted.add(symbol_name)
        lines.append(_cutlass_conv_transform_source(helper))
    for candidate in used_candidate_plan.get("candidates", ()):
        if not isinstance(candidate, Mapping):
            continue
        symbol_name = str(candidate.get("kernel_symbol", ""))
        if not symbol_name or symbol_name in emitted:
            continue
        emitted.add(symbol_name)
        if str(candidate.get("status", "")) == _CUTLASS_CONV_RUNTIME_STATUS:
            lines.append(_cutlass_conv_runtime_launcher_source(symbol_name, candidate))
        else:
            lines.append(_cutlass_conv_stub_launcher_source(symbol_name))
    candidates_by_profiler = {
        str(candidate.get("profiler_symbol", "")): candidate
        for candidate in used_candidate_plan.get("candidates", ())
        if isinstance(candidate, Mapping)
    }
    for symbol in used_candidate_plan.get("profiler_symbols", ()):
        symbol_name = str(symbol)
        if not symbol_name or symbol_name in emitted:
            continue
        emitted.add(symbol_name)
        candidate = candidates_by_profiler.get(symbol_name)
        if isinstance(candidate, Mapping) and str(candidate.get("status", "")) == _CUTLASS_CONV_RUNTIME_STATUS:
            lines.append(_cutlass_conv_runtime_profiler_source(symbol_name, candidate))
        else:
            lines.append(_cutlass_conv_stub_profiler_source(symbol_name))
    return "\n\n".join(lines) if lines else "// no CUTLASS Conv stub exports requested"


def _cutlass_conv_stub_launcher_source(symbol: str) -> str:
    return f"""extern "C" int {symbol}(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {{
  (void)activation_nhwc;
  (void)weight_ohwi;
  (void)bias;
  (void)output_nhwc;
  (void)n;
  (void)h;
  (void)w;
  (void)c;
  (void)out_h;
  (void)out_w;
  (void)out_c;
  (void)kernel_h;
  (void)kernel_w;
  (void)stride_h;
  (void)stride_w;
  (void)pad_h;
  (void)pad_w;
  (void)dilation_h;
  (void)dilation_w;
  (void)stream;
  return {_CUTLASS_CONV_STUB_RETURN_CODE};
}}"""


def _cutlass_conv_runtime_launcher_source(symbol: str, candidate: Mapping[str, Any]) -> str:
    dtype = str(candidate.get("dtype", ""))
    iterator_algorithm = str(candidate.get("cutlass", {}).get("iterator_algorithm", "analytic"))
    if dtype == "float32":
        if iterator_algorithm != "analytic":
            raise ValueError(f"Unsupported CUTLASS Conv float32 launcher iterator algorithm {iterator_algorithm!r}")
        launch_function = "dinoml_cutlass_conv_launch_fp32_simt_implicit_gemm_bias"
    elif iterator_algorithm == "few_channels":
        launch_function = "dinoml_cutlass_conv_launch_fp16_tensorop_few_channels_bias"
    elif iterator_algorithm == "fixed_channels":
        input_channels = int(candidate.get("selection_predicate", {}).get("input_channels", 0) or 0)
        if input_channels not in _CUTLASS_CONV_FIXED_CHANNELS_SYMBOL_IDS:
            raise ValueError(f"Unsupported CUTLASS Conv fixed-channel launcher input channel count {input_channels}")
        launch_function = f"dinoml_cutlass_conv_launch_fp16_tensorop_fixed_channels_c{input_channels}_bias"
    elif iterator_algorithm == "optimized":
        launch_function = "dinoml_cutlass_conv_launch_fp16_tensorop_optimized_bias"
    else:
        launch_function = "dinoml_cutlass_conv_launch_fp16_simt_implicit_gemm_bias"
    return f"""extern "C" int {symbol}(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {{
  if (activation_nhwc == nullptr || weight_ohwi == nullptr || bias == nullptr || output_nhwc == nullptr) {{
    return static_cast<int>(cudaErrorInvalidValue);
  }}
  if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || out_h <= 0 || out_w <= 0 || out_c <= 0 ||
      kernel_h <= 0 || kernel_w <= 0 || stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0 ||
      pad_h < 0 || pad_w < 0) {{
    return static_cast<int>(cudaErrorInvalidValue);
  }}
  return {launch_function}(
      activation_nhwc,
      weight_ohwi,
      bias,
      output_nhwc,
      n,
      h,
      w,
      c,
      out_h,
      out_w,
      out_c,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      stream);
}}"""


def _cutlass_conv_stub_profiler_source(symbol: str) -> str:
    return f"""extern "C" float {symbol}(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int iterations,
    cudaStream_t stream) {{
  (void)activation_nhwc;
  (void)weight_ohwi;
  (void)bias;
  (void)output_nhwc;
  (void)n;
  (void)h;
  (void)w;
  (void)c;
  (void)out_h;
  (void)out_w;
  (void)out_c;
  (void)kernel_h;
  (void)kernel_w;
  (void)stride_h;
  (void)stride_w;
  (void)pad_h;
  (void)pad_w;
  (void)dilation_h;
  (void)dilation_w;
  (void)iterations;
  (void)stream;
  return {_CUTLASS_CONV_STUB_PROFILE_MS}f;
}}"""


def _cutlass_conv_runtime_profiler_source(symbol: str, candidate: Mapping[str, Any]) -> str:
    launcher_symbol = str(candidate.get("kernel_symbol", ""))
    if not launcher_symbol:
        raise ValueError(f"CUTLASS Conv runtime profiler {symbol!r} is missing a launcher symbol")
    return f"""extern "C" float {symbol}(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int iterations,
    cudaStream_t stream) {{
  if (iterations <= 0) {{
    return {_CUTLASS_CONV_STUB_PROFILE_MS}f;
  }}
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaError_t event_status = cudaEventCreate(&start);
  if (event_status != cudaSuccess) {{
    return {_CUTLASS_CONV_STUB_PROFILE_MS}f;
  }}
  event_status = cudaEventCreate(&stop);
  if (event_status != cudaSuccess) {{
    cudaEventDestroy(start);
    return {_CUTLASS_CONV_STUB_PROFILE_MS}f;
  }}
  cudaStream_t profile_stream = stream;
  cudaEventRecord(start, profile_stream);
  for (int iter = 0; iter < iterations; ++iter) {{
    int status = {launcher_symbol}(
        activation_nhwc,
        weight_ohwi,
        bias,
        output_nhwc,
        n,
        h,
        w,
        c,
        out_h,
        out_w,
        out_c,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        profile_stream);
    if (status != 0) {{
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return {_CUTLASS_CONV_STUB_PROFILE_MS}f;
    }}
  }}
  cudaEventRecord(stop, profile_stream);
  event_status = cudaEventSynchronize(stop);
  if (event_status != cudaSuccess) {{
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {_CUTLASS_CONV_STUB_PROFILE_MS}f;
  }}
  float elapsed_ms = 0.0f;
  event_status = cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (event_status != cudaSuccess) {{
    return {_CUTLASS_CONV_STUB_PROFILE_MS}f;
  }}
  return elapsed_ms;
}}"""


def _cutlass_conv_transform_runtime_support_source() -> str:
    return """#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/tensor.h"

namespace {

template <typename T>
__global__ void dinoml_cutlass_conv_nchw_to_nhwc_kernel(
    const T* src,
    T* dst,
    int n,
    int c,
    int h,
    int w) {
  int linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int total = n * c * h * w;
  if (linear >= total) {
    return;
  }
  int x = linear % w;
  int tmp = linear / w;
  int y = tmp % h;
  tmp /= h;
  int channel = tmp % c;
  int batch = tmp / c;
  int dst_index = ((batch * h + y) * w + x) * c + channel;
  dst[dst_index] = src[linear];
}

template <typename T>
__global__ void dinoml_cutlass_conv_nhwc_to_nchw_kernel(
    const T* src,
    T* dst,
    int n,
    int c,
    int h,
    int w) {
  int linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int total = n * c * h * w;
  if (linear >= total) {
    return;
  }
  int x = linear % w;
  int tmp = linear / w;
  int y = tmp % h;
  tmp /= h;
  int channel = tmp % c;
  int batch = tmp / c;
  int src_index = ((batch * h + y) * w + x) * c + channel;
  dst[linear] = src[src_index];
}

template <typename T>
__global__ void dinoml_cutlass_conv_oihw_to_ohwi_kernel(
    const T* src,
    T* dst,
    int out_c,
    int in_c,
    int kernel_h,
    int kernel_w) {
  int linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int total = out_c * in_c * kernel_h * kernel_w;
  if (linear >= total) {
    return;
  }
  int kw = linear % kernel_w;
  int tmp = linear / kernel_w;
  int kh = tmp % kernel_h;
  tmp /= kernel_h;
  int in_channel = tmp % in_c;
  int out_channel = tmp / in_c;
  int dst_index = ((out_channel * kernel_h + kh) * kernel_w + kw) * in_c + in_channel;
  dst[dst_index] = src[linear];
}

template <typename T>
int dinoml_cutlass_conv_launch_nchw_to_nhwc(
    const void* src,
    void* dst,
    int n,
    int c,
    int h,
    int w,
    cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || n <= 0 || c <= 0 || h <= 0 || w <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int total = n * c * h * w;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  dinoml_cutlass_conv_nchw_to_nhwc_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const T*>(src),
      static_cast<T*>(dst),
      n,
      c,
      h,
      w);
  return static_cast<int>(cudaGetLastError());
}

template <typename T>
int dinoml_cutlass_conv_launch_nhwc_to_nchw(
    const void* src,
    void* dst,
    int n,
    int c,
    int h,
    int w,
    cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || n <= 0 || c <= 0 || h <= 0 || w <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int total = n * c * h * w;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  dinoml_cutlass_conv_nhwc_to_nchw_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const T*>(src),
      static_cast<T*>(dst),
      n,
      c,
      h,
      w);
  return static_cast<int>(cudaGetLastError());
}

template <typename T>
int dinoml_cutlass_conv_launch_oihw_to_ohwi(
    const void* src,
    void* dst,
    int out_c,
    int in_c,
    int kernel_h,
    int kernel_w,
    cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || out_c <= 0 || in_c <= 0 || kernel_h <= 0 || kernel_w <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int total = out_c * in_c * kernel_h * kernel_w;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  dinoml_cutlass_conv_oihw_to_ohwi_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const T*>(src),
      static_cast<T*>(dst),
      out_c,
      in_c,
      kernel_h,
      kernel_w);
  return static_cast<int>(cudaGetLastError());
}

using DinomlCutlassConvFp16Element = cutlass::half_t;
using DinomlCutlassConvFp16Accumulator = float;
using DinomlCutlassConvFp16Compute = float;
using DinomlCutlassConvFp16Layout = cutlass::layout::TensorNHWC;
using DinomlCutlassConvFp16MmaOp = cutlass::arch::OpClassSimt;
using DinomlCutlassConvFp16SmArch = cutlass::arch::Sm80;
using DinomlCutlassConvFp16ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 8>;
using DinomlCutlassConvFp16WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
using DinomlCutlassConvFp16InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
using DinomlCutlassConvFp16Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using DinomlCutlassConvFp16Epilogue = cutlass::epilogue::thread::LinearCombination<
    DinomlCutlassConvFp16Element,
    1,
    DinomlCutlassConvFp16Accumulator,
    DinomlCutlassConvFp16Compute>;
using DinomlCutlassConvFp16Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    DinomlCutlassConvFp16Element,
    DinomlCutlassConvFp16Layout,
    DinomlCutlassConvFp16Element,
    DinomlCutlassConvFp16Layout,
    DinomlCutlassConvFp16Element,
    DinomlCutlassConvFp16Layout,
    DinomlCutlassConvFp16Accumulator,
    DinomlCutlassConvFp16MmaOp,
    DinomlCutlassConvFp16SmArch,
    DinomlCutlassConvFp16ThreadblockShape,
    DinomlCutlassConvFp16WarpShape,
    DinomlCutlassConvFp16InstructionShape,
    DinomlCutlassConvFp16Epilogue,
    DinomlCutlassConvFp16Swizzle,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic,
    cutlass::conv::StrideSupport::kStrided,
    1,
    1>::Kernel;
using DinomlCutlassConvFp16ImplicitGemm =
    cutlass::conv::device::ImplicitGemmConvolution<DinomlCutlassConvFp16Kernel>;

using DinomlCutlassConvFp16FewChannelsMmaOp = cutlass::arch::OpClassTensorOp;
using DinomlCutlassConvFp16FewChannelsThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using DinomlCutlassConvFp16FewChannelsWarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using DinomlCutlassConvFp16FewChannelsInstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using DinomlCutlassConvFp16FewChannelsEpilogue = cutlass::epilogue::thread::LinearCombination<
    DinomlCutlassConvFp16Element,
    1,
    DinomlCutlassConvFp16Accumulator,
    DinomlCutlassConvFp16Compute>;
using DinomlCutlassConvFp16FewChannelsKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    DinomlCutlassConvFp16Element,
    DinomlCutlassConvFp16Layout,
    DinomlCutlassConvFp16Element,
    DinomlCutlassConvFp16Layout,
    DinomlCutlassConvFp16Element,
    DinomlCutlassConvFp16Layout,
    DinomlCutlassConvFp16Accumulator,
    DinomlCutlassConvFp16FewChannelsMmaOp,
    DinomlCutlassConvFp16SmArch,
    DinomlCutlassConvFp16FewChannelsThreadblockShape,
    DinomlCutlassConvFp16FewChannelsWarpShape,
    DinomlCutlassConvFp16FewChannelsInstructionShape,
    DinomlCutlassConvFp16FewChannelsEpilogue,
    DinomlCutlassConvFp16Swizzle,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kFewChannels,
    cutlass::conv::StrideSupport::kStrided,
    1,
    1>::Kernel;
using DinomlCutlassConvFp16FewChannelsImplicitGemm =
    cutlass::conv::device::ImplicitGemmConvolution<DinomlCutlassConvFp16FewChannelsKernel>;

using DinomlCutlassConvFp16FixedChannelsMmaOp = cutlass::arch::OpClassTensorOp;
using DinomlCutlassConvFp16FixedChannelsThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
using DinomlCutlassConvFp16FixedChannelsWarpShape = cutlass::gemm::GemmShape<64, 16, 32>;
using DinomlCutlassConvFp16FixedChannelsInstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using DinomlCutlassConvFp16FixedChannelsEpilogue = cutlass::epilogue::thread::LinearCombination<
    DinomlCutlassConvFp16Element,
    1,
    DinomlCutlassConvFp16Accumulator,
    DinomlCutlassConvFp16Compute>;
template <int ChannelCount>
using DinomlCutlassConvFp16FixedChannelsKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    DinomlCutlassConvFp16Element,
    DinomlCutlassConvFp16Layout,
    DinomlCutlassConvFp16Element,
    DinomlCutlassConvFp16Layout,
    DinomlCutlassConvFp16Element,
    DinomlCutlassConvFp16Layout,
    DinomlCutlassConvFp16Accumulator,
    DinomlCutlassConvFp16FixedChannelsMmaOp,
    DinomlCutlassConvFp16SmArch,
    DinomlCutlassConvFp16FixedChannelsThreadblockShape,
    DinomlCutlassConvFp16FixedChannelsWarpShape,
    DinomlCutlassConvFp16FixedChannelsInstructionShape,
    DinomlCutlassConvFp16FixedChannelsEpilogue,
    DinomlCutlassConvFp16Swizzle,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kFixedChannels,
    cutlass::conv::StrideSupport::kStrided,
    ChannelCount,
    ChannelCount>::Kernel;

using DinomlCutlassConvFp16OptimizedMmaOp = cutlass::arch::OpClassTensorOp;
using DinomlCutlassConvFp16OptimizedThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using DinomlCutlassConvFp16OptimizedWarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using DinomlCutlassConvFp16OptimizedInstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using DinomlCutlassConvFp16OptimizedEpilogue = cutlass::epilogue::thread::LinearCombination<
    DinomlCutlassConvFp16Element,
    1,
    DinomlCutlassConvFp16Accumulator,
    DinomlCutlassConvFp16Compute>;
using DinomlCutlassConvFp16OptimizedKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    DinomlCutlassConvFp16Element,
    DinomlCutlassConvFp16Layout,
    DinomlCutlassConvFp16Element,
    DinomlCutlassConvFp16Layout,
    DinomlCutlassConvFp16Element,
    DinomlCutlassConvFp16Layout,
    DinomlCutlassConvFp16Accumulator,
    DinomlCutlassConvFp16OptimizedMmaOp,
    DinomlCutlassConvFp16SmArch,
    DinomlCutlassConvFp16OptimizedThreadblockShape,
    DinomlCutlassConvFp16OptimizedWarpShape,
    DinomlCutlassConvFp16OptimizedInstructionShape,
    DinomlCutlassConvFp16OptimizedEpilogue,
    DinomlCutlassConvFp16Swizzle,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    8,
    8>::Kernel;
using DinomlCutlassConvFp16OptimizedImplicitGemm =
    cutlass::conv::device::ImplicitGemmConvolution<DinomlCutlassConvFp16OptimizedKernel>;

using DinomlCutlassConvFp32Element = float;
using DinomlCutlassConvFp32Accumulator = float;
using DinomlCutlassConvFp32Compute = float;
using DinomlCutlassConvFp32Layout = cutlass::layout::TensorNHWC;
using DinomlCutlassConvFp32MmaOp = cutlass::arch::OpClassSimt;
using DinomlCutlassConvFp32SmArch = cutlass::arch::Sm80;
using DinomlCutlassConvFp32ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 8>;
using DinomlCutlassConvFp32WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
using DinomlCutlassConvFp32InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
using DinomlCutlassConvFp32Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using DinomlCutlassConvFp32Epilogue = cutlass::epilogue::thread::LinearCombination<
    DinomlCutlassConvFp32Element,
    1,
    DinomlCutlassConvFp32Accumulator,
    DinomlCutlassConvFp32Compute>;
using DinomlCutlassConvFp32Kernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    DinomlCutlassConvFp32Element,
    DinomlCutlassConvFp32Layout,
    DinomlCutlassConvFp32Element,
    DinomlCutlassConvFp32Layout,
    DinomlCutlassConvFp32Element,
    DinomlCutlassConvFp32Layout,
    DinomlCutlassConvFp32Accumulator,
    DinomlCutlassConvFp32MmaOp,
    DinomlCutlassConvFp32SmArch,
    DinomlCutlassConvFp32ThreadblockShape,
    DinomlCutlassConvFp32WarpShape,
    DinomlCutlassConvFp32InstructionShape,
    DinomlCutlassConvFp32Epilogue,
    DinomlCutlassConvFp32Swizzle,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic,
    cutlass::conv::StrideSupport::kStrided,
    1,
    1>::Kernel;
using DinomlCutlassConvFp32ImplicitGemm =
    cutlass::conv::device::ImplicitGemmConvolution<DinomlCutlassConvFp32Kernel>;

template <typename ImplicitGemm>
int dinoml_cutlass_conv_launch_fp16_kernel_bias(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  cutlass::conv::Conv2dProblemSize problem_size(
      n,
      h,
      w,
      c,
      out_c,
      kernel_h,
      kernel_w,
      out_h,
      out_w,
      pad_h,
      pad_w,
      stride_h,
      stride_w,
      dilation_h,
      dilation_w,
      cutlass::conv::Mode::kCrossCorrelation,
      1,
      1);
  DinomlCutlassConvFp16Layout activation_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(n, h, w, c));
  DinomlCutlassConvFp16Layout weight_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(out_c, kernel_h, kernel_w, c));
  DinomlCutlassConvFp16Layout output_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(n, out_h, out_w, out_c));
  typename ImplicitGemm::Arguments arguments{
      problem_size,
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(activation_nhwc)),
       activation_layout},
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(weight_ohwi)),
       weight_layout},
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(bias)),
       DinomlCutlassConvFp16Layout::Stride(0)},
      {static_cast<DinomlCutlassConvFp16Element*>(output_nhwc), output_layout},
      {DinomlCutlassConvFp16Compute(1), DinomlCutlassConvFp16Compute(1)}};
  ImplicitGemm implicit_gemm;
  cutlass::Status status = implicit_gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return 1000 + static_cast<int>(status);
  }
  status = implicit_gemm.initialize(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return 1100 + static_cast<int>(status);
  }
  status = implicit_gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return 1200 + static_cast<int>(status);
  }
  return static_cast<int>(cudaGetLastError());
}

template <typename ImplicitGemm>
int dinoml_cutlass_conv_launch_fp32_kernel_bias(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  cutlass::conv::Conv2dProblemSize problem_size(
      n,
      h,
      w,
      c,
      out_c,
      kernel_h,
      kernel_w,
      out_h,
      out_w,
      pad_h,
      pad_w,
      stride_h,
      stride_w,
      dilation_h,
      dilation_w,
      cutlass::conv::Mode::kCrossCorrelation,
      1,
      1);
  DinomlCutlassConvFp32Layout activation_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(n, h, w, c));
  DinomlCutlassConvFp32Layout weight_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(out_c, kernel_h, kernel_w, c));
  DinomlCutlassConvFp32Layout output_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(n, out_h, out_w, out_c));
  typename ImplicitGemm::Arguments arguments{
      problem_size,
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(activation_nhwc)),
       activation_layout},
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(weight_ohwi)),
       weight_layout},
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(bias)),
       DinomlCutlassConvFp32Layout::Stride(0)},
      {static_cast<DinomlCutlassConvFp32Element*>(output_nhwc), output_layout},
      {DinomlCutlassConvFp32Compute(1), DinomlCutlassConvFp32Compute(1)}};
  ImplicitGemm implicit_gemm;
  cutlass::Status status = implicit_gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return 1000 + static_cast<int>(status);
  }
  status = implicit_gemm.initialize(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return 1100 + static_cast<int>(status);
  }
  status = implicit_gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return 1200 + static_cast<int>(status);
  }
  return static_cast<int>(cudaGetLastError());
}

int dinoml_cutlass_conv_launch_fp16_simt_implicit_gemm_bias(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  return dinoml_cutlass_conv_launch_fp16_kernel_bias<DinomlCutlassConvFp16ImplicitGemm>(
      activation_nhwc,
      weight_ohwi,
      bias,
      output_nhwc,
      n,
      h,
      w,
      c,
      out_h,
      out_w,
      out_c,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      stream);
}

int dinoml_cutlass_conv_launch_fp32_simt_implicit_gemm_bias(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  return dinoml_cutlass_conv_launch_fp32_kernel_bias<DinomlCutlassConvFp32ImplicitGemm>(
      activation_nhwc,
      weight_ohwi,
      bias,
      output_nhwc,
      n,
      h,
      w,
      c,
      out_h,
      out_w,
      out_c,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      stream);
}

int dinoml_cutlass_conv_launch_fp16_tensorop_few_channels_bias(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  if (c != 3) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return dinoml_cutlass_conv_launch_fp16_kernel_bias<DinomlCutlassConvFp16FewChannelsImplicitGemm>(
      activation_nhwc,
      weight_ohwi,
      bias,
      output_nhwc,
      n,
      h,
      w,
      c,
      out_h,
      out_w,
      out_c,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      stream);
}

template <int ChannelCount>
int dinoml_cutlass_conv_launch_fp16_tensorop_fixed_channels_bias(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  if (c != ChannelCount) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<
      DinomlCutlassConvFp16FixedChannelsKernel<ChannelCount>>;
  return dinoml_cutlass_conv_launch_fp16_kernel_bias<ImplicitGemm>(
      activation_nhwc,
      weight_ohwi,
      bias,
      output_nhwc,
      n,
      h,
      w,
      c,
      out_h,
      out_w,
      out_c,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      stream);
}

int dinoml_cutlass_conv_launch_fp16_tensorop_fixed_channels_c4_bias(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  return dinoml_cutlass_conv_launch_fp16_tensorop_fixed_channels_bias<4>(
      activation_nhwc,
      weight_ohwi,
      bias,
      output_nhwc,
      n,
      h,
      w,
      c,
      out_h,
      out_w,
      out_c,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      stream);
}

int dinoml_cutlass_conv_launch_fp16_tensorop_fixed_channels_c8_bias(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  return dinoml_cutlass_conv_launch_fp16_tensorop_fixed_channels_bias<8>(
      activation_nhwc,
      weight_ohwi,
      bias,
      output_nhwc,
      n,
      h,
      w,
      c,
      out_h,
      out_w,
      out_c,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      stream);
}

int dinoml_cutlass_conv_launch_fp16_tensorop_optimized_bias(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  if (c < 16 || (c % 8) != 0 || (out_c % 8) != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return dinoml_cutlass_conv_launch_fp16_kernel_bias<DinomlCutlassConvFp16OptimizedImplicitGemm>(
      activation_nhwc,
      weight_ohwi,
      bias,
      output_nhwc,
      n,
      h,
      w,
      c,
      out_h,
      out_w,
      out_c,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      stream);
}

}  // namespace"""


def _cutlass_conv_transform_source(helper: Mapping[str, Any]) -> str:
    symbol = str(helper["symbol"])
    dtype = str(helper["dtype"])
    transform = str(helper["transform"])
    ctype = "__half" if dtype == "float16" else "float"
    if transform == "nchw_to_nhwc_temporary":
        return f"""extern "C" int {symbol}(
    const void* src_nchw,
    void* dst_nhwc,
    int n,
    int c,
    int h,
    int w,
    cudaStream_t stream) {{
  return dinoml_cutlass_conv_launch_nchw_to_nhwc<{ctype}>(
      src_nchw, dst_nhwc, n, c, h, w, stream);
}}"""
    if transform == "nhwc_to_nchw_temporary":
        return f"""extern "C" int {symbol}(
    const void* src_nhwc,
    void* dst_nchw,
    int n,
    int c,
    int h,
    int w,
    cudaStream_t stream) {{
  return dinoml_cutlass_conv_launch_nhwc_to_nchw<{ctype}>(
      src_nhwc, dst_nchw, n, c, h, w, stream);
}}"""
    if transform == "oihw_to_ohwi_temporary":
        return f"""extern "C" int {symbol}(
    const void* src_oihw,
    void* dst_ohwi,
    int out_c,
    int in_c,
    int kernel_h,
    int kernel_w,
    cudaStream_t stream) {{
  return dinoml_cutlass_conv_launch_oihw_to_ohwi<{ctype}>(
      src_oihw, dst_ohwi, out_c, in_c, kernel_h, kernel_w, stream);
}}"""
    raise ValueError(f"Unsupported CUTLASS Conv transform helper {transform!r}")


def _validate_conv_op_name(op_name: str) -> None:
    if op_name not in CONV_OPS:
        supported = ", ".join(CONV_OPS)
        raise ValueError(f"Unsupported CUTLASS conv op {op_name!r}; supported ops: {supported}")


def _conv_candidate_status(dtype: str) -> str:
    _normalize_conv_dtype(dtype)
    return _CUTLASS_CONV_RUNTIME_STATUS


def _conv_plan_status(
    dtype: str,
    *,
    input_shape: Any,
    weight_shape: Any,
    output_shape: Any,
    conv_config: Mapping[str, Any],
) -> str:
    _normalize_conv_dtype(dtype)
    return _CUTLASS_CONV_RUNTIME_STATUS


def _conv_plan_status_payload(status: str) -> dict[str, Any]:
    if status == _CUTLASS_CONV_RUNTIME_STATUS:
        return {
            "runtime": {
                "status": _CUTLASS_CONV_RUNTIME_STATUS,
                "launcher": "cutlass_implicit_gemm_conv2d_fprop_bias",
            },
            "profiler_status": _CUTLASS_CONV_RUNTIME_PROFILER_STATUS,
        }
    return {"blocked_reason": _CUTLASS_CONV_SCAFFOLD_BLOCKED_REASON}


def _cutlass_conv_exact_runtime_slice_compatible(
    exact_slice: Any,
    *,
    input_shape: Any,
    weight_shape: Any,
    output_shape: Any,
    conv_config: Mapping[str, Any],
) -> bool:
    if not isinstance(exact_slice, Mapping):
        return False
    input_dims = _shape_list_or_none(input_shape)
    weight_dims = _shape_list_or_none(weight_shape)
    output_dims = _shape_list_or_none(output_shape)
    if input_dims is None or weight_dims is None or output_dims is None:
        return False
    if not _shape_matches_template(input_dims, exact_slice.get("activation_shape")):
        return False
    if not _shape_matches_template(weight_dims, exact_slice.get("weight_shape")):
        return False
    if not _shape_matches_template(output_dims, exact_slice.get("output_shape")):
        return False
    if "batch" in exact_slice.get("activation_shape", ()) and input_dims[0] != output_dims[0]:
        return False
    checks = (
        ("stride", "stride"),
        ("padding", "padding"),
        ("dilation", "dilation"),
    )
    for field, config_field in checks:
        expected = _shape_list_or_none(exact_slice.get(field))
        actual = _shape_list_or_none(conv_config.get(config_field))
        if expected is None or actual != expected:
            return False
    return int(conv_config.get("groups", -1) or -1) == int(exact_slice.get("groups", -1) or -1)


def _shape_list_or_none(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)):
        return None
    result = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            return None
    return result


def _shape_matches_template(actual: list[int], template: Any) -> bool:
    if not isinstance(template, (list, tuple)) or len(actual) != len(template):
        return False
    for actual_dim, expected_dim in zip(actual, template, strict=True):
        if expected_dim == "batch":
            if actual_dim <= 0:
                return False
            continue
        if isinstance(expected_dim, bool):
            return False
        try:
            if actual_dim != int(expected_dim):
                return False
        except (TypeError, ValueError):
            return False
    return True


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


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


__all__ = [
    "CONV_OPS",
    "CONV_SUPPORTED_DTYPES",
    "CUTLASS_CONV_CANDIDATE_SET_SCHEMA_VERSION",
    "CUTLASS_CONV_USED_CANDIDATE_PLAN_SCHEMA_VERSION",
    "cutlass_conv_candidate_set",
    "cutlass_conv_candidate_set_id",
    "cutlass_conv_candidate_compatible_with_plan",
    "cutlass_conv_candidates",
    "cutlass_conv_input_pack_symbol",
    "cutlass_conv_layout_plan",
    "cutlass_conv_output_unpack_symbol",
    "cutlass_conv_profiler_symbol",
    "cutlass_conv_symbol",
    "cutlass_conv_used_candidate_plan",
    "cutlass_conv_wrapper_stages",
    "cutlass_conv_weight_pack_symbol",
    "normalize_cutlass_conv_used_candidate_plan",
    "render_cutlass_conv_scaffold_source",
    "validate_cutlass_conv_plan",
    "validate_cutlass_conv_scaffold_plan",
]
