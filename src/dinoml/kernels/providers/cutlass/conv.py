from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.ir import canonical_json, dtype_nbytes


CUTLASS_CONV_CANDIDATE_SET_SCHEMA_VERSION = 1
CUTLASS_CONV_USED_CANDIDATE_PLAN_SCHEMA_VERSION = 1
CONV_OPS = ("conv2d_bias",)
CONV_SUPPORTED_DTYPES = ("float16", "float32")
_CUTLASS_CONV_DEFAULT_SYMBOL_ID = "scaffold_sm80_nhwc_ohwi_bias"
_CUTLASS_CONV_SCAFFOLD_KIND = "cutlass_conv2d_bias_manifest_scaffold"
_CUTLASS_CONV_SCAFFOLD_STATUS = "manifest_scaffold_only"
_CUTLASS_CONV_SCAFFOLD_BLOCKED_REASON = "cutlass_conv_runtime_launcher_not_implemented"
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
    return validate_cutlass_conv_scaffold_plan(
        {
        "schema_version": 1,
        "kind": _CUTLASS_CONV_SCAFFOLD_KIND,
        "status": _CUTLASS_CONV_SCAFFOLD_STATUS,
        "blocked_reason": _CUTLASS_CONV_SCAFFOLD_BLOCKED_REASON,
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
    if str(payload.get("status")) != _CUTLASS_CONV_SCAFFOLD_STATUS:
        raise ValueError(f"Unsupported CUTLASS Conv scaffold status {payload.get('status')!r}")
    if str(payload.get("blocked_reason")) != _CUTLASS_CONV_SCAFFOLD_BLOCKED_REASON:
        raise ValueError(
            "CUTLASS Conv scaffold blocked_reason must record "
            f"{_CUTLASS_CONV_SCAFFOLD_BLOCKED_REASON!r}"
        )
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
    if int(layout_translation.get("output_unpack_nbytes", -1)) != expected_output_nbytes:
        raise ValueError(
            "CUTLASS Conv scaffold output_unpack_nbytes mismatch: "
            f"expected {expected_output_nbytes}, got {layout_translation.get('output_unpack_nbytes')!r}"
        )
    weight_transform = dict(payload.get("weight_transform", {}))
    expected_weight_nbytes = _nbytes(weight_shape, dtype_size)
    if weight_transform.get("from") != "oihw":
        raise ValueError("CUTLASS Conv scaffold weight_transform.from must be 'oihw'")
    if weight_transform.get("to") != "ohwi":
        raise ValueError("CUTLASS Conv scaffold weight_transform.to must be 'ohwi'")
    if weight_transform.get("pack") != "oihw_to_ohwi_temporary":
        raise ValueError("CUTLASS Conv scaffold weight_transform.pack must be 'oihw_to_ohwi_temporary'")
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
    payload["dtype"] = dtype
    payload["semantic_layout"] = semantic_layout
    payload["provider_layout"] = provider_layout
    payload["layout_translation"] = layout_translation
    payload["weight_transform"] = weight_transform
    payload["conv_config"] = {"stride": stride, "padding": padding, "dilation": dilation, "groups": 1}
    payload["input_shape"] = x_shape
    payload["weight_shape"] = weight_shape
    payload["bias_shape"] = bias_shape
    payload["output_shape"] = output_shape
    payload["temporary_buffers"] = normalized_buffers
    payload["workspace_nbytes"] = 0
    payload["temporary_nbytes"] = expected_temporary_nbytes
    return payload


def validate_cutlass_conv_plan(
    plan: Mapping[str, Any] | None,
    *,
    candidate: Mapping[str, Any] | None = None,
    node_id: str | None = None,
) -> dict[str, Any]:
    # Compatibility alias for existing callers; this validator remains scoped to
    # the current NHWC/OHWI manifest-only scaffold.
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
    ):
        if payload.get(field) != expected.get(field):
            raise ValueError(
                f"CUTLASS Conv used candidate plan entry candidate_set.{field} mismatch: "
                f"expected {expected.get(field)!r}, got {payload.get(field)!r}"
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
        "optional",
        "target_policy",
        "status",
    ):
        if candidate_payload.get(field) != expected.get(field):
            raise ValueError(
                f"CUTLASS Conv used candidate plan entry candidate.{field} mismatch: "
                f"expected {expected.get(field)!r}, got {candidate_payload.get(field)!r}"
            )


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


def _cutlass_conv_stub_source_exports(used_candidate_plan: Mapping[str, Any]) -> str:
    lines: list[str] = []
    emitted: set[str] = set()
    for symbol in used_candidate_plan.get("kernel_symbols", ()):
        symbol_name = str(symbol)
        if not symbol_name or symbol_name in emitted:
            continue
        emitted.add(symbol_name)
        lines.append(_cutlass_conv_stub_launcher_source(symbol_name))
    for symbol in used_candidate_plan.get("profiler_symbols", ()):
        symbol_name = str(symbol)
        if not symbol_name or symbol_name in emitted:
            continue
        emitted.add(symbol_name)
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
    "cutlass_conv_candidates",
    "cutlass_conv_layout_plan",
    "cutlass_conv_profiler_symbol",
    "cutlass_conv_symbol",
    "cutlass_conv_used_candidate_plan",
    "normalize_cutlass_conv_used_candidate_plan",
    "render_cutlass_conv_scaffold_source",
    "validate_cutlass_conv_plan",
    "validate_cutlass_conv_scaffold_plan",
]
