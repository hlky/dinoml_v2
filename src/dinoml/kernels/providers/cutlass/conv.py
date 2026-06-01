from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.ir import canonical_json, dtype_nbytes
from dinoml.kernels.providers.cutlass.gemm import (
    CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE,
    cutlass_gemm_target_policy,
)


CUTLASS_CONV_CANDIDATE_SET_SCHEMA_VERSION = 1
CUTLASS_CONV_USED_CANDIDATE_PLAN_SCHEMA_VERSION = 1
CONV_OPS = ("conv2d_bias", "conv2d_bias_relu", "conv2d_bias_add", "conv2d_bias_add_relu")
CONV_SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")
_CUTLASS_CONV_DEFAULT_SYMBOL_ID = "conv_analytic_simt_sm80_f32_128x64x8_s2_w32x64x8_f32_align1"
_CUTLASS_CONV_PLAN_KIND = "cutlass_conv2d_bias_plan"
_CUTLASS_CONV_STATUS = "runtime"
_CUTLASS_CONV_PROFILER_STATUS = "runtime_profiler"
_CUTLASS_CONV_TRANSFORM_ABI = "dinoml_cutlass_layout_transform_v1"
_CUTLASS_CONV_BASE_SEMANTIC_LAYOUT = {
    "activation": "nchw",
    "weight": "oihw",
    "bias": "o",
    "output": "nchw",
}
_CUTLASS_CONV_BASE_PROVIDER_LAYOUT = {
    "activation": "nhwc",
    "weight": "ohwi",
    "bias": "o",
    "output": "nhwc",
}
_CUTLASS_CONV_EPILOGUE_BY_OP = {
    "conv2d_bias": "bias",
    "conv2d_bias_relu": "bias_relu",
    "conv2d_bias_add": "bias_add",
    "conv2d_bias_add_relu": "bias_add_relu",
}
_CUTLASS_CONV_RUNTIME_LAUNCHER_BY_OP = {
    "conv2d_bias": "cutlass_implicit_gemm_conv2d_fprop_bias",
    "conv2d_bias_relu": "cutlass_implicit_gemm_conv2d_fprop_bias_relu",
    "conv2d_bias_add": "cutlass_implicit_gemm_conv2d_fprop_bias_add",
    "conv2d_bias_add_relu": "cutlass_implicit_gemm_conv2d_fprop_bias_add_relu",
}
_CUTLASS_CONV_LAUNCH_ABI_BY_OP = {
    "conv2d_bias": "dinoml_cutlass_conv2d_bias_v1",
    "conv2d_bias_relu": "dinoml_cutlass_conv2d_bias_relu_v1",
    "conv2d_bias_add": "dinoml_cutlass_conv2d_bias_add_v1",
    "conv2d_bias_add_relu": "dinoml_cutlass_conv2d_bias_add_relu_v1",
}


def cutlass_conv_symbol(op_name: str, dtype: str, symbol_id: str | None = None) -> str:
    normalized_dtype = _normalize_conv_dtype(dtype)
    symbol_suffix = symbol_id or _CUTLASS_CONV_DEFAULT_SYMBOL_ID
    return f"dinoml_cutlass_{op_name}_{normalized_dtype}_{symbol_suffix}"


def cutlass_conv_profiler_symbol(op_name: str, dtype: str, symbol_id: str | None = None) -> str:
    normalized_dtype = _normalize_conv_dtype(dtype)
    symbol_suffix = symbol_id or _CUTLASS_CONV_DEFAULT_SYMBOL_ID
    return f"dinoml_profile_cutlass_{op_name}_{normalized_dtype}_{symbol_suffix}"


def cutlass_conv_static_library_name(op_name: str, dtype: str) -> str:
    _validate_conv_op_name(op_name)
    normalized_dtype = _normalize_conv_dtype(dtype)
    return f"libdinoml_cutlass_{op_name}_{normalized_dtype}.a"


def cutlass_conv_cmake_target(op_name: str, dtype: str) -> str:
    _validate_conv_op_name(op_name)
    normalized_dtype = _normalize_conv_dtype(dtype)
    return f"dinoml_cutlass_conv_{op_name}_{normalized_dtype}"


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
    return f"cutlass_conv_{op_name}_{normalized_dtype}_nhwc_ohwi_{_conv_epilogue(op_name)}_v1"


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
    candidates = []
    for config in _cutlass_conv_candidate_configs_for_target(normalized_dtype, target):
        cutlass_config = dict(config["cutlass"])
        if cutlass_config["opclass"] == "simt":
            candidates.append(
                _cutlass_conv_candidate_from_gemm_config(
                    op_name,
                    normalized_dtype,
                    config,
                    iterator_algorithm="analytic",
                    align_a=1,
                    align_b=1,
                    target_policy=normalized_target,
                    status=status,
                    selection_predicate={"kind": "fallback"},
                    optional=bool(config.get("optional", False)),
                )
            )
            continue
        if normalized_dtype not in {"float16", "bfloat16"}:
            continue
        candidates.append(
            _cutlass_conv_candidate_from_gemm_config(
                op_name,
                normalized_dtype,
                config,
                iterator_algorithm="optimized",
                align_a=int(cutlass_config["align"]),
                align_b=int(cutlass_config["align"]),
                target_policy=normalized_target,
                status=status,
                selection_predicate={
                    "kind": "natural_alignment",
                    "dtype": normalized_dtype,
                    "groups": 1,
                    "min_input_channels": int(cutlass_config["align"]),
                    "input_channels_multiple": int(cutlass_config["align"]),
                    "output_channels_multiple": int(cutlass_config["align"]),
                    "requires_layout_translation": "nchw_oihw_to_nhwc_ohwi",
                    "padding_policy": "none",
                },
                optional=bool(config.get("optional", False)),
            )
        )
        if int(cutlass_config["align"]) in {4, 8}:
            channel_count = int(cutlass_config["align"])
            candidates.append(
                _cutlass_conv_candidate_from_gemm_config(
                    op_name,
                    normalized_dtype,
                    config,
                    iterator_algorithm="fixed_channels",
                    align_a=channel_count,
                    align_b=channel_count,
                    target_policy=normalized_target,
                    status=status,
                    selection_predicate={
                        "kind": "semantic_input_channels",
                        "input_channels": channel_count,
                        "dtype": normalized_dtype,
                        "groups": 1,
                        "requires_layout_translation": "nchw_oihw_to_nhwc_ohwi",
                        "padding_policy": "none",
                    },
                    optional=True,
                )
            )
        candidates.append(
            _cutlass_conv_candidate_from_gemm_config(
                op_name,
                normalized_dtype,
                config,
                iterator_algorithm="few_channels",
                align_a=1,
                align_b=1,
                target_policy=normalized_target,
                status=status,
                selection_predicate={
                    "kind": "semantic_input_channels",
                    "input_channels": 3,
                    "dtype": normalized_dtype,
                    "groups": 1,
                    "requires_layout_translation": "nchw_oihw_to_nhwc_ohwi",
                    "padding_policy": "none",
                },
                optional=True,
            )
        )
    return candidates


def _cutlass_conv_candidate_configs_for_target(
    dtype: str,
    target: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    configs = tuple(
        config
        for config in CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE[dtype]
        if str(config.get("cutlass", {}).get("math_operator", "multiply_add")) == "multiply_add"
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


def _cutlass_conv_candidate_from_gemm_config(
    op_name: str,
    dtype: str,
    candidate_config: Mapping[str, Any],
    *,
    iterator_algorithm: str,
    align_a: int,
    align_b: int,
    target_policy: Mapping[str, Any],
    status: str,
    selection_predicate: Mapping[str, Any],
    optional: bool,
) -> dict[str, Any]:
    base_cutlass = dict(candidate_config["cutlass"])
    cutlass_config = {
        **base_cutlass,
        "api": "device_implicit_gemm_convolution",
        "iterator_algorithm": iterator_algorithm,
        "instruction_shape": list(base_cutlass.get("instruction", base_cutlass.get("instruction_shape", []))),
        "align_a": int(align_a),
        "align_b": int(align_b),
        "kind": "implicit_gemm_runtime_launcher",
        "policy_source": "cutlass_gemm_candidate_config",
        "base_gemm_symbol_id": str(candidate_config["symbol_id"]),
    }
    symbol_id = f"conv_{iterator_algorithm}_{candidate_config['symbol_id']}_a{int(align_a)}b{int(align_b)}"
    return _cutlass_conv_candidate(
        op_name,
        dtype,
        symbol_id=symbol_id,
        target_policy=target_policy,
        status=status,
        accumulator_dtype=str(candidate_config["accumulator_dtype"]),
        cutlass=cutlass_config,
        selection_predicate=selection_predicate,
        optional=optional,
    )


def cutlass_conv_candidate_compatible_with_plan(
    candidate: Mapping[str, Any],
    cutlass_conv_plan: Mapping[str, Any],
) -> bool:
    op_family = str(cutlass_conv_plan.get("op_family", ""))
    if op_family:
        try:
            expected_epilogue = _conv_epilogue(op_family)
        except ValueError:
            return False
        if str(candidate.get("epilogue", "")) != expected_epilogue:
            return False
        if dict(candidate.get("epilogue_config", {})) != _conv_epilogue_config(op_family):
            return False
        if str(candidate.get("launch_abi", "")) != _cutlass_conv_launch_abi(op_family):
            return False
    dtype = str(cutlass_conv_plan.get("dtype", ""))
    if str(candidate.get("dtype", "")) != dtype:
        return False
    candidate_layouts = dict(candidate.get("layouts", {}))
    semantic_layout = dict(cutlass_conv_plan.get("semantic_layout", {}))
    provider_layout = dict(cutlass_conv_plan.get("provider_layout", {}))
    expected_layouts = _cutlass_conv_candidate_layouts(op_family or str(candidate.get("op", "conv2d_bias")))
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
        "epilogue": _conv_epilogue(op_name),
        "epilogue_config": _conv_epilogue_config(op_name),
        "launch_abi": _cutlass_conv_launch_abi(op_name),
        "layouts": _cutlass_conv_candidate_layouts(op_name),
        "cutlass": dict(cutlass),
        "selection_predicate": dict(selection_predicate),
        "optional": bool(optional),
        "target_policy": dict(target_policy),
        "status": status,
        "profiler_status": _CUTLASS_CONV_PROFILER_STATUS,
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
    status = _CUTLASS_CONV_STATUS
    normalized_target = _normalize_target_policy(target)
    config = {
        "schema_version": CUTLASS_CONV_CANDIDATE_SET_SCHEMA_VERSION,
        "provider": "cutlass",
        "family": "conv2d_fprop",
        "op": op_name,
        "dtype": normalized_dtype,
        "accumulator_dtype": "float32",
        "epilogue": _conv_epilogue(op_name),
        "epilogue_config": _conv_epilogue_config(op_name),
        "launch_abi": _cutlass_conv_launch_abi(op_name),
        "semantic_layout": _cutlass_conv_semantic_layout(op_name),
        "provider_layout": _cutlass_conv_provider_layout(op_name),
        "supported_groups": [1],
        "supported_dtypes": list(CONV_SUPPORTED_DTYPES),
        "candidate_count": len(candidates),
        "candidate_config_keys": [str(candidate["candidate_config_key"]) for candidate in candidates],
        "target_policy": normalized_target,
        "status": status,
        "profiler_status": _CUTLASS_CONV_PROFILER_STATUS,
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
        conv_plan_payload = validate_cutlass_conv_plan(
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


def render_cutlass_conv_source(source_text: str, used_candidate_plan: Mapping[str, Any]) -> str:
    exports = _cutlass_conv_source_exports(used_candidate_plan)
    marker = "// DINOML_CUTLASS_CONV_EXPORTS"
    if marker not in source_text:
        raise ValueError("CUTLASS Conv source is missing the export marker")
    return source_text.replace(marker, exports)


def normalize_cutlass_conv_used_candidate_plan(used_candidate_plan: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(used_candidate_plan, Mapping):
        raise ValueError("CUTLASS Conv support requires a used candidate plan mapping")
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
    op_name = str(node.get("op", ""))
    _validate_conv_op_name(op_name)
    if op_name not in CONV_OPS:
        raise ValueError(f"Unsupported CUTLASS conv plan op {node.get('op')!r}")
    attrs = dict(node.get("attrs", {}))
    bridge_metadata = {
        key: str(attrs[key])
        for key in ("source_op", "bias_mode")
        if attrs.get(key) is not None
    }
    x_name, weight_name, bias_name = [str(name) for name in node.get("inputs", ())[:3]]
    residual_name = (
        str(node.get("inputs", (None, None, None, ""))[3]) if _cutlass_conv_op_has_residual(op_name) else None
    )
    output_name = str(node.get("outputs", ("",))[0])
    x_shape = [int(dim) for dim in tensor_map[x_name]["shape"]]
    weight_shape = [int(dim) for dim in tensor_map[weight_name]["shape"]]
    bias_shape = [int(dim) for dim in tensor_map[bias_name]["shape"]]
    residual_shape = None if residual_name is None else [int(dim) for dim in tensor_map[residual_name]["shape"]]
    output_shape = [int(dim) for dim in tensor_map[output_name]["shape"]]
    dtype = str(tensor_map[output_name]["dtype"])
    dtype_size = dtype_nbytes(dtype)
    weight_is_constant = str(tensor_map[weight_name].get("kind", "")) == "constant"
    stride = [int(item) for item in attrs.get("stride", (1, 1))]
    padding = [int(item) for item in attrs.get("padding", (0, 0))]
    dilation = [int(item) for item in attrs.get("dilation", (1, 1))]
    groups = int(attrs.get("groups", 1))
    temporary_buffers = [
        {"name": "activation_nhwc", "kind": "layout_pack", "layout": "nhwc", "nbytes": _nbytes(x_shape, dtype_size)},
        *(
            []
            if weight_is_constant
            else [{"name": "weight_ohwi", "kind": "layout_pack", "layout": "ohwi", "nbytes": _nbytes(weight_shape, dtype_size)}]
        ),
        *(
            [
                {
                    "name": "residual_nhwc",
                    "kind": "layout_pack",
                    "layout": "nhwc",
                    "nbytes": _nbytes(output_shape, dtype_size),
                }
            ]
            if residual_shape is not None
            else []
        ),
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
    return validate_cutlass_conv_plan(
        {
        "schema_version": 1,
        "kind": _CUTLASS_CONV_PLAN_KIND,
        "status": plan_status,
        **_conv_plan_status_payload(plan_status, op_name=op_name),
        "node_id": str(node.get("id", "")),
        "op_family": op_name,
        **bridge_metadata,
        "dtype": dtype,
        "epilogue": _conv_epilogue(op_name),
        "epilogue_config": _conv_epilogue_config(op_name),
        "semantic_layout": _cutlass_conv_semantic_layout(op_name),
        "provider_layout": _cutlass_conv_provider_layout(op_name),
        "layout_translation": {
            "input_pack": "nchw_to_nhwc_temporary",
            "output_unpack": "nhwc_to_nchw_temporary",
            "bias": "direct_per_output_channel",
            "input_pack_nbytes": _nbytes(x_shape, dtype_size),
            "output_unpack_nbytes": _nbytes(output_shape, dtype_size),
            "input_pack_symbol": cutlass_conv_input_pack_symbol(dtype),
            "output_unpack_symbol": cutlass_conv_output_unpack_symbol(dtype),
            **(
                {
                    "residual_pack": "nchw_to_nhwc_temporary",
                    "residual_pack_nbytes": _nbytes(output_shape, dtype_size),
                    "residual_pack_symbol": cutlass_conv_input_pack_symbol(dtype),
                }
                if residual_shape is not None
                else {}
            ),
        },
        "weight_transform": {
            "from": "oihw",
            "to": "ohwi",
            "pack": "constants_bin_prepacked" if weight_is_constant else "oihw_to_ohwi_temporary",
            "temporary_nbytes": _nbytes(weight_shape, dtype_size),
            **({} if weight_is_constant else {"pack_symbol": cutlass_conv_weight_pack_symbol(dtype)}),
            "runtime_persistent": bool(weight_is_constant),
            "channel_pad_multiple": 1,
            "padded_input_channels": int(weight_shape[1]),
            "padded_output_channels": int(weight_shape[0]),
            "padding_fill_value": 0.0,
        },
        "conv_config": conv_config,
        "input_shape": x_shape,
        "weight_shape": weight_shape,
        "bias_shape": bias_shape,
        **({"residual_shape": residual_shape} if residual_shape is not None else {}),
        "output_shape": output_shape,
        "workspace_nbytes": 0,
        "temporary_buffers": temporary_buffers,
        "temporary_nbytes": sum(int(buffer["nbytes"]) for buffer in temporary_buffers),
        },
        node_id=_optional_str(node.get("id")),
    )


def validate_cutlass_conv_plan(
    plan: Mapping[str, Any] | None,
    *,
    candidate: Mapping[str, Any] | None = None,
    node_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise ValueError("CUTLASS Conv requires cutlass_conv_plan transform metadata")
    payload = dict(plan)
    if str(payload.get("kind")) != _CUTLASS_CONV_PLAN_KIND:
        raise ValueError(f"Unsupported CUTLASS Conv kind {payload.get('kind')!r}")
    status = str(payload.get("status"))
    if status != _CUTLASS_CONV_STATUS:
        raise ValueError(f"Unsupported CUTLASS Conv status {payload.get('status')!r}")
    op_family = str(payload.get("op_family"))
    _validate_conv_op_name(op_family)
    source_op = _optional_str(payload.get("source_op"))
    bias_mode = _optional_str(payload.get("bias_mode"))
    if source_op is not None or bias_mode is not None:
        if op_family != "conv2d_bias" or source_op != "conv2d" or bias_mode != "explicit_zero_constant":
            raise ValueError(
                "CUTLASS Conv bridge metadata must be "
                "source_op='conv2d' and bias_mode='explicit_zero_constant' on conv2d_bias"
            )
    if node_id is not None and str(payload.get("node_id", "")) != node_id:
        raise ValueError(
            f"CUTLASS Conv node_id mismatch: expected {node_id!r}, got {payload.get('node_id')!r}"
        )
    dtype = _normalize_conv_dtype(str(payload.get("dtype")))
    epilogue = str(payload.get("epilogue", ""))
    if epilogue != _conv_epilogue(op_family):
        raise ValueError(
            f"CUTLASS Conv epilogue mismatch for {op_family!r}: expected {_conv_epilogue(op_family)!r}, got {epilogue!r}"
        )
    epilogue_config = dict(payload.get("epilogue_config", {}))
    if epilogue_config != _conv_epilogue_config(op_family):
        raise ValueError(
            "CUTLASS Conv epilogue_config mismatch: "
            f"expected {_conv_epilogue_config(op_family)!r}, got {epilogue_config!r}"
        )
    semantic_layout = dict(payload.get("semantic_layout", {}))
    provider_layout = dict(payload.get("provider_layout", {}))
    expected_semantic_layout = _cutlass_conv_semantic_layout(op_family)
    expected_provider_layout = _cutlass_conv_provider_layout(op_family)
    if semantic_layout != expected_semantic_layout:
        raise ValueError(
            f"CUTLASS Conv semantic_layout must be {expected_semantic_layout}, got {semantic_layout!r}"
        )
    if provider_layout != expected_provider_layout:
        raise ValueError(
            f"CUTLASS Conv provider_layout must be {expected_provider_layout}, got {provider_layout!r}"
        )
    x_shape = _validate_positive_shape(payload.get("input_shape"), rank=4, name="input_shape")
    weight_shape = _validate_positive_shape(payload.get("weight_shape"), rank=4, name="weight_shape")
    bias_shape = _validate_positive_shape(payload.get("bias_shape"), rank=1, name="bias_shape")
    residual_shape = (
        _validate_positive_shape(payload.get("residual_shape"), rank=4, name="residual_shape")
        if _cutlass_conv_op_has_residual(op_family)
        else None
    )
    output_shape = _validate_positive_shape(payload.get("output_shape"), rank=4, name="output_shape")
    dtype_size = dtype_nbytes(dtype)
    conv_config = dict(payload.get("conv_config", {}))
    stride = _validate_positive_shape(conv_config.get("stride"), rank=2, name="conv_config.stride")
    padding = _validate_non_negative_shape(conv_config.get("padding"), rank=2, name="conv_config.padding")
    dilation = _validate_positive_shape(conv_config.get("dilation"), rank=2, name="conv_config.dilation")
    groups = conv_config.get("groups")
    if not isinstance(groups, int) or isinstance(groups, bool) or groups != 1:
        raise ValueError(f"CUTLASS Conv currently requires conv_config.groups == 1, got {groups!r}")
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
            f"CUTLASS Conv status for dtype/shape slice must be {expected_status!r}, got {status!r}"
        )
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("CUTLASS Conv runtime plan requires runtime metadata")
    if str(runtime.get("status")) != _CUTLASS_CONV_STATUS:
        raise ValueError("CUTLASS Conv runtime metadata has an unexpected status")
    if str(runtime.get("launcher")) != _cutlass_conv_runtime_launcher_name(op_family):
        raise ValueError(
            "CUTLASS Conv runtime metadata has an unexpected launcher: "
            f"expected {_cutlass_conv_runtime_launcher_name(op_family)!r}, got {runtime.get('launcher')!r}"
        )
    if str(payload.get("profiler_status")) != _CUTLASS_CONV_PROFILER_STATUS:
        raise ValueError(
            "CUTLASS Conv runtime plan must record "
            f"profiler_status={_CUTLASS_CONV_PROFILER_STATUS!r}"
        )
    if weight_shape[1] != x_shape[1]:
        raise ValueError(
            "CUTLASS Conv weight/input channel mismatch for groups=1: "
            f"weight I={weight_shape[1]} vs input C={x_shape[1]}"
        )
    if bias_shape[0] != weight_shape[0] or output_shape[1] != weight_shape[0]:
        raise ValueError(
            "CUTLASS Conv output/bias channels must match weight O: "
            f"bias={bias_shape[0]}, output C={output_shape[1]}, weight O={weight_shape[0]}"
        )
    if residual_shape is not None and residual_shape != output_shape:
        raise ValueError(
            f"CUTLASS Conv residual_shape must match output_shape for {op_family}: "
            f"residual={residual_shape}, output={output_shape}"
        )
    layout_translation = dict(payload.get("layout_translation", {}))
    expected_input_nbytes = _nbytes(x_shape, dtype_size)
    expected_output_nbytes = _nbytes(output_shape, dtype_size)
    if layout_translation.get("input_pack") != "nchw_to_nhwc_temporary":
        raise ValueError("CUTLASS Conv layout_translation.input_pack must be 'nchw_to_nhwc_temporary'")
    if layout_translation.get("output_unpack") != "nhwc_to_nchw_temporary":
        raise ValueError("CUTLASS Conv layout_translation.output_unpack must be 'nhwc_to_nchw_temporary'")
    if layout_translation.get("bias") != "direct_per_output_channel":
        raise ValueError("CUTLASS Conv layout_translation.bias must be 'direct_per_output_channel'")
    if int(layout_translation.get("input_pack_nbytes", -1)) != expected_input_nbytes:
        raise ValueError(
            "CUTLASS Conv input_pack_nbytes mismatch: "
            f"expected {expected_input_nbytes}, got {layout_translation.get('input_pack_nbytes')!r}"
        )
    if str(layout_translation.get("input_pack_symbol")) != cutlass_conv_input_pack_symbol(dtype):
        raise ValueError(
            "CUTLASS Conv layout_translation.input_pack_symbol mismatch: "
            f"expected {cutlass_conv_input_pack_symbol(dtype)!r}, got {layout_translation.get('input_pack_symbol')!r}"
        )
    if int(layout_translation.get("output_unpack_nbytes", -1)) != expected_output_nbytes:
        raise ValueError(
            "CUTLASS Conv output_unpack_nbytes mismatch: "
            f"expected {expected_output_nbytes}, got {layout_translation.get('output_unpack_nbytes')!r}"
        )
    if str(layout_translation.get("output_unpack_symbol")) != cutlass_conv_output_unpack_symbol(dtype):
        raise ValueError(
            "CUTLASS Conv layout_translation.output_unpack_symbol mismatch: "
            f"expected {cutlass_conv_output_unpack_symbol(dtype)!r}, got {layout_translation.get('output_unpack_symbol')!r}"
        )
    if _cutlass_conv_op_has_residual(op_family):
        if layout_translation.get("residual_pack") != "nchw_to_nhwc_temporary":
            raise ValueError("CUTLASS Conv layout_translation.residual_pack must be 'nchw_to_nhwc_temporary'")
        if int(layout_translation.get("residual_pack_nbytes", -1)) != expected_output_nbytes:
            raise ValueError(
                "CUTLASS Conv residual_pack_nbytes mismatch: "
                f"expected {expected_output_nbytes}, got {layout_translation.get('residual_pack_nbytes')!r}"
            )
        if str(layout_translation.get("residual_pack_symbol")) != cutlass_conv_input_pack_symbol(dtype):
            raise ValueError(
                "CUTLASS Conv layout_translation.residual_pack_symbol mismatch: "
                f"expected {cutlass_conv_input_pack_symbol(dtype)!r}, got {layout_translation.get('residual_pack_symbol')!r}"
            )
    weight_transform = dict(payload.get("weight_transform", {}))
    expected_weight_nbytes = _nbytes(weight_shape, dtype_size)
    if weight_transform.get("from") != "oihw":
        raise ValueError("CUTLASS Conv weight_transform.from must be 'oihw'")
    if weight_transform.get("to") != "ohwi":
        raise ValueError("CUTLASS Conv weight_transform.to must be 'ohwi'")
    weight_prepacked = str(weight_transform.get("pack")) == "constants_bin_prepacked"
    if str(weight_transform.get("pack")) not in {"oihw_to_ohwi_temporary", "constants_bin_prepacked"}:
        raise ValueError("CUTLASS Conv weight_transform.pack must be an admitted OIHW->OHWI packing mode")
    if weight_prepacked:
        if not bool(weight_transform.get("runtime_persistent")):
            raise ValueError("CUTLASS Conv prepacked weight constants must be runtime_persistent")
        weight_transform.pop("pack_symbol", None)
    elif str(weight_transform.get("pack_symbol")) != cutlass_conv_weight_pack_symbol(dtype):
        raise ValueError(
            "CUTLASS Conv weight_transform.pack_symbol mismatch: "
            f"expected {cutlass_conv_weight_pack_symbol(dtype)!r}, got {weight_transform.get('pack_symbol')!r}"
        )
    if int(weight_transform.get("temporary_nbytes", -1)) != expected_weight_nbytes:
        raise ValueError(
            "CUTLASS Conv weight temporary_nbytes mismatch: "
            f"expected {expected_weight_nbytes}, got {weight_transform.get('temporary_nbytes')!r}"
        )
    if bool(weight_transform.get("runtime_persistent")) != weight_prepacked:
        raise ValueError("CUTLASS Conv weight_transform.runtime_persistent must match the packing mode")
    channel_pad_multiple = weight_transform.get("channel_pad_multiple")
    if not isinstance(channel_pad_multiple, int) or isinstance(channel_pad_multiple, bool) or channel_pad_multiple <= 0:
        raise ValueError(
            "CUTLASS Conv weight_transform.channel_pad_multiple must be a positive integer, "
            f"got {channel_pad_multiple!r}"
        )
    expected_padded_input_channels = _round_up(weight_shape[1], channel_pad_multiple)
    expected_padded_output_channels = _round_up(weight_shape[0], channel_pad_multiple)
    if int(weight_transform.get("padded_input_channels", -1)) != expected_padded_input_channels:
        raise ValueError(
            "CUTLASS Conv padded_input_channels mismatch: "
            f"expected {expected_padded_input_channels}, got {weight_transform.get('padded_input_channels')!r}"
        )
    if int(weight_transform.get("padded_output_channels", -1)) != expected_padded_output_channels:
        raise ValueError(
            "CUTLASS Conv padded_output_channels mismatch: "
            f"expected {expected_padded_output_channels}, got {weight_transform.get('padded_output_channels')!r}"
        )
    if float(weight_transform.get("padding_fill_value", 0.0)) != 0.0:
        raise ValueError("CUTLASS Conv padding_fill_value must be 0.0")
    temporary_buffers = payload.get("temporary_buffers", ())
    if not isinstance(temporary_buffers, (list, tuple)):
        raise ValueError("CUTLASS Conv temporary_buffers must be a list")
    expected_buffers = [
        ("activation_nhwc", "layout_pack", "nhwc", expected_input_nbytes),
        *([] if weight_prepacked else [("weight_ohwi", "layout_pack", "ohwi", expected_weight_nbytes)]),
        *([("residual_nhwc", "layout_pack", "nhwc", expected_output_nbytes)] if residual_shape is not None else []),
        ("output_nhwc", "layout_pack", "nhwc", expected_output_nbytes),
    ]
    if len(temporary_buffers) != len(expected_buffers):
        raise ValueError(
            f"CUTLASS Conv temporary_buffers must contain {len(expected_buffers)} entries, "
            f"got {len(temporary_buffers)}"
        )
    normalized_buffers = []
    for buffer, (expected_name, expected_kind, expected_layout, expected_nbytes) in zip(
        temporary_buffers, expected_buffers, strict=True
    ):
        if not isinstance(buffer, Mapping):
            raise ValueError("CUTLASS Conv temporary_buffers entries must be objects")
        buffer_payload = dict(buffer)
        if str(buffer_payload.get("name")) != expected_name:
            raise ValueError(
                f"CUTLASS Conv temporary buffer name mismatch: expected {expected_name!r}, "
                f"got {buffer_payload.get('name')!r}"
            )
        if str(buffer_payload.get("kind")) != expected_kind:
            raise ValueError(
                f"CUTLASS Conv temporary buffer kind mismatch for {expected_name!r}: "
                f"expected {expected_kind!r}, got {buffer_payload.get('kind')!r}"
            )
        if str(buffer_payload.get("layout")) != expected_layout:
            raise ValueError(
                f"CUTLASS Conv temporary buffer layout mismatch for {expected_name!r}: "
                f"expected {expected_layout!r}, got {buffer_payload.get('layout')!r}"
            )
        if int(buffer_payload.get("nbytes", -1)) != expected_nbytes:
            raise ValueError(
                f"CUTLASS Conv temporary buffer nbytes mismatch for {expected_name!r}: "
                f"expected {expected_nbytes}, got {buffer_payload.get('nbytes')!r}"
            )
        normalized_buffers.append(buffer_payload)
    if int(payload.get("workspace_nbytes", -1)) != 0:
        raise ValueError("CUTLASS Conv workspace_nbytes must be 0 for the current plan")
    expected_temporary_nbytes = sum(int(buffer["nbytes"]) for buffer in normalized_buffers)
    if int(payload.get("temporary_nbytes", -1)) != expected_temporary_nbytes:
        raise ValueError(
            "CUTLASS Conv temporary_nbytes mismatch: "
            f"expected {expected_temporary_nbytes}, got {payload.get('temporary_nbytes')!r}"
        )
    if candidate is not None:
        candidate_layouts = dict(candidate.get("layouts", {}))
        expected_candidate_layouts = _cutlass_conv_candidate_layouts(op_family)
        if candidate_layouts != expected_candidate_layouts:
            raise ValueError(
                "CUTLASS Conv candidate layouts do not match transform plan: "
                f"expected {expected_candidate_layouts}, got {candidate_layouts!r}"
            )
        if str(candidate.get("dtype")) != dtype:
            raise ValueError(
                f"CUTLASS Conv candidate dtype mismatch: expected {dtype!r}, got {candidate.get('dtype')!r}"
            )
        if str(candidate.get("epilogue")) != epilogue:
            raise ValueError(
                f"CUTLASS Conv candidate epilogue mismatch: expected {epilogue!r}, got {candidate.get('epilogue')!r}"
            )
        if dict(candidate.get("epilogue_config", {})) != epilogue_config:
            raise ValueError(
                "CUTLASS Conv candidate epilogue_config mismatch: "
                f"expected {epilogue_config!r}, got {candidate.get('epilogue_config')!r}"
            )
        if str(candidate.get("launch_abi")) != _cutlass_conv_launch_abi(op_family):
            raise ValueError(
                "CUTLASS Conv candidate launch_abi mismatch: "
                f"expected {_cutlass_conv_launch_abi(op_family)!r}, got {candidate.get('launch_abi')!r}"
            )
        selected_candidate = payload.get("selected_candidate")
        if isinstance(selected_candidate, Mapping):
            if str(selected_candidate.get("candidate_id", "")) != str(candidate.get("candidate_id", "")):
                raise ValueError(
                    "CUTLASS Conv selected_candidate.candidate_id mismatch: "
                    f"expected {candidate.get('candidate_id')!r}, got {selected_candidate.get('candidate_id')!r}"
                )
            if str(selected_candidate.get("kernel_symbol", "")) != str(candidate.get("kernel_symbol", "")):
                raise ValueError(
                    "CUTLASS Conv selected_candidate.kernel_symbol mismatch: "
                    f"expected {candidate.get('kernel_symbol')!r}, got {selected_candidate.get('kernel_symbol')!r}"
                )
    payload["dtype"] = dtype
    payload["op_family"] = op_family
    if source_op is not None:
        payload["source_op"] = source_op
        payload["bias_mode"] = str(bias_mode)
    else:
        payload.pop("source_op", None)
        payload.pop("bias_mode", None)
    payload["epilogue"] = epilogue
    payload["epilogue_config"] = epilogue_config
    payload["semantic_layout"] = semantic_layout
    payload["provider_layout"] = provider_layout
    payload["layout_translation"] = layout_translation
    payload["weight_transform"] = weight_transform
    payload["conv_config"] = normalized_conv_config
    payload["input_shape"] = x_shape
    payload["weight_shape"] = weight_shape
    payload["bias_shape"] = bias_shape
    if residual_shape is not None:
        payload["residual_shape"] = residual_shape
    payload["output_shape"] = output_shape
    payload["temporary_buffers"] = normalized_buffers
    payload["workspace_nbytes"] = 0
    payload["temporary_nbytes"] = expected_temporary_nbytes
    payload["status"] = status
    if status == _CUTLASS_CONV_STATUS:
        payload["runtime"] = dict(payload["runtime"])
        payload["profiler_status"] = _CUTLASS_CONV_PROFILER_STATUS
        payload.pop("profiler_blocked_reason", None)
        payload.pop("blocked_reason", None)
    return payload



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
    conv_plan_payload = validate_cutlass_conv_plan(
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
        "epilogue_config",
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
            "CUTLASS Conv used candidate plan entry selected candidate_id is not emitted by the current plan: "
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
        "epilogue_config",
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


def _cutlass_conv_item_wrapper_stages(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    op_name = str(item.get("op", ""))
    _validate_conv_op_name(op_name)
    selected_candidate = _selected_cutlass_conv_candidate(item)
    node_id = _optional_str(item.get("node_id"))
    if node_id is None and isinstance(item.get("cutlass_conv_plan"), Mapping):
        node_id = _optional_str(item["cutlass_conv_plan"].get("node_id"))
    conv_plan = validate_cutlass_conv_plan(
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
        **(
            {"source_op": str(conv_plan["source_op"]), "bias_mode": str(conv_plan["bias_mode"])}
            if conv_plan.get("source_op") is not None
            else {}
        ),
    }
    stages = [
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
    ]
    if not bool(weight_transform.get("runtime_persistent")):
        stages.append(
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
            }
        )
    provider_inputs = [
        _temporary_buffer_descriptor(temporary_buffers["activation_nhwc"]),
        (
            {"kind": "semantic_tensor", "role": "weight", "layout": "ohwi"}
            if bool(weight_transform.get("runtime_persistent"))
            else _temporary_buffer_descriptor(temporary_buffers["weight_ohwi"])
        ),
        {"kind": "semantic_tensor", "role": "bias", "layout": "o"},
    ]
    if _cutlass_conv_op_has_residual(op_name):
        stages.append(
            {
                **stage_common,
                "stage_index": 2,
                "stage_name": "residual_pack",
                "stage_kind": "transform_helper",
                "symbol": str(layout_translation["residual_pack_symbol"]),
                "helper_abi": _CUTLASS_CONV_TRANSFORM_ABI,
                "tensor_role": "residual",
                "layout_from": "nchw",
                "layout_to": "nhwc",
                "source": {"kind": "semantic_tensor", "role": "residual", "layout": "nchw"},
                "destination": _temporary_buffer_descriptor(temporary_buffers["residual_nhwc"]),
                "shape_args": [
                    {"name": "n", "placeholder": "output_n", "value": int(output_shape[0])},
                    {"name": "c", "placeholder": "output_c", "value": int(output_shape[1])},
                    {"name": "h", "placeholder": "output_h", "value": int(output_shape[2])},
                    {"name": "w", "placeholder": "output_w", "value": int(output_shape[3])},
                ],
            }
        )
        provider_inputs.append(_temporary_buffer_descriptor(temporary_buffers["residual_nhwc"]))
    stages.extend(
        [
            {
                **stage_common,
                "stage_index": len(stages),
                "stage_name": "provider_launch",
                "stage_kind": "provider_launcher",
                "symbol": str(selected_candidate["kernel_symbol"]),
                "launch_abi": str(selected_candidate["launch_abi"]),
                "inputs": provider_inputs,
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
                "stage_index": len(stages) + 1,
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
    )
    return stages


def _nbytes(shape: list[int], dtype_size: int) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * int(dtype_size)


def _validate_positive_shape(value: Any, *, rank: int, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != rank:
        raise ValueError(f"CUTLASS Conv {name} must be a rank-{rank} integer shape, got {value!r}")
    dims = []
    for dim in value:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError(f"CUTLASS Conv {name} must contain positive integers, got {value!r}")
        dims.append(int(dim))
    return dims


def _validate_non_negative_shape(value: Any, *, rank: int, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != rank:
        raise ValueError(f"CUTLASS Conv {name} must be a rank-{rank} integer shape, got {value!r}")
    dims = []
    for dim in value:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
            raise ValueError(f"CUTLASS Conv {name} must contain non-negative integers, got {value!r}")
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


def _cutlass_conv_semantic_layout(op_name: str) -> dict[str, str]:
    _validate_conv_op_name(op_name)
    layout = dict(_CUTLASS_CONV_BASE_SEMANTIC_LAYOUT)
    if _cutlass_conv_op_has_residual(op_name):
        layout["residual"] = "nchw"
    return layout


def _cutlass_conv_provider_layout(op_name: str) -> dict[str, str]:
    _validate_conv_op_name(op_name)
    layout = dict(_CUTLASS_CONV_BASE_PROVIDER_LAYOUT)
    if _cutlass_conv_op_has_residual(op_name):
        layout["residual"] = "nhwc"
    return layout


def _cutlass_conv_candidate_layouts(op_name: str) -> dict[str, str]:
    semantic_layout = _cutlass_conv_semantic_layout(op_name)
    provider_layout = _cutlass_conv_provider_layout(op_name)
    layouts = {
        "activation_semantic": semantic_layout["activation"],
        "weight_semantic": semantic_layout["weight"],
        "output_semantic": semantic_layout["output"],
        "activation_provider": provider_layout["activation"],
        "weight_provider": provider_layout["weight"],
        "output_provider": provider_layout["output"],
    }
    if _cutlass_conv_op_has_residual(op_name):
        layouts["residual_semantic"] = semantic_layout["residual"]
        layouts["residual_provider"] = provider_layout["residual"]
    return layouts


def _selected_cutlass_conv_candidate(item: Mapping[str, Any]) -> Mapping[str, Any]:
    selected_id = str(item.get("selected_candidate_id", ""))
    candidates = item.get("candidates", ())
    if isinstance(candidates, (list, tuple)):
        for candidate in candidates:
            if isinstance(candidate, Mapping) and str(candidate.get("candidate_id", "")) == selected_id:
                return dict(candidate)
    raise ValueError(
        "CUTLASS Conv wrapper stages require a selected candidate present in the manifest item"
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
        helpers = [
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
                "symbol": str(layout_translation["output_unpack_symbol"]),
                "dtype": dtype,
                "tensor_role": "output",
                "transform": str(layout_translation["output_unpack"]),
                "layout_from": "nhwc",
                "layout_to": "nchw",
                "shape_order": ["n", "c", "h", "w"],
                "helper_abi": _CUTLASS_CONV_TRANSFORM_ABI,
            },
        ]
        if not bool(weight_transform.get("runtime_persistent")):
            helpers.append(
                {
                    "symbol": str(weight_transform["pack_symbol"]),
                    "dtype": dtype,
                    "tensor_role": "weight",
                    "transform": str(weight_transform["pack"]),
                    "layout_from": "oihw",
                    "layout_to": "ohwi",
                    "shape_order": ["o", "i", "h", "w"],
                    "helper_abi": _CUTLASS_CONV_TRANSFORM_ABI,
                }
            )
        for helper in helpers:
            existing = helpers_by_symbol.get(helper["symbol"])
            if existing is not None and existing != helper:
                raise ValueError(
                    "CUTLASS Conv transform helper metadata drifted for symbol "
                    f"{helper['symbol']!r}: expected {existing!r}, got {helper!r}"
                )
            helpers_by_symbol[helper["symbol"]] = helper
    return sorted(helpers_by_symbol.values(), key=lambda item: (item["dtype"], item["tensor_role"], item["symbol"]))


def _cutlass_conv_source_exports(used_candidate_plan: Mapping[str, Any]) -> str:
    lines: list[str] = []
    emitted: set[str] = set()
    transform_helpers = [
        dict(item)
        for item in used_candidate_plan.get("transform_helpers", ())
        if isinstance(item, Mapping)
    ]
    for helper in transform_helpers:
        symbol_name = str(helper.get("symbol", ""))
        if not symbol_name or symbol_name in emitted:
            continue
        emitted.add(symbol_name)
        lines.append(_cutlass_conv_transform_export(helper))
    for candidate in used_candidate_plan.get("candidates", ()):
        if not isinstance(candidate, Mapping):
            continue
        symbol_name = str(candidate.get("kernel_symbol", ""))
        if not symbol_name or symbol_name in emitted:
            continue
        emitted.add(symbol_name)
        lines.append(_cutlass_conv_launcher_export(candidate))
    return "\n\n".join(lines) if lines else "// no CUTLASS Conv exports requested"


def _cutlass_conv_transform_export(helper: Mapping[str, Any]) -> str:
    symbol = str(helper.get("symbol", ""))
    dtype = str(helper.get("dtype", ""))
    transform = str(helper.get("transform", ""))
    _normalize_conv_dtype(dtype)
    dtype_prefix, launch_prefix = _cutlass_conv_dtype_prefixes(dtype)
    macro_by_transform = {
        "nchw_to_nhwc_temporary": "DINOML_CUTLASS_CONV_NCHW_TO_NHWC_EXPORT",
        "oihw_to_ohwi_temporary": "DINOML_CUTLASS_CONV_OIHW_TO_OHWI_EXPORT",
        "nhwc_to_nchw_temporary": "DINOML_CUTLASS_CONV_NHWC_TO_NCHW_EXPORT",
    }
    macro = macro_by_transform.get(transform)
    if macro is None:
        raise ValueError(f"Unsupported CUTLASS Conv transform helper {transform!r}")
    return f"{macro}({symbol}, {dtype_prefix})"


def _cutlass_conv_launcher_export(candidate: Mapping[str, Any]) -> str:
    dtype = str(candidate.get("dtype", ""))
    epilogue = str(candidate.get("epilogue", ""))
    cutlass_config = dict(candidate.get("cutlass", {}))
    symbol = str(candidate.get("kernel_symbol", ""))
    profiler_symbol = str(candidate.get("profiler_symbol", ""))
    if not symbol or not profiler_symbol:
        raise ValueError(f"CUTLASS Conv candidate is missing launcher/profiler symbols: {candidate!r}")
    dtype_prefix, launch_prefix = _cutlass_conv_dtype_prefixes(dtype)
    macro = {
        "bias": "DINOML_CUTLASS_CONV_BIAS_EXPORT",
        "bias_relu": "DINOML_CUTLASS_CONV_BIAS_RELU_EXPORT",
        "bias_add": "DINOML_CUTLASS_CONV_BIAS_ADD_EXPORT",
        "bias_add_relu": "DINOML_CUTLASS_CONV_BIAS_ADD_RELU_EXPORT",
    }.get(epilogue)
    if macro is None:
        raise ValueError(f"Unsupported CUTLASS Conv epilogue {epilogue!r}")
    opclass = "cutlass::arch::OpClassTensorOp" if str(cutlass_config.get("opclass")) == "tensorop" else "cutlass::arch::OpClassSimt"
    threadblock = _shape_macro_args("threadblock", cutlass_config)
    warp = _shape_macro_args("warp", cutlass_config)
    instruction = _shape_macro_args("instruction_shape", cutlass_config)
    stages = int(cutlass_config.get("stages", 2) or 2)
    align_a = int(cutlass_config.get("align_a", cutlass_config.get("align", 1)) or 1)
    align_b = int(cutlass_config.get("align_b", cutlass_config.get("align", 1)) or 1)
    epilogue_vector_length = max(1, min(8, align_a))
    iterator = {
        "analytic": "cutlass::conv::IteratorAlgorithm::kAnalytic",
        "optimized": "cutlass::conv::IteratorAlgorithm::kOptimized",
        "few_channels": "cutlass::conv::IteratorAlgorithm::kFewChannels",
        "fixed_channels": "cutlass::conv::IteratorAlgorithm::kFixedChannels",
    }[str(cutlass_config.get("iterator_algorithm", "analytic"))]
    math_operator = {
        "multiply_add": "cutlass::arch::OpMultiplyAdd",
    }.get(str(cutlass_config.get("math_operator", "multiply_add")))
    if math_operator is None:
        raise ValueError(f"Unsupported CUTLASS Conv math operator {cutlass_config.get('math_operator')!r}")
    return (
        f"{macro}({symbol}, {profiler_symbol}, {dtype_prefix}, {launch_prefix}, {opclass}, "
        f"{threadblock}, {warp}, {instruction}, {stages}, {math_operator}, "
        f"{iterator}, {align_a}, {align_b}, {epilogue_vector_length})"
    )


def _shape_macro_args(field: str, cutlass_config: Mapping[str, Any]) -> str:
    shape = cutlass_config.get(field)
    if shape is None and field == "instruction_shape":
        shape = cutlass_config.get("instruction")
    if not isinstance(shape, (list, tuple)) or len(shape) != 3:
        raise ValueError(f"CUTLASS Conv candidate is missing {field}: {cutlass_config!r}")
    return ", ".join(str(int(dim)) for dim in shape)


def _cutlass_conv_dtype_prefixes(dtype: str) -> tuple[str, str]:
    normalized = _normalize_conv_dtype(dtype)
    if normalized == "float32":
        return "Fp32", "fp32"
    if normalized == "float16":
        return "Fp16", "fp16"
    if normalized == "bfloat16":
        return "Bf16", "bf16"
    raise ValueError(f"Unsupported CUTLASS Conv dtype: {dtype!r}")


def _cutlass_conv_symbol_requires_residual(symbol: str) -> bool:
    return "_conv2d_bias_add_" in symbol or "_conv2d_bias_add_relu_" in symbol


def _cutlass_conv_op_has_residual(op_name: str) -> bool:
    _validate_conv_op_name(op_name)
    return op_name in {"conv2d_bias_add", "conv2d_bias_add_relu"}


def _cutlass_conv_epilogue_has_residual(epilogue: str) -> bool:
    return epilogue in {"bias_add", "bias_add_relu"}


def _cutlass_conv_runtime_launcher_name(op_name: str) -> str:
    _validate_conv_op_name(op_name)
    return _CUTLASS_CONV_RUNTIME_LAUNCHER_BY_OP[op_name]


def _cutlass_conv_launch_abi(op_name: str) -> str:
    _validate_conv_op_name(op_name)
    return _CUTLASS_CONV_LAUNCH_ABI_BY_OP[op_name]


def _conv_candidate_status(dtype: str) -> str:
    _normalize_conv_dtype(dtype)
    return _CUTLASS_CONV_STATUS


def _conv_epilogue(op_name: str) -> str:
    _validate_conv_op_name(op_name)
    return _CUTLASS_CONV_EPILOGUE_BY_OP[op_name]


def _conv_epilogue_config(op_name: str) -> dict[str, Any]:
    epilogue = _conv_epilogue(op_name)
    if epilogue == "bias":
        return {"inputs": ["bias"]}
    if epilogue == "bias_relu":
        return {"inputs": ["bias"], "activation": "relu"}
    if epilogue == "bias_add":
        return {"inputs": ["bias", "d0"]}
    if epilogue == "bias_add_relu":
        return {"inputs": ["bias", "d0"], "activation": "relu"}
    raise ValueError(f"Unsupported CUTLASS Conv epilogue {epilogue!r}")


def _validate_conv_op_name(op_name: str) -> None:
    if op_name not in CONV_OPS:
        supported = ", ".join(CONV_OPS)
        raise ValueError(f"Unsupported CUTLASS Conv op {op_name!r}; supported ops: {supported}")


def _conv_plan_status(
    dtype: str,
    *,
    input_shape: Any,
    weight_shape: Any,
    output_shape: Any,
    conv_config: Mapping[str, Any],
) -> str:
    _normalize_conv_dtype(dtype)
    return _CUTLASS_CONV_STATUS


def _conv_plan_status_payload(status: str, *, op_name: str) -> dict[str, Any]:
    return {
        "runtime": {
            "status": _CUTLASS_CONV_STATUS,
            "launcher": _cutlass_conv_runtime_launcher_name(op_name),
        },
        "profiler_status": _CUTLASS_CONV_PROFILER_STATUS,
    }


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
    "cutlass_conv_cmake_target",
    "cutlass_conv_input_pack_symbol",
    "cutlass_conv_layout_plan",
    "cutlass_conv_output_unpack_symbol",
    "cutlass_conv_profiler_symbol",
    "cutlass_conv_static_library_name",
    "cutlass_conv_symbol",
    "cutlass_conv_used_candidate_plan",
    "cutlass_conv_wrapper_stages",
    "cutlass_conv_weight_pack_symbol",
    "normalize_cutlass_conv_used_candidate_plan",
        "validate_cutlass_conv_plan",
    ]
