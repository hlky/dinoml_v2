from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import render_op_template, supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.normalization import GROUP_NORM_DTYPES


_GROUP_NORM_OPS = frozenset({"group_norm", "group_norm_swish"})


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, str(node["op"]))
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return render_op_template("group_norm_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return render_op_template("group_norm_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, str(node["op"]))
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    weight = _c_ident(node["inputs"][1])
    bias = _c_ident(node["inputs"][2])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{x}, ptr_{weight}, ptr_{bias}, ptr_{out}, runtime_numel_{out}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, str(node["op"]))
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    x_tensor = tensor_map[node["inputs"][0]]
    weight_tensor = tensor_map[node["inputs"][1]]
    bias_tensor = tensor_map[node["inputs"][2]]
    output_tensor = tensor_map[node["outputs"][0]]
    num_groups = _validate_node_contract(node, x_tensor, weight_tensor, bias_tensor, output_tensor)
    channels = int(x_tensor["shape"][-1])
    sample_dims = [int(dim) for dim in x_tensor["shape"][1:]]
    sample_size = math.prod(sample_dims)
    spatial_size = sample_size // channels
    group_channels = channels // num_groups
    group_size = spatial_size * group_channels
    dtype = str(x_tensor["dtype"])
    func = _function_name(node, tensor_map)
    return {
        "func": func,
        "kernel": f"{func}_kernel",
        "cpu_storage_type": target_storage_type(dtype, "cpu"),
        "storage_type": target_storage_type(dtype, target),
        "sample_size": sample_size,
        "spatial_size": spatial_size,
        "channels": channels,
        "num_groups": num_groups,
        "group_channels": group_channels,
        "group_size": group_size,
        "eps_literal": _float_literal(float(node.get("attrs", {}).get("eps", 1e-5))),
        "inv_group_size_literal": _float_literal(1.0 / float(group_size)),
        "apply_swish": str(node["op"]) == "group_norm_swish",
        "block_size": _gpu_block_size(group_size),
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    x_tensor: Mapping[str, Any],
    weight_tensor: Mapping[str, Any],
    bias_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> int:
    op_name = str(node["op"])
    if op_name not in _GROUP_NORM_OPS:
        raise ValueError(f"Unsupported normalization op: {node['op']}")
    x_dtype = str(x_tensor["dtype"])
    weight_dtype = str(weight_tensor["dtype"])
    bias_dtype = str(bias_tensor["dtype"])
    output_dtype = str(output_tensor["dtype"])
    if x_dtype != weight_dtype or x_dtype != bias_dtype or x_dtype != output_dtype:
        raise NotImplementedError(f"{op_name} lowering currently requires matching input/output dtypes")
    if x_dtype not in GROUP_NORM_DTYPES:
        raise NotImplementedError(f"{op_name} lowering supports float16, float32, and bfloat16 tensors only")
    if list(x_tensor["shape"]) != list(output_tensor["shape"]):
        raise ValueError(f"{op_name} input and output shapes must match")
    if len(x_tensor["shape"]) < 2:
        raise ValueError(f"{op_name} requires rank >= 2 input")
    if len(weight_tensor["shape"]) != 1 or len(bias_tensor["shape"]) != 1:
        raise ValueError(f"{op_name} requires rank-1 weight and bias")
    x_shape_spec = x_tensor.get("shape_spec", x_tensor["shape"])
    weight_shape_spec = weight_tensor.get("shape_spec", weight_tensor["shape"])
    bias_shape_spec = bias_tensor.get("shape_spec", bias_tensor["shape"])
    if any(not isinstance(dim, int) for dim in x_shape_spec[1:]):
        raise ValueError(f"{op_name} lowering requires static non-batch dimensions")
    channels = x_tensor["shape"][-1]
    if not isinstance(x_shape_spec[-1], int) or not isinstance(channels, int) or int(channels) <= 0:
        raise ValueError(f"{op_name} lowering requires a positive static last dimension")
    if any(int(dim) <= 0 for dim in x_tensor["shape"][1:]):
        raise ValueError(f"{op_name} lowering requires positive static non-batch dimensions")
    if not isinstance(weight_shape_spec[0], int) or int(weight_tensor["shape"][0]) != int(channels):
        raise ValueError(f"{op_name} lowering requires weight shape [channels] matching the input channels")
    if not isinstance(bias_shape_spec[0], int) or int(bias_tensor["shape"][0]) != int(channels):
        raise ValueError(f"{op_name} lowering requires bias shape [channels] matching the input channels")
    num_groups = node.get("attrs", {}).get("num_groups")
    if not isinstance(num_groups, int) or isinstance(num_groups, bool) or int(num_groups) <= 0:
        raise ValueError(f"{op_name} lowering requires a positive integer num_groups")
    validated_groups = int(num_groups)
    if int(channels) % validated_groups != 0:
        raise ValueError(f"{op_name} lowering requires channels divisible by num_groups")
    return validated_groups


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    x_tensor = tensor_map[node["inputs"][0]]
    signature = {
        "op": str(node["op"]),
        "shape": list(x_tensor["shape"]),
        "dtype": str(x_tensor["dtype"]),
        "num_groups": int(node.get("attrs", {}).get("num_groups", 0)),
        "eps": float(node.get("attrs", {}).get("eps", 1e-5)),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"{node['op']}_{digest}"


def _float_literal(value: float) -> str:
    return f"{float(value):.9g}f"


def _gpu_block_size(group_size: int) -> int:
    block = 1
    while block < group_size and block < 256:
        block *= 2
    return max(32, block)


GROUP_NORM_LOWERING = OpLowering(
    op_name="group_norm",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


GROUP_NORM_SWISH_LOWERING = OpLowering(
    op_name="group_norm_swish",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
