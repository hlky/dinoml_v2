from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import render_op_template, supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.normalization import LAYER_NORM_DTYPES


_GROUP_LAYERNORM_OPS = frozenset({"group_layernorm", "group_layernorm_sigmoid_mul"})


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    op_name = str(node["op"])
    spec = supported_target_spec(target, op_name)
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return render_op_template("group_layernorm_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return render_op_template("group_layernorm_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    op_name = str(node["op"])
    spec = supported_target_spec(target, op_name)
    context = _context(target, node, tensor_map)
    args = ", ".join(
        f"ptr_{group['x_ident']}, ptr_{group['weight_ident']}, ptr_{group['bias_ident']}, "
        f"ptr_{group['out_ident']}, runtime_numel_{group['out_ident']}"
        for group in context["groups"]
    )
    func = context["func"]
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
    groups = _validate_node_contract(node, tensor_map)
    dtype = str(tensor_map[node["inputs"][0]]["dtype"])
    func = _function_name(node, tensor_map)
    return {
        "func": func,
        "storage_type": target_storage_type(dtype, target),
        "cpu_storage_type": target_storage_type(dtype, "cpu"),
        "eps_literal": _float_literal(float(node.get("attrs", {}).get("eps", 1e-5))),
        "apply_sigmoid_mul": str(node["op"]) == "group_layernorm_sigmoid_mul",
        "groups": groups,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    op_name = str(node["op"])
    if op_name not in _GROUP_LAYERNORM_OPS:
        raise ValueError(f"Unsupported normalization op: {node['op']}")
    group_count = node.get("attrs", {}).get("group_count")
    if not isinstance(group_count, int) or isinstance(group_count, bool) or int(group_count) <= 0:
        raise ValueError(f"{op_name} lowering requires a positive integer group_count")
    normalized_shapes = node.get("attrs", {}).get("normalized_shapes")
    if not isinstance(normalized_shapes, list) or len(normalized_shapes) != int(group_count):
        raise ValueError(f"{op_name} lowering requires normalized_shapes matching group_count")
    inputs = list(node["inputs"])
    outputs = list(node["outputs"])
    if len(inputs) != 3 * int(group_count):
        raise ValueError(f"{op_name} lowering expects flattened [inputs, weights, biases] triples")
    if len(outputs) != int(group_count):
        raise ValueError(f"{op_name} lowering expects one output per group")
    x_tensors = [tensor_map[name] for name in inputs[:group_count]]
    weight_tensors = [tensor_map[name] for name in inputs[group_count : 2 * group_count]]
    bias_tensors = [tensor_map[name] for name in inputs[2 * group_count :]]
    output_tensors = [tensor_map[name] for name in outputs]
    dtype = str(x_tensors[0]["dtype"])
    if dtype not in LAYER_NORM_DTYPES:
        raise NotImplementedError(f"{op_name} lowering supports float16, float32, and bfloat16 tensors only")
    batch_prefix_shape = None
    batch_prefix_spec = None
    groups: list[dict[str, Any]] = []
    for index, (x_tensor, weight_tensor, bias_tensor, output_tensor, norm_shape) in enumerate(
        zip(x_tensors, weight_tensors, bias_tensors, output_tensors, normalized_shapes)
    ):
        if not isinstance(norm_shape, list) or not norm_shape:
            raise ValueError(f"{op_name} lowering requires non-empty normalized_shapes[{index}]")
        if any(not isinstance(dim, int) or int(dim) <= 0 for dim in norm_shape):
            raise ValueError(f"{op_name} lowering requires positive static normalized_shapes[{index}]")
        if any(str(tensor["dtype"]) != dtype for tensor in (weight_tensor, bias_tensor, output_tensor)):
            raise NotImplementedError(f"{op_name} lowering currently requires matching input/output dtypes")
        if list(x_tensor["shape"]) != list(output_tensor["shape"]):
            raise ValueError(f"{op_name} group {index} input and output shapes must match")
        norm_rank = len(norm_shape)
        if len(x_tensor["shape"]) < norm_rank:
            raise ValueError(f"{op_name} group {index} input rank must be at least len(normalized_shape)")
        if list(x_tensor["shape"][-norm_rank:]) != [int(dim) for dim in norm_shape]:
            raise ValueError(f"{op_name} group {index} input suffix must match normalized_shape")
        if list(weight_tensor["shape"]) != [int(dim) for dim in norm_shape]:
            raise ValueError(f"{op_name} group {index} weight shape must match normalized_shape")
        if list(bias_tensor["shape"]) != [int(dim) for dim in norm_shape]:
            raise ValueError(f"{op_name} group {index} bias shape must match normalized_shape")
        x_shape_spec = x_tensor.get("shape_spec", x_tensor["shape"])
        prefix_shape = list(x_tensor["shape"][: len(x_tensor["shape"]) - norm_rank])
        prefix_spec = list(x_shape_spec[: len(x_shape_spec) - norm_rank])
        if any(not isinstance(dim, int) for dim in x_shape_spec[-norm_rank:]):
            raise ValueError(f"{op_name} lowering requires static normalized_shape suffixes")
        if batch_prefix_shape is None:
            batch_prefix_shape = prefix_shape
            batch_prefix_spec = prefix_spec
        elif prefix_shape != batch_prefix_shape or prefix_spec != batch_prefix_spec:
            raise ValueError(f"{op_name} inputs must share the same leading batch dimensions")
        cols = 1
        for dim in norm_shape:
            cols *= int(dim)
        groups.append(
            {
                "index": index,
                "x_ident": _c_ident(inputs[index]),
                "weight_ident": _c_ident(inputs[group_count + index]),
                "bias_ident": _c_ident(inputs[2 * group_count + index]),
                "out_ident": _c_ident(outputs[index]),
                "kernel": f"{_function_name(node, tensor_map)}_g{index}_kernel",
                "warp_kernel": f"{_function_name(node, tensor_map)}_g{index}_warp_kernel",
                "cols": cols,
                "block_size": _cuda_block_size(cols),
                "cols_per_thread": (cols + 31) // 32,
                "rows_per_block": _cuda_rows_per_block(cols),
                "use_warp_kernel": cols < 512,
                "inv_cols_literal": _float_literal(1.0 / float(cols)),
            }
        )
    return groups


def _cuda_block_size(cols: int) -> int:
    block = 1
    while block < cols and block < 256:
        block *= 2
    return max(32, block)


def _cuda_rows_per_block(cols: int) -> int:
    if cols <= 256:
        return 8
    if cols <= 512:
        return 4
    return 2


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    group_count = int(node.get("attrs", {}).get("group_count", 0))
    signature = {
        "op": str(node["op"]),
        "shapes": [list(tensor_map[name]["shape"]) for name in node["inputs"][:group_count]],
        "normalized_shapes": list(node.get("attrs", {}).get("normalized_shapes", [])),
        "dtype": str(tensor_map[node["inputs"][0]]["dtype"]),
        "eps": float(node.get("attrs", {}).get("eps", 1e-5)),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"{str(node['op'])}_{digest}"


def _float_literal(value: float) -> str:
    return f"{float(value):.9g}f"


GROUP_LAYERNORM_LOWERING = OpLowering(
    op_name="group_layernorm",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


GROUP_LAYERNORM_SIGMOID_MUL_LOWERING = OpLowering(
    op_name="group_layernorm_sigmoid_mul",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
