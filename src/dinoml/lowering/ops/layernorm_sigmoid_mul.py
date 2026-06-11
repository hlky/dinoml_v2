from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import render_op_template, supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.normalization import LAYER_NORM_DTYPES


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "layernorm_sigmoid_mul")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return render_op_template("layernorm_sigmoid_mul_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return render_op_template("layernorm_sigmoid_mul_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "layernorm_sigmoid_mul")
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
    supported_target_spec(target, "layernorm_sigmoid_mul")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    x_tensor = tensor_map[node["inputs"][0]]
    weight_tensor = tensor_map[node["inputs"][1]]
    bias_tensor = tensor_map[node["inputs"][2]]
    output_tensor = tensor_map[node["outputs"][0]]
    cols = _validate_node_contract(node, x_tensor, weight_tensor, bias_tensor, output_tensor)
    dtype = str(x_tensor["dtype"])
    func = _function_name(node, tensor_map)
    return {
        "func": func,
        "kernel": f"{func}_kernel",
        "warp_kernel": f"{func}_warp_kernel",
        "cpu_storage_type": target_storage_type(dtype, "cpu"),
        "storage_type": target_storage_type(dtype, target),
        "cols": cols,
        "eps_literal": _float_literal(float(node.get("attrs", {}).get("eps", 1e-5))),
        "inv_cols_literal": _float_literal(1.0 / float(cols)),
        "block_size": _cuda_block_size(cols),
        "cols_per_thread": (cols + 31) // 32,
        "rows_per_block": _cuda_rows_per_block(cols),
        "use_warp_kernel": cols < 512,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    x_tensor: Mapping[str, Any],
    weight_tensor: Mapping[str, Any],
    bias_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> int:
    if str(node["op"]) != "layernorm_sigmoid_mul":
        raise ValueError(f"Unsupported normalization op: {node['op']}")
    x_dtype = str(x_tensor["dtype"])
    if x_dtype not in LAYER_NORM_DTYPES:
        raise NotImplementedError("layernorm_sigmoid_mul lowering supports float16, float32, and bfloat16 tensors only")
    if any(str(tensor["dtype"]) != x_dtype for tensor in (weight_tensor, bias_tensor, output_tensor)):
        raise NotImplementedError("layernorm_sigmoid_mul lowering currently requires matching input/output dtypes")
    if list(x_tensor["shape"]) != list(output_tensor["shape"]):
        raise ValueError("layernorm_sigmoid_mul input and output shapes must match")
    normalized_shape = node.get("attrs", {}).get("normalized_shape")
    if not isinstance(normalized_shape, list) or not normalized_shape:
        raise ValueError("layernorm_sigmoid_mul lowering requires a non-empty normalized_shape attr")
    if any(not isinstance(dim, int) or int(dim) <= 0 for dim in normalized_shape):
        raise ValueError("layernorm_sigmoid_mul lowering requires positive static normalized_shape dims")
    norm_rank = len(normalized_shape)
    if len(x_tensor["shape"]) < norm_rank:
        raise ValueError("layernorm_sigmoid_mul input rank must be at least len(normalized_shape)")
    if list(x_tensor["shape"][-norm_rank:]) != [int(dim) for dim in normalized_shape]:
        raise ValueError("layernorm_sigmoid_mul input suffix must match normalized_shape")
    if list(weight_tensor["shape"]) != [int(dim) for dim in normalized_shape]:
        raise ValueError("layernorm_sigmoid_mul weight shape must match normalized_shape")
    if list(bias_tensor["shape"]) != [int(dim) for dim in normalized_shape]:
        raise ValueError("layernorm_sigmoid_mul bias shape must match normalized_shape")
    x_shape_spec = x_tensor.get("shape_spec", x_tensor["shape"])
    if any(not isinstance(dim, int) for dim in x_shape_spec[-norm_rank:]):
        raise ValueError("layernorm_sigmoid_mul lowering requires a static normalized_shape suffix")
    cols = 1
    for dim in normalized_shape:
        cols *= int(dim)
    return cols


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
    x_tensor = tensor_map[node["inputs"][0]]
    signature = {
        "op": "layernorm_sigmoid_mul",
        "shape": list(x_tensor["shape"]),
        "dtype": str(x_tensor["dtype"]),
        "normalized_shape": list(node.get("attrs", {}).get("normalized_shape", [])),
        "eps": float(node.get("attrs", {}).get("eps", 1e-5)),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"layernorm_sigmoid_mul_{digest}"


def _float_literal(value: float) -> str:
    return f"{float(value):.9g}f"


LAYERNORM_SIGMOID_MUL_LOWERING = OpLowering(
    op_name="layernorm_sigmoid_mul",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
