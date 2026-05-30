from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.normalization import ADD_LAYER_NORM_DTYPES


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "add_layer_norm")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("add_layer_norm_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("add_layer_norm_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "add_layer_norm")
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    residual = _c_ident(node["inputs"][1])
    weight = _c_ident(node["inputs"][2])
    bias = _c_ident(node["inputs"][3])
    summed = _c_ident(node["outputs"][0])
    normalized = _c_ident(node["outputs"][1])
    args = (
        f"ptr_{x}, ptr_{residual}, ptr_{weight}, ptr_{bias}, "
        f"ptr_{summed}, ptr_{normalized}, runtime_numel_{summed}"
    )
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "add_layer_norm")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    x_tensor = tensor_map[node["inputs"][0]]
    residual_tensor = tensor_map[node["inputs"][1]]
    weight_tensor = tensor_map[node["inputs"][2]]
    bias_tensor = tensor_map[node["inputs"][3]]
    summed_tensor = tensor_map[node["outputs"][0]]
    normalized_tensor = tensor_map[node["outputs"][1]]
    _validate_node_contract(node, x_tensor, residual_tensor, weight_tensor, bias_tensor, summed_tensor, normalized_tensor)
    cols = int(x_tensor["shape"][-1])
    dtype = str(x_tensor["dtype"])
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "warp_kernel": f"{_function_name(node, tensor_map)}_warp_kernel",
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
    residual_tensor: Mapping[str, Any],
    weight_tensor: Mapping[str, Any],
    bias_tensor: Mapping[str, Any],
    summed_tensor: Mapping[str, Any],
    normalized_tensor: Mapping[str, Any],
) -> None:
    if str(node["op"]) != "add_layer_norm":
        raise ValueError(f"Unsupported normalization op: {node['op']}")
    x_dtype = str(x_tensor["dtype"])
    if any(str(tensor["dtype"]) != x_dtype for tensor in (residual_tensor, weight_tensor, bias_tensor, summed_tensor, normalized_tensor)):
        raise NotImplementedError("add_layer_norm lowering currently requires matching input/output dtypes")
    if x_dtype not in ADD_LAYER_NORM_DTYPES:
        raise NotImplementedError("add_layer_norm lowering supports float16, float32, and bfloat16 tensors only")
    if list(x_tensor["shape"]) != list(residual_tensor["shape"]):
        raise ValueError("add_layer_norm input and residual shapes must match")
    if list(x_tensor["shape"]) != list(summed_tensor["shape"]) or list(x_tensor["shape"]) != list(normalized_tensor["shape"]):
        raise ValueError("add_layer_norm output shapes must match input shape")
    if not x_tensor["shape"]:
        raise ValueError("add_layer_norm requires rank >= 1 input")
    if len(weight_tensor["shape"]) != 1 or len(bias_tensor["shape"]) != 1:
        raise ValueError("add_layer_norm requires rank-1 weight and bias")
    x_shape_spec = x_tensor.get("shape_spec", x_tensor["shape"])
    weight_shape_spec = weight_tensor.get("shape_spec", weight_tensor["shape"])
    bias_shape_spec = bias_tensor.get("shape_spec", bias_tensor["shape"])
    cols = x_tensor["shape"][-1]
    if not isinstance(x_shape_spec[-1], int) or not isinstance(cols, int) or int(cols) <= 0:
        raise ValueError("add_layer_norm lowering requires a positive static last dimension")
    if not isinstance(weight_shape_spec[0], int) or int(weight_tensor["shape"][0]) != int(cols):
        raise ValueError("add_layer_norm lowering requires weight shape [hidden] matching input hidden size")
    if not isinstance(bias_shape_spec[0], int) or int(bias_tensor["shape"][0]) != int(cols):
        raise ValueError("add_layer_norm lowering requires bias shape [hidden] matching input hidden size")


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
        "op": "add_layer_norm",
        "shape": list(x_tensor["shape"]),
        "dtype": str(x_tensor["dtype"]),
        "eps": float(node.get("attrs", {}).get("eps", 1e-5)),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"add_layer_norm_{digest}"


def _float_literal(value: float) -> str:
    return f"{float(value):.9g}f"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


ADD_LAYER_NORM_LOWERING = OpLowering(
    op_name="add_layer_norm",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
