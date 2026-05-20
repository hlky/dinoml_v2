from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.ops.reductions import TOPK_DTYPES, normalize_topk_dim, resolve_topk_shape
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "topk")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("topk_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("topk_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "topk")
    func = _function_name(node, tensor_map)
    inp = _c_ident(node["inputs"][0])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{inp}, ptr_{out}, runtime_numel_{out}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "topk")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    _validate_node_contract(node, input_tensor, output_tensor)
    input_dtype = str(input_tensor["dtype"])
    input_storage_type = target_storage_type(input_dtype, target)
    output_storage_type = _output_storage_type(target, node["op"], input_dtype)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "input_storage_type": input_storage_type,
        "output_storage_type": output_storage_type,
        "input_dtype": input_dtype,
        "output_kind": "indices" if node["op"] == "topk_indices" else "values",
        "cols": int(input_tensor["shape"][-1]),
        "k": int(node.get("attrs", {})["k"]),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> None:
    if node["op"] not in {"topk_values", "topk_indices"}:
        raise ValueError(f"Unsupported topk op: {node['op']}")
    if len(node.get("inputs", [])) != 1:
        raise ValueError("topk expects exactly one input")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("topk expects exactly one output")
    input_dtype = str(input_tensor["dtype"])
    if input_dtype not in TOPK_DTYPES:
        raise NotImplementedError(f"topk lowering does not support dtype {input_dtype}")
    expected_dtype = input_dtype if node["op"] == "topk_values" else "int64"
    if str(output_tensor["dtype"]) != expected_dtype:
        raise ValueError(f"{node['op']} output dtype must be {expected_dtype}, got {output_tensor['dtype']}")
    if not input_tensor["shape"]:
        raise ValueError("topk requires a ranked tensor")
    attrs = node.get("attrs", {})
    dim = normalize_topk_dim(attrs.get("dim", -1), len(input_tensor["shape"]))
    if dim != len(input_tensor["shape"]) - 1:
        raise NotImplementedError("topk lowering currently supports only the last dimension")
    if not bool(attrs.get("largest", True)):
        raise NotImplementedError("topk lowering currently supports only largest=True")
    if not bool(attrs.get("sorted", True)):
        raise NotImplementedError("topk lowering currently supports only sorted=True")
    shape_spec = input_tensor.get("shape_spec", input_tensor["shape"])
    cols = input_tensor["shape"][-1]
    if not isinstance(shape_spec[-1], int) or not isinstance(cols, int) or int(cols) <= 0:
        raise ValueError("topk lowering requires a positive static last dimension")
    expected = resolve_topk_shape(input_tensor["shape"], attrs.get("k"), dim, True, True)
    if list(output_tensor["shape"]) != expected:
        raise ValueError("topk output shape does not match topk attrs")


def _output_storage_type(target: str, op_name: str, input_dtype: str) -> str:
    if op_name == "topk_indices":
        return "int64_t"
    return target_storage_type(input_dtype, target)


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    attrs = node.get("attrs", {})
    dim = normalize_topk_dim(attrs.get("dim", -1), len(input_tensor["shape"]))
    signature = {
        "op": str(node["op"]),
        "shape": list(input_tensor["shape"]),
        "dtype": str(input_tensor["dtype"]),
        "k": int(attrs["k"]),
        "dim": dim,
        "largest": bool(attrs.get("largest", True)),
        "sorted": bool(attrs.get("sorted", True)),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"{node['op']}_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


TOPK_VALUES_LOWERING = OpLowering(
    op_name="topk_values",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
TOPK_INDICES_LOWERING = OpLowering(
    op_name="topk_indices",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
TOPK_LOWERINGS = {
    TOPK_VALUES_LOWERING.op_name: TOPK_VALUES_LOWERING,
    TOPK_INDICES_LOWERING.op_name: TOPK_INDICES_LOWERING,
}
