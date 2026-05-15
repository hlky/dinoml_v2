from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.cpp_types import cpu_storage_type, cuda_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.ops.collections import COLLECTION_DTYPES, normalize_index_select_attrs, resolve_index_select_shape
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    context = _context(target, node, tensor_map)
    if target == "cpu":
        return _render_template("index_select_cpu.cpp.j2", context)
    if target == "cuda":
        return _render_template("index_select_cuda.cu.j2", context)
    raise ValueError(f"Unsupported index_select target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{x}, ptr_{out}, runtime_numel_{out}"
    if target == "cpu":
        return f"if (int err = {func}({args})) return err;"
    if target == "cuda":
        return f"if (int err = {func}({args}, session->stream)) return err;"
    raise ValueError(f"Unsupported index_select target: {target}")


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    if target not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported index_select target: {target}")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    dim, indices = _validate_node_contract(node, input_tensor, output_tensor)
    dtype = str(output_tensor["dtype"])
    storage_type = cpu_storage_type(dtype) if target == "cpu" else cuda_storage_type(dtype)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "copy_body": _copy_body(input_tensor, output_tensor, dim, indices),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> tuple[int, list[int]]:
    if node["op"] != "index_select":
        raise ValueError(f"Unsupported collection op: {node['op']}")
    if len(node.get("inputs", [])) != 1:
        raise ValueError("index_select expects one tensor input")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("index_select expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in COLLECTION_DTYPES:
        raise NotImplementedError(f"index_select lowering does not support dtype {output_tensor['dtype']}")
    if str(input_tensor["dtype"]) != dtype:
        raise ValueError("index_select input and output dtype must match")
    attrs = node.get("attrs", {})
    dim, indices = normalize_index_select_attrs(attrs.get("dim", 0), attrs.get("indices"), input_tensor["shape"])
    expected_shape = resolve_index_select_shape(input_tensor["shape"], dim, indices)
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("index_select output shape does not match selected indices")
    return dim, indices


def _copy_body(
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
    dim: int,
    indices: list[int],
) -> str:
    input_shape = [int(axis) for axis in input_tensor["shape"]]
    output_shape = [int(axis) for axis in output_tensor["shape"]]
    input_strides = _dense_strides(input_shape)
    lines = [
        "  int64_t remaining = idx;",
        "  int64_t input_idx = 0;",
        "  int64_t coord = 0;",
        "  int64_t selected_index = 0;",
    ]
    for axis in range(len(output_shape) - 1, -1, -1):
        output_extent = output_shape[axis]
        input_stride = input_strides[axis]
        lines.append(f"  coord = remaining % {output_extent};")
        lines.append(f"  remaining = remaining / {output_extent};")
        if axis == dim:
            lines.append(f"  selected_index = {_static_index_expr(indices, 'coord')};")
            lines.append(f"  input_idx += selected_index * {input_stride};")
        else:
            lines.append(f"  input_idx += coord * {input_stride};")
    lines.append("  y[idx] = x[input_idx];")
    return "\n".join(lines)


def _static_index_expr(indices: list[int], coord_expr: str) -> str:
    expr = str(indices[-1])
    for position in range(len(indices) - 2, -1, -1):
        expr = f"({coord_expr} == {position} ? {indices[position]} : {expr})"
    return expr


def _dense_strides(shape: list[int]) -> list[int]:
    strides = [1] * len(shape)
    running = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = running
        running *= shape[axis]
    return strides


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    attrs = node.get("attrs", {})
    signature = {
        "op": "index_select",
        "input_shape": list(input_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "dim": int(attrs.get("dim", 0)),
        "indices": list(attrs.get("indices", [])),
        "dtype": str(output_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"index_select_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


INDEX_SELECT_LOWERING = OpLowering(
    op_name="index_select",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
