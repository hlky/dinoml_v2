from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.ops.collections import (
    COLLECTION_DTYPES,
    normalize_repeat_interleave_dim,
    normalize_repeat_interleave_repeats,
    resolve_repeat_interleave_shape,
)
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "repeat_interleave")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("repeat_interleave_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("repeat_interleave_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "repeat_interleave")
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{x}, ptr_{out}, runtime_numel_{out}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "repeat_interleave")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    repeats, dim = _validate_node_contract(node, input_tensor, output_tensor)
    dtype = str(output_tensor["dtype"])
    storage_type = target_storage_type(dtype, target)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "copy_body": _copy_body(input_tensor, output_tensor, repeats, dim),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> tuple[int, int]:
    if node["op"] != "repeat_interleave":
        raise ValueError(f"Unsupported collection op: {node['op']}")
    if len(node.get("inputs", [])) != 1:
        raise ValueError("repeat_interleave expects one tensor input")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("repeat_interleave expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in COLLECTION_DTYPES:
        raise NotImplementedError(f"repeat_interleave lowering does not support dtype {output_tensor['dtype']}")
    if str(input_tensor["dtype"]) != dtype:
        raise ValueError("repeat_interleave input and output dtype must match")
    repeats = normalize_repeat_interleave_repeats(node.get("attrs", {}).get("repeats"))
    dim = normalize_repeat_interleave_dim(node.get("attrs", {}).get("dim"), len(input_tensor["shape"]))
    expected_shape = resolve_repeat_interleave_shape(input_tensor["shape"], repeats, dim)
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("repeat_interleave output shape does not match input shape")
    return repeats, dim


def _copy_body(
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
    repeats: int,
    dim: int,
) -> str:
    input_shape = [int(axis) for axis in input_tensor["shape"]]
    output_shape = [int(axis) for axis in output_tensor["shape"]]
    lines = [
        f"  const int64_t repeat_interleave_repeats = {repeats};",
        "  int64_t remaining = idx;",
        "  int64_t input_idx = 0;",
        "  int64_t input_stride = 1;",
        "  int64_t coord = 0;",
    ]
    for axis in range(len(output_shape) - 1, -1, -1):
        output_extent = output_shape[axis]
        input_extent = input_shape[axis]
        lines.append(f"  coord = remaining % {output_extent};")
        lines.append(f"  remaining = remaining / {output_extent};")
        if axis == dim:
            lines.append("  coord = coord / repeat_interleave_repeats;")
        lines.append("  input_idx += coord * input_stride;")
        lines.append(f"  input_stride *= {input_extent};")
    lines.append("  y[idx] = x[input_idx];")
    return "\n".join(lines)


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    signature = {
        "op": "repeat_interleave",
        "input_shape": list(input_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "repeats": int(node.get("attrs", {}).get("repeats", 0)),
        "dim": int(node.get("attrs", {}).get("dim", 0)),
        "dtype": str(output_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"repeat_interleave_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


REPEAT_INTERLEAVE_LOWERING = OpLowering(
    op_name="repeat_interleave",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
