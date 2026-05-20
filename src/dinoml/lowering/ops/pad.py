from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.ops.collections import COLLECTION_DTYPES, normalize_pad_widths, resolve_pad_shape
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "pad")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("pad_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("pad_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "pad")
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{x}, ptr_{out}, runtime_numel_{out}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "pad")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    left, _ = _validate_node_contract(node, input_tensor, output_tensor)
    dtype = str(output_tensor["dtype"])
    storage_type = target_storage_type(dtype, target)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "fill_literal": _float_literal(float(node.get("attrs", {}).get("value", 0.0))),
        "copy_body": _copy_body(input_tensor, output_tensor, left),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> tuple[list[int], list[int]]:
    if node["op"] != "pad":
        raise ValueError(f"Unsupported collection op: {node['op']}")
    if len(node.get("inputs", [])) != 1:
        raise ValueError("pad expects one tensor input")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("pad expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in COLLECTION_DTYPES:
        raise NotImplementedError(f"pad lowering does not support dtype {output_tensor['dtype']}")
    if str(input_tensor["dtype"]) != dtype:
        raise ValueError("pad input and output dtype must match")
    left, right = normalize_pad_widths(node.get("attrs", {}).get("pad"), len(input_tensor["shape"]))
    expected_shape = resolve_pad_shape(input_tensor["shape"], node.get("attrs", {}).get("pad"))
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("pad output shape does not match pad attrs")
    return left, right


def _copy_body(input_tensor: Mapping[str, Any], output_tensor: Mapping[str, Any], left: list[int]) -> str:
    input_shape = [int(axis) for axis in input_tensor["shape"]]
    output_shape = [int(axis) for axis in output_tensor["shape"]]
    lines = [
        "  int64_t remaining = idx;",
        "  int64_t input_idx = 0;",
        "  int64_t input_stride = 1;",
        "  bool inside = true;",
        "  int64_t coord = 0;",
        "  int64_t input_coord = 0;",
    ]
    for axis in range(len(output_shape) - 1, -1, -1):
        lines.append(f"  coord = remaining % {output_shape[axis]};")
        lines.append(f"  remaining = remaining / {output_shape[axis]};")
        lines.append(f"  input_coord = coord - {left[axis]};")
        lines.append(f"  if (input_coord < 0 || input_coord >= {input_shape[axis]}) inside = false;")
        lines.append("  input_idx += input_coord * input_stride;")
        lines.append(f"  input_stride *= {input_shape[axis]};")
    lines.append("  y[idx] = inside ? x[input_idx] : fill_value;")
    return "\n".join(lines)


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    signature = {
        "op": "pad",
        "input_shape": list(input_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "pad": list(node.get("attrs", {}).get("pad", [])),
        "value": float(node.get("attrs", {}).get("value", 0.0)),
        "dtype": str(output_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"pad_{digest}"


def _float_literal(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("pad lowering supports only finite constant values for now")
    literal = f"{value:.9g}"
    if "." not in literal and "e" not in literal and "E" not in literal:
        literal = f"{literal}.0"
    return f"{literal}f"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


PAD_LOWERING = OpLowering(
    op_name="pad",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
