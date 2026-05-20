from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.ops.collections import COLLECTION_DTYPES, normalize_dynamic_slice_attrs, resolve_dynamic_slice_shape
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "dynamic_slice")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("dynamic_slice_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("dynamic_slice_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "dynamic_slice")
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{x}, ptr_{out}, runtime_numel_{out}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "dynamic_slice")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    starts, sizes = _validate_node_contract(node, input_tensor, output_tensor)
    dtype = str(output_tensor["dtype"])
    storage_type = target_storage_type(dtype, target)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "copy_body": _copy_body(input_tensor, output_tensor, starts, sizes),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> tuple[list[int], list[int]]:
    if node["op"] != "dynamic_slice":
        raise ValueError(f"Unsupported collection op: {node['op']}")
    if len(node.get("inputs", [])) != 1:
        raise ValueError("dynamic_slice expects one tensor input")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("dynamic_slice expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in COLLECTION_DTYPES:
        raise NotImplementedError(f"dynamic_slice lowering does not support dtype {output_tensor['dtype']}")
    if str(input_tensor["dtype"]) != dtype:
        raise ValueError("dynamic_slice input and output dtype must match")
    attrs = node.get("attrs", {})
    starts, sizes = normalize_dynamic_slice_attrs(
        attrs.get("start_indices"),
        attrs.get("slice_sizes"),
        input_tensor["shape"],
    )
    expected_shape = resolve_dynamic_slice_shape(input_tensor["shape"], starts, sizes)
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("dynamic_slice output shape does not match slice_sizes")
    return starts, sizes


def _copy_body(
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
    starts: list[int],
    sizes: list[int],
) -> str:
    input_shape = [int(axis) for axis in input_tensor["shape"]]
    output_shape = [int(axis) for axis in output_tensor["shape"]]
    input_strides = _dense_strides(input_shape)
    lines = [
        "  int64_t remaining = idx;",
        "  int64_t input_idx = 0;",
        "  int64_t coord = 0;",
    ]
    for axis in range(len(output_shape) - 1, -1, -1):
        output_extent = sizes[axis]
        input_stride = input_strides[axis]
        start = starts[axis]
        lines.append(f"  coord = remaining % {output_extent};")
        lines.append(f"  remaining = remaining / {output_extent};")
        lines.append(f"  input_idx += (coord + {start}) * {input_stride};")
    lines.append("  y[idx] = x[input_idx];")
    return "\n".join(lines)


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
        "op": "dynamic_slice",
        "input_shape": list(input_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "start_indices": list(attrs.get("start_indices", [])),
        "slice_sizes": list(attrs.get("slice_sizes", [])),
        "dtype": str(output_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"dynamic_slice_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


DYNAMIC_SLICE_LOWERING = OpLowering(
    op_name="dynamic_slice",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
