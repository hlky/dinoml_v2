from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.cpp_types import cpu_storage_type, cuda_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.ops.collections import COLLECTION_DTYPES, normalize_slice_scatter_attrs, resolve_slice_scatter_shape


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    context = _context(target, node, tensor_map)
    if target == "cpu":
        return _render_template("slice_scatter_cpu.cpp.j2", context)
    if target == "cuda":
        return _render_template("slice_scatter_cuda.cu.j2", context)
    raise ValueError(f"Unsupported slice_scatter target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    update = _c_ident(node["inputs"][1])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{x}, ptr_{update}, ptr_{out}, runtime_numel_{out}"
    if target == "cpu":
        return f"if (int err = {func}({args})) return err;"
    if target == "cuda":
        return f"if (int err = {func}({args}, session->stream)) return err;"
    raise ValueError(f"Unsupported slice_scatter target: {target}")


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    if target not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported slice_scatter target: {target}")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    update_tensor = tensor_map[node["inputs"][1]]
    output_tensor = tensor_map[node["outputs"][0]]
    starts = _validate_node_contract(node, input_tensor, update_tensor, output_tensor)
    dtype = str(output_tensor["dtype"])
    storage_type = cpu_storage_type(dtype) if target == "cpu" else cuda_storage_type(dtype)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "copy_body": _copy_body(input_tensor, update_tensor, starts),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    update_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> list[int]:
    if node["op"] != "slice_scatter":
        raise ValueError(f"Unsupported collection op: {node['op']}")
    if len(node.get("inputs", [])) != 2:
        raise ValueError("slice_scatter expects two tensor inputs")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("slice_scatter expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in COLLECTION_DTYPES:
        raise NotImplementedError(f"slice_scatter lowering does not support dtype {output_tensor['dtype']}")
    if str(input_tensor["dtype"]) != dtype or str(update_tensor["dtype"]) != dtype:
        raise ValueError("slice_scatter input, update, and output dtype must match")
    attrs = node.get("attrs", {})
    starts = normalize_slice_scatter_attrs(
        attrs.get("start_indices"),
        input_tensor["shape"],
        update_tensor["shape"],
    )
    expected_shape = resolve_slice_scatter_shape(input_tensor["shape"], update_tensor["shape"], starts)
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("slice_scatter output shape does not match input shape")
    return starts


def _copy_body(
    input_tensor: Mapping[str, Any],
    update_tensor: Mapping[str, Any],
    starts: list[int],
) -> str:
    output_shape = [int(axis) for axis in input_tensor["shape"]]
    update_shape = [int(axis) for axis in update_tensor["shape"]]
    update_strides = _dense_strides(update_shape)
    lines = [
        "  int64_t remaining = idx;",
        "  int64_t update_idx = 0;",
        "  int64_t coord = 0;",
        "  bool in_slice = true;",
    ]
    for axis in range(len(output_shape) - 1, -1, -1):
        output_extent = output_shape[axis]
        update_extent = update_shape[axis]
        update_stride = update_strides[axis]
        start = starts[axis]
        stop = start + update_extent
        lines.append(f"  coord = remaining % {output_extent};")
        lines.append(f"  remaining = remaining / {output_extent};")
        lines.append(f"  in_slice = in_slice && coord >= {start} && coord < {stop};")
        lines.append(f"  update_idx += (coord - {start}) * {update_stride};")
    lines.append("  y[idx] = in_slice ? update[update_idx] : x[idx];")
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
    update_tensor = tensor_map[node["inputs"][1]]
    output_tensor = tensor_map[node["outputs"][0]]
    attrs = node.get("attrs", {})
    signature = {
        "op": "slice_scatter",
        "input_shape": list(input_tensor["shape"]),
        "update_shape": list(update_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "start_indices": list(attrs.get("start_indices", [])),
        "dtype": str(output_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"slice_scatter_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


def _c_ident(name: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"_{ident}"
    return ident


SLICE_SCATTER_LOWERING = OpLowering(
    op_name="slice_scatter",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
