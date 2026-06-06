from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident, shape_spec_dim_expr
from dinoml.ops.collections import COLLECTION_DTYPES, normalize_dynamic_slice_attrs
from dinoml.shapes import max_shape, normalize_shape


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
    extra_args = _launch_args(node, tensor_map)
    args = f"ptr_{x}, ptr_{out}, runtime_numel_{out}"
    if extra_args:
        args = f"{args}, {', '.join(extra_args)}"
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
    dynamic_dims = _dynamic_dim_sources(input_tensor, output_tensor)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "copy_body": _copy_body(input_tensor, output_tensor),
        "rank": len(output_tensor["shape"]),
        "start_exprs": [shape_spec_dim_expr(start, dynamic_dims) for start in starts],
        "size_exprs": [shape_spec_dim_expr(size, dynamic_dims) for size in sizes],
        "stride_exprs": _input_stride_exprs(input_tensor, dynamic_dims),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> tuple[list[Any], list[Any]]:
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
        input_tensor.get("shape_spec", input_tensor["shape"]),
    )
    expected_shape_spec = normalize_shape(sizes)
    expected_shape = max_shape(expected_shape_spec)
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("dynamic_slice output shape does not match slice_sizes")
    if expected_shape_spec != normalize_shape(output_tensor.get("shape_spec", output_tensor["shape"])):
        raise ValueError("dynamic_slice output shape_spec does not match slice_sizes")
    return starts, sizes


def _copy_body(
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> str:
    output_shape = [int(axis) for axis in output_tensor["shape"]]
    lines = [
        "  int64_t remaining = idx;",
        "  int64_t input_idx = 0;",
        "  int64_t coord = 0;",
    ]
    for axis in range(len(output_shape) - 1, -1, -1):
        lines.append(f"  coord = remaining % size_{axis};")
        lines.append(f"  remaining = remaining / size_{axis};")
        lines.append(f"  input_idx += (coord + start_{axis}) * stride_{axis};")
    lines.append("  y[idx] = x[input_idx];")
    return "\n".join(lines)

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


def _launch_args(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> list[str]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    starts, sizes = _validate_node_contract(node, input_tensor, output_tensor)
    dynamic_dims = _dynamic_dim_sources(input_tensor, output_tensor)
    stride_exprs = _input_stride_exprs(input_tensor, dynamic_dims)
    args: list[str] = []
    for axis in range(len(starts)):
        args.append(shape_spec_dim_expr(starts[axis], dynamic_dims))
        args.append(shape_spec_dim_expr(sizes[axis], dynamic_dims))
        args.append(stride_exprs[axis])
    return args


def _input_stride_exprs(input_tensor: Mapping[str, Any], dynamic_dims: Mapping[str, str]) -> list[str]:
    shape_spec = input_tensor.get("shape_spec", input_tensor["shape"])
    rank = len(shape_spec)
    strides = ["1"] * rank
    running: list[str] = []
    for axis in range(rank - 1, -1, -1):
        if running:
            strides[axis] = " * ".join(f"({item})" for item in running)
        running.insert(0, shape_spec_dim_expr(shape_spec[axis], dynamic_dims))
    return strides


def _dynamic_dim_sources(
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> dict[str, str]:
    sources: dict[str, str] = {}
    _record_tensor_dim_sources(sources, input_tensor, f"shape_{_c_ident(str(input_tensor['name']))}")
    _record_tensor_dim_sources(sources, output_tensor, f"shape_{_c_ident(str(output_tensor['name']))}")
    return sources


def _record_tensor_dim_sources(
    sources: dict[str, str],
    tensor: Mapping[str, Any],
    prefix: str,
) -> None:
    for axis, dim in enumerate(tensor.get("shape_spec", tensor["shape"])):
        _record_dim_sources(sources, dim, f"{prefix}_{axis}")


def _record_dim_sources(sources: dict[str, str], dim: Any, expr: str) -> None:
    if isinstance(dim, int):
        return
    kind = dim.get("kind")
    if kind == "dim":
        sources.setdefault(str(dim["name"]), expr)
        return
    if kind == "int_expr":
        _record_dim_sources(sources, dim["lhs"], expr)
        _record_dim_sources(sources, dim["rhs"], expr)
        return
    raise ValueError(f"Unsupported shape dimension kind: {kind!r}")


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
