from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.ops.collections import COLLECTION_DTYPES, SPECIALIZED_PERMUTE_DIMS, normalize_permute_dims, resolve_permute_shape
from dinoml.lowering.shape_buffers import shape_spec_dim_expr


PERMUTE_OPS = ("permute", *SPECIALIZED_PERMUTE_DIMS)
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "permute")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("permute_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("permute_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "permute")
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
    supported_target_spec(target, "permute")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    dims = _validate_node_contract(node, input_tensor, output_tensor)
    dtype = str(output_tensor["dtype"])
    storage_type = target_storage_type(dtype, target)
    input_ident = _c_ident(str(input_tensor["name"]))
    output_ident = _c_ident(str(output_tensor["name"]))
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "copy_body": _copy_body(input_tensor, output_tensor, dims),
        "rank": len(output_tensor["shape"]),
        "extent_exprs": [f"shape_{output_ident}_{axis}" for axis in range(len(output_tensor["shape"]))],
        "stride_exprs": _input_stride_exprs(input_tensor, input_ident),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> list[int]:
    if str(node["op"]) not in PERMUTE_OPS:
        raise ValueError(f"Unsupported collection op: {node['op']}")
    if len(node.get("inputs", [])) != 1:
        raise ValueError("permute expects one tensor input")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("permute expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in COLLECTION_DTYPES:
        raise NotImplementedError(f"permute lowering does not support dtype {output_tensor['dtype']}")
    if str(input_tensor["dtype"]) != dtype:
        raise ValueError("permute input and output dtype must match")
    dims = _node_dims(node, len(input_tensor["shape"]))
    expected_shape = resolve_permute_shape(input_tensor["shape"], dims)
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("permute output shape does not match input shape")
    return dims


def _copy_body(
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
    dims: list[int],
) -> str:
    output_shape = [int(axis) for axis in output_tensor["shape"]]
    lines = [
        "  int64_t remaining = idx;",
        "  int64_t input_idx = 0;",
        "  int64_t coord = 0;",
    ]
    for output_axis in range(len(output_shape) - 1, -1, -1):
        input_axis = dims[output_axis]
        lines.append(f"  coord = remaining % extent_{output_axis};")
        lines.append(f"  remaining = remaining / extent_{output_axis};")
        lines.append(f"  input_idx += coord * stride_{input_axis};")
    lines.append("  y[idx] = x[input_idx];")
    return "\n".join(lines)


def _launch_args(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> list[str]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    _validate_node_contract(node, input_tensor, output_tensor)
    input_ident = _c_ident(str(input_tensor["name"]))
    output_ident = _c_ident(str(output_tensor["name"]))
    extents = [f"shape_{output_ident}_{axis}" for axis in range(len(output_tensor["shape"]))]
    strides = _input_stride_exprs(input_tensor, input_ident)
    args: list[str] = []
    for axis in range(len(extents)):
        args.append(extents[axis])
        args.append(strides[axis])
    return args


def _input_stride_exprs(input_tensor: Mapping[str, Any], input_ident: str) -> list[str]:
    rank = len(input_tensor["shape"])
    strides = ["1"] * rank
    running: list[str] = []
    for axis in range(rank - 1, -1, -1):
        if running:
            strides[axis] = " * ".join(f"({item})" for item in running)
        running.insert(0, f"shape_{input_ident}_{axis}")
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


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    dims = _node_dims(node, len(input_tensor["shape"]))
    op_name = str(node["op"])
    dims_id = "".join(str(dim) for dim in dims)
    signature = {
        "op": op_name,
        "input_shape": list(input_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "dims": dims,
        "dtype": str(output_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"{op_name}_{dims_id}_{digest}"


def _node_dims(node: Mapping[str, Any], rank: int) -> list[int]:
    op_name = str(node["op"])
    if op_name in SPECIALIZED_PERMUTE_DIMS:
        fixed_dims = list(SPECIALIZED_PERMUTE_DIMS[op_name])
        attrs_dims = node.get("attrs", {}).get("dims")
        if attrs_dims is None:
            return fixed_dims
        normalized_dims = normalize_permute_dims(attrs_dims, rank)
        if tuple(normalized_dims) != SPECIALIZED_PERMUTE_DIMS[op_name]:
            raise ValueError(f"{op_name} lowering requires fixed dims {fixed_dims}")
        return normalized_dims
    return normalize_permute_dims(node.get("attrs", {}).get("dims"), rank)


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


PERMUTE_LOWERINGS = {
    op_name: OpLowering(
        op_name=op_name,
        render_generated_kernel=render_generated_kernel,
        render_launch=render_launch,
        source_key=source_key,
        generated_function_name=generated_function_name,
    )
    for op_name in PERMUTE_OPS
}

PERMUTE_LOWERING = PERMUTE_LOWERINGS["permute"]
