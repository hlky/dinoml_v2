from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.gather import _index_storage_type
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.collections import INDEX_ADD_DTYPES, GATHER_INDEX_DTYPES, normalize_index_add_attrs, resolve_index_add_shape


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "index_add")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("index_add_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("index_add_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "index_add")
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    index = _c_ident(node["inputs"][1])
    source = _c_ident(node["inputs"][2])
    out = _c_ident(node["outputs"][0])
    rank = len(tensor_map[node["outputs"][0]]["shape"])
    args = [
        f"ptr_{x}",
        f"ptr_{index}",
        f"ptr_{source}",
        f"ptr_{out}",
        f"runtime_numel_{out}",
        f"runtime_numel_{source}",
        *[f"static_cast<int64_t>(shape_{out}_{axis})" for axis in range(rank)],
        *[f"static_cast<int64_t>(shape_{source}_{axis})" for axis in range(rank)],
    ]
    joined = ", ".join(args)
    if not spec.is_gpu:
        return f"if (int err = {func}({joined})) return err;"
    return f"if (int err = {func}({joined}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "index_add")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    index_tensor = tensor_map[node["inputs"][1]]
    source_tensor = tensor_map[node["inputs"][2]]
    output_tensor = tensor_map[node["outputs"][0]]
    dim = _validate_node_contract(node, input_tensor, index_tensor, source_tensor, output_tensor)
    dtype = str(output_tensor["dtype"])
    rank = len(output_tensor["shape"])
    return {
        "func": _function_name(node, tensor_map),
        "copy_kernel": f"{_function_name(node, tensor_map)}_copy_kernel",
        "scatter_kernel": f"{_function_name(node, tensor_map)}_scatter_kernel",
        "storage_type": target_storage_type(dtype, target),
        "index_storage_type": _index_storage_type(str(index_tensor["dtype"])),
        "copy_body": "  y[idx] = x[idx];",
        "scatter_body": _scatter_body(rank, dim, target, _function_name(node, tensor_map)),
        "rank": rank,
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    index_tensor: Mapping[str, Any],
    source_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> int:
    if node["op"] != "index_add":
        raise ValueError(f"Unsupported collection op: {node['op']}")
    if len(node.get("inputs", [])) != 3:
        raise ValueError("index_add expects three tensor inputs")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("index_add expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in INDEX_ADD_DTYPES:
        raise NotImplementedError(f"index_add lowering does not support dtype {output_tensor['dtype']}")
    if str(input_tensor["dtype"]) != dtype or str(source_tensor["dtype"]) != dtype:
        raise ValueError("index_add input, source, and output dtype must match")
    if str(index_tensor["dtype"]) not in GATHER_INDEX_DTYPES:
        raise ValueError(f"index_add index must have dtype int64 or int32, got {index_tensor['dtype']}")
    dim = normalize_index_add_attrs(
        node.get("attrs", {}).get("dim", 0),
        input_tensor.get("shape_spec", input_tensor["shape"]),
        index_tensor.get("shape_spec", index_tensor["shape"]),
        source_tensor.get("shape_spec", source_tensor["shape"]),
    )
    expected_shape = resolve_index_add_shape(input_tensor["shape"], index_tensor["shape"], source_tensor["shape"], dim)
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("index_add output shape must match input shape")
    expected_shape_spec = list(input_tensor.get("shape_spec", input_tensor["shape"]))
    if list(output_tensor.get("shape_spec", output_tensor["shape"])) != expected_shape_spec:
        raise ValueError("index_add output shape_spec must match input shape_spec")
    return dim


def _scatter_body(rank: int, dim: int, target: str, func_name: str) -> str:
    lines = [
        "  int64_t remaining = src_idx;",
        "  int64_t out_idx = 0;",
        "  int64_t out_stride = 1;",
        "  int64_t coord = 0;",
        "  int64_t selected_index = 0;",
    ]
    for axis in range(rank - 1, -1, -1):
        lines.append(f"  coord = remaining % source_extent_{axis};")
        lines.append(f"  remaining = remaining / source_extent_{axis};")
        if axis == dim:
            lines.append("  selected_index = static_cast<int64_t>(index[coord]);")
            lines.append(f"  if (selected_index < 0 || selected_index >= out_extent_{axis}) {{")
            if target == "cpu":
                lines.append('    return dino_runtime_fail("index_add index out of bounds");')
            else:
                lines.append(f"    assert(selected_index >= 0 && selected_index < out_extent_{axis});")
                lines.append("    return;")
            lines.append("  }")
            lines.append("  out_idx += selected_index * out_stride;")
        else:
            lines.append("  out_idx += coord * out_stride;")
        if axis > 0:
            lines.append(f"  out_stride *= out_extent_{axis};")
    lines.append(f"  {func_name}_atomic_add(y + out_idx, dinoml::math::cast<float>(source[src_idx]));")
    return "\n".join(lines)


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    index_tensor = tensor_map[node["inputs"][1]]
    source_tensor = tensor_map[node["inputs"][2]]
    output_tensor = tensor_map[node["outputs"][0]]
    attrs = node.get("attrs", {})
    signature = {
        "op": "index_add",
        "input_shape": list(input_tensor["shape"]),
        "index_shape": list(index_tensor["shape"]),
        "source_shape": list(source_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "dim": int(attrs.get("dim", 0)),
        "dtype": str(output_tensor["dtype"]),
        "index_dtype": str(index_tensor["dtype"]),
        "rank": len(output_tensor["shape"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"index_add_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


INDEX_ADD_LOWERING = OpLowering(
    op_name="index_add",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
