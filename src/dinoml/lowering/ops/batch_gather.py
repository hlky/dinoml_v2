from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.cpp_types import cpu_storage_type, cuda_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.gather import _index_storage_type
from dinoml.ops.collections import (
    COLLECTION_DTYPES,
    GATHER_INDEX_DTYPES,
    normalize_batch_gather_attrs,
    resolve_batch_gather_shape,
)


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    context = _context(target, node, tensor_map)
    if target == "cpu":
        return _render_template("gather_cpu.cpp.j2", context)
    if target == "cuda":
        return _render_template("gather_cuda.cu.j2", context)
    raise ValueError(f"Unsupported batch_gather target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    indices = _c_ident(node["inputs"][1])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{x}, ptr_{indices}, ptr_{out}, runtime_numel_{out}"
    if target == "cpu":
        return f"if (int err = {func}({args})) return err;"
    if target == "cuda":
        return f"if (int err = {func}({args}, session->stream)) return err;"
    raise ValueError(f"Unsupported batch_gather target: {target}")


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    if target not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported batch_gather target: {target}")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    index_tensor = tensor_map[node["inputs"][1]]
    output_tensor = tensor_map[node["outputs"][0]]
    _validate_node_contract(node, input_tensor, index_tensor, output_tensor)
    dtype = str(output_tensor["dtype"])
    storage_type = cpu_storage_type(dtype) if target == "cpu" else cuda_storage_type(dtype)
    index_storage_type = _index_storage_type(str(index_tensor["dtype"]))
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "index_storage_type": index_storage_type,
        "copy_body": _copy_body(input_tensor, index_tensor, target),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    index_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> None:
    if node["op"] != "batch_gather":
        raise ValueError(f"Unsupported collection op: {node['op']}")
    if len(node.get("inputs", [])) != 2:
        raise ValueError("batch_gather expects two tensor inputs")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("batch_gather expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in COLLECTION_DTYPES:
        raise NotImplementedError(f"batch_gather lowering does not support dtype {output_tensor['dtype']}")
    if str(input_tensor["dtype"]) != dtype:
        raise ValueError("batch_gather input and output dtype must match")
    if str(index_tensor["dtype"]) not in GATHER_INDEX_DTYPES:
        raise ValueError(f"batch_gather indices must have dtype int64 or int32, got {index_tensor['dtype']}")
    normalize_batch_gather_attrs(input_tensor["shape"], index_tensor["shape"])
    expected_shape = resolve_batch_gather_shape(input_tensor["shape"], index_tensor["shape"])
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("batch_gather output shape does not match expected [B, K, ...]")


def _copy_body(
    input_tensor: Mapping[str, Any],
    index_tensor: Mapping[str, Any],
    target: str,
) -> str:
    input_shape = [int(axis) for axis in input_tensor["shape"]]
    index_shape = [int(axis) for axis in index_tensor["shape"]]
    batch = input_shape[0]
    source_rows = input_shape[1]
    gather_count = index_shape[1]
    slice_size = _numel(input_shape[2:])
    input_batch_stride = source_rows * slice_size
    output_batch_stride = gather_count * slice_size
    lines = [
        f"  const int64_t slice_offset = idx % {slice_size};",
        f"  const int64_t batch = idx / {output_batch_stride};",
        f"  const int64_t within_batch = idx - batch * {output_batch_stride};",
        f"  const int64_t k = within_batch / {slice_size};",
        f"  const int64_t selected_index = static_cast<int64_t>(index[batch * {gather_count} + k]);",
        f"  if (batch < 0 || batch >= {batch} || selected_index < 0 || selected_index >= {source_rows}) {{",
    ]
    if target == "cpu":
        lines.append('    return dino_runtime_fail("batch_gather index out of bounds");')
    else:
        lines.append(f"    assert(batch >= 0 && batch < {batch} && selected_index >= 0 && selected_index < {source_rows});")
        lines.append("    return;")
    lines.extend(
        [
            "  }",
            f"  const int64_t input_idx = batch * {input_batch_stride} + selected_index * {slice_size} + slice_offset;",
            "  y[idx] = x[input_idx];",
        ]
    )
    return "\n".join(lines)


def _numel(shape: list[int]) -> int:
    total = 1
    for axis in shape:
        total *= axis
    return total


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    index_tensor = tensor_map[node["inputs"][1]]
    output_tensor = tensor_map[node["outputs"][0]]
    signature = {
        "op": "batch_gather",
        "input_shape": list(input_tensor["shape"]),
        "index_shape": list(index_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "dtype": str(output_tensor["dtype"]),
        "index_dtype": str(index_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"batch_gather_{digest}"


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
    ident = re.sub(r"_(\d+)$", r"__\1", ident)
    return ident


BATCH_GATHER_LOWERING = OpLowering(
    op_name="batch_gather",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
