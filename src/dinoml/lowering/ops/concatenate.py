from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.cpp_types import cpu_storage_type, cuda_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.ops.collections import COLLECTION_DTYPES, normalize_concatenate_dim, resolve_concatenate_shape


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    context = _context(target, node, tensor_map)
    if target == "cpu":
        return _render_template("concatenate_cpu.cpp.j2", context)
    if target == "cuda":
        return _render_template("concatenate_cuda.cu.j2", context)
    raise ValueError(f"Unsupported concatenate target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    func = _function_name(node, tensor_map)
    inputs = ", ".join(f"ptr_{_c_ident(name)}" for name in node["inputs"])
    out = _c_ident(node["outputs"][0])
    args = f"{inputs}, ptr_{out}, runtime_numel_{out}"
    if target == "cpu":
        return f"if (int err = {func}({args})) return err;"
    if target == "cuda":
        return f"if (int err = {func}({args}, session->stream)) return err;"
    raise ValueError(f"Unsupported concatenate target: {target}")


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    if target not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported concatenate target: {target}")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensors = [tensor_map[name] for name in node["inputs"]]
    output_tensor = tensor_map[node["outputs"][0]]
    dim = _validate_node_contract(node, input_tensors, output_tensor)
    dtype = str(output_tensor["dtype"])
    storage_type = cpu_storage_type(dtype) if target == "cpu" else cuda_storage_type(dtype)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "input_params": _input_params(node, storage_type),
        "input_args": _input_args(node),
        "null_checks": _null_checks(node),
        "copy_body": _copy_body(input_tensors, output_tensor, dim),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensors: list[Mapping[str, Any]],
    output_tensor: Mapping[str, Any],
) -> int:
    if node["op"] != "concatenate":
        raise ValueError(f"Unsupported collection op: {node['op']}")
    if not input_tensors:
        raise ValueError("concatenate expects a non-empty sequence of tensors")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("concatenate expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in COLLECTION_DTYPES:
        raise NotImplementedError(f"concatenate lowering does not support dtype {output_tensor['dtype']}")
    if any(str(tensor["dtype"]) != dtype for tensor in input_tensors):
        raise ValueError("concatenate input and output dtype must match")
    dim = normalize_concatenate_dim(node.get("attrs", {}).get("dim", 0), len(input_tensors[0]["shape"]))
    expected_shape = resolve_concatenate_shape([tensor["shape"] for tensor in input_tensors], dim)
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("concatenate output shape does not match input shapes")
    return dim


def _input_params(node: Mapping[str, Any], storage_type: str) -> list[dict[str, str]]:
    return [{"ident": _c_ident(name), "storage_type": storage_type} for name in node["inputs"]]


def _input_args(node: Mapping[str, Any]) -> str:
    return ", ".join(f"ptr_{_c_ident(name)}" for name in node["inputs"])


def _null_checks(node: Mapping[str, Any]) -> str:
    names = [f"x{idx}" for idx, _ in enumerate(node["inputs"])] + ["y"]
    return " || ".join(f"{name} == nullptr" for name in names)


def _copy_body(input_tensors: list[Mapping[str, Any]], output_tensor: Mapping[str, Any], dim: int) -> str:
    output_shape = [int(axis) for axis in output_tensor["shape"]]
    inner = 1
    for axis in output_shape[dim + 1 :]:
        inner *= int(axis)
    concat_extent = int(output_shape[dim])
    lines = [
        f"  const int64_t inner = {inner};",
        f"  const int64_t concat_extent = {concat_extent};",
        "  const int64_t inner_idx = idx % inner;",
        "  const int64_t concat_idx = (idx / inner) % concat_extent;",
        "  const int64_t outer_idx = idx / (inner * concat_extent);",
    ]
    offset = 0
    for index, tensor in enumerate(input_tensors):
        axis_extent = int(tensor["shape"][dim])
        prefix = "if" if index == 0 else "else if"
        lines.append(f"  {prefix} (concat_idx < {offset + axis_extent}) {{")
        lines.append(
            f"    y[idx] = x{index}[(outer_idx * {axis_extent} + (concat_idx - {offset})) * inner + inner_idx];"
        )
        lines.append("  }")
        offset += axis_extent
    return "\n".join(lines)


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensors = [tensor_map[name] for name in node["inputs"]]
    output_tensor = tensor_map[node["outputs"][0]]
    signature = {
        "op": "concatenate",
        "input_shapes": [list(tensor["shape"]) for tensor in input_tensors],
        "output_shape": list(output_tensor["shape"]),
        "dim": int(node.get("attrs", {}).get("dim", 0)),
        "dtype": str(output_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"concatenate_{digest}"


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


CONCATENATE_LOWERING = OpLowering(
    op_name="concatenate",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
