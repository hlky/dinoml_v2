from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.ops.base import OpLowering


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    context = _context(node, tensor_map)
    if target == "cpu":
        return _render_template("shape_buffer_count_true_cpu.cpp.j2", context)
    if target == "cuda":
        return _render_template("shape_buffer_count_true_cuda.cu.j2", context)
    raise ValueError(f"Unsupported shape-buffer count target: {target}")


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
    if target == "cpu":
        args = f"ptr_{x}, ptr_{out}, runtime_numel_{x}, session->shape_{out}.data()"
        return f"if (int err = {func}({args})) return err;"
    if target == "cuda":
        args = f"ptr_{x}, ptr_{out}, runtime_numel_{x}, session->shape_{out}, session->stream"
        return f"if (int err = {func}({args})) return err;"
    raise ValueError(f"Unsupported shape-buffer count target: {target}")


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    if target not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported shape-buffer count target: {target}")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    _validate_node_contract(node, input_tensor, output_tensor)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> None:
    if node["op"] != "_shape_buffer_count_true":
        raise ValueError(f"Unsupported internal op: {node['op']}")
    if len(node.get("inputs", [])) != 1:
        raise ValueError("_shape_buffer_count_true expects one tensor input")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("_shape_buffer_count_true expects exactly one output")
    if str(input_tensor["dtype"]) != "bool" or str(output_tensor["dtype"]) != "bool":
        raise NotImplementedError("_shape_buffer_count_true supports only bool tensors")
    if len(output_tensor["shape"]) != 1:
        raise NotImplementedError("_shape_buffer_count_true requires a rank-1 output")
    input_numel = 1
    for dim in input_tensor["shape"]:
        input_numel *= int(dim)
    if list(output_tensor["shape"]) != [input_numel]:
        raise ValueError("_shape_buffer_count_true output shape must equal flattened input capacity")


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    signature = {
        "op": "_shape_buffer_count_true",
        "input_shape": list(input_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"shape_buffer_count_true_{digest}"


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


SHAPE_BUFFER_COUNT_TRUE_LOWERING = OpLowering(
    op_name="_shape_buffer_count_true",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
