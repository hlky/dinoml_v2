from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.cpp_types import cpu_storage_type, cuda_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.ops.creation import RANDN_DTYPES, infer_randn_shape_with_attrs
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    context = _context(target, node, tensor_map)
    if target == "cpu":
        return _render_template("randn_cpu.cpp.j2", context)
    if target == "cuda":
        return _render_template("randn_cuda.cu.j2", context)
    raise ValueError(f"Unsupported randn target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    func = _function_name(node, tensor_map)
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{out}, runtime_numel_{out}"
    if target == "cpu":
        return f"if (int err = {func}({args})) return err;"
    if target == "cuda":
        return f"if (int err = {func}({args}, session->stream)) return err;"
    raise ValueError(f"Unsupported randn target: {target}")


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    if target not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported randn target: {target}")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output_tensor = tensor_map[node["outputs"][0]]
    _validate_node_contract(node, output_tensor)
    dtype = str(output_tensor["dtype"])
    storage_type = cpu_storage_type(dtype) if target == "cpu" else cuda_storage_type(dtype)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "seed_literal": f"{int(node.get('attrs', {}).get('seed', 0))}ull",
        "block_size": 256,
    }


def _validate_node_contract(node: Mapping[str, Any], output_tensor: Mapping[str, Any]) -> None:
    if node["op"] != "randn":
        raise ValueError(f"Unsupported creation op: {node['op']}")
    if node.get("inputs"):
        raise ValueError("randn expects no tensor inputs")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("randn expects exactly one output")
    if str(output_tensor["dtype"]) not in RANDN_DTYPES:
        raise NotImplementedError(f"randn lowering does not support dtype {output_tensor['dtype']}")
    expected_shape = infer_randn_shape_with_attrs([], node.get("attrs", {}))
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("randn output shape does not match shape attr")


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    output_tensor = tensor_map[node["outputs"][0]]
    signature = {
        "op": "randn",
        "shape": list(output_tensor["shape"]),
        "dtype": str(output_tensor["dtype"]),
        "seed": int(node.get("attrs", {}).get("seed", 0)),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"randn_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


RANDN_LOWERING = OpLowering(
    op_name="randn",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
