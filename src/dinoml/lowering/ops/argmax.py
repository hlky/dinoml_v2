from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.ops.reductions import ARGMAX_DTYPES, normalize_argmax_dim, resolve_argmax_shape
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "argmax")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("argmax_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("argmax_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "argmax")
    func = _function_name(node, tensor_map)
    inp = _c_ident(node["inputs"][0])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{inp}, ptr_{out}, runtime_numel_{out}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "argmax")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    _validate_node_contract(node, input_tensor, output_tensor)
    input_dtype = str(input_tensor["dtype"])
    input_storage_type = target_storage_type(input_dtype, target)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "warp_kernel": f"{_function_name(node, tensor_map)}_warp_kernel",
        "input_storage_type": input_storage_type,
        "output_storage_type": "int64_t",
        "input_dtype": input_dtype,
        "input_is_bool": input_dtype == "bool",
        "input_is_integer": input_dtype in {"int32", "int64"},
        "input_needs_nan_handling": input_dtype in {"float16", "float32", "bfloat16"},
        "cols": int(input_tensor["shape"][-1]),
        "cols_per_thread": (int(input_tensor["shape"][-1]) + 31) // 32,
        "use_warp_kernel": int(input_tensor["shape"][-1]) <= 4096,
        "rows_per_block": 8,
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> None:
    if node["op"] != "argmax":
        raise ValueError(f"Unsupported argmax op: {node['op']}")
    if len(node.get("inputs", [])) != 1:
        raise ValueError("argmax expects exactly one input")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("argmax expects exactly one output")
    input_dtype = str(input_tensor["dtype"])
    if input_dtype not in ARGMAX_DTYPES:
        raise NotImplementedError(f"argmax lowering does not support dtype {input_dtype}")
    if str(output_tensor["dtype"]) != "int64":
        raise ValueError(f"argmax output dtype must be int64, got {output_tensor['dtype']}")
    if not input_tensor["shape"]:
        raise ValueError("argmax requires a ranked tensor")
    attrs = node.get("attrs", {})
    dim = normalize_argmax_dim(attrs.get("dim", -1), len(input_tensor["shape"]))
    if dim != len(input_tensor["shape"]) - 1:
        raise NotImplementedError("argmax lowering currently supports only the last dimension")
    shape_spec = input_tensor.get("shape_spec", input_tensor["shape"])
    cols = input_tensor["shape"][-1]
    if not isinstance(shape_spec[-1], int) or not isinstance(cols, int) or int(cols) <= 0:
        raise ValueError("argmax lowering requires a positive static last dimension")
    expected = resolve_argmax_shape(input_tensor["shape"], dim, bool(attrs.get("keepdim", False)))
    if list(output_tensor["shape"]) != expected:
        raise ValueError("argmax output shape does not match argmax attrs")


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    attrs = node.get("attrs", {})
    dim = normalize_argmax_dim(attrs.get("dim", -1), len(input_tensor["shape"]))
    signature = {
        "op": "argmax",
        "shape": list(input_tensor["shape"]),
        "dtype": str(input_tensor["dtype"]),
        "dim": dim,
        "keepdim": bool(attrs.get("keepdim", False)),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"argmax_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


ARGMAX_LOWERING = OpLowering(
    op_name="argmax",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
