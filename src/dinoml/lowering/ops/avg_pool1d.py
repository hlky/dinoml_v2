from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.ops.pooling import POOLING_DTYPES, normalize_avg_pool1d_attrs, resolve_avg_pool1d_shape
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "avg_pool1d")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("avg_pool1d_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("avg_pool1d_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "avg_pool1d")
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    out = _c_ident(node["outputs"][0])
    args = f"ptr_{x}, ptr_{out}, runtime_numel_{out}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "avg_pool1d")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    kernel, stride, padding = _validate_node_contract(node, input_tensor, output_tensor)
    dtype = str(output_tensor["dtype"])
    storage_type = target_storage_type(dtype, target)
    n, c, length = [int(dim) for dim in input_tensor["shape"]]
    out_n, out_c, out_length = [int(dim) for dim in output_tensor["shape"]]
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "n": n,
        "c": c,
        "length": length,
        "out_n": out_n,
        "out_c": out_c,
        "out_length": out_length,
        "kernel_size": kernel[0],
        "stride": stride[0],
        "padding": padding[0],
        "divisor": kernel[0],
        "block_size": _gpu_block_size(target),
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> tuple[list[int], list[int], list[int]]:
    if node["op"] != "avg_pool1d":
        raise ValueError(f"Unsupported pooling op: {node['op']}")
    if len(node.get("inputs", [])) != 1:
        raise ValueError("avg_pool1d expects one tensor input")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("avg_pool1d expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in POOLING_DTYPES:
        raise NotImplementedError(f"avg_pool1d lowering does not support dtype {output_tensor['dtype']}")
    if str(input_tensor["dtype"]) != dtype:
        raise ValueError("avg_pool1d input and output dtype must match")
    attrs = node.get("attrs", {})
    kernel, stride, padding = normalize_avg_pool1d_attrs(
        attrs.get("kernel_size"),
        attrs.get("stride"),
        attrs.get("padding", (0,)),
    )
    expected_shape = resolve_avg_pool1d_shape(input_tensor["shape"], kernel, stride, padding)
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("avg_pool1d output shape does not match attrs")
    return kernel, stride, padding


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    attrs = node.get("attrs", {})
    kernel, stride, padding = normalize_avg_pool1d_attrs(
        attrs.get("kernel_size"),
        attrs.get("stride"),
        attrs.get("padding", (0,)),
    )
    signature = {
        "op": "avg_pool1d",
        "input_shape": list(input_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "kernel_size": kernel,
        "stride": stride,
        "padding": padding,
        "dtype": str(output_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"avg_pool1d_{digest}"


def _gpu_block_size(target: str) -> int:
    if target == "rocm":
        return 128
    return 256


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


AVG_POOL1D_LOWERING = OpLowering(
    op_name="avg_pool1d",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
