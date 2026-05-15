from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.cpp_types import cpu_storage_type, cuda_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.ops.pooling import POOLING_DTYPES, normalize_max_pool2d_attrs, resolve_max_pool2d_shape
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    context = _context(target, node, tensor_map)
    if target == "cpu":
        return _render_template("max_pool2d_cpu.cpp.j2", context)
    if target == "cuda":
        return _render_template("max_pool2d_cuda.cu.j2", context)
    raise ValueError(f"Unsupported max_pool2d target: {target}")


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
    args = f"ptr_{x}, ptr_{out}, runtime_numel_{out}"
    if target == "cpu":
        return f"if (int err = {func}({args})) return err;"
    if target == "cuda":
        return f"if (int err = {func}({args}, session->stream)) return err;"
    raise ValueError(f"Unsupported max_pool2d target: {target}")


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    if target not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported max_pool2d target: {target}")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    kernel, stride, padding = _validate_node_contract(node, input_tensor, output_tensor)
    dtype = str(output_tensor["dtype"])
    storage_type = cpu_storage_type(dtype) if target == "cpu" else cuda_storage_type(dtype)
    n, c, height, width = [int(dim) for dim in input_tensor["shape"]]
    out_n, out_c, out_height, out_width = [int(dim) for dim in output_tensor["shape"]]
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": storage_type,
        "n": n,
        "c": c,
        "height": height,
        "width": width,
        "out_n": out_n,
        "out_c": out_c,
        "out_height": out_height,
        "out_width": out_width,
        "kernel_h": kernel[0],
        "kernel_w": kernel[1],
        "stride_h": stride[0],
        "stride_w": stride[1],
        "pad_h": padding[0],
        "pad_w": padding[1],
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> tuple[list[int], list[int], list[int]]:
    if node["op"] != "max_pool2d":
        raise ValueError(f"Unsupported pooling op: {node['op']}")
    if len(node.get("inputs", [])) != 1:
        raise ValueError("max_pool2d expects one tensor input")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("max_pool2d expects exactly one output")
    dtype = str(output_tensor["dtype"])
    if dtype not in POOLING_DTYPES:
        raise NotImplementedError(f"max_pool2d lowering does not support dtype {output_tensor['dtype']}")
    if str(input_tensor["dtype"]) != dtype:
        raise ValueError("max_pool2d input and output dtype must match")
    attrs = node.get("attrs", {})
    kernel, stride, padding = normalize_max_pool2d_attrs(
        attrs.get("kernel_size"),
        attrs.get("stride"),
        attrs.get("padding", (0, 0)),
    )
    expected_shape = resolve_max_pool2d_shape(input_tensor["shape"], kernel, stride, padding)
    if list(expected_shape) != list(output_tensor["shape"]):
        raise ValueError("max_pool2d output shape does not match attrs")
    return kernel, stride, padding


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    attrs = node.get("attrs", {})
    kernel, stride, padding = normalize_max_pool2d_attrs(
        attrs.get("kernel_size"),
        attrs.get("stride"),
        attrs.get("padding", (0, 0)),
    )
    signature = {
        "op": "max_pool2d",
        "input_shape": list(input_tensor["shape"]),
        "output_shape": list(output_tensor["shape"]),
        "kernel_size": kernel,
        "stride": stride,
        "padding": padding,
        "dtype": str(output_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"max_pool2d_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


MAX_POOL2D_LOWERING = OpLowering(
    op_name="max_pool2d",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
