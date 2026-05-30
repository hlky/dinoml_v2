from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "qkv_split")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template("qkv_split_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return _render_template("qkv_split_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "qkv_split")
    func = _function_name(node, tensor_map)
    qkv = _c_ident(node["inputs"][0])
    q, k, v = (_c_ident(name) for name in node["outputs"])
    args = f"ptr_{qkv}, ptr_{q}, ptr_{k}, ptr_{v}, runtime_numel_{q}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "qkv_split")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    qkv_tensor = tensor_map[node["inputs"][0]]
    output_tensors = [tensor_map[name] for name in node["outputs"]]
    _validate_node_contract(node, qkv_tensor, output_tensors)
    dtype = str(qkv_tensor["dtype"])
    hidden = int(output_tensors[0]["shape"][-1])
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": target_storage_type(dtype, target),
        "cpu_storage_type": target_storage_type(dtype, "cpu"),
        "hidden": hidden,
        "vector_width": 8 if dtype in {"float16", "bfloat16"} and hidden % 8 == 0 else 1,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    qkv_tensor: Mapping[str, Any],
    output_tensors: list[Mapping[str, Any]],
) -> None:
    if str(node["op"]) != "qkv_split":
        raise ValueError(f"Unsupported qkv split op: {node['op']}")
    if len(output_tensors) != 3:
        raise ValueError("qkv_split expects exactly three outputs")
    dtype = str(qkv_tensor["dtype"])
    if dtype not in {"float16", "float32", "bfloat16"}:
        raise NotImplementedError(f"qkv_split does not support dtype {dtype}")
    if not qkv_tensor["shape"] or not isinstance(qkv_tensor["shape"][-1], int):
        raise ValueError("qkv_split requires a static last dimension")
    if int(qkv_tensor["shape"][-1]) % 3 != 0:
        raise ValueError("qkv_split input last dimension must be divisible by 3")
    expected = list(qkv_tensor["shape"])
    expected[-1] = int(expected[-1]) // 3
    for output_tensor in output_tensors:
        if str(output_tensor["dtype"]) != dtype:
            raise ValueError("qkv_split output dtype must match input dtype")
        if list(output_tensor["shape"]) != expected:
            raise ValueError("qkv_split output shape must equal input shape with last dim / 3")


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    qkv_tensor = tensor_map[node["inputs"][0]]
    signature = {
        "op": "qkv_split",
        "shape": list(qkv_tensor["shape"]),
        "dtype": str(qkv_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"qkv_split_{digest}"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


QKV_SPLIT_LOWERING = OpLowering(
    op_name="qkv_split",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
