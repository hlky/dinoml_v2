from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import render_op_template, supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "swiglu")
    context = _context(target, node, tensor_map)
    if spec.is_gpu:
        context.update(spec.gpu_template_context())
    return render_op_template(f"swiglu_{'gpu' if spec.is_gpu else 'cpu.cpp'}.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "swiglu")
    input_name = str(node["inputs"][0])
    output_name = str(node["outputs"][0])
    func = _function_name(node, tensor_map)
    args = f"ptr_{_c_ident(input_name)}, ptr_{_c_ident(output_name)}, runtime_numel_{_c_ident(output_name)}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "swiglu")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[str(node["inputs"][0])]
    output_tensor = tensor_map[str(node["outputs"][0])]
    dtype = str(input_tensor["dtype"])
    input_shape = list(input_tensor["shape"])
    output_shape = list(output_tensor["shape"])
    if dtype not in {"float16", "float32", "bfloat16"}:
        raise NotImplementedError(f"swiglu does not support dtype {dtype}")
    if not input_shape or int(input_shape[-1]) % 2 != 0:
        raise ValueError("swiglu expects a static even input last dimension")
    hidden = int(input_shape[-1]) // 2
    expected_output_shape = list(input_shape)
    expected_output_shape[-1] = hidden
    if output_shape != expected_output_shape:
        raise ValueError("swiglu output shape must equal input shape with last dim / 2")
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": target_storage_type(dtype, target),
        "cpu_storage_type": target_storage_type(dtype, "cpu"),
        "hidden": hidden,
    }


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[str(node["inputs"][0])]
    signature = {
        "op": "swiglu",
        "shape": list(input_tensor["shape"]),
        "dtype": str(input_tensor["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"swiglu_{digest}"


SWIGLU_LOWERING = OpLowering(
    op_name="swiglu",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
