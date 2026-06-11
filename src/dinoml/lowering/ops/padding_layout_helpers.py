from __future__ import annotations

from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import render_op_template, supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.collections import (
    PADDING_LAYOUT_HELPER_DTYPES,
    PADDING_LAYOUT_HELPER_OPS,
    resolve_padding_layout_shape,
)


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, str(node["op"]))
    context = _context(target, node, tensor_map)
    if spec.is_gpu:
        context.update(spec.gpu_template_context())
    template_name = f"padding_layout_{'gpu' if spec.is_gpu else 'cpu.cpp'}.j2"
    return render_op_template(template_name, context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, str(node["op"]))
    func = generated_function_name(target, node, tensor_map)
    x_ident = _c_ident(str(node["inputs"][0]))
    out_ident = _c_ident(str(node["outputs"][0]))
    args = f"ptr_{x_ident}, ptr_{out_ident}, runtime_numel_{out_ident}"
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, str(node["op"]))
    return f"{target}:{generated_function_name(target, node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    validated = _validate_node_contract(node, tensor_map)
    return f"{validated['op_name']}_{validated['dtype']}"


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    validated = _validate_node_contract(node, tensor_map)
    return {
        "func": generated_function_name(target, node, tensor_map),
        "kernel": f"{generated_function_name(target, node, tensor_map)}_kernel",
        "storage_type": target_storage_type(str(validated["dtype"]), target),
        "padded_channels": int(validated["padded_channels"]),
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    op_name = str(node["op"])
    if op_name not in PADDING_LAYOUT_HELPER_OPS:
        raise ValueError(f"Unsupported padding layout helper op: {op_name}")
    if len(node.get("inputs", ())) != 1 or len(node.get("outputs", ())) != 1:
        raise ValueError(f"{op_name} expects one input and one output")
    input_tensor = tensor_map[str(node["inputs"][0])]
    output_tensor = tensor_map[str(node["outputs"][0])]
    input_dtype = str(input_tensor["dtype"])
    output_dtype = str(output_tensor["dtype"])
    if input_dtype not in PADDING_LAYOUT_HELPER_DTYPES or output_dtype not in PADDING_LAYOUT_HELPER_DTYPES:
        raise NotImplementedError(f"{op_name} lowering does not support dtype {output_dtype}")
    if input_dtype != output_dtype:
        raise ValueError(f"{op_name} lowering requires matching input/output dtypes")
    if op_name == "nhwc3to4":
        expected_rank = 4
        padded_channels = 4
    elif op_name == "nhwc3to8":
        expected_rank = 4
        padded_channels = 8
    else:
        expected_rank = 5
        padded_channels = 8
    expected_shape = resolve_padding_layout_shape(
        op_name,
        input_tensor["shape"],
        expected_rank=expected_rank,
        padded_channels=padded_channels,
    )
    if list(output_tensor["shape"]) != expected_shape:
        raise ValueError(f"{op_name} output shape does not match input layout contract")
    return {
        "op_name": op_name,
        "dtype": output_dtype,
        "padded_channels": padded_channels,
    }


PADDING_LAYOUT_HELPER_LOWERINGS = {
    op_name: OpLowering(
        op_name=op_name,
        render_generated_kernel=render_generated_kernel,
        render_launch=render_launch,
        source_key=source_key,
        generated_function_name=generated_function_name,
    )
    for op_name in PADDING_LAYOUT_HELPER_OPS
}
