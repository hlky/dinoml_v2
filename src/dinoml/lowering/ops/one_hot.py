from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import render_op_template, supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.collections import ONE_HOT_INPUT_DTYPES, normalize_one_hot_num_classes, resolve_one_hot_shape


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, "one_hot")
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return render_op_template("one_hot_cpu.cpp.j2", context)
    context.update(spec.gpu_template_context())
    return render_op_template("one_hot_gpu.j2", context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, "one_hot")
    func = _function_name(node, tensor_map)
    x = _c_ident(node["inputs"][0])
    out = _c_ident(node["outputs"][0])
    args = [f"ptr_{x}", f"ptr_{out}", f"runtime_numel_{x}", f"runtime_numel_{out}"]
    joined = ", ".join(args)
    if not spec.is_gpu:
        return f"if (int err = {func}({joined})) return err;"
    return f"if (int err = {func}({joined}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, "one_hot")
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    num_classes = _validate_node_contract(node, input_tensor, output_tensor)
    return {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "input_storage_type": target_storage_type(str(input_tensor["dtype"]), target),
        "output_storage_type": target_storage_type("int64", target),
        "num_classes": num_classes,
        "block_size": 256,
    }


def _validate_node_contract(
    node: Mapping[str, Any],
    input_tensor: Mapping[str, Any],
    output_tensor: Mapping[str, Any],
) -> int:
    if node["op"] != "one_hot":
        raise ValueError(f"Unsupported one_hot op: {node['op']}")
    if len(node.get("inputs", [])) != 1:
        raise ValueError("one_hot expects exactly one input")
    if len(node.get("outputs", [])) != 1:
        raise ValueError("one_hot expects exactly one output")
    input_dtype = str(input_tensor["dtype"])
    if input_dtype not in ONE_HOT_INPUT_DTYPES:
        raise NotImplementedError(f"one_hot lowering does not support dtype {input_dtype}")
    if str(output_tensor["dtype"]) != "int64":
        raise ValueError(f"one_hot output dtype must be int64, got {output_tensor['dtype']}")
    num_classes = normalize_one_hot_num_classes(node.get("attrs", {}).get("num_classes"))
    expected_shape = resolve_one_hot_shape(input_tensor["shape"], num_classes)
    if list(output_tensor["shape"]) != list(expected_shape):
        raise ValueError("one_hot output shape does not match input shape plus num_classes")
    expected_shape_spec = [*list(input_tensor.get("shape_spec", input_tensor["shape"])), num_classes]
    if list(output_tensor.get("shape_spec", output_tensor["shape"])) != expected_shape_spec:
        raise ValueError("one_hot output shape_spec must match input shape_spec plus num_classes")
    return num_classes


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    input_tensor = tensor_map[node["inputs"][0]]
    signature = {
        "op": "one_hot",
        "input_dtype": str(input_tensor["dtype"]),
        "num_classes": int(node.get("attrs", {}).get("num_classes", 0)),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"one_hot_{digest}"


ONE_HOT_LOWERING = OpLowering(
    op_name="one_hot",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)
