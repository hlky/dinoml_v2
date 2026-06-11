from __future__ import annotations

import hashlib
from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import render_op_template, supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.tensor_filters import (
    TENSOR_FILTER_HELPER_DTYPES,
    TENSOR_FILTER_HELPER_OPS,
    normalize_fir_upsample2d_attrs,
    normalize_tensor_filter_channels,
    resolve_fir_downsample2d_shape,
    resolve_fir_filter_pad2_shape,
    resolve_fir_upsample2d_shape,
    resolve_tensor_filter_weight_shape,
)


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, str(node["op"]))
    context = _context(target, node, tensor_map)
    if spec.is_gpu:
        context.update(spec.gpu_template_context())
    template_name = f"{node['op']}_{'gpu' if spec.is_gpu else 'cpu.cpp'}.j2"
    return render_op_template(template_name, context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, str(node["op"]))
    op_name = str(node["op"])
    func = _function_name(node, tensor_map)
    if op_name in {"kdownsample2d_weight", "kupsample2d_weight"}:
        out_ident = _c_ident(str(node["outputs"][0]))
        args = f"ptr_{out_ident}, runtime_numel_{out_ident}"
    else:
        x_ident = _c_ident(str(node["inputs"][0]))
        out_ident = _c_ident(str(node["outputs"][0]))
        args = ", ".join(
            [
                f"ptr_{x_ident}",
                f"ptr_{out_ident}",
                f"runtime_numel_{out_ident}",
                *[f"shape_{x_ident}_{axis}" for axis in range(len(tensor_map[str(node['inputs'][0])]['shape']))],
                *[f"shape_{out_ident}_{axis}" for axis in range(len(tensor_map[str(node['outputs'][0])]['shape']))],
            ]
        )
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, str(node["op"]))
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    validated = _validate_node_contract(node, tensor_map)
    context = {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": target_storage_type(str(validated["output_dtype"]), target),
        "block_size": 256,
    }
    context.update(validated.get("template_attrs", {}))
    return context


def _validate_node_contract(
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    op_name = str(node["op"])
    if op_name not in TENSOR_FILTER_HELPER_OPS:
        raise ValueError(f"Unsupported tensor filter helper op: {op_name}")
    if op_name in {"kdownsample2d_weight", "kupsample2d_weight"}:
        if node.get("inputs"):
            raise ValueError(f"{op_name} expects zero inputs")
        if len(node.get("outputs", ())) != 1:
            raise ValueError(f"{op_name} expects one output")
        output_tensor = tensor_map[str(node["outputs"][0])]
        output_dtype = str(output_tensor["dtype"])
        if output_dtype not in TENSOR_FILTER_HELPER_DTYPES:
            raise NotImplementedError(f"{op_name} lowering does not support dtype {output_dtype}")
        channels = normalize_tensor_filter_channels(node.get("attrs", {}).get("channels"), f"{op_name} channels")
        expected_shape = resolve_tensor_filter_weight_shape(channels)
        if list(output_tensor["shape"]) != expected_shape:
            raise ValueError(f"{op_name} output shape does not match channels attr")
        if str(node.get("attrs", {}).get("dtype", output_dtype)) != output_dtype:
            raise ValueError(f"{op_name} output dtype does not match dtype attr")
        return {
            "output_dtype": output_dtype,
            "template_attrs": {"channels": channels},
        }

    if len(node.get("inputs", ())) != 1 or len(node.get("outputs", ())) != 1:
        raise ValueError(f"{op_name} expects one input and one output")
    input_tensor = tensor_map[str(node["inputs"][0])]
    output_tensor = tensor_map[str(node["outputs"][0])]
    input_dtype = str(input_tensor["dtype"])
    output_dtype = str(output_tensor["dtype"])
    if input_dtype not in TENSOR_FILTER_HELPER_DTYPES or output_dtype not in TENSOR_FILTER_HELPER_DTYPES:
        raise NotImplementedError(f"{op_name} lowering does not support dtype {output_dtype}")
    if input_dtype != output_dtype:
        raise ValueError(f"{op_name} lowering requires matching input/output dtypes")

    if op_name == "fir_downsample2d":
        expected_shape = resolve_fir_downsample2d_shape(input_tensor["shape"])
        template_attrs: dict[str, Any] = {}
    elif op_name == "fir_filter_pad2":
        expected_shape = resolve_fir_filter_pad2_shape(input_tensor["shape"])
        template_attrs = {}
    else:
        normalized = normalize_fir_upsample2d_attrs(
            up=node.get("attrs", {}).get("up", 2),
            pad0=node.get("attrs", {}).get("pad0", 2),
            pad1=node.get("attrs", {}).get("pad1", 1),
        )
        expected_shape = resolve_fir_upsample2d_shape(
            input_tensor["shape"],
            up=int(normalized["up"]),
            pad0=int(normalized["pad0"]),
            pad1=int(normalized["pad1"]),
        )
        template_attrs = {
            "up": int(normalized["up"]),
            "pad0": int(normalized["pad0"]),
            "pad1": int(normalized["pad1"]),
        }
    if list(output_tensor["shape"]) != expected_shape:
        raise ValueError(f"{op_name} output shape does not match inputs and attrs")
    return {
        "output_dtype": output_dtype,
        "template_attrs": template_attrs,
    }


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    signature = {
        "op": str(node["op"]),
        "attrs": dict(node.get("attrs", {})),
        "inputs": [(list(tensor_map[str(name)]["shape"]), str(tensor_map[str(name)]["dtype"])) for name in node["inputs"]],
        "outputs": [(list(tensor_map[str(name)]["shape"]), str(tensor_map[str(name)]["dtype"])) for name in node["outputs"]],
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"{node['op']}_{digest}"


TENSOR_FILTER_HELPER_LOWERINGS = {
    op_name: OpLowering(
        op_name=op_name,
        render_generated_kernel=render_generated_kernel,
        render_launch=render_launch,
        source_key=source_key,
        generated_function_name=generated_function_name,
    )
    for op_name in TENSOR_FILTER_HELPER_OPS
}
