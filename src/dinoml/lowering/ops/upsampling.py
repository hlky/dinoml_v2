from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.upsampling import (
    UPSAMPLING_DTYPES,
    normalize_upsampling_attrs,
    resolve_upsampling1d_shape,
    resolve_upsampling2d_shape,
    resolve_upsampling3d_compress_time_shape,
    resolve_upsampling3d_shape,
)


UPSAMPLING_FAMILY_OPS = (
    "upsampling1d",
    "upsampling1d_add",
    "upsampling2d",
    "upsampling2d_add",
    "upsampling3d",
    "upsampling3d_add",
    "upsampling3d_compress_time",
)


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, str(node["op"]))
    context = _context(target, node, tensor_map)
    if not spec.is_gpu:
        return _render_template(_template_name(str(node["op"]), is_gpu=False), context)
    context.update(spec.gpu_template_context())
    return _render_template(_template_name(str(node["op"]), is_gpu=True), context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    op_name = str(node["op"])
    spec = supported_target_spec(target, op_name)
    _validate_node_contract(node, tensor_map)
    x_ident = _c_ident(str(node["inputs"][0]))
    out_ident = _c_ident(str(node["outputs"][0]))
    residual_ptr = "nullptr"
    if _has_residual(op_name):
        residual_ptr = f"ptr_{_c_ident(str(node['inputs'][1]))}"
    extra_args = _shape_args(node, tensor_map)
    args = ", ".join([f"ptr_{x_ident}", residual_ptr, f"ptr_{out_ident}", f"runtime_numel_{out_ident}", *extra_args])
    func = _function_name(node, tensor_map)
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
    op_name = str(node["op"])
    spec = supported_target_spec(target, op_name)
    _validate_node_contract(node, tensor_map)
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    dtype = str(output_tensor["dtype"])
    input_ident = _c_ident(str(node["inputs"][0]))
    output_ident = _c_ident(str(node["outputs"][0]))
    channel_count = int(output_tensor["shape"][-1])
    mode = "nearest" if op_name == "upsampling3d_compress_time" else str(node["attrs"]["mode"])
    context = {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
        "storage_type": target_storage_type(dtype, target),
        "has_residual": _has_residual(op_name),
        "input_shape_vars": [f"shape_{input_ident}_{axis}" for axis in range(len(input_tensor["shape"]))],
        "output_shape_vars": [f"shape_{output_ident}_{axis}" for axis in range(len(output_tensor["shape"]))],
        "block_size": 256,
        "channel_count": channel_count,
    }
    context.update(_gpu_vector_plan(dtype, channel_count, mode))
    if op_name != "upsampling3d_compress_time":
        scale_factor = float(node["attrs"]["scale_factor"])
        context["scale_factor"] = scale_factor
        context["inverse_scale_factor_literal"] = _float_literal(1.0 / scale_factor)
    if op_name.startswith("upsampling1d"):
        context["mode"] = mode
        context["align_corners"] = bool(node["attrs"].get("align_corners", False))
    elif op_name.startswith("upsampling2d"):
        context["mode"] = mode
        context["align_corners"] = bool(node["attrs"].get("align_corners", False))
    elif op_name.startswith("upsampling3d") and op_name != "upsampling3d_compress_time":
        context["mode"] = mode
        context["align_corners"] = bool(node["attrs"].get("align_corners", False))
    return context


def _validate_node_contract(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> None:
    op_name = str(node["op"])
    if op_name not in UPSAMPLING_FAMILY_OPS:
        raise ValueError(f"Unsupported upsampling op: {op_name}")
    expected_inputs = 2 if _has_residual(op_name) else 1
    if len(node.get("inputs", ())) != expected_inputs:
        raise ValueError(f"{op_name} expects {expected_inputs} tensor inputs")
    if len(node.get("outputs", ())) != 1:
        raise ValueError(f"{op_name} expects exactly one output")
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    dtype = str(output_tensor["dtype"])
    if dtype not in UPSAMPLING_DTYPES:
        raise NotImplementedError(f"{op_name} lowering does not support dtype {dtype}")
    if str(input_tensor["dtype"]) != dtype:
        raise ValueError(f"{op_name} lowering requires matching input/output dtypes")
    if op_name == "upsampling3d_compress_time":
        input_shape_spec = input_tensor.get("shape_spec", input_tensor["shape"])
        if not isinstance(input_shape_spec[1], int):
            raise ValueError("upsampling3d_compress_time lowering requires a static frame dimension")
    output_shape = list(output_tensor["shape"])
    input_shape = input_tensor["shape"]
    if op_name == "upsampling3d_compress_time":
        expected_shape = resolve_upsampling3d_compress_time_shape(input_shape)
    else:
        attrs = normalize_upsampling_attrs(
            op_name,
            scale_factor=node.get("attrs", {}).get("scale_factor"),
            mode=node.get("attrs", {}).get("mode"),
            align_corners=node.get("attrs", {}).get("align_corners", False),
        )
        if op_name.startswith("upsampling1d"):
            expected_shape = resolve_upsampling1d_shape(input_shape, float(attrs["scale_factor"]))
        elif op_name.startswith("upsampling2d"):
            expected_shape = resolve_upsampling2d_shape(input_shape, float(attrs["scale_factor"]))
        else:
            expected_shape = resolve_upsampling3d_shape(input_shape, float(attrs["scale_factor"]))
    if [int(dim) for dim in expected_shape] != [int(dim) for dim in output_shape]:
        raise ValueError(f"{op_name} output shape does not match attrs")
    if _has_residual(op_name):
        residual_tensor = tensor_map[node["inputs"][1]]
        if str(residual_tensor["dtype"]) != dtype:
            raise ValueError(f"{op_name} lowering requires matching residual dtype {dtype}")
        if [int(dim) for dim in residual_tensor["shape"]] != [int(dim) for dim in output_shape]:
            raise ValueError(f"{op_name} residual shape must match the output shape {output_shape}")


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    op_name = str(node["op"])
    output_tensor = tensor_map[node["outputs"][0]]
    signature = {
        "op": op_name,
        "input_shapes": [list(tensor_map[input_name]["shape"]) for input_name in node["inputs"]],
        "output_shape": list(output_tensor["shape"]),
        "dtype": str(output_tensor["dtype"]),
        "attrs": dict(node.get("attrs", {})),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"{op_name}_{digest}"


def _shape_args(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> list[str]:
    input_tensor = tensor_map[node["inputs"][0]]
    output_tensor = tensor_map[node["outputs"][0]]
    input_ident = _c_ident(str(node["inputs"][0]))
    output_ident = _c_ident(str(node["outputs"][0]))
    args = [f"shape_{input_ident}_{axis}" for axis in range(len(input_tensor["shape"]))]
    args.extend(f"shape_{output_ident}_{axis}" for axis in range(len(output_tensor["shape"])))
    return args


def _template_name(op_name: str, *, is_gpu: bool) -> str:
    suffix = "gpu.j2" if is_gpu else "cpu.cpp.j2"
    if op_name.startswith("upsampling1d"):
        return f"upsampling1d_{suffix}"
    if op_name.startswith("upsampling2d"):
        return f"upsampling2d_{suffix}"
    if op_name == "upsampling3d_compress_time":
        return f"upsampling3d_compress_time_{suffix}"
    return f"upsampling3d_{suffix}"


def _has_residual(op_name: str) -> bool:
    return op_name.endswith("_add")


def _gpu_vector_plan(dtype: str, channels: int, mode: str) -> dict[str, Any]:
    if mode in {"linear", "bilinear", "trilinear"}:
        use_packed_interpolation = (channels % 2) == 0
        return {
            "use_packed_interpolation": use_packed_interpolation,
            "interpolation_vec_type": _pack2_vec_type(dtype),
            "interpolation_alignment": 2 if use_packed_interpolation else 1,
            "nearest_vec_type": _nearest_vec_type(dtype, channels),
            "nearest_alignment": _nearest_alignment(dtype, channels),
        }
    return {
        "use_packed_interpolation": False,
        "interpolation_vec_type": _pack2_vec_type(dtype),
        "interpolation_alignment": 1,
        "nearest_vec_type": _nearest_vec_type(dtype, channels),
        "nearest_alignment": _nearest_alignment(dtype, channels),
    }


def _pack2_vec_type(dtype: str) -> str:
    if dtype == "float32":
        return "float2"
    if dtype == "float16":
        return "half2"
    if dtype == "bfloat16":
        return "dinoml::bfloat162"
    raise NotImplementedError(f"Unsupported upsampling dtype for packed interpolation: {dtype}")


def _nearest_vec_type(dtype: str, channels: int) -> str:
    if dtype == "float32":
        return "float2" if (channels % 2) == 0 else "float"
    if dtype == "float16":
        if (channels % 8) == 0:
            return "float4"
        if (channels % 2) == 0:
            return "half2"
        return "half"
    if dtype == "bfloat16":
        if (channels % 8) == 0:
            return "float4"
        if (channels % 2) == 0:
            return "dinoml::bfloat162"
        return "dinoml::bfloat16"
    raise NotImplementedError(f"Unsupported upsampling dtype for nearest vectorization: {dtype}")


def _nearest_alignment(dtype: str, channels: int) -> int:
    if dtype == "float32":
        return 2 if (channels % 2) == 0 else 1
    if dtype in {"float16", "bfloat16"}:
        if (channels % 8) == 0:
            return 8
        if (channels % 2) == 0:
            return 2
        return 1
    raise NotImplementedError(f"Unsupported upsampling dtype for nearest vectorization: {dtype}")


def _float_literal(value: float) -> str:
    literal = format(float(value), ".9g")
    if "e" not in literal and "." not in literal:
        literal += ".0"
    return f"{literal}f"


def _render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


UPSAMPLING_FAMILY_LOWERINGS = {
    op_name: OpLowering(
        op_name=op_name,
        render_generated_kernel=render_generated_kernel,
        render_launch=render_launch,
        source_key=source_key,
        generated_function_name=generated_function_name,
    )
    for op_name in UPSAMPLING_FAMILY_OPS
}
