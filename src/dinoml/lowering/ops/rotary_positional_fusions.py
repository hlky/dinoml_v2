from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import render_op_template, supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.positional import (
    ROTARY_POSITIONAL_FUSION_DTYPES,
    ROTARY_POSITIONAL_FUSION_OPS,
    normalize_get_2d_rotary_pos_embed_attrs,
    normalize_get_2d_rotary_pos_embed_lumina_attrs,
    normalize_get_3d_rotary_pos_embed_allegro_attrs,
    normalize_get_3d_rotary_pos_embed_attrs,
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
    func = _function_name(node, tensor_map)
    output_names = [_c_ident(name) for name in node["outputs"]]
    numel_args = ", ".join(f"runtime_numel_{name}" for name in output_names)
    ptr_args = ", ".join(f"ptr_{name}" for name in output_names)
    args = f"{ptr_args}, {numel_args}"
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
    output_tensors = [tensor_map[str(name)] for name in node["outputs"]]
    normalized, output_dtype = _validate_node_contract(node, output_tensors)
    context = {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
    }
    if op_name == "get_2d_rotary_pos_embed":
        context.update(
            {
                "storage_type": target_storage_type(output_dtype, target),
                "rows": int(normalized["grid_h"]) * int(normalized["grid_w"]),
                "grid_h": int(normalized["grid_h"]),
                "grid_w": int(normalized["grid_w"]),
                "embed_dim": int(normalized["embed_dim"]),
                "half_embed_dim": int(normalized["embed_dim"]) // 2,
                "pair_dim": int(normalized["embed_dim"]) // 4,
                "crop_start_h_literal": _float_literal(float(normalized["crop_start_h"])),
                "crop_start_w_literal": _float_literal(float(normalized["crop_start_w"])),
                "grid_h_step_literal": _linspace_step_literal(
                    float(normalized["crop_start_h"]),
                    float(normalized["crop_stop_h"]),
                    int(normalized["grid_h"]),
                ),
                "grid_w_step_literal": _linspace_step_literal(
                    float(normalized["crop_start_w"]),
                    float(normalized["crop_stop_w"]),
                    int(normalized["grid_w"]),
                ),
                "neg_log_theta_literal": _float_literal(-math.log(float(normalized["theta"]))),
                "block_size": 256,
            }
        )
    elif op_name == "get_2d_rotary_pos_embed_lumina":
        context.update(
            {
                "storage_type": target_storage_type(output_dtype, target),
                "len_h": int(normalized["len_h"]),
                "len_w": int(normalized["len_w"]),
                "output_cols": int(normalized["embed_dim"]) // 2,
                "quarter_dim": int(normalized["embed_dim"]) // 4,
                "neg_log_scaled_theta_literal": _float_literal(
                    -math.log(10000.0 * float(normalized["ntk_factor"]))
                ),
                "inv_linear_factor_literal": _float_literal(1.0 / float(normalized["linear_factor"])),
                "block_size": 256,
            }
        )
    elif op_name == "get_3d_rotary_pos_embed":
        dim_t = int(normalized["embed_dim"]) // 4
        dim_h = (int(normalized["embed_dim"]) // 8) * 3
        dim_w = dim_h
        context.update(
            {
                "storage_type": target_storage_type(output_dtype, target),
                "rows": int(normalized["temporal_size"]) * int(normalized["grid_h"]) * int(normalized["grid_w"]),
                "grid_h": int(normalized["grid_h"]),
                "grid_w": int(normalized["grid_w"]),
                "temporal_size": int(normalized["temporal_size"]),
                "embed_dim": int(normalized["embed_dim"]),
                "dim_t": dim_t,
                "dim_h": dim_h,
                "dim_w": dim_w,
                "pair_dim_t": dim_t // 2,
                "pair_dim_h": dim_h // 2,
                "pair_dim_w": dim_w // 2,
                "grid_is_slice": str(normalized["grid_type"]) == "slice",
                "crop_start_h_literal": _float_literal(float(normalized["crop_start_h"])),
                "crop_start_w_literal": _float_literal(float(normalized["crop_start_w"])),
                "grid_h_step_literal": _linspace_step_literal(
                    float(normalized["crop_start_h"]),
                    float(normalized["crop_stop_h"]),
                    int(normalized["grid_h"]),
                ),
                "grid_w_step_literal": _linspace_step_literal(
                    float(normalized["crop_start_w"]),
                    float(normalized["crop_stop_w"]),
                    int(normalized["grid_w"]),
                ),
                "neg_log_theta_literal": _float_literal(-math.log(float(normalized["theta"]))),
                "block_size": 256,
            }
        )
    elif op_name == "get_3d_rotary_pos_embed_allegro":
        dim_axis = int(normalized["attention_head_dim"]) // 3
        context.update(
            {
                "storage_type": target_storage_type(output_dtype, target),
                "grid_storage_type": target_storage_type("int64", target),
                "num_frames": int(normalized["num_frames"]),
                "grid_h": int(normalized["grid_h"]),
                "grid_w": int(normalized["grid_w"]),
                "dim_axis": dim_axis,
                "pair_dim": dim_axis // 2,
                "inv_interp_t_literal": _float_literal(1.0 / float(normalized["interpolation_scale_t"])),
                "inv_interp_h_literal": _float_literal(1.0 / float(normalized["interpolation_scale_h"])),
                "inv_interp_w_literal": _float_literal(1.0 / float(normalized["interpolation_scale_w"])),
                "neg_log_theta_literal": _float_literal(-math.log(10000.0)),
                "block_size": 256,
            }
        )
    else:
        raise ValueError(f"Unsupported rotary positional fusion op: {op_name}")
    return context


def _validate_node_contract(
    node: Mapping[str, Any],
    output_tensors: list[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], str]:
    op_name = str(node["op"])
    attrs = node.get("attrs", {})
    output_dtype = str(attrs.get("dtype", output_tensors[0]["dtype"]))
    if op_name == "get_2d_rotary_pos_embed":
        if len(output_tensors) != 2:
            raise ValueError("get_2d_rotary_pos_embed expects exactly two outputs")
        normalized = normalize_get_2d_rotary_pos_embed_attrs(
            embed_dim=attrs.get("embed_dim"),
            crop_start_h=attrs.get("crop_start_h"),
            crop_start_w=attrs.get("crop_start_w"),
            crop_stop_h=attrs.get("crop_stop_h"),
            crop_stop_w=attrs.get("crop_stop_w"),
            grid_h=attrs.get("grid_h"),
            grid_w=attrs.get("grid_w"),
            theta=attrs.get("theta", 10000.0),
            use_real=attrs.get("use_real", True),
        )
        expected_shape = [int(normalized["grid_h"]) * int(normalized["grid_w"]), int(normalized["embed_dim"])]
        _validate_float_outputs(op_name, output_tensors, expected_shape, output_dtype)
        return normalized, output_dtype
    if op_name == "get_2d_rotary_pos_embed_lumina":
        if len(output_tensors) != 2:
            raise ValueError("get_2d_rotary_pos_embed_lumina expects exactly two outputs")
        normalized = normalize_get_2d_rotary_pos_embed_lumina_attrs(
            embed_dim=attrs.get("embed_dim"),
            len_h=attrs.get("len_h"),
            len_w=attrs.get("len_w"),
            linear_factor=attrs.get("linear_factor", 1.0),
            ntk_factor=attrs.get("ntk_factor", 1.0),
        )
        expected_shape = [int(normalized["len_h"]), int(normalized["len_w"]), int(normalized["embed_dim"]) // 2]
        _validate_float_outputs(op_name, output_tensors, expected_shape, output_dtype)
        return normalized, output_dtype
    if op_name == "get_3d_rotary_pos_embed":
        if len(output_tensors) != 2:
            raise ValueError("get_3d_rotary_pos_embed expects exactly two outputs")
        normalized = normalize_get_3d_rotary_pos_embed_attrs(
            embed_dim=attrs.get("embed_dim"),
            crop_start_h=attrs.get("crop_start_h"),
            crop_start_w=attrs.get("crop_start_w"),
            crop_stop_h=attrs.get("crop_stop_h"),
            crop_stop_w=attrs.get("crop_stop_w"),
            grid_h=attrs.get("grid_h"),
            grid_w=attrs.get("grid_w"),
            temporal_size=attrs.get("temporal_size"),
            theta=attrs.get("theta", 10000.0),
            use_real=attrs.get("use_real", True),
            grid_type=attrs.get("grid_type", "linspace"),
            max_h=attrs.get("max_h", 0),
            max_w=attrs.get("max_w", 0),
        )
        expected_shape = [
            int(normalized["temporal_size"]) * int(normalized["grid_h"]) * int(normalized["grid_w"]),
            int(normalized["embed_dim"]),
        ]
        _validate_float_outputs(op_name, output_tensors, expected_shape, output_dtype)
        return normalized, output_dtype
    if op_name == "get_3d_rotary_pos_embed_allegro":
        if len(output_tensors) != 9:
            raise ValueError("get_3d_rotary_pos_embed_allegro expects exactly nine outputs")
        normalized = normalize_get_3d_rotary_pos_embed_allegro_attrs(
            height=attrs.get("height"),
            width=attrs.get("width"),
            num_frames=attrs.get("num_frames"),
            vae_scale_factor_spatial=attrs.get("vae_scale_factor_spatial", 8),
            patch_size=attrs.get("patch_size", 2),
            interpolation_scale_h=attrs.get("interpolation_scale_h", 2.0),
            interpolation_scale_t=attrs.get("interpolation_scale_t", 2.2),
            interpolation_scale_w=attrs.get("interpolation_scale_w", 2.0),
            attention_head_dim=attrs.get("attention_head_dim", 96),
        )
        if output_dtype not in ROTARY_POSITIONAL_FUSION_DTYPES:
            raise NotImplementedError(f"{op_name} does not support dtype {output_dtype}")
        dim_axis = int(normalized["attention_head_dim"]) // 3
        grid_shape = [1, int(normalized["num_frames"]) * int(normalized["grid_h"]) * int(normalized["grid_w"])]
        expected_shapes = [
            [int(normalized["num_frames"]), dim_axis],
            [int(normalized["num_frames"]), dim_axis],
            [int(normalized["grid_h"]), dim_axis],
            [int(normalized["grid_h"]), dim_axis],
            [int(normalized["grid_w"]), dim_axis],
            [int(normalized["grid_w"]), dim_axis],
            grid_shape,
            grid_shape,
            grid_shape,
        ]
        expected_dtypes = [output_dtype] * 6 + ["int64", "int64", "int64"]
        for output_tensor, expected_shape, expected_dtype in zip(output_tensors, expected_shapes, expected_dtypes):
            if list(output_tensor["shape"]) != expected_shape:
                raise ValueError(f"{op_name} output shape does not match attrs")
            if str(output_tensor["dtype"]) != expected_dtype:
                raise NotImplementedError(f"{op_name} output dtype does not match attrs")
        return normalized, output_dtype
    raise ValueError(f"Unsupported rotary positional fusion op: {op_name}")


def _validate_float_outputs(
    op_name: str,
    output_tensors: list[Mapping[str, Any]],
    expected_shape: list[int],
    output_dtype: str,
) -> None:
    if output_dtype not in ROTARY_POSITIONAL_FUSION_DTYPES:
        raise NotImplementedError(f"{op_name} does not support dtype {output_dtype}")
    for output_tensor in output_tensors:
        if list(output_tensor["shape"]) != expected_shape:
            raise ValueError(f"{op_name} output shape does not match attrs")
        if str(output_tensor["dtype"]) != output_dtype:
            raise NotImplementedError(f"{op_name} output dtype does not match attrs")


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    signature = {
        "op": str(node["op"]),
        "attrs": dict(node.get("attrs", {})),
        "outputs": [(list(tensor_map[str(name)]["shape"]), str(tensor_map[str(name)]["dtype"])) for name in node["outputs"]],
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"{node['op']}_{digest}"


def _float_literal(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("rotary positional fusion lowering supports only finite attrs")
    literal = f"{value:.9g}"
    if "." not in literal and "e" not in literal and "E" not in literal:
        literal = f"{literal}.0"
    return f"{literal}f"


def _linspace_step_literal(start: float, stop: float, size: int) -> str:
    if size <= 1:
        return "0.0f"
    scaled_stop = stop * float(size - 1) / float(size)
    return _float_literal((scaled_stop - start) / float(size - 1))


ROTARY_POSITIONAL_FUSION_LOWERINGS = {
    op_name: OpLowering(
        op_name=op_name,
        render_generated_kernel=render_generated_kernel,
        render_launch=render_launch,
        source_key=source_key,
        generated_function_name=generated_function_name,
    )
    for op_name in ROTARY_POSITIONAL_FUSION_OPS
}
