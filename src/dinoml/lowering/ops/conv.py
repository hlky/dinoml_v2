from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.kernels.providers.cutlass.conv import CUTLASS_TRANSPOSED_CONV_OPS, cutlass_conv_wrapper_stages
from dinoml.kernels.providers.ck.conv import CK_CONV_OPS, CK_TRANSPOSED_CONV_OPS
from dinoml.lowering.cpp_types import cpu_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.ops.definitions import get_op_def
from dinoml.ops.conv import (
    CONV2D_BIAS_DTYPES,
    CONV1D_BIAS_FAMILY_OPS,
    CONV2D_BIAS_FAMILY_OPS,
    CONV3D_FAMILY_OPS,
    TRANSPOSED_CONV1D_OPS,
    TRANSPOSED_CONV2D_FAMILY_OPS,
    normalize_transposed_conv1d_attrs,
    normalize_conv1d_bias_attrs,
    normalize_conv2d_bias_attrs,
    normalize_conv3d_attrs,
    resolve_conv1d_bias_add_relu_shape,
    resolve_conv1d_bias_add_shape,
    resolve_conv1d_bias_shape,
    resolve_conv1d_bias_relu_shape,
    resolve_conv3d_bias_shape,
    resolve_depthwise_conv3d_shape,
    resolve_transposed_conv1d_shape,
    normalize_transposed_conv2d_attrs,
    resolve_conv2d_bias_add_relu_shape,
    resolve_conv2d_bias_add_shape,
    resolve_conv2d_bias_shape,
    resolve_conv2d_bias_relu_shape,
    resolve_transposed_conv2d_bias_add_relu_shape,
    resolve_transposed_conv2d_bias_add_shape,
    resolve_transposed_conv2d_bias_relu_shape,
    resolve_transposed_conv2d_bias_shape,
    resolve_transposed_conv2d_shape,
)


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    if target == "cpu":
        if str(node["op"]) in {*CONV1D_BIAS_FAMILY_OPS, *TRANSPOSED_CONV1D_OPS}:
            template_name = "conv1d_cpu.cpp.j2"
        elif str(node["op"]) in CONV3D_FAMILY_OPS:
            template_name = "conv3d_cpu.cpp.j2"
        else:
            template_name = "conv_cpu.cpp.j2"
        return _render_cpu_template(template_name, _cpu_context(node, tensor_map))
    if target in {"cuda", "rocm"}:
        return None
    raise ValueError(f"Unsupported Conv lowering target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    op_name = str(node["op"])
    if target == "cpu":
        func = _cpu_function_name(node, tensor_map)
        x_ident = _c_ident(str(node["inputs"][0]))
        weight_ident = _c_ident(str(node["inputs"][1]))
        bias_ident = None if not _cpu_op_has_bias(op_name) else _c_ident(str(node["inputs"][2]))
        call_args = [
            f"ptr_{x_ident}",
            f"runtime_numel_{x_ident}",
            f"ptr_{weight_ident}",
            f"runtime_numel_{weight_ident}",
        ]
        if bias_ident is not None:
            call_args.extend([f"ptr_{bias_ident}", f"runtime_numel_{bias_ident}"])
        if _conv_family_has_residual(op_name):
            residual_input_index = 3 if _cpu_op_has_bias(op_name) else 2
            residual_ident = _c_ident(str(node["inputs"][residual_input_index]))
            call_args.extend([f"ptr_{residual_ident}", f"runtime_numel_{residual_ident}"])
        out_ident = _c_ident(str(node["outputs"][0]))
        call_args.extend([f"ptr_{out_ident}", f"runtime_numel_{out_ident}"])
        return (
            "if (int err = " f"{func}({', '.join(call_args)})) return err;"
        )
    if target == "rocm":
        return _render_rocm_launch(node, tensor_map, kernel_manifest)
    if target != "cuda":
        raise ValueError(f"Unsupported Conv lowering target: {target}")
    if op_name in TRANSPOSED_CONV2D_FAMILY_OPS and op_name not in CUTLASS_TRANSPOSED_CONV_OPS:
        raise NotImplementedError(
            f"{op_name} CUDA Conv lowering is unsupported; only transposed_conv2d has native CUTLASS support"
        )
    if op_name not in {*CONV1D_BIAS_FAMILY_OPS, *CONV2D_BIAS_FAMILY_OPS, *CONV3D_FAMILY_OPS, *CUTLASS_TRANSPOSED_CONV_OPS}:
        raise NotImplementedError(f"{op_name} CUDA Conv lowering is not implemented")
    item = _manifest_kernel_item(kernel_manifest, op_name, node_id=str(node.get("id", "")))
    if item is None:
        raise ValueError(f"{op_name} CUDA lowering requires a CUTLASS Conv manifest entry")
    stages = cutlass_conv_wrapper_stages({"required_kernels": [item]})
    stage_names = [str(stage.get("stage_name")) for stage in stages]
    expected_stage_names = ["activation_pack"]
    if "weight_pack" in stage_names:
        expected_stage_names.append("weight_pack")
    if _conv_family_has_residual(op_name):
        expected_stage_names.append("residual_pack")
    expected_stage_names.extend(["provider_launch", "output_unpack"])
    if stage_names != expected_stage_names:
        raise ValueError(f"{op_name} CUTLASS Conv wrapper stages are malformed")

    input_names = [str(name) for name in node.get("inputs", ())]
    expected_inputs = 2 if op_name in CUTLASS_TRANSPOSED_CONV_OPS else (4 if _conv_family_has_residual(op_name) else 3)
    if len(input_names) != expected_inputs:
        raise ValueError(f"{op_name} CUDA lowering expects {expected_inputs} inputs")
    output_names = [str(name) for name in node.get("outputs", ())]
    if len(output_names) != 1:
        raise ValueError(f"{op_name} CUDA lowering expects one output")
    roles = {
        "activation": _c_ident(input_names[0]),
        "weight": _c_ident(input_names[1]),
        "output": _c_ident(output_names[0]),
    }
    if op_name not in CUTLASS_TRANSPOSED_CONV_OPS:
        roles["bias"] = _c_ident(input_names[2])
    if _conv_family_has_residual(op_name):
        roles["residual"] = _c_ident(input_names[3])
    _validate_runtime_shape_contract(op_name, tensor_map, input_names, output_names[0], item)

    lines = [
        f'// CUTLASS Conv wrapper lowering for {op_name}: public NCHW tensors enter a provider-internal NHWC Conv plan.',
    ]
    for stage in stages:
        lines.extend(_render_runtime_stage(stage, roles=roles))
    return "\n".join(lines)


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    if target == "cpu":
        return f"{target}:{generated_function_name(target, node, tensor_map)}"
    if target in {"cuda", "rocm"}:
        return None
    raise ValueError(f"Unsupported Conv lowering target: {target}")


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    if target == "cpu":
        return _cpu_function_name(node, tensor_map)
    if target in {"cuda", "rocm"}:
        return None
    raise ValueError(f"Unsupported Conv lowering target: {target}")


def _render_rocm_launch(
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None,
) -> str:
    op_name = str(node["op"])
    is_transposed = op_name in TRANSPOSED_CONV2D_FAMILY_OPS
    has_bias = _cpu_op_has_bias(op_name)
    if op_name not in {*CK_CONV_OPS, *CK_TRANSPOSED_CONV_OPS}:
        raise NotImplementedError(f"{op_name} ROCm CK Conv lowering is not implemented")
    input_names = [str(name) for name in node.get("inputs", ())]
    expected_inputs = 2 + int(has_bias) + int(_conv_family_has_residual(op_name))
    if len(input_names) != expected_inputs:
        raise ValueError(f"{op_name} ROCm lowering expects {expected_inputs} inputs")
    output_names = [str(name) for name in node.get("outputs", ())]
    if len(output_names) != 1:
        raise ValueError(f"{op_name} ROCm lowering expects one output")
    x_name, weight_name = input_names[:2]
    bias_name = input_names[2] if has_bias else None
    residual_name = input_names[3] if has_bias and _conv_family_has_residual(op_name) else (input_names[2] if _conv_family_has_residual(op_name) else None)
    output_name = output_names[0]
    x = tensor_map[x_name]
    weight = tensor_map[weight_name]
    bias = None if bias_name is None else tensor_map[bias_name]
    residual = None if residual_name is None else tensor_map[residual_name]
    output = tensor_map[output_name]
    attrs = _validate_cpu_contract(node, x, weight, bias, output, residual)
    dtype = str(output["dtype"])
    item = _manifest_kernel_item(kernel_manifest, op_name, node_id=str(node.get("id", "")), kernel_library="ck_conv")
    if item is None:
        item = _manifest_kernel_item(
            kernel_manifest,
            op_name,
            node_id=str(node.get("id", "")),
            kernel_library="rocm_tile_conv",
        )
    symbol = (
        str(item["kernel_symbol"])
        if item is not None
        else get_op_def(op_name).backend_kernels["rocm"].resolve(dtype).symbol
    )
    kernel_library = str(item.get("kernel_library", "ck_conv")) if item is not None else "ck_conv"
    runtime_plan = dict(item.get("ck_conv_runtime_plan", {})) if isinstance(item, Mapping) else {}
    weight_pack_mode = str(runtime_plan.get("weight_pack_mode", ""))
    dispatches = _rocm_dispatch_selections(item, str(node.get("id", ""))) if kernel_library == "ck_conv" else []
    x_ident = _c_ident(x_name)
    weight_ident = _c_ident(weight_name)
    bias_ident = None if bias_name is None else _c_ident(bias_name)
    residual_ident = None if residual_name is None else _c_ident(residual_name)
    out_ident = _c_ident(output_name)
    if op_name in {*CONV1D_BIAS_FAMILY_OPS, *TRANSPOSED_CONV1D_OPS}:
        stride_w = attrs["stride"][0]
        pad_w = attrs["padding"][0]
        output_pad_w = 0 if attrs["output_padding"] is None else attrs["output_padding"][0]
        dilation_w = attrs["dilation"][0]
        weight_is_prepacked = weight_pack_mode == "constants_bin_prepacked_kxc"
        if attrs["transposed"]:
            expected_out_w = (
                f"((shape_{x_ident}_2 - 1) * {stride_w} - {2 * pad_w} + {dilation_w} * (shape_{weight_ident}_2 - 1) + {output_pad_w} + 1)"
            )
        else:
            expected_out_w = (
                f"((shape_{x_ident}_2 + {2 * pad_w} - {dilation_w} * (shape_{weight_ident}_2 - 1) - 1) / {stride_w} + 1)"
            )
        input_channel_check = (
            f'if (shape_{weight_ident}_0 != shape_{x_ident}_1) return dinoml::module::fail("{op_name} input channel mismatch");'
            if attrs["transposed"]
            else f'if (shape_{weight_ident}_1 != shape_{x_ident}_1) return dinoml::module::fail("{op_name} input channel mismatch");'
        )
        out_channel_dim = 1 if attrs["transposed"] else 0
        lines = [
            input_channel_check,
        ]
        if bias_ident is not None:
            lines.append(
                f'if (shape_{bias_ident}_0 != shape_{weight_ident}_{out_channel_dim}) return dinoml::module::fail("{op_name} bias shape mismatch");'
            )
        lines.append(
            (
                f"if (shape_{out_ident}_0 != shape_{x_ident}_0 || "
                f"shape_{out_ident}_1 != shape_{weight_ident}_{out_channel_dim} || "
                f"shape_{out_ident}_2 != {expected_out_w}) "
                f'return dinoml::module::fail("{op_name} output shape mismatch");'
            )
        )
        if residual_ident is not None:
            lines.append(
                (
                    f"if (shape_{residual_ident}_0 != shape_{out_ident}_0 || "
                    f"shape_{residual_ident}_1 != shape_{out_ident}_1 || "
                    f"shape_{residual_ident}_2 != shape_{out_ident}_2) "
                    f'return dinoml::module::fail("{op_name} residual shape mismatch");'
                )
            )
        default_launch = (
            _rocm_launch_lines_transposed_1d(
                op_name=op_name,
                symbol=symbol,
                failure_label="ROCm Tile" if kernel_library == "rocm_tile_conv" else "CK Conv",
                x_ident=x_ident,
                weight_ident=weight_ident,
                out_ident=out_ident,
                stride_w=stride_w,
                pad_w=pad_w,
                output_pad_w=output_pad_w,
                dilation_w=dilation_w,
            )
            if attrs["transposed"]
            else _rocm_launch_lines_1d(
                op_name=op_name,
                symbol=symbol,
                failure_label="ROCm Tile" if kernel_library == "rocm_tile_conv" else "CK Conv",
                x_ident=x_ident,
                weight_ident=weight_ident,
                bias_ident=bias_ident,
                residual_ident=residual_ident,
                out_ident=out_ident,
                stride_w=stride_w,
                pad_w=pad_w,
                dilation_w=dilation_w,
                weight_is_prepacked=weight_is_prepacked,
            )
        )
        if dispatches:
            lines.extend(
                _rocm_dispatch_lines(
                    op_name=op_name,
                    item=item,
                    dispatches=dispatches,
                    default_launch=default_launch,
                    x_ident=x_ident,
                    weight_ident=weight_ident,
                    bias_ident=bias_ident,
                    residual_ident=residual_ident,
                    out_ident=out_ident,
                    stride_h=1,
                    stride_w=stride_w,
                    pad_h=0,
                    pad_w=pad_w,
                    output_pad_h=0,
                    output_pad_w=output_pad_w,
                    dilation_h=1,
                    dilation_w=dilation_w,
                    transposed=attrs["transposed"],
                    weight_is_prepacked=weight_is_prepacked,
                )
            )
        else:
            lines.extend(default_launch)
        return "\n".join(lines)
    if op_name == "conv3d_bias":
        stride_d, stride_h, stride_w = attrs["stride"]
        pad_d, pad_h, pad_w = attrs["padding"]
        dilation_d, dilation_h, dilation_w = attrs["dilation"]
        groups = int(attrs["groups"])
        expected_out_d = (
            f"((shape_{x_ident}_2 + {2 * pad_d} - {dilation_d} * (shape_{weight_ident}_2 - 1) - 1) / {stride_d} + 1)"
        )
        expected_out_h = (
            f"((shape_{x_ident}_3 + {2 * pad_h} - {dilation_h} * (shape_{weight_ident}_3 - 1) - 1) / {stride_h} + 1)"
        )
        expected_out_w = (
            f"((shape_{x_ident}_4 + {2 * pad_w} - {dilation_w} * (shape_{weight_ident}_4 - 1) - 1) / {stride_w} + 1)"
        )
        lines = [
            f'if ((shape_{x_ident}_1 % {groups}) != 0) return dinoml::module::fail("{op_name} input channels/groups mismatch");',
            f'if (shape_{weight_ident}_1 != (shape_{x_ident}_1 / {groups})) return dinoml::module::fail("{op_name} input channel mismatch");',
            f'if ((shape_{weight_ident}_0 % {groups}) != 0) return dinoml::module::fail("{op_name} output channels/groups mismatch");',
        ]
        if bias_ident is not None:
            lines.append(
                f'if (shape_{bias_ident}_0 != shape_{weight_ident}_0) return dinoml::module::fail("{op_name} bias shape mismatch");'
            )
        lines.append(
            (
                f"if (shape_{out_ident}_0 != shape_{x_ident}_0 || "
                f"shape_{out_ident}_1 != shape_{weight_ident}_0 || "
                f"shape_{out_ident}_2 != {expected_out_d} || "
                f"shape_{out_ident}_3 != {expected_out_h} || "
                f"shape_{out_ident}_4 != {expected_out_w}) "
                f'return dinoml::module::fail("{op_name} output shape mismatch");'
            )
        )
        lines.extend(
            _rocm_launch_lines_3d(
                op_name=op_name,
                symbol=symbol,
                failure_label="CK Conv",
                x_ident=x_ident,
                weight_ident=weight_ident,
                bias_ident=bias_ident,
                out_ident=out_ident,
                stride_d=stride_d,
                stride_h=stride_h,
                stride_w=stride_w,
                pad_d=pad_d,
                pad_h=pad_h,
                pad_w=pad_w,
                dilation_d=dilation_d,
                dilation_h=dilation_h,
                dilation_w=dilation_w,
                weight_is_prepacked=weight_pack_mode == "constants_bin_prepacked_kzyxc",
                groups=groups,
            )
        )
        return "\n".join(lines)
    stride_h, stride_w = attrs["stride"]
    pad_h, pad_w = attrs["padding"]
    output_pad_h = 0 if attrs["output_padding"] is None else attrs["output_padding"][0]
    output_pad_w = 0 if attrs["output_padding"] is None else attrs["output_padding"][1]
    dilation_h, dilation_w = attrs["dilation"]
    if is_transposed:
        expected_out_h = (
            f"((shape_{x_ident}_2 - 1) * {stride_h} - {2 * pad_h} + {dilation_h} * (shape_{weight_ident}_2 - 1) + {output_pad_h} + 1)"
        )
        expected_out_w = (
            f"((shape_{x_ident}_3 - 1) * {stride_w} - {2 * pad_w} + {dilation_w} * (shape_{weight_ident}_3 - 1) + {output_pad_w} + 1)"
        )
        lines = [
            f'if (shape_{weight_ident}_0 != shape_{x_ident}_1) return dinoml::module::fail("{op_name} input channel mismatch");',
        ]
        if bias_ident is not None:
            lines.append(
                f'if (shape_{bias_ident}_0 != shape_{weight_ident}_1) return dinoml::module::fail("{op_name} bias shape mismatch");'
            )
        lines.append(
            (
                f"if (shape_{out_ident}_0 != shape_{x_ident}_0 || "
                f"shape_{out_ident}_1 != shape_{weight_ident}_1 || "
                f"shape_{out_ident}_2 != {expected_out_h} || "
                f"shape_{out_ident}_3 != {expected_out_w}) "
                f'return dinoml::module::fail("{op_name} output shape mismatch");'
            )
        )
    else:
        expected_out_h = (
            f"((shape_{x_ident}_2 + {2 * pad_h} - {dilation_h} * (shape_{weight_ident}_2 - 1) - 1) / {stride_h} + 1)"
        )
        expected_out_w = (
            f"((shape_{x_ident}_3 + {2 * pad_w} - {dilation_w} * (shape_{weight_ident}_3 - 1) - 1) / {stride_w} + 1)"
        )
        lines = [
            f'if (shape_{weight_ident}_1 != shape_{x_ident}_1) return dinoml::module::fail("{op_name} input channel mismatch");',
        ]
        if bias_ident is not None:
            lines.append(
                f'if (shape_{bias_ident}_0 != shape_{weight_ident}_0) return dinoml::module::fail("{op_name} bias shape mismatch");'
            )
        lines.append(
            (
                f"if (shape_{out_ident}_0 != shape_{x_ident}_0 || "
                f"shape_{out_ident}_1 != shape_{weight_ident}_0 || "
                f"shape_{out_ident}_2 != {expected_out_h} || "
                f"shape_{out_ident}_3 != {expected_out_w}) "
                f'return dinoml::module::fail("{op_name} output shape mismatch");'
            )
        )
    if residual_ident is not None:
        lines.append(
            (
                f"if (shape_{residual_ident}_0 != shape_{out_ident}_0 || "
                f"shape_{residual_ident}_1 != shape_{out_ident}_1 || "
                f"shape_{residual_ident}_2 != shape_{out_ident}_2 || "
                f"shape_{residual_ident}_3 != shape_{out_ident}_3) "
                f'return dinoml::module::fail("{op_name} residual shape mismatch");'
            )
        )
        residual_arg = f"ptr_{residual_ident}, "
    weight_is_prepacked = kernel_library == "ck_conv" and weight_pack_mode == "constants_bin_prepacked_kyxc"
    default_launch = [
        *(
            _rocm_launch_lines_2d_ck(
                op_name=op_name,
                symbol=symbol,
                failure_label="CK Conv",
                x_ident=x_ident,
                weight_ident=weight_ident,
                bias_ident=bias_ident,
                residual_ident=residual_ident,
                out_ident=out_ident,
                stride_h=stride_h,
                stride_w=stride_w,
                pad_h=pad_h,
                pad_w=pad_w,
                dilation_h=dilation_h,
                dilation_w=dilation_w,
                weight_is_prepacked=weight_is_prepacked,
            )
            if kernel_library == "ck_conv" and not is_transposed
            else _rocm_launch_lines(
                op_name=op_name,
                symbol=symbol,
                failure_label="ROCm Tile" if kernel_library == "rocm_tile_conv" else "CK Conv",
                x_ident=x_ident,
                weight_ident=weight_ident,
                bias_ident=bias_ident,
                residual_ident=residual_ident,
                out_ident=out_ident,
                stride_h=stride_h,
                stride_w=stride_w,
                pad_h=pad_h,
                pad_w=pad_w,
                output_pad_h=output_pad_h,
                output_pad_w=output_pad_w,
                dilation_h=dilation_h,
                dilation_w=dilation_w,
                transposed=is_transposed,
            )
        )
    ]
    if dispatches:
        lines.extend(
            _rocm_dispatch_lines(
                op_name=op_name,
                item=item,
                dispatches=dispatches,
                default_launch=default_launch,
                x_ident=x_ident,
                weight_ident=weight_ident,
                bias_ident=bias_ident,
                residual_ident=residual_ident,
                out_ident=out_ident,
                stride_h=stride_h,
                stride_w=stride_w,
                pad_h=pad_h,
                pad_w=pad_w,
                output_pad_h=output_pad_h,
                output_pad_w=output_pad_w,
                dilation_h=dilation_h,
                dilation_w=dilation_w,
                transposed=is_transposed,
            )
        )
    else:
        lines.extend(default_launch)
    return "\n".join(lines)


def _cpu_context(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    x_name = str(node["inputs"][0])
    weight_name = str(node["inputs"][1])
    bias_name = None if not _cpu_op_has_bias(str(node["op"])) else str(node["inputs"][2])
    out_name = str(node["outputs"][0])
    x = tensor_map[x_name]
    weight = tensor_map[weight_name]
    bias = None if bias_name is None else tensor_map[bias_name]
    residual = None
    if _conv_family_has_residual(str(node["op"])):
        residual_index = 3 if bias_name is not None else 2
        residual = tensor_map[str(node["inputs"][residual_index])]
    output = tensor_map[out_name]
    attrs = _validate_cpu_contract(node, x, weight, bias, output, residual)
    if attrs["rank"] == 3:
        batch, in_channels, in_width = [int(dim) for dim in x["shape"]]
        if attrs["transposed"]:
            weight_in_channels, out_channels, kernel_w = [int(dim) for dim in weight["shape"]]
        else:
            out_channels, weight_in_channels, kernel_w = [int(dim) for dim in weight["shape"]]
        out_batch, out_channels_out, out_width = [int(dim) for dim in output["shape"]]
        return {
            "func": _cpu_function_name(node, tensor_map),
            "storage_type": cpu_storage_type(str(output["dtype"])),
            "input_numel": batch * in_channels * in_width,
            "weight_numel": weight_in_channels * out_channels * kernel_w,
            "bias_numel": None if bias is None else out_channels,
            "residual_numel": None if residual is None else out_batch * out_channels_out * out_width,
            "output_numel": out_batch * out_channels_out * out_width,
            "batch": batch,
            "in_channels": in_channels,
            "in_width": in_width,
            "out_channels": out_channels,
            "kernel_w": kernel_w,
            "out_width": out_width,
            "stride_w": attrs["stride"][0],
            "pad_w": attrs["padding"][0],
            "output_pad_w": 0 if attrs["output_padding"] is None else attrs["output_padding"][0],
            "dilation_w": attrs["dilation"][0],
            "transposed": attrs["transposed"],
            "has_bias": attrs["has_bias"],
            "apply_residual": attrs["apply_residual"],
            "apply_relu": attrs["apply_relu"],
        }
    if attrs["rank"] == 5:
        batch, in_channels, in_depth, in_height, in_width = [int(dim) for dim in x["shape"]]
        out_channels, weight_in_channels, kernel_d, kernel_h, kernel_w = [int(dim) for dim in weight["shape"]]
        out_batch, out_channels_out, out_depth, out_height, out_width = [int(dim) for dim in output["shape"]]
        groups = int(attrs["groups"])
        return {
            "func": _cpu_function_name(node, tensor_map),
            "storage_type": cpu_storage_type(str(output["dtype"])),
            "input_numel": batch * in_channels * in_depth * in_height * in_width,
            "weight_numel": weight_in_channels * out_channels * kernel_d * kernel_h * kernel_w,
            "bias_numel": None if bias is None else out_channels,
            "residual_numel": None if residual is None else out_batch * out_channels_out * out_depth * out_height * out_width,
            "output_numel": out_batch * out_channels_out * out_depth * out_height * out_width,
            "batch": batch,
            "in_channels": in_channels,
            "in_depth": in_depth,
            "in_height": in_height,
            "in_width": in_width,
            "out_channels": out_channels,
            "weight_in_channels": weight_in_channels,
            "groups": groups,
            "in_channels_per_group": in_channels // groups,
            "out_channels_per_group": out_channels // groups,
            "kernel_d": kernel_d,
            "kernel_h": kernel_h,
            "kernel_w": kernel_w,
            "out_depth": out_depth,
            "out_height": out_height,
            "out_width": out_width,
            "stride_d": attrs["stride"][0],
            "stride_h": attrs["stride"][1],
            "stride_w": attrs["stride"][2],
            "pad_d": attrs["padding"][0],
            "pad_h": attrs["padding"][1],
            "pad_w": attrs["padding"][2],
            "dilation_d": attrs["dilation"][0],
            "dilation_h": attrs["dilation"][1],
            "dilation_w": attrs["dilation"][2],
            "has_bias": attrs["has_bias"],
        }
    batch, in_channels, in_height, in_width = [int(dim) for dim in x["shape"]]
    weight_dims = [int(dim) for dim in weight["shape"]]
    if attrs["transposed"]:
        weight_in_channels, out_channels, kernel_h, kernel_w = weight_dims
    else:
        out_channels, weight_in_channels, kernel_h, kernel_w = weight_dims
    out_batch, out_channels_out, out_height, out_width = [int(dim) for dim in output["shape"]]
    return {
        "func": _cpu_function_name(node, tensor_map),
        "storage_type": cpu_storage_type(str(output["dtype"])),
        "input_numel": batch * in_channels * in_height * in_width,
        "weight_numel": weight_in_channels * out_channels * kernel_h * kernel_w,
        "bias_numel": None if bias is None else out_channels,
        "residual_numel": None if residual is None else out_batch * out_channels_out * out_height * out_width,
        "output_numel": out_batch * out_channels_out * out_height * out_width,
        "batch": batch,
        "in_channels": in_channels,
        "in_height": in_height,
        "in_width": in_width,
        "out_channels": out_channels,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "out_height": out_height,
        "out_width": out_width,
        "stride_h": attrs["stride"][0],
        "stride_w": attrs["stride"][1],
        "pad_h": attrs["padding"][0],
        "pad_w": attrs["padding"][1],
        "dilation_h": attrs["dilation"][0],
        "dilation_w": attrs["dilation"][1],
        "transposed": attrs["transposed"],
        "has_bias": attrs["has_bias"],
        "apply_residual": attrs["apply_residual"],
        "apply_relu": attrs["apply_relu"],
    }


def _cpu_function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    x_name = str(node["inputs"][0])
    weight_name = str(node["inputs"][1])
    bias_name = None if not _cpu_op_has_bias(str(node["op"])) else str(node["inputs"][2])
    out_name = str(node["outputs"][0])
    x = tensor_map[x_name]
    weight = tensor_map[weight_name]
    bias = None if bias_name is None else tensor_map[bias_name]
    residual = None
    if _conv_family_has_residual(str(node["op"])):
        residual_index = 3 if bias_name is not None else 2
        residual = tensor_map[str(node["inputs"][residual_index])]
    output = tensor_map[out_name]
    attrs = _validate_cpu_contract(node, x, weight, bias, output, residual)
    signature = {
        "op": str(node["op"]),
        "input_shape": [int(dim) for dim in x["shape"]],
        "weight_shape": [int(dim) for dim in weight["shape"]],
        "bias_shape": None if bias is None else [int(dim) for dim in bias["shape"]],
        "residual_shape": None if residual is None else [int(dim) for dim in residual["shape"]],
        "output_shape": [int(dim) for dim in output["shape"]],
        "stride": attrs["stride"],
        "padding": attrs["padding"],
        "output_padding": attrs["output_padding"],
        "dilation": attrs["dilation"],
        "groups": attrs["groups"],
        "dtype": str(output["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"{str(node['op'])}_{digest}"


def _validate_cpu_contract(
    node: Mapping[str, Any],
    x: Mapping[str, Any],
    weight: Mapping[str, Any],
    bias: Mapping[str, Any] | None,
    output: Mapping[str, Any],
    residual: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    op_name = str(node["op"])
    if op_name not in {
        *CONV1D_BIAS_FAMILY_OPS,
        *CONV2D_BIAS_FAMILY_OPS,
        *CONV3D_FAMILY_OPS,
        *TRANSPOSED_CONV1D_OPS,
        *TRANSPOSED_CONV2D_FAMILY_OPS,
    }:
        raise ValueError(f"Unsupported Conv op for CPU lowering: {node['op']}")
    dtype = str(output["dtype"])
    if dtype not in CONV2D_BIAS_DTYPES:
        raise NotImplementedError(f"{op_name} CPU lowering does not support dtype {dtype!r}")
    for tensor in (x, weight, bias, output, residual):
        if tensor is None:
            continue
        if str(tensor["dtype"]) != dtype:
            raise ValueError(f"{op_name} CPU lowering requires matching tensor dtypes")
    attrs = node.get("attrs", {})
    if op_name in {*CONV1D_BIAS_FAMILY_OPS, *TRANSPOSED_CONV1D_OPS}:
        if op_name in TRANSPOSED_CONV1D_OPS:
            stride, padding, output_padding, dilation, groups = normalize_transposed_conv1d_attrs(
                attrs.get("stride", (1,)),
                attrs.get("padding", (0,)),
                attrs.get("output_padding", (0,)),
                attrs.get("dilation", (1,)),
                attrs.get("groups", 1),
            )
            expected_shape = resolve_transposed_conv1d_shape(
                x["shape"],
                weight["shape"],
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                groups=groups,
            )
        else:
            stride, padding, dilation, groups = normalize_conv1d_bias_attrs(
                attrs.get("stride", (1,)),
                attrs.get("padding", (0,)),
                attrs.get("dilation", (1,)),
                attrs.get("groups", 1),
            )
            output_padding = None
            if bias is None:
                raise ValueError(f"{op_name} CPU lowering requires a bias tensor")
            if op_name == "conv1d_bias":
                expected_shape = resolve_conv1d_bias_shape(
                    x["shape"], weight["shape"], bias["shape"], stride=stride, padding=padding, dilation=dilation, groups=groups
                )
            elif op_name == "conv1d_bias_relu":
                expected_shape = resolve_conv1d_bias_relu_shape(
                    x["shape"], weight["shape"], bias["shape"], stride=stride, padding=padding, dilation=dilation, groups=groups
                )
            elif op_name == "conv1d_bias_add":
                if residual is None:
                    raise ValueError("conv1d_bias_add CPU lowering requires a residual tensor")
                expected_shape = resolve_conv1d_bias_add_shape(
                    x["shape"],
                    weight["shape"],
                    bias["shape"],
                    residual["shape"],
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
            else:
                if residual is None:
                    raise ValueError("conv1d_bias_add_relu CPU lowering requires a residual tensor")
                expected_shape = resolve_conv1d_bias_add_relu_shape(
                    x["shape"],
                    weight["shape"],
                    bias["shape"],
                    residual["shape"],
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
    elif op_name in CONV2D_BIAS_FAMILY_OPS:
        stride, padding, dilation, groups = normalize_conv2d_bias_attrs(
            attrs.get("stride", (1, 1)),
            attrs.get("padding", (0, 0)),
            attrs.get("dilation", (1, 1)),
            attrs.get("groups", 1),
        )
        output_padding = None
        if bias is None:
            raise ValueError(f"{op_name} CPU lowering requires a bias tensor")
        if op_name == "conv2d_bias":
            expected_shape = resolve_conv2d_bias_shape(
                x["shape"], weight["shape"], bias["shape"], stride=stride, padding=padding, dilation=dilation, groups=groups
            )
        elif op_name == "conv2d_bias_relu":
            expected_shape = resolve_conv2d_bias_relu_shape(
                x["shape"], weight["shape"], bias["shape"], stride=stride, padding=padding, dilation=dilation, groups=groups
            )
        elif op_name == "conv2d_bias_add":
            if residual is None:
                raise ValueError("conv2d_bias_add CPU lowering requires a residual tensor")
            expected_shape = resolve_conv2d_bias_add_shape(
                x["shape"],
                weight["shape"],
                bias["shape"],
                residual["shape"],
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        else:
            if residual is None:
                raise ValueError("conv2d_bias_add_relu CPU lowering requires a residual tensor")
            expected_shape = resolve_conv2d_bias_add_relu_shape(
                x["shape"],
                weight["shape"],
                bias["shape"],
                residual["shape"],
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
    elif op_name in CONV3D_FAMILY_OPS:
        stride, padding, dilation, groups = normalize_conv3d_attrs(
            attrs.get("stride", (1, 1, 1)),
            attrs.get("padding", (0, 0, 0)),
            attrs.get("dilation", (1, 1, 1)),
            attrs.get("groups", 1),
        )
        output_padding = None
        if op_name == "conv3d_bias":
            if bias is None:
                raise ValueError("conv3d_bias CPU lowering requires a bias tensor")
            expected_shape = resolve_conv3d_bias_shape(
                x["shape"], weight["shape"], bias["shape"], stride=stride, padding=padding, dilation=dilation, groups=groups
            )
        else:
            expected_shape = resolve_depthwise_conv3d_shape(
                x["shape"], weight["shape"], stride=stride, padding=padding, dilation=dilation, groups=groups
            )
    else:
        stride, padding, output_padding, dilation, groups = normalize_transposed_conv2d_attrs(
            attrs.get("stride", (1, 1)),
            attrs.get("padding", (0, 0)),
            attrs.get("output_padding", (0, 0)),
            attrs.get("dilation", (1, 1)),
            attrs.get("groups", 1),
        )
        if op_name == "transposed_conv2d":
            expected_shape = resolve_transposed_conv2d_shape(
                x["shape"],
                weight["shape"],
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                groups=groups,
            )
        elif op_name == "transposed_conv2d_bias":
            if bias is None:
                raise ValueError("transposed_conv2d_bias CPU lowering requires a bias tensor")
            expected_shape = resolve_transposed_conv2d_bias_shape(
                x["shape"],
                weight["shape"],
                bias["shape"],
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                groups=groups,
            )
        elif op_name == "transposed_conv2d_bias_relu":
            if bias is None:
                raise ValueError("transposed_conv2d_bias_relu CPU lowering requires a bias tensor")
            expected_shape = resolve_transposed_conv2d_bias_relu_shape(
                x["shape"],
                weight["shape"],
                bias["shape"],
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                groups=groups,
            )
        elif op_name == "transposed_conv2d_bias_add":
            if bias is None or residual is None:
                raise ValueError("transposed_conv2d_bias_add CPU lowering requires bias and residual tensors")
            expected_shape = resolve_transposed_conv2d_bias_add_shape(
                x["shape"],
                weight["shape"],
                bias["shape"],
                residual["shape"],
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                groups=groups,
            )
        else:
            if bias is None or residual is None:
                raise ValueError("transposed_conv2d_bias_add_relu CPU lowering requires bias and residual tensors")
            expected_shape = resolve_transposed_conv2d_bias_add_relu_shape(
                x["shape"],
                weight["shape"],
                bias["shape"],
                residual["shape"],
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                groups=groups,
            )
    if [int(dim) for dim in output["shape"]] != expected_shape:
        raise ValueError(f"{op_name} CPU lowering output shape does not match attrs")
    return {
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "dilation": dilation,
        "groups": groups,
        "rank": 3 if op_name in {*CONV1D_BIAS_FAMILY_OPS, *TRANSPOSED_CONV1D_OPS} else (5 if op_name in CONV3D_FAMILY_OPS else 4),
        "transposed": op_name in {*TRANSPOSED_CONV1D_OPS, *TRANSPOSED_CONV2D_FAMILY_OPS},
        "has_bias": _cpu_op_has_bias(op_name),
        "apply_residual": _conv_family_has_residual(op_name),
        "apply_relu": op_name in {
            "conv1d_bias_relu",
            "conv1d_bias_add_relu",
            "conv2d_bias_relu",
            "conv2d_bias_add_relu",
            "transposed_conv2d_bias_relu",
            "transposed_conv2d_bias_add_relu",
        },
    }


def _render_cpu_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


def render_conv_wrapper_stage(stage: Mapping[str, Any]) -> str:
    stage_kind = str(stage.get("stage_kind", ""))
    symbol = str(stage.get("symbol", ""))
    if not symbol:
        raise ValueError("CUTLASS Conv wrapper stage is missing a symbol")
    if stage_kind == "transform_helper":
        source = _descriptor_placeholder(stage.get("source"))
        destination = _descriptor_placeholder(stage.get("destination"))
        shape_args = _shape_placeholders(stage)
        return (
            f"DINO_CUDA_CHECK({symbol}("
            f"{', '.join([source, destination, *shape_args, 'stream'])}));"
        )
    if stage_kind == "provider_launcher":
        inputs = stage.get("inputs")
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("CUTLASS Conv provider_launcher stage inputs must be a list")
        output = _descriptor_placeholder(stage.get("output"))
        pointer_args = [_descriptor_placeholder(item) for item in inputs]
        shape_args = _shape_placeholders(stage)
        status_name = f"status_{_c_ident(str(stage.get('stage_name', 'cutlass_conv')))}"
        return (
            f"int {status_name} = {symbol}("
            f"{', '.join([*pointer_args, output, *shape_args, 'stream'])});\n"
            f"if ({status_name} != 0) {{\n"
            f"  return {status_name};\n"
            f"}}"
        )
    raise ValueError(f"Unsupported CUTLASS Conv wrapper stage kind {stage_kind!r}")


def _render_runtime_stage(stage: Mapping[str, Any], *, roles: Mapping[str, str]) -> list[str]:
    stage_kind = str(stage.get("stage_kind", ""))
    symbol = str(stage.get("symbol", ""))
    if not symbol:
        raise ValueError("CUTLASS Conv wrapper stage is missing a symbol")
    if stage_kind == "transform_helper":
        source = _runtime_descriptor_expr(stage.get("source"), roles=roles, node_id=stage.get("node_id"))
        destination = _runtime_descriptor_expr(stage.get("destination"), roles=roles, node_id=stage.get("node_id"))
        shape_args = _runtime_shape_args(stage, roles=roles)
        return [
            f"if (int err = {symbol}({', '.join([source, destination, *shape_args, 'session->stream'])})) "
            f'return dinoml::module::fail("{stage.get("op", "conv2d_bias")} CUTLASS Conv {stage.get("stage_name", "transform")} failed");'
        ]
    if stage_kind == "provider_launcher":
        inputs = stage.get("inputs")
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("CUTLASS Conv provider launcher stage inputs must be a list")
        pointer_args = [_runtime_descriptor_expr(item, roles=roles, node_id=stage.get("node_id")) for item in inputs]
        output = _runtime_descriptor_expr(stage.get("output"), roles=roles, node_id=stage.get("node_id"))
        shape_args = _runtime_shape_args(stage, roles=roles)
        op_name = str(stage.get("op") or "conv2d_bias")
        node_scope = _c_ident(str(stage.get("node_id") or "conv"))
        stage_scope = _c_ident(str(stage.get("stage_name", "provider_launch")))
        status_name = f"status_{node_scope}_{stage_scope}"
        if str(stage.get("status", "")) == "bounded_runtime":
            failure_message = f"{op_name} CUTLASS Conv provider launcher failed"
        else:
            failure_message = f"{op_name} CUTLASS Conv provider launcher is unsupported by the current Conv implementation"
        return [
            f"int {status_name} = {symbol}({', '.join([*pointer_args, output, *shape_args, 'session->stream'])});",
            f"if ({status_name} != 0) {{",
            f'  return dinoml::module::fail("{failure_message}");',
            "}",
        ]
    raise ValueError(f"Unsupported CUTLASS Conv wrapper stage kind {stage_kind!r}")


def _runtime_descriptor_expr(descriptor: Any, *, roles: Mapping[str, str], node_id: Any = None) -> str:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"Malformed CUTLASS Conv descriptor: {descriptor!r}")
    kind = str(descriptor.get("kind", ""))
    if kind == "semantic_tensor":
        role = str(descriptor.get("role", ""))
        if role not in roles:
            raise ValueError(f"Unsupported CUTLASS Conv semantic tensor role {role!r}")
        return f"ptr_{roles[role]}"
    if kind == "temporary_buffer":
        node_id = str(descriptor.get("node_id") or node_id or "")
        name = str(descriptor.get("name", ""))
        if not node_id or not name:
            raise ValueError("CUTLASS Conv temporary buffer descriptor is missing node_id or name")
        return f"session->cutlass_conv_tmp_{_c_ident(node_id)}_{_c_ident(name)}"
    raise ValueError(f"Unsupported CUTLASS Conv descriptor kind {kind!r}")


def _runtime_shape_args(stage: Mapping[str, Any], *, roles: Mapping[str, str]) -> list[str]:
    public_rank = int(stage.get("public_rank", 4) or 4)
    args = []
    for item in _shape_arg_items(stage):
        name = str(item.get("name", ""))
        placeholder = str(item.get("placeholder", ""))
        if placeholder == "activation_n":
            args.append(f"static_cast<int>(shape_{roles['activation']}_0)")
        elif placeholder == "activation_c":
            args.append(f"static_cast<int>(shape_{roles['activation']}_1)")
        elif placeholder == "activation_d":
            args.append(
                "1"
                if public_rank < 5
                else f"static_cast<int>(shape_{roles['activation']}_2)"
            )
        elif placeholder == "activation_h":
            args.append(
                "1"
                if public_rank == 3
                else (
                    f"static_cast<int>(shape_{roles['activation']}_3)"
                    if public_rank == 5
                    else f"static_cast<int>(shape_{roles['activation']}_2)"
                )
            )
        elif placeholder == "activation_w":
            args.append(
                f"static_cast<int>(shape_{roles['activation']}_{2 if public_rank == 3 else (4 if public_rank == 5 else 3)})"
            )
        elif placeholder == "output_n":
            args.append(f"static_cast<int>(shape_{roles['output']}_0)")
        elif placeholder == "output_c":
            args.append(f"static_cast<int>(shape_{roles['output']}_1)")
        elif placeholder == "output_d":
            args.append(
                "1"
                if public_rank < 5
                else f"static_cast<int>(shape_{roles['output']}_2)"
            )
        elif placeholder == "output_h":
            args.append(
                "1"
                if public_rank == 3
                else (
                    f"static_cast<int>(shape_{roles['output']}_3)"
                    if public_rank == 5
                    else f"static_cast<int>(shape_{roles['output']}_2)"
                )
            )
        elif placeholder == "output_w":
            args.append(
                f"static_cast<int>(shape_{roles['output']}_{2 if public_rank == 3 else (4 if public_rank == 5 else 3)})"
            )
        elif placeholder == "weight_o":
            args.append(f"static_cast<int>(shape_{roles['weight']}_0)")
        elif placeholder == "weight_i":
            args.append(f"static_cast<int>(shape_{roles['weight']}_1)")
        elif placeholder == "weight_0":
            args.append(f"static_cast<int>(shape_{roles['weight']}_0)")
        elif placeholder == "weight_1":
            args.append(f"static_cast<int>(shape_{roles['weight']}_1)")
        elif placeholder == "kernel_d":
            args.append(
                "1"
                if public_rank < 5
                else f"static_cast<int>(shape_{roles['weight']}_2)"
            )
        elif placeholder == "kernel_h":
            args.append(
                "1"
                if public_rank == 3
                else (
                    f"static_cast<int>(shape_{roles['weight']}_3)"
                    if public_rank == 5
                    else f"static_cast<int>(shape_{roles['weight']}_2)"
                )
            )
        elif placeholder == "kernel_w":
            args.append(
                f"static_cast<int>(shape_{roles['weight']}_{2 if public_rank == 3 else (4 if public_rank == 5 else 3)})"
            )
        elif name in {"stride_h", "stride_w", "pad_h", "pad_w", "dilation_h", "dilation_w"}:
            args.append(str(int(item["value"])))
        else:
            args.append(str(int(item["value"])))
    return args


def _shape_arg_items(stage: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_args = stage.get("shape_args")
    if not isinstance(raw_args, (list, tuple)):
        raise ValueError("CUTLASS Conv wrapper stage shape_args must be a list")
    items = []
    for item in raw_args:
        if not isinstance(item, Mapping):
            raise ValueError(f"Malformed CUTLASS Conv shape arg descriptor: {item!r}")
        items.append(item)
    return items


def _manifest_kernel_item(
    kernel_manifest: Mapping[str, Any] | None,
    op_name: str,
    *,
    node_id: str,
    kernel_library: str = "cutlass_conv",
) -> Mapping[str, Any] | None:
    if kernel_manifest is None:
        return None
    matches = []
    for item in kernel_manifest.get("required_kernels", []):
        if not isinstance(item, Mapping):
            continue
        if item.get("op") != op_name or item.get("kernel_library") != kernel_library:
            continue
        if kernel_library == "ck_conv":
            runtime_plan = item.get("ck_conv_runtime_plan")
            if isinstance(runtime_plan, Mapping) and str(runtime_plan.get("node_id", "")) == node_id:
                return item
            matches.append(item)
            continue
        if kernel_library != "cutlass_conv":
            matches.append(item)
            continue
        conv_plan = item.get("cutlass_conv_plan")
        if isinstance(conv_plan, Mapping) and str(conv_plan.get("node_id", "")) == node_id:
            return item
        matches.append(item)
    return matches[0] if matches else None


def _rocm_dispatch_selections(item: Mapping[str, Any] | None, node_id: str) -> list[Mapping[str, Any]]:
    if not isinstance(item, Mapping):
        return []
    return [
        selection
        for selection in item.get("execution_plan_dispatch", [])
        if isinstance(selection, Mapping) and str(selection.get("node_id")) == node_id
    ]


def _rocm_dispatch_lines(
    *,
    op_name: str,
    item: Mapping[str, Any] | None,
    dispatches: list[Mapping[str, Any]],
    default_launch: list[str],
    x_ident: str,
    weight_ident: str,
    bias_ident: str,
    residual_ident: str | None,
    out_ident: str,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    output_pad_h: int,
    output_pad_w: int,
    dilation_h: int,
    dilation_w: int,
    transposed: bool,
    weight_is_prepacked: bool = False,
) -> list[str]:
    lines = []
    for index, dispatch in enumerate(dispatches):
        candidate = _candidate_by_id(item, str(dispatch.get("selected_candidate_id", "")))
        branch = "if" if index == 0 else "else if"
        lines.append(
            f"{branch} ({_rocm_dispatch_guard(dispatch, x_ident=x_ident, weight_ident=weight_ident, out_ident=out_ident)}) {{"
        )
        if op_name in CONV1D_BIAS_FAMILY_OPS:
            body = _rocm_launch_lines_1d(
                op_name=op_name,
                symbol=str(dispatch.get("kernel_symbol") or candidate.get("kernel_symbol")),
                failure_label="CK Conv",
                x_ident=x_ident,
                weight_ident=weight_ident,
                bias_ident=bias_ident,
                residual_ident=residual_ident,
                out_ident=out_ident,
                stride_w=stride_w,
                pad_w=pad_w,
                dilation_w=dilation_w,
                weight_is_prepacked=weight_is_prepacked,
            )
        elif op_name in TRANSPOSED_CONV1D_OPS:
            body = _rocm_launch_lines_transposed_1d(
                op_name=op_name,
                symbol=str(dispatch.get("kernel_symbol") or candidate.get("kernel_symbol")),
                failure_label="CK Conv",
                x_ident=x_ident,
                weight_ident=weight_ident,
                out_ident=out_ident,
                stride_w=stride_w,
                pad_w=pad_w,
                output_pad_w=output_pad_w,
                dilation_w=dilation_w,
            )
        else:
            body = (
                _rocm_launch_lines(
                    op_name=op_name,
                    symbol=str(dispatch.get("kernel_symbol") or candidate.get("kernel_symbol")),
                    failure_label="CK Conv",
                    x_ident=x_ident,
                    weight_ident=weight_ident,
                    bias_ident=bias_ident,
                    residual_ident=residual_ident,
                    out_ident=out_ident,
                    stride_h=stride_h,
                    stride_w=stride_w,
                    pad_h=pad_h,
                    pad_w=pad_w,
                    output_pad_h=output_pad_h,
                    output_pad_w=output_pad_w,
                    dilation_h=dilation_h,
                    dilation_w=dilation_w,
                    transposed=transposed,
                )
                if transposed
                else _rocm_launch_lines_2d_ck(
                    op_name=op_name,
                    symbol=str(dispatch.get("kernel_symbol") or candidate.get("kernel_symbol")),
                    failure_label="CK Conv",
                    x_ident=x_ident,
                    weight_ident=weight_ident,
                    bias_ident=bias_ident,
                    residual_ident=residual_ident,
                    out_ident=out_ident,
                    stride_h=stride_h,
                    stride_w=stride_w,
                    pad_h=pad_h,
                    pad_w=pad_w,
                    dilation_h=dilation_h,
                    dilation_w=dilation_w,
                    weight_is_prepacked=weight_is_prepacked,
                )
            )
        lines.extend(f"  {line}" for line in body)
        lines.append("}")
    lines.append("else {")
    lines.extend(f"  {line}" for line in default_launch)
    lines.append("}")
    return lines


def _rocm_dispatch_guard(
    selection: Mapping[str, Any],
    *,
    x_ident: str,
    weight_ident: str,
    out_ident: str,
) -> str:
    shape = selection.get("shape")
    if not isinstance(shape, Mapping):
        return "false"
    is_conv1d = "h" not in shape and "out_h" not in shape and "kernel_h" not in shape
    required = (
        {
            "n": f"shape_{x_ident}_0",
            "c": f"shape_{x_ident}_1",
            "w": f"shape_{x_ident}_2",
            "out_n": f"shape_{out_ident}_0",
            "out_c": f"shape_{out_ident}_1",
            "out_w": f"shape_{out_ident}_2",
            "kernel_w": f"shape_{weight_ident}_2",
        }
        if is_conv1d
        else {
            "n": f"shape_{x_ident}_0",
            "c": f"shape_{x_ident}_1",
            "h": f"shape_{x_ident}_2",
            "w": f"shape_{x_ident}_3",
            "out_n": f"shape_{out_ident}_0",
            "out_c": f"shape_{out_ident}_1",
            "out_h": f"shape_{out_ident}_2",
            "out_w": f"shape_{out_ident}_3",
            "kernel_h": f"shape_{weight_ident}_2",
            "kernel_w": f"shape_{weight_ident}_3",
        }
    )
    conditions = []
    for field, expr in required.items():
        value = shape.get(field)
        if type(value) is not int or value <= 0:
            return "false"
        conditions.append(f"({expr}) == {int(value)}")
    return " && ".join(conditions)


def _rocm_launch_lines(
    *,
    op_name: str,
    symbol: str,
    failure_label: str,
    x_ident: str,
    weight_ident: str,
    bias_ident: str,
    residual_ident: str | None,
    out_ident: str,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    output_pad_h: int,
    output_pad_w: int,
    dilation_h: int,
    dilation_w: int,
    transposed: bool,
) -> list[str]:
    call_args = [f"ptr_{x_ident}", f"ptr_{weight_ident}"]
    if bias_ident is not None:
        call_args.append(f"ptr_{bias_ident}")
    if residual_ident is not None:
        call_args.append(f"ptr_{residual_ident}")
    call_args.extend(
        [
            f"ptr_{out_ident}",
            f"static_cast<int>(shape_{x_ident}_0)",
            f"static_cast<int>(shape_{x_ident}_1)",
            f"static_cast<int>(shape_{x_ident}_2)",
            f"static_cast<int>(shape_{x_ident}_3)",
            (
                f"static_cast<int>(shape_{weight_ident}_1)"
                if transposed
                else f"static_cast<int>(shape_{weight_ident}_0)"
            ),
            f"static_cast<int>(shape_{weight_ident}_2)",
            f"static_cast<int>(shape_{weight_ident}_3)",
            f"static_cast<int>(shape_{out_ident}_2)",
            f"static_cast<int>(shape_{out_ident}_3)",
            str(stride_h),
            str(stride_w),
            str(pad_h),
            str(pad_w),
        ]
    )
    if transposed:
        call_args.extend([str(output_pad_h), str(output_pad_w)])
    call_args.extend([str(dilation_h), str(dilation_w), "session->stream"])
    return [
        (
            f"if (int err = {symbol}({', '.join(call_args)})) "
            f'return dinoml::module::fail("{op_name} {failure_label} launcher failed");'
        )
    ]


def _rocm_launch_lines_1d(
    *,
    op_name: str,
    symbol: str,
    failure_label: str,
    x_ident: str,
    weight_ident: str,
    bias_ident: str | None,
    residual_ident: str | None,
    out_ident: str,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    weight_is_prepacked: bool,
) -> list[str]:
    call_args = [f"ptr_{x_ident}", f"ptr_{weight_ident}"]
    if bias_ident is not None:
        call_args.append(f"ptr_{bias_ident}")
    if residual_ident is not None:
        call_args.append(f"ptr_{residual_ident}")
    call_args.extend(
        [
            f"ptr_{out_ident}",
            f"static_cast<int>(shape_{x_ident}_0)",
            f"static_cast<int>(shape_{x_ident}_1)",
            f"static_cast<int>(shape_{x_ident}_2)",
            f"static_cast<int>(shape_{weight_ident}_0)",
            f"static_cast<int>(shape_{weight_ident}_2)",
            f"static_cast<int>(shape_{out_ident}_2)",
            str(stride_w),
            str(pad_w),
            str(dilation_w),
            "1" if weight_is_prepacked else "0",
            "session->stream",
        ]
    )
    return [
        (
            f"if (int err = {symbol}({', '.join(call_args)})) "
            f'return dinoml::module::fail("{op_name} {failure_label} launcher failed");'
        )
    ]


def _rocm_launch_lines_2d_ck(
    *,
    op_name: str,
    symbol: str,
    failure_label: str,
    x_ident: str,
    weight_ident: str,
    bias_ident: str | None,
    residual_ident: str | None,
    out_ident: str,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    weight_is_prepacked: bool,
) -> list[str]:
    call_args = [f"ptr_{x_ident}", f"ptr_{weight_ident}"]
    if bias_ident is not None:
        call_args.append(f"ptr_{bias_ident}")
    if residual_ident is not None:
        call_args.append(f"ptr_{residual_ident}")
    call_args.extend(
        [
            f"ptr_{out_ident}",
            f"static_cast<int>(shape_{x_ident}_0)",
            f"static_cast<int>(shape_{x_ident}_1)",
            f"static_cast<int>(shape_{x_ident}_2)",
            f"static_cast<int>(shape_{x_ident}_3)",
            f"static_cast<int>(shape_{weight_ident}_0)",
            f"static_cast<int>(shape_{weight_ident}_2)",
            f"static_cast<int>(shape_{weight_ident}_3)",
            f"static_cast<int>(shape_{out_ident}_2)",
            f"static_cast<int>(shape_{out_ident}_3)",
            str(stride_h),
            str(stride_w),
            str(pad_h),
            str(pad_w),
            str(dilation_h),
            str(dilation_w),
            "1" if weight_is_prepacked else "0",
            "session->stream",
        ]
    )
    return [
        (
            f"if (int err = {symbol}({', '.join(call_args)})) "
            f'return dinoml::module::fail("{op_name} {failure_label} launcher failed");'
        )
    ]


def _rocm_launch_lines_transposed_1d(
    *,
    op_name: str,
    symbol: str,
    failure_label: str,
    x_ident: str,
    weight_ident: str,
    out_ident: str,
    stride_w: int,
    pad_w: int,
    output_pad_w: int,
    dilation_w: int,
) -> list[str]:
    call_args = [
        f"ptr_{x_ident}",
        f"ptr_{weight_ident}",
        f"ptr_{out_ident}",
        f"static_cast<int>(shape_{x_ident}_0)",
        f"static_cast<int>(shape_{x_ident}_1)",
        "1",
        f"static_cast<int>(shape_{x_ident}_2)",
        f"static_cast<int>(shape_{weight_ident}_1)",
        "1",
        f"static_cast<int>(shape_{weight_ident}_2)",
        "1",
        f"static_cast<int>(shape_{out_ident}_2)",
        "1",
        str(stride_w),
        "0",
        str(pad_w),
        "0",
        str(output_pad_w),
        "1",
        str(dilation_w),
        "session->stream",
    ]
    return [
        (
            f"if (int err = {symbol}({', '.join(call_args)})) "
            f'return dinoml::module::fail("{op_name} {failure_label} launcher failed");'
        )
    ]


def _rocm_launch_lines_3d(
    *,
    op_name: str,
    symbol: str,
    failure_label: str,
    x_ident: str,
    weight_ident: str,
    bias_ident: str | None,
    out_ident: str,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    dilation_d: int,
    dilation_h: int,
    dilation_w: int,
    weight_is_prepacked: bool,
    groups: int,
) -> list[str]:
    call_args = [f"ptr_{x_ident}", f"ptr_{weight_ident}"]
    if bias_ident is not None:
        call_args.append(f"ptr_{bias_ident}")
    call_args.extend(
        [
            f"ptr_{out_ident}",
            f"static_cast<int>(shape_{x_ident}_0)",
            f"static_cast<int>(shape_{x_ident}_1)",
            f"static_cast<int>(shape_{x_ident}_2)",
            f"static_cast<int>(shape_{x_ident}_3)",
            f"static_cast<int>(shape_{x_ident}_4)",
            f"static_cast<int>(shape_{weight_ident}_0)",
            f"static_cast<int>(shape_{weight_ident}_2)",
            f"static_cast<int>(shape_{weight_ident}_3)",
            f"static_cast<int>(shape_{weight_ident}_4)",
            f"static_cast<int>(shape_{out_ident}_2)",
            f"static_cast<int>(shape_{out_ident}_3)",
            f"static_cast<int>(shape_{out_ident}_4)",
            str(stride_d),
            str(stride_h),
            str(stride_w),
            str(pad_d),
            str(pad_h),
            str(pad_w),
            str(dilation_d),
            str(dilation_h),
            str(dilation_w),
            "1" if weight_is_prepacked else "0",
            str(groups),
            "session->stream",
        ]
    )
    return [
        (
            f"if (int err = {symbol}({', '.join(call_args)})) "
            f'return dinoml::module::fail("{op_name} {failure_label} launcher failed");'
        )
    ]


def _candidate_by_id(item: Mapping[str, Any] | None, candidate_id: str) -> Mapping[str, Any]:
    if not isinstance(item, Mapping):
        return {}
    for candidate in item.get("candidates", []):
        if isinstance(candidate, Mapping) and str(candidate.get("candidate_id")) == candidate_id:
            return candidate
    return {}


def _validate_runtime_shape_contract(
    op_name: str,
    tensor_map: Mapping[str, Mapping[str, Any]],
    input_names: list[str],
    output_name: str,
    item: Mapping[str, Any],
) -> None:
    conv_plan = item.get("cutlass_conv_plan")
    if not isinstance(conv_plan, Mapping):
        raise ValueError(f"{op_name} CUTLASS Conv manifest entry is missing cutlass_conv_plan")
    x_name, weight_name = input_names[:2]
    x = tensor_map[x_name]
    weight = tensor_map[weight_name]
    output = tensor_map[output_name]
    expected = {
        "input_shape": x["shape"],
        "weight_shape": weight["shape"],
        "output_shape": output["shape"],
    }
    if op_name == "transposed_conv1d":
        expected = {
            "input_shape": [int(x["shape"][0]), int(x["shape"][1]), 1, int(x["shape"][2])],
            "weight_shape": [int(weight["shape"][0]), int(weight["shape"][1]), 1, int(weight["shape"][2])],
            "output_shape": [int(output["shape"][0]), int(output["shape"][1]), 1, int(output["shape"][2])],
        }
    if op_name not in CUTLASS_TRANSPOSED_CONV_OPS:
        bias_name = input_names[2]
        expected["bias_shape"] = tensor_map[bias_name]["shape"]
        residual = None if not _conv_family_has_residual(op_name) else tensor_map[input_names[3]]
        if residual is not None:
            expected["residual_shape"] = residual["shape"]
    for field, shape in expected.items():
        if [int(dim) for dim in conv_plan.get(field, [])] != [int(dim) for dim in shape]:
            raise ValueError(f"{op_name} CUTLASS Conv manifest {field} does not match lowered tensor shape")


def render_conv_wrapper_stages(stages: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> list[str]:
    return [render_conv_wrapper_stage(stage) for stage in stages]


def render_conv_wrapper_source(
    stages: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    op_name: str | None = None,
    node_id: str | None = None,
) -> str:
    if not stages:
        raise ValueError("CUTLASS Conv wrapper source requires at least one stage")
    first_stage = stages[0]
    op_name = str(op_name or first_stage.get("op") or "conv2d_bias")
    node_id = None if node_id is None and first_stage.get("node_id") is None else str(node_id or first_stage.get("node_id"))
    snippet_lines = []
    for stage in stages:
        snippet_lines.extend(render_conv_wrapper_stage(stage).splitlines())
    function_suffix = _c_ident(node_id or op_name)
    lines = [
        "// CUTLASS Conv only: emitted for artifact/source inspection.",
        "// This debug wrapper snippet is intentionally not compiled into the runtime module.",
        f"// op: {op_name}",
        f"// node_id: {node_id or '<unknown>'}",
        "#if 0",
        f'extern "C" int dinoml_cutlass_conv_wrapper_{function_suffix}(cudaStream_t stream) {{',
    ]
    lines.extend(f"  {line}" for line in snippet_lines)
    lines.extend(
        [
            "  return 0;",
            "}",
            "#endif",
            "",
        ]
    )
    return "\n".join(lines)


def _descriptor_placeholder(descriptor: Any) -> str:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"Malformed CUTLASS Conv descriptor: {descriptor!r}")
    kind = str(descriptor.get("kind", ""))
    if kind == "semantic_tensor":
        role = str(descriptor.get("role", ""))
        if role not in {"activation", "weight", "bias", "output"}:
            if role != "residual":
                raise ValueError(f"Unsupported CUTLASS Conv semantic tensor role {role!r}")
        return f"ptr_{role}"
    if kind == "temporary_buffer":
        name = str(descriptor.get("name", ""))
        if not name:
            raise ValueError("CUTLASS Conv temporary buffer descriptor is missing name")
        return f"tmp_{_c_ident(name)}"
    raise ValueError(f"Unsupported CUTLASS Conv descriptor kind {kind!r}")


def _shape_placeholders(stage: Mapping[str, Any]) -> list[str]:
    raw_args = stage.get("shape_args")
    if not isinstance(raw_args, (list, tuple)):
        raise ValueError("CUTLASS Conv wrapper stage shape_args must be a list")
    placeholders = []
    for item in raw_args:
        if not isinstance(item, Mapping) or not str(item.get("placeholder", "")):
            raise ValueError(f"Malformed CUTLASS Conv shape arg descriptor: {item!r}")
        placeholders.append(str(item["placeholder"]))
    return placeholders


CONV1D_BIAS_LOWERING = OpLowering(
    op_name="conv1d_bias",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


CONV1D_BIAS_RELU_LOWERING = OpLowering(
    op_name="conv1d_bias_relu",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


CONV1D_BIAS_ADD_LOWERING = OpLowering(
    op_name="conv1d_bias_add",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


CONV1D_BIAS_ADD_RELU_LOWERING = OpLowering(
    op_name="conv1d_bias_add_relu",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


CONV2D_BIAS_LOWERING = OpLowering(
    op_name="conv2d_bias",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


CONV2D_BIAS_RELU_LOWERING = OpLowering(
    op_name="conv2d_bias_relu",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


CONV2D_BIAS_ADD_LOWERING = OpLowering(
    op_name="conv2d_bias_add",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


CONV2D_BIAS_ADD_RELU_LOWERING = OpLowering(
    op_name="conv2d_bias_add_relu",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


CONV3D_BIAS_LOWERING = OpLowering(
    op_name="conv3d_bias",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


DEPTHWISE_CONV3D_LOWERING = OpLowering(
    op_name="depthwise_conv3d",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


TRANSPOSED_CONV1D_LOWERING = OpLowering(
    op_name="transposed_conv1d",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


TRANSPOSED_CONV2D_LOWERING = OpLowering(
    op_name="transposed_conv2d",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


TRANSPOSED_CONV2D_BIAS_LOWERING = OpLowering(
    op_name="transposed_conv2d_bias",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


TRANSPOSED_CONV2D_BIAS_RELU_LOWERING = OpLowering(
    op_name="transposed_conv2d_bias_relu",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


TRANSPOSED_CONV2D_BIAS_ADD_LOWERING = OpLowering(
    op_name="transposed_conv2d_bias_add",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


TRANSPOSED_CONV2D_BIAS_ADD_RELU_LOWERING = OpLowering(
    op_name="transposed_conv2d_bias_add_relu",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


__all__ = [
    "CONV1D_BIAS_ADD_LOWERING",
    "CONV1D_BIAS_ADD_RELU_LOWERING",
    "CONV1D_BIAS_LOWERING",
    "CONV1D_BIAS_RELU_LOWERING",
    "CONV2D_BIAS_ADD_LOWERING",
    "CONV2D_BIAS_ADD_RELU_LOWERING",
    "CONV2D_BIAS_LOWERING",
    "CONV2D_BIAS_RELU_LOWERING",
    "CONV3D_BIAS_LOWERING",
    "DEPTHWISE_CONV3D_LOWERING",
    "TRANSPOSED_CONV2D_BIAS_ADD_LOWERING",
    "TRANSPOSED_CONV2D_BIAS_ADD_RELU_LOWERING",
    "TRANSPOSED_CONV2D_BIAS_LOWERING",
    "TRANSPOSED_CONV2D_BIAS_RELU_LOWERING",
    "TRANSPOSED_CONV2D_LOWERING",
    "render_conv_wrapper_source",
    "render_conv_wrapper_stage",
    "render_conv_wrapper_stages",
]


def _conv2d_bias_family_has_residual(op_name: str) -> bool:
    return op_name in {"conv2d_bias_add", "conv2d_bias_add_relu"}


def _cpu_op_has_bias(op_name: str) -> bool:
    return op_name not in {"transposed_conv1d", "transposed_conv2d", "depthwise_conv3d"}


def _conv_family_has_residual(op_name: str) -> bool:
    return op_name in {
        "conv1d_bias_add",
        "conv1d_bias_add_relu",
        "conv2d_bias_add",
        "conv2d_bias_add_relu",
        "transposed_conv2d_bias_add",
        "transposed_conv2d_bias_add_relu",
    }
