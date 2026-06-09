from __future__ import annotations

from pathlib import Path

import pytest

from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.kernels.profiling import build_profile_workloads
from dinoml.kernels.providers.cutlass.conv import (
    cutlass_conv_static_library_name,
    cutlass_conv_used_candidate_plan,
    cutlass_transposed_conv_candidate_set_id,
    render_cutlass_conv_source,
)
from dinoml.lowering.cuda import render_cuda_module
from dinoml.ops.conv import TRANSPOSED_CONV2D_BIAS_FAMILY_OPS


def test_cuda_transposed_conv2d_cmake_targets_cover_provider_ops():
    cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
    cutlass_conv_ops_block = cmake.split("set(DINOML_CUTLASS_CONV_OPS", 1)[1].split(")", 1)[0]

    assert "transposed_conv2d" in cutlass_conv_ops_block


def test_cuda_transposed_conv2d_uses_cutlass_manifest_and_profile_workloads():
    ir = _cuda_transposed_conv2d_ir(
        "float16",
        op_name="transposed_conv2d",
        batch=2,
        in_channels=8,
        out_channels=64,
        height=16,
        width=16,
        stride=(2, 2),
        padding=(1, 1),
        output_padding=(1, 1),
    )
    manifest = build_kernel_manifest(ir, {"name": "cuda", "arch": "sm_86"})
    item = manifest["required_kernels"][0]
    workloads = build_profile_workloads(ir, manifest)

    assert item["kernel_library"] == "cutlass_conv"
    assert item["support_archive"] == cutlass_conv_static_library_name("transposed_conv2d", "float16")
    assert item["candidate_set_id"] == cutlass_transposed_conv_candidate_set_id("transposed_conv2d", "float16")
    assert item["candidate_set"]["family"] == "conv2d_dgrad"
    assert item["candidate_set"]["epilogue"] == "identity"
    assert item["candidate_set"]["launch_abi"] == "dinoml_cutlass_transposed_conv2d_v1"
    assert item["candidate_set"]["semantic_layout"] == {"activation": "nchw", "weight": "iohw", "output": "nchw"}
    assert item["candidate_set"]["provider_layout"] == {"activation": "nhwc", "weight": "ihwo", "output": "nhwc"}
    assert {candidate["cutlass"]["opclass"] for candidate in item["candidates"]} == {"tensorop"}
    assert {candidate["cutlass"]["iterator_algorithm"] for candidate in item["candidates"]} == {"optimized"}

    conv_plan = item["cutlass_conv_plan"]
    assert conv_plan["kind"] == "cutlass_transposed_conv2d_plan"
    assert conv_plan["conv_config"]["output_padding"] == [1, 1]
    assert conv_plan["layout_translation"]["input_pack"] == "nchw_to_nhwc_temporary"
    assert conv_plan["layout_translation"]["output_unpack"] == "nhwc_to_nchw_temporary"
    assert conv_plan["weight_transform"]["from"] == "iohw"
    assert conv_plan["weight_transform"]["to"] == "ihwo"
    assert conv_plan["weight_transform"]["pack"] == "iohw_to_ihwo_temporary"

    assert {workload.kernel_library for workload in workloads} == {"cutlass_conv"}
    assert {workload.op for workload in workloads} == {"transposed_conv2d"}
    assert all(workload.bias_tensor is None for workload in workloads)
    assert all(workload.bias_shape is None for workload in workloads)
    assert all(workload.conv_config["output_padding"] == [1, 1] for workload in workloads)
    assert all(workload.weight_shape == (8, 64, 3, 3) for workload in workloads)


def test_cuda_transposed_conv2d_module_declares_and_calls_cutlass_symbol():
    ir = _cuda_transposed_conv2d_ir(
        "float16",
        op_name="transposed_conv2d",
        in_channels=8,
        out_channels=64,
        height=16,
        width=16,
        stride=(2, 2),
        padding=(1, 1),
        output_padding=(1, 1),
    )
    manifest = build_kernel_manifest(ir, {"name": "cuda", "arch": "sm_86"})
    item = manifest["required_kernels"][0]
    source = render_cuda_module(ir, kernel_manifest=manifest)
    declaration = (
        f'extern "C" int {item["kernel_symbol"]}(\n'
        "    const void* activation_nhwc,\n"
        "    const void* weight_provider,\n"
        "    void* output_nhwc,"
    )

    assert declaration in source
    assert "dinoml_cutlass_conv_weight_pack_iohw_to_ihwo_float16_v1" in source
    assert f'{item["kernel_symbol"]}(session->cutlass_conv_tmp_n0_activation_nhwc, session->cutlass_conv_tmp_n0_weight_ihwo, session->cutlass_conv_tmp_n0_output_nhwc' in source
    assert "transposed_conv2d CUTLASS Conv activation_pack failed" in source
    assert "transposed_conv2d CUTLASS Conv weight_pack failed" in source
    assert "transposed_conv2d CUTLASS Conv output_unpack failed" in source


def test_cutlass_transposed_conv2d_unit_generation_exports_weight_pack_and_dgrad_kernel():
    ir = _cuda_transposed_conv2d_ir(
        "float16",
        op_name="transposed_conv2d",
        in_channels=8,
        out_channels=64,
        height=16,
        width=16,
        stride=(2, 2),
        padding=(1, 1),
        output_padding=(1, 1),
    )
    manifest = build_kernel_manifest(ir, {"name": "cuda", "arch": "sm_86"})
    item = manifest["required_kernels"][0]
    used_candidate_plan = cutlass_conv_used_candidate_plan(manifest)
    rendered = render_cutlass_conv_source(
        Path("kernels/cuda/src/cutlass_conv.cu").read_text(encoding="utf-8"),
        used_candidate_plan,
    )

    assert "DINOML_CUTLASS_CONV_IOHW_TO_IHWO_EXPORT" in rendered
    assert "DINOML_CUTLASS_TRANSPOSED_CONV2D_EXPORT" in rendered
    assert "dinoml_cutlass_conv_weight_pack_iohw_to_ihwo_float16_v1" in rendered
    assert item["kernel_symbol"] in rendered
    assert item["profiler_symbol"] in rendered


@pytest.mark.parametrize("op_name", TRANSPOSED_CONV2D_BIAS_FAMILY_OPS)
def test_cuda_transposed_conv2d_bias_family_ops_fail_explicitly(op_name: str):
    ir = _cuda_transposed_conv2d_ir("float16", op_name=op_name)

    with pytest.raises(
        NotImplementedError,
        match=rf"{op_name} CUDA backend is unsupported; only transposed_conv2d has native CUTLASS support",
    ):
        build_kernel_manifest(ir, {"name": "cuda", "arch": "sm_86"})


def _cuda_transposed_conv2d_ir(
    dtype: str,
    *,
    op_name: str = "transposed_conv2d",
    batch: int = 2,
    in_channels: int = 4,
    out_channels: int = 6,
    height: int = 8,
    width: int = 8,
    kernel_h: int = 3,
    kernel_w: int = 3,
    stride: list[int] | tuple[int, int] | None = None,
    padding: list[int] | tuple[int, int] | None = None,
    output_padding: list[int] | tuple[int, int] | None = None,
    dilation: list[int] | tuple[int, int] | None = None,
    groups: int = 1,
) -> dict:
    stride = [1, 1] if stride is None else [int(stride[0]), int(stride[1])]
    padding = [1, 1] if padding is None else [int(padding[0]), int(padding[1])]
    output_padding = [0, 0] if output_padding is None else [int(output_padding[0]), int(output_padding[1])]
    dilation = [1, 1] if dilation is None else [int(dilation[0]), int(dilation[1])]
    output_h = (height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_h - 1) + output_padding[0] + 1
    output_w = (width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_w - 1) + output_padding[1] + 1
    has_bias = op_name != "transposed_conv2d"
    has_residual = op_name in {"transposed_conv2d_bias_add", "transposed_conv2d_bias_add_relu"}
    tensors = [
        _tensor("x", [batch, in_channels, height, width], dtype, "input"),
        _tensor("weight", [in_channels, out_channels, kernel_h, kernel_w], dtype, "input"),
        *([_tensor("bias", [out_channels], dtype, "input")] if has_bias else []),
        *(
            [_tensor("residual", [batch, out_channels, output_h, output_w], dtype, "input")]
            if has_residual
            else []
        ),
        _tensor("y", [batch, out_channels, output_h, output_w], dtype, "output"),
    ]
    inputs = [
        _io("x", [batch, in_channels, height, width], dtype),
        _io("weight", [in_channels, out_channels, kernel_h, kernel_w], dtype),
        *([_io("bias", [out_channels], dtype)] if has_bias else []),
        *(
            [_io("residual", [batch, out_channels, output_h, output_w], dtype)]
            if has_residual
            else []
        ),
    ]
    node_inputs = ["x", "weight", *(["bias"] if has_bias else []), *(["residual"] if has_residual else [])]
    return {
        "schema_version": 1,
        "name": f"cuda_{op_name}_smoke",
        "inputs": inputs,
        "constants": [],
        "outputs": [_io("y", [batch, out_channels, output_h, output_w], dtype)],
        "nodes": [
            {
                "id": "n0",
                "op": op_name,
                "inputs": node_inputs,
                "outputs": ["y"],
                "attrs": {
                    "stride": stride,
                    "padding": padding,
                    "output_padding": output_padding,
                    "dilation": dilation,
                    "groups": groups,
                },
            }
        ],
        "tensors": tensors,
        "metadata": {},
    }


def _io(name: str, shape: list[int], dtype: str) -> dict:
    return {
        "name": name,
        "tensor": name,
        "shape": shape,
        "shape_spec": shape,
        "layout": _dense_layout(shape),
        "dtype": dtype,
    }


def _tensor(name: str, shape: list[int], dtype: str, kind: str) -> dict:
    nbytes = 2 if dtype in {"float16", "bfloat16"} else 4
    for dim in shape:
        nbytes *= dim
    return {
        "name": name,
        "shape": shape,
        "shape_spec": shape,
        "layout": _dense_layout(shape),
        "dtype": dtype,
        "kind": kind,
        "nbytes": nbytes,
    }


def _dense_layout(shape: list[int]) -> dict:
    stride = 1
    strides = []
    for dim in reversed(shape):
        strides.insert(0, stride)
        stride *= dim
    return {
        "schema_version": 1,
        "kind": "dense",
        "order": "row_major",
        "strides": strides,
        "storage_offset": 0,
    }
