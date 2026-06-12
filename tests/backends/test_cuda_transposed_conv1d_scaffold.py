from __future__ import annotations

from pathlib import Path

from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.kernels.profiling import build_profile_workloads
from dinoml.kernels.providers.cutlass.conv import (
    cutlass_conv_static_library_name,
    cutlass_conv_used_candidate_plan,
    cutlass_transposed_conv_candidate_set_id,
    render_cutlass_conv_source,
)
from dinoml.lowering.cuda import render_cuda_module


def test_cuda_transposed_conv1d_cmake_targets_cover_provider_ops():
    cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
    cutlass_conv_ops_block = cmake.split("set(DINOML_CUTLASS_CONV_OPS", 1)[1].split(")", 1)[0]

    assert "transposed_conv1d" in cutlass_conv_ops_block


def test_cuda_transposed_conv1d_uses_cutlass_manifest_and_profile_workloads():
    ir = _cuda_transposed_conv1d_ir(
        "float16",
        batch=2,
        in_channels=8,
        out_channels=16,
        width=16,
        stride=2,
        padding=1,
        output_padding=1,
    )
    manifest = build_kernel_manifest(ir, {"name": "cuda", "arch": "sm_86"})
    item = manifest["required_kernels"][0]
    workloads = build_profile_workloads(ir, manifest)

    assert item["kernel_library"] == "cutlass_conv"
    assert item["support_archive"] == cutlass_conv_static_library_name("transposed_conv1d", "float16")
    assert item["candidate_set_id"] == cutlass_transposed_conv_candidate_set_id("transposed_conv1d", "float16")
    assert item["candidate_set"]["family"] == "conv2d_dgrad"
    assert item["candidate_set"]["epilogue"] == "identity"
    assert item["candidate_set"]["launch_abi"] == "dinoml_cutlass_transposed_conv2d_v1"
    assert item["candidate_set"]["semantic_layout"] == {"activation": "ncw", "weight": "iow", "output": "ncw"}
    assert item["candidate_set"]["provider_layout"] == {"activation": "nhwc", "weight": "ihwo", "output": "nhwc"}

    conv_plan = item["cutlass_conv_plan"]
    assert conv_plan["kind"] == "cutlass_transposed_conv2d_plan"
    assert conv_plan["public_rank"] == 3
    assert conv_plan["input_shape"] == [2, 8, 1, 16]
    assert conv_plan["weight_shape"] == [8, 16, 1, 3]
    assert conv_plan["output_shape"] == [2, 16, 1, 32]
    assert conv_plan["conv_config"]["output_padding"] == [0, 1]
    assert conv_plan["layout_translation"]["input_pack"] == "nchw_to_nhwc_temporary"
    assert conv_plan["layout_translation"]["output_unpack"] == "nhwc_to_nchw_temporary"
    assert conv_plan["weight_transform"]["from"] == "iohw"
    assert conv_plan["weight_transform"]["to"] == "ihwo"

    assert {workload.kernel_library for workload in workloads} == {"cutlass_conv"}
    assert {workload.op for workload in workloads} == {"transposed_conv1d"}
    assert all(workload.bias_tensor is None for workload in workloads)
    assert all(workload.bias_shape is None for workload in workloads)
    assert all(workload.conv_config["output_padding"] == [0, 1] for workload in workloads)
    assert all(workload.x_shape == (2, 8, 1, 16) for workload in workloads)
    assert all(workload.weight_shape == (8, 16, 1, 3) for workload in workloads)
    assert all(workload.output_shape == (2, 16, 1, 32) for workload in workloads)


def test_cuda_transposed_conv1d_module_declares_and_calls_cutlass_symbol():
    ir = _cuda_transposed_conv1d_ir(
        "float16",
        in_channels=8,
        out_channels=16,
        width=16,
        stride=2,
        padding=1,
        output_padding=1,
    )
    manifest = build_kernel_manifest(ir, {"name": "cuda", "arch": "sm_86"})
    item = manifest["required_kernels"][0]
    source = render_cuda_module(ir, kernel_manifest=manifest)

    assert f'extern "C" int {item["kernel_symbol"]}(' in source
    assert "dinoml_cutlass_conv_weight_pack_iohw_to_ihwo_float16_v1" in source
    assert f'{item["kernel_symbol"]}(session->cutlass_conv_tmp_n0_activation_nhwc, session->cutlass_conv_tmp_n0_weight_ihwo, session->cutlass_conv_tmp_n0_output_nhwc' in source
    assert "transposed_conv1d CUTLASS Conv activation_pack failed" in source
    assert "transposed_conv1d CUTLASS Conv weight_pack failed" in source
    assert "transposed_conv1d CUTLASS Conv output_unpack failed" in source


def test_cutlass_transposed_conv1d_unit_generation_exports_weight_pack_and_dgrad_kernel():
    ir = _cuda_transposed_conv1d_ir(
        "float16",
        in_channels=8,
        out_channels=16,
        width=16,
        stride=2,
        padding=1,
        output_padding=1,
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


def _cuda_transposed_conv1d_ir(
    dtype: str,
    *,
    batch: int = 2,
    in_channels: int = 4,
    out_channels: int = 6,
    width: int = 8,
    kernel_w: int = 3,
    stride: int = 1,
    padding: int = 1,
    output_padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> dict:
    output_w = (width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1
    return {
        "schema_version": 1,
        "name": "cuda_transposed_conv1d_smoke",
        "inputs": [
            _io("x", [batch, in_channels, width], dtype),
            _io("weight", [in_channels, out_channels, kernel_w], dtype),
        ],
        "constants": [],
        "outputs": [_io("y", [batch, out_channels, output_w], dtype)],
        "nodes": [
            {
                "id": "n0",
                "op": "transposed_conv1d",
                "inputs": ["x", "weight"],
                "outputs": ["y"],
                "attrs": {
                    "stride": [stride],
                    "padding": [padding],
                    "output_padding": [output_padding],
                    "dilation": [dilation],
                    "groups": groups,
                },
            }
        ],
        "tensors": [
            _tensor("x", [batch, in_channels, width], dtype, "input"),
            _tensor("weight", [in_channels, out_channels, kernel_w], dtype, "input"),
            _tensor("y", [batch, out_channels, output_w], dtype, "output"),
        ],
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
    return {
        "name": name,
        "shape": shape,
        "shape_spec": shape,
        "layout": _dense_layout(shape),
        "dtype": dtype,
        "kind": kind,
    }


def _dense_layout(shape: list[int]) -> list[int]:
    stride = 1
    layout = []
    for dim in reversed(shape):
        layout.append(stride)
        stride *= int(dim)
    return list(reversed(layout))
