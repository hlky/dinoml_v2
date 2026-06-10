from __future__ import annotations

from pathlib import Path

import pytest

from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.kernels.profiling import build_profile_workloads
from dinoml.kernels.providers.cutlass.conv import (
    cutlass_conv_candidate_set_id,
    cutlass_conv_static_library_name,
    cutlass_conv_used_candidate_plan,
    render_cutlass_conv_source,
)
from dinoml.lowering.cuda import render_cuda_module


_CONV1D_OPS = ("conv1d_bias", "conv1d_bias_relu", "conv1d_bias_add", "conv1d_bias_add_relu")


def test_cuda_conv1d_cmake_targets_cover_provider_ops():
    cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
    cutlass_conv_ops_block = cmake.split("set(DINOML_CUTLASS_CONV_OPS", 1)[1].split(")", 1)[0]

    for op_name in _CONV1D_OPS:
        assert op_name in cutlass_conv_ops_block


@pytest.mark.parametrize("op_name", _CONV1D_OPS)
def test_cuda_conv1d_uses_cutlass_manifest_and_profile_workloads(op_name: str):
    ir = _cuda_conv1d_ir("float16", op_name=op_name, batch=2, in_channels=8, out_channels=64, width=16)
    manifest = build_kernel_manifest(ir, {"name": "cuda", "arch": "sm_86"})
    item = manifest["required_kernels"][0]
    workloads = build_profile_workloads(ir, manifest)

    assert item["kernel_library"] == "cutlass_conv"
    assert item["support_archive"] == cutlass_conv_static_library_name(op_name, "float16")
    assert item["candidate_set_id"] == cutlass_conv_candidate_set_id(op_name, "float16")
    assert item["candidate_set"]["family"] == "conv1d_fprop"
    assert item["candidate_set"]["semantic_layout"] == {
        "activation": "ncw",
        "weight": "oiw",
        "bias": "o",
        "output": "ncw",
        **({"residual": "ncw"} if op_name in {"conv1d_bias_add", "conv1d_bias_add_relu"} else {}),
    }
    assert item["candidate_set"]["provider_layout"] == {
        "activation": "nwc",
        "weight": "owi",
        "bias": "o",
        "output": "nwc",
        **({"residual": "nwc"} if op_name in {"conv1d_bias_add", "conv1d_bias_add_relu"} else {}),
    }
    conv_plan = item["cutlass_conv_plan"]
    assert conv_plan["kind"] == "cutlass_conv1d_bias_plan"
    assert conv_plan["layout_translation"]["input_pack"] == "ncw_to_nwc_temporary"
    assert conv_plan["layout_translation"]["output_unpack"] == "nwc_to_ncw_temporary"
    assert conv_plan["weight_transform"]["from"] == "oiw"
    assert conv_plan["weight_transform"]["to"] == "owi"
    assert conv_plan["weight_transform"]["pack"] == "oiw_to_owi_temporary"
    if op_name in {"conv1d_bias_add", "conv1d_bias_add_relu"}:
        assert conv_plan["layout_translation"]["residual_pack"] == "ncw_to_nwc_temporary"

    assert {workload.kernel_library for workload in workloads} == {"cutlass_conv"}
    assert {workload.op for workload in workloads} == {op_name}
    assert all(len(workload.x_shape) == 3 for workload in workloads)
    assert all(len(workload.weight_shape) == 3 for workload in workloads)
    assert all(workload.conv_config["stride"] == [2] for workload in workloads)
    assert all(workload.conv_config["padding"] == [1] for workload in workloads)
    assert all(workload.conv_config["dilation"] == [1] for workload in workloads)


def test_cuda_conv1d_module_declares_and_calls_cutlass_symbol():
    ir = _cuda_conv1d_ir("float16", op_name="conv1d_bias_add_relu", in_channels=8, out_channels=64, width=16)
    manifest = build_kernel_manifest(ir, {"name": "cuda", "arch": "sm_86"})
    item = manifest["required_kernels"][0]
    source = render_cuda_module(ir, kernel_manifest=manifest)

    assert 'extern "C" int dinoml_cutlass_conv1d_input_pack_ncw_to_nwc_float16_v1' in source
    assert 'extern "C" int dinoml_cutlass_conv1d_weight_pack_oiw_to_owi_float16_v1' in source
    assert 'extern "C" int dinoml_cutlass_conv1d_output_unpack_nwc_to_ncw_float16_v1' in source
    assert f'extern "C" int {item["kernel_symbol"]}(' in source
    assert "conv1d_bias_add_relu CUTLASS Conv activation_pack failed" in source
    assert "conv1d_bias_add_relu CUTLASS Conv weight_pack failed" in source
    assert "conv1d_bias_add_relu CUTLASS Conv output_unpack failed" in source


def test_cutlass_conv1d_unit_generation_exports_weight_pack_and_kernel():
    ir = _cuda_conv1d_ir("float16", op_name="conv1d_bias_add_relu", in_channels=8, out_channels=64, width=16)
    manifest = build_kernel_manifest(ir, {"name": "cuda", "arch": "sm_86"})
    item = manifest["required_kernels"][0]
    used_candidate_plan = cutlass_conv_used_candidate_plan(manifest)
    rendered = render_cutlass_conv_source(
        Path("kernels/cuda/src/cutlass_conv.cu").read_text(encoding="utf-8"),
        used_candidate_plan,
    )

    assert "DINOML_CUTLASS_CONV1D_NCW_TO_NWC_EXPORT" in rendered
    assert "DINOML_CUTLASS_CONV1D_OIW_TO_OWI_EXPORT" in rendered
    assert "DINOML_CUTLASS_CONV1D_NWC_TO_NCW_EXPORT" in rendered
    assert "DINOML_CUTLASS_CONV1D_BIAS_ADD_RELU_EXPORT" in rendered
    assert "dinoml_cutlass_conv1d_weight_pack_oiw_to_owi_float16_v1" in rendered
    assert item["kernel_symbol"] in rendered
    assert item["profiler_symbol"] in rendered


@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_cuda_conv1d_unaligned_tensorop_shapes_fail_clearly(dtype: str):
    ir = _cuda_conv1d_ir(dtype, op_name="conv1d_bias", in_channels=3, out_channels=5, width=9)

    with pytest.raises(ValueError, match="CUTLASS Conv manifest candidate selection found no candidate compatible with the transform plan"):
        build_kernel_manifest(ir, {"name": "cuda", "arch": "sm_86"})


def _cuda_conv1d_ir(
    dtype: str,
    *,
    op_name: str = "conv1d_bias",
    batch: int = 2,
    in_channels: int = 4,
    out_channels: int = 6,
    width: int = 8,
    kernel_w: int = 3,
    stride: int = 2,
    padding: int = 1,
    dilation: int = 1,
    groups: int = 1,
) -> dict:
    out_width = (width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    has_residual = op_name in {"conv1d_bias_add", "conv1d_bias_add_relu"}
    tensors = [
        _tensor("x", [batch, in_channels, width], dtype, "input"),
        _tensor("weight", [out_channels, in_channels, kernel_w], dtype, "input"),
        _tensor("bias", [out_channels], dtype, "input"),
        *([_tensor("residual", [batch, out_channels, out_width], dtype, "input")] if has_residual else []),
        _tensor("y", [batch, out_channels, out_width], dtype, "output"),
    ]
    inputs = [
        _io("x", [batch, in_channels, width], dtype),
        _io("weight", [out_channels, in_channels, kernel_w], dtype),
        _io("bias", [out_channels], dtype),
        *([_io("residual", [batch, out_channels, out_width], dtype)] if has_residual else []),
    ]
    node_inputs = ["x", "weight", "bias", *(["residual"] if has_residual else [])]
    return {
        "schema_version": 1,
        "name": f"cuda_{op_name}_smoke",
        "inputs": inputs,
        "constants": [],
        "outputs": [_io("y", [batch, out_channels, out_width], dtype)],
        "nodes": [
            {
                "id": "n0",
                "op": op_name,
                "inputs": node_inputs,
                "outputs": ["y"],
                "attrs": {
                    "stride": [stride],
                    "padding": [padding],
                    "dilation": [dilation],
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
