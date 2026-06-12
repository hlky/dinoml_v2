from __future__ import annotations

from pathlib import Path

from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.kernels.profiling import build_profile_workloads
from dinoml.kernels.providers.cutlass.conv import (
    cutlass_conv_candidate_set_id,
    cutlass_conv_static_library_name,
    cutlass_conv_used_candidate_plan,
    render_cutlass_conv_source,
)
from dinoml.lowering.cuda import render_cuda_module


def test_cuda_conv3d_cmake_targets_cover_provider_ops():
    cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
    cutlass_conv_ops_block = cmake.split("set(DINOML_CUTLASS_CONV_OPS", 1)[1].split(")", 1)[0]

    assert "conv3d_bias" in cutlass_conv_ops_block


def test_cuda_conv3d_uses_cutlass_manifest_and_profile_workloads():
    ir = _cuda_conv3d_ir("float16", batch=1, in_channels=4, out_channels=4, depth=5, height=6, width=7, groups=4)
    manifest = build_kernel_manifest(ir, {"name": "cuda", "arch": "sm_86"})
    item = manifest["required_kernels"][0]
    workloads = build_profile_workloads(ir, manifest)

    assert item["kernel_library"] == "cutlass_conv"
    assert item["support_archive"] == cutlass_conv_static_library_name("conv3d_bias", "float16")
    assert item["candidate_set_id"] == cutlass_conv_candidate_set_id("conv3d_bias", "float16")
    assert item["candidate_set"]["family"] == "conv3d_fprop"
    assert item["candidate_set"]["semantic_layout"] == {
        "activation": "ncdhw",
        "weight": "oidhw",
        "bias": "o",
        "output": "ncdhw",
    }
    assert item["candidate_set"]["provider_layout"] == {
        "activation": "ndhwc",
        "weight": "ktrsc",
        "bias": "o",
        "output": "ndhwc",
    }
    assert item["candidate_set"]["supported_groups"] == ["any_positive_int"]
    conv_plan = item["cutlass_conv_plan"]
    assert conv_plan["kind"] == "cutlass_conv3d_bias_plan"
    assert conv_plan["source_op"] == "depthwise_conv3d"
    assert conv_plan["bias_mode"] == "explicit_zero_constant"
    assert conv_plan["conv_config"]["groups"] == 4
    assert conv_plan["runtime"]["provider_groups"] == 1
    assert conv_plan["layout_translation"]["input_pack"] == "ncdhw_to_ndhwc_temporary"
    assert conv_plan["layout_translation"]["output_unpack"] == "ndhwc_to_ncdhw_temporary"
    assert conv_plan["weight_transform"]["from"] == "oidhw"
    assert conv_plan["weight_transform"]["to"] == "ktrsc"
    assert conv_plan["weight_transform"]["pack"] == "depthwise_oidhw_to_ktrsc_temporary"

    assert {workload.kernel_library for workload in workloads} == {"cutlass_conv"}
    assert {workload.op for workload in workloads} == {"conv3d_bias"}
    assert all(len(workload.x_shape) == 5 for workload in workloads)
    assert all(len(workload.weight_shape) == 5 for workload in workloads)
    assert all(workload.conv_config["groups"] == 4 for workload in workloads)


def test_cuda_conv3d_module_declares_and_calls_cutlass_symbol():
    ir = _cuda_conv3d_ir("float16", batch=1, in_channels=4, out_channels=4, depth=5, height=6, width=7, groups=4)
    manifest = build_kernel_manifest(ir, {"name": "cuda", "arch": "sm_86"})
    item = manifest["required_kernels"][0]
    source = render_cuda_module(ir, kernel_manifest=manifest)

    assert 'extern "C" int dinoml_cutlass_conv3d_input_pack_ncdhw_to_ndhwc_float16_v1' in source
    assert 'extern "C" int dinoml_cutlass_conv3d_weight_pack_depthwise_oidhw_to_ktrsc_float16_v1' in source
    assert 'extern "C" int dinoml_cutlass_conv3d_output_unpack_ndhwc_to_ncdhw_float16_v1' in source
    assert f'extern "C" int {item["kernel_symbol"]}(' in source
    assert "conv3d_bias CUTLASS Conv activation_pack failed" in source
    assert "conv3d_bias CUTLASS Conv weight_pack failed" in source
    assert "conv3d_bias CUTLASS Conv output_unpack failed" in source


def test_cuda_conv3d_module_uses_rank5_runtime_shape_indices():
    ir = _cuda_conv3d_ir(
        "float32",
        batch=2,
        in_channels=3,
        out_channels=5,
        depth=5,
        height=6,
        width=7,
        groups=1,
        kernel_w=2,
        stride=(2, 1, 1),
        padding=(1, 1, 0),
    )
    manifest = build_kernel_manifest(ir, {"name": "cuda", "arch": "sm_89"})
    item = manifest["required_kernels"][0]
    source = render_cuda_module(ir, kernel_manifest=manifest)

    expected_call = (
        f"int status_n0_provider_launch = {item['kernel_symbol']}("
        "session->cutlass_conv_tmp_n0_activation_ndhwc, "
        "session->cutlass_conv_tmp_n0_weight_ktrsc, "
        "ptr_bias, "
        "session->cutlass_conv_tmp_n0_output_ndhwc, "
        "static_cast<int>(shape_x_0), "
        "static_cast<int>(shape_x_2), "
        "static_cast<int>(shape_x_3), "
        "static_cast<int>(shape_x_4), "
        "static_cast<int>(shape_x_1), "
        "static_cast<int>(shape_y_2), "
        "static_cast<int>(shape_y_3), "
        "static_cast<int>(shape_y_4), "
        "static_cast<int>(shape_y_1), "
        "static_cast<int>(shape_weight_2), "
        "static_cast<int>(shape_weight_3), "
        "static_cast<int>(shape_weight_4)"
    )
    assert expected_call in source


def test_cutlass_conv3d_unit_generation_exports_weight_pack_and_kernel():
    ir = _cuda_conv3d_ir("float16", batch=1, in_channels=4, out_channels=4, depth=5, height=6, width=7, groups=4)
    manifest = build_kernel_manifest(ir, {"name": "cuda", "arch": "sm_86"})
    item = manifest["required_kernels"][0]
    used_candidate_plan = cutlass_conv_used_candidate_plan(manifest)
    rendered = render_cutlass_conv_source(
        Path("kernels/cuda/src/cutlass_conv.cu").read_text(encoding="utf-8"),
        used_candidate_plan,
    )

    assert "DINOML_CUTLASS_CONV3D_NCDHW_TO_NDHWC_EXPORT" in rendered
    assert "DINOML_CUTLASS_CONV3D_NDHWC_TO_NCDHW_EXPORT" in rendered
    assert "DINOML_CUTLASS_CONV3D_DEPTHWISE_OIDHW_TO_KTRSC_EXPORT" in rendered
    assert "dinoml_cutlass_conv3d_weight_pack_depthwise_oidhw_to_ktrsc_float16_v1" in rendered
    assert item["kernel_symbol"] in rendered
    assert item["profiler_symbol"] in rendered


def _cuda_conv3d_ir(
    dtype: str,
    *,
    batch: int,
    in_channels: int,
    out_channels: int,
    depth: int,
    height: int,
    width: int,
    groups: int,
    kernel_d: int = 3,
    kernel_h: int = 3,
    kernel_w: int = 3,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (1, 1, 1),
    dilation: tuple[int, int, int] = (1, 1, 1),
) -> dict:
    out_d = (depth + 2 * padding[0] - dilation[0] * (kernel_d - 1) - 1) // stride[0] + 1
    out_h = (height + 2 * padding[1] - dilation[1] * (kernel_h - 1) - 1) // stride[1] + 1
    out_w = (width + 2 * padding[2] - dilation[2] * (kernel_w - 1) - 1) // stride[2] + 1
    weight_i = in_channels // groups
    return {
        "schema_version": 1,
        "name": "cuda_conv3d_smoke",
        "inputs": [
            _io("x", [batch, in_channels, depth, height, width], dtype),
            _io("weight", [out_channels, weight_i, kernel_d, kernel_h, kernel_w], dtype),
            _io("bias", [out_channels], dtype),
        ],
        "constants": [],
        "outputs": [_io("y", [batch, out_channels, out_d, out_h, out_w], dtype)],
        "nodes": [
            {
                "id": "n0",
                "op": "conv3d_bias",
                "inputs": ["x", "weight", "bias"],
                "outputs": ["y"],
                "attrs": {
                    "stride": list(stride),
                    "padding": list(padding),
                    "dilation": list(dilation),
                    "groups": groups,
                    "source_op": "depthwise_conv3d",
                    "bias_mode": "explicit_zero_constant",
                },
            }
        ],
        "tensors": [
            _tensor("x", [batch, in_channels, depth, height, width], dtype, "input"),
            _tensor("weight", [out_channels, weight_i, kernel_d, kernel_h, kernel_w], dtype, "input"),
            _tensor("bias", [out_channels], dtype, "input"),
            _tensor("y", [batch, out_channels, out_d, out_h, out_w], dtype, "output"),
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
