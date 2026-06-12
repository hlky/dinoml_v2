from __future__ import annotations

from pathlib import Path

from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.kernels.profiling import build_profile_workloads
from dinoml.kernels.providers.ck.conv import ck_conv_static_library_name, ck_conv_used_candidate_plan, render_ck_conv_source
from dinoml.lowering.rocm import render_rocm_module


def test_rocm_transposed_conv1d_cmake_targets_cover_provider_ops():
    cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
    ck_conv_ops_block = cmake.split("set(DINOML_CK_CONV_OPS", 1)[1].split("CACHE STRING", 1)[0]

    assert "transposed_conv1d" in ck_conv_ops_block


def test_rocm_transposed_conv1d_uses_ck_manifest_and_profile_workloads():
    ir = _rocm_transposed_conv1d_ir(
        "float16",
        batch=2,
        in_channels=8,
        out_channels=16,
        width=9,
        stride=2,
        padding=1,
        output_padding=1,
    )
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]
    workloads = build_profile_workloads(ir, manifest)

    assert item["kernel_library"] == "ck_conv"
    assert item["candidate_set_id"] == "ck_transposed_conv1d_float16_identity_v1"
    assert item["kernel_symbol"] == "dinoml_ck_transposed_conv1d_float16_grouped_bwd_data_v1"
    assert item["support_archive"] == ck_conv_static_library_name("transposed_conv1d", "float16")
    assert item["candidate_set"]["epilogue"] == "identity"
    assert item["candidate_set"]["epilogue_config"]["inputs"] == []
    assert item["candidate_set"]["epilogue_config"]["launch_abi"] == "dinoml_ck_transposed_conv2d_v1"
    assert item["candidate_set"]["semantic_layout"] == {"activation": "ncw", "weight": "iow", "output": "ncw"}
    assert item["candidate_set"]["provider_layout"] == {
        "activation": "g_nhw_k_strided",
        "weight": "g_k_c_yx_strided",
        "output": "g_nhw_c_strided",
    }
    assert {workload.kernel_library for workload in workloads} == {"ck_conv"}
    assert {workload.op for workload in workloads} == {"transposed_conv1d"}
    assert {workload.candidate_id for workload in workloads} == {"ck_transposed_conv1d_float16_grouped_bwd_data_v1"}
    assert all(workload.bias_tensor is None for workload in workloads)
    assert all(workload.bias_shape is None for workload in workloads)
    assert all(workload.conv_config["output_padding"] == [1] for workload in workloads)
    assert all(workload.weight_shape == (8, 16, 3) for workload in workloads)


def test_rocm_transposed_conv1d_module_declares_and_calls_ck_symbol():
    ir = _rocm_transposed_conv1d_ir(
        "float16",
        stride=2,
        padding=1,
        output_padding=1,
    )
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    source = render_rocm_module(ir, kernel_manifest=manifest)

    assert 'extern "C" int dinoml_ck_transposed_conv1d_float16_grouped_bwd_data_v1' in source
    assert "const half* weight" in source
    assert (
        "dinoml_ck_transposed_conv1d_float16_grouped_bwd_data_v1("
        "ptr_x, ptr_weight, ptr_y, static_cast<int>(shape_x_0)"
        in source
    )
    assert "shape_y_1 != shape_weight_1" in source
    assert "transposed_conv1d CK Conv launcher failed" in source


def test_ck_conv_unit_generation_supports_transposed_conv1d_base_op():
    manifest = build_kernel_manifest(_rocm_transposed_conv1d_ir("float16"), {"name": "rocm", "arch": "gfx1201"})
    rendered = render_ck_conv_source(
        Path("kernels/rocm/src/ck_conv.hip").read_text(encoding="utf-8"),
        ck_conv_used_candidate_plan(manifest),
    )

    assert "DINOML_CK_TRANSPOSED_CONV2D_EXPORT(transposed_conv1d, float16, half, grouped_bwd_data_v1)" in rendered


def _rocm_transposed_conv1d_ir(
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
        "name": "rocm_transposed_conv1d_smoke",
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
