from __future__ import annotations

import numpy as np

import dinoml as dml
from dinoml import compiler as dml_compiler
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.kernels.profiling import _CkConvProfiler, build_profile_workloads
from dinoml.kernels.providers.ck.conv import ck_conv_candidate_set_id
from dinoml.lowering.rocm import render_rocm_module


def test_rocm_conv3d_bias_grouped_manifest_and_module_source():
    ir = _rocm_conv3d_ir(
        "float16",
        input_shape=[1, 4, 5, 6, 7],
        weight_shape=[4, 1, 3, 3, 3],
        bias_shape=[4],
        output_shape=[1, 4, 5, 6, 7],
        groups=4,
    )
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]
    source = render_rocm_module(ir, kernel_manifest=manifest)

    assert item["kernel_library"] == "ck_conv"
    assert item["candidate_set_id"] == ck_conv_candidate_set_id("conv3d_bias", "float16")
    assert item["candidate_set"]["family"] == "conv3d_fprop"
    assert item["candidate_set"]["semantic_layout"] == {
        "activation": "ncdhw",
        "weight": "oidhw",
        "bias": "o",
        "output": "ncdhw",
    }
    assert item["candidate_set"]["provider_layout"] == {
        "activation": "g_ndhw_c_strided",
        "weight": "g_k_zyx_c_strided",
        "bias": "g_k",
        "output": "g_ndhw_k_strided",
    }
    assert 'extern "C" int dinoml_ck_conv3d_bias_float16_' in source
    assert "int groups," in source
    assert "shape_weight_1 != (shape_x_1 / 4)" in source
    assert "conv3d_bias CK Conv launcher failed" in source


def test_rocm_conv3d_bias_grouped_profile_workloads_are_not_blocked():
    ir = _rocm_conv3d_ir(
        "float16",
        input_shape=[1, 4, 5, 6, 7],
        weight_shape=[4, 1, 3, 3, 3],
        bias_shape=[4],
        output_shape=[1, 4, 5, 6, 7],
        groups=4,
    )
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]

    workloads = build_profile_workloads(ir, manifest)

    assert "profile_blocked_reason" not in item
    assert workloads
    assert workloads[0].op == "conv3d_bias"
    assert workloads[0].conv_config["groups"] == 4


def test_ck_conv3d_profiler_passes_rank3_problem_metadata():
    ir = _rocm_conv3d_ir(
        "float16",
        input_shape=[1, 4, 5, 6, 7],
        weight_shape=[4, 1, 3, 3, 3],
        bias_shape=[4],
        output_shape=[1, 4, 5, 6, 7],
        groups=4,
    )
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    workloads = build_profile_workloads(ir, manifest)
    calls = []

    class FakeModule:
        def profile_conv(self, **kwargs):
            calls.append(kwargs)
            symbols = list(kwargs["profiler_symbols"])
            return [
                {"profiler_symbol": symbols[0], "samples_ms": [0.11, 0.10], "workspace_nbytes": 16, "ok": True},
            ]

    profiler = _CkConvProfiler({(workloads[0].op, workloads[0].dtype): FakeModule()})
    rows = profiler.profile_problem(workloads, iterations=9, repeats=2)

    assert calls[0]["profiler_symbols"] == [workload.profiler_symbol for workload in workloads]
    assert calls[0]["spatial_rank"] == 3
    assert calls[0]["in_depth"] == 5
    assert calls[0]["kernel_d"] == 3
    assert calls[0]["out_depth"] == 5
    assert calls[0]["groups"] == 4
    assert rows[0]["samples_ms"] == [0.11, 0.10]


def test_rocm_conv3d_constant_weight_prepack_is_recorded_in_metadata_and_launch_source(tmp_path):
    spec, weight_value = _trace_rocm_constant_weight_conv3d_spec("float16")
    ir = dml_compiler._write_constants(tmp_path, spec.ir, spec.constants, target=dml.Target("rocm"))
    constants_by_name = {constant["name"]: constant for constant in ir["constants"]}
    weight_constant = constants_by_name["weight"]

    assert weight_constant["storage"]["kind"] == "ck_conv3d_weight"
    assert weight_constant["storage"]["logical_layout"] == "oidhw"
    assert weight_constant["storage"]["storage_layout"] == "kzyxc"

    stored = np.frombuffer((tmp_path / "constants.bin").read_bytes(), dtype=np.float16)
    stored = stored[
        weight_constant["offset"] // np.dtype(np.float16).itemsize : (
            weight_constant["offset"] + weight_constant["nbytes"]
        )
        // np.dtype(np.float16).itemsize
    ].reshape(weight_value.shape[0], weight_value.shape[2], weight_value.shape[3], weight_value.shape[4], weight_value.shape[1])
    expected = np.ascontiguousarray(np.transpose(weight_value, (0, 2, 3, 4, 1)))
    np.testing.assert_array_equal(stored, expected)

    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = next(
        kernel
        for kernel in manifest["required_kernels"]
        if kernel["op"] == "conv3d_bias" and kernel["kernel_library"] == "ck_conv"
    )

    assert item["ck_conv_runtime_plan"] == {
        "node_id": "n0",
        "weight_pack_mode": "constants_bin_prepacked_kzyxc",
        "constant_tensor": "weight",
    }

    source = render_rocm_module(ir, kernel_manifest=manifest)

    assert "dinoml_ck_conv3d_bias_float16_" in source
    assert "ptr_x, ptr_weight, ptr_bias, ptr_t0" in source
    assert ", 1, 1, session->stream" in source


def _trace_rocm_constant_weight_conv3d_spec(dtype: str) -> tuple[dml.ModelSpec, np.ndarray]:
    weight_value = (np.arange(5 * 3 * 2 * 3 * 2, dtype=np.float32).reshape(5, 3, 2, 3, 2) / 29.0).astype(np.float16)
    bias_value = (np.arange(5, dtype=np.float32) / 19.0).astype(np.float16)

    class TinyConstConv3d(dml.Module):
        def __init__(self):
            self.weight = dml.Parameter(list(weight_value.shape), dtype=dtype, value=weight_value)
            self.bias = dml.Parameter([5], dtype=dtype, value=bias_value)

        def forward(self, x):
            y = dml.ops.conv3d_bias(x, self.weight, self.bias, stride=(1, 1, 1), padding=(0, 1, 0))
            return dml.ops.output(y, "y")

    spec = dml.trace(
        TinyConstConv3d(),
        {"x": dml.TensorSpec([2, 3, 4, 6, 5], dtype)},
        name=f"rocm_conv3d_constant_weight_{dtype}",
    )
    return spec, weight_value


def _rocm_conv3d_ir(
    dtype: str,
    *,
    input_shape: list[int],
    weight_shape: list[int],
    bias_shape: list[int],
    output_shape: list[int],
    groups: int,
) -> dict:
    return {
        "schema_version": 1,
        "name": "rocm_conv3d_scaffold",
        "inputs": [
            _io("x", input_shape, dtype),
            _io("weight", weight_shape, dtype),
            _io("bias", bias_shape, dtype),
        ],
        "constants": [],
        "outputs": [_io("y", output_shape, dtype)],
        "nodes": [
            {
                "id": "n0",
                "op": "conv3d_bias",
                "inputs": ["x", "weight", "bias"],
                "outputs": ["y"],
                "attrs": {
                    "stride": [1, 1, 1],
                    "padding": [1, 1, 1],
                    "dilation": [1, 1, 1],
                    "groups": groups,
                },
            }
        ],
        "tensors": [
            _tensor("x", input_shape, dtype, "input"),
            _tensor("weight", weight_shape, dtype, "input"),
            _tensor("bias", bias_shape, dtype, "input"),
            _tensor("y", output_shape, dtype, "output"),
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
    nbytes = (2 if dtype in {"float16", "bfloat16"} else 4)
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
