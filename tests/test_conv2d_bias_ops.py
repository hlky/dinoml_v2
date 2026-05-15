import ctypes
import shutil
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import dinoml as dml
from dinoml import runtime
from dinoml.backends.cutlass import ensure_cutlass_conv_support_scaffold
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import array_from_storage, array_to_storage, read_json
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.kernels.providers.cutlass.conv import (
    cutlass_conv_candidate_set,
    cutlass_conv_candidates,
    cutlass_conv_input_pack_symbol,
    cutlass_conv_layout_plan,
    cutlass_conv_output_unpack_symbol,
    cutlass_conv_used_candidate_plan,
    cutlass_conv_weight_pack_symbol,
)
from dinoml.lowering.cuda import render_cuda_module
from dinoml.lowering.ops.conv import render_scaffold_wrapper_source, render_scaffold_wrapper_stages
from dinoml.kernels.profiling import build_profile_workloads, profile_artifact
from dinoml.passes import validate_ir
from dinoml.passes.validation import ValidationError
from dinoml.shapes import Dim


class Conv2dBiasModule(dml.Module):
    def __init__(self, stride=1, padding=0, dilation=1, groups=1):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x, weight, bias):
        return dml.ops.output(
            dml.ops.conv2d_bias(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            ),
            "out",
        )


class TwoConv2dBiasModule(dml.Module):
    def forward(self, x0, weight0, bias0, x1, weight1, bias1):
        y0 = dml.ops.output(dml.ops.conv2d_bias(x0, weight0, bias0, stride=(2, 1), padding=(1, 0)), "out0")
        y1 = dml.ops.output(dml.ops.conv2d_bias(x1, weight1, bias1, stride=(2, 1), padding=(1, 0)), "out1")
        return y0, y1


def _trace_conv2d_bias(
    dtype="float32",
    x_shape=(2, 3, 7, 8),
    weight_shape=(4, 3, 3, 2),
    bias_shape=(4,),
    stride=(2, 1),
    padding=(1, 0),
    dilation=(1, 2),
    groups=1,
):
    return dml.trace(
        Conv2dBiasModule(stride=stride, padding=padding, dilation=dilation, groups=groups),
        inputs={
            "x": dml.TensorSpec(x_shape, dtype),
            "weight": dml.TensorSpec(weight_shape, dtype),
            "bias": dml.TensorSpec(bias_shape, dtype),
        },
        name=f"conv2d_bias_{dtype}",
    )


def _trace_two_conv2d_bias(dtype="float16"):
    return dml.trace(
        TwoConv2dBiasModule(),
        inputs={
            "x0": dml.TensorSpec((2, 3, 7, 8), dtype),
            "weight0": dml.TensorSpec((4, 3, 3, 2), dtype),
            "bias0": dml.TensorSpec((4,), dtype),
            "x1": dml.TensorSpec((2, 3, 7, 8), dtype),
            "weight1": dml.TensorSpec((4, 3, 3, 2), dtype),
            "bias1": dml.TensorSpec((4,), dtype),
        },
        name=f"two_conv2d_bias_{dtype}",
    )


def _input(shape, dtype, start, stop):
    value = np.linspace(start, stop, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    if dtype == "float16":
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return value


def _storage_roundtrip(value, dtype):
    if dtype == "float16":
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return np.asarray(value, dtype=np.float32)


def _torch_conv2d_bias_reference(x, weight, bias, *, stride, padding, dilation):
    return (
        F.conv2d(
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(weight, dtype=np.float32)),
            torch.from_numpy(np.asarray(bias, dtype=np.float32)),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )
        .detach()
        .cpu()
        .numpy()
    )


def _expected_transform_helper_exports(layout_plan, *, status):
    dtype = layout_plan["dtype"]
    return [
        {
            "kind": "transform_helper",
            "symbol": layout_plan["layout_translation"]["input_pack_symbol"],
            "helper_abi": "dinoml_cutlass_layout_transform_v1",
            "tensor_role": "activation",
            "transform": "nchw_to_nhwc_temporary",
            "dtype": dtype,
            "layout_from": "nchw",
            "layout_to": "nhwc",
            "shape_order": ["n", "c", "h", "w"],
            "status": status,
            "success_return_code": 0,
        },
        {
            "kind": "transform_helper",
            "symbol": layout_plan["layout_translation"]["output_unpack_symbol"],
            "helper_abi": "dinoml_cutlass_layout_transform_v1",
            "tensor_role": "output",
            "transform": "nhwc_to_nchw_temporary",
            "dtype": dtype,
            "layout_from": "nhwc",
            "layout_to": "nchw",
            "shape_order": ["n", "c", "h", "w"],
            "status": status,
            "success_return_code": 0,
        },
        {
            "kind": "transform_helper",
            "symbol": layout_plan["weight_transform"]["pack_symbol"],
            "helper_abi": "dinoml_cutlass_layout_transform_v1",
            "tensor_role": "weight",
            "transform": "oihw_to_ohwi_temporary",
            "dtype": dtype,
            "layout_from": "oihw",
            "layout_to": "ohwi",
            "shape_order": ["o", "i", "h", "w"],
            "status": status,
            "success_return_code": 0,
        },
    ]


def test_conv2d_bias_frontend_ir_preserves_nchw_oihw_attrs_and_dtype():
    spec = _trace_conv2d_bias(
        "float32",
        x_shape=(2, 3, 7, 8),
        weight_shape=(4, 3, 3, 2),
        bias_shape=(4,),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 2),
    )

    assert spec.ir["outputs"][0]["shape"] == [2, 4, 4, 6]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 4, 4, 6]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "conv2d_bias"
    assert node["inputs"] == ["x", "weight", "bias"]
    assert node["attrs"] == {
        "stride": [2, 1],
        "padding": [1, 0],
        "dilation": [1, 2],
        "groups": 1,
    }


@pytest.mark.parametrize("dtype,atol,rtol", [("float32", 1e-6, 1e-6), ("float16", 1e-3, 1e-3)])
def test_cpu_reference_conv2d_bias_matches_torch(dtype, atol, rtol):
    spec = _trace_conv2d_bias(
        dtype,
        x_shape=(2, 3, 6, 7),
        weight_shape=(4, 3, 2, 3),
        bias_shape=(4,),
        stride=(1, 2),
        padding=(1, 1),
        dilation=(2, 1),
    )
    x = _input((2, 3, 6, 7), dtype, -1.5, 2.5)
    weight = _input((4, 3, 2, 3), dtype, -0.75, 1.25)
    bias = _input((4,), dtype, -0.5, 0.5)

    actual = execute_cpu(spec, {"x": x, "weight": weight, "bias": bias})["out"]

    expected = _storage_roundtrip(
        _torch_conv2d_bias_reference(
            x,
            weight,
            bias,
            stride=(1, 2),
            padding=(1, 1),
            dilation=(2, 1),
        ),
        dtype,
    )
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype,atol,rtol", [("float32", 1e-6, 1e-6), ("float16", 1e-3, 1e-3)])
def test_cpu_artifact_runs_generated_naive_conv2d_bias(dtype, atol, rtol, tmp_path, monkeypatch):
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec = _trace_conv2d_bias(
        dtype,
        x_shape=(2, 3, 6, 7),
        weight_shape=(4, 3, 2, 3),
        bias_shape=(4,),
        stride=(1, 2),
        padding=(1, 1),
        dilation=(2, 1),
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"conv2d_bias_{dtype}_cpu.dinoml")

    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "static int conv2d_bias_" in generated

    x = _input((2, 3, 6, 7), dtype, -1.5, 2.5)
    weight = _input((4, 3, 2, 3), dtype, -0.75, 1.25)
    bias = _input((4,), dtype, -0.5, 0.5)
    expected = _storage_roundtrip(
        _torch_conv2d_bias_reference(
            x,
            weight,
            bias,
            stride=(1, 2),
            padding=(1, 1),
            dilation=(2, 1),
        ),
        dtype,
    )

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy({"x": x, "weight": weight, "bias": bias})["out"]
    finally:
        session.close()
        module.close()

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_conv2d_bias_frontend_rejects_dynamic_shapes_bad_ranks_dtype_and_groups():
    class DynamicConv(dml.Module):
        def forward(self, x, weight, bias):
            return dml.ops.conv2d_bias(x, weight, bias)

    with pytest.raises(ValueError, match="static activation, weight, and bias shapes"):
        dml.trace(
            DynamicConv(),
            inputs={
                "x": dml.TensorSpec([1, 3, Dim("h", 4, 8), 8], "float32"),
                "weight": dml.TensorSpec([4, 3, 3, 3], "float32"),
                "bias": dml.TensorSpec([4], "float32"),
            },
        )
    with pytest.raises(ValueError, match="rank-4 NCHW activation"):
        _trace_conv2d_bias("float32", x_shape=(2, 3, 7))
    with pytest.raises(ValueError, match="rank-4 OIHW weight"):
        _trace_conv2d_bias("float32", weight_shape=(4, 3, 3))
    with pytest.raises(ValueError, match="rank-1 bias"):
        _trace_conv2d_bias("float32", bias_shape=(4, 1))
    with pytest.raises(ValueError, match="does not support dtype bfloat16"):
        _trace_conv2d_bias("bfloat16")
    with pytest.raises(NotImplementedError, match="groups=1 only"):
        _trace_conv2d_bias("float32", groups=2)
    with pytest.raises(ValueError, match="positive integers"):
        _trace_conv2d_bias("float32", stride=(1, 0))
    with pytest.raises(ValueError, match="non-negative integers"):
        _trace_conv2d_bias("float32", padding=(0, -1))
    with pytest.raises(ValueError, match="positive integers"):
        _trace_conv2d_bias("float32", dilation=(1, 0))
    with pytest.raises(ValueError, match="must match activation channels"):
        _trace_conv2d_bias("float32", x_shape=(2, 3, 7, 8), weight_shape=(4, 2, 3, 2))
    with pytest.raises(ValueError, match="bias length must match weight output channels"):
        _trace_conv2d_bias("float32", weight_shape=(4, 3, 3, 2), bias_shape=(3,))
    with pytest.raises(ValueError, match="output height must be positive"):
        _trace_conv2d_bias("float32", x_shape=(1, 3, 3, 8), weight_shape=(4, 3, 5, 2), dilation=(2, 1))


def test_conv2d_bias_validation_rejects_dynamic_shape_bad_groups_shape_and_dtype():
    spec = _trace_conv2d_bias("float32")
    spec.ir["inputs"][0]["shape_spec"] = [2, 3, Dim("h", 4, 7).to_json(), 8]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [2, 3, Dim("h", 4, 7).to_json(), 8]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace_conv2d_bias("float32")
    spec.ir["nodes"][0]["attrs"]["groups"] = 2
    with pytest.raises(ValidationError, match="groups=1 only"):
        validate_ir(spec.ir)

    spec = _trace_conv2d_bias("float32")
    spec.ir["outputs"][0]["shape"] = [2, 4, 4, 5]
    spec.ir["outputs"][0]["shape_spec"] = [2, 4, 4, 5]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 4, 4, 5]
    output_tensor["shape_spec"] = [2, 4, 4, 5]
    output_tensor["layout"]["strides"] = [80, 20, 5, 1]
    with pytest.raises(ValidationError, match=r"expected \[2, 4, 4, 6\]"):
        validate_ir(spec.ir)

    spec = _trace_conv2d_bias("float32")
    spec.ir["outputs"][0]["dtype"] = "float16"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "float16"
    with pytest.raises(ValidationError, match="expected float32"):
        validate_ir(spec.ir)


def test_cutlass_conv2d_bias_scaffold_records_layout_transform_metadata(tmp_path):
    spec = _trace_conv2d_bias(
        "float16",
        x_shape=(2, 3, 7, 8),
        weight_shape=(4, 3, 3, 2),
        bias_shape=(4,),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 2),
    )
    target = {"name": "cuda", "arch": "sm_86", "no_tf32": True}
    node = spec.ir["nodes"][0]
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}

    candidate_set = cutlass_conv_candidate_set("conv2d_bias", "float16", target=target)
    candidates = cutlass_conv_candidates("conv2d_bias", "float16", target=target)
    assert candidate_set["status"] == "bounded_runtime"
    assert candidate_set["profiler_status"] == "bounded_runtime_profiler"
    assert candidate_set["semantic_layout"] == {
        "activation": "nchw",
        "weight": "oihw",
        "bias": "o",
        "output": "nchw",
    }
    assert candidate_set["provider_layout"] == {
        "activation": "nhwc",
        "weight": "ohwi",
        "bias": "o",
        "output": "nhwc",
    }
    assert candidates[0]["status"] == "bounded_runtime"
    assert candidates[0]["profiler_status"] == "bounded_runtime_profiler"
    assert candidates[0]["cutlass"]["opclass"] == "simt"
    assert candidates[0]["cutlass"]["kind"] == "implicit_gemm_runtime_launcher"
    assert candidates[1]["status"] == "bounded_runtime"
    assert candidates[1]["cutlass"]["opclass"] == "tensorop"
    assert candidates[1]["cutlass"]["iterator_algorithm"] == "few_channels"
    assert candidates[1]["selection_predicate"] == {
        "kind": "semantic_input_channels",
        "input_channels": 3,
        "dtype": "float16",
        "groups": 1,
        "requires_layout_translation": "nchw_oihw_to_nhwc_ohwi",
        "padding_policy": "none",
    }
    assert [candidate["cutlass"]["iterator_algorithm"] for candidate in candidates] == [
        "analytic",
        "few_channels",
        "fixed_channels",
        "fixed_channels",
        "optimized",
    ]
    assert [candidate["selection_predicate"].get("input_channels") for candidate in candidates[2:4]] == [4, 8]
    assert [candidate["cutlass"]["align_a"] for candidate in candidates[2:4]] == [4, 8]
    assert [candidate["cutlass"]["align_b"] for candidate in candidates[2:4]] == [4, 8]
    assert [candidate["cutlass"]["stages"] for candidate in candidates[2:4]] == [3, 3]
    optimized = candidates[4]
    assert optimized["selection_predicate"] == {
        "kind": "natural_alignment",
        "dtype": "float16",
        "groups": 1,
        "min_input_channels": 16,
        "input_channels_multiple": 8,
        "output_channels_multiple": 8,
        "requires_layout_translation": "nchw_oihw_to_nhwc_ohwi",
        "padding_policy": "none",
    }
    assert optimized["cutlass"]["align_a"] == 8
    assert optimized["cutlass"]["align_b"] == 8
    assert optimized["cutlass"]["stages"] == 3
    selected = candidates[1]

    layout_plan = cutlass_conv_layout_plan(node, tensor_map=tensor_map)
    assert layout_plan["status"] == "bounded_runtime"
    assert layout_plan["runtime"]["launcher"] == "cutlass_implicit_gemm_conv2d_fprop_bias"
    assert layout_plan["profiler_status"] == "bounded_runtime_profiler"
    assert "profiler_blocked_reason" not in layout_plan
    assert layout_plan["semantic_layout"]["activation"] == "nchw"
    assert layout_plan["provider_layout"]["activation"] == "nhwc"
    assert layout_plan["layout_translation"]["input_pack"] == "nchw_to_nhwc_temporary"
    assert layout_plan["layout_translation"]["output_unpack"] == "nhwc_to_nchw_temporary"
    assert layout_plan["layout_translation"]["input_pack_symbol"] == cutlass_conv_input_pack_symbol("float16")
    assert layout_plan["layout_translation"]["output_unpack_symbol"] == cutlass_conv_output_unpack_symbol("float16")
    assert layout_plan["weight_transform"]["from"] == "oihw"
    assert layout_plan["weight_transform"]["to"] == "ohwi"
    assert layout_plan["weight_transform"]["pack_symbol"] == cutlass_conv_weight_pack_symbol("float16")
    assert layout_plan["weight_transform"]["channel_pad_multiple"] == 1

    required = {
        "op": "conv2d_bias",
        "node_id": node["id"],
        "kernel_symbol": selected["kernel_symbol"],
        "kernel_library": "cutlass_conv",
        "profiler_symbol": selected["profiler_symbol"],
        "selected_candidate_id": selected["candidate_id"],
        "candidates": candidates,
        "candidate_set_id": candidate_set["candidate_set_id"],
        "candidate_set_key": candidate_set["candidate_set_key"],
        "candidate_set": candidate_set,
        "cutlass_conv_plan": layout_plan,
    }
    manifest = {
        "target": target,
        "cache_key": "test-conv-cache-key",
        "support_cache_key": "test-conv-support-key",
        "required_kernels": [required],
    }
    used_plan = cutlass_conv_used_candidate_plan(manifest)
    assert used_plan["library_name"] == "cutlass_conv"
    assert used_plan["candidates"] == candidates
    assert used_plan["candidate_config_keys"] == [candidate["candidate_config_key"] for candidate in candidates]
    assert used_plan["entries"][0]["cutlass_conv_plan"] == layout_plan
    assert used_plan["entries"][0]["candidates"] == candidates
    assert used_plan["entries"][0]["node_id"] == node["id"]
    assert len(used_plan["entries"][0]["cutlass_conv_plan_key"]) == 64
    assert [item["symbol"] for item in used_plan["transform_helpers"]] == [
        cutlass_conv_input_pack_symbol("float16"),
        cutlass_conv_output_unpack_symbol("float16"),
        cutlass_conv_weight_pack_symbol("float16"),
    ]

    codegen_plan = create_codegen_plan(manifest, tmp_path / "cache")
    [support_lib] = codegen_plan.external_support_libraries
    assert support_lib["name"] == "cutlass_conv"
    assert support_lib["library"] == "lib/libdinoml_cutlass_conv.so"
    assert support_lib["used_candidate_plan_key"] == used_plan["used_candidate_plan_key"]
    assert support_lib["candidate_config_keys"] == [candidate["candidate_config_key"] for candidate in candidates]
    assert support_lib["transform_helper_symbols"] == [
        cutlass_conv_input_pack_symbol("float16"),
        cutlass_conv_output_unpack_symbol("float16"),
        cutlass_conv_weight_pack_symbol("float16"),
    ]
    assert [stage["stage_name"] for stage in codegen_plan.wrapper_stages] == [
        "activation_pack",
        "weight_pack",
        "provider_launch",
        "output_unpack",
    ]
    assert codegen_plan.wrapper_stages[0]["source"] == {
        "kind": "semantic_tensor",
        "role": "activation",
        "layout": "nchw",
    }
    assert codegen_plan.wrapper_stages[0]["destination"] == {
        "kind": "temporary_buffer",
        "name": "activation_nhwc",
        "layout": "nhwc",
        "nbytes": layout_plan["layout_translation"]["input_pack_nbytes"],
    }
    assert codegen_plan.wrapper_stages[1]["destination"]["name"] == "weight_ohwi"
    assert codegen_plan.wrapper_stages[2]["stage_kind"] == "provider_launcher"
    assert codegen_plan.wrapper_stages[2]["symbol"] == selected["kernel_symbol"]
    assert [arg["placeholder"] for arg in codegen_plan.wrapper_stages[2]["shape_args"]] == [
        "activation_n",
        "activation_h",
        "activation_w",
        "activation_c",
        "output_h",
        "output_w",
        "output_c",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "pad_h",
        "pad_w",
        "dilation_h",
        "dilation_w",
    ]
    assert codegen_plan.wrapper_stages[3]["destination"] == {
        "kind": "semantic_tensor",
        "role": "output",
        "layout": "nchw",
    }


def test_cutlass_conv2d_bias_codegen_wrapper_stages_render_source_snippets(tmp_path):
    spec = _trace_conv2d_bias("float16")
    kernel_manifest = build_kernel_manifest(spec.ir, {"name": "cuda", "arch": "sm_86"})
    [required] = kernel_manifest["required_kernels"]

    codegen_plan = create_codegen_plan(kernel_manifest, tmp_path / "cache")
    rendered = render_scaffold_wrapper_stages(codegen_plan.wrapper_stages)

    assert rendered[0] == (
        "DINO_CUDA_CHECK(dinoml_cutlass_conv_input_pack_nchw_to_nhwc_float16_v1("
        "ptr_activation, tmp_activation_nhwc, activation_n, activation_c, activation_h, activation_w, stream));"
    )
    assert rendered[1] == (
        "DINO_CUDA_CHECK(dinoml_cutlass_conv_weight_pack_oihw_to_ohwi_float16_v1("
        "ptr_weight, tmp_weight_ohwi, weight_o, weight_i, kernel_h, kernel_w, stream));"
    )
    assert rendered[2] == (
        f"int status_provider_launch = {required['kernel_symbol']}("
        "tmp_activation_nhwc, tmp_weight_ohwi, ptr_bias, tmp_output_nhwc, activation_n, activation_h, "
        "activation_w, activation_c, output_h, output_w, output_c, kernel_h, kernel_w, stride_h, stride_w, "
        "pad_h, pad_w, dilation_h, dilation_w, stream);\n"
        "if (status_provider_launch != 0) {\n"
        "  return status_provider_launch;\n"
        "}"
    )
    assert rendered[3] == (
        "DINO_CUDA_CHECK(dinoml_cutlass_conv_output_unpack_nhwc_to_nchw_float16_v1("
        "tmp_output_nhwc, ptr_output, output_n, output_c, output_h, output_w, stream));"
    )
    rendered_source = render_scaffold_wrapper_source(
        codegen_plan.wrapper_stages,
        op_name="conv2d_bias",
        node_id="conv_node_0",
    )
    assert rendered_source == (
        "// CUTLASS Conv scaffold only: emitted for artifact/source inspection.\n"
        "// This debug wrapper snippet is intentionally not compiled into the runtime module.\n"
        "// op: conv2d_bias\n"
        "// node_id: conv_node_0\n"
        "#if 0\n"
        "extern \"C\" int dinoml_cutlass_conv_wrapper_scaffold_conv_node__0(cudaStream_t stream) {\n"
        "  DINO_CUDA_CHECK(dinoml_cutlass_conv_input_pack_nchw_to_nhwc_float16_v1("
        "ptr_activation, tmp_activation_nhwc, activation_n, activation_c, activation_h, activation_w, stream));\n"
        "  DINO_CUDA_CHECK(dinoml_cutlass_conv_weight_pack_oihw_to_ohwi_float16_v1("
        "ptr_weight, tmp_weight_ohwi, weight_o, weight_i, kernel_h, kernel_w, stream));\n"
        f"  int status_provider_launch = {required['kernel_symbol']}("
        "tmp_activation_nhwc, tmp_weight_ohwi, ptr_bias, tmp_output_nhwc, activation_n, activation_h, "
        "activation_w, activation_c, output_h, output_w, output_c, kernel_h, kernel_w, stride_h, stride_w, "
        "pad_h, pad_w, dilation_h, dilation_w, stream);\n"
        "  if (status_provider_launch != 0) {\n"
        "    return status_provider_launch;\n"
        "  }\n"
        "  DINO_CUDA_CHECK(dinoml_cutlass_conv_output_unpack_nhwc_to_nchw_float16_v1("
        "tmp_output_nhwc, ptr_output, output_n, output_c, output_h, output_w, stream));\n"
        "  return 0;\n"
        "}\n"
        "#endif\n"
    )


def test_cutlass_conv2d_bias_manifest_selects_tensorop_for_c3_c4_c8_and_optimized_without_padding():
    few_spec = _trace_conv2d_bias(
        "float16",
        x_shape=(2, 3, 7, 8),
        weight_shape=(4, 3, 3, 2),
        bias_shape=(4,),
    )
    fixed4_spec = _trace_conv2d_bias(
        "float16",
        x_shape=(2, 4, 7, 8),
        weight_shape=(4, 4, 3, 2),
        bias_shape=(4,),
    )
    fixed8_spec = _trace_conv2d_bias(
        "float16",
        x_shape=(2, 8, 7, 8),
        weight_shape=(4, 8, 3, 2),
        bias_shape=(4,),
    )
    fallback_spec = _trace_conv2d_bias(
        "float16",
        x_shape=(2, 5, 7, 8),
        weight_shape=(4, 5, 3, 2),
        bias_shape=(4,),
    )
    optimized_spec = _trace_conv2d_bias(
        "float16",
        x_shape=(2, 16, 7, 8),
        weight_shape=(16, 16, 3, 2),
        bias_shape=(16,),
    )
    unaligned_spec = _trace_conv2d_bias(
        "float16",
        x_shape=(2, 16, 7, 8),
        weight_shape=(15, 16, 3, 2),
        bias_shape=(15,),
    )

    [few_required] = build_kernel_manifest(few_spec.ir, {"name": "cuda", "arch": "sm_86"})["required_kernels"]
    [fixed4_required] = build_kernel_manifest(fixed4_spec.ir, {"name": "cuda", "arch": "sm_86"})["required_kernels"]
    [fixed8_required] = build_kernel_manifest(fixed8_spec.ir, {"name": "cuda", "arch": "sm_86"})["required_kernels"]
    [fallback_required] = build_kernel_manifest(fallback_spec.ir, {"name": "cuda", "arch": "sm_86"})["required_kernels"]
    [optimized_required] = build_kernel_manifest(optimized_spec.ir, {"name": "cuda", "arch": "sm_86"})["required_kernels"]
    [unaligned_required] = build_kernel_manifest(unaligned_spec.ir, {"name": "cuda", "arch": "sm_86"})["required_kernels"]

    assert few_required["selected_candidate_id"].endswith("few_channels_c3")
    assert few_required["kernel_symbol"].endswith("tensorop_sm80_nhwc_ohwi_bias_few_channels_c3")
    assert few_required["cutlass_conv_plan"]["selected_candidate"]["opclass"] == "tensorop"
    assert few_required["cutlass_conv_plan"]["selected_candidate"]["iterator_algorithm"] == "few_channels"
    assert few_required["cutlass_conv_plan"]["weight_transform"]["channel_pad_multiple"] == 1
    assert few_required["cutlass_conv_plan"]["weight_transform"]["padded_input_channels"] == 3

    for required, channel_count in ((fixed4_required, 4), (fixed8_required, 8)):
        assert required["selected_candidate_id"].endswith(f"fixed_channels_c{channel_count}")
        assert required["kernel_symbol"].endswith(f"tensorop_sm80_nhwc_ohwi_bias_fixed_channels_c{channel_count}")
        assert required["cutlass_conv_plan"]["selected_candidate"]["opclass"] == "tensorop"
        assert required["cutlass_conv_plan"]["selected_candidate"]["iterator_algorithm"] == "fixed_channels"
        assert required["cutlass_conv_plan"]["selected_candidate"]["selection_predicate"] == {
            "kind": "semantic_input_channels",
            "input_channels": channel_count,
            "dtype": "float16",
            "groups": 1,
            "requires_layout_translation": "nchw_oihw_to_nhwc_ohwi",
            "padding_policy": "none",
        }
        assert required["cutlass_conv_plan"]["weight_transform"]["channel_pad_multiple"] == 1
        assert required["cutlass_conv_plan"]["weight_transform"]["padded_input_channels"] == channel_count
        selected_candidate = next(
            candidate
            for candidate in sorted(required["candidates"], key=lambda item: item["profiler_symbol"])
            if candidate["candidate_id"] == required["selected_candidate_id"]
        )
        assert selected_candidate["cutlass"]["align_a"] == channel_count
        assert selected_candidate["cutlass"]["align_b"] == channel_count
        assert selected_candidate["cutlass"]["stages"] == 3

    assert fallback_required["selected_candidate_id"].endswith("simt_sm80_nhwc_ohwi_bias")
    assert fallback_required["kernel_symbol"].endswith("simt_sm80_nhwc_ohwi_bias")
    assert fallback_required["cutlass_conv_plan"]["selected_candidate"]["opclass"] == "simt"
    assert fallback_required["cutlass_conv_plan"]["selected_candidate"]["iterator_algorithm"] == "analytic"
    assert fallback_required["cutlass_conv_plan"]["weight_transform"]["channel_pad_multiple"] == 1
    assert fallback_required["cutlass_conv_plan"]["weight_transform"]["padded_input_channels"] == 5

    assert optimized_required["selected_candidate_id"].endswith("optimized_align8")
    assert optimized_required["kernel_symbol"].endswith("tensorop_sm80_nhwc_ohwi_bias_optimized_align8")
    assert optimized_required["cutlass_conv_plan"]["selected_candidate"]["opclass"] == "tensorop"
    assert optimized_required["cutlass_conv_plan"]["selected_candidate"]["iterator_algorithm"] == "optimized"
    assert optimized_required["cutlass_conv_plan"]["selected_candidate"]["selection_predicate"] == {
        "kind": "natural_alignment",
        "dtype": "float16",
        "groups": 1,
        "min_input_channels": 16,
        "input_channels_multiple": 8,
        "output_channels_multiple": 8,
        "requires_layout_translation": "nchw_oihw_to_nhwc_ohwi",
        "padding_policy": "none",
    }
    assert optimized_required["cutlass_conv_plan"]["weight_transform"]["channel_pad_multiple"] == 1
    assert optimized_required["cutlass_conv_plan"]["weight_transform"]["padded_input_channels"] == 16
    assert optimized_required["cutlass_conv_plan"]["weight_transform"]["padded_output_channels"] == 16

    assert unaligned_required["selected_candidate_id"].endswith("simt_sm80_nhwc_ohwi_bias")
    assert unaligned_required["cutlass_conv_plan"]["selected_candidate"]["iterator_algorithm"] == "analytic"
    assert unaligned_required["cutlass_conv_plan"]["weight_transform"]["padded_input_channels"] == 16
    assert unaligned_required["cutlass_conv_plan"]["weight_transform"]["padded_output_channels"] == 15


def test_cutlass_conv2d_bias_cuda_runtime_provider_status_names_are_node_scoped(tmp_path, monkeypatch):
    spec = _trace_two_conv2d_bias("float16")
    kernel_manifest = build_kernel_manifest(spec.ir, {"name": "cuda", "arch": "sm_86"})
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    module_source = render_cuda_module(spec.ir, kernel_manifest=kernel_manifest)

    assert "int status_n0_provider_launch =" in module_source
    assert "int status_n1_provider_launch =" in module_source
    assert module_source.count("int status_provider_launch =") == 0


def test_cutlass_conv2d_bias_profile_workload_scaffold_records_provider_transforms():
    spec = _trace_conv2d_bias(
        "float16",
        x_shape=(2, 3, 7, 8),
        weight_shape=(4, 3, 3, 2),
        bias_shape=(4,),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 2),
    )
    target = {"name": "cuda", "arch": "sm_86", "no_tf32": True}
    kernel_manifest = build_kernel_manifest(spec.ir, target)

    workloads = build_profile_workloads(spec.ir, kernel_manifest)

    assert len(workloads) == 2
    assert {workload.candidate["cutlass"]["opclass"] for workload in workloads} == {"simt", "tensorop"}
    assert {workload.candidate["cutlass"]["iterator_algorithm"] for workload in workloads} == {
        "analytic",
        "few_channels",
    }
    workload = next(item for item in workloads if item.candidate["cutlass"]["opclass"] == "tensorop")
    assert workload.kernel_library == "cutlass_conv"
    assert workload.op == "conv2d_bias"
    assert workload.dtype == "float16"
    assert workload.x_shape == (2, 3, 7, 8)
    assert workload.weight_shape == (4, 3, 3, 2)
    assert workload.bias_shape == (4,)
    assert workload.output_shape == (2, 4, 4, 6)
    assert workload.conv_config == {"stride": [2, 1], "padding": [1, 0], "dilation": [1, 2], "groups": 1}
    assert workload.semantic_layout == {"activation": "nchw", "weight": "oihw", "bias": "o", "output": "nchw"}
    assert workload.provider_layout == {"activation": "nhwc", "weight": "ohwi", "bias": "o", "output": "nhwc"}
    assert workload.layout_translation["input_pack"] == "nchw_to_nhwc_temporary"
    assert workload.layout_translation["output_unpack"] == "nhwc_to_nchw_temporary"
    assert workload.weight_transform["from"] == "oihw"
    assert workload.weight_transform["to"] == "ohwi"
    assert workload.weight_transform["channel_pad_multiple"] == 1
    assert workload.shape_source == "graph_max_shape"
    assert workload.shape_case_id == "max"
    payload = workload.to_json()
    assert payload["profile_variant"] == {"kind": "bounded_runtime", "profiler_status": "bounded_runtime_profiler"}
    assert payload["inputs"] == {"x": [2, 3, 7, 8], "weight": [4, 3, 3, 2], "bias": [4]}
    assert payload["output"] == {workload.output_tensor: [2, 4, 4, 6]}
    assert payload["temporary_buffers"]
    assert payload["candidate"]["status"] == "bounded_runtime"
    assert payload["candidate"]["profiler_status"] == "bounded_runtime_profiler"
    assert payload["candidate"]["cutlass"]["iterator_algorithm"] == "few_channels"


@pytest.mark.parametrize(
    ("x_shape", "weight_shape", "bias_shape", "expected_algorithms"),
    [
        ((2, 3, 7, 8), (4, 3, 3, 2), (4,), {"analytic", "few_channels"}),
        ((2, 4, 7, 8), (8, 4, 3, 2), (8,), {"analytic", "fixed_channels"}),
        ((2, 16, 7, 8), (16, 16, 3, 2), (16,), {"analytic", "optimized"}),
    ],
)
def test_cutlass_conv2d_bias_profile_workload_filters_candidates_by_shape_predicate(
    x_shape,
    weight_shape,
    bias_shape,
    expected_algorithms,
):
    spec = _trace_conv2d_bias(
        "float16",
        x_shape=x_shape,
        weight_shape=weight_shape,
        bias_shape=bias_shape,
    )
    kernel_manifest = build_kernel_manifest(spec.ir, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(spec.ir, kernel_manifest)

    assert {workload.candidate["cutlass"]["iterator_algorithm"] for workload in workloads} == expected_algorithms
    assert len(workloads) == len(expected_algorithms)
    for workload in workloads:
        predicate = workload.candidate["selection_predicate"]
        if predicate["kind"] == "semantic_input_channels":
            assert predicate["input_channels"] == x_shape[1]
        elif predicate["kind"] == "natural_alignment":
            assert x_shape[1] >= predicate["min_input_channels"]
            assert x_shape[1] % predicate["input_channels_multiple"] == 0
            assert weight_shape[0] % predicate["output_channels_multiple"] == 0
        else:
            assert predicate["kind"] == "fallback"


def test_cutlass_conv2d_bias_profile_workload_requires_manifest_transform_metadata():
    spec = _trace_conv2d_bias("float16")
    kernel_manifest = build_kernel_manifest(spec.ir, {"name": "cuda", "arch": "sm_86"})
    kernel_manifest["required_kernels"][0].pop("cutlass_conv_plan")

    with pytest.raises(ValueError, match="cutlass_conv_plan transform metadata"):
        build_profile_workloads(spec.ir, kernel_manifest)


def test_cutlass_conv2d_bias_profile_workload_rejects_incoherent_transform_nbytes():
    spec = _trace_conv2d_bias("float16")
    kernel_manifest = build_kernel_manifest(spec.ir, {"name": "cuda", "arch": "sm_86"})
    kernel_manifest["required_kernels"][0]["cutlass_conv_plan"]["layout_translation"]["input_pack_nbytes"] -= 2

    with pytest.raises(ValueError, match="input_pack_nbytes mismatch"):
        build_profile_workloads(spec.ir, kernel_manifest)


def test_cutlass_conv2d_bias_codegen_plan_rejects_candidate_layout_drift(tmp_path):
    spec = _trace_conv2d_bias("float16")
    kernel_manifest = build_kernel_manifest(spec.ir, {"name": "cuda", "arch": "sm_86"})
    required = kernel_manifest["required_kernels"][0]
    selected = next(
        candidate
        for candidate in required["candidates"]
        if candidate["candidate_id"] == required["selected_candidate_id"]
    )
    selected["layouts"]["activation_provider"] = "nchw"

    with pytest.raises(ValueError, match="candidate layouts do not match transform plan"):
        create_codegen_plan(kernel_manifest, tmp_path / "cache")


@pytest.mark.parametrize(
    ("mutator", "error_match"),
    [
        (
            lambda used_plan: used_plan["entries"][0]["candidates"][1]["layouts"].__setitem__("activation_provider", "nchw"),
            r"candidate\.layouts mismatch",
        ),
        (
            lambda used_plan: used_plan["entries"][0]["candidates"][1].__setitem__("dtype", "float32"),
            r"(candidate\.(candidate_config_key|kernel_symbol|profiler_symbol|dtype) mismatch|selected candidate_id is not emitted)",
        ),
    ],
)
def test_cutlass_conv_support_scaffold_rejects_mutated_used_plan_candidate_before_writing_manifests(
    tmp_path,
    monkeypatch,
    mutator,
    error_match,
):
    spec = _trace_conv2d_bias("float16")
    kernel_manifest = build_kernel_manifest(spec.ir, {"name": "cuda", "arch": "sm_86"})
    used_plan = cutlass_conv_used_candidate_plan(kernel_manifest)
    mutator(used_plan)
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    support_root = (
        tmp_path
        / "cache"
        / "support"
        / "cuda-86"
        / "cutlass-conv"
        / str(used_plan["support_cache_key"])[:16]
    )

    with pytest.raises(ValueError, match=error_match):
        ensure_cutlass_conv_support_scaffold("sm_86", used_candidate_plan=used_plan)

    assert not (support_root / "lib" / "cutlass_conv_manifest.json").exists()
    assert not (support_root / "src" / "source_manifest.json").exists()


def test_conv2d_bias_cpu_compile_builds_generated_bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec = _trace_conv2d_bias("float32")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "conv2d_bias_cpu.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "static int conv2d_bias_" in generated


def test_conv2d_bias_cuda_compile_builds_guarded_wrapper_with_cutlass_runtime_boundary(tmp_path, monkeypatch):
    spec = _trace_conv2d_bias("float16")
    artifact_dir = tmp_path / "conv2d_bias_cuda.dinoml"
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    nvcc_available = shutil.which("nvcc") is not None
    if nvcc_available:
        dml.compile(spec, dml.Target("cuda", arch="sm_86"), artifact_dir)
    else:
        with pytest.raises(NotImplementedError, match="compiled support library"):
            dml.compile(spec, dml.Target("cuda", arch="sm_86"), artifact_dir)

    kernel_manifest = read_json(artifact_dir / "kernel_manifest.json")
    [required] = kernel_manifest["required_kernels"]
    expected_node_id = spec.ir["nodes"][0]["id"]
    assert required["op"] == "conv2d_bias"
    assert required["kernel_library"] == "cutlass_conv"
    assert required["candidate_set"]["status"] == "bounded_runtime"
    assert required["candidate_set"]["profiler_status"] == "bounded_runtime_profiler"
    assert required["selected_candidate_id"].endswith("few_channels_c3")
    assert required["cutlass_conv_plan"]["selected_candidate"]["opclass"] == "tensorop"
    assert required["cutlass_conv_plan"]["selected_candidate"]["iterator_algorithm"] == "few_channels"
    assert required["cutlass_conv_plan"]["selected_candidate"]["kernel_symbol"] == required["kernel_symbol"]
    assert required["cutlass_conv_plan"]["status"] == "bounded_runtime"
    assert required["cutlass_conv_plan"]["runtime"]["launcher"] == "cutlass_implicit_gemm_conv2d_fprop_bias"
    assert required["cutlass_conv_plan"]["profiler_status"] == "bounded_runtime_profiler"
    assert "profiler_blocked_reason" not in required["cutlass_conv_plan"]

    codegen_plan = read_json(artifact_dir / "kernel_codegen_plan.json")
    [support_lib] = codegen_plan["external_support_libraries"]
    assert support_lib["name"] == "cutlass_conv"
    assert support_lib["library"] == "lib/libdinoml_cutlass_conv.so"
    assert [stage["stage_name"] for stage in codegen_plan["wrapper_stages"]] == [
        "activation_pack",
        "weight_pack",
        "provider_launch",
        "output_unpack",
    ]
    assert [stage["symbol"] for stage in codegen_plan["wrapper_stages"][:2]] == [
        required["cutlass_conv_plan"]["layout_translation"]["input_pack_symbol"],
        required["cutlass_conv_plan"]["weight_transform"]["pack_symbol"],
    ]
    assert codegen_plan["wrapper_stages"][2]["symbol"] == required["kernel_symbol"]
    assert codegen_plan["wrapper_stages"][3]["symbol"] == required["cutlass_conv_plan"]["layout_translation"]["output_unpack_symbol"]
    assert codegen_plan["wrapper_scaffold_manifest"] == "debug/generated_src/scaffold_source_manifest.json"
    [wrapper_source] = codegen_plan["wrapper_scaffold_sources"]
    assert wrapper_source["source_kind"] == "cutlass_conv_wrapper_scaffold"
    assert wrapper_source["op"] == "conv2d_bias"
    assert wrapper_source["node_id"] == expected_node_id
    assert wrapper_source["stage_names"] == [
        "activation_pack",
        "weight_pack",
        "provider_launch",
        "output_unpack",
    ]
    wrapper_source_path = artifact_dir / wrapper_source["emitted_source_path"]
    assert wrapper_source_path.exists()
    wrapper_source_text = wrapper_source_path.read_text(encoding="utf-8")
    assert wrapper_source_text.startswith("// CUTLASS Conv scaffold only: emitted for artifact/source inspection.\n")
    assert f"// node_id: {expected_node_id}\n" in wrapper_source_text
    assert "#if 0\n" in wrapper_source_text
    assert required["cutlass_conv_plan"]["layout_translation"]["input_pack_symbol"] in wrapper_source_text
    assert required["cutlass_conv_plan"]["weight_transform"]["pack_symbol"] in wrapper_source_text
    assert required["kernel_symbol"] in wrapper_source_text
    assert required["cutlass_conv_plan"]["layout_translation"]["output_unpack_symbol"] in wrapper_source_text
    scaffold_manifest = read_json(artifact_dir / codegen_plan["wrapper_scaffold_manifest"])
    assert scaffold_manifest["kind"] == "dinoml.wrapper_scaffold_source_manifest"
    assert scaffold_manifest["target"]["name"] == "cuda"
    assert scaffold_manifest["target"]["arch"] == "sm_86"
    assert scaffold_manifest["sources"] == codegen_plan["wrapper_scaffold_sources"]
    support_cache_dir = tmp_path / "cache" / "support" / "cuda-86" / "cutlass-conv" / kernel_manifest["support_cache_key"][:16]
    assert Path(support_lib["cache_dir"]) == support_cache_dir
    support_manifest = read_json(support_cache_dir / "lib" / "cutlass_conv_manifest.json")
    source_manifest = read_json(support_cache_dir / "src" / "source_manifest.json")
    assert support_manifest["profiler_status"] == "bounded_runtime_profiler"
    assert "profiler_blocked_reason" not in support_manifest
    assert support_manifest["source_manifest"] == "../src/source_manifest.json"
    assert support_manifest["library"] == "libdinoml_cutlass_conv.so"
    export_status = support_manifest["status"]
    expected_candidate_config_keys = [candidate["candidate_config_key"] for candidate in required["candidates"]]
    expected_exports = [
        *_expected_transform_helper_exports(required["cutlass_conv_plan"], status=export_status),
        *[
            {
                "kind": "launcher",
                "symbol": candidate["kernel_symbol"],
                "launch_abi": "dinoml_cutlass_conv2d_bias_v1",
                "status": export_status,
                "candidate_status": "bounded_runtime",
                "success_return_code": 0,
            }
            for candidate in required["candidates"]
        ],
        *[
            {
                "kind": "profiler",
                "symbol": candidate["profiler_symbol"],
                "launch_abi": "dinoml_cutlass_conv2d_bias_v1",
                "status": export_status,
                "profiler_status": "bounded_runtime_profiler",
                "success_return_min_ms": 0.0,
            }
            for candidate in sorted(required["candidates"], key=lambda item: item["profiler_symbol"])
        ],
    ]
    assert support_manifest["exports"] == expected_exports
    assert support_manifest["used_candidate_plan"]["entries"][0]["node_id"] == expected_node_id
    assert support_manifest["used_candidate_plan"]["entries"][0]["cutlass_conv_plan"] == required["cutlass_conv_plan"]
    assert support_manifest["used_candidate_plan"]["candidate_config_keys"] == expected_candidate_config_keys
    assert source_manifest["kind"] == "dinoml.support_source_manifest"
    assert source_manifest["provider"] == "cutlass"
    assert source_manifest["library"] == "cutlass_conv"
    assert source_manifest["used_candidate_plan"]["entries"][0]["node_id"] == expected_node_id
    assert source_manifest["used_candidate_plan"]["entries"][0]["cutlass_conv_plan"] == required["cutlass_conv_plan"]
    assert source_manifest["used_candidate_plan"]["candidate_config_keys"] == expected_candidate_config_keys
    assert source_manifest["sources"][0]["source_role"] == "support_library"
    assert source_manifest["sources"][0]["candidate_set_keys"] == [required["candidate_set_key"]]
    assert source_manifest["sources"][0]["candidate_config_keys"] == sorted(expected_candidate_config_keys)
    source_symbols = {item["name"] for item in source_manifest["sources"][0]["symbols"]}
    helper_symbols = {
        required["cutlass_conv_plan"]["layout_translation"]["input_pack_symbol"],
        required["cutlass_conv_plan"]["layout_translation"]["output_unpack_symbol"],
        required["cutlass_conv_plan"]["weight_transform"]["pack_symbol"],
    }
    assert helper_symbols.issubset(source_symbols)
    assert {candidate["kernel_symbol"] for candidate in required["candidates"]}.issubset(source_symbols)
    assert {candidate["profiler_symbol"] for candidate in required["candidates"]}.issubset(source_symbols)
    helper_entries = [item for item in source_manifest["sources"][0]["symbols"] if item["kind"] == "transform_helper"]
    assert [(item["tensor_role"], item["name"]) for item in helper_entries] == [
        ("activation", required["cutlass_conv_plan"]["layout_translation"]["input_pack_symbol"]),
        ("output", required["cutlass_conv_plan"]["layout_translation"]["output_unpack_symbol"]),
        ("weight", required["cutlass_conv_plan"]["weight_transform"]["pack_symbol"]),
    ]
    if shutil.which("nvcc") is None:
        assert support_manifest["status"] == "source_bounded_runtime"
        assert support_manifest["compile"]["status"] == "source_bounded_runtime"
        assert support_manifest["compile"]["blocked_reason"] == "nvcc_unavailable"
        assert "library_sha256" not in support_manifest
        assert not (support_cache_dir / "lib" / "libdinoml_cutlass_conv.so").exists()
        assert not (artifact_dir / "module.so").exists()
        return
    assert support_manifest["status"] == "compiled_bounded_runtime"
    assert len(support_manifest["library_sha256"]) == 64
    assert support_manifest["compile"]["status"] == "compiled_bounded_runtime"
    assert support_manifest["compile"]["command"][0] == "nvcc"
    assert any("cutlass/include" in root for root in support_manifest["compile"]["include_roots"])
    support_library = support_cache_dir / "lib" / "libdinoml_cutlass_conv.so"
    assert support_library.exists()
    stub = ctypes.CDLL(str(support_library))
    stub.dinoml_cutlass_conv_stub_status.restype = ctypes.c_int
    assert stub.dinoml_cutlass_conv_stub_status() == 901
    launcher = getattr(stub, required["kernel_symbol"])
    launcher.restype = ctypes.c_int
    launcher.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        *([ctypes.c_int] * 15),
        ctypes.c_void_p,
    ]
    assert launcher(
        None,
        None,
        None,
        None,
        2,
        7,
        8,
        3,
        4,
        6,
        4,
        3,
        2,
        2,
        1,
        1,
        0,
        1,
        2,
        None,
    ) != 0
    profiler = getattr(stub, required["profiler_symbol"])
    profiler.restype = ctypes.c_float
    profiler.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        *([ctypes.c_int] * 16),
        ctypes.c_void_p,
    ]
    assert profiler(
        None,
        None,
        None,
        None,
        2,
        7,
        8,
        3,
        4,
        6,
        4,
        3,
        2,
        2,
        1,
        1,
        0,
        1,
        2,
        5,
        None,
    ) == pytest.approx(-1.0)
    artifact_manifest = read_json(artifact_dir / "manifest.json")
    assert artifact_manifest["files"]["cutlass_conv_library"] == "lib/libdinoml_cutlass_conv.so"
    assert (artifact_dir / "lib" / "libdinoml_cutlass_conv.so").exists()
    assert (artifact_dir / "module.so").exists()
    module_source = (artifact_dir / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "void* cutlass_conv_tmp_" in module_source
    assert f"extern \"C\" int {required['cutlass_conv_plan']['layout_translation']['input_pack_symbol']}(" in module_source
    assert f"extern \"C\" int {required['cutlass_conv_plan']['weight_transform']['pack_symbol']}(" in module_source
    assert f"extern \"C\" int {required['kernel_symbol']}(" in module_source
    assert required["cutlass_conv_plan"]["layout_translation"]["input_pack_symbol"] in module_source
    assert required["cutlass_conv_plan"]["weight_transform"]["pack_symbol"] in module_source
    assert required["kernel_symbol"] in module_source
    assert required["cutlass_conv_plan"]["layout_translation"]["output_unpack_symbol"] in module_source
    assert "CUTLASS Conv provider launcher failed" in module_source
    if torch.cuda.is_available():
        module = runtime.load(artifact_dir, load_constants=False)
        session = module.create_session()
        try:
            x = _input((2, 3, 7, 8), "float16", -1.0, 1.0)
            weight = _input((4, 3, 3, 2), "float16", -0.5, 0.5)
            bias = _input((4,), "float16", -0.25, 0.25)
            actual = session.run_numpy({"x": x, "weight": weight, "bias": bias})["out"]
            expected = _storage_roundtrip(
                _torch_conv2d_bias_reference(
                    x,
                    weight,
                    bias,
                    stride=(2, 1),
                    padding=(1, 0),
                    dilation=(1, 2),
                ),
                "float16",
            )
            np.testing.assert_allclose(actual, expected, atol=2e-2, rtol=2e-2)
        finally:
            session.close()
            module.close()
        profile_report = profile_artifact(artifact_dir, iterations=1, repeats=1, refresh=True)
        assert profile_report["summary"]["profiled"] >= 2
        assert profile_report["summary"]["failed"] == 0
        assert (artifact_dir / "debug" / "profile_report.json").exists()


def test_conv2d_bias_cuda_runtime_fixed_channels_c4_matches_torch(tmp_path, monkeypatch):
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("fixed-channel CUTLASS Conv runtime parity requires nvcc and torch CUDA")

    spec = _trace_conv2d_bias(
        "float16",
        x_shape=(2, 4, 7, 8),
        weight_shape=(8, 4, 3, 2),
        bias_shape=(8,),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 2),
    )
    artifact_dir = tmp_path / "conv2d_bias_cuda_fixed_c4.dinoml"
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    dml.compile(spec, dml.Target("cuda", arch="sm_86"), artifact_dir)

    kernel_manifest = read_json(artifact_dir / "kernel_manifest.json")
    [required] = kernel_manifest["required_kernels"]
    assert required["selected_candidate_id"].endswith("fixed_channels_c4")
    assert required["cutlass_conv_plan"]["selected_candidate"]["iterator_algorithm"] == "fixed_channels"
    assert required["cutlass_conv_plan"]["weight_transform"]["channel_pad_multiple"] == 1
    assert required["cutlass_conv_plan"]["weight_transform"]["padded_input_channels"] == 4

    module = runtime.load(artifact_dir, load_constants=False)
    session = module.create_session()
    try:
        x = _input((2, 4, 7, 8), "float16", -1.0, 1.0)
        weight = _input((8, 4, 3, 2), "float16", -0.5, 0.5)
        bias = _input((8,), "float16", -0.25, 0.25)
        actual = session.run_numpy({"x": x, "weight": weight, "bias": bias})["out"]
        expected = _storage_roundtrip(
            _torch_conv2d_bias_reference(
                x,
                weight,
                bias,
                stride=(2, 1),
                padding=(1, 0),
                dilation=(1, 2),
            ),
            "float16",
        )
        np.testing.assert_allclose(actual, expected, atol=2e-2, rtol=2e-2)
    finally:
        session.close()
        module.close()


def test_conv2d_bias_cuda_runtime_optimized_aligned_c16_matches_torch(tmp_path, monkeypatch):
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("optimized CUTLASS Conv runtime parity requires nvcc and torch CUDA")

    spec = _trace_conv2d_bias(
        "float16",
        x_shape=(2, 16, 7, 8),
        weight_shape=(16, 16, 3, 2),
        bias_shape=(16,),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 2),
    )
    artifact_dir = tmp_path / "conv2d_bias_cuda_optimized_c16.dinoml"
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    dml.compile(spec, dml.Target("cuda", arch="sm_86"), artifact_dir)

    kernel_manifest = read_json(artifact_dir / "kernel_manifest.json")
    [required] = kernel_manifest["required_kernels"]
    assert required["selected_candidate_id"].endswith("optimized_align8")
    assert required["cutlass_conv_plan"]["selected_candidate"]["iterator_algorithm"] == "optimized"
    assert required["cutlass_conv_plan"]["weight_transform"]["channel_pad_multiple"] == 1
    assert required["cutlass_conv_plan"]["weight_transform"]["padded_input_channels"] == 16
    assert required["cutlass_conv_plan"]["weight_transform"]["padded_output_channels"] == 16

    module = runtime.load(artifact_dir, load_constants=False)
    session = module.create_session()
    try:
        x = _input((2, 16, 7, 8), "float16", -1.0, 1.0)
        weight = _input((16, 16, 3, 2), "float16", -0.5, 0.5)
        bias = _input((16,), "float16", -0.25, 0.25)
        actual = session.run_numpy({"x": x, "weight": weight, "bias": bias})["out"]
        expected = _storage_roundtrip(
            _torch_conv2d_bias_reference(
                x,
                weight,
                bias,
                stride=(2, 1),
                padding=(1, 0),
                dilation=(1, 2),
            ),
            "float16",
        )
        np.testing.assert_allclose(actual, expected, atol=2e-2, rtol=2e-2)
    finally:
        session.close()
        module.close()


def test_conv2d_bias_cuda_runtime_float32_simt_clip_patch_shape_matches_torch(tmp_path, monkeypatch):
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("float32 CUTLASS Conv runtime parity requires nvcc and torch CUDA")

    spec = _trace_conv2d_bias(
        "float32",
        x_shape=(2, 3, 4, 4),
        weight_shape=(6, 3, 2, 2),
        bias_shape=(6,),
        stride=(2, 2),
        padding=(0, 0),
        dilation=(1, 1),
    )
    artifact_dir = tmp_path / "conv2d_bias_float32_clip_patch_cuda.dinoml"
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    dml.compile(spec, dml.Target("cuda", arch="sm_86"), artifact_dir)

    kernel_manifest = read_json(artifact_dir / "kernel_manifest.json")
    [required] = kernel_manifest["required_kernels"]
    assert required["selected_candidate_id"].endswith("simt_sm80_nhwc_ohwi_bias")
    assert required["cutlass_conv_plan"]["status"] == "bounded_runtime"
    assert required["cutlass_conv_plan"]["selected_candidate"]["opclass"] == "simt"
    assert required["cutlass_conv_plan"]["selected_candidate"]["iterator_algorithm"] == "analytic"
    assert required["cutlass_conv_plan"]["runtime"]["launcher"] == "cutlass_implicit_gemm_conv2d_fprop_bias"

    module = runtime.load(artifact_dir, load_constants=False)
    session = module.create_session()
    try:
        x = _input((2, 3, 4, 4), "float32", -1.0, 1.0)
        weight = _input((6, 3, 2, 2), "float32", -0.5, 0.5)
        bias = _input((6,), "float32", -0.25, 0.25)
        actual = session.run_numpy({"x": x, "weight": weight, "bias": bias})["out"]
        expected = _torch_conv2d_bias_reference(
            x,
            weight,
            bias,
            stride=(2, 2),
            padding=(0, 0),
            dilation=(1, 1),
        )
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    finally:
        session.close()
        module.close()


def test_cutlass_conv2d_bias_float32_non_clip_shape_uses_simt_runtime():
    spec = _trace_conv2d_bias(
        "float32",
        x_shape=(2, 3, 6, 7),
        weight_shape=(4, 3, 2, 3),
        bias_shape=(4,),
        stride=(1, 2),
        padding=(1, 1),
        dilation=(2, 1),
    )

    kernel_manifest = build_kernel_manifest(spec.ir, {"name": "cuda", "arch": "sm_86"})
    [required] = kernel_manifest["required_kernels"]

    assert required["selected_candidate_id"].endswith("simt_sm80_nhwc_ohwi_bias")
    assert required["candidate_set"]["status"] == "bounded_runtime"
    assert required["candidate_set"]["profiler_status"] == "bounded_runtime_profiler"
    assert "profiler_blocked_reason" not in required["candidate_set"]
    assert required["cutlass_conv_plan"]["status"] == "bounded_runtime"
    assert required["cutlass_conv_plan"]["runtime"]["launcher"] == "cutlass_implicit_gemm_conv2d_fprop_bias"
    assert required["cutlass_conv_plan"]["profiler_status"] == "bounded_runtime_profiler"
    assert required["cutlass_conv_plan"]["selected_candidate"]["selection_predicate"]["kind"] == "fallback"

    workloads = build_profile_workloads(spec.ir, kernel_manifest)
    assert len(workloads) == 1
    assert workloads[0].candidate["dtype"] == "float32"
    assert workloads[0].candidate["status"] == "bounded_runtime"
    assert workloads[0].candidate["cutlass"]["opclass"] == "simt"
    assert workloads[0].candidate["cutlass"]["iterator_algorithm"] == "analytic"


def test_conv2d_bias_cuda_runtime_float32_simt_general_shape_matches_torch(tmp_path, monkeypatch):
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("general float32 CUTLASS Conv runtime parity requires nvcc and torch CUDA")

    spec = _trace_conv2d_bias(
        "float32",
        x_shape=(2, 3, 6, 7),
        weight_shape=(4, 3, 2, 3),
        bias_shape=(4,),
        stride=(1, 2),
        padding=(1, 1),
        dilation=(2, 1),
    )
    artifact_dir = tmp_path / "conv2d_bias_float32_general_cuda.dinoml"
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    dml.compile(spec, dml.Target("cuda", arch="sm_86"), artifact_dir)

    kernel_manifest = read_json(artifact_dir / "kernel_manifest.json")
    [required] = kernel_manifest["required_kernels"]
    assert required["selected_candidate_id"].endswith("simt_sm80_nhwc_ohwi_bias")
    assert required["cutlass_conv_plan"]["status"] == "bounded_runtime"
    assert required["cutlass_conv_plan"]["selected_candidate"]["opclass"] == "simt"
    assert required["cutlass_conv_plan"]["selected_candidate"]["iterator_algorithm"] == "analytic"

    module = runtime.load(artifact_dir, load_constants=False)
    session = module.create_session()
    try:
        x = _input((2, 3, 6, 7), "float32", -1.0, 1.0)
        weight = _input((4, 3, 2, 3), "float32", -0.5, 0.5)
        bias = _input((4,), "float32", -0.25, 0.25)
        actual = session.run_numpy({"x": x, "weight": weight, "bias": bias})["out"]
        expected = _torch_conv2d_bias_reference(
            x,
            weight,
            bias,
            stride=(1, 2),
            padding=(1, 1),
            dilation=(2, 1),
        )
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    finally:
        session.close()
        module.close()


def test_cutlass_conv_support_scaffold_marks_exports_source_only_without_nvcc(tmp_path, monkeypatch):
    spec = _trace_conv2d_bias("float16")
    kernel_manifest = build_kernel_manifest(spec.ir, {"name": "cuda", "arch": "sm_86"})
    [required] = kernel_manifest["required_kernels"]
    used_plan = cutlass_conv_used_candidate_plan(kernel_manifest)
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    support_root = tmp_path / "cache" / "support" / "cuda-86" / "cutlass-conv" / str(used_plan["support_cache_key"])[:16]
    stale_library = support_root / "lib" / "libdinoml_cutlass_conv.so"
    stale_library.parent.mkdir(parents=True)
    stale_library.write_bytes(b"stale compiled stub")

    scaffold = ensure_cutlass_conv_support_scaffold("sm_86", used_candidate_plan=used_plan)

    support_manifest = read_json(scaffold.manifest)
    assert support_manifest["status"] == "source_bounded_runtime"
    assert support_manifest["compile"]["status"] == "source_bounded_runtime"
    assert support_manifest["compile"]["blocked_reason"] == "nvcc_unavailable"
    assert "library_sha256" not in support_manifest
    assert not stale_library.exists()
    assert support_manifest["exports"] == [
        *_expected_transform_helper_exports(required["cutlass_conv_plan"], status="source_bounded_runtime"),
        *[
            {
                "kind": "launcher",
                "symbol": candidate["kernel_symbol"],
                "launch_abi": "dinoml_cutlass_conv2d_bias_v1",
                "status": "source_bounded_runtime",
                "candidate_status": "bounded_runtime",
                "success_return_code": 0,
            }
            for candidate in required["candidates"]
        ],
        *[
            {
                "kind": "profiler",
                "symbol": candidate["profiler_symbol"],
                "launch_abi": "dinoml_cutlass_conv2d_bias_v1",
                "status": "source_bounded_runtime",
                "profiler_status": "bounded_runtime_profiler",
                "success_return_min_ms": 0.0,
            }
            for candidate in sorted(required["candidates"], key=lambda item: item["profiler_symbol"])
        ],
    ]


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_cutlass_conv_support_scaffold_runtime_transform_helpers_match_torch(tmp_path, monkeypatch, dtype):
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("CUDA runtime transform helper parity requires nvcc and torch CUDA")

    spec = _trace_conv2d_bias(
        dtype,
        x_shape=(2, 3, 4, 5),
        weight_shape=(4, 3, 2, 3),
        bias_shape=(4,),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
    )
    kernel_manifest = build_kernel_manifest(spec.ir, {"name": "cuda", "arch": "sm_86"})
    [required] = kernel_manifest["required_kernels"]
    used_plan = cutlass_conv_used_candidate_plan(kernel_manifest)
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    scaffold = ensure_cutlass_conv_support_scaffold("sm_86", used_candidate_plan=used_plan)
    support_library = Path(scaffold.manifest).parent / "libdinoml_cutlass_conv.so"
    stub = ctypes.CDLL(str(support_library))
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    x = torch.arange(2 * 3 * 4 * 5, device="cuda", dtype=getattr(torch, dtype)).reshape(2, 3, 4, 5)
    x_nhwc = torch.empty((2, 4, 5, 3), device="cuda", dtype=x.dtype)
    input_pack = getattr(stub, required["cutlass_conv_plan"]["layout_translation"]["input_pack_symbol"])
    input_pack.restype = ctypes.c_int
    input_pack.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    assert input_pack(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(x_nhwc.data_ptr()),
        2,
        3,
        4,
        5,
        stream,
    ) == 0
    torch.cuda.synchronize()
    assert torch.equal(x_nhwc, x.permute(0, 2, 3, 1).contiguous())

    weight = torch.arange(4 * 3 * 2 * 3, device="cuda", dtype=getattr(torch, dtype)).reshape(4, 3, 2, 3)
    weight_ohwi = torch.empty((4, 2, 3, 3), device="cuda", dtype=weight.dtype)
    weight_pack = getattr(stub, required["cutlass_conv_plan"]["weight_transform"]["pack_symbol"])
    weight_pack.restype = ctypes.c_int
    weight_pack.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    assert weight_pack(
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(weight_ohwi.data_ptr()),
        4,
        3,
        2,
        3,
        stream,
    ) == 0
    torch.cuda.synchronize()
    assert torch.equal(weight_ohwi, weight.permute(0, 2, 3, 1).contiguous())

    provider_output = torch.arange(2 * 3 * 4 * 4, device="cuda", dtype=getattr(torch, dtype)).reshape(2, 3, 4, 4)
    output_nchw = torch.empty((2, 4, 3, 4), device="cuda", dtype=provider_output.dtype)
    output_unpack = getattr(stub, required["cutlass_conv_plan"]["layout_translation"]["output_unpack_symbol"])
    output_unpack.restype = ctypes.c_int
    output_unpack.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    assert output_unpack(
        ctypes.c_void_p(provider_output.data_ptr()),
        ctypes.c_void_p(output_nchw.data_ptr()),
        2,
        4,
        3,
        4,
        stream,
    ) == 0
    torch.cuda.synchronize()
    assert torch.equal(output_nchw, provider_output.permute(0, 3, 1, 2).contiguous())
