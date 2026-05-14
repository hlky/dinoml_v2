import numpy as np
import pytest
import torch
import torch.nn.functional as F

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import array_from_storage, array_to_storage, read_json
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.kernels.providers.cutlass.conv import (
    cutlass_conv_candidate_set,
    cutlass_conv_candidates,
    cutlass_conv_layout_plan,
    cutlass_conv_used_candidate_plan,
)
from dinoml.kernels.profiling import build_profile_workloads
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
    assert candidate_set["status"] == "manifest_scaffold_only"
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
    assert candidates[0]["status"] == "manifest_scaffold_only"

    layout_plan = cutlass_conv_layout_plan(node, tensor_map=tensor_map)
    assert layout_plan["status"] == "manifest_scaffold_only"
    assert layout_plan["blocked_reason"] == "cutlass_conv_runtime_launcher_not_implemented"
    assert layout_plan["semantic_layout"]["activation"] == "nchw"
    assert layout_plan["provider_layout"]["activation"] == "nhwc"
    assert layout_plan["layout_translation"]["input_pack"] == "nchw_to_nhwc_temporary"
    assert layout_plan["layout_translation"]["output_unpack"] == "nhwc_to_nchw_temporary"
    assert layout_plan["weight_transform"]["from"] == "oihw"
    assert layout_plan["weight_transform"]["to"] == "ohwi"
    assert layout_plan["weight_transform"]["channel_pad_multiple"] == 1

    required = {
        "op": "conv2d_bias",
        "kernel_symbol": candidates[0]["kernel_symbol"],
        "kernel_library": "cutlass_conv",
        "profiler_symbol": candidates[0]["profiler_symbol"],
        "selected_candidate_id": candidates[0]["candidate_id"],
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
    assert used_plan["entries"][0]["cutlass_conv_plan"] == layout_plan
    assert len(used_plan["entries"][0]["cutlass_conv_plan_key"]) == 64

    codegen_plan = create_codegen_plan(manifest, tmp_path / "cache")
    [support_lib] = codegen_plan.external_support_libraries
    assert support_lib["name"] == "cutlass_conv"
    assert support_lib["library"] == "lib/libdinoml_cutlass_conv.so"
    assert support_lib["used_candidate_plan_key"] == used_plan["used_candidate_plan_key"]


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

    assert len(workloads) == 1
    workload = workloads[0]
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
    assert payload["profile_variant"] == {"kind": "manifest_scaffold_only"}
    assert payload["inputs"] == {"x": [2, 3, 7, 8], "weight": [4, 3, 3, 2], "bias": [4]}
    assert payload["output"] == {workload.output_tensor: [2, 4, 4, 6]}
    assert payload["temporary_buffers"]
    assert payload["candidate"]["status"] == "manifest_scaffold_only"


def test_cutlass_conv2d_bias_profile_workload_requires_manifest_transform_metadata():
    spec = _trace_conv2d_bias("float16")
    kernel_manifest = build_kernel_manifest(spec.ir, {"name": "cuda", "arch": "sm_86"})
    kernel_manifest["required_kernels"][0].pop("cutlass_conv_plan")

    with pytest.raises(ValueError, match="cutlass_conv_plan transform metadata"):
        build_profile_workloads(spec.ir, kernel_manifest)


def test_conv2d_bias_cpu_compile_rejects_unlowered_reference_only_surface(tmp_path):
    spec = _trace_conv2d_bias("float32")

    with pytest.raises(NotImplementedError, match="cpu backend does not support op conv2d_bias"):
        dml.compile(spec, dml.Target("cpu"), tmp_path / "conv2d_bias_cpu.dinoml")


def test_conv2d_bias_cuda_compile_emits_manifest_scaffold_then_rejects(tmp_path):
    spec = _trace_conv2d_bias("float16")
    artifact_dir = tmp_path / "conv2d_bias_cuda.dinoml"

    with pytest.raises(NotImplementedError, match="manifest/codegen scaffold only"):
        dml.compile(spec, dml.Target("cuda", arch="sm_86"), artifact_dir)

    kernel_manifest = read_json(artifact_dir / "kernel_manifest.json")
    [required] = kernel_manifest["required_kernels"]
    assert required["op"] == "conv2d_bias"
    assert required["kernel_library"] == "cutlass_conv"
    assert required["candidate_set"]["status"] == "manifest_scaffold_only"
    assert required["cutlass_conv_plan"]["status"] == "manifest_scaffold_only"
    assert required["cutlass_conv_plan"]["blocked_reason"] == "cutlass_conv_runtime_launcher_not_implemented"

    codegen_plan = read_json(artifact_dir / "kernel_codegen_plan.json")
    [support_lib] = codegen_plan["external_support_libraries"]
    assert support_lib["name"] == "cutlass_conv"
    assert support_lib["library"] == "lib/libdinoml_cutlass_conv.so"
    assert not (artifact_dir / "manifest.json").exists()
    assert not (artifact_dir / "module.so").exists()
