import sys
from pathlib import Path
import shutil

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
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import read_json
from dinoml.shapes import Dim


class Conv2dModule(dml.Module):
    def __init__(self, stride=1, padding=0, dilation=1, groups=1):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x, weight):
        return dml.ops.output(
            dml.ops.conv2d(
                x,
                weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            ),
            "out",
        )


def _trace_conv2d(
    dtype="float32",
    x_shape=(2, 3, 7, 8),
    weight_shape=(4, 3, 3, 2),
    stride=(2, 1),
    padding=(1, 0),
    dilation=(1, 2),
    groups=1,
):
    return dml.trace(
        Conv2dModule(stride=stride, padding=padding, dilation=dilation, groups=groups),
        inputs={
            "x": dml.TensorSpec(x_shape, dtype),
            "weight": dml.TensorSpec(weight_shape, dtype),
        },
        name=f"conv2d_{dtype}",
    )


def _input(shape, dtype, start, stop):
    value = np.linspace(start, stop, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    if dtype == "float16":
        return np.asarray(value, dtype=np.float16)
    return value


def _storage_roundtrip(value, dtype):
    if dtype == "float16":
        return np.asarray(value, dtype=np.float16)
    return np.asarray(value, dtype=np.float32)


def _torch_conv2d_reference(x, weight, *, stride, padding, dilation):
    return (
        F.conv2d(
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(weight, dtype=np.float32)),
            None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )
        .detach()
        .cpu()
        .numpy()
    )


def _assert_explicit_zero_bias_bridge(lowered_path, *, output_channels):
    lowered = read_json(lowered_path)
    [node] = lowered["nodes"]
    assert node["op"] == "conv2d_bias"
    assert node["attrs"]["source_op"] == "conv2d"
    assert node["attrs"]["bias_mode"] == "explicit_zero_constant"
    zero_bias_name = node["inputs"][2]
    zero_bias_tensor = next(tensor for tensor in lowered["tensors"] if tensor["name"] == zero_bias_name)
    assert zero_bias_tensor["kind"] == "constant"
    zero_bias_constant = next(constant for constant in lowered["constants"] if constant["name"] == zero_bias_name)
    assert zero_bias_constant["shape"] == [output_channels]


def _assert_conv2d_cuda_runtime_matches_torch(
    tmp_path,
    *,
    x_shape,
    weight_shape,
    stride,
    padding,
    dilation,
    artifact_name,
    expected_candidate_suffix,
    expected_iterator_algorithm,
    expected_padded_input_channels,
    expected_padded_output_channels=None,
):
    spec = _trace_conv2d(
        "float16",
        x_shape=x_shape,
        weight_shape=weight_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    artifact_dir = tmp_path / artifact_name

    dml.compile(spec, dml.Target("cuda", arch="sm_86"), artifact_dir)

    _assert_explicit_zero_bias_bridge(
        artifact_dir / "graph.dinoir.json",
        output_channels=weight_shape[0],
    )

    kernel_manifest = read_json(artifact_dir / "kernel_manifest.json")
    [required] = kernel_manifest["required_kernels"]
    assert required["op"] == "conv2d_bias"
    assert required["kernel_library"] == "cutlass_conv"
    assert required["candidate_set"]["status"] == "bounded_runtime"
    assert required["cutlass_conv_plan"]["status"] == "bounded_runtime"
    assert required["selected_candidate_id"].endswith(expected_candidate_suffix)
    assert required["cutlass_conv_plan"]["selected_candidate"]["opclass"] == "tensorop"
    assert required["cutlass_conv_plan"]["selected_candidate"]["iterator_algorithm"] == expected_iterator_algorithm
    assert required["cutlass_conv_plan"]["runtime"]["launcher"] == "cutlass_implicit_gemm_conv2d_fprop_bias"
    assert required["cutlass_conv_plan"]["weight_transform"]["channel_pad_multiple"] == 1
    assert required["cutlass_conv_plan"]["weight_transform"]["padded_input_channels"] == expected_padded_input_channels
    if expected_padded_output_channels is not None:
        assert required["cutlass_conv_plan"]["weight_transform"]["padded_output_channels"] == expected_padded_output_channels

    x = _input(x_shape, "float16", -1.0, 1.0)
    weight = _input(weight_shape, "float16", -0.5, 0.5)
    expected = _storage_roundtrip(
        _torch_conv2d_reference(
            x,
            weight,
            stride=stride,
            padding=padding,
            dilation=dilation,
        ),
        "float16",
    )

    module = runtime.load(artifact_dir)
    session = module.create_session()
    try:
        actual = session.run_numpy({"x": x, "weight": weight})["out"]
    finally:
        session.close()
        module.close()

    np.testing.assert_allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_conv2d_frontend_lowers_through_explicit_zero_bias_constant():
    spec = _trace_conv2d(
        "float32",
        x_shape=(2, 3, 7, 8),
        weight_shape=(4, 3, 3, 2),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 2),
    )

    assert spec.ir["outputs"][0]["shape"] == [2, 4, 4, 6]
    [node] = spec.ir["nodes"]
    assert node["op"] == "conv2d_bias"
    assert len(node["inputs"]) == 3
    assert node["attrs"] == {
        "stride": [2, 1],
        "padding": [1, 0],
        "dilation": [1, 2],
        "groups": 1,
        "bias_mode": "explicit_zero_constant",
        "source_op": "conv2d",
    }
    assert len(spec.ir["constants"]) == 1
    [constant] = spec.ir["constants"]
    assert constant["shape"] == [4]
    np.testing.assert_array_equal(spec.constants[constant["name"]], np.zeros((4,), dtype=np.float32))


@pytest.mark.parametrize("dtype,atol,rtol", [("float32", 1e-6, 1e-6), ("float16", 1e-3, 1e-3)])
def test_conv2d_cpu_reference_matches_torch(dtype, atol, rtol):
    spec = _trace_conv2d(
        dtype,
        x_shape=(2, 3, 6, 7),
        weight_shape=(4, 3, 2, 3),
        stride=(1, 2),
        padding=(1, 1),
        dilation=(2, 1),
    )
    x = _input((2, 3, 6, 7), dtype, -1.5, 2.5)
    weight = _input((4, 3, 2, 3), dtype, -0.75, 1.25)

    actual = execute_cpu(spec, {"x": x, "weight": weight})["out"]
    expected = _storage_roundtrip(
        _torch_conv2d_reference(
            x,
            weight,
            stride=(1, 2),
            padding=(1, 1),
            dilation=(2, 1),
        ),
        dtype,
    )

    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype,atol,rtol", [("float32", 1e-6, 1e-6), ("float16", 1e-3, 1e-3)])
def test_conv2d_cpu_artifact_reuses_generated_conv2d_bias_bridge(dtype, atol, rtol, tmp_path, monkeypatch):
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec = _trace_conv2d(
        dtype,
        x_shape=(2, 3, 6, 7),
        weight_shape=(4, 3, 2, 3),
        stride=(1, 2),
        padding=(1, 1),
        dilation=(2, 1),
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"conv2d_{dtype}_cpu.dinoml")

    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "static int conv2d_bias_" in generated

    x = _input((2, 3, 6, 7), dtype, -1.5, 2.5)
    weight = _input((4, 3, 2, 3), dtype, -0.75, 1.25)
    expected = _storage_roundtrip(
        _torch_conv2d_reference(
            x,
            weight,
            stride=(1, 2),
            padding=(1, 1),
            dilation=(2, 1),
        ),
        dtype,
    )

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy({"x": x, "weight": weight})["out"]
    finally:
        session.close()
        module.close()

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_conv2d_cuda_compile_records_explicit_zero_bias_bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec = _trace_conv2d(
        "float16",
        x_shape=(2, 3, 7, 8),
        weight_shape=(4, 3, 3, 2),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 2),
    )
    artifact_dir = tmp_path / "conv2d_cuda.dinoml"

    nvcc_available = shutil.which("nvcc") is not None
    if nvcc_available:
        dml.compile(spec, dml.Target("cuda", arch="sm_86"), artifact_dir)
    else:
        with pytest.raises(NotImplementedError, match="compiled support library"):
            dml.compile(spec, dml.Target("cuda", arch="sm_86"), artifact_dir)

    _assert_explicit_zero_bias_bridge(artifact_dir / "graph.dinoir.json", output_channels=4)

    kernel_manifest = read_json(artifact_dir / "kernel_manifest.json")
    [required] = kernel_manifest["required_kernels"]
    assert required["op"] == "conv2d_bias"
    assert required["kernel_library"] == "cutlass_conv"
    assert required["candidate_set"]["status"] == "bounded_runtime"
    assert required["cutlass_conv_plan"]["status"] == "bounded_runtime"
    assert required["cutlass_conv_plan"]["selected_candidate"]["iterator_algorithm"] == "few_channels"


def test_conv2d_cuda_runtime_few_channels_c3_matches_torch(tmp_path, use_shared_dinoml_cuda_cache):
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("few-channel CUTLASS Conv runtime parity requires nvcc and torch CUDA")

    _assert_conv2d_cuda_runtime_matches_torch(
        tmp_path,
        x_shape=(2, 3, 7, 8),
        weight_shape=(4, 3, 3, 2),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 2),
        artifact_name="conv2d_cuda_few_c3.dinoml",
        expected_candidate_suffix="few_channels_c3",
        expected_iterator_algorithm="few_channels",
        expected_padded_input_channels=3,
        expected_padded_output_channels=4,
    )


def test_conv2d_cuda_runtime_fixed_channels_c4_matches_torch(tmp_path, use_shared_dinoml_cuda_cache):
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("fixed-channel CUTLASS Conv runtime parity requires nvcc and torch CUDA")

    _assert_conv2d_cuda_runtime_matches_torch(
        tmp_path,
        x_shape=(2, 4, 7, 8),
        weight_shape=(8, 4, 3, 2),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 2),
        artifact_name="conv2d_cuda_fixed_c4.dinoml",
        expected_candidate_suffix="fixed_channels_c4",
        expected_iterator_algorithm="fixed_channels",
        expected_padded_input_channels=4,
        expected_padded_output_channels=8,
    )


def test_conv2d_cuda_runtime_fixed_channels_c8_matches_torch(tmp_path, use_shared_dinoml_cuda_cache):
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("fixed-channel CUTLASS Conv runtime parity requires nvcc and torch CUDA")

    _assert_conv2d_cuda_runtime_matches_torch(
        tmp_path,
        x_shape=(2, 8, 7, 8),
        weight_shape=(8, 8, 3, 2),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 2),
        artifact_name="conv2d_cuda_fixed_c8.dinoml",
        expected_candidate_suffix="fixed_channels_c8",
        expected_iterator_algorithm="fixed_channels",
        expected_padded_input_channels=8,
    )


def test_conv2d_cuda_runtime_optimized_aligned_c16_matches_torch(tmp_path, use_shared_dinoml_cuda_cache):
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("optimized CUTLASS Conv runtime parity requires nvcc and torch CUDA")

    _assert_conv2d_cuda_runtime_matches_torch(
        tmp_path,
        x_shape=(2, 16, 7, 8),
        weight_shape=(16, 16, 3, 2),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 2),
        artifact_name="conv2d_cuda_optimized_c16.dinoml",
        expected_candidate_suffix="optimized_align8",
        expected_iterator_algorithm="optimized",
        expected_padded_input_channels=16,
        expected_padded_output_channels=16,
    )


@pytest.mark.skipif(not torch.cuda.is_available() or shutil.which("nvcc") is None, reason="CUDA runtime smoke requires torch CUDA and nvcc")
def test_conv2d_cuda_runtime_smoke_matches_torch(tmp_path, monkeypatch):
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec = _trace_conv2d(
        "float32",
        x_shape=(1, 3, 4, 4),
        weight_shape=(6, 3, 2, 2),
        stride=(2, 2),
        padding=(0, 0),
        dilation=(1, 1),
    )
    major, minor = torch.cuda.get_device_capability()
    artifact = dml.compile(spec, dml.Target("cuda", arch=f"sm_{major}{minor}"), tmp_path / "conv2d_cuda_runtime.dinoml")

    x = _input((1, 3, 4, 4), "float32", -1.25, 1.75)
    weight = _input((6, 3, 2, 2), "float32", -0.75, 1.25)
    expected = _torch_conv2d_reference(x, weight, stride=(2, 2), padding=(0, 0), dilation=(1, 1))

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_torch(
            {
                "x": torch.from_numpy(x).to("cuda"),
                "weight": torch.from_numpy(weight).to("cuda"),
            }
        )["out"]
    finally:
        session.close()
        module.close()

    np.testing.assert_allclose(actual.detach().cpu().numpy(), expected, rtol=1e-5, atol=1e-5)


def test_conv2d_frontend_rejects_dynamic_shapes_bad_ranks_dtype_and_groups():
    class DynamicConv(dml.Module):
        def forward(self, x, weight):
            return dml.ops.conv2d(x, weight)

    with pytest.raises(ValueError, match="static activation and weight shapes"):
        dml.trace(
            DynamicConv(),
            inputs={
                "x": dml.TensorSpec([1, 3, Dim("h", 4, 8), 8], "float32"),
                "weight": dml.TensorSpec([4, 3, 3, 3], "float32"),
            },
        )
    with pytest.raises(ValueError, match="rank-4 NCHW activation"):
        _trace_conv2d("float32", x_shape=(2, 3, 7))
    with pytest.raises(ValueError, match="rank-4 OIHW weight"):
        _trace_conv2d("float32", weight_shape=(4, 3, 3))
    with pytest.raises(ValueError, match="does not support dtype bfloat16"):
        _trace_conv2d("bfloat16")
    with pytest.raises(NotImplementedError, match="groups=1 only"):
        _trace_conv2d("float32", groups=2)
    with pytest.raises(ValueError, match="positive integers"):
        _trace_conv2d("float32", stride=(1, 0))
    with pytest.raises(ValueError, match="non-negative integers"):
        _trace_conv2d("float32", padding=(0, -1))
    with pytest.raises(ValueError, match="positive integers"):
        _trace_conv2d("float32", dilation=(1, 0))
    with pytest.raises(ValueError, match="must match activation channels"):
        _trace_conv2d("float32", x_shape=(2, 3, 7, 8), weight_shape=(4, 2, 3, 2))
    with pytest.raises(ValueError, match="output height must be positive"):
        _trace_conv2d("float32", x_shape=(1, 3, 3, 8), weight_shape=(4, 3, 5, 2), dilation=(2, 1))
