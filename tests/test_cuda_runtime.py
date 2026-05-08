import shutil
import subprocess
import sys

import numpy as np
import pytest

import dinoml as dml
from dinoml import runtime
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import read_json


pytestmark = pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")


def test_cuda_artifact_runs_without_torch(tmp_path):
    from tests.models.fused_elementwise import build_spec, build_validation_inputs

    spec = build_spec()
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "fused_elementwise.dinoml")
    assert (artifact.path / "lib" / "libdinoml_runtime.so").exists()
    assert (artifact.path / "lib" / "libdinoml_cuda_runtime.so").exists()
    assert (artifact.path / "lib" / "libdinoml_cuda_kernels.so").exists()
    generated_module = artifact.path / "debug" / "generated_src" / "module.cu"
    generated_text = generated_module.read_text(encoding="utf-8")
    assert (artifact.path / "metadata.json").exists()
    assert read_json(artifact.path / "manifest.json")["files"]["metadata"] == "metadata.json"
    assert "kMetadataJson" not in generated_text
    assert "R\"DINOJSON" not in generated_text
    assert "fused_elementwise_" in generated_text
    assert "dino_fused_" not in generated_text
    assert "dinoml::math::mul" in generated_text
    assert "dinoml::math::sigmoid" in generated_text
    assert "dino_session_set_stream" in generated_text
    assert "dino_session_get_output_shape" in generated_text
    assert "last_output_shapes" in generated_text
    assert "session->stream" in generated_text
    assert ", session->stream)) return err;" in generated_text
    assert "if (!session->external_stream)" in generated_text

    inputs = build_validation_inputs()
    expected = execute_cpu(spec, inputs)

    module = runtime.load(artifact.path)
    assert module.metadata == read_json(artifact.path / "metadata.json")
    assert hasattr(module._dll, "dino_session_set_stream")
    assert hasattr(module._dll, "dino_session_get_output_shape")
    session = module.create_session()
    session.set_stream(0)
    actual = session.run_numpy(inputs)
    assert session.get_output_shape("y") == actual["y"].shape
    repeated = session.run_numpy(inputs)
    assert session._cuda_buffers
    session.close()
    module.close()

    np.testing.assert_allclose(actual["y"], expected["y"], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(repeated["y"], expected["y"], atol=1e-4, rtol=1e-4)


def test_runtime_constant_update_changes_output(tmp_path):
    from tests.models.fused_elementwise import build_spec, build_validation_inputs

    spec = build_spec()
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "fused_elementwise_constants.dinoml")
    inputs = build_validation_inputs()

    module = runtime.load(artifact.path)
    session = module.create_session()
    module.set_constant_numpy("scale", np.zeros_like(spec.constants["scale"]))
    module.set_constant_numpy("bias", np.zeros_like(spec.constants["bias"]))
    actual = session.run_numpy(inputs)
    session.close()
    module.close()

    np.testing.assert_allclose(actual["y"], np.zeros([2, 3, 4], dtype=np.float32), atol=1e-6, rtol=0)


class DTypeFusedElementwise(dml.Module):
    def __init__(self, dtype: str):
        self.scale = dml.Parameter([4], dtype=dtype)
        self.bias = dml.Parameter([4], dtype=dtype)

    def forward(self, x):
        y = dml.ops.mul(x, self.scale)
        y = dml.ops.add(y, self.bias)
        y = dml.ops.relu(y)
        y = dml.ops.mul(y, 0.5)
        return dml.ops.output(y, "y")


@pytest.mark.parametrize(
    ("dtype", "torch_dtype", "atol", "rtol"),
    [
        ("float16", "float16", 2e-3, 2e-3),
        ("bfloat16", "bfloat16", 2e-2, 2e-2),
    ],
)
def test_cuda_fused_elementwise_supports_reduced_precision_torch_pointers(tmp_path, dtype, torch_dtype, atol, rtol):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    torch_dtype_obj = getattr(torch, torch_dtype)
    constants = {
        "scale": np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float32),
        "bias": np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32),
    }
    spec = dml.trace(
        DTypeFusedElementwise(dtype),
        inputs={"x": dml.TensorSpec([2, 3, 4], dtype)},
        constants=constants,
        name=f"fused_elementwise_{dtype}",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / f"fused_elementwise_{dtype}.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert ("half" if dtype == "float16" else "__nv_bfloat16") in generated

    torch.manual_seed(123)
    x = torch.randn((2, 3, 4), device="cuda", dtype=torch_dtype_obj)
    scale = torch.tensor(constants["scale"], device="cuda", dtype=torch_dtype_obj)
    bias = torch.tensor(constants["bias"], device="cuda", dtype=torch_dtype_obj)
    expected = torch.relu(x.float() * scale.float() + bias.float()) * 0.5
    expected = expected.to(torch_dtype_obj).float().cpu().numpy()

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual_torch = session.run_torch({"x": x})["y"]
    actual_numpy = session.run_numpy({"x": x.float().cpu().numpy()})["y"]
    session.close()
    module.close()

    assert actual_torch.dtype == torch_dtype_obj
    np.testing.assert_allclose(actual_torch.float().cpu().numpy(), expected, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual_numpy.astype(np.float32), expected, atol=atol, rtol=rtol)


class VectorizableScalarChain(dml.Module):
    def forward(self, x):
        y = dml.ops.mul(x, 1.125)
        y = dml.ops.add(y, 0.375)
        y = dml.ops.relu(y)
        return dml.ops.output(y, "y")


class SoftmaxLastDim(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.softmax(x, dim=-1), "y")


class ReductionLastDim(dml.Module):
    def __init__(self, op_name: str, keepdim: bool = False):
        self.op_name = op_name
        self.keepdim = keepdim

    def forward(self, x):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(x, dim=-1, keepdim=self.keepdim), "y")


def test_cuda_fused_elementwise_emits_float4_vector_path(tmp_path):
    spec = dml.trace(
        VectorizableScalarChain(),
        inputs={"x": dml.TensorSpec([2, 32], "float32")},
        name="vectorizable_scalar_chain",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "vectorized_scalar_chain.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "float4 raw_x" in generated
    assert "_vec" in generated

    x = np.random.default_rng(42).standard_normal([2, 32]).astype(np.float32)
    expected = np.maximum(x * np.float32(1.125) + np.float32(0.375), 0.0).astype(np.float32)
    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({"x": x})["y"]
    session.close()
    module.close()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_cuda_generated_softmax_matches_numpy_for_attention_rows(tmp_path):
    spec = dml.trace(
        SoftmaxLastDim(),
        inputs={"x": dml.TensorSpec([256, 1024], "float32")},
        name="attention_row_softmax_cuda",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "attention_row_softmax_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "softmax_" in generated
    assert "_packed_kernel" in generated
    assert "float4" in generated
    assert "__shfl_down_sync" in generated
    assert "expf" in generated

    rng = np.random.default_rng(321)
    x = rng.standard_normal((256, 1024)).astype(np.float32) * 2.5
    shifted = x - np.max(x, axis=-1, keepdims=True)
    expected = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({"x": x})["y"]
    session.close()
    module.close()

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("op_name", "numpy_op"),
    [
        ("reduce_sum", np.sum),
        ("reduce_max", np.max),
        ("reduce_min", np.min),
        ("reduce_mean", np.mean),
    ],
)
def test_cuda_generated_reductions_match_numpy(tmp_path, op_name, numpy_op):
    spec = dml.trace(
        ReductionLastDim(op_name),
        inputs={"x": dml.TensorSpec([64, 257], "float32")},
        name=f"{op_name}_cuda",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / f"{op_name}_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert f"{op_name}_" in generated
    assert "_warp_kernel" in generated
    assert "__shfl_down_sync" in generated

    x = np.random.default_rng(91).standard_normal((64, 257)).astype(np.float32)
    expected = numpy_op(x, axis=-1).astype(np.float32)
    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({"x": x})["y"]
    session.close()
    module.close()

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_cuda_generated_reduction_keepdim_shape(tmp_path):
    spec = dml.trace(
        ReductionLastDim("reduce_sum", keepdim=True),
        inputs={"x": dml.TensorSpec([8, 16, 33], "float32")},
        name="reduce_sum_keepdim_cuda",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "reduce_sum_keepdim_cuda.dinoml")
    x = np.random.default_rng(92).standard_normal((8, 16, 33)).astype(np.float32)
    expected = np.sum(x, axis=-1, keepdims=True).astype(np.float32)
    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({"x": x})["y"]
    session.close()
    module.close()

    assert actual.shape == (8, 16, 1)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


class DynamicChannelBias(dml.Module):
    def __init__(self):
        self.scale = dml.Parameter([4], dtype="float32")
        self.bias = dml.Parameter([4], dtype="float32")

    def forward(self, x):
        return dml.ops.output(dml.ops.relu(x * self.scale + self.bias), "y")


class DynamicGenericBroadcast(dml.Module):
    def forward(self, x, z):
        return dml.ops.output(dml.ops.relu(x + z), "y")


def test_cuda_runtime_supports_dynamic_shapes(tmp_path):
    constants = {
        "scale": np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float32),
        "bias": np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32),
    }
    batch = dml.Dim("batch", min=1, max=4)
    height = dml.Dim("height", min=8, max=16, divisible_by=8)
    spec = dml.trace(
        DynamicChannelBias(),
        inputs={"x": dml.TensorSpec([batch, height, 4], "float32")},
        constants=constants,
        name="dynamic_channel_bias_cuda",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "dynamic_channel_bias_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "runtime_total" in generated
    assert "check_tensor_dynamic" in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    for shape in ((1, 8, 4), (3, 16, 4)):
        x = np.random.default_rng(sum(shape)).standard_normal(shape).astype(np.float32)
        expected = np.maximum(x * constants["scale"] + constants["bias"], 0.0).astype(np.float32)
        actual = session.run_numpy({"x": x})["y"]
        assert actual.shape == shape
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        x_torch = torch.randn((2, 8, 4), device="cuda", dtype=torch.float32)
        y_torch = torch.empty_like(x_torch)
        expected_torch = torch.relu(x_torch * torch.tensor(constants["scale"], device="cuda") + torch.tensor(constants["bias"], device="cuda"))
        session.run_device_pointers(
            {"x": x_torch.data_ptr()},
            {"y": y_torch.data_ptr()},
            input_shapes={"x": tuple(int(dim) for dim in x_torch.shape)},
        )
        np.testing.assert_allclose(y_torch.cpu().numpy(), expected_torch.cpu().numpy(), atol=1e-5, rtol=1e-5)
    session.close()
    module.close()


def test_cuda_runtime_supports_dynamic_generic_broadcast(tmp_path):
    batch = dml.Dim("batch", min=1, max=4)
    height = dml.Dim("height", min=8, max=16, divisible_by=8)
    spec = dml.trace(
        DynamicGenericBroadcast(),
        inputs={
            "x": dml.TensorSpec([batch, height, 4], "float32"),
            "z": dml.TensorSpec([1, height, 1], "float32"),
        },
        name="dynamic_generic_broadcast_cuda",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "dynamic_generic_broadcast_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "const int64_t* input_shape" in generated
    assert "session->shape_z" in generated
    assert "session->shape_t1" in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    for shape in ((1, 8, 4), (3, 16, 4)):
        rng = np.random.default_rng(sum(shape) + 23)
        x = rng.standard_normal(shape).astype(np.float32)
        z = rng.standard_normal((1, shape[1], 1)).astype(np.float32)
        expected = np.maximum(x + z, 0.0).astype(np.float32)
        actual = session.run_numpy({"x": x, "z": z})["y"]
        assert actual.shape == shape
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    session.close()
    module.close()


def test_cli_compile_inspect_validate(tmp_path):
    fixture = "examples/fused_elementwise.py"
    artifact = tmp_path / "cli_fused_elementwise.dinoml"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dinoml.cli",
            "compile",
            fixture,
            "--target",
            "cuda",
            "--arch",
            "sm_86",
            "--out",
            str(artifact),
        ],
        check=True,
        cwd="/workspace/dinoml_v2",
    )
    inspect = subprocess.run(
        [sys.executable, "-m", "dinoml.cli", "inspect", str(artifact)],
        check=True,
        cwd="/workspace/dinoml_v2",
        text=True,
        stdout=subprocess.PIPE,
    )
    assert '"nodes": 1' in inspect.stdout
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dinoml.cli",
            "validate",
            str(artifact),
            "--against",
            fixture,
        ],
        check=True,
        cwd="/workspace/dinoml_v2",
    )
