import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml import runtime
from dinoml.reference import reference_numpy
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.ops import collect_generated_sources
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError


class T5LayerNormModule(dml.Module):
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def forward(self, x, weight):
        return dml.ops.output(dml.ops.t5_layer_norm(x, weight, eps=self.eps), "out")


def _trace_t5_layer_norm(
    dtype: str = "float32",
    x_shape=(2, 3, 8),
    weight_shape=(8,),
    eps: float = 1e-6,
):
    return dml.trace(
        T5LayerNormModule(eps=eps),
        inputs={
            "x": dml.TensorSpec(x_shape, dtype),
            "weight": dml.TensorSpec(weight_shape, dtype),
        },
        name=f"t5_layer_norm_{dtype}",
    )


def _storage_roundtrip(value, dtype: str) -> np.ndarray:
    if dtype == "float32":
        return np.asarray(value, dtype=np.float32)
    return array_from_storage(array_to_storage(np.asarray(value, dtype=np.float32), dtype), dtype)


def _reference_t5_layer_norm(x: np.ndarray, weight: np.ndarray, *, eps: float, dtype: str) -> np.ndarray:
    x_ref = _storage_roundtrip(x, dtype).astype(np.float32)
    weight_ref = _storage_roundtrip(weight, dtype).astype(np.float32)
    mean_square = np.mean(x_ref * x_ref, axis=-1, keepdims=True)
    result = x_ref * (1.0 / np.sqrt(mean_square + float(eps))) * weight_ref
    return _storage_roundtrip(result, dtype)


def test_t5_layer_norm_frontend_ir_preserves_shape_spec_eps_and_dtype():
    batch = dml.Dim("batch", min=1, max=4)
    spec = dml.trace(
        T5LayerNormModule(eps=1e-5),
        inputs={
            "x": dml.TensorSpec([batch, 8], "float32"),
            "weight": dml.TensorSpec([8], "float32"),
        },
        name="t5_layer_norm_dynamic_batch",
    )

    node = spec.ir["nodes"][0]
    output = spec.ir["outputs"][0]
    assert node["op"] == "t5_layer_norm"
    assert node["inputs"] == ["x", "weight"]
    assert node["attrs"] == {"eps": 1e-05}
    assert output["shape"] == [4, 8]
    assert output["shape_spec"] == [batch.to_json(), 8]
    assert output["dtype"] == "float32"


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [("float32", 1e-6, 1e-6), ("float16", 2e-3, 2e-3), ("bfloat16", 2e-2, 2e-2)],
)
def test_cpu_reference_t5_layer_norm_matches_expected(dtype, atol, rtol):
    spec = _trace_t5_layer_norm(dtype=dtype, x_shape=(2, 4, 8), weight_shape=(8,), eps=1e-5)
    rng = np.random.default_rng(123)
    x = (rng.standard_normal((2, 4, 8)).astype(np.float32) * 1.5) + 0.25
    weight = (rng.standard_normal((8,)).astype(np.float32) * 0.5) + 1.0

    actual = reference_numpy(spec, {"x": x, "weight": weight})["out"]
    expected = _reference_t5_layer_norm(x, weight, eps=1e-5, dtype=dtype)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=atol, rtol=rtol)


def test_t5_layer_norm_frontend_rejects_dynamic_last_dim_bad_weight_and_dtype():
    hidden = dml.Dim("hidden", min=4, max=16)

    with pytest.raises(ValueError, match="static last dimension"):
        dml.trace(
            T5LayerNormModule(),
            inputs={
                "x": dml.TensorSpec([2, hidden], "float32"),
                "weight": dml.TensorSpec([16], "float32"),
            },
            name="t5_layer_norm_dynamic_hidden",
        )

    with pytest.raises(ValueError, match="rank-1 weight"):
        _trace_t5_layer_norm(dtype="float32", x_shape=(2, 8), weight_shape=(1, 8))

    with pytest.raises(ValueError, match="static weight shape"):
        dml.trace(
            T5LayerNormModule(),
            inputs={
                "x": dml.TensorSpec([2, 8], "float32"),
                "weight": dml.TensorSpec([hidden], "float32"),
            },
            name="t5_layer_norm_dynamic_weight",
        )

    with pytest.raises(ValueError, match="weight length must match"):
        _trace_t5_layer_norm(dtype="float32", x_shape=(2, 8), weight_shape=(6,))

    with pytest.raises(ValueError, match="does not support dtype bool"):
        _trace_t5_layer_norm(dtype="bool", x_shape=(2, 8), weight_shape=(8,))


def test_t5_layer_norm_validation_rejects_mismatched_weight_shape_output_and_dtype():
    spec = _trace_t5_layer_norm(dtype="float32", x_shape=(2, 3, 8), weight_shape=(8,))

    spec.ir["inputs"][1]["shape"] = [7]
    weight_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "weight")
    weight_tensor["shape"] = [7]
    weight_tensor["shape_spec"] = [7]
    weight_tensor["layout"]["strides"] = [1]
    with pytest.raises(ValidationError, match="weight length must match"):
        validate_ir(spec.ir)

    spec = _trace_t5_layer_norm(dtype="float32", x_shape=(2, 3, 8), weight_shape=(8,))
    spec.ir["outputs"][0]["shape"] = [2, 3, 7]
    spec.ir["outputs"][0]["shape_spec"] = [2, 3, 7]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 3, 7]
    output_tensor["shape_spec"] = [2, 3, 7]
    output_tensor["layout"]["strides"] = [21, 7, 1]
    with pytest.raises(ValidationError, match=r"expected \[2, 3, 8\]"):
        validate_ir(spec.ir)

    spec = _trace_t5_layer_norm(dtype="float32", x_shape=(2, 3, 8), weight_shape=(8,))
    spec.ir["outputs"][0]["dtype"] = "float16"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "float16"
    with pytest.raises(ValidationError, match="expected float32"):
        validate_ir(spec.ir)


def test_t5_layer_norm_manifest_and_generated_sources_are_model_owned():
    spec = _trace_t5_layer_norm(dtype="float32", x_shape=(16, 257), weight_shape=(257,), eps=1e-5)
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cpu", "arch": "native"})
    [required] = manifest["required_kernels"]
    assert required["op"] == "t5_layer_norm"
    assert required["kernel_symbol"] == "generated_t5_layer_norm"
    assert required["kernel_library"] == "model"
    assert required["profiler_symbol"] is None
    assert required["has_profiler"] is False
    assert required["generated_source"]["generated_function_name"].startswith("t5_layer_norm_")
    assert required["generated_source"]["source_key"].startswith("cpu:")

    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    assert len(sources["kernels"]) == 1
    assert "t5_layer_norm_" in sources["kernels"][0]
    assert "rsqrtf" in sources["kernels"][0]
    assert "dinoml::math::cast<float>(weight[col])" in sources["kernels"][0]
    assert "_warp_kernel" in sources["kernels"][0]


def test_cpu_artifact_runs_generated_t5_layer_norm_with_dynamic_leading_dims(tmp_path):
    batch = dml.Dim("batch", min=1, max=4)
    spec = dml.trace(
        T5LayerNormModule(eps=1e-5),
        inputs={
            "x": dml.TensorSpec([batch, 8], "float32"),
            "weight": dml.TensorSpec([8], "float32"),
        },
        name="t5_layer_norm_dynamic_cpu",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "t5_layer_norm_dynamic_cpu.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "t5_layer_norm_" in generated
    assert "sum_sq" in generated
    assert "sqrtf" in generated

    rng = np.random.default_rng(321)
    weight = (rng.standard_normal((8,)).astype(np.float32) * 0.25) + 1.0

    module = runtime.load(artifact.path)
    session = module.create_session()
    for rows in (2, 4):
        x = (rng.standard_normal((rows, 8)).astype(np.float32) * 2.0) - 0.75
        expected = _reference_t5_layer_norm(x, weight, eps=1e-5, dtype="float32")
        actual = session.run_numpy({"x": x, "weight": weight})["out"]
        assert actual.shape == (rows, 8)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    session.close()
    module.close()


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_cuda_artifact_runs_generated_t5_layer_norm(tmp_path):
    spec = _trace_t5_layer_norm(dtype="float32", x_shape=(64, 257), weight_shape=(257,), eps=1e-5)
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "t5_layer_norm_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "t5_layer_norm_" in generated
    assert "_warp_kernel" in generated
    assert "__shfl_down_sync" in generated
    assert "rsqrtf" in generated

    rng = np.random.default_rng(222)
    x = (rng.standard_normal((64, 257)).astype(np.float32) * 1.75) - 0.5
    weight = (rng.standard_normal((257,)).astype(np.float32) * 0.2) + 1.0
    expected = _reference_t5_layer_norm(x, weight, eps=1e-5, dtype="float32")

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({"x": x, "weight": weight})["out"]
    session.close()
    module.close()

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
@pytest.mark.parametrize(
    ("dtype", "torch_dtype", "atol", "rtol"),
    [("float16", "float16", 2e-3, 2e-3), ("bfloat16", "bfloat16", 2e-2, 2e-2)],
)
def test_t5_layer_norm_reduced_precision_cuda_runtime_matches_reference_and_uses_fp32_accumulation(
    tmp_path, dtype, torch_dtype, atol, rtol
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    spec = _trace_t5_layer_norm(dtype=dtype, x_shape=(8, 33), weight_shape=(33,), eps=1e-5)
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / f"t5_layer_norm_{dtype}_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    storage_type = "half" if dtype == "float16" else "__nv_bfloat16"
    assert f"const {storage_type}* DINO_RESTRICT x" in generated
    assert f"const {storage_type}* DINO_RESTRICT weight" in generated
    assert f"{storage_type}* DINO_RESTRICT y" in generated
    assert "float thread_sum_sq" in generated
    assert "dinoml::math::cast<float>(x[base + col])" in generated
    assert f"dinoml::math::cast<{storage_type}>" in generated

    rng = np.random.default_rng(456)
    x = (rng.standard_normal((8, 33)).astype(np.float32) * 1.25) - 0.4
    weight = (rng.standard_normal((33,)).astype(np.float32) * 0.2) + 1.0
    expected = _reference_t5_layer_norm(x, weight, eps=1e-5, dtype=dtype)
    x_torch = torch.tensor(x, device="cuda", dtype=getattr(torch, torch_dtype))
    weight_torch = torch.tensor(weight, device="cuda", dtype=getattr(torch, torch_dtype))

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_torch({"x": x_torch, "weight": weight_torch})["out"]
    session.close()
    module.close()

    assert actual.dtype == getattr(torch, torch_dtype)
    np.testing.assert_allclose(actual.float().cpu().numpy(), expected.astype(np.float32), atol=atol, rtol=rtol)
