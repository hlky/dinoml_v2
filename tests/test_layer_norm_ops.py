import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml import runtime
from dinoml.backends.cpu import execute_cpu
from dinoml.frontend import GraphBuilder
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.ops import collect_generated_sources
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError


class LayerNormModule(dml.Module):
    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def forward(self, x, weight, bias):
        return dml.ops.output(dml.ops.layer_norm(x, weight, bias, eps=self.eps), "out")


def _trace_layer_norm(
    dtype: str = "float32",
    x_shape=(2, 3, 8),
    weight_shape=(8,),
    bias_shape=(8,),
    eps: float = 1e-5,
):
    return dml.trace(
        LayerNormModule(eps=eps),
        inputs={
            "x": dml.TensorSpec(x_shape, dtype),
            "weight": dml.TensorSpec(weight_shape, dtype),
            "bias": dml.TensorSpec(bias_shape, dtype),
        },
        name=f"layer_norm_{dtype}",
    )


def _storage_roundtrip(value, dtype: str) -> np.ndarray:
    if dtype == "float32":
        return np.asarray(value, dtype=np.float32)
    return array_from_storage(array_to_storage(np.asarray(value, dtype=np.float32), dtype), dtype)


def _reference_layer_norm(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    eps: float,
    dtype: str,
) -> np.ndarray:
    x_ref = _storage_roundtrip(x, dtype).astype(np.float32)
    weight_ref = _storage_roundtrip(weight, dtype).astype(np.float32)
    bias_ref = _storage_roundtrip(bias, dtype).astype(np.float32)
    mean = np.mean(x_ref, axis=-1, keepdims=True)
    variance = np.maximum(np.mean(x_ref * x_ref, axis=-1, keepdims=True) - mean * mean, 0.0)
    result = (x_ref - mean) * (1.0 / np.sqrt(variance + float(eps))) * weight_ref + bias_ref
    return _storage_roundtrip(result, dtype)


def _torch_reference_layer_norm(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    torch = pytest.importorskip("torch")
    expected = torch.nn.functional.layer_norm(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        normalized_shape=(int(weight.shape[0]),),
        weight=torch.from_numpy(np.asarray(weight, dtype=np.float32)),
        bias=torch.from_numpy(np.asarray(bias, dtype=np.float32)),
        eps=float(eps),
    )
    return expected.detach().cpu().numpy().astype(np.float32)


def _clip_like_layer_norm_inputs(x_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = int(x_shape[-1])
    coords = np.arange(int(np.prod(x_shape)), dtype=np.float32).reshape(x_shape)
    x = (0.8 * np.sin(coords / 37.0) + 0.35 * np.cos(coords / 19.0)).astype(np.float32)
    x += np.linspace(-0.2, 0.25, num=hidden, dtype=np.float32).reshape((1,) * (len(x_shape) - 1) + (hidden,))
    if len(x_shape) > 1:
        row_offsets = np.arange(int(np.prod(x_shape[:-1])), dtype=np.float32).reshape(x_shape[:-1] + (1,))
        x += (row_offsets * 0.03125).astype(np.float32)
    weight = (1.0 + 0.05 * np.sin(np.linspace(-1.5, 1.5, num=hidden, dtype=np.float32))).astype(np.float32)
    bias = (0.02 * np.cos(np.linspace(-0.75, 0.75, num=hidden, dtype=np.float32))).astype(np.float32)
    return x, weight, bias


def test_layer_norm_frontend_ir_preserves_shape_spec_eps_and_dtype():
    batch = dml.Dim("batch", min=1, max=4)
    spec = dml.trace(
        LayerNormModule(eps=1e-5),
        inputs={
            "x": dml.TensorSpec([batch, 8], "float32"),
            "weight": dml.TensorSpec([8], "float32"),
            "bias": dml.TensorSpec([8], "float32"),
        },
        name="layer_norm_dynamic_batch",
    )

    node = spec.ir["nodes"][0]
    output = spec.ir["outputs"][0]
    assert node["op"] == "layer_norm"
    assert node["inputs"] == ["x", "weight", "bias"]
    assert node["attrs"] == {"eps": 1e-05}
    assert output["shape"] == [4, 8]
    assert output["shape_spec"] == [batch.to_json(), 8]
    assert output["dtype"] == "float32"


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [("float32", 1e-6, 1e-6), ("float16", 2e-3, 2e-3), ("bfloat16", 2e-2, 2e-2)],
)
def test_cpu_reference_layer_norm_matches_expected(dtype, atol, rtol):
    spec = _trace_layer_norm(dtype=dtype, x_shape=(2, 4, 8), weight_shape=(8,), bias_shape=(8,), eps=1e-5)
    rng = np.random.default_rng(123)
    x = (rng.standard_normal((2, 4, 8)).astype(np.float32) * 1.5) + 0.25
    weight = (rng.standard_normal((8,)).astype(np.float32) * 0.5) + 1.0
    bias = (rng.standard_normal((8,)).astype(np.float32) * 0.25) - 0.1

    actual = execute_cpu(spec, {"x": x, "weight": weight, "bias": bias})["out"]
    expected = _reference_layer_norm(x, weight, bias, eps=1e-5, dtype=dtype)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=atol, rtol=rtol)


def test_layer_norm_frontend_rejects_dynamic_last_dim_bad_affine_shapes_and_dtype():
    hidden = dml.Dim("hidden", min=4, max=16)

    with pytest.raises(ValueError, match="static last dimension"):
        dml.trace(
            LayerNormModule(),
            inputs={
                "x": dml.TensorSpec([2, hidden], "float32"),
                "weight": dml.TensorSpec([16], "float32"),
                "bias": dml.TensorSpec([16], "float32"),
            },
            name="layer_norm_dynamic_hidden",
        )

    with pytest.raises(ValueError, match="rank-1 weight"):
        _trace_layer_norm(dtype="float32", x_shape=(2, 8), weight_shape=(1, 8), bias_shape=(8,))

    with pytest.raises(ValueError, match="rank-1 bias"):
        _trace_layer_norm(dtype="float32", x_shape=(2, 8), weight_shape=(8,), bias_shape=(1, 8))

    with pytest.raises(ValueError, match="static weight shape"):
        dml.trace(
            LayerNormModule(),
            inputs={
                "x": dml.TensorSpec([2, 8], "float32"),
                "weight": dml.TensorSpec([hidden], "float32"),
                "bias": dml.TensorSpec([8], "float32"),
            },
            name="layer_norm_dynamic_weight",
        )

    with pytest.raises(ValueError, match="static bias shape"):
        dml.trace(
            LayerNormModule(),
            inputs={
                "x": dml.TensorSpec([2, 8], "float32"),
                "weight": dml.TensorSpec([8], "float32"),
                "bias": dml.TensorSpec([hidden], "float32"),
            },
            name="layer_norm_dynamic_bias",
        )

    with pytest.raises(ValueError, match="weight length must match"):
        _trace_layer_norm(dtype="float32", x_shape=(2, 8), weight_shape=(6,), bias_shape=(8,))

    with pytest.raises(ValueError, match="bias length must match"):
        _trace_layer_norm(dtype="float32", x_shape=(2, 8), weight_shape=(8,), bias_shape=(6,))

    with pytest.raises(ValueError, match="does not support dtype bool"):
        _trace_layer_norm(dtype="bool", x_shape=(2, 8), weight_shape=(8,), bias_shape=(8,))

    with pytest.raises(ValueError, match="dtype mismatch"):
        with GraphBuilder("layer_norm_dtype_mismatch") as builder:
            x = builder.input("x", dml.TensorSpec([2, 8], "float32"))
            weight = builder.input("weight", dml.TensorSpec([8], "float16"))
            bias = builder.input("bias", dml.TensorSpec([8], "float32"))
            dml.ops.layer_norm(x, weight, bias)

    with pytest.raises(ValueError, match="different DinoML traces"):
        with GraphBuilder("layer_norm_builder_x") as x_builder:
            x = x_builder.input("x", dml.TensorSpec([2, 8], "float32"))
        with GraphBuilder("layer_norm_builder_w") as weight_builder:
            weight = weight_builder.input("weight", dml.TensorSpec([8], "float32"))
        with GraphBuilder("layer_norm_builder_b") as bias_builder:
            bias = bias_builder.input("bias", dml.TensorSpec([8], "float32"))
        with GraphBuilder("layer_norm_builder_out"):
            dml.ops.layer_norm(x, weight, bias)


def test_layer_norm_validation_rejects_mismatched_affine_shape_output_and_dtype():
    spec = _trace_layer_norm(dtype="float32", x_shape=(2, 3, 8), weight_shape=(8,), bias_shape=(8,))

    spec.ir["inputs"][1]["shape"] = [7]
    weight_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "weight")
    weight_tensor["shape"] = [7]
    weight_tensor["shape_spec"] = [7]
    weight_tensor["layout"]["strides"] = [1]
    with pytest.raises(ValidationError, match="weight length must match"):
        validate_ir(spec.ir)

    spec = _trace_layer_norm(dtype="float32", x_shape=(2, 3, 8), weight_shape=(8,), bias_shape=(8,))
    spec.ir["inputs"][2]["shape"] = [7]
    bias_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "bias")
    bias_tensor["shape"] = [7]
    bias_tensor["shape_spec"] = [7]
    bias_tensor["layout"]["strides"] = [1]
    with pytest.raises(ValidationError, match="bias length must match"):
        validate_ir(spec.ir)

    spec = _trace_layer_norm(dtype="float32", x_shape=(2, 3, 8), weight_shape=(8,), bias_shape=(8,))
    spec.ir["outputs"][0]["shape"] = [2, 3, 7]
    spec.ir["outputs"][0]["shape_spec"] = [2, 3, 7]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 3, 7]
    output_tensor["shape_spec"] = [2, 3, 7]
    output_tensor["layout"]["strides"] = [21, 7, 1]
    with pytest.raises(ValidationError, match=r"expected \[2, 3, 8\]"):
        validate_ir(spec.ir)

    spec = _trace_layer_norm(dtype="float32", x_shape=(2, 3, 8), weight_shape=(8,), bias_shape=(8,))
    spec.ir["outputs"][0]["dtype"] = "float16"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "float16"
    with pytest.raises(ValidationError, match="expected float32"):
        validate_ir(spec.ir)


def test_layer_norm_manifest_and_generated_sources_are_model_owned():
    spec = _trace_layer_norm(dtype="float32", x_shape=(16, 257), weight_shape=(257,), bias_shape=(257,), eps=1e-5)
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cpu", "arch": "native"})
    [required] = manifest["required_kernels"]
    assert required["op"] == "layer_norm"
    assert required["kernel_symbol"] == "generated_layer_norm"
    assert required["kernel_library"] == "model"
    assert required["profiler_symbol"] is None
    assert required["has_profiler"] is False
    assert required["generated_source"]["generated_function_name"].startswith("layer_norm_")
    assert required["generated_source"]["source_key"].startswith("cpu:")

    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    assert len(sources["kernels"]) == 1
    assert "layer_norm_" in sources["kernels"][0]
    assert "const float mean" in sources["kernels"][0]
    assert "dinoml::math::cast<float>(bias[col])" in sources["kernels"][0]
    assert "_warp_kernel" in sources["kernels"][0]


def test_layer_norm_generated_source_preserves_small_eps_literals():
    spec = _trace_layer_norm(dtype="float32", x_shape=(2, 768), weight_shape=(768,), bias_shape=(768,), eps=1e-12)
    lowered, _ = PassManager().run(spec.ir)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    assert "constexpr float eps = 1e-12f;" in sources["kernels"][0]
    assert "constexpr float eps = 0.00000000f;" not in sources["kernels"][0]


def test_cpu_artifact_runs_generated_layer_norm_with_dynamic_leading_dims(tmp_path):
    batch = dml.Dim("batch", min=1, max=4)
    spec = dml.trace(
        LayerNormModule(eps=1e-5),
        inputs={
            "x": dml.TensorSpec([batch, 8], "float32"),
            "weight": dml.TensorSpec([8], "float32"),
            "bias": dml.TensorSpec([8], "float32"),
        },
        name="layer_norm_dynamic_cpu",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "layer_norm_dynamic_cpu.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "layer_norm_" in generated
    assert "sum_sq" in generated
    assert "mean" in generated
    assert "bias[col]" in generated

    rng = np.random.default_rng(321)
    weight = (rng.standard_normal((8,)).astype(np.float32) * 0.25) + 1.0
    bias = (rng.standard_normal((8,)).astype(np.float32) * 0.2) - 0.1

    module = runtime.load(artifact.path)
    session = module.create_session()
    for rows in (2, 4):
        x = (rng.standard_normal((rows, 8)).astype(np.float32) * 2.0) - 0.75
        expected = _reference_layer_norm(x, weight, bias, eps=1e-5, dtype="float32")
        actual = session.run_numpy({"x": x, "weight": weight, "bias": bias})["out"]
        assert actual.shape == (rows, 8)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    session.close()
    module.close()


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_cuda_artifact_runs_generated_layer_norm(tmp_path):
    spec = _trace_layer_norm(dtype="float32", x_shape=(64, 257), weight_shape=(257,), bias_shape=(257,), eps=1e-5)
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "layer_norm_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "layer_norm_" in generated
    assert "_warp_kernel" in generated
    assert "__shfl_down_sync" in generated
    assert "const float mean" in generated
    assert "fmaxf" in generated

    rng = np.random.default_rng(222)
    x = (rng.standard_normal((64, 257)).astype(np.float32) * 1.75) - 0.5
    weight = (rng.standard_normal((257,)).astype(np.float32) * 0.2) + 1.0
    bias = (rng.standard_normal((257,)).astype(np.float32) * 0.15) - 0.05
    expected = _reference_layer_norm(x, weight, bias, eps=1e-5, dtype="float32")

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({"x": x, "weight": weight, "bias": bias})["out"]
    session.close()
    module.close()

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
@pytest.mark.parametrize(
    ("label", "x_shape"),
    [
        ("clip_text_base", (1, 4, 512)),
        ("clip_vision_base", (1, 50, 768)),
    ],
)
def test_cuda_artifact_runs_generated_layer_norm_at_clip_base_shapes_against_torch(
    tmp_path, use_shared_dinoml_cuda_cache, label, x_shape
):
    spec = _trace_layer_norm(dtype="float32", x_shape=x_shape, weight_shape=(x_shape[-1],), bias_shape=(x_shape[-1],), eps=1e-5)
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / f"{label}_layer_norm_cuda.dinoml")

    x, weight, bias = _clip_like_layer_norm_inputs(x_shape)
    expected = _torch_reference_layer_norm(x, weight, bias, eps=1e-5)

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({"x": x, "weight": weight, "bias": bias})["out"]
    session.close()
    module.close()

    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=1e-6)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
@pytest.mark.parametrize(
    ("dtype", "torch_dtype", "atol", "rtol"),
    [("float16", "float16", 2e-3, 2e-3), ("bfloat16", "bfloat16", 2e-2, 2e-2)],
)
def test_layer_norm_reduced_precision_cuda_runtime_matches_reference_and_uses_fp32_accumulation(
    tmp_path, dtype, torch_dtype, atol, rtol
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    spec = _trace_layer_norm(dtype=dtype, x_shape=(8, 33), weight_shape=(33,), bias_shape=(33,), eps=1e-5)
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / f"layer_norm_{dtype}_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    storage_type = "half" if dtype == "float16" else "__nv_bfloat16"
    assert f"const {storage_type}* DINO_RESTRICT x" in generated
    assert f"const {storage_type}* DINO_RESTRICT weight" in generated
    assert f"const {storage_type}* DINO_RESTRICT bias" in generated
    assert f"{storage_type}* DINO_RESTRICT y" in generated
    assert "float thread_sum" in generated
    assert "float thread_sum_sq" in generated
    assert "dinoml::math::cast<float>(x[base + col])" in generated
    assert f"dinoml::math::cast<{storage_type}>" in generated

    rng = np.random.default_rng(456)
    x = (rng.standard_normal((8, 33)).astype(np.float32) * 1.25) - 0.4
    weight = (rng.standard_normal((33,)).astype(np.float32) * 0.2) + 1.0
    bias = (rng.standard_normal((33,)).astype(np.float32) * 0.1) - 0.05
    expected = _reference_layer_norm(x, weight, bias, eps=1e-5, dtype=dtype)
    x_torch = torch.tensor(x, device="cuda", dtype=getattr(torch, torch_dtype))
    weight_torch = torch.tensor(weight, device="cuda", dtype=getattr(torch, torch_dtype))
    bias_torch = torch.tensor(bias, device="cuda", dtype=getattr(torch, torch_dtype))

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_torch({"x": x_torch, "weight": weight_torch, "bias": bias_torch})["out"]
    session.close()
    module.close()

    assert actual.dtype == getattr(torch, torch_dtype)
    np.testing.assert_allclose(actual.float().cpu().numpy(), expected.astype(np.float32), atol=atol, rtol=rtol)
