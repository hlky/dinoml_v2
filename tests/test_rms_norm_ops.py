import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml import runtime
from dinoml.backends.cpu import execute_cpu
from dinoml.frontend import GraphBuilder, Parameter
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.ops.definitions import OP_REGISTRY
from dinoml.passes import PassManager, validate_ir


class RMSNormModule(dml.Module):
    def __init__(self, *, use_weight: bool, eps: float = 1e-6):
        self.use_weight = use_weight
        self.eps = eps

    def forward(self, x, weight=None):
        if self.use_weight:
            return dml.ops.output(dml.ops.rms_norm(x, weight, eps=self.eps), "out")
        return dml.ops.output(dml.ops.rms_norm(x, eps=self.eps), "out")


def _trace_rms_norm(
    *,
    dtype: str = "float32",
    x_shape=(2, 3, 8),
    weight_shape=(8,),
    use_weight: bool = True,
    eps: float = 1e-6,
):
    inputs = {"x": dml.TensorSpec(x_shape, dtype)}
    if use_weight:
        inputs["weight"] = dml.TensorSpec(weight_shape, dtype)
    return dml.trace(
        RMSNormModule(use_weight=use_weight, eps=eps),
        inputs=inputs,
        name=f"rms_norm_{dtype}_{'weighted' if use_weight else 'unweighted'}",
    )


def _storage_roundtrip(value, dtype: str) -> np.ndarray:
    if dtype == "float32":
        return np.asarray(value, dtype=np.float32)
    return array_from_storage(array_to_storage(np.asarray(value, dtype=np.float32), dtype), dtype)


def _reference_rms_norm(x: np.ndarray, weight: np.ndarray | None, *, eps: float, dtype: str) -> np.ndarray:
    x_ref = _storage_roundtrip(x, dtype).astype(np.float32)
    weight_ref = (
        np.ones((x_ref.shape[-1],), dtype=np.float32)
        if weight is None
        else _storage_roundtrip(weight, dtype).astype(np.float32)
    )
    mean_square = np.mean(x_ref * x_ref, axis=-1, keepdims=True)
    result = x_ref * (1.0 / np.sqrt(mean_square + float(eps))) * weight_ref
    return _storage_roundtrip(result, dtype)


def test_rms_norm_helper_stays_out_of_registry_and_composes_t5_layer_norm():
    weighted = _trace_rms_norm(dtype="float16", x_shape=(2, 4, 8), weight_shape=(8,), use_weight=True, eps=1e-5)
    unweighted = _trace_rms_norm(dtype="float16", x_shape=(2, 4, 8), use_weight=False, eps=1e-5)

    assert "rms_norm" not in OP_REGISTRY.frontend_names()

    weighted_node = weighted.ir["nodes"][0]
    assert weighted_node["op"] == "t5_layer_norm"
    assert weighted_node["inputs"] == ["x", "weight"]
    assert weighted_node["attrs"] == {"eps": 1e-05}
    assert len(weighted.ir["constants"]) == 0

    unweighted_node = unweighted.ir["nodes"][0]
    assert unweighted_node["op"] == "t5_layer_norm"
    assert unweighted_node["attrs"] == {"eps": 1e-05}
    assert unweighted_node["inputs"][0] == "x"
    assert len(unweighted.ir["constants"]) == 1
    ones_name = unweighted_node["inputs"][1]
    ones_constant = next(constant for constant in unweighted.ir["constants"] if constant["tensor"] == ones_name)
    assert ones_constant["shape"] == [8]
    assert ones_constant["dtype"] == "float16"
    np.testing.assert_allclose(unweighted.constants[ones_name].astype(np.float32), np.ones((8,), dtype=np.float32))

    for spec in (weighted, unweighted):
        assert all(node["op"] != "rms_norm" for node in spec.ir["nodes"])
        lowered, _ = PassManager().run(spec.ir)
        validate_ir(lowered)
        assert [node["op"] for node in lowered["nodes"]] == ["t5_layer_norm"]


@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [("float32", 1e-6, 1e-6), ("float16", 2e-3, 2e-3), ("bfloat16", 2e-2, 2e-2)],
)
def test_rms_norm_cpu_reference_matches_expected(dtype, atol, rtol, use_weight):
    spec = _trace_rms_norm(dtype=dtype, x_shape=(2, 4, 8), weight_shape=(8,), use_weight=use_weight, eps=1e-5)
    rng = np.random.default_rng(123 if use_weight else 456)
    x = (rng.standard_normal((2, 4, 8)).astype(np.float32) * 1.5) + 0.25
    weight = None
    inputs = {"x": x}
    if use_weight:
        weight = (rng.standard_normal((8,)).astype(np.float32) * 0.5) + 1.0
        inputs["weight"] = weight

    actual = execute_cpu(spec, inputs)["out"]
    expected = _reference_rms_norm(x, weight, eps=1e-5, dtype=dtype)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=atol, rtol=rtol)


def test_rms_norm_frontend_rejects_dynamic_hidden_bad_rank_dtype_and_weight_contracts():
    hidden = dml.Dim("hidden", min=4, max=16)

    with pytest.raises(ValueError, match="static last dimension"):
        dml.trace(
            RMSNormModule(use_weight=False),
            inputs={"x": dml.TensorSpec([2, hidden], "float32")},
            name="rms_norm_dynamic_hidden",
        )

    with pytest.raises(ValueError, match="rank >= 1 input"):
        with GraphBuilder("rms_norm_scalar") as builder:
            dml.ops.rms_norm(builder.constant(Parameter(np.array(1.0, dtype=np.float32))))

    with pytest.raises(ValueError, match="does not support dtype bool"):
        _trace_rms_norm(dtype="bool", x_shape=(2, 8), use_weight=False)

    with pytest.raises(ValueError, match="rank-1 weight"):
        _trace_rms_norm(dtype="float32", x_shape=(2, 8), weight_shape=(1, 8), use_weight=True)

    with pytest.raises(ValueError, match="weight length must match"):
        _trace_rms_norm(dtype="float32", x_shape=(2, 8), weight_shape=(6,), use_weight=True)

    with pytest.raises(ValueError, match="dtype mismatch"):
        with GraphBuilder("rms_norm_dtype_mismatch") as builder:
            x = builder.input("x", dml.TensorSpec([2, 8], "float32"))
            weight = builder.input("weight", dml.TensorSpec([8], "float16"))
            dml.ops.rms_norm(x, weight)

    with pytest.raises(ValueError, match="different DinoML traces"):
        with GraphBuilder("rms_norm_builder_x") as x_builder:
            x = x_builder.input("x", dml.TensorSpec([2, 8], "float32"))
        with GraphBuilder("rms_norm_builder_w") as weight_builder:
            weight = weight_builder.input("weight", dml.TensorSpec([8], "float32"))
        with GraphBuilder("rms_norm_builder_out"):
            dml.ops.rms_norm(x, weight)


@pytest.mark.parametrize("use_weight", [True, False])
def test_cpu_artifact_runs_generated_rms_norm_with_dynamic_leading_dims(tmp_path, use_weight):
    batch = dml.Dim("batch", min=1, max=4)
    inputs = {"x": dml.TensorSpec([batch, 8], "float32")}
    if use_weight:
        inputs["weight"] = dml.TensorSpec([8], "float32")
    spec = dml.trace(
        RMSNormModule(use_weight=use_weight, eps=1e-5),
        inputs=inputs,
        name=f"rms_norm_dynamic_cpu_{'weighted' if use_weight else 'unweighted'}",
    )
    artifact_name = "rms_norm_dynamic_weighted_cpu.dinoml" if use_weight else "rms_norm_dynamic_unweighted_cpu.dinoml"
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / artifact_name)
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "t5_layer_norm_" in generated
    assert "sum_sq" in generated
    assert "sqrtf" in generated

    rng = np.random.default_rng(321 if use_weight else 654)
    weight = None
    if use_weight:
        weight = (rng.standard_normal((8,)).astype(np.float32) * 0.25) + 1.0

    module = runtime.load(artifact.path)
    session = module.create_session()
    for rows in (2, 4):
        x = (rng.standard_normal((rows, 8)).astype(np.float32) * 2.0) - 0.75
        expected = _reference_rms_norm(x, weight, eps=1e-5, dtype="float32")
        runtime_inputs = {"x": x}
        if weight is not None:
            runtime_inputs["weight"] = weight
        actual = session.run_numpy(runtime_inputs)["out"]
        assert actual.shape == (rows, 8)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    session.close()
    module.close()


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
@pytest.mark.parametrize("use_weight", [True, False])
def test_cuda_artifact_runs_generated_rms_norm(tmp_path, use_weight):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    spec = _trace_rms_norm(dtype="float32", x_shape=(16, 33), weight_shape=(33,), use_weight=use_weight, eps=1e-5)
    artifact_name = "rms_norm_weighted_cuda.dinoml" if use_weight else "rms_norm_unweighted_cuda.dinoml"
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / artifact_name)
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "t5_layer_norm_" in generated
    assert "float thread_sum_sq" in generated

    rng = np.random.default_rng(789 if use_weight else 987)
    x = (rng.standard_normal((16, 33)).astype(np.float32) * 1.25) - 0.4
    weight = None
    inputs = {"x": torch.tensor(x, device="cuda", dtype=torch.float32)}
    if use_weight:
        weight = (rng.standard_normal((33,)).astype(np.float32) * 0.2) + 1.0
        inputs["weight"] = torch.tensor(weight, device="cuda", dtype=torch.float32)
    expected = _reference_rms_norm(x, weight, eps=1e-5, dtype="float32")

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_torch(inputs)["out"]
    session.close()
    module.close()

    assert actual.dtype == torch.float32
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize(
    ("dtype", "torch_dtype", "atol", "rtol"),
    [("float16", "float16", 2e-3, 2e-3), ("bfloat16", "bfloat16", 2e-2, 2e-2)],
)
def test_rms_norm_reduced_precision_cuda_runtime_matches_reference_and_uses_fp32_accumulation(
    tmp_path, use_weight, dtype, torch_dtype, atol, rtol
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    spec = _trace_rms_norm(dtype=dtype, x_shape=(8, 33), weight_shape=(33,), use_weight=use_weight, eps=1e-5)
    artifact_name = f"rms_norm_{'weighted' if use_weight else 'unweighted'}_{dtype}_cuda.dinoml"
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / artifact_name)
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    storage_type = "half" if dtype == "float16" else "__nv_bfloat16"
    assert f"const {storage_type}* DINO_RESTRICT x" in generated
    assert f"{storage_type}* DINO_RESTRICT y" in generated
    assert "float thread_sum_sq" in generated
    assert "dinoml::math::cast<float>(x[base + col])" in generated
    assert f"dinoml::math::cast<{storage_type}>" in generated
    if use_weight:
        assert f"const {storage_type}* DINO_RESTRICT weight" in generated

    rng = np.random.default_rng(456 if use_weight else 654)
    x = (rng.standard_normal((8, 33)).astype(np.float32) * 1.25) - 0.4
    weight = None
    runtime_inputs = {"x": torch.tensor(x, device="cuda", dtype=getattr(torch, torch_dtype))}
    if use_weight:
        weight = (rng.standard_normal((33,)).astype(np.float32) * 0.2) + 1.0
        runtime_inputs["weight"] = torch.tensor(weight, device="cuda", dtype=getattr(torch, torch_dtype))
    expected = _reference_rms_norm(x, weight, eps=1e-5, dtype=dtype)

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_torch(runtime_inputs)["out"]
    session.close()
    module.close()

    assert actual.dtype == getattr(torch, torch_dtype)
    np.testing.assert_allclose(actual.float().cpu().numpy(), expected.astype(np.float32), atol=atol, rtol=rtol)
