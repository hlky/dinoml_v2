import numpy as np
import pytest
import shutil

import dinoml as dml
from dinoml import runtime
from dinoml.reference import reference_numpy
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.ops import collect_generated_sources
from dinoml.ops.definitions import OP_REGISTRY
from dinoml.passes import PassManager, validate_ir


class TimestepEmbeddingModule(dml.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1.0,
        scale: float = 1.0,
        max_period: int = 10000,
    ):
        self.embedding_dim = embedding_dim
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps):
        return dml.ops.output(
            dml.ops.get_timestep_embedding(
                timesteps,
                self.embedding_dim,
                flip_sin_to_cos=self.flip_sin_to_cos,
                downscale_freq_shift=self.downscale_freq_shift,
                scale=self.scale,
                max_period=self.max_period,
            ),
            "out",
        )


def _trace_timestep_embedding(
    *,
    dtype: str = "float32",
    timesteps_shape=(4,),
    embedding_dim: int = 6,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
):
    return dml.trace(
        TimestepEmbeddingModule(
            embedding_dim=embedding_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=downscale_freq_shift,
            scale=scale,
            max_period=max_period,
        ),
        inputs={"timesteps": dml.TensorSpec(timesteps_shape, dtype)},
        name=f"get_timestep_embedding_{dtype}_{embedding_dim}",
    )


def _storage_roundtrip(value, dtype: str) -> np.ndarray:
    if dtype == "float32":
        return np.asarray(value, dtype=np.float32)
    return array_from_storage(array_to_storage(np.asarray(value, dtype=np.float32), dtype), dtype)


def _reference_get_timestep_embedding(
    timesteps: np.ndarray,
    *,
    embedding_dim: int,
    flip_sin_to_cos: bool,
    downscale_freq_shift: float,
    scale: float,
    max_period: int,
    dtype: str,
) -> np.ndarray:
    timesteps_ref = _storage_roundtrip(timesteps, dtype).astype(np.float32)
    batch = timesteps_ref.shape[0]
    half_dim = embedding_dim // 2
    if half_dim == 0:
        return _storage_roundtrip(np.zeros((batch, 1), dtype=np.float32), dtype)

    exponent = (
        -np.log(np.float32(max_period))
        * np.arange(half_dim, dtype=np.float32)
        / np.float32(float(half_dim) - float(downscale_freq_shift))
    )
    frequencies = np.exp(exponent).astype(np.float32, copy=False)
    args = timesteps_ref[:, None] * frequencies[None, :]
    args = args * np.float32(scale)
    sin_part = np.sin(args).astype(np.float32, copy=False)
    cos_part = np.cos(args).astype(np.float32, copy=False)
    pieces = [cos_part, sin_part] if flip_sin_to_cos else [sin_part, cos_part]
    embedding = np.concatenate(pieces, axis=1)
    if embedding_dim % 2 == 1:
        embedding = np.concatenate([embedding, np.zeros((batch, 1), dtype=np.float32)], axis=1)
    return _storage_roundtrip(embedding, dtype)


def test_get_timestep_embedding_frontend_ir_registers_op_and_preserves_dynamic_shape_spec_and_dtype():
    batch = dml.Dim("batch", min=1, max=4)
    spec = dml.trace(
        TimestepEmbeddingModule(
            embedding_dim=5,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.5,
            scale=1.25,
            max_period=64,
        ),
        inputs={"timesteps": dml.TensorSpec([batch], "float16")},
        name="get_timestep_embedding_dynamic_batch",
    )

    node = spec.ir["nodes"][0]
    output = spec.ir["outputs"][0]
    assert "get_timestep_embedding" in OP_REGISTRY.frontend_names()
    assert len(spec.ir["nodes"]) == 1
    assert node["op"] == "get_timestep_embedding"
    assert node["inputs"] == ["timesteps"]
    assert node["attrs"] == {
        "embedding_dim": 5,
        "flip_sin_to_cos": True,
        "downscale_freq_shift": 0.5,
        "scale": 1.25,
        "max_period": 64.0,
    }
    assert output["shape"] == [4, 5]
    assert output["shape_spec"] == [batch.to_json(), 5]
    assert output["dtype"] == "float16"

    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [lowered_node["op"] for lowered_node in lowered["nodes"]] == ["get_timestep_embedding"]


def test_get_timestep_embedding_manifest_and_generated_sources_are_model_owned():
    spec = _trace_timestep_embedding(
        dtype="float32",
        timesteps_shape=(16,),
        embedding_dim=5,
        flip_sin_to_cos=True,
        downscale_freq_shift=0.5,
        scale=1.25,
        max_period=64,
    )
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)

    manifest = build_kernel_manifest(lowered, {"name": "cpu", "arch": "native"})
    [required] = manifest["required_kernels"]
    assert required["op"] == "get_timestep_embedding"
    assert required["kernel_symbol"] == "generated_get_timestep_embedding"
    assert required["kernel_library"] == "model"
    assert required["profiler_symbol"] is None
    assert required["has_profiler"] is False
    assert required["generated_source"]["generated_function_name"].startswith("get_timestep_embedding_")
    assert required["generated_source"]["source_key"].startswith("cpu:")

    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    assert len(sources["kernels"]) == 1
    generated = sources["kernels"][0]
    assert "get_timestep_embedding_" in generated
    assert "sinf" in generated
    assert "cosf" in generated
    assert "expf" in generated
    assert "concatenate_" not in generated
    assert "fused_elementwise" not in generated


def test_get_timestep_embedding_supports_embedding_dim_one_zero_column():
    spec = _trace_timestep_embedding(dtype="float32", embedding_dim=1)
    timesteps = np.array([0.5, 1.5, -2.0, 4.0], dtype=np.float32)

    actual = reference_numpy(spec, {"timesteps": timesteps})["out"]

    np.testing.assert_array_equal(actual, np.zeros((4, 1), dtype=np.float32))


def test_get_timestep_embedding_frontend_rejects_dynamic_rank_dtype_and_bad_parameters():
    dynamic_n = dml.Dim("n", min=1, max=4)
    dynamic_spec = dml.trace(
        TimestepEmbeddingModule(embedding_dim=6),
        inputs={"timesteps": dml.TensorSpec([dynamic_n], "float32")},
        name="get_timestep_embedding_dynamic",
    )

    assert dynamic_spec.ir["outputs"][0]["shape"] == [4, 6]
    assert dynamic_spec.ir["outputs"][0]["shape_spec"] == [dynamic_n.to_json(), 6]

    with pytest.raises(ValueError, match="rank-1 timesteps"):
        _trace_timestep_embedding(dtype="float32", timesteps_shape=(2, 2), embedding_dim=6)

    with pytest.raises(ValueError, match="does not support dtype bool"):
        _trace_timestep_embedding(dtype="bool", embedding_dim=6)

    with pytest.raises(ValueError, match="positive integer"):
        _trace_timestep_embedding(dtype="float32", embedding_dim=0)

    with pytest.raises(ValueError, match="non-zero"):
        _trace_timestep_embedding(dtype="float32", embedding_dim=2)

    with pytest.raises(ValueError, match="downscale_freq_shift must be finite"):
        _trace_timestep_embedding(dtype="float32", embedding_dim=6, downscale_freq_shift=float("inf"))

    with pytest.raises(ValueError, match="scale must be finite"):
        _trace_timestep_embedding(dtype="float32", embedding_dim=6, scale=float("nan"))

    with pytest.raises(ValueError, match="positive finite number"):
        _trace_timestep_embedding(dtype="float32", embedding_dim=6, max_period=0)


@pytest.mark.parametrize(
    ("dtype", "embedding_dim", "flip_sin_to_cos", "downscale_freq_shift", "scale", "max_period", "atol", "rtol"),
    [
        ("float32", 6, False, 1.0, 1.0, 10000, 1e-6, 1e-6),
        ("float32", 5, True, 0.5, 0.75, 64, 1e-6, 1e-6),
        ("float16", 7, False, 1.5, 2.0, 1000, 2e-3, 2e-3),
        ("bfloat16", 7, True, 0.25, 0.5, 256, 2e-2, 2e-2),
    ],
)
def test_cpu_reference_get_timestep_embedding_matches_formula(
    dtype,
    embedding_dim,
    flip_sin_to_cos,
    downscale_freq_shift,
    scale,
    max_period,
    atol,
    rtol,
):
    spec = _trace_timestep_embedding(
        dtype=dtype,
        embedding_dim=embedding_dim,
        flip_sin_to_cos=flip_sin_to_cos,
        downscale_freq_shift=downscale_freq_shift,
        scale=scale,
        max_period=max_period,
    )
    timesteps = np.array([0.0, 1.25, 10.5, -0.75], dtype=np.float32)
    expected = _reference_get_timestep_embedding(
        timesteps,
        embedding_dim=embedding_dim,
        flip_sin_to_cos=flip_sin_to_cos,
        downscale_freq_shift=downscale_freq_shift,
        scale=scale,
        max_period=max_period,
        dtype=dtype,
    )

    actual = reference_numpy(spec, {"timesteps": timesteps})["out"]

    assert actual.shape == (4, embedding_dim)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=atol, rtol=rtol)


def test_cpu_artifact_runs_generated_get_timestep_embedding_with_dynamic_timesteps_length(tmp_path):
    batch = dml.Dim("batch", min=1, max=4)
    spec = dml.trace(
        TimestepEmbeddingModule(
            embedding_dim=7,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.5,
            scale=0.75,
            max_period=64,
        ),
        inputs={"timesteps": dml.TensorSpec([batch], "float32")},
        name="get_timestep_embedding_dynamic_cpu",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "get_timestep_embedding_dynamic_cpu.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "get_timestep_embedding_" in generated
    assert "sinf" in generated
    assert "cosf" in generated
    assert "concatenate_" not in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    for timesteps in (
        np.array([0.0, 1.5], dtype=np.float32),
        np.array([0.0, 1.25, 10.5, -0.75], dtype=np.float32),
    ):
        expected = _reference_get_timestep_embedding(
            timesteps,
            embedding_dim=7,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.5,
            scale=0.75,
            max_period=64,
            dtype="float32",
        )
        actual = session.run_numpy({"timesteps": timesteps})["out"]
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    session.close()
    module.close()


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_get_timestep_embedding_float32_cuda_artifact_compiles_for_even_and_odd_cases(tmp_path):
    for embedding_dim, flip_sin_to_cos in ((6, False), (5, True)):
        spec = _trace_timestep_embedding(
            dtype="float32",
            embedding_dim=embedding_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=0.5,
            scale=0.75,
            max_period=64,
        )
        artifact = dml.compile(
            spec,
            dml.Target("cuda", arch="sm_86"),
            tmp_path / f"get_timestep_embedding_{embedding_dim}_cuda.dinoml",
        )
        generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
        assert "get_timestep_embedding_" in generated
        assert "sinf" in generated
        assert "cosf" in generated
        assert "concatenate_" not in generated


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_get_timestep_embedding_float32_cuda_runtime_matches_reference(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    spec = _trace_timestep_embedding(
        dtype="float32",
        embedding_dim=6,
        flip_sin_to_cos=False,
        downscale_freq_shift=1.0,
        scale=1.0,
        max_period=10000,
    )
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch="sm_86"),
        tmp_path / "get_timestep_embedding_runtime_float32_cuda.dinoml",
    )
    timesteps = np.array([0.0, 1.25, 10.5, -0.75], dtype=np.float32)
    expected = _reference_get_timestep_embedding(
        timesteps,
        embedding_dim=6,
        flip_sin_to_cos=False,
        downscale_freq_shift=1.0,
        scale=1.0,
        max_period=10000,
        dtype="float32",
    )

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_torch({"timesteps": torch.tensor(timesteps, device="cuda", dtype=torch.float32)})["out"]
    session.close()
    module.close()

    assert actual.dtype == torch.float32
    np.testing.assert_allclose(actual.float().cpu().numpy(), expected.astype(np.float32), atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_get_timestep_embedding_dynamic_cuda_runtime_matches_reference_across_lengths(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    batch = dml.Dim("batch", min=1, max=4)
    spec = dml.trace(
        TimestepEmbeddingModule(
            embedding_dim=7,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.5,
            scale=0.75,
            max_period=64,
        ),
        inputs={"timesteps": dml.TensorSpec([batch], "float32")},
        name="get_timestep_embedding_dynamic_cuda",
    )
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch="sm_86"),
        tmp_path / "get_timestep_embedding_dynamic_cuda.dinoml",
    )
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "get_timestep_embedding_" in generated
    assert "sinf" in generated
    assert "cosf" in generated
    assert "concatenate_" not in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        for timesteps in (
            np.array([0.0, 1.5], dtype=np.float32),
            np.array([0.0, 1.25, 10.5, -0.75], dtype=np.float32),
        ):
            expected = _reference_get_timestep_embedding(
                timesteps,
                embedding_dim=7,
                flip_sin_to_cos=True,
                downscale_freq_shift=0.5,
                scale=0.75,
                max_period=64,
                dtype="float32",
            )
            actual = session.run_torch({"timesteps": torch.tensor(timesteps, device="cuda", dtype=torch.float32)})["out"]
            assert actual.dtype == torch.float32
            assert tuple(actual.shape) == expected.shape
            np.testing.assert_allclose(
                actual.float().cpu().numpy(),
                expected.astype(np.float32),
                atol=2e-6,
                rtol=1e-6,
            )
    finally:
        session.close()
        module.close()


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
@pytest.mark.parametrize(
    ("dtype", "torch_dtype", "atol", "rtol"),
    [
        ("float16", "float16", 2e-3, 2e-3),
        ("bfloat16", "bfloat16", 2e-2, 2e-2),
    ],
)
def test_get_timestep_embedding_reduced_precision_cuda_runtime_matches_reference(
    tmp_path,
    dtype,
    torch_dtype,
    atol,
    rtol,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    runtime_dtype = getattr(torch, torch_dtype)

    spec = _trace_timestep_embedding(
        dtype=dtype,
        embedding_dim=7,
        flip_sin_to_cos=(dtype == "bfloat16"),
        downscale_freq_shift=0.5,
        scale=0.75,
        max_period=64,
    )
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch="sm_86"),
        tmp_path / f"get_timestep_embedding_runtime_{dtype}_cuda.dinoml",
    )
    timesteps = np.array([0.0, 1.25, 10.5, -0.75], dtype=np.float32)
    expected = _reference_get_timestep_embedding(
        timesteps,
        embedding_dim=7,
        flip_sin_to_cos=(dtype == "bfloat16"),
        downscale_freq_shift=0.5,
        scale=0.75,
        max_period=64,
        dtype=dtype,
    )

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_torch({"timesteps": torch.tensor(timesteps, device="cuda", dtype=runtime_dtype)})["out"]
    session.close()
    module.close()

    assert actual.dtype == runtime_dtype
    np.testing.assert_allclose(actual.float().cpu().numpy(), expected.astype(np.float32), atol=atol, rtol=rtol)
