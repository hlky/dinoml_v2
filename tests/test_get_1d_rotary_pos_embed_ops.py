import numpy as np
import pytest
import shutil

import dinoml as dml
from dinoml import runtime
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import array_from_storage, array_to_storage, read_json
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.ops import collect_generated_sources
from dinoml.ops.definitions import get_op_def
from dinoml.passes import PassManager, validate_ir


COMPONENT_OPS = ("get_1d_rotary_pos_embed_cos", "get_1d_rotary_pos_embed_sin")


class RotaryIntModule(dml.Module):
    def __init__(
        self,
        *,
        dim: int,
        pos: int,
        theta: float = 10000.0,
        use_real: bool = True,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
        repeat_interleave_real: bool = True,
        dtype: str = "float32",
    ):
        self.dim = dim
        self.pos = pos
        self.theta = theta
        self.use_real = use_real
        self.linear_factor = linear_factor
        self.ntk_factor = ntk_factor
        self.repeat_interleave_real = repeat_interleave_real
        self.dtype = dtype

    def forward(self):
        cos_part, sin_part = dml.ops.get_1d_rotary_pos_embed(
            self.dim,
            self.pos,
            theta=self.theta,
            use_real=self.use_real,
            linear_factor=self.linear_factor,
            ntk_factor=self.ntk_factor,
            repeat_interleave_real=self.repeat_interleave_real,
            dtype=self.dtype,
        )
        return dml.ops.output(cos_part, "cos"), dml.ops.output(sin_part, "sin")


class RotaryTensorModule(dml.Module):
    def __init__(
        self,
        *,
        dim: int,
        theta: float = 10000.0,
        use_real: bool = True,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
        repeat_interleave_real: bool = True,
        dtype: str = "float32",
    ):
        self.dim = dim
        self.theta = theta
        self.use_real = use_real
        self.linear_factor = linear_factor
        self.ntk_factor = ntk_factor
        self.repeat_interleave_real = repeat_interleave_real
        self.dtype = dtype

    def forward(self, pos):
        cos_part, sin_part = dml.ops.get_1d_rotary_pos_embed(
            self.dim,
            pos,
            theta=self.theta,
            use_real=self.use_real,
            linear_factor=self.linear_factor,
            ntk_factor=self.ntk_factor,
            repeat_interleave_real=self.repeat_interleave_real,
            dtype=self.dtype,
        )
        return dml.ops.output(cos_part, "cos"), dml.ops.output(sin_part, "sin")


class MixedRotaryVariantModule(dml.Module):
    def forward(self, pos):
        int_cos, int_sin = dml.ops.get_1d_rotary_pos_embed(
            8,
            3,
            theta=4096.0,
            use_real=True,
            linear_factor=1.25,
            ntk_factor=1.1,
            repeat_interleave_real=True,
            dtype="float32",
        )
        tensor_cos, tensor_sin = dml.ops.get_1d_rotary_pos_embed(
            6,
            pos,
            theta=512.0,
            use_real=False,
            linear_factor=1.5,
            ntk_factor=0.75,
            repeat_interleave_real=False,
            dtype="float32",
        )
        return (
            dml.ops.output(int_cos, "int_cos"),
            dml.ops.output(int_sin, "int_sin"),
            dml.ops.output(tensor_cos, "tensor_cos"),
            dml.ops.output(tensor_sin, "tensor_sin"),
        )


def _trace_rotary_int(
    *,
    dim: int = 8,
    pos: int = 4,
    theta: float = 10000.0,
    use_real: bool = True,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    dtype: str = "float32",
):
    return dml.trace(
        RotaryIntModule(
            dim=dim,
            pos=pos,
            theta=theta,
            use_real=use_real,
            linear_factor=linear_factor,
            ntk_factor=ntk_factor,
            repeat_interleave_real=repeat_interleave_real,
            dtype=dtype,
        ),
        inputs={},
        name=f"get_1d_rotary_pos_embed_int_{dtype}_{dim}_{use_real}",
    )


def _trace_rotary_tensor(
    *,
    dim: int = 8,
    pos_shape=(4,),
    pos_dtype: str = "float32",
    theta: float = 10000.0,
    use_real: bool = True,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    dtype: str = "float32",
):
    return dml.trace(
        RotaryTensorModule(
            dim=dim,
            theta=theta,
            use_real=use_real,
            linear_factor=linear_factor,
            ntk_factor=ntk_factor,
            repeat_interleave_real=repeat_interleave_real,
            dtype=dtype,
        ),
        inputs={"pos": dml.TensorSpec(pos_shape, pos_dtype)},
        name=f"get_1d_rotary_pos_embed_tensor_{dtype}_{dim}_{use_real}",
    )


def _storage_roundtrip(value, dtype: str) -> np.ndarray:
    if dtype == "float32":
        return np.asarray(value, dtype=np.float32)
    return array_from_storage(array_to_storage(np.asarray(value, dtype=np.float32), dtype), dtype)


def _reference_get_1d_rotary_pos_embed(
    *,
    dim: int,
    positions: np.ndarray,
    theta: float,
    use_real: bool,
    linear_factor: float,
    ntk_factor: float,
    repeat_interleave_real: bool,
    dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    positions_fp32 = positions.astype(np.float32, copy=False)
    theta_scaled = np.float32(theta * ntk_factor)
    rotary_dim = dim // 2
    exponent = -np.log(theta_scaled) * np.arange(rotary_dim, dtype=np.float32) / np.float32(rotary_dim)
    inv_freqs = np.exp(exponent).astype(np.float32, copy=False) / np.float32(linear_factor)
    freqs = positions_fp32[:, None] * inv_freqs[None, :]
    cos_base = np.cos(freqs).astype(np.float32, copy=False)
    sin_base = np.sin(freqs).astype(np.float32, copy=False)
    if not use_real:
        return _storage_roundtrip(cos_base, dtype), _storage_roundtrip(sin_base, dtype)
    if repeat_interleave_real:
        cos_part = np.repeat(cos_base, 2, axis=1)
        sin_part = np.repeat(sin_base, 2, axis=1)
    else:
        cos_part = np.concatenate([cos_base, cos_base], axis=1)
        sin_part = np.concatenate([sin_base, sin_base], axis=1)
    return _storage_roundtrip(cos_part, dtype), _storage_roundtrip(sin_part, dtype)


def test_get_1d_rotary_pos_embed_int_frontend_lowers_to_two_generated_component_nodes_without_arange_helper():
    spec = _trace_rotary_int(
        dim=8,
        pos=5,
        theta=4096.0,
        use_real=True,
        linear_factor=1.25,
        ntk_factor=1.1,
        repeat_interleave_real=True,
        dtype="float16",
    )

    assert [output["shape"] for output in spec.ir["outputs"]] == [[5, 8], [5, 8]]
    assert [output["dtype"] for output in spec.ir["outputs"]] == ["float16", "float16"]
    assert [node["op"] for node in spec.ir["nodes"]] == list(COMPONENT_OPS)
    assert all(not node["inputs"] for node in spec.ir["nodes"])

    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    lowered_ops = [node["op"] for node in lowered["nodes"]]
    assert lowered_ops == list(COMPONENT_OPS)
    for helper_op in {"concatenate", "repeat_interleave", "cos", "sin", "fused_elementwise"}:
        assert helper_op not in lowered_ops


def test_get_1d_rotary_pos_embed_dynamic_tensor_pos_and_use_real_false_preserve_shape_spec():
    seq = dml.Dim("seq", min=1, max=4)
    spec = dml.trace(
        RotaryTensorModule(
            dim=6,
            theta=64.0,
            use_real=False,
            linear_factor=1.5,
            ntk_factor=1.25,
            repeat_interleave_real=False,
            dtype="float32",
        ),
        inputs={"pos": dml.TensorSpec([seq], "float32")},
        name="get_1d_rotary_pos_embed_dynamic_use_real_false",
    )

    assert [node["op"] for node in spec.ir["nodes"]] == list(COMPONENT_OPS)
    assert [output["shape"] for output in spec.ir["outputs"]] == [[4, 3], [4, 3]]
    assert [output["shape_spec"] for output in spec.ir["outputs"]] == [[seq.to_json(), 3], [seq.to_json(), 3]]

    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == list(COMPONENT_OPS)


@pytest.mark.parametrize("op_name", COMPONENT_OPS)
def test_get_1d_rotary_pos_embed_component_registry_truthfully_reports_zero_or_one_input(op_name):
    op_def = get_op_def(op_name)

    assert op_def.accepts_input_count(0)
    assert op_def.accepts_input_count(1)
    assert not op_def.accepts_input_count(2)


def test_get_1d_rotary_pos_embed_manifest_and_generated_sources_are_two_model_owned_component_kernels():
    spec = _trace_rotary_tensor(
        dim=8,
        pos_shape=(5,),
        pos_dtype="float32",
        theta=4096.0,
        use_real=True,
        linear_factor=1.25,
        ntk_factor=1.1,
        repeat_interleave_real=False,
        dtype="float32",
    )
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)

    manifest = build_kernel_manifest(lowered, {"name": "cpu", "arch": "native"})
    assert [item["op"] for item in manifest["required_kernels"]] == list(COMPONENT_OPS)
    for item in manifest["required_kernels"]:
        assert item["kernel_symbol"] == "generated_get_1d_rotary_pos_embed"
        assert item["kernel_library"] == "model"
        assert item["profiler_symbol"] is None
        assert item["has_profiler"] is False
        assert item["generated_source"]["generated_function_name"].startswith(f"{item['op']}_")
        assert item["generated_source"]["source_key"].startswith("cpu:")
        assert len(item["generated_source"]["source_hash"]) == 16

    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    assert len(sources["kernels"]) == 2
    combined = "\n".join(sources["kernels"])
    assert "get_1d_rotary_pos_embed_cos_" in combined
    assert "get_1d_rotary_pos_embed_sin_" in combined
    assert "sinf" in combined
    assert "cosf" in combined
    assert "generated_concatenate" not in combined
    assert "generated_repeat_interleave" not in combined


def test_get_1d_rotary_pos_embed_mixed_variants_keep_distinct_model_generated_provenance(tmp_path):
    spec = dml.trace(
        MixedRotaryVariantModule(),
        inputs={"pos": dml.TensorSpec([4], "float32")},
        name="get_1d_rotary_pos_embed_mixed_variants",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "get_1d_rotary_pos_embed_mixed_variants_cpu.dinoml")

    kernel_manifest = read_json(artifact.path / "kernel_manifest.json")
    codegen_plan = read_json(artifact.path / "kernel_codegen_plan.json")
    source_manifest = read_json(artifact.path / "debug" / "generated_src" / "source_manifest.json")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")

    required = kernel_manifest["required_kernels"]
    assert [item["op"] for item in required] == [
        "get_1d_rotary_pos_embed_cos",
        "get_1d_rotary_pos_embed_sin",
        "get_1d_rotary_pos_embed_cos",
        "get_1d_rotary_pos_embed_sin",
    ]
    assert all(item["kernel_symbol"] == "generated_get_1d_rotary_pos_embed" for item in required)
    assert len({item["generated_source"]["source_key"] for item in required}) == 4
    assert len({item["generated_source"]["generated_function_name"] for item in required}) == 4

    assert len(codegen_plan["generated_sources"]) == 4
    assert len({entry["source_key"] for entry in codegen_plan["generated_sources"]}) == 4
    assert len({entry["generated_function_name"] for entry in codegen_plan["generated_sources"]}) == 4

    assert len(source_manifest["sources"]) == 4
    assert len({entry["source_key"] for entry in source_manifest["sources"]}) == 4
    assert generated.count("static int get_1d_rotary_pos_embed_cos_") == 2
    assert generated.count("static int get_1d_rotary_pos_embed_sin_") == 2


@pytest.mark.parametrize(
    ("dtype", "use_real", "repeat_interleave_real", "theta", "linear_factor", "ntk_factor", "atol", "rtol"),
    [
        ("float32", True, True, 10000.0, 1.0, 1.0, 1e-6, 1e-6),
        ("float32", True, False, 4096.0, 2.0, 1.5, 1e-6, 1e-6),
        ("float16", True, True, 1000.0, 1.25, 1.1, 2e-3, 2e-3),
        ("bfloat16", True, False, 256.0, 0.75, 1.25, 2e-2, 2e-2),
        ("float32", False, True, 512.0, 1.25, 0.75, 1e-6, 1e-6),
    ],
)
def test_cpu_reference_get_1d_rotary_pos_embed_matches_formula(
    dtype,
    use_real,
    repeat_interleave_real,
    theta,
    linear_factor,
    ntk_factor,
    atol,
    rtol,
):
    spec = _trace_rotary_int(
        dim=8,
        pos=4,
        theta=theta,
        use_real=use_real,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
        repeat_interleave_real=repeat_interleave_real,
        dtype=dtype,
    )
    positions = np.arange(4, dtype=np.float32)
    expected_cos, expected_sin = _reference_get_1d_rotary_pos_embed(
        dim=8,
        positions=positions,
        theta=theta,
        use_real=use_real,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
        repeat_interleave_real=repeat_interleave_real,
        dtype=dtype,
    )

    actual = execute_cpu(spec, {})

    assert actual["cos"].dtype == expected_cos.dtype
    assert actual["sin"].dtype == expected_sin.dtype
    np.testing.assert_allclose(actual["cos"].astype(np.float32), expected_cos.astype(np.float32), atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual["sin"].astype(np.float32), expected_sin.astype(np.float32), atol=atol, rtol=rtol)


def test_cpu_reference_get_1d_rotary_pos_embed_matches_formula_for_static_tensor_positions():
    spec = _trace_rotary_tensor(
        dim=6,
        pos_shape=(3,),
        pos_dtype="float16",
        theta=64.0,
        use_real=False,
        linear_factor=1.5,
        ntk_factor=1.25,
        repeat_interleave_real=False,
        dtype="float32",
    )
    positions = np.array([0.0, 1.5, 3.25], dtype=np.float32)
    expected_cos, expected_sin = _reference_get_1d_rotary_pos_embed(
        dim=6,
        positions=positions,
        theta=64.0,
        use_real=False,
        linear_factor=1.5,
        ntk_factor=1.25,
        repeat_interleave_real=False,
        dtype="float32",
    )

    actual = execute_cpu(spec, {"pos": positions})

    np.testing.assert_allclose(actual["cos"], expected_cos, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual["sin"], expected_sin, atol=1e-6, rtol=1e-6)


def test_cpu_artifact_runs_two_component_rotary_kernels_with_dynamic_pos_length_and_use_real_false(tmp_path):
    seq = dml.Dim("seq", min=1, max=4)
    spec = dml.trace(
        RotaryTensorModule(
            dim=6,
            theta=64.0,
            use_real=False,
            linear_factor=1.5,
            ntk_factor=1.25,
            repeat_interleave_real=False,
            dtype="float32",
        ),
        inputs={"pos": dml.TensorSpec([seq], "float32")},
        name="get_1d_rotary_pos_embed_dynamic_cpu",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "get_1d_rotary_pos_embed_dynamic_cpu.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "get_1d_rotary_pos_embed_cos_" in generated
    assert "get_1d_rotary_pos_embed_sin_" in generated
    assert "generated_repeat_interleave" not in generated
    assert "generated_concatenate" not in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    for positions in (
        np.array([0.0, 1.5], dtype=np.float32),
        np.array([0.0, 1.25, 3.25, 7.5], dtype=np.float32),
    ):
        expected_cos, expected_sin = _reference_get_1d_rotary_pos_embed(
            dim=6,
            positions=positions,
            theta=64.0,
            use_real=False,
            linear_factor=1.5,
            ntk_factor=1.25,
            repeat_interleave_real=False,
            dtype="float32",
        )
        actual = session.run_numpy({"pos": positions})
        assert actual["cos"].shape == expected_cos.shape
        assert actual["sin"].shape == expected_sin.shape
        np.testing.assert_allclose(actual["cos"], expected_cos, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(actual["sin"], expected_sin, atol=1e-6, rtol=1e-6)
    session.close()
    module.close()


def test_cpu_artifact_runs_no_input_integer_pos_rotary_component_kernels_without_helper_composition(tmp_path):
    spec = _trace_rotary_int(
        dim=8,
        pos=5,
        theta=4096.0,
        use_real=True,
        linear_factor=1.25,
        ntk_factor=1.1,
        repeat_interleave_real=False,
        dtype="float16",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "get_1d_rotary_pos_embed_static_int_cpu.dinoml")
    kernel_manifest = read_json(artifact.path / "kernel_manifest.json")
    codegen_plan = read_json(artifact.path / "kernel_codegen_plan.json")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")

    assert len(kernel_manifest["required_kernels"]) == 2
    assert [item["op"] for item in kernel_manifest["required_kernels"]] == list(COMPONENT_OPS)
    assert len(codegen_plan["generated_sources"]) == 2
    assert "get_1d_rotary_pos_embed_cos_" in generated
    assert "get_1d_rotary_pos_embed_sin_" in generated
    assert "generated_arange" not in generated
    assert "generated_repeat_interleave" not in generated
    assert "generated_concatenate" not in generated

    expected_cos, expected_sin = _reference_get_1d_rotary_pos_embed(
        dim=8,
        positions=np.arange(5, dtype=np.float32),
        theta=4096.0,
        use_real=True,
        linear_factor=1.25,
        ntk_factor=1.1,
        repeat_interleave_real=False,
        dtype="float16",
    )

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({})
    session.close()
    module.close()

    np.testing.assert_allclose(actual["cos"].astype(np.float32), expected_cos.astype(np.float32), atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(actual["sin"].astype(np.float32), expected_sin.astype(np.float32), atol=2e-3, rtol=2e-3)


def test_get_1d_rotary_pos_embed_frontend_rejects_unsupported_inputs():
    with pytest.raises(ValueError, match="positive integer"):
        _trace_rotary_int(dim=0)
    with pytest.raises(ValueError, match="even dim"):
        _trace_rotary_int(dim=7)
    with pytest.raises(ValueError, match="does not support dtype bool"):
        _trace_rotary_int(dtype="bool")
    with pytest.raises(ValueError, match="positive sequence length"):
        _trace_rotary_int(pos=0)
    with pytest.raises(ValueError, match="rank-1 pos tensor"):
        _trace_rotary_tensor(pos_shape=(2, 2))
    with pytest.raises(ValueError, match="does not support pos dtype bool"):
        _trace_rotary_tensor(pos_dtype="bool")
    with pytest.raises(ValueError, match="theta must be a positive finite number"):
        _trace_rotary_int(theta=float("inf"))
    with pytest.raises(ValueError, match="linear_factor must be a positive finite number"):
        _trace_rotary_int(linear_factor=0.0)
    with pytest.raises(ValueError, match="ntk_factor must be a positive finite number"):
        _trace_rotary_int(ntk_factor=-1.0)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
@pytest.mark.parametrize(
    ("use_real", "repeat_interleave_real"),
    [
        (True, False),
        (False, False),
    ],
)
def test_get_1d_rotary_pos_embed_float32_cuda_artifact_compiles_as_two_component_kernels(
    tmp_path,
    use_real,
    repeat_interleave_real,
):
    spec = _trace_rotary_tensor(
        dim=8,
        pos_shape=(5,),
        pos_dtype="float32",
        theta=4096.0,
        use_real=use_real,
        linear_factor=1.25,
        ntk_factor=1.1,
        repeat_interleave_real=repeat_interleave_real,
        dtype="float32",
    )
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch="sm_86"),
        tmp_path / f"get_1d_rotary_pos_embed_{use_real}_cuda.dinoml",
    )
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "get_1d_rotary_pos_embed_cos_" in generated
    assert "get_1d_rotary_pos_embed_sin_" in generated
    assert "generated_repeat_interleave" not in generated
    assert "generated_concatenate" not in generated


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_get_1d_rotary_pos_embed_float32_cuda_runtime_matches_reference_for_use_real_false(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    spec = _trace_rotary_tensor(
        dim=8,
        pos_shape=(4,),
        pos_dtype="float32",
        theta=512.0,
        use_real=False,
        linear_factor=1.25,
        ntk_factor=0.75,
        repeat_interleave_real=False,
        dtype="float32",
    )
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch="sm_86"),
        tmp_path / "get_1d_rotary_pos_embed_runtime_float32_cuda.dinoml",
    )
    positions = np.array([0.0, 1.25, 10.5, -0.75], dtype=np.float32)
    expected_cos, expected_sin = _reference_get_1d_rotary_pos_embed(
        dim=8,
        positions=positions,
        theta=512.0,
        use_real=False,
        linear_factor=1.25,
        ntk_factor=0.75,
        repeat_interleave_real=False,
        dtype="float32",
    )

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_torch({"pos": torch.tensor(positions, device="cuda", dtype=torch.float32)})
    session.close()
    module.close()

    assert actual["cos"].dtype == torch.float32
    assert actual["sin"].dtype == torch.float32
    np.testing.assert_allclose(actual["cos"].float().cpu().numpy(), expected_cos.astype(np.float32), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual["sin"].float().cpu().numpy(), expected_sin.astype(np.float32), atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_get_1d_rotary_pos_embed_dynamic_tensor_pos_cuda_runtime_matches_reference_across_lengths(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    seq = dml.Dim("seq", min=1, max=4)
    spec = dml.trace(
        RotaryTensorModule(
            dim=8,
            theta=4096.0,
            use_real=True,
            linear_factor=1.25,
            ntk_factor=1.1,
            repeat_interleave_real=False,
            dtype="float32",
        ),
        inputs={"pos": dml.TensorSpec([seq], "float32")},
        name="get_1d_rotary_pos_embed_dynamic_cuda",
    )
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch="sm_86"),
        tmp_path / "get_1d_rotary_pos_embed_dynamic_cuda.dinoml",
    )
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "get_1d_rotary_pos_embed_cos_" in generated
    assert "get_1d_rotary_pos_embed_sin_" in generated
    assert "generated_repeat_interleave" not in generated
    assert "generated_concatenate" not in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        for positions in (
            np.array([0.0, 1.5], dtype=np.float32),
            np.array([0.0, 1.25, 3.25, 7.5], dtype=np.float32),
        ):
            expected_cos, expected_sin = _reference_get_1d_rotary_pos_embed(
                dim=8,
                positions=positions,
                theta=4096.0,
                use_real=True,
                linear_factor=1.25,
                ntk_factor=1.1,
                repeat_interleave_real=False,
                dtype="float32",
            )
            actual = session.run_numpy({"pos": positions})
            assert actual["cos"].shape == expected_cos.shape
            assert actual["sin"].shape == expected_sin.shape
            np.testing.assert_allclose(actual["cos"], expected_cos, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(actual["sin"], expected_sin, atol=1e-6, rtol=1e-6)
    finally:
        session.close()
        module.close()


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_get_1d_rotary_pos_embed_no_input_integer_pos_cuda_runtime_matches_reference(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    spec = _trace_rotary_int(
        dim=6,
        pos=4,
        theta=512.0,
        use_real=False,
        linear_factor=1.5,
        ntk_factor=0.75,
        repeat_interleave_real=False,
        dtype="float32",
    )
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch="sm_86"),
        tmp_path / "get_1d_rotary_pos_embed_static_int_cuda.dinoml",
    )
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "get_1d_rotary_pos_embed_cos_" in generated
    assert "get_1d_rotary_pos_embed_sin_" in generated
    assert "generated_arange" not in generated
    assert "generated_repeat_interleave" not in generated
    assert "generated_concatenate" not in generated

    expected_cos, expected_sin = _reference_get_1d_rotary_pos_embed(
        dim=6,
        positions=np.arange(4, dtype=np.float32),
        theta=512.0,
        use_real=False,
        linear_factor=1.5,
        ntk_factor=0.75,
        repeat_interleave_real=False,
        dtype="float32",
    )

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({})
    session.close()
    module.close()

    np.testing.assert_allclose(actual["cos"], expected_cos, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual["sin"], expected_sin, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
@pytest.mark.parametrize(
    ("dtype", "torch_dtype", "atol", "rtol"),
    [
        ("float16", "float16", 2e-3, 2e-3),
        ("bfloat16", "bfloat16", 2e-2, 2e-2),
    ],
)
def test_get_1d_rotary_pos_embed_reduced_precision_cuda_runtime_matches_reference(
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

    spec = _trace_rotary_tensor(
        dim=8,
        pos_shape=(4,),
        pos_dtype="float32",
        theta=1000.0,
        use_real=True,
        linear_factor=1.25,
        ntk_factor=1.1,
        repeat_interleave_real=(dtype == "float16"),
        dtype=dtype,
    )
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch="sm_86"),
        tmp_path / f"get_1d_rotary_pos_embed_runtime_{dtype}_cuda.dinoml",
    )
    positions = np.array([0.0, 1.25, 10.5, -0.75], dtype=np.float32)
    expected_cos, expected_sin = _reference_get_1d_rotary_pos_embed(
        dim=8,
        positions=positions,
        theta=1000.0,
        use_real=True,
        linear_factor=1.25,
        ntk_factor=1.1,
        repeat_interleave_real=(dtype == "float16"),
        dtype=dtype,
    )

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_torch({"pos": torch.tensor(positions, device="cuda", dtype=torch.float32)})
    session.close()
    module.close()

    assert actual["cos"].dtype == runtime_dtype
    assert actual["sin"].dtype == runtime_dtype
    np.testing.assert_allclose(actual["cos"].float().cpu().numpy(), expected_cos.astype(np.float32), atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual["sin"].float().cpu().numpy(), expected_sin.astype(np.float32), atol=atol, rtol=rtol)
