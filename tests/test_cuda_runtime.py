import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import dinoml as dml
from dinoml import runtime
from dinoml.backends.cuda_libraries import discover_cuda_libraries
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import ModelSpec, read_json
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_candidates


DEFAULT_CUDA_TARGET = {"name": "cuda", "arch": "sm_86"}


def _cutlass_default_symbol_id(dtype: str, *, op_name: str = "gemm_rrr", target=None) -> str:
    return str(cutlass_gemm_candidates(op_name, dtype, target=target or DEFAULT_CUDA_TARGET)[0]["symbol_id"])


def _cutlass_default_candidate_id(dtype: str, *, op_name: str = "gemm_rrr", target=None) -> str:
    return str(cutlass_gemm_candidates(op_name, dtype, target=target or DEFAULT_CUDA_TARGET)[0]["candidate_id"])


def _cutlass_candidate_count(dtype: str, *, op_name: str = "gemm_rrr", target=None) -> int:
    return len(cutlass_gemm_candidates(op_name, dtype, target=target or DEFAULT_CUDA_TARGET))


pytestmark = pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _write_minimal_gguf_tensor(path, *, name, gguf_shape, qtype_value, payload):
    data = bytearray()
    data += b"GGUF"
    data += struct.pack("<IQQ", 3, 1, 0)
    data += _gguf_string(name)
    data += struct.pack("<I", len(gguf_shape))
    data += struct.pack("<" + "Q" * len(gguf_shape), *gguf_shape)
    data += struct.pack("<IQ", int(qtype_value), 0)
    data += b"\0" * ((32 - len(data) % 32) % 32)
    data += payload
    path.write_bytes(data)


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
    assert "dino_module_unload_constants" in generated_text
    assert "dino_module_load_deferred" in generated_text
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
    assert hasattr(module._dll, "dino_module_unload_constants")
    assert hasattr(module._dll, "dino_module_load_deferred")
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
    expected_loaded = {constant["name"]: True for constant in spec.ir["constants"]}
    expected_unloaded = {constant["name"]: False for constant in spec.ir["constants"]}

    module = runtime.load(artifact.path)
    assert module.constant_load_state() == expected_loaded
    session = module.create_session()
    module.set_constant_numpy("scale", np.zeros_like(spec.constants["scale"]))
    module.set_constant_numpy("bias", np.zeros_like(spec.constants["bias"]))
    assert module.constant_load_state() == expected_loaded
    actual = session.run_numpy(inputs)
    module.unload_constants()
    assert module.constant_load_state() == expected_unloaded
    with pytest.raises(RuntimeError, match="Constant scale has not been loaded"):
        session.run_numpy(inputs)
    module.load_constants_from_file()
    assert module.constant_load_state() == expected_loaded
    reloaded = session.run_numpy(inputs)
    session.close()
    module.close()

    np.testing.assert_allclose(actual["y"], np.zeros([2, 3, 4], dtype=np.float32), atol=1e-6, rtol=0)
    np.testing.assert_allclose(reloaded["y"], execute_cpu(spec, inputs)["y"], atol=1e-4, rtol=1e-4)

    module = runtime.load(artifact.path, load_constants=False)
    assert module.constant_load_state() == expected_unloaded
    session = module.create_session()
    try:
        with pytest.raises(RuntimeError, match="Constant scale has not been loaded"):
            session.run_numpy(inputs)
        module.load_constants_from_file()
        assert module.constant_load_state() == expected_loaded
        deferred = session.run_numpy(inputs)
    finally:
        session.close()
        module.close()
    np.testing.assert_allclose(deferred["y"], execute_cpu(spec, inputs)["y"], atol=1e-4, rtol=1e-4)


def test_cuda_runtime_mixed_dense_and_manual_encoded_constant_reload_requires_explicit_encoded_load(
    monkeypatch, tmp_path
):
    class MixedConstantModel(dml.Module):
        def __init__(self):
            self.dense_bias = dml.Parameter([2], dtype="float32")
            self.manual_bias = dml.Parameter([2], dtype="float32")

        def forward(self, x):
            return dml.ops.output(x + self.dense_bias + self.manual_bias, "y")

    dense_bias = np.array([0.5, -0.25], dtype=np.float32)
    manual_bias = np.array([1.5, 2.0], dtype=np.float32)
    x = np.array([3.0, -1.0], dtype=np.float32)

    tensor_info = SimpleNamespace(
        name="blk.0.ffn.manual_bias",
        qtype="F32",
        qtype_value=0,
        shape=(2,),
        data_offset=64,
    )
    fake_file = SimpleNamespace(
        get_tensor=lambda name: tensor_info,
        read_tensor_bytes=lambda tensor: manual_bias.tobytes(order="C"),
    )
    monkeypatch.setitem(sys.modules, "libgguf", SimpleNamespace(open_gguf=lambda path: fake_file))

    source = dml.gguf_constant(
        tmp_path / "weights.gguf",
        "blk.0.ffn.manual_bias",
        residency="manual_runtime_load",
    )
    traced = dml.trace(
        MixedConstantModel(),
        inputs={"x": dml.TensorSpec([2], "float32")},
        constants={"dense_bias": dense_bias, "manual_bias": source},
        name="mixed_dense_manual_encoded_constants_cuda",
    )
    spec = ModelSpec(
        name=traced.name,
        ir=traced.ir,
        constants={"dense_bias": dense_bias, "manual_bias": source},
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "mixed_dense_manual_encoded_constants_cuda.dinoml")

    module = runtime.load(artifact.path)
    session = module.create_session()
    expected = x + dense_bias + manual_bias
    try:
        assert module.constant_load_state() == {"dense_bias": True, "manual_bias": False}
        with pytest.raises(RuntimeError, match="Constant manual_bias has not been loaded"):
            session.run_numpy({"x": x})

        module.load_encoded_constants(names=["manual_bias"])
        assert module.constant_load_state() == {"dense_bias": True, "manual_bias": True}
        loaded = session.run_numpy({"x": x})

        module.unload_constants()
        assert module.constant_load_state() == {"dense_bias": False, "manual_bias": False}
        module.load_constants_from_file()
        assert module.constant_load_state() == {"dense_bias": True, "manual_bias": False}
        with pytest.raises(RuntimeError, match="Constant manual_bias has not been loaded"):
            session.run_numpy({"x": x})

        module.load_encoded_constants(names=["manual_bias"])
        assert module.constant_load_state() == {"dense_bias": True, "manual_bias": True}
        reloaded = session.run_numpy({"x": x})
    finally:
        session.close()
        module.close()

    np.testing.assert_allclose(loaded["y"], expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(reloaded["y"], expected, atol=1e-5, rtol=1e-5)


def test_cuda_runtime_load_encoded_constants_from_real_libgguf_q4_0_uses_device_dequant(
    monkeypatch, tmp_path
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    libgguf = pytest.importorskip("libgguf")
    pytest.importorskip("libgguf.libgguf_cuda")
    if not hasattr(torch.ops, "_C_gguf") or not hasattr(torch.ops._C_gguf, "dequantize"):
        pytest.skip("libgguf CUDA dequantize op is not registered")

    class EncodedRowsWeightModel(dml.Module):
        def __init__(self):
            self.weight = dml.Parameter([2, 32], dtype="float32")

        def forward(self, x):
            return dml.ops.output(x + self.weight, "y")

    rows = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(2, 32)
    qtype = libgguf.GGMLQuantizationType.Q4_0
    encoded = libgguf.quantize_rows(rows, qtype)
    expected_weight = libgguf.dequantize_rows(encoded, qtype, n_per_row=32).reshape(2, 32)
    gguf_path = tmp_path / "weights.gguf"
    _write_minimal_gguf_tensor(
        gguf_path,
        name="blk.0.ffn.weight",
        gguf_shape=(32, 2),
        qtype_value=int(qtype),
        payload=encoded.tobytes(order="C"),
    )

    source = dml.gguf_constant(
        gguf_path,
        "blk.0.ffn.weight",
        qtype="Q4_0",
        encoded_nbytes=encoded.nbytes,
        n_per_row=32,
        residency="manual_runtime_load",
    )
    traced = dml.trace(
        EncodedRowsWeightModel(),
        inputs={"x": dml.TensorSpec([2, 32], "float32")},
        constants={"weight": source},
        name="encoded_weight_real_libgguf_cuda",
    )
    spec = ModelSpec(name=traced.name, ir=traced.ir, constants={"weight": source})
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch="sm_86"),
        tmp_path / "encoded_weight_real_libgguf_cuda.dinoml",
        constant_load_policy="deferred",
    )

    def fail_cpu_dequantize_rows(*args, **kwargs):
        raise AssertionError("runtime load should use libgguf CUDA dequantize")

    monkeypatch.setattr(libgguf, "dequantize_rows", fail_cpu_dequantize_rows)

    x = np.linspace(-0.25, 0.5, 64, dtype=np.float32).reshape(2, 32)
    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        assert module.constant_load_state() == {"weight": False}
        with pytest.raises(RuntimeError, match="Constant weight has not been loaded"):
            session.run_numpy({"x": x})

        module.load_encoded_constants(names=["weight"])
        assert module.constant_load_state() == {"weight": True}
        loaded = session.run_numpy({"x": x})

        module.unload_constants()
        assert module.constant_load_state() == {"weight": False}
        module.load_constants_from_file()
        assert module.constant_load_state() == {"weight": False}
        with pytest.raises(RuntimeError, match="Constant weight has not been loaded"):
            session.run_numpy({"x": x})

        module.load_encoded_constants()
        assert module.constant_load_state() == {"weight": True}
        reloaded = session.run_numpy({"x": x})
    finally:
        session.close()
        module.close()

    np.testing.assert_allclose(loaded["y"], x + expected_weight, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(reloaded["y"], x + expected_weight, atol=0.0, rtol=0.0)


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


class GemmModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(a, b), "y")


class GemmBiasModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b, bias):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(a, b, bias), "y")


class BmmAddModule(dml.Module):
    def forward(self, a, b, d0):
        return dml.ops.output(dml.ops.bmm_rrr_add(a, b, d0), "y")


class IdentityModule(dml.Module):
    def forward(self, x):
        return dml.ops.output(x, "y")


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
    compile_constants = {
        "scale": np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float32),
        "bias": np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32),
    }
    runtime_constants = {
        "scale": np.array([1.25, -0.75, 0.5, 2.0], dtype=np.float32),
        "bias": np.array([-0.4, 0.6, 0.15, -0.2], dtype=np.float32),
    }
    spec = dml.trace(
        DTypeFusedElementwise(dtype),
        inputs={"x": dml.TensorSpec([2, 3, 4], dtype)},
        constants=compile_constants,
        name=f"fused_elementwise_{dtype}",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / f"fused_elementwise_{dtype}.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert ("half" if dtype == "float16" else "__nv_bfloat16") in generated

    torch.manual_seed(123)
    x = torch.randn((2, 3, 4), device="cuda", dtype=torch_dtype_obj)
    scale = torch.tensor(runtime_constants["scale"], device="cuda", dtype=torch_dtype_obj)
    bias = torch.tensor(runtime_constants["bias"], device="cuda", dtype=torch_dtype_obj)
    expected = torch.relu(x.float() * scale.float() + bias.float()) * 0.5
    expected = expected.to(torch_dtype_obj).float().cpu().numpy()

    module = runtime.load(artifact.path)
    module.set_constant_device_pointer("scale", scale.data_ptr(), tuple(scale.shape), "fp16" if dtype == "float16" else "bf16")
    module.set_constant_torch("bias", bias)
    session = module.create_session()
    actual_raw = torch.empty_like(x)
    session.run_device_pointers(
        {"x": x.data_ptr()},
        {"y": actual_raw.data_ptr()},
        {"x": tuple(int(dim) for dim in x.shape)},
        {"y": tuple(int(dim) for dim in actual_raw.shape)},
    )
    module.set_constant_numpy("bias", runtime_constants["bias"])
    actual_torch = session.run_torch({"x": x})["y"]
    actual_numpy = session.run_numpy({"x": x.float().cpu().numpy()})["y"]
    session.close()
    module.close()

    np.testing.assert_allclose(actual_raw.float().cpu().numpy(), expected, atol=atol, rtol=rtol)
    assert actual_torch.dtype == torch_dtype_obj
    np.testing.assert_allclose(actual_torch.float().cpu().numpy(), expected, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual_numpy.astype(np.float32), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("op_name", "a_shape", "b_shape", "dtype", "torch_dtype", "suffix", "atol", "rtol"),
    [
        ("gemm_rrr", (16, 32), (32, 24), "float32", "float32", "float32", 1e-2, 1e-2),
        ("gemm_rcr", (16, 32), (24, 32), "float32", "float32", "float32", 1e-2, 1e-2),
        ("gemm_rrr", (16, 32), (32, 24), "float16", "float16", "float16", 2e-2, 2e-2),
        ("gemm_rcr", (16, 32), (24, 32), "float16", "float16", "float16", 2e-2, 2e-2),
        ("gemm_rrr", (16, 32), (32, 24), "bfloat16", "bfloat16", "bfloat16", 3e-2, 3e-2),
        ("gemm_rcr", (16, 32), (24, 32), "bfloat16", "bfloat16", "bfloat16", 3e-2, 3e-2),
    ],
)
def test_cuda_cutlass_gemm_runtime_matches_torch(tmp_path, monkeypatch, op_name, a_shape, b_shape, dtype, torch_dtype, suffix, atol, rtol):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    if not discover_cuda_libraries()["cutlass"].available:
        pytest.skip("CUTLASS headers are not available")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    torch_dtype_obj = getattr(torch, torch_dtype)
    spec = dml.trace(
        GemmModule(op_name),
        inputs={"a": dml.TensorSpec(a_shape, dtype), "b": dml.TensorSpec(b_shape, dtype)},
        name=f"{op_name}_{dtype}_cutlass_cuda",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / f"{op_name}_{dtype}_cutlass_cuda.dinoml")
    manifest = read_json(artifact.path / "manifest.json")
    kernel_manifest = read_json(artifact.path / "kernel_manifest.json")
    source_manifest = read_json(artifact.path / "debug" / "generated_src" / "source_manifest.json")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")

    assert manifest["files"]["cutlass_gemm_library"] == "lib/libdinoml_cutlass_gemm.so"
    assert (artifact.path / "lib" / "libdinoml_cutlass_gemm.so").exists()
    assert kernel_manifest["required_kernels"][0]["kernel_library"] == "cutlass_gemm"
    symbol = f"dinoml_cutlass_{op_name}_{suffix}_{_cutlass_default_symbol_id(dtype)}"
    assert kernel_manifest["required_kernels"][0]["kernel_symbol"] == symbol
    assert kernel_manifest["required_kernels"][0]["candidate_set_id"] == f"cutlass_{op_name}_{suffix}_linear_combination_v1"
    assert kernel_manifest["required_kernels"][0]["candidate_set"]["candidate_count"] == _cutlass_candidate_count(dtype, op_name=op_name)
    assert kernel_manifest["required_kernels"][0]["selected_candidate_id"] == _cutlass_default_candidate_id(dtype)
    assert len(kernel_manifest["required_kernels"][0]["candidates"]) == _cutlass_candidate_count(dtype, op_name=op_name)
    assert kernel_manifest["required_kernels"][0]["candidates"][0]["candidate_id"] == _cutlass_default_candidate_id(dtype)
    assert kernel_manifest["required_kernels"][0]["candidates"][0]["kernel_symbol"] == symbol
    assert source_manifest["sources"] == []
    assert symbol in generated

    torch.manual_seed(47)
    a = torch.randn(a_shape, device="cuda", dtype=torch_dtype_obj)
    b = torch.randn(b_shape, device="cuda", dtype=torch_dtype_obj)
    expected = a @ (b if op_name == "gemm_rrr" else b.t())

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual_torch = session.run_torch({"a": a, "b": b})["y"]
    actual_numpy = session.run_numpy({"a": a.float().cpu().numpy(), "b": b.float().cpu().numpy()})["y"]
    session.close()
    module.close()

    assert actual_torch.dtype == torch_dtype_obj
    torch.testing.assert_close(actual_torch, expected, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual_numpy.astype(np.float32), expected.float().cpu().numpy(), atol=atol, rtol=rtol)


def test_cuda_runtime_materializes_reported_smaller_output_shape(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")

    x = torch.arange(8, device="cuda", dtype=torch.float32).reshape(2, 4)
    spec = dml.trace(IdentityModule(), inputs={"x": dml.TensorSpec([2, 4], "float32")}, name="materialize_output_shape_cuda")
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "materialize_output_shape_cuda.dinoml")
    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        monkeypatch.setattr(session, "get_output_shape", lambda _name: (1, 4))
        actual = session.run_torch({"x": x})["y"]
        assert actual.shape == (1, 4)
        torch.testing.assert_close(actual, x[:1], rtol=0, atol=0)

        device_out = torch.empty((2, 4), device="cuda", dtype=torch.float32)
        session.run_device_pointers(
            {"x": x.data_ptr()},
            {"y": device_out.data_ptr()},
            {"x": tuple(int(dim) for dim in x.shape)},
            {"y": (2, 4)},
        )
        reported_shape = session.get_output_shape("y")
        assert reported_shape == (1, 4)
        torch.testing.assert_close(device_out[:1], x[:1], rtol=0, atol=0)

        monkeypatch.setattr(session, "get_output_shape", lambda _name: (4, 2))
        actual = session.run_torch({"x": x})["y"]
        assert actual.shape == (4, 2)
        torch.testing.assert_close(actual, x.reshape(4, 2), rtol=0, atol=0)

        monkeypatch.setattr(session, "get_output_shape", lambda _name: (3, 4))
        direct_out = torch.empty((2, 4), device="cuda", dtype=torch.float32)
        with pytest.raises(ValueError, match="has more elements than allocated"):
            session.run_device_pointers(
                {"x": x.data_ptr()},
                {"y": direct_out.data_ptr()},
                {"x": tuple(int(dim) for dim in x.shape)},
                {"y": (2, 4)},
            )
        with pytest.raises(ValueError, match="has more elements than allocated"):
            session.run_torch({"x": x})
    finally:
        session.close()
        module.close()


@pytest.mark.parametrize(
    ("d0_shape",),
    [
        ((2, 4, 6),),
        ((6,),),
        ((1, 6),),
    ],
)
def test_cuda_cutlass_bmm_add_runtime_matches_torch(tmp_path, monkeypatch, d0_shape):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    if not discover_cuda_libraries()["cutlass"].available:
        pytest.skip("CUTLASS headers are not available")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    spec = dml.trace(
        BmmAddModule(),
        inputs={
            "a": dml.TensorSpec([2, 4, 8], "float32"),
            "b": dml.TensorSpec([2, 8, 6], "float32"),
            "d0": dml.TensorSpec(d0_shape, "float32"),
        },
        name="bmm_rrr_add_float32_cutlass_cuda",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "bmm_rrr_add_float32_cutlass_cuda.dinoml")
    manifest = read_json(artifact.path / "manifest.json")
    kernel_manifest = read_json(artifact.path / "kernel_manifest.json")

    symbol = kernel_manifest["required_kernels"][0]["kernel_symbol"]
    assert manifest["files"]["cutlass_bmm_library"] == "lib/libdinoml_cutlass_bmm.so"
    assert symbol.startswith("dinoml_cutlass_bmm_rrr_add_float32_")
    assert kernel_manifest["required_kernels"][0]["candidate_set_id"] == "cutlass_bmm_rrr_add_float32_add_v1"
    assert kernel_manifest["required_kernels"][0]["candidate_set"]["epilogue_config"]["launch_abi"] == "dinoml_cutlass_bmm_add_v1"

    torch.manual_seed(49)
    a = torch.randn((2, 4, 8), device="cuda", dtype=torch.float32)
    b = torch.randn((2, 8, 6), device="cuda", dtype=torch.float32)
    d0 = torch.randn(d0_shape, device="cuda", dtype=torch.float32)
    expected = torch.bmm(a, b) + d0

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual_torch = session.run_torch({"a": a, "b": b, "d0": d0})["y"]
    session.close()
    module.close()

    torch.testing.assert_close(actual_torch, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    ("op_name", "a_shape", "b_shape", "dtype", "torch_dtype", "suffix", "atol", "rtol"),
    [
        ("gemm_rrr_bias", (16, 32), (32, 24), "float32", "float32", "float32", 1e-2, 1e-2),
        ("gemm_rcr_bias", (16, 32), (24, 32), "float32", "float32", "float32", 1e-2, 1e-2),
        ("gemm_rrr_bias_relu", (16, 32), (32, 24), "float32", "float32", "float32", 1e-2, 1e-2),
        ("gemm_rcr_bias_relu", (16, 32), (24, 32), "float32", "float32", "float32", 1e-2, 1e-2),
        ("gemm_rcr_bias_elup1", (16, 32), (24, 32), "float32", "float32", "float32", 1e-2, 1e-2),
    ],
)
def test_cuda_cutlass_gemm_bias_runtime_matches_torch(
    tmp_path,
    monkeypatch,
    op_name,
    a_shape,
    b_shape,
    dtype,
    torch_dtype,
    suffix,
    atol,
    rtol,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    if not discover_cuda_libraries()["cutlass"].available:
        pytest.skip("CUTLASS headers are not available")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    torch_dtype_obj = getattr(torch, torch_dtype)
    spec = dml.trace(
        GemmBiasModule(op_name),
        inputs={
            "a": dml.TensorSpec(a_shape, dtype),
            "b": dml.TensorSpec(b_shape, dtype),
            "bias": dml.TensorSpec([24], dtype),
        },
        name=f"{op_name}_{dtype}_cutlass_cuda",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / f"{op_name}_{dtype}_cutlass_cuda.dinoml")
    kernel_manifest = read_json(artifact.path / "kernel_manifest.json")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")

    required = kernel_manifest["required_kernels"][0]
    if op_name.endswith("_bias_relu"):
        epilogue = "bias_relu"
    elif op_name.endswith("_bias_elup1"):
        epilogue = "bias_elup1"
    else:
        epilogue = "bias"
    symbol = f"dinoml_cutlass_{op_name}_{suffix}_{_cutlass_default_symbol_id(dtype)}"
    assert required["kernel_symbol"] == symbol
    assert required["candidate_set_id"] == f"cutlass_{op_name}_{suffix}_{epilogue}_v1"
    assert required["candidate_set"]["epilogue_config"]["inputs"] == ["bias"]
    assert required["candidate_set"]["epilogue_config"]["name"] == epilogue
    assert required["candidates"][0]["launch_abi"] == "dinoml_cutlass_gemm_bias_v1"
    assert symbol in generated

    torch.manual_seed(48)
    a = torch.randn(a_shape, device="cuda", dtype=torch_dtype_obj)
    b = torch.randn(b_shape, device="cuda", dtype=torch_dtype_obj)
    bias = torch.randn((24,), device="cuda", dtype=torch_dtype_obj)
    expected = a @ (b if op_name.startswith("gemm_rrr") else b.t()) + bias
    if op_name.endswith("_bias_relu"):
        expected = torch.relu(expected)
    elif op_name.endswith("_bias_elup1"):
        expected = torch.where(expected >= 0, expected + 1, torch.exp(expected))

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual_torch = session.run_torch({"a": a, "b": b, "bias": bias})["y"]
    actual_numpy = session.run_numpy(
        {"a": a.float().cpu().numpy(), "b": b.float().cpu().numpy(), "bias": bias.float().cpu().numpy()}
    )["y"]
    session.close()
    module.close()

    assert actual_torch.dtype == torch_dtype_obj
    torch.testing.assert_close(actual_torch, expected, atol=atol, rtol=rtol)
    np.testing.assert_allclose(actual_numpy.astype(np.float32), expected.float().cpu().numpy(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("op_name", "b_spec_factory", "b_shape"),
    [
        ("gemm_rrr", lambda tokens: [32, tokens], (32, 11)),
        ("gemm_rcr", lambda tokens: [tokens, 32], (11, 32)),
    ],
)
def test_cuda_cutlass_gemm_supports_dynamic_mn_shapes(tmp_path, monkeypatch, op_name, b_spec_factory, b_shape):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    if not discover_cuda_libraries()["cutlass"].available:
        pytest.skip("CUTLASS headers are not available")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    batch = dml.Dim("batch", min=1, max=16)
    tokens = dml.Dim("tokens", min=1, max=24)
    spec = dml.trace(
        GemmModule(op_name),
        inputs={"a": dml.TensorSpec([batch, 32], "float32"), "b": dml.TensorSpec(b_spec_factory(tokens), "float32")},
        name=f"{op_name}_dynamic_cutlass_cuda",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / f"{op_name}_dynamic_cutlass_cuda.dinoml")

    torch.manual_seed(1234)
    a = torch.randn((7, 32), device="cuda", dtype=torch.float32)
    b = torch.randn(b_shape, device="cuda", dtype=torch.float32)
    expected = a @ (b if op_name == "gemm_rrr" else b.t())

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_torch({"a": a, "b": b})["y"]
    assert tuple(actual.shape) == (7, 11)
    assert session.get_output_shape("y") == (7, 11)
    session.close()
    module.close()

    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_cuda_cutlass_linear_model_uses_runtime_constants_and_dynamic_batch(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    if not discover_cuda_libraries()["cutlass"].available:
        pytest.skip("CUTLASS headers are not available")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    from examples.cuda_linear import (
        IN_FEATURES,
        MAX_BATCH,
        build_constants,
        build_spec,
        build_validation_inputs,
        numpy_reference,
    )

    spec = build_spec()
    target = dml.Target("cuda", arch="sm_86", no_tf32=True)
    artifact = dml.compile(spec, target, tmp_path / "cuda_linear.dinoml")
    manifest = read_json(artifact.path / "manifest.json")
    kernel_manifest = read_json(artifact.path / "kernel_manifest.json")
    metadata = read_json(artifact.path / "metadata.json")

    required = kernel_manifest["required_kernels"][0]
    assert manifest["files"]["cutlass_gemm_library"] == "lib/libdinoml_cutlass_gemm.so"
    assert required["kernel_library"] == "cutlass_gemm"
    assert required["candidate_set_id"] == "cutlass_gemm_rrr_bias_float32_bias_v1"
    assert required["candidate_set"]["epilogue_config"]["inputs"] == ["bias"]
    assert required["candidate_set"]["target_policy"]["no_tf32"] is True
    expected_candidate_count = _cutlass_candidate_count(
        "float32",
        op_name="gemm_rrr_bias",
        target=target.to_json(),
    )
    assert required["candidate_set"]["candidate_count"] == expected_candidate_count
    assert len(required["candidates"]) == expected_candidate_count
    assert required["kernel_symbol"] == required["candidates"][0]["kernel_symbol"]
    assert required["selected_candidate_id"] == required["candidates"][0]["candidate_id"]
    assert required["cutlass_alignment_cap"] == 2
    assert required["candidates"][0]["cutlass"]["align"] <= required["cutlass_alignment_cap"]
    assert [constant["name"] for constant in metadata["constants"]] == ["weight", "bias"]
    assert metadata["inputs"][0]["shape_spec"][0]["buckets"] == [1, 3, 4]

    runtime_constants = {
        "weight": build_constants()["weight"] * np.float32(-0.5),
        "bias": build_constants()["bias"] + np.float32(0.25),
    }
    validation_inputs = build_validation_inputs()

    def random_inputs(batch: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(2026 + batch)
        return {"x": rng.standard_normal((batch, IN_FEATURES)).astype(np.float32)}

    module = runtime.load(artifact.path)
    weight = torch.tensor(runtime_constants["weight"], device="cuda", dtype=torch.float32)
    bias = torch.tensor(runtime_constants["bias"], device="cuda", dtype=torch.float32)
    module.set_constant_torch("weight", weight)
    module.set_constant_torch("bias", bias)
    session = module.create_session()

    input_cases = [
        {"x": validation_inputs["x"][:2].copy()},
        validation_inputs,
        random_inputs(1),
        random_inputs(MAX_BATCH),
    ]
    for inputs in input_cases:
        expected = numpy_reference(inputs, runtime_constants)["y"]
        x = torch.tensor(inputs["x"], device="cuda", dtype=torch.float32)
        actual = session.run_torch({"x": x})["y"]
        assert tuple(actual.shape) == expected.shape
        assert session.get_output_shape("y") == expected.shape
        torch.testing.assert_close(actual.cpu(), torch.from_numpy(expected), atol=1e-2, rtol=1e-2)

    session.close()
    module.close()


def test_cuda_cutlass_gemm_no_tf32_runtime_matches_torch(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    if not discover_cuda_libraries()["cutlass"].available:
        pytest.skip("CUTLASS headers are not available")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    target = dml.Target("cuda", arch="sm_86", no_tf32=True)
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec((16, 32), "float32"), "b": dml.TensorSpec((32, 24), "float32")},
        name="gemm_rrr_float32_no_tf32_cutlass_cuda",
    )
    artifact = dml.compile(spec, target, tmp_path / "gemm_rrr_float32_no_tf32_cutlass_cuda.dinoml")
    kernel_manifest = read_json(artifact.path / "kernel_manifest.json")
    required = kernel_manifest["required_kernels"][0]
    symbol = f"dinoml_cutlass_gemm_rrr_float32_{_cutlass_default_symbol_id('float32', target=target.to_json())}"
    assert required["kernel_symbol"] == symbol
    assert required["candidate_set"]["candidate_count"] == _cutlass_candidate_count("float32", target=target.to_json())
    assert {candidate["cutlass"]["opclass"] for candidate in required["candidates"]} == {"simt"}
    assert "simt_sm80_f32" in symbol

    torch.manual_seed(47)
    a = torch.randn((16, 32), device="cuda", dtype=torch.float32)
    b = torch.randn((32, 24), device="cuda", dtype=torch.float32)
    expected = a @ b

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_torch({"a": a, "b": b})["y"]
    session.close()
    module.close()

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_cuda_cutlass_gemm_rrr_runtime_dequantizes_gguf_q4_0_rhs_before_launch(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    if not discover_cuda_libraries()["cutlass"].available:
        pytest.skip("CUTLASS headers are not available")
    libgguf = pytest.importorskip("libgguf")
    pytest.importorskip("libgguf.libgguf_cuda")
    if runtime._libgguf_cuda_native_dequantize_rows_on_stream() is None:
        pytest.skip("libgguf CUDA native dequantize ABI is not available")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    class EncodedRhsGemm(dml.Module):
        def __init__(self):
            self.weight = dml.Parameter([32, 32], dtype="float32")

        def forward(self, x):
            return dml.ops.output(dml.ops.gemm_rrr(x, self.weight), "y")

    rows = np.linspace(-1.5, 1.5, 32 * 32, dtype=np.float32).reshape(32, 32)
    qtype = libgguf.GGMLQuantizationType.Q4_0
    encoded = libgguf.quantize_rows(rows, qtype)
    expected_weight = libgguf.dequantize_rows(encoded, qtype, n_per_row=32).reshape(32, 32)
    gguf_path = tmp_path / "weights.gguf"
    _write_minimal_gguf_tensor(
        gguf_path,
        name="blk.0.ffn.weight",
        gguf_shape=(32, 32),
        qtype_value=int(qtype),
        payload=encoded.tobytes(order="C"),
    )
    source = dml.gguf_constant(
        gguf_path,
        "blk.0.ffn.weight",
        qtype="Q4_0",
        encoded_nbytes=encoded.nbytes,
        n_per_row=32,
        materialization="dequantize_on_gpu_before_launch",
        residency="manual_runtime_load",
    )
    traced = dml.trace(
        EncodedRhsGemm(),
        inputs={"x": dml.TensorSpec([2, 32], "float32")},
        constants={"weight": source},
        name="gguf_q4_0_runtime_dequant_gemm_rrr",
    )
    spec = ModelSpec(name=traced.name, ir=traced.ir, constants={"weight": source})
    target = dml.Target("cuda", arch="sm_86", no_tf32=True)
    artifact = dml.compile(
        spec,
        target,
        tmp_path / "gguf_q4_0_runtime_dequant_gemm_rrr.dinoml",
        constant_load_policy="deferred",
    )

    kernel_manifest = read_json(artifact.path / "kernel_manifest.json")
    plan = kernel_manifest["required_kernels"][0]["gguf_runtime_dequant"]
    assert plan["status"] == "lowered_runtime_dequant_scratch"
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "module->libgguf_cuda_dequantize_rows_on_stream" in generated
    assert generated.index("module->libgguf_cuda_dequantize_rows_on_stream") < generated.index("ptr_x, ptr_weight_dequant")

    x = np.linspace(-0.5, 0.75, 64, dtype=np.float32).reshape(2, 32)
    expected = x @ expected_weight
    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        assert module.constant_load_state() == {"weight": False}
        with pytest.raises(RuntimeError, match="Constant weight has not been loaded"):
            session.run_numpy({"x": x})
        module.load_encoded_constants(["weight"])
        actual = session.run_numpy({"x": x})["y"]
    finally:
        session.close()
        module.close()

    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


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


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cuda_generated_reduced_precision_softmax_source_avoids_packed_float_reinterpret(tmp_path, dtype):
    spec = dml.trace(
        SoftmaxLastDim(),
        inputs={"x": dml.TensorSpec([8, 1024], dtype)},
        name=f"softmax_{dtype}_cuda",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / f"softmax_{dtype}_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    storage_type = "half" if dtype == "float16" else "__nv_bfloat16"
    assert f"const {storage_type}* DINO_RESTRICT x" in generated
    assert f"{storage_type}* DINO_RESTRICT y" in generated
    assert "_warp_kernel" in generated
    assert "_packed_kernel" not in generated
    assert "float4" not in generated
    assert "float2" not in generated
    assert "dinoml::math::cast<float>(x[base + col])" in generated
    assert f"dinoml::math::cast<{storage_type}>" in generated


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


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("op_name", ["reduce_sum", "reduce_max", "reduce_min", "reduce_mean"])
def test_cuda_generated_reduced_precision_reduction_source(tmp_path, dtype, op_name):
    spec = dml.trace(
        ReductionLastDim(op_name),
        inputs={"x": dml.TensorSpec([8, 33], dtype)},
        name=f"{op_name}_{dtype}_cuda",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / f"{op_name}_{dtype}_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    storage_type = "half" if dtype == "float16" else "__nv_bfloat16"
    assert f"const {storage_type}* DINO_RESTRICT x" in generated
    assert f"{storage_type}* DINO_RESTRICT y" in generated
    assert "float acc" in generated
    assert "dinoml::math::cast<float>(x[base + col])" in generated
    assert f"dinoml::math::cast<{storage_type}>" in generated


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


class DynamicConstantBias(dml.Module):
    def __init__(self, batch):
        self.bias = dml.Parameter([batch, 1], dtype="float32")

    def forward(self, x):
        return dml.ops.output(x + self.bias, "y")


class DirectIdentityModel(dml.Module):
    def forward(self, x):
        return dml.ops.output(x, "y")


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


def test_cuda_runtime_set_constant_accepts_dynamic_shape(tmp_path):
    batch = dml.Dim("batch", min=1, max=4)
    spec = dml.trace(
        DynamicConstantBias(batch),
        inputs={"x": dml.TensorSpec([batch, 4], "float32")},
        constants={"bias": np.zeros((4, 1), dtype=np.float32)},
        name="dynamic_constant_bias_cuda",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "dynamic_constant_bias_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "std::vector<int64_t> const_shape_bias;" in generated
    assert 'check_tensor_dynamic(\n            *tensor,\n            "bias"' in generated
    assert "const int64_t shape_bias_0 = module->const_shape_bias[0];" in generated
    assert "Dynamic dimension batch mismatch between x and bias" in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    bias = np.array([[10.0], [20.0]], dtype=np.float32)
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    try:
        module.set_constant_numpy("bias", bias)
        actual = session.run_numpy({"x": x})["y"]
        with pytest.raises(ValueError, match=r"bias axis 0 .*expected \[1, 4\]"):
            module.set_constant_numpy("bias", np.zeros((5, 1), dtype=np.float32))
        module.set_constant_numpy("bias", np.zeros((3, 1), dtype=np.float32))
        with pytest.raises(RuntimeError, match="Dynamic dimension batch mismatch between x and bias"):
            session.run_numpy({"x": x})
    finally:
        session.close()
        module.close()

    np.testing.assert_allclose(actual, x + bias, atol=1e-5, rtol=1e-5)


def test_cuda_runtime_materializes_direct_input_output(tmp_path):
    spec = dml.trace(DirectIdentityModel(), inputs={"x": dml.TensorSpec([2, 3], "float32")}, name="direct_identity_cuda")
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "direct_identity_cuda.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cu").read_text(encoding="utf-8")
    assert "cudaMemcpyAsync(dinoml::module::tensor_data(outputs[0])" in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    try:
        actual = session.run_numpy({"x": x})["y"]
    finally:
        session.close()
        module.close()

    np.testing.assert_array_equal(actual, x)


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
