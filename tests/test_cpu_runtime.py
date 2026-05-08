import ctypes

import numpy as np

import dinoml as dml
from dinoml import runtime
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import read_json


class DynamicChannelBias(dml.Module):
    def __init__(self):
        self.scale = dml.Parameter([4], dtype="float32")
        self.bias = dml.Parameter([4], dtype="float32")

    def forward(self, x):
        return dml.ops.output(dml.ops.relu(x * self.scale + self.bias), "y")


class DynamicGenericBroadcast(dml.Module):
    def forward(self, x, z):
        return dml.ops.output(dml.ops.relu(x + z), "y")


def test_cpu_artifact_uses_shared_runtime_and_generated_elementwise(tmp_path):
    from tests.models.fused_elementwise import build_spec, build_validation_inputs

    spec = build_spec()
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "fused_elementwise_cpu.dinoml")
    assert (artifact.path / "lib" / "libdinoml_runtime.so").exists()
    assert (artifact.path / "lib" / "libdinoml_cpu_kernels.so").exists()
    assert (artifact.path / "kernel_manifest.json").exists()
    assert (artifact.path / "metadata.json").exists()
    assert read_json(artifact.path / "manifest.json")["files"]["metadata"] == "metadata.json"
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "kMetadataJson" not in generated
    assert "R\"DINOJSON" not in generated
    assert "dino_session_set_stream" in generated
    source_manifest = read_json(artifact.path / "debug" / "generated_src" / "source_manifest.json")
    sources = source_manifest["sources"]
    assert source_manifest["deduplication"] == "exact_source_key"
    assert len(sources) == 1
    assert sources[0]["op"] == "fused_elementwise"
    assert sources[0]["target"] == "cpu"
    assert sources[0]["emitted_new_source"] is True
    per_op_source = artifact.path / "debug" / "generated_src" / sources[0]["emitted_source_path"]
    assert per_op_source.exists()
    assert per_op_source.read_text(encoding="utf-8") in generated

    inputs = build_validation_inputs()
    expected = execute_cpu(spec, inputs)

    module = runtime.load(artifact.path)
    assert module.metadata == read_json(artifact.path / "metadata.json")
    assert hasattr(module._dll, "dino_session_set_stream")
    session = module.create_session()
    session.set_stream(ctypes.c_void_p(0))
    session.set_stream(None)
    actual = session.run_numpy(inputs)
    module.set_constant_numpy("scale", np.zeros_like(spec.constants["scale"]))
    module.set_constant_numpy("bias", np.zeros_like(spec.constants["bias"]))
    zeroed = session.run_numpy(inputs)
    session.close()
    module.close()

    np.testing.assert_allclose(actual["y"], expected["y"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(zeroed["y"], np.zeros([2, 3, 4], dtype=np.float32), atol=1e-6, rtol=0)


def test_cpu_generated_fused_elementwise_supports_generic_subgraph(tmp_path):
    from tests.models.fused_elementwise import build_spec, build_validation_inputs

    spec = build_spec()
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "generic_elementwise.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "fused_elementwise_" in generated
    assert "dino_fused_" not in generated
    assert "dinoml::math::mul" in generated
    assert "dinoml::math::sub" in generated
    assert "dinoml::math::sigmoid" in generated
    assert "dinoml::math::relu" in generated

    inputs = build_validation_inputs()
    expected = execute_cpu(spec, inputs)

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy(inputs)
    session.close()
    module.close()

    np.testing.assert_allclose(actual["y"], expected["y"], atol=1e-5, rtol=1e-5)


def test_cpu_runtime_supports_dynamic_shapes(tmp_path):
    batch = dml.Dim("batch", min=1, max=4)
    height = dml.Dim("height", min=8, max=16, divisible_by=8)
    constants = {
        "scale": np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float32),
        "bias": np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32),
    }
    spec = dml.trace(
        DynamicChannelBias(),
        inputs={"x": dml.TensorSpec([batch, height, 4], "float32")},
        constants=constants,
        name="dynamic_channel_bias",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "dynamic_channel_bias_cpu.dinoml")
    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        session.get_output_shape("y")
    except RuntimeError as exc:
        assert "before dino_session_run" in str(exc)
    else:
        raise AssertionError("output shape was available before dino_session_run")

    for shape in ((2, 8, 4), (4, 16, 4)):
        x = np.random.default_rng(sum(shape)).standard_normal(shape).astype(np.float32)
        expected = np.maximum(x * constants["scale"] + constants["bias"], 0.0).astype(np.float32)
        actual = session.run_numpy({"x": x})["y"]
        assert actual.shape == shape
        assert session.get_output_shape("y") == shape
        assert session.get_output_shape(0) == shape
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    small = (ctypes.c_int64 * 1)()
    ndim = ctypes.c_size_t(1)
    err = module._dll.dino_session_get_output_shape(session._handle, ctypes.c_size_t(0), small, ctypes.byref(ndim))
    assert err
    assert ndim.value == 3
    assert b"too small" in module._last_error_message()

    bad = np.zeros((2, 10, 4), dtype=np.float32)
    try:
        session.run_numpy({"x": bad})
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("dynamic shape divisibility violation was not rejected")

    session.close()
    module.close()


def test_cpu_runtime_supports_dynamic_generic_broadcast(tmp_path):
    batch = dml.Dim("batch", min=1, max=4)
    height = dml.Dim("height", min=8, max=16, divisible_by=8)
    spec = dml.trace(
        DynamicGenericBroadcast(),
        inputs={
            "x": dml.TensorSpec([batch, height, 4], "float32"),
            "z": dml.TensorSpec([1, height, 1], "float32"),
        },
        name="dynamic_generic_broadcast",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "dynamic_generic_broadcast_cpu.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "const int64_t* input_shape" in generated
    assert "session->shape_z" in generated
    assert "session->shape_t1" in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    for shape in ((2, 8, 4), (4, 16, 4)):
        rng = np.random.default_rng(sum(shape) + 17)
        x = rng.standard_normal(shape).astype(np.float32)
        z = rng.standard_normal((1, shape[1], 1)).astype(np.float32)
        expected = np.maximum(x + z, 0.0).astype(np.float32)
        actual = session.run_numpy({"x": x, "z": z})["y"]
        assert actual.shape == shape
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    session.close()
    module.close()
