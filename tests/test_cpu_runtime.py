import ctypes
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import dinoml as dml
from dinoml import runtime
from dinoml.backends.cpu import execute_cpu
from dinoml.constant_sources import MaterializedConstant
from dinoml.ir import (
    IR_SCHEMA_VERSION,
    RUNTIME_ABI_VERSION,
    ModelSpec,
    array_from_storage,
    array_to_storage,
    dtype_runtime_enum,
    read_json,
    write_json,
)
from dinoml.ops.definitions import get_op_def
from dinoml.passes import PassManager


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


class SoftmaxLastDim(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.softmax(x, dim=-1), "y")


class SoftmaxNonLastDim(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.softmax(x, dim=0), "y")


class ReductionLastDim(dml.Module):
    def __init__(self, op_name: str, keepdim: bool = False, **attrs):
        self.op_name = op_name
        self.keepdim = keepdim
        self.attrs = attrs

    def forward(self, x):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(x, dim=-1, keepdim=self.keepdim, **self.attrs), "y")


class PublicShapeViewOutputs(dml.Module):
    def forward(self, x, z):
        return {
            "id": dml.ops.identity(x),
            "reshaped": dml.ops.reshape(x, [3, 2]),
            "flat": dml.ops.flatten(x),
            "squeezed": dml.ops.squeeze(z),
            "unsqueezed": dml.ops.unsqueeze(x, 0),
        }


class AddZeroModel(dml.Module):
    def forward(self, x):
        return dml.ops.output(x + 0.0, "y")


class EncodedWeightModel(dml.Module):
    def __init__(self):
        self.weight = dml.Parameter([2, 2], dtype="float32")

    def forward(self, x):
        return dml.ops.output(x + self.weight, "y")


class MaterializingConstant:
    def __init__(self, value, storage):
        self.value = np.asarray(value)
        self.storage = storage

    def materialize(self, dtype, shape):
        return MaterializedConstant(array_to_storage(self.value, dtype), self.storage)


class DirectIdentityModel(dml.Module):
    def forward(self, x):
        return dml.ops.output(x, "y")


class AddZeroUnusedShapeSource(dml.Module):
    def forward(self, x, z):
        return dml.ops.output(x + 0.0, "y")


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


class BmmModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b, d0=None):
        op = getattr(dml.ops, self.op_name)
        if d0 is not None:
            return dml.ops.output(op(a, b, d0), "y")
        return dml.ops.output(op(a, b), "y")


class BmmHelperModule(dml.Module):
    def __init__(self, helper_name: str, layout: str = "rrr"):
        self.helper_name = helper_name
        self.layout = layout

    def forward(self, a, b, d0=None):
        if self.helper_name == "bmm":
            return dml.ops.output(dml.ops.bmm(a, b), "y")
        if self.helper_name == "bmm_xxx":
            return dml.ops.output(dml.ops.bmm_xxx(a, b, layout=self.layout), "y")
        if d0 is None:
            raise ValueError("bmm_xxx_add helper test requires d0")
        return dml.ops.output(dml.ops.bmm_xxx_add(a, b, d0, layout=self.layout), "y")


class GemmBiasModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b, bias):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(a, b, bias), "y")


class GemmResidualModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b, bias, d0, d1=None):
        op = getattr(dml.ops, self.op_name)
        if d1 is None:
            return dml.ops.output(op(a, b, bias, d0), "y")
        return dml.ops.output(op(a, b, bias, d0, d1), "y")


GEMM_BIAS_RESIDUAL_CASES = tuple(
    (f"gemm_{layout}_bias_{suffix}", layout, epilogue, inputs)
    for layout in ("rcr", "rrr")
    for suffix, epilogue, inputs in (
        ("add", "bias_add", ("bias", "d0")),
        ("add_add", "bias_add_add", ("bias", "d0", "d1")),
        ("mul", "bias_mul", ("bias", "d0")),
        ("mul_add", "bias_mul_add", ("bias", "d0", "d1")),
    )
)
GEMM_BIAS_RESIDUAL_RELU_CASES = tuple(
    (f"gemm_{layout}_bias_{suffix}", layout, epilogue, inputs)
    for layout in ("rcr", "rrr")
    for suffix, epilogue, inputs in (
        ("add_relu", "bias_add_relu", ("bias", "d0")),
        ("add_add_relu", "bias_add_add_relu", ("bias", "d0", "d1")),
    )
)
GEMM_BIAS_RESIDUAL_COMPOUND_CASES = tuple(
    (f"gemm_{layout}_bias_{suffix}", layout, epilogue, inputs)
    for layout in ("rcr", "rrr")
    for suffix, epilogue, inputs in (
        ("mul_tanh", "bias_mul_tanh", ("bias", "d0")),
        ("sigmoid_mul", "bias_sigmoid_mul", ("bias", "d0")),
        ("sigmoid_mul_tanh", "bias_sigmoid_mul_tanh", ("bias", "d0")),
    )
)
GEMM_BIAS_RESIDUAL_CASES = (
    *GEMM_BIAS_RESIDUAL_CASES,
    *GEMM_BIAS_RESIDUAL_RELU_CASES,
    *GEMM_BIAS_RESIDUAL_COMPOUND_CASES,
)


def _storage_roundtrip(value, dtype: str):
    return array_from_storage(array_to_storage(value, dtype), dtype)


def test_python_runtime_populates_dino_tensor_metadata():
    tensor, keepalive = runtime._make_dino_tensor(
        ctypes.c_void_p(0x1000),
        (2, 3, 4),
        dtype_runtime_enum("float32"),
        nbytes=2 * 3 * 4 * 4,
        device_type=runtime.DINO_DEVICE_CPU,
    )

    assert RUNTIME_ABI_VERSION == 7
    assert len(keepalive) == 2
    assert [tensor.shape[idx] for idx in range(tensor.ndim)] == [2, 3, 4]
    assert [tensor.strides[idx] for idx in range(tensor.ndim)] == [12, 4, 1]
    assert tensor.byte_offset == 0
    assert tensor.nbytes == 96
    assert tensor.device_type == runtime.DINO_DEVICE_CPU
    assert tensor.flags & runtime.DINO_TENSOR_FLAG_CONTIGUOUS
    assert tensor.alignment >= 16


def test_cpu_runtime_copies_non_contiguous_numpy_inputs(tmp_path):
    spec = dml.trace(AddZeroModel(), inputs={"x": dml.TensorSpec([2, 3], "float32")}, name="add_zero_non_contiguous")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "identity_non_contiguous.dinoml")
    base = np.arange(12, dtype=np.float32).reshape(2, 6)
    x = base[:, ::2]
    assert not x.flags.c_contiguous

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy({"x": x})["y"]
    finally:
        session.close()
        module.close()

    np.testing.assert_array_equal(actual, np.ascontiguousarray(x))


def test_cpu_runtime_rejects_non_contiguous_abi_strides(tmp_path):
    spec = dml.trace(AddZeroModel(), inputs={"x": dml.TensorSpec([2, 3], "float32")}, name="add_zero_bad_strides")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "identity_bad_strides.dinoml")

    module = runtime.load(artifact.path)
    session = module.create_session()
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    y = np.empty_like(x)
    input_shape = runtime._shape_buffer(x.shape)
    output_shape = runtime._shape_buffer(y.shape)
    bad_input_strides = (ctypes.c_int64 * 2)(1, 2)
    output_strides = runtime._stride_buffer(y.shape)
    input_tensor = runtime._DinoTensor(
        ctypes.c_void_p(x.ctypes.data),
        input_shape,
        x.ndim,
        dtype_runtime_enum("float32"),
        bad_input_strides,
        0,
        x.nbytes,
        runtime.DINO_DEVICE_CPU,
        0,
        16,
    )
    output_tensor = runtime._DinoTensor(
        ctypes.c_void_p(y.ctypes.data),
        output_shape,
        y.ndim,
        dtype_runtime_enum("float32"),
        output_strides,
        0,
        y.nbytes,
        runtime.DINO_DEVICE_CPU,
        runtime.DINO_TENSOR_FLAG_CONTIGUOUS,
        16,
    )
    inputs = (runtime._DinoTensor * 1)(input_tensor)
    outputs = (runtime._DinoTensor * 1)(output_tensor)
    try:
        with pytest.raises(RuntimeError, match="contiguous row-major strides"):
            module._check(module._dll.dino_session_run(session._handle, inputs, ctypes.c_size_t(1), outputs, ctypes.c_size_t(1)))
    finally:
        session.close()
        module.close()


def test_cpu_runtime_honors_byte_offsets_in_abi_tensors(tmp_path):
    spec = dml.trace(AddZeroModel(), inputs={"x": dml.TensorSpec([2, 3], "float32")}, name="add_zero_byte_offset")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "identity_byte_offset.dinoml")

    module = runtime.load(artifact.path)
    session = module.create_session()
    backing_input = np.arange(8, dtype=np.float32)
    backing_output = np.full(8, -1.0, dtype=np.float32)
    input_shape = runtime._shape_buffer((2, 3))
    output_shape = runtime._shape_buffer((2, 3))
    input_strides = runtime._stride_buffer((2, 3))
    output_strides = runtime._stride_buffer((2, 3))
    input_tensor = runtime._DinoTensor(
        ctypes.c_void_p(backing_input.ctypes.data),
        input_shape,
        2,
        dtype_runtime_enum("float32"),
        input_strides,
        backing_input.itemsize,
        backing_input.nbytes,
        runtime.DINO_DEVICE_CPU,
        runtime.DINO_TENSOR_FLAG_CONTIGUOUS,
        0,
    )
    output_tensor = runtime._DinoTensor(
        ctypes.c_void_p(backing_output.ctypes.data),
        output_shape,
        2,
        dtype_runtime_enum("float32"),
        output_strides,
        backing_output.itemsize,
        backing_output.nbytes,
        runtime.DINO_DEVICE_CPU,
        runtime.DINO_TENSOR_FLAG_CONTIGUOUS,
        0,
    )
    inputs = (runtime._DinoTensor * 1)(input_tensor)
    outputs = (runtime._DinoTensor * 1)(output_tensor)

    try:
        module._check(module._dll.dino_session_run(session._handle, inputs, ctypes.c_size_t(1), outputs, ctypes.c_size_t(1)))
    finally:
        session.close()
        module.close()

    np.testing.assert_array_equal(backing_output[1:7].reshape(2, 3), backing_input[1:7].reshape(2, 3))
    assert backing_output[0] == -1.0
    assert backing_output[7] == -1.0


def test_cpu_runtime_materializes_direct_input_output(tmp_path):
    spec = dml.trace(DirectIdentityModel(), inputs={"x": dml.TensorSpec([2, 3], "float32")}, name="direct_identity_cpu")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "direct_identity_cpu.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "const float* ptr_" in generated
    assert "std::memcpy(dinoml::module::tensor_data(outputs[0])" in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    try:
        actual = session.run_numpy({"x": x})["y"]
    finally:
        session.close()
        module.close()

    np.testing.assert_array_equal(actual, x)


def test_run_numpy_rejects_unexpected_input_names(tmp_path):
    spec = dml.trace(DirectIdentityModel(), inputs={"x": dml.TensorSpec([2, 3], "float32")}, name="unexpected_input_cpu")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "unexpected_input_cpu.dinoml")
    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        with pytest.raises(ValueError, match="Unexpected input: z"):
            session.run_numpy(
                {
                    "x": np.zeros((2, 3), dtype=np.float32),
                    "z": np.zeros((2, 3), dtype=np.float32),
                }
            )
    finally:
        session.close()
        module.close()


def test_run_numpy_rejects_non_mapping_inputs(tmp_path):
    spec = dml.trace(DirectIdentityModel(), inputs={"x": dml.TensorSpec([2, 3], "float32")}, name="bad_input_map_cpu")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "bad_input_map_cpu.dinoml")
    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        with pytest.raises(TypeError, match="run_numpy inputs must be a mapping"):
            session.run_numpy([("x", np.zeros((2, 3), dtype=np.float32))])
    finally:
        session.close()
        module.close()


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cpu_runtime_materializes_reduced_precision_direct_input_output(tmp_path, dtype):
    spec = dml.trace(DirectIdentityModel(), inputs={"x": dml.TensorSpec([2, 3], dtype)}, name=f"direct_identity_{dtype}_cpu")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"direct_identity_{dtype}_cpu.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert ("dinoml::math::float16" if dtype == "float16" else "dinoml::math::bfloat16") in generated
    assert "std::memcpy(dinoml::module::tensor_data(outputs[0])" in generated

    x = np.array([[-2.25, -1.0, 0.0], [1.125, 2.5, 3.75]], dtype=np.float32)
    expected = _storage_roundtrip(x, dtype)
    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy({"x": x})["y"]
    finally:
        session.close()
        module.close()

    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=0, rtol=0)


def test_cpu_artifact_uses_shared_runtime_and_generated_elementwise(tmp_path):
    from tests.models.fused_elementwise import build_spec, build_validation_inputs

    spec = build_spec()
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "fused_elementwise_cpu.dinoml")
    assert (artifact.path / "lib" / "libdinoml_runtime.so").exists()
    assert (artifact.path / "lib" / "libdinoml_cpu_kernels.so").exists()
    assert (artifact.path / "kernel_manifest.json").exists()
    assert (artifact.path / "metadata.json").exists()
    manifest = read_json(artifact.path / "manifest.json")
    assert manifest["files"]["metadata"] == "metadata.json"
    assert manifest["constant_load_policy"] == "eager"
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "kMetadataJson" not in generated
    assert "R\"DINOJSON" not in generated
    assert "dino_session_set_stream" in generated
    assert "dino_module_unload_constants" in generated
    assert "dino_module_load_deferred" in generated
    assert "session->last_output_shapes_valid[i] = false;" in generated
    assert "session->last_output_shapes[i].clear();" in generated
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
    expected_loaded = {constant["name"]: True for constant in spec.ir["constants"]}
    expected_unloaded = {constant["name"]: False for constant in spec.ir["constants"]}

    module = runtime.load(artifact.path)
    assert module.metadata == read_json(artifact.path / "metadata.json")
    assert hasattr(module._dll, "dino_session_set_stream")
    assert hasattr(module._dll, "dino_module_load_deferred")
    assert module.constant_load_state() == expected_loaded
    assert module.is_constant_loaded("scale") is True
    with pytest.raises(ValueError, match="Unknown constant"):
        module.is_constant_loaded("missing")
    session = module.create_session()
    session.set_stream(ctypes.c_void_p(0))
    session.set_stream(None)
    actual = session.run_numpy(inputs)
    module.set_constant_numpy("scale", np.zeros_like(spec.constants["scale"]))
    module.set_constant_numpy("bias", np.zeros_like(spec.constants["bias"]))
    assert module.constant_load_state() == expected_loaded
    zeroed = session.run_numpy(inputs)
    module.unload_constants()
    assert module.constant_load_state() == expected_unloaded
    with pytest.raises(RuntimeError, match="Constant scale has not been loaded"):
        session.run_numpy(inputs)
    with pytest.raises(RuntimeError, match="Output shape is unavailable before dino_session_run"):
        session.get_output_shape("y")
    module.load_constants_from_file()
    assert module.constant_load_state() == expected_loaded
    reloaded = session.run_numpy(inputs)
    session.close()
    module.close()

    np.testing.assert_allclose(actual["y"], expected["y"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(zeroed["y"], np.zeros([2, 3, 4], dtype=np.float32), atol=1e-6, rtol=0)
    np.testing.assert_allclose(reloaded["y"], expected["y"], atol=1e-5, rtol=1e-5)

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
    np.testing.assert_allclose(deferred["y"], expected["y"], atol=1e-5, rtol=1e-5)


def test_cpu_artifact_deferred_constant_load_policy(tmp_path):
    from tests.models.fused_elementwise import build_spec, build_validation_inputs

    spec = build_spec()
    artifact = dml.compile(
        spec,
        dml.Target("cpu"),
        tmp_path / "deferred_constants_cpu.dinoml",
        constant_load_policy="deferred",
    )
    manifest = read_json(artifact.path / "manifest.json")
    assert manifest["constant_load_policy"] == "deferred"

    inputs = build_validation_inputs()
    expected = execute_cpu(spec, inputs)
    expected_loaded = {constant["name"]: True for constant in spec.ir["constants"]}
    expected_unloaded = {constant["name"]: False for constant in spec.ir["constants"]}
    module = runtime.load(artifact.path)
    assert module.constant_load_state() == expected_unloaded
    session = module.create_session()
    try:
        with pytest.raises(RuntimeError, match="Constant scale has not been loaded"):
            session.run_numpy(inputs)
        module.load_constants_from_file()
        assert module.constant_load_state() == expected_loaded
        actual = session.run_numpy(inputs)
    finally:
        session.close()
        module.close()
    np.testing.assert_allclose(actual["y"], expected["y"], atol=1e-5, rtol=1e-5)

    eager_module = runtime.load(artifact.path, load_constants=True)
    assert eager_module.constant_load_state() == expected_loaded
    eager_session = eager_module.create_session()
    try:
        eager_actual = eager_session.run_numpy(inputs)
    finally:
        eager_session.close()
        eager_module.close()
    np.testing.assert_allclose(eager_actual["y"], expected["y"], atol=1e-5, rtol=1e-5)


def test_cpu_constant_reload_failure_preserves_resident_values(tmp_path):
    from tests.models.fused_elementwise import build_spec, build_validation_inputs

    spec = build_spec()
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "constant_reload_transactional_cpu.dinoml")
    metadata = read_json(artifact.path / "metadata.json")
    constants = {constant["name"]: constant for constant in metadata["constants"]}
    partial_reload = bytearray((artifact.path / "constants.bin").read_bytes())
    scale = constants["scale"]
    bias = constants["bias"]
    partial_reload[scale["offset"] : scale["offset"] + scale["nbytes"]] = b"\x00" * scale["nbytes"]
    truncated_path = tmp_path / "truncated_constants.bin"
    truncated_path.write_bytes(bytes(partial_reload[: bias["offset"] + bias["nbytes"] - 1]))

    inputs = build_validation_inputs()
    expected = execute_cpu(spec, inputs)
    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        before = session.run_numpy(inputs)
        with pytest.raises(RuntimeError, match="constants file is too small for bias"):
            module.load_constants_from_file(truncated_path)
        after = session.run_numpy(inputs)
        assert module.constant_load_state() == {name: True for name in constants}
    finally:
        session.close()
        module.close()

    np.testing.assert_allclose(before["y"], expected["y"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(after["y"], expected["y"], atol=1e-5, rtol=1e-5)


def test_create_session_rejects_closed_runtime_module(tmp_path):
    from tests.models.fused_elementwise import build_spec

    artifact = dml.compile(build_spec(), dml.Target("cpu"), tmp_path / "closed_module_cpu.dinoml")
    module = runtime.load(artifact.path)
    module.close()

    with pytest.raises(RuntimeError, match="RuntimeModule is closed"):
        module.create_session()


def test_session_methods_reject_closed_session(tmp_path):
    from tests.models.fused_elementwise import build_spec

    artifact = dml.compile(build_spec(), dml.Target("cpu"), tmp_path / "closed_session_cpu.dinoml")
    module = runtime.load(artifact.path)
    session = module.create_session()
    session.close()

    x = np.zeros((2, 3, 4), dtype=np.float32)
    with pytest.raises(RuntimeError, match="Session is closed"):
        session.run_numpy({"x": x})
    with pytest.raises(RuntimeError, match="Session is closed"):
        session.get_output_shape("y")
    with pytest.raises(RuntimeError, match="Session is closed"):
        session.set_stream(None)

    module.close()


def test_closing_runtime_module_closes_live_sessions(tmp_path):
    from tests.models.fused_elementwise import build_spec

    artifact = dml.compile(build_spec(), dml.Target("cpu"), tmp_path / "module_closes_sessions_cpu.dinoml")
    module = runtime.load(artifact.path)
    session = module.create_session()

    module.close()

    assert not session._handle
    with pytest.raises(RuntimeError, match="Session is closed"):
        session.run_numpy({"x": np.zeros((2, 3, 4), dtype=np.float32)})
    with pytest.raises(RuntimeError, match="Session is closed"):
        session.set_stream(None)


def test_runtime_module_close_attempts_all_sessions_before_reporting_failure():
    closed = []
    freed = []

    class FakeSession:
        def __init__(self, name, should_fail=False):
            self.name = name
            self.should_fail = should_fail

        def close(self):
            closed.append(self.name)
            if self.should_fail:
                raise RuntimeError(f"{self.name} close failed")

    class FakeDll:
        def dino_module_free(self, handle):
            freed.append(handle.value)
            return 0

    module = runtime.RuntimeModule.__new__(runtime.RuntimeModule)
    module._handle = ctypes.c_void_p(0x1234)
    module._sessions = [
        FakeSession("first", should_fail=True),
        FakeSession("second"),
        FakeSession("third"),
    ]
    module._dll = FakeDll()
    module._check = lambda err: None

    with pytest.raises(RuntimeError, match="first close failed"):
        module.close()

    assert closed == ["first", "second", "third"]
    assert freed == []
    assert module._handle.value == 0x1234

    closed.clear()
    module._sessions[0].should_fail = False
    module.close()

    assert closed == ["first", "second", "third"]
    assert freed == [0x1234]
    assert not module._handle


def test_session_init_destroys_partial_native_handle_when_create_fails():
    destroys = []

    class FakeDll:
        def dino_session_create(self, module_handle, out_handle):
            out_handle._obj.value = 0x1234
            return 13

        def dino_session_destroy(self, handle):
            destroys.append(handle.value)
            return 0

    def check(err):
        if err == 13:
            raise RuntimeError("session create failed")
        if err:
            raise RuntimeError("unexpected runtime failure")

    module = SimpleNamespace(
        _handle=ctypes.c_void_p(0x9999),
        _dll=FakeDll(),
        _sessions=set(),
        _check=check,
    )

    with pytest.raises(RuntimeError, match="session create failed"):
        runtime.Session(module)

    assert destroys == [0x1234]
    assert module._sessions == set()


def test_session_init_rejects_success_with_null_native_handle():
    destroys = []

    class FakeDll:
        def dino_session_create(self, module_handle, out_handle):
            return 0

        def dino_session_destroy(self, handle):
            destroys.append(handle.value)
            return 0

    module = SimpleNamespace(
        _handle=ctypes.c_void_p(0x9999),
        _dll=FakeDll(),
        _sessions=set(),
        _check=lambda err: None,
    )

    with pytest.raises(RuntimeError, match="Native session create returned a null session handle"):
        runtime.Session(module)

    assert destroys == []
    assert module._sessions == set()


def test_session_init_notes_cleanup_failure_when_tracking_fails():
    destroys = []

    class FailingSessionSet:
        def add(self, session):
            raise RuntimeError("session tracking failed")

    class FakeDll:
        def dino_session_create(self, module_handle, out_handle):
            out_handle._obj.value = 0x1234
            return 0

        def dino_session_destroy(self, handle):
            destroys.append(handle.value)
            return 17

    def check(err):
        if err == 17:
            raise RuntimeError("session destroy failed")
        if err:
            raise RuntimeError("unexpected runtime failure")

    module = SimpleNamespace(
        _handle=ctypes.c_void_p(0x9999),
        _dll=FakeDll(),
        _sessions=FailingSessionSet(),
        _check=check,
    )

    with pytest.raises(RuntimeError, match="session tracking failed") as excinfo:
        runtime.Session(module)

    notes = getattr(excinfo.value, "__notes__", [])
    assert any("Additionally failed to destroy partially created runtime session" in note for note in notes)
    assert 0x1234 in destroys


def test_runtime_module_init_frees_native_module_when_metadata_load_fails(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "metadata_load_failure.dinoml"
    (artifact_dir / "lib").mkdir(parents=True)
    write_json(
        artifact_dir / "manifest.json",
        {
            "runtime_abi_version": RUNTIME_ABI_VERSION,
            "target": {"name": "cpu"},
            "constant_load_policy": "eager",
            "files": {
                "runtime_library": "lib/libdinoml_runtime.so",
                "kernel_library": "lib/libdinoml_cuda_kernels.so",
            },
        },
    )
    (artifact_dir / "module.so").write_bytes(b"")
    (artifact_dir / "lib" / "libdinoml_runtime.so").write_bytes(b"")
    (artifact_dir / "lib" / "libdinoml_cuda_kernels.so").write_bytes(b"")

    freed = []

    class FakeSymbol:
        def __init__(self, func):
            self.func = func

        def __call__(self, *args):
            return self.func(*args)

    class FakeRuntimeDll:
        def __init__(self):
            self.dino_abi_version = FakeSymbol(lambda: RUNTIME_ABI_VERSION)
            self.dino_get_last_error = FakeSymbol(lambda: b"")

    class FakeModuleDll:
        def __init__(self):
            self.dino_module_load = FakeSymbol(self._load)
            self.dino_module_load_deferred = FakeSymbol(self._load)
            self.dino_module_free = FakeSymbol(self._free)
            self.dino_module_get_metadata_json = FakeSymbol(lambda handle: b"{not-json")
            self.dino_module_load_constants = FakeSymbol(lambda *args: 0)
            self.dino_module_unload_constants = FakeSymbol(lambda *args: 0)
            self.dino_module_set_constant = FakeSymbol(lambda *args: 0)
            self.dino_session_create = FakeSymbol(lambda *args: 0)
            self.dino_session_destroy = FakeSymbol(lambda *args: 0)
            self.dino_session_set_stream = FakeSymbol(lambda *args: 0)
            self.dino_session_get_output_shape = FakeSymbol(lambda *args: 0)
            self.dino_session_run = FakeSymbol(lambda *args: 0)

        def _load(self, artifact_path, out_handle):
            out_handle._obj.value = 0x1234
            return 0

        def _free(self, handle):
            freed.append(handle.value)
            return 0

    runtime_dll = FakeRuntimeDll()
    kernels_dll = SimpleNamespace()
    module_dll = FakeModuleDll()

    def fake_cdll(path, mode=0):
        path = str(path)
        if path.endswith("libdinoml_runtime.so"):
            return runtime_dll
        if path.endswith("libdinoml_cuda_kernels.so"):
            return kernels_dll
        if path.endswith("module.so"):
            return module_dll
        raise AssertionError(f"Unexpected library load: {path}")

    monkeypatch.setattr(runtime.ctypes, "CDLL", fake_cdll)

    with pytest.raises(ValueError, match="Expecting property name"):
        runtime.load(artifact_dir)

    assert freed == [0x1234]


def test_runtime_module_init_rejects_success_with_null_native_handle(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "null_module_handle.dinoml"
    (artifact_dir / "lib").mkdir(parents=True)
    write_json(
        artifact_dir / "manifest.json",
        {
            "runtime_abi_version": RUNTIME_ABI_VERSION,
            "target": {"name": "cpu"},
            "constant_load_policy": "eager",
            "files": {
                "runtime_library": "lib/libdinoml_runtime.so",
                "kernel_library": "lib/libdinoml_cuda_kernels.so",
            },
        },
    )
    (artifact_dir / "module.so").write_bytes(b"")
    (artifact_dir / "lib" / "libdinoml_runtime.so").write_bytes(b"")
    (artifact_dir / "lib" / "libdinoml_cuda_kernels.so").write_bytes(b"")

    freed = []
    metadata_reads = []

    class FakeSymbol:
        def __init__(self, func):
            self.func = func

        def __call__(self, *args):
            return self.func(*args)

    class FakeRuntimeDll:
        def __init__(self):
            self.dino_abi_version = FakeSymbol(lambda: RUNTIME_ABI_VERSION)
            self.dino_get_last_error = FakeSymbol(lambda: b"")

    class FakeModuleDll:
        def __init__(self):
            self.dino_module_load = FakeSymbol(lambda artifact_path, out_handle: 0)
            self.dino_module_load_deferred = FakeSymbol(lambda artifact_path, out_handle: 0)
            self.dino_module_free = FakeSymbol(self._free)
            self.dino_module_get_metadata_json = FakeSymbol(self._metadata)
            self.dino_module_load_constants = FakeSymbol(lambda *args: 0)
            self.dino_module_unload_constants = FakeSymbol(lambda *args: 0)
            self.dino_module_set_constant = FakeSymbol(lambda *args: 0)
            self.dino_session_create = FakeSymbol(lambda *args: 0)
            self.dino_session_destroy = FakeSymbol(lambda *args: 0)
            self.dino_session_set_stream = FakeSymbol(lambda *args: 0)
            self.dino_session_get_output_shape = FakeSymbol(lambda *args: 0)
            self.dino_session_run = FakeSymbol(lambda *args: 0)

        def _free(self, handle):
            freed.append(handle.value)
            return 0

        def _metadata(self, handle):
            metadata_reads.append(handle.value)
            return b"{}"

    runtime_dll = FakeRuntimeDll()
    kernels_dll = SimpleNamespace()
    module_dll = FakeModuleDll()

    def fake_cdll(path, mode=0):
        path = str(path)
        if path.endswith("libdinoml_runtime.so"):
            return runtime_dll
        if path.endswith("libdinoml_cuda_kernels.so"):
            return kernels_dll
        if path.endswith("module.so"):
            return module_dll
        raise AssertionError(f"Unexpected library load: {path}")

    monkeypatch.setattr(runtime.ctypes, "CDLL", fake_cdll)

    with pytest.raises(RuntimeError, match="Native module load returned a null module handle"):
        runtime.load(artifact_dir)

    assert freed == []
    assert metadata_reads == []


def test_cuda_session_entrypoints_reject_closed_session_before_backend_checks():
    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p()
    session.module = SimpleNamespace(target_name="cpu", metadata={"inputs": [], "outputs": []})

    with pytest.raises(RuntimeError, match="Session is closed"):
        session.run_device_pointers({}, {})
    with pytest.raises(RuntimeError, match="Session is closed"):
        session.run_torch({})


def test_cuda_session_tracks_external_stream_state():
    calls = []

    class FakeModuleDll:
        def dino_session_set_stream(self, handle, stream):
            calls.append((handle.value, stream.value))
            return 0

    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(0x1234)
    session._external_stream = False
    session.module = SimpleNamespace(
        _dll=FakeModuleDll(),
        _check=lambda err: err,
    )

    session.set_stream(ctypes.c_void_p(0xBEEF))
    assert session._external_stream is True
    session.set_stream(None)
    assert session._external_stream is False
    assert calls == [(0x1234, 0xBEEF), (0x1234, None)]


def test_cuda_device_pointer_run_skips_external_stream_shape_buffer_capacity_check():
    runs = []

    class FakeModuleDll:
        def dino_session_run(self, handle, inputs, input_count, outputs, output_count):
            runs.append((handle.value, input_count.value, output_count.value))
            return 0

    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(0x1234)
    session._external_stream = True
    session.module = SimpleNamespace(
        target_name="cuda",
        metadata={
            "inputs": [{"name": "x", "shape": [2], "shape_spec": [2], "dtype": "float32"}],
            "outputs": [{"name": "y", "shape": [2], "shape_spec": [2], "dtype": "float32"}],
            "output_shape_reports": {
                "version": 1,
                "reports": [{"output": "y", "kind": "shape_buffer"}],
            },
        },
        _dll=FakeModuleDll(),
        _check=lambda err: err,
    )
    session.get_output_shape = lambda _name: (_ for _ in ()).throw(
        AssertionError("shape-buffer reports should be unavailable on external-stream runs")
    )

    session.run_device_pointers(
        {"x": 0x1000},
        {"y": 0x2000},
        {"x": (2,)},
        {"y": (2,)},
    )

    assert runs == [(0x1234, 1, 1)]


def test_cuda_device_pointer_run_keeps_caller_shape_capacity_check_on_external_stream():
    class FakeModuleDll:
        def dino_session_run(self, handle, inputs, input_count, outputs, output_count):
            return 0

    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(0x1234)
    session._external_stream = True
    session.module = SimpleNamespace(
        target_name="cuda",
        metadata={
            "inputs": [{"name": "x", "shape": [2], "shape_spec": [2], "dtype": "float32"}],
            "outputs": [{"name": "y", "shape": [2], "shape_spec": [2], "dtype": "float32"}],
            "output_shape_reports": {"version": 1, "reports": []},
        },
        _dll=FakeModuleDll(),
        _check=lambda err: err,
    )
    session.get_output_shape = lambda _name: (3,)

    with pytest.raises(ValueError, match="has more elements than allocated"):
        session.run_device_pointers(
            {"x": 0x1000},
            {"y": 0x2000},
            {"x": (2,)},
            {"y": (2,)},
        )


def test_cuda_staging_allocator_keeps_cached_buffer_when_grow_allocation_fails():
    frees = []

    class FakeCudaRuntime:
        def dino_device_malloc(self, out_ptr, nbytes):
            assert nbytes.value == 32
            return 7

        def dino_device_free(self, ptr):
            frees.append(ptr.value)
            return 0

    def check(err):
        if err:
            raise RuntimeError("device allocation failed")

    session = object.__new__(runtime.Session)
    session._cuda_buffers = {"input:x": (ctypes.c_void_p(0x1000), 16)}
    session.module = SimpleNamespace(_cuda_runtime_dll=FakeCudaRuntime(), _check=check, _check_cuda_runtime=check)

    with pytest.raises(RuntimeError, match="device allocation failed"):
        session._device_buffer("input:x", 32)

    assert session._cuda_buffers["input:x"][0].value == 0x1000
    assert session._cuda_buffers["input:x"][1] == 16
    assert frees == []


def test_cuda_staging_allocator_grow_swaps_cache_after_freeing_old_buffer():
    frees = []

    class FakeCudaRuntime:
        def dino_device_malloc(self, out_ptr, nbytes):
            assert nbytes.value == 32
            out_ptr._obj.value = 0x2000
            return 0

        def dino_device_free(self, ptr):
            frees.append(ptr.value)
            return 0

    def check(err):
        if err:
            raise RuntimeError("unexpected runtime failure")

    session = object.__new__(runtime.Session)
    session._cuda_buffers = {"input:x": (ctypes.c_void_p(0x1000), 16)}
    session.module = SimpleNamespace(_cuda_runtime_dll=FakeCudaRuntime(), _check=check, _check_cuda_runtime=check)

    ptr = session._device_buffer("input:x", 32)

    assert ptr.value == 0x2000
    assert session._cuda_buffers["input:x"][0].value == 0x2000
    assert session._cuda_buffers["input:x"][1] == 32
    assert frees == [0x1000]


def test_cuda_staging_allocator_forget_successful_frees_when_close_free_fails():
    frees = []
    fail_ptr = 0x1000

    class FakeCudaRuntime:
        def dino_device_free(self, ptr):
            frees.append(ptr.value)
            return 9 if ptr.value == fail_ptr else 0

    def check(err):
        if err:
            raise RuntimeError("device free failed")

    session = object.__new__(runtime.Session)
    session._cuda_buffers = {
        "input:x": (ctypes.c_void_p(0x1000), 16),
        "output:y": (ctypes.c_void_p(0x2000), 32),
    }
    session.module = SimpleNamespace(_cuda_runtime_dll=FakeCudaRuntime(), _check=check, _check_cuda_runtime=check)

    with pytest.raises(RuntimeError, match="device free failed"):
        session._free_cuda_buffers()

    assert frees == [0x2000, 0x1000]
    assert list(session._cuda_buffers) == ["input:x"]
    assert session._cuda_buffers["input:x"][0].value == 0x1000
    assert session._cuda_buffers["input:x"][1] == 16

    fail_ptr = 0
    session._free_cuda_buffers()

    assert frees == [0x2000, 0x1000, 0x1000]
    assert session._cuda_buffers == {}


def test_session_close_still_destroys_native_session_when_staging_free_fails():
    frees = []
    destroys = []
    fail_ptr = 0x1000

    class FakeCudaRuntime:
        def dino_device_free(self, ptr):
            frees.append(ptr.value)
            return 9 if ptr.value == fail_ptr else 0

    class FakeModuleDll:
        def dino_session_destroy(self, handle):
            destroys.append(handle.value)
            return 0

    def check(err):
        if err:
            raise RuntimeError("device free failed")

    sessions = set()
    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(0x1234)
    session._cuda_buffers = {
        "input:x": (ctypes.c_void_p(0x1000), 16),
        "output:y": (ctypes.c_void_p(0x2000), 32),
    }
    session.module = SimpleNamespace(
        _cuda_runtime_dll=FakeCudaRuntime(),
        _dll=FakeModuleDll(),
        _sessions=sessions,
        _check=check,
        _check_cuda_runtime=check,
    )
    sessions.add(session)

    with pytest.raises(RuntimeError, match="device free failed"):
        session.close()

    assert frees == [0x2000, 0x1000]
    assert destroys == [0x1234]
    assert not session._handle
    assert session in sessions
    assert list(session._cuda_buffers) == ["input:x"]

    fail_ptr = 0
    session.close()

    assert frees == [0x2000, 0x1000, 0x1000]
    assert destroys == [0x1234]
    assert session._cuda_buffers == {}
    assert session not in sessions


def test_session_close_keeps_native_handle_retryable_when_destroy_fails_after_buffer_cleanup():
    frees = []
    destroys = []
    destroy_fails = True

    class FakeCudaRuntime:
        def dino_device_free(self, ptr):
            frees.append(ptr.value)
            return 0

    class FakeModuleDll:
        def dino_session_destroy(self, handle):
            destroys.append(handle.value)
            return 11 if destroy_fails else 0

    def check(err):
        if err == 11:
            raise RuntimeError("destroy failed")
        if err:
            raise RuntimeError("unexpected runtime failure")

    sessions = set()
    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(0x1234)
    session._cuda_buffers = {"input:x": (ctypes.c_void_p(0x1000), 16)}
    session.module = SimpleNamespace(
        _cuda_runtime_dll=FakeCudaRuntime(),
        _dll=FakeModuleDll(),
        _sessions=sessions,
        _check=check,
        _check_cuda_runtime=check,
    )
    sessions.add(session)

    with pytest.raises(RuntimeError, match="destroy failed"):
        session.close()

    assert frees == [0x1000]
    assert destroys == [0x1234]
    assert session._handle.value == 0x1234
    assert session._cuda_buffers == {}
    assert session in sessions

    destroy_fails = False
    session.close()

    assert frees == [0x1000]
    assert destroys == [0x1234, 0x1234]
    assert not session._handle
    assert session not in sessions


def test_runtime_check_reads_cuda_runtime_last_error():
    class FakeGetter:
        restype = None

        def __call__(self):
            return b"cuda helper failed"

    class FakeCudaRuntime:
        dino_get_last_error = FakeGetter()

    module = object.__new__(runtime.RuntimeModule)
    module._dll = SimpleNamespace()
    module._runtime_dll = SimpleNamespace()
    module._cuda_runtime_dll = FakeCudaRuntime()

    with pytest.raises(RuntimeError, match="cuda helper failed"):
        module._check(1)


def test_cuda_runtime_check_ignores_stale_module_last_error():
    class FakeGetter:
        restype = None

        def __init__(self, message):
            self.message = message

        def __call__(self):
            return self.message

    module = object.__new__(runtime.RuntimeModule)
    module._dll = SimpleNamespace(dino_get_last_error=FakeGetter(b"stale module failure"))
    module._runtime_dll = SimpleNamespace()
    module._cuda_runtime_dll = SimpleNamespace(dino_get_last_error=FakeGetter(b"fresh cuda helper failure"))

    with pytest.raises(RuntimeError, match="fresh cuda helper failure"):
        module._check_cuda_runtime(1)


def test_constant_load_unload_rejects_closed_runtime_module(tmp_path):
    from tests.models.fused_elementwise import build_spec

    artifact = dml.compile(build_spec(), dml.Target("cpu"), tmp_path / "closed_module_constants_cpu.dinoml")
    module = runtime.load(artifact.path)
    module.close()

    with pytest.raises(RuntimeError, match="RuntimeModule is closed"):
        module.load_constants_from_file()
    with pytest.raises(RuntimeError, match="RuntimeModule is closed"):
        module.unload_constants()


def test_constant_setters_reject_closed_runtime_module_after_name_validation(tmp_path):
    from tests.models.fused_elementwise import build_spec

    spec = build_spec()
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "closed_module_constant_setters_cpu.dinoml")
    module = runtime.load(artifact.path)
    module.close()

    value = np.zeros_like(spec.constants["scale"])
    tensor = SimpleNamespace(is_cuda=False, is_contiguous=lambda: False, shape=value.shape)

    with pytest.raises(ValueError, match="Unknown constant: missing"):
        module.set_constant_numpy("missing", value)
    with pytest.raises(ValueError, match="Unknown constant: missing"):
        module.set_constant_device_pointer("missing", 0x1000, value.shape, "float32")
    with pytest.raises(ValueError, match="Unknown constant: missing"):
        module.set_constant_torch("missing", tensor)

    with pytest.raises(RuntimeError, match="RuntimeModule is closed"):
        module.set_constant_numpy("scale", value)
    with pytest.raises(RuntimeError, match="RuntimeModule is closed"):
        module.set_constant_device_pointer("scale", 0x1000, value.shape, "float32")
    with pytest.raises(RuntimeError, match="RuntimeModule is closed"):
        module.set_constant_torch("scale", tensor)


def test_set_constant_torch_rejects_unknown_constant_before_tensor_checks():
    module = object.__new__(runtime.RuntimeModule)
    module.metadata = {
        "constants": [
            {
                "name": "scale",
                "shape": [4],
                "shape_spec": [4],
                "dtype": "float32",
            }
        ]
    }

    tensor = SimpleNamespace(is_cuda=False, is_contiguous=lambda: False, shape=(4,))
    with pytest.raises(ValueError, match="Unknown constant: missing"):
        module.set_constant_torch("missing", tensor)


def test_set_constant_device_pointer_rejects_unknown_constant_before_backend_checks():
    module = object.__new__(runtime.RuntimeModule)
    module.target_name = "cpu"
    module.metadata = {
        "constants": [
            {
                "name": "scale",
                "shape": [4],
                "shape_spec": [4],
                "dtype": "float32",
            }
        ]
    }

    with pytest.raises(ValueError, match="Unknown constant: missing"):
        module.set_constant_device_pointer("missing", 0x1000, (4,), "float32")


def test_cuda_set_constant_numpy_preserves_setter_failure_when_temp_free_fails():
    events = []

    class FakeCudaRuntime:
        def dino_device_malloc(self, out_ptr, nbytes):
            events.append(("malloc", nbytes.value))
            out_ptr._obj.value = 0x2000
            return 0

        def dino_copy_host_to_device(self, dst, src, nbytes):
            events.append(("copy_h2d", dst.value, nbytes.value))
            return 0

        def dino_device_free(self, ptr):
            events.append(("free", ptr.value))
            return 17

    class FakeModuleDll:
        def dino_module_set_constant(self, handle, name, tensor):
            tensor_value = ctypes.cast(tensor, ctypes.POINTER(runtime._DinoTensor)).contents
            events.append(("set", handle.value, name.decode("utf-8"), tensor_value.data))
            return 13

    def check(err):
        if err == 13:
            raise RuntimeError("set constant failed")
        if err == 17:
            raise RuntimeError("temporary free failed")
        if err:
            raise RuntimeError("unexpected runtime failure")

    module = object.__new__(runtime.RuntimeModule)
    module.target_name = "cuda"
    module._handle = ctypes.c_void_p(0x1234)
    module._cuda_runtime_dll = FakeCudaRuntime()
    module._dll = FakeModuleDll()
    module._check = check
    module._check_cuda_runtime = check
    module.metadata = {
        "constants": [
            {
                "name": "scale",
                "shape": [2],
                "shape_spec": [2],
                "dtype": "float32",
            }
        ]
    }
    module._constant_loaded = {"scale": False}

    with pytest.raises(RuntimeError, match="set constant failed"):
        module.set_constant_numpy("scale", np.array([1.0, 2.0], dtype=np.float32))

    assert events == [
        ("malloc", 8),
        ("copy_h2d", 0x2000, 8),
        ("set", 0x1234, "scale", 0x2000),
        ("free", 0x2000),
    ]
    assert module.constant_load_state() == {"scale": False}


@pytest.mark.parametrize("previously_loaded", [False, True])
def test_cpu_set_constant_numpy_failure_preserves_loaded_state(previously_loaded):
    events = []

    class FakeModuleDll:
        def dino_module_set_constant(self, handle, name, tensor):
            tensor_value = ctypes.cast(tensor, ctypes.POINTER(runtime._DinoTensor)).contents
            events.append(("set", handle.value, name.decode("utf-8"), tensor_value.data))
            return 19

    def check(err):
        if err == 19:
            raise RuntimeError("set constant failed")
        if err:
            raise RuntimeError("unexpected runtime failure")

    module = object.__new__(runtime.RuntimeModule)
    module.target_name = "cpu"
    module._handle = ctypes.c_void_p(0x1234)
    module._dll = FakeModuleDll()
    module._check = check
    module.metadata = {
        "constants": [
            {
                "name": "scale",
                "shape": [2],
                "shape_spec": [2],
                "dtype": "float32",
            }
        ]
    }
    module._constant_loaded = {"scale": previously_loaded}

    with pytest.raises(RuntimeError, match="set constant failed"):
        module.set_constant_numpy("scale", np.array([1.0, 2.0], dtype=np.float32))

    assert len(events) == 1
    assert events[0][0:2] == ("set", 0x1234)
    assert module.constant_load_state() == {"scale": previously_loaded}


@pytest.mark.parametrize("previously_loaded", [False, True])
def test_cuda_set_constant_device_pointer_failure_preserves_loaded_state(previously_loaded):
    events = []

    class FakeModuleDll:
        def dino_module_set_constant(self, handle, name, tensor):
            tensor_value = ctypes.cast(tensor, ctypes.POINTER(runtime._DinoTensor)).contents
            events.append(
                (
                    "set",
                    handle.value,
                    name.decode("utf-8"),
                    tensor_value.data,
                    tensor_value.device_type,
                )
            )
            return 23

    def check(err):
        if err == 23:
            raise RuntimeError("set constant failed")
        if err:
            raise RuntimeError("unexpected runtime failure")

    module = object.__new__(runtime.RuntimeModule)
    module.target_name = "cuda"
    module._handle = ctypes.c_void_p(0x5678)
    module._dll = FakeModuleDll()
    module._check = check
    module.metadata = {
        "constants": [
            {
                "name": "scale",
                "shape": [2],
                "shape_spec": [2],
                "dtype": "float32",
            }
        ]
    }
    module._constant_loaded = {"scale": previously_loaded}

    with pytest.raises(RuntimeError, match="set constant failed"):
        module.set_constant_device_pointer("scale", 0x4000, (2,), "float32")

    assert events == [("set", 0x5678, "scale", 0x4000, runtime.DINO_DEVICE_CUDA)]
    assert module.constant_load_state() == {"scale": previously_loaded}


def test_compile_rejects_unknown_constant_load_policy(tmp_path):
    from tests.models.fused_elementwise import build_spec

    with pytest.raises(ValueError, match="Unsupported constant_load_policy"):
        dml.compile(
            build_spec(),
            dml.Target("cpu"),
            tmp_path / "bad_constant_load_policy.dinoml",
            constant_load_policy="lazy",
        )


def test_compile_writes_encoded_constants_manifest(tmp_path):
    storage = {
        "kind": "gguf",
        "path": "weights.gguf",
        "tensor": "blk.0.ffn.weight",
        "logical_dtype": "float32",
        "shape": [2, 2],
        "qtype": "F32",
        "encoded_nbytes": 16,
        "materialization": "dequantize_full_before_launch",
        "residency": "eager_dense_device",
    }
    traced = dml.trace(
        EncodedWeightModel(),
        inputs={"x": dml.TensorSpec([2, 2])},
        constants={"weight": MaterializingConstant([[1.0, 2.0], [3.0, 4.0]], storage)},
        name="encoded_weight",
    )
    spec = ModelSpec(
        name=traced.name,
        ir=traced.ir,
        constants={"weight": MaterializingConstant([[1.0, 2.0], [3.0, 4.0]], storage)},
    )

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "encoded_weight.dinoml")

    manifest = read_json(artifact.path / "manifest.json")
    assert manifest["files"]["encoded_constants"] == "encoded_constants.json"
    encoded_constants = read_json(artifact.path / "encoded_constants.json")
    assert encoded_constants["summary"] == {
        "constant_count": 1,
        "logical_nbytes": 16,
        "encoded_nbytes": 16,
        "runtime_supported_count": 1,
    }
    assert encoded_constants["constants"][0]["policy"]["residency_status"] == "runtime_supported"


def test_runtime_encoded_constant_load_plan_reads_encoded_constants_manifest(tmp_path):
    module = runtime.RuntimeModule.__new__(runtime.RuntimeModule)
    module.artifact_dir = tmp_path
    module.manifest = {"files": {"encoded_constants": "encoded_constants.json"}}
    module.metadata = {"constants": []}
    write_json(
        tmp_path / "encoded_constants.json",
        {
            "schema_version": 1,
            "kind": "dinoml.encoded_constants",
            "constants": [
                {
                    "name": "weight",
                    "dtype": "float32",
                    "shape": [3, 2],
                    "logical_nbytes": 24,
                    "storage": {
                        "kind": "gguf",
                        "path": "weights.gguf",
                        "tensor": "blk.0.ffn.weight",
                        "logical_dtype": "float32",
                        "shape": [3, 2],
                        "qtype": "F32",
                        "encoded_nbytes": 24,
                        "materialization": "dequantize_full_before_launch",
                        "residency": "eager_dense_device",
                    },
                    "policy": {
                        "materialization": "dequantize_full_before_launch",
                        "materialization_status": "runtime_supported",
                        "residency": "eager_dense_device",
                        "residency_status": "runtime_supported",
                    },
                }
            ],
        },
    )

    plan = module.encoded_constant_load_plan()

    assert plan == [
        {
            "name": "weight",
            "dtype": "float32",
            "shape": [3, 2],
            "logical_nbytes": 24,
            "storage_kind": "gguf",
            "storage_path": "weights.gguf",
            "storage_tensor": "blk.0.ffn.weight",
            "storage": {
                "kind": "gguf",
                "path": "weights.gguf",
                "tensor": "blk.0.ffn.weight",
                "logical_dtype": "float32",
                "shape": [3, 2],
                "qtype": "F32",
                "encoded_nbytes": 24,
                "materialization": "dequantize_full_before_launch",
                "residency": "eager_dense_device",
            },
            "policy": {
                "materialization": "dequantize_full_before_launch",
                "materialization_status": "runtime_supported",
                "residency": "eager_dense_device",
                "residency_status": "runtime_supported",
            },
            "runtime_supported": True,
            "loadable_now": True,
        }
    ]


def test_runtime_encoded_constant_load_plan_falls_back_to_metadata_constants(tmp_path):
    module = runtime.RuntimeModule.__new__(runtime.RuntimeModule)
    module.artifact_dir = tmp_path
    module.manifest = {"files": {}}
    module.metadata = {
        "constants": [
            {
                "name": "weight",
                "dtype": "float32",
                "shape": [2, 2],
                "nbytes": 16,
                "storage": {
                    "kind": "gguf",
                    "path": "weights.gguf",
                    "tensor": "blk.0.ffn.weight",
                    "logical_dtype": "float32",
                    "shape": [2, 2],
                },
            }
        ]
    }

    plan = module.encoded_constant_load_plan()

    assert plan[0]["name"] == "weight"
    assert plan[0]["logical_nbytes"] == 16
    assert plan[0]["storage_path"] == "weights.gguf"
    assert plan[0]["storage_tensor"] == "blk.0.ffn.weight"
    assert plan[0]["policy"] == {
        "materialization": "dequantize_full_before_launch",
        "materialization_status": "runtime_supported",
        "residency": "eager_dense_device",
        "residency_status": "runtime_supported",
    }
    assert plan[0]["loadable_now"] is True


def test_runtime_load_encoded_constants_filters_names_and_rejects_unknown(monkeypatch, tmp_path):
    values = {
        "weight": np.array([1.0, 2.0], dtype=np.float32),
        "bias": np.array([3.0, 4.0], dtype=np.float32),
    }

    def get_tensor(name):
        return SimpleNamespace(name=name, qtype="F32", qtype_value=0, shape=(2,), data_offset=128)

    fake_file = SimpleNamespace(
        get_tensor=get_tensor,
        read_tensor_bytes=lambda tensor: values[tensor.name].tobytes(order="C"),
    )
    monkeypatch.setitem(sys.modules, "libgguf", SimpleNamespace(open_gguf=lambda path: fake_file))

    module = runtime.RuntimeModule.__new__(runtime.RuntimeModule)
    module.artifact_dir = tmp_path
    module.manifest = {"files": {"encoded_constants": "encoded_constants.json"}}
    module.metadata = {
        "constants": [
            {"name": name, "shape": [2], "shape_spec": [2], "dtype": "float32"}
            for name in values
        ]
    }
    module._constant_loaded = {name: False for name in values}
    module._handle = ctypes.c_void_p(1)
    write_json(
        tmp_path / "encoded_constants.json",
        {
            "schema_version": 1,
            "kind": "dinoml.encoded_constants",
            "constants": [
                {
                    "name": name,
                    "dtype": "float32",
                    "shape": [2],
                    "logical_nbytes": value.nbytes,
                    "storage": {
                        "kind": "gguf",
                        "path": "weights.gguf",
                        "tensor": name,
                        "logical_dtype": "float32",
                        "shape": [2],
                        "qtype": "F32",
                        "encoded_nbytes": value.nbytes,
                        "materialization": "dequantize_full_before_launch",
                        "residency": "manual_runtime_load" if name == "bias" else "eager_dense_device",
                    },
                }
                for name, value in values.items()
            ],
        },
    )
    captured = {}
    def set_constant_numpy(name, value):
        captured.setdefault(name, np.array(value, copy=True))
        module._mark_constant_loaded(name, True)

    module.set_constant_numpy = set_constant_numpy

    plan = module.encoded_constant_load_plan(names=["bias"])
    assert plan[0]["policy"]["residency"] == "manual_runtime_load"
    assert plan[0]["policy"]["residency_status"] == "runtime_supported"
    assert plan[0]["loadable_now"] is True

    module.load_encoded_constants(names=["bias"])

    assert list(captured) == ["bias"]
    np.testing.assert_array_equal(captured["bias"], values["bias"])
    assert module.constant_load_state() == {"weight": False, "bias": True}
    with pytest.raises(ValueError, match="Unknown encoded constant"):
        module.encoded_constant_load_plan(names=["missing"])
    with pytest.raises(ValueError, match="Unknown encoded constant"):
        module.load_encoded_constants(names=["missing"])


def test_runtime_load_encoded_constants_rejects_missing_runtime_constant_before_materialize(monkeypatch, tmp_path):
    def fail_open_gguf(path):
        raise AssertionError("malformed encoded constants should fail before opening GGUF storage")

    monkeypatch.setitem(sys.modules, "libgguf", SimpleNamespace(open_gguf=fail_open_gguf))
    module = runtime.RuntimeModule.__new__(runtime.RuntimeModule)
    module.artifact_dir = tmp_path
    module.manifest = {"files": {"encoded_constants": "encoded_constants.json"}}
    module.metadata = {"constants": [{"name": "weight", "shape": [2], "shape_spec": [2], "dtype": "float32"}]}
    module._handle = ctypes.c_void_p(1)
    write_json(
        tmp_path / "encoded_constants.json",
        {
            "schema_version": 1,
            "kind": "dinoml.encoded_constants",
            "constants": [
                {
                    "name": "missing_runtime",
                    "dtype": "float32",
                    "shape": [2],
                    "logical_nbytes": 8,
                    "storage": {
                        "kind": "gguf",
                        "path": "weights.gguf",
                        "tensor": "missing_runtime",
                        "logical_dtype": "float32",
                        "shape": [2],
                        "qtype": "F32",
                        "encoded_nbytes": 8,
                        "materialization": "dequantize_full_before_launch",
                        "residency": "manual_runtime_load",
                    },
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="Encoded constant\\(s\\) are not runtime constants: missing_runtime"):
        module.load_encoded_constants()


def test_runtime_load_encoded_constants_rejects_duplicate_manifest_names_before_materialize(monkeypatch, tmp_path):
    def fail_open_gguf(path):
        raise AssertionError("duplicate encoded constants should fail before opening GGUF storage")

    monkeypatch.setitem(sys.modules, "libgguf", SimpleNamespace(open_gguf=fail_open_gguf))
    module = runtime.RuntimeModule.__new__(runtime.RuntimeModule)
    module.artifact_dir = tmp_path
    module.manifest = {"files": {"encoded_constants": "encoded_constants.json"}}
    module.metadata = {"constants": [{"name": "weight", "shape": [2], "shape_spec": [2], "dtype": "float32"}]}
    module._handle = ctypes.c_void_p(1)
    write_json(
        tmp_path / "encoded_constants.json",
        {
            "schema_version": 1,
            "kind": "dinoml.encoded_constants",
            "constants": [
                {
                    "name": "weight",
                    "dtype": "float32",
                    "shape": [2],
                    "logical_nbytes": 8,
                    "storage": {
                        "kind": "gguf",
                        "path": "weights.gguf",
                        "tensor": "weight",
                        "logical_dtype": "float32",
                        "shape": [2],
                        "qtype": "F32",
                        "encoded_nbytes": 8,
                        "materialization": "dequantize_full_before_launch",
                        "residency": "manual_runtime_load",
                    },
                },
                {
                    "name": "weight",
                    "dtype": "float32",
                    "shape": [2],
                    "logical_nbytes": 8,
                    "storage": {
                        "kind": "gguf",
                        "path": "weights.gguf",
                        "tensor": "weight_duplicate",
                        "logical_dtype": "float32",
                        "shape": [2],
                        "qtype": "F32",
                        "encoded_nbytes": 8,
                        "materialization": "dequantize_full_before_launch",
                        "residency": "manual_runtime_load",
                    },
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="duplicate encoded constant name: weight"):
        module.encoded_constant_load_plan()
    with pytest.raises(ValueError, match="duplicate encoded constant name: weight"):
        module.load_encoded_constants()


def test_runtime_load_encoded_constants_materializes_all_before_setting(monkeypatch, tmp_path):
    values = {
        "first": np.array([1.0, 2.0], dtype=np.float32),
        "second": np.array([3.0, 4.0], dtype=np.float32),
    }

    def get_tensor(name):
        return SimpleNamespace(name=name, qtype="F32", qtype_value=0, shape=(2,), data_offset=128)

    def read_tensor_bytes(tensor):
        if tensor.name == "second":
            raise ValueError("second tensor read failed")
        return values[tensor.name].tobytes(order="C")

    fake_file = SimpleNamespace(get_tensor=get_tensor, read_tensor_bytes=read_tensor_bytes)
    monkeypatch.setitem(sys.modules, "libgguf", SimpleNamespace(open_gguf=lambda path: fake_file))

    module = runtime.RuntimeModule.__new__(runtime.RuntimeModule)
    module.artifact_dir = tmp_path
    module.manifest = {"files": {"encoded_constants": "encoded_constants.json"}}
    module.metadata = {
        "constants": [
            {"name": name, "shape": [2], "shape_spec": [2], "dtype": "float32"}
            for name in values
        ]
    }
    module._handle = ctypes.c_void_p(1)
    write_json(
        tmp_path / "encoded_constants.json",
        {
            "schema_version": 1,
            "kind": "dinoml.encoded_constants",
            "constants": [
                {
                    "name": name,
                    "dtype": "float32",
                    "shape": [2],
                    "logical_nbytes": value.nbytes,
                    "storage": {
                        "kind": "gguf",
                        "path": "weights.gguf",
                        "tensor": name,
                        "logical_dtype": "float32",
                        "shape": [2],
                        "qtype": "F32",
                        "encoded_nbytes": value.nbytes,
                        "materialization": "dequantize_full_before_launch",
                        "residency": "manual_runtime_load",
                    },
                }
                for name, value in values.items()
            ],
        },
    )
    captured = {}
    module.set_constant_numpy = lambda name, value: captured.setdefault(name, np.array(value, copy=True))

    with pytest.raises(ValueError, match="second tensor read failed"):
        module.load_encoded_constants()

    assert captured == {}


def test_runtime_load_encoded_constants_restores_loaded_state_when_setter_fails(monkeypatch, tmp_path):
    values = {
        "first": np.array([1.0, 2.0], dtype=np.float32),
        "second": np.array([3.0, 4.0], dtype=np.float32),
    }

    def get_tensor(name):
        return SimpleNamespace(name=name, qtype="F32", qtype_value=0, shape=(2,), data_offset=128)

    fake_file = SimpleNamespace(
        get_tensor=get_tensor,
        read_tensor_bytes=lambda tensor: values[tensor.name].tobytes(order="C"),
    )
    monkeypatch.setitem(sys.modules, "libgguf", SimpleNamespace(open_gguf=lambda path: fake_file))

    module = runtime.RuntimeModule.__new__(runtime.RuntimeModule)
    module.artifact_dir = tmp_path
    module.manifest = {"files": {"encoded_constants": "encoded_constants.json"}}
    module.metadata = {
        "constants": [
            {"name": name, "shape": [2], "shape_spec": [2], "dtype": "float32"}
            for name in values
        ]
    }
    module._constant_loaded = {"first": False, "second": True}
    module._handle = ctypes.c_void_p(1)
    write_json(
        tmp_path / "encoded_constants.json",
        {
            "schema_version": 1,
            "kind": "dinoml.encoded_constants",
            "constants": [
                {
                    "name": name,
                    "dtype": "float32",
                    "shape": [2],
                    "logical_nbytes": value.nbytes,
                    "storage": {
                        "kind": "gguf",
                        "path": "weights.gguf",
                        "tensor": name,
                        "logical_dtype": "float32",
                        "shape": [2],
                        "qtype": "F32",
                        "encoded_nbytes": value.nbytes,
                        "materialization": "dequantize_full_before_launch",
                        "residency": "manual_runtime_load",
                    },
                }
                for name, value in values.items()
            ],
        },
    )
    captured = {}

    def set_constant_numpy(name, value):
        captured[name] = np.array(value, copy=True)
        if name == "second":
            raise RuntimeError("second setter failed")
        module._mark_constant_loaded(name, True)

    module.set_constant_numpy = set_constant_numpy

    with pytest.raises(RuntimeError, match="second setter failed"):
        module.load_encoded_constants()

    assert list(captured) == ["first", "second"]
    assert module.constant_load_state() == {"first": False, "second": True}


def test_runtime_load_encoded_constants_rejects_closed_module_before_materialize(monkeypatch, tmp_path):
    def fail_open_gguf(path):
        raise AssertionError("closed modules should fail before opening GGUF storage")

    monkeypatch.setitem(sys.modules, "libgguf", SimpleNamespace(open_gguf=fail_open_gguf))
    module = runtime.RuntimeModule.__new__(runtime.RuntimeModule)
    module.artifact_dir = tmp_path
    module.manifest = {"files": {"encoded_constants": "encoded_constants.json"}}
    module.metadata = {"constants": [{"name": "weight", "shape": [2], "shape_spec": [2], "dtype": "float32"}]}
    module._handle = ctypes.c_void_p()
    write_json(
        tmp_path / "encoded_constants.json",
        {
            "schema_version": 1,
            "kind": "dinoml.encoded_constants",
            "constants": [
                {
                    "name": "weight",
                    "dtype": "float32",
                    "shape": [2],
                    "logical_nbytes": 8,
                    "storage": {
                        "kind": "gguf",
                        "path": "weights.gguf",
                        "tensor": "weight",
                        "logical_dtype": "float32",
                        "shape": [2],
                        "qtype": "F32",
                        "encoded_nbytes": 8,
                        "materialization": "dequantize_full_before_launch",
                        "residency": "manual_runtime_load",
                    },
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="Unknown encoded constant"):
        module.load_encoded_constants(names=["missing"])
    with pytest.raises(RuntimeError, match="RuntimeModule is closed"):
        module.load_encoded_constants(names=["weight"])


def test_runtime_load_encoded_constants_rejects_future_policy_before_materialize(monkeypatch, tmp_path):
    def fail_open_gguf(path):
        raise AssertionError("future policy should fail before opening GGUF storage")

    monkeypatch.setitem(sys.modules, "libgguf", SimpleNamespace(open_gguf=fail_open_gguf))
    module = runtime.RuntimeModule.__new__(runtime.RuntimeModule)
    module.artifact_dir = tmp_path
    module.manifest = {"files": {"encoded_constants": "encoded_constants.json"}}
    module.metadata = {
        "constants": [
            {"name": "offload_weight", "shape": [2], "shape_spec": [2], "dtype": "float32"},
            {"name": "weight", "shape": [2], "shape_spec": [2], "dtype": "float32"},
        ]
    }
    module._handle = ctypes.c_void_p()
    write_json(
        tmp_path / "encoded_constants.json",
        {
            "schema_version": 1,
            "kind": "dinoml.encoded_constants",
            "constants": [
                {
                    "name": "weight",
                    "dtype": "float32",
                    "shape": [2],
                    "logical_nbytes": 8,
                    "storage": {
                        "kind": "gguf",
                        "path": "weights.gguf",
                        "tensor": "weight",
                        "logical_dtype": "float32",
                        "shape": [2],
                        "materialization": "dequantize_on_gpu_before_launch",
                        "residency": "eager_dense_device",
                    },
                },
                {
                    "name": "offload_weight",
                    "dtype": "float32",
                    "shape": [2],
                    "logical_nbytes": 8,
                    "storage": {
                        "kind": "gguf",
                        "path": "weights.gguf",
                        "tensor": "offload_weight",
                        "logical_dtype": "float32",
                        "shape": [2],
                        "materialization": "dequantize_full_before_launch",
                        "residency": "cpu_until_first_use",
                    },
                }
            ],
        },
    )

    plan = module.encoded_constant_load_plan()
    assert plan[0]["policy"]["materialization_status"] == "future"
    assert plan[0]["loadable_now"] is False
    assert plan[1]["policy"]["residency_status"] == "future"
    assert plan[1]["loadable_now"] is False
    with pytest.raises(NotImplementedError, match="Encoded constant policy is not runtime-supported"):
        module.load_encoded_constants()


def test_runtime_load_encoded_constants_materializes_gguf_metadata(monkeypatch, tmp_path):
    values = np.arange(6, dtype=np.float32).reshape(3, 2)
    tensor_info = SimpleNamespace(qtype="F32", qtype_value=0, shape=(2, 3), data_offset=128)
    fake_file = SimpleNamespace(
        get_tensor=lambda name: tensor_info,
        read_tensor_bytes=lambda tensor: values.tobytes(order="C"),
    )
    monkeypatch.setitem(sys.modules, "libgguf", SimpleNamespace(open_gguf=lambda path: fake_file))

    module = runtime.RuntimeModule.__new__(runtime.RuntimeModule)
    module.artifact_dir = tmp_path
    module.manifest = {"files": {"encoded_constants": "encoded_constants.json"}}
    module.metadata = {"constants": [{"name": "weight", "shape": [3, 2], "shape_spec": [3, 2], "dtype": "float32"}]}
    module._constant_loaded = {"weight": False}
    module._handle = ctypes.c_void_p(1)
    write_json(
        tmp_path / "encoded_constants.json",
        {
            "schema_version": 1,
            "kind": "dinoml.encoded_constants",
            "constants": [
                {
                    "name": "weight",
                    "dtype": "float32",
                    "shape": [3, 2],
                    "logical_nbytes": values.nbytes,
                    "storage": {
                        "kind": "gguf",
                        "path": "weights.gguf",
                        "tensor": "blk.0.ffn.weight",
                        "logical_dtype": "float32",
                        "shape": [3, 2],
                        "qtype": "F32",
                        "encoded_nbytes": values.nbytes,
                        "n_per_row": 2,
                    },
                }
            ],
            "summary": {"constant_count": 1, "logical_nbytes": values.nbytes, "encoded_nbytes": values.nbytes},
        },
    )
    captured = {}

    def set_constant_numpy(name, value):
        captured[name] = np.array(value, copy=True)

    module.set_constant_numpy = set_constant_numpy

    module.load_encoded_constants()

    np.testing.assert_array_equal(captured["weight"], values)


def test_runtime_load_encoded_constants_resolves_artifact_relative_gguf_path(monkeypatch, tmp_path):
    values = np.arange(4, dtype=np.float32).reshape(2, 2)
    opened_paths = []
    tensor_info = SimpleNamespace(qtype="F32", qtype_value=0, shape=(2, 2), data_offset=64)

    fake_file = SimpleNamespace(
        get_tensor=lambda name: tensor_info,
        read_tensor_bytes=lambda tensor: values.tobytes(order="C"),
    )

    def open_gguf(path):
        opened_paths.append(path)
        return fake_file

    monkeypatch.setitem(sys.modules, "libgguf", SimpleNamespace(open_gguf=open_gguf))

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    module = runtime.RuntimeModule.__new__(runtime.RuntimeModule)
    module.artifact_dir = artifact_dir
    module.manifest = {"files": {"encoded_constants": "encoded_constants.json"}}
    module.metadata = {"constants": [{"name": "weight", "shape": [2, 2], "shape_spec": [2, 2], "dtype": "float32"}]}
    module._constant_loaded = {"weight": False}
    module._handle = ctypes.c_void_p(1)
    write_json(
        artifact_dir / "encoded_constants.json",
        {
            "schema_version": 1,
            "kind": "dinoml.encoded_constants",
            "constants": [
                {
                    "name": "weight",
                    "dtype": "float32",
                    "shape": [2, 2],
                    "logical_nbytes": values.nbytes,
                    "storage": {
                        "kind": "gguf",
                        "path": "weights.gguf",
                        "tensor": "blk.0.ffn.weight",
                        "logical_dtype": "float32",
                        "shape": [2, 2],
                        "qtype": "F32",
                        "encoded_nbytes": values.nbytes,
                        "materialization": "dequantize_full_before_launch",
                        "residency": "manual_runtime_load",
                    },
                }
            ],
        },
    )
    captured = {}

    def set_constant_numpy(name, value):
        captured[name] = np.array(value, copy=True)
        module._mark_constant_loaded(name, True)

    module.set_constant_numpy = set_constant_numpy

    module.load_encoded_constants()

    assert opened_paths == [str((artifact_dir / "weights.gguf").resolve())]
    np.testing.assert_array_equal(captured["weight"], values)
    assert module.constant_load_state() == {"weight": True}


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        ("float16", 2e-3, 2e-3),
        ("bfloat16", 2e-2, 2e-2),
    ],
)
def test_cpu_generated_fused_elementwise_supports_reduced_precision(tmp_path, dtype, atol, rtol):
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
        name=f"cpu_fused_elementwise_{dtype}",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"cpu_fused_elementwise_{dtype}.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert ("dinoml::math::float16" if dtype == "float16" else "dinoml::math::bfloat16") in generated
    assert "using fused_elementwise_" in generated
    assert "compute_t = float" in generated

    x = np.random.default_rng(13).standard_normal((2, 3, 4)).astype(np.float32)
    expected = execute_cpu(spec.bind_constants(runtime_constants), {"x": x})["y"]

    module = runtime.load(artifact.path)
    session = module.create_session()
    module.set_constant_numpy("scale", runtime_constants["scale"])
    module.set_constant_numpy("bias", runtime_constants["bias"])
    try:
        actual = session.run_numpy({"x": x})["y"]
    finally:
        session.close()
        module.close()

    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=atol, rtol=rtol)


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


def test_cpu_runtime_materializes_output_view_of_input_on_repeated_runs(tmp_path):
    spec = ModelSpec("shape_view_input_alias", _shape_view_ir(), constants={})
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "shape_view_input_alias.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "const float* ptr_y = ptr_x;" in generated
    assert "std::memcpy(dinoml::module::tensor_data(outputs[0]), ptr_y, runtime_numel_y * sizeof(float));" in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    x0 = np.arange(6, dtype=np.float32).reshape(2, 3)
    x1 = (np.arange(6, dtype=np.float32) + 100.0).reshape(2, 3)
    try:
        first = session.run_numpy({"x": x0})["y"]
        second = session.run_numpy({"x": x1})["y"]
    finally:
        session.close()
        module.close()

    np.testing.assert_array_equal(first, x0.reshape(3, 2))
    np.testing.assert_array_equal(second, x1.reshape(3, 2))


def test_cpu_runtime_materializes_internal_shape_buffer_count_fixture(tmp_path):
    spec = ModelSpec("shape_buffer_count_true_fixture", _shape_buffer_count_true_ir(), constants={})
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "shape_buffer_count_true_fixture.dinoml")
    metadata = read_json(artifact.path / "metadata.json")
    assert metadata["output_shape_reports"] == {
        "version": 1,
        "reports": [{"output": "y", "kind": "shape_buffer"}],
    }

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        mixed = session.run_numpy({"x": np.array([[True, False, True], [False, True, False]], dtype=np.bool_)})["y"]
        assert mixed.shape == (3,)
        assert session.get_output_shape("y") == (3,)
        np.testing.assert_array_equal(mixed, np.ones((3,), dtype=np.bool_))

        empty = session.run_numpy({"x": np.zeros((2, 3), dtype=np.bool_)})["y"]
        assert empty.shape == (0,)
        assert session.get_output_shape("y") == (0,)
        np.testing.assert_array_equal(empty, np.empty((0,), dtype=np.bool_))

        full = session.run_numpy({"x": np.ones((2, 3), dtype=np.bool_)})["y"]
        assert full.shape == (6,)
        assert session.get_output_shape("y") == (6,)
        np.testing.assert_array_equal(full, np.ones((6,), dtype=np.bool_))
    finally:
        session.close()
        module.close()


def test_cpu_runtime_materializes_public_shape_view_ops_on_repeated_runs(tmp_path):
    spec = dml.trace(
        PublicShapeViewOutputs(),
        inputs={"x": dml.TensorSpec([2, 3]), "z": dml.TensorSpec([1, 2, 1, 3])},
        name="public_shape_views",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "public_shape_views.dinoml")
    metadata = read_json(artifact.path / "metadata.json")
    graph = read_json(artifact.path / "graph.dinoir.json")
    assert metadata["memory_plan"]["views"]["views"]
    assert graph["nodes"] == []

    module = runtime.load(artifact.path)
    session = module.create_session()
    x0 = np.arange(6, dtype=np.float32).reshape(2, 3)
    x1 = (np.arange(6, dtype=np.float32) + 50.0).reshape(2, 3)
    z0 = (np.arange(6, dtype=np.float32) + 100.0).reshape(1, 2, 1, 3)
    z1 = (np.arange(6, dtype=np.float32) + 200.0).reshape(1, 2, 1, 3)
    try:
        first = session.run_numpy({"x": x0, "z": z0})
        second = session.run_numpy({"x": x1, "z": z1})
    finally:
        session.close()
        module.close()

    np.testing.assert_array_equal(first["id"], x0)
    np.testing.assert_array_equal(first["reshaped"], x0.reshape(3, 2))
    np.testing.assert_array_equal(first["flat"], x0.reshape(6))
    np.testing.assert_array_equal(first["squeezed"], z0.reshape(2, 3))
    np.testing.assert_array_equal(first["unsqueezed"], x0.reshape(1, 2, 3))
    np.testing.assert_array_equal(second["id"], x1)
    np.testing.assert_array_equal(second["reshaped"], x1.reshape(3, 2))
    np.testing.assert_array_equal(second["flat"], x1.reshape(6))
    np.testing.assert_array_equal(second["squeezed"], z1.reshape(2, 3))
    np.testing.assert_array_equal(second["unsqueezed"], x1.reshape(1, 2, 3))


def test_cpu_runtime_materializes_output_view_of_constant(tmp_path):
    constant = np.arange(6, dtype=np.float32).reshape(2, 3) + 7.0
    spec = ModelSpec("shape_view_constant_alias", _shape_view_ir(source_kind="constant"), constants={"c": constant})
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "shape_view_constant_alias.dinoml")

    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy({"x": np.zeros((1,), dtype=np.float32)})["y"]
    finally:
        session.close()
        module.close()

    np.testing.assert_array_equal(actual, constant.reshape(3, 2))


def test_cpu_runtime_materializes_output_view_of_temporary(tmp_path):
    spec = ModelSpec("shape_view_temporary_alias", _shape_view_ir(source_kind="temporary"), constants={})
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "shape_view_temporary_alias.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "std::vector<float> tmp_t0;" in generated
    assert "const float* ptr_y = ptr_t0;" in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    x = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    try:
        actual = session.run_numpy({"x": x})["y"]
    finally:
        session.close()
        module.close()

    np.testing.assert_array_equal(actual, np.maximum(x, 0.0).reshape(3, 2))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cpu_runtime_materializes_reduced_precision_output_view_of_temporary(tmp_path, dtype):
    spec = ModelSpec(f"shape_view_temporary_alias_{dtype}", _shape_view_ir(source_kind="temporary", dtype=dtype), constants={})
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"shape_view_temporary_alias_{dtype}.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    expected_type = "dinoml::math::float16" if dtype == "float16" else "dinoml::math::bfloat16"
    assert f"std::vector<{expected_type}> tmp_t0;" in generated
    assert f"const {expected_type}* ptr_y = ptr_t0;" in generated
    assert f"runtime_numel_y * sizeof({expected_type})" in generated

    module = runtime.load(artifact.path)
    session = module.create_session()
    x = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    expected = _storage_roundtrip(np.maximum(_storage_roundtrip(x, dtype).astype(np.float32), 0.0), dtype).reshape(3, 2)
    try:
        actual = session.run_numpy({"x": x})["y"]
    finally:
        session.close()
        module.close()

    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=0, rtol=0)


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

    x = np.zeros((2, 8, 4), dtype=np.float32)
    y = np.empty_like(x)
    input_tensor, input_keepalive = runtime._make_dino_tensor(
        ctypes.c_void_p(x.ctypes.data),
        x.shape,
        dtype_runtime_enum("float32"),
        nbytes=x.nbytes,
        device_type=runtime.DINO_DEVICE_CPU,
    )
    output_tensor, output_keepalive = runtime._make_dino_tensor(
        ctypes.c_void_p(y.ctypes.data),
        y.shape,
        dtype_runtime_enum("float32"),
        nbytes=y.nbytes,
        device_type=runtime.DINO_DEVICE_CPU,
    )
    inputs = (runtime._DinoTensor * 1)(input_tensor)
    outputs = (runtime._DinoTensor * 1)(output_tensor)
    with pytest.raises(RuntimeError, match="wrong input/output count"):
        module._check(
            module._dll.dino_session_run(
                session._handle,
                inputs,
                ctypes.c_size_t(1),
                outputs,
                ctypes.c_size_t(0),
            )
        )
    assert input_keepalive and output_keepalive
    with pytest.raises(RuntimeError, match="Output shape is unavailable before dino_session_run"):
        session.get_output_shape("y")

    bad = np.zeros((2, 10, 4), dtype=np.float32)
    try:
        session.run_numpy({"x": bad})
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("dynamic shape divisibility violation was not rejected")

    session.close()
    module.close()


def test_get_output_shape_rejects_negative_reported_dimensions():
    def fake_get_output_shape(_handle, _index, shape, ndim_ptr):
        if shape is None:
            ndim_ptr._obj.value = 2
        else:
            shape[0] = -1
            shape[1] = 4
        return 0

    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(123)
    session.module = SimpleNamespace(
        metadata={"outputs": [{"name": "y"}]},
        _dll=SimpleNamespace(dino_session_get_output_shape=fake_get_output_shape),
        _check=lambda code: code,
    )

    with pytest.raises(ValueError, match="negative dimension"):
        session.get_output_shape("y")


def test_get_output_shape_accepts_zero_reported_dimensions():
    def fake_get_output_shape(_handle, _index, shape, ndim_ptr):
        if shape is None:
            ndim_ptr._obj.value = 1
        else:
            shape[0] = 0
        return 0

    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(123)
    session.module = SimpleNamespace(
        metadata={"outputs": [{"name": "y"}]},
        _dll=SimpleNamespace(dino_session_get_output_shape=fake_get_output_shape),
        _check=lambda code: code,
    )

    assert session.get_output_shape("y") == (0,)


def test_get_output_shape_rejects_rank_growth_between_abi_calls():
    def fake_get_output_shape(_handle, _index, shape, ndim_ptr):
        if shape is None:
            ndim_ptr._obj.value = 1
        else:
            shape[0] = 4
            ndim_ptr._obj.value = 2
        return 0

    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(123)
    session.module = SimpleNamespace(
        metadata={"outputs": [{"name": "y"}]},
        _dll=SimpleNamespace(dino_session_get_output_shape=fake_get_output_shape),
        _check=lambda code: code,
    )

    with pytest.raises(RuntimeError, match="exceeds shape buffer capacity"):
        session.get_output_shape("y")


def test_cpu_runtime_materializes_reported_smaller_output_shape(tmp_path, monkeypatch):
    spec = dml.trace(DirectIdentityModel(), inputs={"x": dml.TensorSpec([2, 4], "float32")}, name="materialize_output_shape_cpu")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "materialize_output_shape_cpu.dinoml")
    module = runtime.load(artifact.path)
    session = module.create_session()
    try:
        monkeypatch.setattr(session, "get_output_shape", lambda _name: (1, 4))
        x = np.arange(8, dtype=np.float32).reshape(2, 4)
        actual = session.run_numpy({"x": x})["y"]
        assert actual.shape == (1, 4)
        np.testing.assert_allclose(actual, x[:1].astype(np.float32), atol=0, rtol=0)

        monkeypatch.setattr(session, "get_output_shape", lambda _name: (4, 2))
        actual = session.run_numpy({"x": x})["y"]
        assert actual.shape == (4, 2)
        np.testing.assert_allclose(actual, x.astype(np.float32).reshape(4, 2), atol=0, rtol=0)

        monkeypatch.setattr(session, "get_output_shape", lambda _name: (0,))
        actual = session.run_numpy({"x": x})["y"]
        assert actual.shape == (0,)
        assert actual.size == 0

        monkeypatch.setattr(session, "get_output_shape", lambda _name: (3, 4))
        with pytest.raises(ValueError, match="has more elements than allocated"):
            session.run_numpy({"x": x})

        monkeypatch.setattr(session, "get_output_shape", lambda _name: (-1, 4))
        with pytest.raises(ValueError, match="negative dimension"):
            session.run_numpy({"x": x})
    finally:
        session.close()
        module.close()


def test_device_pointer_run_rejects_reported_shape_larger_than_bound_output(monkeypatch):
    calls = []

    def fake_run(*_args):
        calls.append("run")
        return 0

    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(123)
    session.module = SimpleNamespace(
        target_name="cuda",
        metadata={
            "inputs": [
                {
                    "name": "x",
                    "shape": [2, 4],
                    "shape_spec": [2, 4],
                    "dtype": "float32",
                }
            ],
            "outputs": [
                {
                    "name": "y",
                    "shape": [4, 4],
                    "shape_spec": [{"kind": "dim", "name": "rows", "min": 1, "max": 4}, 4],
                    "dtype": "float32",
                }
            ],
        },
        _dll=SimpleNamespace(dino_session_run=fake_run),
        _check=lambda code: code,
    )

    monkeypatch.setattr(session, "get_output_shape", lambda _name: (1, 4))
    session.run_device_pointers(
        {"x": 0x1000},
        {"y": 0x2000},
        {"x": (2, 4)},
        {"y": (2, 4)},
    )
    assert calls == ["run"]

    monkeypatch.setattr(session, "get_output_shape", lambda _name: (4, 2))
    session.run_device_pointers(
        {"x": 0x1000},
        {"y": 0x2000},
        {"x": (2, 4)},
        {"y": (2, 4)},
    )
    assert calls == ["run", "run"]

    monkeypatch.setattr(session, "get_output_shape", lambda _name: (0,))
    session.run_device_pointers(
        {"x": 0x1000},
        {"y": 0x2000},
        {"x": (2, 4)},
        {"y": (2, 4)},
    )
    assert calls == ["run", "run", "run"]

    monkeypatch.setattr(session, "get_output_shape", lambda _name: (3, 4))
    with pytest.raises(ValueError, match="has more elements than allocated"):
        session.run_device_pointers(
            {"x": 0x1000},
            {"y": 0x2000},
            {"x": (2, 4)},
            {"y": (2, 4)},
        )
    assert calls == ["run", "run", "run", "run"]

    monkeypatch.setattr(session, "get_output_shape", lambda _name: (-1, 4))
    with pytest.raises(ValueError, match="negative dimension"):
        session.run_device_pointers(
            {"x": 0x1000},
            {"y": 0x2000},
            {"x": (2, 4)},
            {"y": (2, 4)},
        )
    assert calls == ["run", "run", "run", "run", "run"]


@pytest.mark.parametrize(
    ("inputs", "outputs", "input_shapes", "output_shapes", "expected_message"),
    [
        ({"x": 0x1000, "z": 0x3000}, {"y": 0x2000}, {"x": (2, 4)}, {"y": (2, 4)}, "Unexpected input pointer: z"),
        ({"x": 0x1000}, {"y": 0x2000, "z": 0x3000}, {"x": (2, 4)}, {"y": (2, 4)}, "Unexpected output pointer: z"),
        ({"x": 0x1000}, {"y": 0x2000}, {"x": (2, 4), "z": (2, 4)}, {"y": (2, 4)}, "Unexpected input shape: z"),
        ({"x": 0x1000}, {"y": 0x2000}, {"x": (2, 4)}, {"y": (2, 4), "z": (2, 4)}, "Unexpected output shape: z"),
    ],
)
def test_device_pointer_run_rejects_unexpected_names(inputs, outputs, input_shapes, output_shapes, expected_message):
    calls = []

    def fake_run(*_args):
        calls.append("run")
        return 0

    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(123)
    session.module = SimpleNamespace(
        target_name="cuda",
        metadata={
            "inputs": [
                {
                    "name": "x",
                    "shape": [2, 4],
                    "shape_spec": [2, 4],
                    "dtype": "float32",
                }
            ],
            "outputs": [
                {
                    "name": "y",
                    "shape": [2, 4],
                    "shape_spec": [2, 4],
                    "dtype": "float32",
                }
            ],
        },
        _dll=SimpleNamespace(dino_session_run=fake_run),
        _check=lambda code: code,
    )

    with pytest.raises(ValueError, match=expected_message):
        session.run_device_pointers(inputs, outputs, input_shapes, output_shapes)

    assert calls == []


@pytest.mark.parametrize(
    ("inputs", "outputs", "input_shapes", "output_shapes", "expected_message"),
    [
        ([("x", 0x1000)], {"y": 0x2000}, {"x": (2, 4)}, {"y": (2, 4)}, "device input pointers must be a mapping"),
        ({"x": 0x1000}, [("y", 0x2000)], {"x": (2, 4)}, {"y": (2, 4)}, "device output pointers must be a mapping"),
        ({"x": 0x1000}, {"y": 0x2000}, [("x", (2, 4))], {"y": (2, 4)}, "device input shapes must be a mapping"),
        ({"x": 0x1000}, {"y": 0x2000}, {"x": (2, 4)}, [("y", (2, 4))], "device output shapes must be a mapping"),
    ],
)
def test_device_pointer_run_rejects_non_mapping_bindings(inputs, outputs, input_shapes, output_shapes, expected_message):
    calls = []

    def fake_run(*_args):
        calls.append("run")
        return 0

    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(123)
    session.module = SimpleNamespace(
        target_name="cuda",
        metadata={
            "inputs": [{"name": "x", "shape": [2, 4], "shape_spec": [2, 4], "dtype": "float32"}],
            "outputs": [{"name": "y", "shape": [2, 4], "shape_spec": [2, 4], "dtype": "float32"}],
        },
        _dll=SimpleNamespace(dino_session_run=fake_run),
        _check=lambda code: code,
    )

    with pytest.raises(TypeError, match=expected_message):
        session.run_device_pointers(inputs, outputs, input_shapes, output_shapes)

    assert calls == []


def test_run_torch_rejects_unexpected_input_names_before_tensor_checks():
    pytest.importorskip("torch")
    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(123)
    session.module = SimpleNamespace(
        target_name="cuda",
        metadata={
            "inputs": [
                {
                    "name": "x",
                    "shape": [2, 4],
                    "shape_spec": [2, 4],
                    "dtype": "float32",
                }
            ],
            "outputs": [
                {
                    "name": "y",
                    "shape": [2, 4],
                    "shape_spec": [2, 4],
                    "dtype": "float32",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="Unexpected input: z"):
        session.run_torch({"x": object(), "z": object()})


def test_run_torch_rejects_non_mapping_inputs_before_tensor_checks():
    pytest.importorskip("torch")
    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(123)
    session.module = SimpleNamespace(
        target_name="cuda",
        metadata={
            "inputs": [{"name": "x", "shape": [2, 4], "shape_spec": [2, 4], "dtype": "float32"}],
            "outputs": [{"name": "y", "shape": [2, 4], "shape_spec": [2, 4], "dtype": "float32"}],
        },
    )

    with pytest.raises(TypeError, match="run_torch inputs must be a mapping"):
        session.run_torch([("x", object())])


def test_run_torch_rejects_zero_input_artifacts_before_output_allocation():
    pytest.importorskip("torch")
    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(123)
    session.module = SimpleNamespace(
        target_name="cuda",
        metadata={
            "inputs": [],
            "outputs": [{"name": "y", "shape": [2, 4], "shape_spec": [2, 4], "dtype": "float32"}],
        },
    )

    with pytest.raises(ValueError, match="run_torch requires at least one CUDA input tensor to infer the output device"):
        session.run_torch({})


def test_run_torch_rejects_mixed_cuda_devices_before_output_allocation():
    torch = pytest.importorskip("torch")
    session = object.__new__(runtime.Session)
    session._handle = ctypes.c_void_p(123)
    session.module = SimpleNamespace(
        target_name="cuda",
        metadata={
            "inputs": [
                {"name": "x", "shape": [2, 4], "shape_spec": [2, 4], "dtype": "float32"},
                {"name": "z", "shape": [2, 4], "shape_spec": [2, 4], "dtype": "float32"},
            ],
            "outputs": [{"name": "y", "shape": [2, 4], "shape_spec": [2, 4], "dtype": "float32"}],
        },
    )

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.float32
        shape = (2, 4)

        def __init__(self, device):
            self.device = device

        def is_contiguous(self):
            return True

    with pytest.raises(
        ValueError,
        match=r"All run_torch inputs must be on the same CUDA device; x is on cuda:0, z is on cuda:1",
    ):
        session.run_torch({"x": FakeCudaTensor("cuda:0"), "z": FakeCudaTensor("cuda:1")})


def test_cpu_runtime_set_constant_accepts_dynamic_shape(tmp_path):
    batch = dml.Dim("batch", min=1, max=4)
    spec = dml.trace(
        DynamicConstantBias(batch),
        inputs={"x": dml.TensorSpec([batch, 4], "float32")},
        constants={"bias": np.zeros((4, 1), dtype=np.float32)},
        name="dynamic_constant_bias",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "dynamic_constant_bias_cpu.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
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

    np.testing.assert_allclose(actual, x + bias, atol=1e-6, rtol=0)


def _shape_view_ir(source_kind: str = "input", dtype: str = "float32"):
    output_shape = [3, 2]
    dtype_nbytes = 2 if dtype in {"float16", "bfloat16"} else 4
    output_nbytes = 6 * dtype_nbytes
    tensors = [
        {
            "name": "x",
            "shape": [2, 3] if source_kind != "constant" else [1],
            "shape_spec": [2, 3] if source_kind != "constant" else [1],
            "dtype": dtype,
            "kind": "input",
            "nbytes": 6 * dtype_nbytes if source_kind != "constant" else dtype_nbytes,
        },
        {
            "name": "y",
            "shape": output_shape,
            "shape_spec": output_shape,
            "dtype": dtype,
            "kind": "output",
            "nbytes": output_nbytes,
        },
    ]
    constants = []
    nodes = []
    source = "x"
    if source_kind == "constant":
        tensors.append({"name": "c", "shape": [2, 3], "shape_spec": [2, 3], "dtype": dtype, "kind": "constant", "nbytes": output_nbytes})
        constants = [{"name": "c", "tensor": "c", "shape": [2, 3], "shape_spec": [2, 3], "dtype": dtype}]
        source = "c"
    elif source_kind == "temporary":
        tensors.append({"name": "t0", "shape": [2, 3], "shape_spec": [2, 3], "dtype": dtype, "kind": "intermediate", "nbytes": output_nbytes})
        nodes = [
            {
                "id": "relu_to_t0",
                "op": "fused_elementwise",
                "inputs": ["x"],
                "outputs": ["t0"],
                "attrs": {"sub_ops": [{"op": "relu", "inputs": ["x"], "outputs": ["t0"], "attrs": {}}]},
            }
        ]
        source = "t0"
    elif source_kind != "input":
        raise ValueError(f"Unsupported source_kind: {source_kind}")
    return {
        "schema_version": IR_SCHEMA_VERSION,
        "name": f"shape_view_{source_kind}_alias",
        "inputs": [{"name": "x", "tensor": "x", "shape": tensors[0]["shape"], "shape_spec": tensors[0]["shape_spec"], "dtype": dtype}],
        "constants": constants,
        "outputs": [{"name": "y", "tensor": "y", "shape": output_shape, "shape_spec": output_shape, "dtype": dtype}],
        "nodes": nodes,
        "tensors": tensors,
        "metadata": {
            "views": {
                "version": 1,
                "views": [
                    {
                        "tensor": "y",
                        "source": source,
                        "kind": "shape_view",
                        "transform": "reshape",
                        "shape": output_shape,
                        "shape_spec": output_shape,
                    }
                ],
            }
        },
    }


def _shape_buffer_count_true_ir():
    return {
        "schema_version": IR_SCHEMA_VERSION,
        "name": "shape_buffer_count_true_fixture",
        "inputs": [{"name": "x", "tensor": "x", "shape": [2, 3], "shape_spec": [2, 3], "dtype": "bool"}],
        "constants": [],
        "outputs": [{"name": "y", "tensor": "y", "shape": [6], "shape_spec": [6], "dtype": "bool"}],
        "nodes": [{"id": "n0", "op": "_shape_buffer_count_true", "inputs": ["x"], "outputs": ["y"], "attrs": {}}],
        "tensors": [
            {"name": "x", "shape": [2, 3], "shape_spec": [2, 3], "dtype": "bool", "kind": "input", "nbytes": 6},
            {"name": "y", "shape": [6], "shape_spec": [6], "dtype": "bool", "kind": "output", "nbytes": 6},
        ],
        "metadata": {
            "output_shape_reports": {
                "version": 1,
                "reports": [{"output": "y", "kind": "shape_buffer"}],
            }
        },
    }


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


def test_cpu_reference_softmax_matches_stable_numpy():
    spec = dml.trace(
        SoftmaxLastDim(),
        inputs={"x": dml.TensorSpec([4, 8], "float32")},
        name="softmax_reference",
    )
    x = np.array(
        [
            [-1000.0, -999.0, -998.0, -997.0, 0.0, 1.0, 2.0, 3.0],
            [20.0, 0.0, -20.0, -40.0, 5.0, 6.0, 7.0, 8.0],
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
            [3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
        ],
        dtype=np.float32,
    )
    shifted = x - np.max(x, axis=-1, keepdims=True)
    expected = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)

    actual = execute_cpu(spec, {"x": x})["y"]

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_cpu_artifact_runs_generated_softmax_for_attention_rows(tmp_path):
    spec = dml.trace(
        SoftmaxLastDim(),
        inputs={"x": dml.TensorSpec([256, 1024], "float32")},
        name="attention_row_softmax",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "attention_row_softmax_cpu.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "softmax_" in generated
    assert "row_max" in generated
    assert "std::exp" in generated

    rng = np.random.default_rng(123)
    x = rng.standard_normal((256, 1024)).astype(np.float32) * 3.0
    shifted = x - np.max(x, axis=-1, keepdims=True)
    expected = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({"x": x})["y"]
    session.close()
    module.close()

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        ("float16", 2e-3, 2e-3),
        ("bfloat16", 2e-2, 2e-2),
    ],
)
def test_cpu_artifact_runs_generated_reduced_precision_softmax(tmp_path, dtype, atol, rtol):
    spec = dml.trace(
        SoftmaxLastDim(),
        inputs={"x": dml.TensorSpec([8, 33], dtype)},
        name=f"softmax_{dtype}_cpu",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"softmax_{dtype}_cpu.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    storage_type = "dinoml::math::float16" if dtype == "float16" else "dinoml::math::bfloat16"
    assert f"const {storage_type}* DINO_RESTRICT x" in generated
    assert "std::vector<float> row_values" in generated
    assert "dinoml::math::cast<float>(x[base + col])" in generated
    assert f"dinoml::math::cast<{storage_type}>" in generated

    x = np.random.default_rng(1234).standard_normal((8, 33)).astype(np.float32) * 2.0
    x_reference = array_from_storage(array_to_storage(x, dtype), dtype).astype(np.float32)
    shifted = x_reference - np.max(x_reference, axis=-1, keepdims=True)
    expected_float = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    expected = array_from_storage(array_to_storage(expected_float, dtype), dtype)

    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({"x": x})["y"]
    session.close()
    module.close()

    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("op_name", "numpy_op"),
    [
        ("reduce_sum", np.sum),
        ("reduce_max", np.max),
        ("reduce_min", np.min),
        ("reduce_mean", np.mean),
        ("var", np.var),
        ("vector_norm", lambda value, axis: np.sqrt(np.sum(value * value, axis=axis))),
    ],
)
def test_cpu_reference_reductions_match_numpy(op_name, numpy_op):
    spec = dml.trace(
        ReductionLastDim(op_name),
        inputs={"x": dml.TensorSpec([3, 5, 7], "float32")},
        name=f"{op_name}_reference",
    )
    x = np.random.default_rng(41).standard_normal((3, 5, 7)).astype(np.float32)
    expected = numpy_op(x, axis=-1).astype(np.float32)
    actual = execute_cpu(spec, {"x": x})["y"]
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("op_name", "a_shape", "b_shape", "dtype", "atol", "rtol"),
    [
        ("gemm_rrr", (4, 8), (8, 6), "float32", 1e-5, 1e-5),
        ("gemm_rcr", (4, 8), (6, 8), "float32", 1e-5, 1e-5),
        ("gemm_rrr", (4, 8), (8, 6), "float16", 2e-3, 2e-3),
        ("gemm_rcr", (4, 8), (6, 8), "float16", 2e-3, 2e-3),
        ("gemm_rrr", (4, 8), (8, 6), "bfloat16", 2e-2, 2e-2),
        ("gemm_rcr", (4, 8), (6, 8), "bfloat16", 2e-2, 2e-2),
    ],
)
def test_cpu_reference_gemm_matches_numpy(op_name, a_shape, b_shape, dtype, atol, rtol):
    spec = dml.trace(
        GemmModule(op_name),
        inputs={"a": dml.TensorSpec(a_shape, dtype), "b": dml.TensorSpec(b_shape, dtype)},
        name=f"{op_name}_{dtype}_reference",
    )
    rng = np.random.default_rng(991)
    a = rng.standard_normal(a_shape).astype(np.float32)
    b = rng.standard_normal(b_shape).astype(np.float32)
    a_reference = array_from_storage(array_to_storage(a, dtype), dtype).astype(np.float32)
    b_reference = array_from_storage(array_to_storage(b, dtype), dtype).astype(np.float32)
    expected = array_from_storage(array_to_storage(a_reference @ (b_reference if op_name == "gemm_rrr" else b_reference.T), dtype), dtype)
    actual = execute_cpu(spec, {"a": a, "b": b})["y"]
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


BMM_LAYOUT_CASES = tuple(
    (
        f"bmm_{layout}",
        (2, 4, 3) if layout[0] == "c" else (2, 3, 4),
        (2, 5, 4) if layout[1] == "c" else (2, 4, 5),
        layout,
    )
    for layout in ("ccc", "ccr", "crc", "crr", "rcc", "rcr", "rrc", "rrr")
)
BMM_ADD_LAYOUT_CASES = tuple((f"{op_name}_add", a_shape, b_shape, layout) for op_name, a_shape, b_shape, layout in BMM_LAYOUT_CASES)


@pytest.mark.parametrize(("op_name", "a_shape", "b_shape", "layout"), BMM_LAYOUT_CASES)
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [("float32", 1e-5, 1e-5), ("float16", 2e-3, 2e-3), ("bfloat16", 2e-2, 2e-2)],
)
def test_cpu_reference_bmm_base_layouts_match_numpy(op_name, a_shape, b_shape, layout, dtype, atol, rtol):
    spec = dml.trace(
        BmmModule(op_name),
        inputs={"a": dml.TensorSpec(a_shape, dtype), "b": dml.TensorSpec(b_shape, dtype)},
        name=f"{op_name}_{dtype}_reference",
    )
    rng = np.random.default_rng(994)
    a = rng.standard_normal(a_shape).astype(np.float32)
    b = rng.standard_normal(b_shape).astype(np.float32)
    a_reference = array_from_storage(array_to_storage(a, dtype), dtype).astype(np.float32)
    b_reference = array_from_storage(array_to_storage(b, dtype), dtype).astype(np.float32)
    logical_a = np.swapaxes(a_reference, -1, -2) if layout[0] == "c" else a_reference
    logical_b = np.swapaxes(b_reference, -1, -2) if layout[1] == "c" else b_reference
    result = np.matmul(logical_a, logical_b)
    if layout[2] == "c":
        result = np.swapaxes(result, -1, -2)
    expected = array_from_storage(array_to_storage(result, dtype), dtype)

    actual = execute_cpu(spec, {"a": a, "b": b})["y"]

    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize(("op_name", "a_shape", "b_shape", "layout"), BMM_ADD_LAYOUT_CASES)
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [("float32", 1e-5, 1e-5), ("float16", 2e-3, 2e-3), ("bfloat16", 2e-2, 2e-2)],
)
def test_cpu_reference_bmm_add_layouts_match_numpy(op_name, a_shape, b_shape, layout, dtype, atol, rtol):
    out_shape = (2, 5, 3) if layout[2] == "c" else (2, 3, 5)
    spec = dml.trace(
        BmmModule(op_name),
        inputs={
            "a": dml.TensorSpec(a_shape, dtype),
            "b": dml.TensorSpec(b_shape, dtype),
            "d0": dml.TensorSpec(out_shape, dtype),
        },
        name=f"{op_name}_{dtype}_reference",
    )
    rng = np.random.default_rng(996)
    a = rng.standard_normal(a_shape).astype(np.float32)
    b = rng.standard_normal(b_shape).astype(np.float32)
    d0 = rng.standard_normal(out_shape).astype(np.float32)
    a_reference = array_from_storage(array_to_storage(a, dtype), dtype).astype(np.float32)
    b_reference = array_from_storage(array_to_storage(b, dtype), dtype).astype(np.float32)
    d0_reference = array_from_storage(array_to_storage(d0, dtype), dtype).astype(np.float32)
    logical_a = np.swapaxes(a_reference, -1, -2) if layout[0] == "c" else a_reference
    logical_b = np.swapaxes(b_reference, -1, -2) if layout[1] == "c" else b_reference
    result = np.matmul(logical_a, logical_b)
    if layout[2] == "c":
        result = np.swapaxes(result, -1, -2)
    expected = array_from_storage(array_to_storage(result + d0_reference, dtype), dtype)

    actual = execute_cpu(spec, {"a": a, "b": b, "d0": d0})["y"]

    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def test_cpu_reference_bmm_add_bias_broadcast_matches_numpy():
    spec = dml.trace(
        BmmModule("bmm_rrr_add"),
        inputs={
            "a": dml.TensorSpec([2, 3, 4], "float32"),
            "b": dml.TensorSpec([2, 4, 5], "float32"),
            "d0": dml.TensorSpec([5], "float32"),
        },
        name="bmm_rrr_add_bias_reference",
    )
    rng = np.random.default_rng(997)
    a = rng.standard_normal((2, 3, 4)).astype(np.float32)
    b = rng.standard_normal((2, 4, 5)).astype(np.float32)
    d0 = rng.standard_normal((5,)).astype(np.float32)
    expected = (np.matmul(a, b) + d0).astype(np.float32)

    actual = execute_cpu(spec, {"a": a, "b": b, "d0": d0})["y"]

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("module", "inputs", "arrays", "expected"),
    [
        (
            BmmHelperModule("bmm"),
            {"a": dml.TensorSpec([2, 3, 4], "float32"), "b": dml.TensorSpec([2, 4, 5], "float32")},
            ("a", "b"),
            "rrr",
        ),
        (
            BmmHelperModule("bmm_xxx", layout="rcr"),
            {"a": dml.TensorSpec([2, 3, 4], "float32"), "b": dml.TensorSpec([2, 5, 4], "float32")},
            ("a", "b"),
            "rcr",
        ),
        (
            BmmHelperModule("bmm_xxx_add", layout="rrc"),
            {
                "a": dml.TensorSpec([2, 3, 4], "float32"),
                "b": dml.TensorSpec([2, 4, 5], "float32"),
                "d0": dml.TensorSpec([2, 5, 3], "float32"),
            },
            ("a", "b", "d0"),
            "rrc_add",
        ),
    ],
)
def test_cpu_reference_bmm_direct_helpers_match_numpy(module, inputs, arrays, expected):
    spec = dml.trace(module, inputs=inputs, name=f"bmm_helper_{expected}_reference")
    rng = np.random.default_rng(998)
    values = {
        name: rng.standard_normal(tuple(tensor_spec.shape)).astype(np.float32)
        for name, tensor_spec in inputs.items()
        if name in arrays
    }
    a = values["a"]
    b = values["b"]
    logical_b = np.swapaxes(b, -1, -2) if expected.startswith("rcr") else b
    result = np.matmul(a, logical_b)
    if expected.startswith("rrc"):
        result = np.swapaxes(result, -1, -2)
    if expected.endswith("_add"):
        result = result + values["d0"]

    actual = execute_cpu(spec, values)["y"]

    np.testing.assert_allclose(actual, result.astype(np.float32), atol=1e-5, rtol=1e-5)


def test_cpu_reference_bmm_batch_broadcast_matches_numpy():
    spec = dml.trace(
        BmmModule("bmm_rcr"),
        inputs={"a": dml.TensorSpec([1, 3, 4], "float32"), "b": dml.TensorSpec([2, 5, 4], "float32")},
        name="bmm_rcr_broadcast_reference",
    )
    rng = np.random.default_rng(995)
    a = rng.standard_normal((1, 3, 4)).astype(np.float32)
    b = rng.standard_normal((2, 5, 4)).astype(np.float32)
    expected = np.matmul(a, np.swapaxes(b, -1, -2)).astype(np.float32)

    actual = execute_cpu(spec, {"a": a, "b": b})["y"]

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("op_name", "a_shape", "b_shape", "dtype", "atol", "rtol"),
    [
        ("gemm_rrr_bias", (4, 8), (8, 6), "float32", 1e-5, 1e-5),
        ("gemm_rcr_bias", (4, 8), (6, 8), "float32", 1e-5, 1e-5),
        ("gemm_rrr_bias_relu", (4, 8), (8, 6), "float32", 1e-5, 1e-5),
        ("gemm_rcr_bias_relu", (4, 8), (6, 8), "float32", 1e-5, 1e-5),
        ("gemm_rrr_bias", (4, 8), (8, 6), "float16", 2e-3, 2e-3),
        ("gemm_rcr_bias", (4, 8), (6, 8), "float16", 2e-3, 2e-3),
        ("gemm_rrr_bias_relu", (4, 8), (8, 6), "float16", 2e-3, 2e-3),
        ("gemm_rcr_bias_relu", (4, 8), (6, 8), "float16", 2e-3, 2e-3),
        ("gemm_rrr_bias", (4, 8), (8, 6), "bfloat16", 2e-2, 2e-2),
        ("gemm_rcr_bias", (4, 8), (6, 8), "bfloat16", 2e-2, 2e-2),
        ("gemm_rrr_bias_relu", (4, 8), (8, 6), "bfloat16", 2e-2, 2e-2),
        ("gemm_rcr_bias_relu", (4, 8), (6, 8), "bfloat16", 2e-2, 2e-2),
    ],
)
def test_cpu_reference_gemm_bias_matches_numpy(op_name, a_shape, b_shape, dtype, atol, rtol):
    spec = dml.trace(
        GemmBiasModule(op_name),
        inputs={"a": dml.TensorSpec(a_shape, dtype), "b": dml.TensorSpec(b_shape, dtype), "bias": dml.TensorSpec([6], dtype)},
        name=f"{op_name}_{dtype}_reference",
    )
    rng = np.random.default_rng(991)
    a = rng.standard_normal(a_shape).astype(np.float32)
    b = rng.standard_normal(b_shape).astype(np.float32)
    bias = rng.standard_normal((6,)).astype(np.float32)
    a_reference = array_from_storage(array_to_storage(a, dtype), dtype).astype(np.float32)
    b_reference = array_from_storage(array_to_storage(b, dtype), dtype).astype(np.float32)
    bias_reference = array_from_storage(array_to_storage(bias, dtype), dtype).astype(np.float32)
    matmul = a_reference @ (b_reference if op_name.startswith("gemm_rrr") else b_reference.T)
    result = matmul + bias_reference
    if op_name.endswith("_bias_relu"):
        result = np.maximum(result, 0.0)
    expected = array_from_storage(array_to_storage(result, dtype), dtype)
    actual = execute_cpu(spec, {"a": a, "b": b, "bias": bias})["y"]
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize(("op_name", "layout", "epilogue", "epilogue_inputs"), GEMM_BIAS_RESIDUAL_CASES)
def test_gemm_bias_residual_frontend_emits_shape_and_schema(op_name, layout, epilogue, epilogue_inputs):
    batch = dml.Dim("batch", min=1, max=4)
    tokens = dml.Dim("tokens", min=1, max=6)
    input_specs = {
        "a": dml.TensorSpec([batch, 8], "float32"),
        "b": dml.TensorSpec([tokens, 8] if layout == "rcr" else [8, tokens], "float32"),
        "bias": dml.TensorSpec([tokens], "float32"),
        "d0": dml.TensorSpec([batch, tokens], "float32"),
    }
    if "d1" in epilogue_inputs:
        input_specs["d1"] = dml.TensorSpec([batch, tokens], "float32")

    spec = dml.trace(GemmResidualModule(op_name), inputs=input_specs, name=f"{op_name}_frontend")
    node = spec.ir["nodes"][0]
    output = spec.ir["outputs"][0]

    assert get_op_def(op_name).schema.inputs == ("a", "b", *epilogue_inputs)
    assert node["op"] == op_name
    assert node["inputs"] == ["a", "b", *epilogue_inputs]
    assert output["shape"] == [4, 6]
    assert output["shape_spec"][0]["name"] == "batch"
    assert output["shape_spec"][1]["name"] == "tokens"
    assert get_op_def(op_name).backend_kernels["cuda"].resolve("float32").candidate_set["epilogue"] == epilogue


@pytest.mark.parametrize(("op_name", "layout", "_epilogue", "epilogue_inputs"), GEMM_BIAS_RESIDUAL_CASES)
def test_cpu_reference_gemm_bias_residual_epilogues_match_numpy(op_name, layout, _epilogue, epilogue_inputs):
    input_specs = {
        "a": dml.TensorSpec([4, 8], "float32"),
        "b": dml.TensorSpec([6, 8] if layout == "rcr" else [8, 6], "float32"),
        "bias": dml.TensorSpec([1, 6], "float32"),
        "d0": dml.TensorSpec([4, 6], "float32"),
    }
    if "d1" in epilogue_inputs:
        input_specs["d1"] = dml.TensorSpec([4, 6], "float32")
    spec = dml.trace(GemmResidualModule(op_name), inputs=input_specs, name=f"{op_name}_float32_reference")
    rng = np.random.default_rng(991)
    inputs = {
        "a": rng.standard_normal((4, 8)).astype(np.float32),
        "b": rng.standard_normal((6, 8) if layout == "rcr" else (8, 6)).astype(np.float32),
        "bias": rng.standard_normal((1, 6)).astype(np.float32),
        "d0": rng.standard_normal((4, 6)).astype(np.float32),
    }
    if "d1" in epilogue_inputs:
        inputs["d1"] = rng.standard_normal((4, 6)).astype(np.float32)
    result = inputs["a"] @ (inputs["b"].T if layout == "rcr" else inputs["b"]) + inputs["bias"]
    if epilogue_inputs == ("bias", "d0") and op_name.endswith("_bias_add"):
        result = result + inputs["d0"]
    elif epilogue_inputs == ("bias", "d0", "d1") and op_name.endswith("_bias_add_add"):
        result = result + inputs["d0"] + inputs["d1"]
    elif epilogue_inputs == ("bias", "d0") and op_name.endswith("_bias_add_relu"):
        result = np.maximum(result + inputs["d0"], 0.0)
    elif epilogue_inputs == ("bias", "d0", "d1") and op_name.endswith("_bias_add_add_relu"):
        result = np.maximum(result + inputs["d0"] + inputs["d1"], 0.0)
    elif epilogue_inputs == ("bias", "d0") and op_name.endswith("_bias_mul"):
        result = result * inputs["d0"]
    elif epilogue_inputs == ("bias", "d0", "d1") and op_name.endswith("_bias_mul_add"):
        result = result * inputs["d0"] + inputs["d1"]
    elif epilogue_inputs == ("bias", "d0") and op_name.endswith("_bias_mul_tanh"):
        result = np.tanh(result * inputs["d0"])
    elif epilogue_inputs == ("bias", "d0") and op_name.endswith("_bias_sigmoid_mul"):
        result = (1.0 / (1.0 + np.exp(-result))) * inputs["d0"]
    elif epilogue_inputs == ("bias", "d0") and op_name.endswith("_bias_sigmoid_mul_tanh"):
        result = np.tanh((1.0 / (1.0 + np.exp(-result))) * inputs["d0"])
    else:
        raise AssertionError(f"Unhandled residual GEMM op: {op_name}")
    expected = result.astype(np.float32)

    actual = execute_cpu(spec, inputs)["y"]

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("op_name", "layout"),
    [
        (f"gemm_{layout}_bias_{suffix}", layout)
        for layout in ("rcr", "rrr")
        for suffix in ("add", "add_relu", "mul", "mul_tanh", "sigmoid_mul", "sigmoid_mul_tanh")
    ],
)
def test_cpu_reference_gemm_single_residual_folded_m_matches_numpy(op_name, layout):
    spec = dml.trace(
        GemmResidualModule(op_name),
        inputs={
            "a": dml.TensorSpec([2, 3, 8], "float32"),
            "b": dml.TensorSpec([6, 8] if layout == "rcr" else [8, 6], "float32"),
            "bias": dml.TensorSpec([6], "float32"),
            "d0": dml.TensorSpec([2, 3, 6], "float32"),
        },
        name=f"{op_name}_folded_m_reference",
    )
    rng = np.random.default_rng(992)
    inputs = {
        "a": rng.standard_normal((2, 3, 8)).astype(np.float32),
        "b": rng.standard_normal((6, 8) if layout == "rcr" else (8, 6)).astype(np.float32),
        "bias": rng.standard_normal((6,)).astype(np.float32),
        "d0": rng.standard_normal((2, 3, 6)).astype(np.float32),
    }
    result = inputs["a"] @ (inputs["b"].T if layout == "rcr" else inputs["b"]) + inputs["bias"]
    if op_name.endswith("_bias_add"):
        result = result + inputs["d0"]
    elif op_name.endswith("_bias_add_relu"):
        result = np.maximum(result + inputs["d0"], 0.0)
    elif op_name.endswith("_bias_mul"):
        result = result * inputs["d0"]
    elif op_name.endswith("_bias_mul_tanh"):
        result = np.tanh(result * inputs["d0"])
    elif op_name.endswith("_bias_sigmoid_mul"):
        result = (1.0 / (1.0 + np.exp(-result))) * inputs["d0"]
    else:
        result = np.tanh((1.0 / (1.0 + np.exp(-result))) * inputs["d0"])

    actual = execute_cpu(spec, inputs)["y"]

    np.testing.assert_allclose(actual, result.astype(np.float32), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("op_name", "layout"),
    [
        (f"gemm_{layout}_bias_{suffix}", layout)
        for layout in ("rcr", "rrr")
        for suffix in ("add_add", "mul_add", "add_add_relu")
    ],
)
def test_cpu_reference_gemm_dual_residual_folded_m_matches_numpy(op_name, layout):
    spec = dml.trace(
        GemmResidualModule(op_name),
        inputs={
            "a": dml.TensorSpec([2, 3, 8], "float32"),
            "b": dml.TensorSpec([6, 8] if layout == "rcr" else [8, 6], "float32"),
            "bias": dml.TensorSpec([6], "float32"),
            "d0": dml.TensorSpec([2, 3, 6], "float32"),
            "d1": dml.TensorSpec([2, 3, 6], "float32"),
        },
        name=f"{op_name}_folded_m_reference",
    )
    rng = np.random.default_rng(993)
    inputs = {
        "a": rng.standard_normal((2, 3, 8)).astype(np.float32),
        "b": rng.standard_normal((6, 8) if layout == "rcr" else (8, 6)).astype(np.float32),
        "bias": rng.standard_normal((6,)).astype(np.float32),
        "d0": rng.standard_normal((2, 3, 6)).astype(np.float32),
        "d1": rng.standard_normal((2, 3, 6)).astype(np.float32),
    }
    result = inputs["a"] @ (inputs["b"].T if layout == "rcr" else inputs["b"]) + inputs["bias"]
    if op_name.endswith("_bias_add_add"):
        result = result + inputs["d0"] + inputs["d1"]
    elif op_name.endswith("_bias_mul_add"):
        result = result * inputs["d0"] + inputs["d1"]
    else:
        result = np.maximum(result + inputs["d0"] + inputs["d1"], 0.0)

    actual = execute_cpu(spec, inputs)["y"]

    np.testing.assert_allclose(actual, result.astype(np.float32), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("activation", ["gelu", "fast_gelu", "sigmoid", "tanh", "swish", "hardswish", "elup1"])
@pytest.mark.parametrize(("layout", "a_shape", "b_shape"), [("rrr", (4, 8), (8, 6)), ("rcr", (4, 8), (6, 8))])
def test_cpu_reference_gemm_bias_activation_matches_numpy(layout, a_shape, b_shape, activation):
    op_name = f"gemm_{layout}_bias_{activation}"
    spec = dml.trace(
        GemmBiasModule(op_name),
        inputs={
            "a": dml.TensorSpec(a_shape, "float32"),
            "b": dml.TensorSpec(b_shape, "float32"),
            "bias": dml.TensorSpec([6], "float32"),
        },
        name=f"{op_name}_float32_reference",
    )
    rng = np.random.default_rng(991)
    a = rng.standard_normal(a_shape).astype(np.float32)
    b = rng.standard_normal(b_shape).astype(np.float32)
    bias = rng.standard_normal((6,)).astype(np.float32)
    result = a @ (b if layout == "rrr" else b.T) + bias
    if activation == "gelu":
        result = 0.5 * result * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (result + 0.044715 * result * result * result)))
    elif activation == "fast_gelu":
        result = result / (1.0 + np.exp(-1.702 * result))
    elif activation == "sigmoid":
        result = 1.0 / (1.0 + np.exp(-result))
    elif activation == "tanh":
        result = np.tanh(result)
    elif activation == "swish":
        result = result / (1.0 + np.exp(-result))
    elif activation == "hardswish":
        result = result * np.clip(result + 3.0, 0.0, 6.0) / 6.0
    elif activation == "elup1":
        result = np.where(result >= 0.0, result + 1.0, np.exp(result))
    expected = result.astype(np.float32)

    actual = execute_cpu(spec, {"a": a, "b": b, "bias": bias})["y"]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_cpu_compile_rejects_cuda_only_gemm(tmp_path):
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="gemm_rrr_cpu_reject",
    )
    with pytest.raises(NotImplementedError, match="cpu backend does not support op gemm_rrr"):
        dml.compile(spec, dml.Target("cpu"), tmp_path / "gemm_rrr_cpu_reject.dinoml")


def test_cpu_artifact_runs_generated_reduction_keepdim(tmp_path):
    spec = dml.trace(
        ReductionLastDim("reduce_mean", keepdim=True),
        inputs={"x": dml.TensorSpec([4, 8, 16], "float32")},
        name="reduce_mean_keepdim_cpu",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "reduce_mean_keepdim_cpu.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "reduce_mean_" in generated
    assert "runtime_rows" in generated

    x = np.random.default_rng(42).standard_normal((4, 8, 16)).astype(np.float32)
    expected = np.mean(x, axis=-1, keepdims=True).astype(np.float32)
    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({"x": x})["y"]
    session.close()
    module.close()

    assert actual.shape == (4, 8, 1)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    ("op_name", "numpy_op", "atol", "rtol"),
    [
        ("reduce_sum", np.sum, 2e-2, 2e-2),
        ("reduce_max", np.max, 0.0, 0.0),
        ("reduce_min", np.min, 0.0, 0.0),
        ("reduce_mean", np.mean, 2e-3, 2e-2),
    ],
)
def test_cpu_artifact_runs_generated_reduced_precision_reductions(tmp_path, dtype, op_name, numpy_op, atol, rtol):
    spec = dml.trace(
        ReductionLastDim(op_name),
        inputs={"x": dml.TensorSpec([4, 8, 16], dtype)},
        name=f"{op_name}_{dtype}_cpu",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"{op_name}_{dtype}_cpu.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    storage_type = "dinoml::math::float16" if dtype == "float16" else "dinoml::math::bfloat16"
    assert f"const {storage_type}* DINO_RESTRICT x" in generated
    assert "float acc" in generated
    assert f"dinoml::math::cast<{storage_type}>" in generated

    x = (np.random.default_rng(44).standard_normal((4, 8, 16)).astype(np.float32) * 0.5)
    x_reference = array_from_storage(array_to_storage(x, dtype), dtype).astype(np.float32)
    expected = array_from_storage(array_to_storage(numpy_op(x_reference, axis=-1), dtype), dtype)
    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({"x": x})["y"]
    session.close()
    module.close()

    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("op_name", "attrs", "expected_fn", "source_snippet"),
    [
        ("var", {}, lambda x: np.var(x, axis=-1, keepdims=True).astype(np.float32), "sum_sq_acc"),
        ("var", {"unbiased": True}, lambda x: np.var(x, axis=-1, keepdims=True, ddof=1).astype(np.float32), "/ 15.00000000f"),
        ("vector_norm", {}, lambda x: np.sqrt(np.sum(x * x, axis=-1, keepdims=True)).astype(np.float32), "sqrtf(acc)"),
    ],
)
def test_cpu_artifact_runs_generated_var_and_vector_norm(tmp_path, op_name, attrs, expected_fn, source_snippet):
    spec = dml.trace(
        ReductionLastDim(op_name, keepdim=True, **attrs),
        inputs={"x": dml.TensorSpec([4, 8, 16], "float32")},
        name=f"{op_name}_{'_'.join(attrs) if attrs else 'default'}_keepdim_cpu",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"{spec.name}.dinoml")
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert f"{op_name}_" in generated
    assert source_snippet in generated

    x = np.random.default_rng(43).standard_normal((4, 8, 16)).astype(np.float32)
    expected = expected_fn(x)
    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({"x": x})["y"]
    session.close()
    module.close()

    assert actual.shape == (4, 8, 1)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_cpu_artifact_runs_with_expression_output_shape(tmp_path):
    batch = {"kind": "dim", "name": "batch", "min": 1, "max": 4}
    half = {"kind": "dim", "name": "half", "min": 2, "max": 8}
    length = {"kind": "dim", "name": "length", "min": 4, "max": 16}
    pooled_length = {"kind": "int_expr", "op": "div", "lhs": length, "rhs": 2}
    spec = dml.trace(
        AddZeroUnusedShapeSource(),
        inputs={"x": dml.TensorSpec([4, 8], "float32"), "z": dml.TensorSpec([16], "float32")},
        name="add_zero_expr_shape_cpu",
    )
    spec = spec.clone()
    for item in spec.ir["inputs"]:
        if item["name"] == "x":
            item["shape_spec"] = [batch, half]
        elif item["name"] == "z":
            item["shape_spec"] = [length]
    for item in spec.ir["outputs"]:
        if item["name"] == "y":
            item["shape_spec"] = [batch, pooled_length]
    for tensor in spec.ir["tensors"]:
        if tensor["name"] == "x":
            tensor["shape_spec"] = [batch, half]
        elif tensor["name"] == "z":
            tensor["shape_spec"] = [length]
        elif tensor["name"] == "t0":
            tensor["shape_spec"] = [batch, pooled_length]

    artifact = dml.compile(
        spec,
        dml.Target("cpu"),
        tmp_path / "add_zero_expr_shape_cpu.dinoml",
        pass_manager=PassManager(
            pipeline=("canonicalize", "constant_bind", "dead_code_eliminate", "elementwise_fusion", "memory_plan", "backend_lower")
        ),
    )
    generated = (artifact.path / "debug" / "generated_src" / "module.cpp").read_text(encoding="utf-8")
    assert "dinoml::module::floor_div(inputs[1].shape[0], 2)" in generated
    assert "Shape expression mismatch for y axis 1" in generated

    x = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)
    z = np.zeros((16,), dtype=np.float32)
    module = runtime.load(artifact.path)
    session = module.create_session()
    actual = session.run_numpy({"x": x, "z": z})["y"]
    assert session.get_output_shape("y") == (2, 8)
    session.close()
    module.close()
    assert actual.shape == (2, 8)
    np.testing.assert_allclose(actual, x, atol=1e-6, rtol=1e-6)


def test_reduction_rejects_non_last_dim():
    class ReduceNonLastDim(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.reduce_sum(x, dim=0), "y")

    with pytest.raises(NotImplementedError, match="last dimension"):
        dml.trace(
            ReduceNonLastDim(),
            inputs={"x": dml.TensorSpec([4, 8], "float32")},
            name="reduce_non_last",
        )


def test_softmax_rejects_non_last_dim():
    try:
        dml.trace(
            SoftmaxNonLastDim(),
            inputs={"x": dml.TensorSpec([4, 8], "float32")},
            name="softmax_non_last",
        )
    except NotImplementedError as exc:
        assert "last dimension" in str(exc)
    else:
        raise AssertionError("softmax accepted a non-last dimension")


def test_softmax_rejects_rank_one_input():
    class RankOneSoftmax(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.softmax(x), "y")

    with pytest.raises(ValueError, match="rank >= 2"):
        dml.trace(
            RankOneSoftmax(),
            inputs={"x": dml.TensorSpec([8], "float32")},
            name="softmax_rank_one",
        )


def test_softmax_rejects_dynamic_last_dim():
    keys = dml.Dim("keys", min=1, max=16)
    try:
        dml.trace(
            SoftmaxLastDim(),
            inputs={"x": dml.TensorSpec([4, keys], "float32")},
            name="softmax_dynamic_last",
        )
    except ValueError as exc:
        assert "static last dimension" in str(exc)
    else:
        raise AssertionError("softmax accepted a dynamic last dimension")
