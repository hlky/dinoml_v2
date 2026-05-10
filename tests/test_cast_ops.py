import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.lowering.ops import render_generated_kernels
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError
from dinoml.runtime import load


class CastModule(dml.Module):
    def __init__(self, dtype: str):
        self.dtype = dtype

    def forward(self, x):
        return dml.ops.output(dml.ops.cast(x, self.dtype), "out")


class CastRoundtripModule(dml.Module):
    def forward(self, x):
        y = dml.ops.cast(x, "float16")
        return dml.ops.output(dml.ops.cast(y, "float32"), "out")


def _trace_cast(input_dtype: str = "float32", output_dtype: str = "float16"):
    return dml.trace(
        CastModule(output_dtype),
        inputs={"x": dml.TensorSpec([2, 3], input_dtype)},
        name=f"cast_{input_dtype}_to_{output_dtype}",
    )


def test_cast_frontend_ir_preserves_shape_and_sets_output_dtype():
    spec = _trace_cast("float32", "bool")

    assert spec.ir["outputs"][0]["shape"] == [2, 3]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 3]
    assert spec.ir["outputs"][0]["dtype"] == "bool"
    node = spec.ir["nodes"][0]
    assert node["op"] == "cast"
    assert node["attrs"] == {"dtype": "bool"}
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    assert output_tensor["dtype"] == "bool"
    assert output_tensor["nbytes"] == 6


def test_cpu_reference_cast_to_bool():
    spec = _trace_cast("float32", "bool")
    x = np.array([[0.0, -1.5, 2.0], [0.0, np.nan, 3.0]], dtype=np.float32)

    actual = execute_cpu(spec, {"x": x})["out"]

    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual, x.astype(np.bool_))


def test_cpu_reference_cast_from_bool_to_float32():
    spec = _trace_cast("bool", "float32")
    x = np.array([[False, True, True], [False, False, True]], dtype=np.bool_)

    actual = execute_cpu(spec, {"x": x})["out"]

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, x.astype(np.float32))


def test_cast_lowers_to_fused_elementwise_with_mixed_pointer_types_and_runs_cpu(tmp_path):
    spec = _trace_cast("float32", "float16")
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["fused_elementwise"]
    fused = lowered["nodes"][0]
    assert fused["attrs"]["sub_ops"][0]["op"] == "cast"
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", [fused], tensor_map)[0]

    assert "const float* DINO_RESTRICT ptr_x" in cpu_source
    assert "dinoml::math::float16* DINO_RESTRICT ptr_" in cpu_source
    assert "dinoml::math::cast<" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "cast_float32_to_float16_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = np.array([[-2.25, -1.0, 0.0], [1.125, 2.5, 3.75]], dtype=np.float32)
    expected = array_from_storage(array_to_storage(x, "float16"), "float16")
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    assert actual.dtype == np.float16
    np.testing.assert_array_equal(actual, expected)


def test_reduced_precision_cast_roundtrip_runs_cpu(tmp_path):
    spec = dml.trace(
        CastRoundtripModule(),
        inputs={"x": dml.TensorSpec([2, 3], "float32")},
        name="cast_float16_roundtrip",
    )
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "cast_float16_roundtrip_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = np.array([[-2.25, -1.0, 0.0], [1.125, 2.5, 3.75]], dtype=np.float32)
    expected = array_from_storage(array_to_storage(x, "float16"), "float16").astype(np.float32)
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def test_cast_frontend_rejects_int_dtype_until_storage_lowering_exists():
    class BadCast(dml.Module):
        def forward(self, x):
            return dml.ops.cast(x, "int32")

    with pytest.raises(ValueError, match="cast does not support dtype int32"):
        dml.trace(BadCast(), inputs={"x": dml.TensorSpec([2, 3], "float32")})


def test_cast_validation_rejects_int_dtype_attr():
    spec = _trace_cast("float32", "bool")
    spec.ir["nodes"][0]["attrs"]["dtype"] = "int64"
    spec.ir["outputs"][0]["dtype"] = "int64"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "int64"

    with pytest.raises(ValidationError, match="cast does not support dtype int64"):
        validate_ir(spec.ir)
