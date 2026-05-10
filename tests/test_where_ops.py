import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.lowering.ops import render_generated_kernels
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError
from dinoml.runtime import load


class WhereModule(dml.Module):
    def forward(self, condition, x, y):
        return dml.ops.output(dml.ops.where(condition, x, y), "out")


class RelationalWhereModule(dml.Module):
    def forward(self, x, y):
        condition = dml.ops.lt(x, y)
        return dml.ops.output(dml.ops.where(condition, x, y), "out")


def _trace_where(dtype: str = "float32"):
    return dml.trace(
        WhereModule(),
        inputs={
            "condition": dml.TensorSpec([2, 1], "bool"),
            "x": dml.TensorSpec([1, 3], dtype),
            "y": dml.TensorSpec([2, 3], dtype),
        },
        name=f"where_{dtype}",
    )


def _storage_roundtrip(value, dtype):
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    if dtype == "bool":
        return np.asarray(value, dtype=np.bool_)
    return np.asarray(value, dtype=np.float32)


def _where_inputs(dtype):
    condition = np.array([[True], [False]], dtype=np.bool_)
    if dtype == "bool":
        x = np.array([[True, False, True]], dtype=np.bool_)
        y = np.array([[False, True, False], [True, True, False]], dtype=np.bool_)
        return condition, x, y
    x = np.array([[1.0, -2.5, 3.25]], dtype=np.float32)
    y = np.array([[10.0, 20.5, -30.25], [40.75, -50.0, 60.125]], dtype=np.float32)
    if dtype == "float16":
        return condition, x.astype(np.float16), y.astype(np.float16)
    return condition, x, y


def test_where_frontend_ir_dtype_shape_and_broadcast():
    spec = _trace_where()

    assert spec.ir["outputs"][0]["shape"] == [2, 3]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    where_node = spec.ir["nodes"][0]
    assert where_node["op"] == "where"
    assert where_node["inputs"] == ["condition", "x", "y"]


def test_where_frontend_allows_bool_values_with_broadcast_shape():
    spec = _trace_where("bool")

    assert spec.ir["outputs"][0]["shape"] == [2, 3]
    assert spec.ir["outputs"][0]["dtype"] == "bool"
    where_node = spec.ir["nodes"][0]
    assert where_node["op"] == "where"
    assert where_node["inputs"] == ["condition", "x", "y"]


def test_cpu_reference_where_executes_broadcast():
    spec = _trace_where()
    condition = np.array([[True], [False]], dtype=np.bool_)
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    y = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)

    actual = execute_cpu(spec, {"condition": condition, "x": x, "y": y})["out"]

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.where(condition, x, y))


def test_cpu_reference_where_executes_bool_values_with_broadcast():
    spec = _trace_where("bool")
    condition, x, y = _where_inputs("bool")

    actual = execute_cpu(spec, {"condition": condition, "x": x, "y": y})["out"]

    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual, np.where(condition, x, y))


@pytest.mark.parametrize(
    ("dtype", "input_pointer", "output_pointer"),
    [
        ("float32", "const float* DINO_RESTRICT ptr_x", "float* DINO_RESTRICT ptr_"),
        ("float16", "const dinoml::math::float16* DINO_RESTRICT ptr_x", "dinoml::math::float16* DINO_RESTRICT ptr_"),
        ("bfloat16", "const dinoml::math::bfloat16* DINO_RESTRICT ptr_x", "dinoml::math::bfloat16* DINO_RESTRICT ptr_"),
        ("bool", "const bool* DINO_RESTRICT ptr_x", "bool* DINO_RESTRICT ptr_"),
    ],
)
def test_where_fused_generated_cpu_source_and_runtime(tmp_path, dtype, input_pointer, output_pointer):
    spec = _trace_where(dtype)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    fused = next(node for node in lowered["nodes"] if node["op"] == "fused_elementwise")
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", [fused], tensor_map)[0]

    assert "const bool* DINO_RESTRICT ptr_condition" in cpu_source
    assert input_pointer in cpu_source
    assert input_pointer.replace("ptr_x", "ptr_y") in cpu_source
    assert output_pointer in cpu_source
    assert "dinoml::math::where" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"where_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    condition, x, y = _where_inputs(dtype)
    try:
        actual = session.run_numpy({"condition": condition, "x": x, "y": y})["out"]
    finally:
        session.close()

    expected = _storage_roundtrip(
        np.where(condition, _storage_roundtrip(x, dtype), _storage_roundtrip(y, dtype)),
        dtype,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "input_pointer", "output_pointer"),
    [
        ("float16", "const half* DINO_RESTRICT ptr_x", "half* DINO_RESTRICT ptr_"),
        ("bfloat16", "const __nv_bfloat16* DINO_RESTRICT ptr_x", "__nv_bfloat16* DINO_RESTRICT ptr_"),
        ("bool", "const bool* DINO_RESTRICT ptr_x", "bool* DINO_RESTRICT ptr_"),
    ],
)
def test_where_generated_cuda_source_uses_mixed_pointer_types(dtype, input_pointer, output_pointer):
    spec = _trace_where(dtype)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    fused = next(node for node in lowered["nodes"] if node["op"] == "fused_elementwise")
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cuda_source = render_generated_kernels("cuda", [fused], tensor_map)[0]

    assert "const bool* DINO_RESTRICT ptr_condition" in cuda_source
    assert input_pointer in cuda_source
    assert input_pointer.replace("ptr_x", "ptr_y") in cuda_source
    assert output_pointer in cuda_source
    assert "dinoml::math::where" in cuda_source


def test_where_fuses_with_relational_condition_and_runs_cpu(tmp_path):
    spec = dml.trace(
        RelationalWhereModule(),
        inputs={"x": dml.TensorSpec([2, 3], "float32"), "y": dml.TensorSpec([1, 3], "float32")},
        name="where_relational_condition",
    )
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["fused_elementwise"]
    fused = lowered["nodes"][0]
    assert [sub_op["op"] for sub_op in fused["attrs"]["sub_ops"]] == ["lt", "where"]

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "where_relational_condition_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = np.array([[0.0, 2.0, 4.0], [7.0, 1.0, 3.0]], dtype=np.float32)
    y = np.array([[1.0, 1.0, 5.0]], dtype=np.float32)
    actual = session.run_numpy({"x": x, "y": y})["out"]
    session.close()

    np.testing.assert_array_equal(actual, np.where(x < y, x, y))


def test_where_frontend_rejects_non_bool_condition():
    class BadCondition(dml.Module):
        def forward(self, condition, x, y):
            return dml.ops.where(condition, x, y)

    with pytest.raises(ValueError, match="condition must have dtype bool"):
        dml.trace(
            BadCondition(),
            inputs={
                "condition": dml.TensorSpec([2, 1], "float32"),
                "x": dml.TensorSpec([1, 3], "float32"),
                "y": dml.TensorSpec([2, 3], "float32"),
            },
        )


def test_where_frontend_rejects_mismatched_x_y_dtypes():
    class BadValues(dml.Module):
        def forward(self, condition, x, y):
            return dml.ops.where(condition, x, y)

    with pytest.raises(ValueError, match="x/y dtype mismatch"):
        dml.trace(
            BadValues(),
            inputs={
                "condition": dml.TensorSpec([2, 1], "bool"),
                "x": dml.TensorSpec([1, 3], "float32"),
                "y": dml.TensorSpec([2, 3], "float16"),
            },
        )


def test_where_validation_rejects_bad_condition_dtype():
    spec = _trace_where()
    condition = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "condition")
    condition["dtype"] = "float32"

    with pytest.raises(ValidationError, match="condition must have dtype bool"):
        validate_ir(spec.ir)


def test_where_validation_rejects_mismatched_x_y_dtypes():
    spec = _trace_where()
    y = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "y")
    y["dtype"] = "float16"

    with pytest.raises(ValidationError, match="x/y dtype mismatch"):
        validate_ir(spec.ir)
