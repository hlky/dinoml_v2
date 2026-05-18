import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from dinoml.passes import validate_ir
from dinoml.passes.validation import ValidationError


class CastModule(dml.Module):
    def __init__(self, dtype: str):
        self.dtype = dtype

    def forward(self, x):
        return dml.ops.output(dml.ops.cast(x, self.dtype), "out")


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


def test_reference_numpy_cast_to_bool():
    spec = _trace_cast("float32", "bool")
    x = np.array([[0.0, -1.5, 2.0], [0.0, np.nan, 3.0]], dtype=np.float32)

    actual = reference_numpy(spec, {"x": x})["out"]

    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual, x.astype(np.bool_))


def test_reference_numpy_cast_from_bool_to_float32():
    spec = _trace_cast("bool", "float32")
    x = np.array([[False, True, True], [False, False, True]], dtype=np.bool_)

    actual = reference_numpy(spec, {"x": x})["out"]

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, x.astype(np.float32))


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
