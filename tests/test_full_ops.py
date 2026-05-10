import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.lowering.ops import render_generated_kernels
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError
from dinoml.runtime import load
from dinoml.shapes import Dim


class FullModule(dml.Module):
    def __init__(self, shape, fill_value, dtype="float32"):
        self.shape = shape
        self.fill_value = fill_value
        self.dtype = dtype

    def forward(self):
        return dml.ops.output(dml.ops.full(self.shape, self.fill_value, dtype=self.dtype), "out")


def _trace_full(shape=(2, 3), fill_value=1.25, dtype="float32"):
    return dml.trace(FullModule(shape, fill_value, dtype), inputs={}, name=f"full_{dtype}")


def test_full_frontend_ir_preserves_shape_spec_dtype_and_attrs():
    spec = _trace_full([2, 3], 1.25, "float32")

    assert spec.ir["inputs"] == []
    assert spec.ir["outputs"][0]["shape"] == [2, 3]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 3]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "full"
    assert node["inputs"] == []
    assert node["attrs"] == {"shape": [2, 3], "fill_value": 1.25, "dtype": "float32"}


@pytest.mark.parametrize(
    ("dtype", "fill_value", "expected_dtype"),
    [
        ("float32", -2.5, np.float32),
        ("float16", 1.125, np.float16),
        ("bfloat16", 1.125, np.float32),
        ("bool", True, np.bool_),
    ],
)
def test_cpu_reference_full(dtype, fill_value, expected_dtype):
    spec = _trace_full([2, 3], fill_value, dtype)

    actual = execute_cpu(spec, {})["out"]

    expected = np.full([2, 3], fill_value, dtype=np.bool_ if dtype == "bool" else np.float32)
    if dtype in {"float16", "bfloat16"}:
        expected = array_from_storage(array_to_storage(expected, dtype), dtype)
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "fill_value"),
    [
        ("float32", -3.5),
        ("float16", 1.125),
        ("bfloat16", 1.125),
        ("bool", True),
    ],
)
def test_full_generated_cpu_source_and_runtime(tmp_path, dtype, fill_value):
    spec = _trace_full([2, 3], fill_value, dtype)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["full"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int full_" in cpu_source
    if dtype == "float32":
        assert "float* DINO_RESTRICT y" in cpu_source
        assert "dinoml::math::cast<float>(-3.5f)" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"full_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    try:
        actual = session.run_numpy({})["out"]
    finally:
        session.close()

    expected = np.full([2, 3], fill_value, dtype=np.bool_ if dtype == "bool" else np.float32)
    if dtype in {"float16", "bfloat16"}:
        expected = array_from_storage(array_to_storage(expected, dtype), dtype)
    np.testing.assert_array_equal(actual, expected)


def test_full_generated_cuda_source_supports_reduced_precision_and_bool():
    for dtype, fill_value, pointer_type, fill_expr in (
        ("float16", 1.25, "half* DINO_RESTRICT y", "dinoml::math::cast<half>(1.25f)"),
        ("bfloat16", 1.25, "__nv_bfloat16* DINO_RESTRICT y", "dinoml::math::cast<__nv_bfloat16>(1.25f)"),
        ("bool", True, "bool* DINO_RESTRICT y", "dinoml::math::cast<bool>(1.0f)"),
    ):
        spec = _trace_full([2, 3], fill_value, dtype)
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert fill_expr in cuda_source


def test_full_frontend_rejects_dynamic_empty_non_positive_and_int_dtype():
    with pytest.raises(ValueError, match="only static shapes"):
        _trace_full([Dim("n", 1, 4), 3], 1.0)
    with pytest.raises(ValueError, match="must not be empty"):
        _trace_full([], 1.0)
    with pytest.raises(ValueError, match="positive"):
        _trace_full([2, 0], 1.0)
    with pytest.raises(ValueError, match="full does not support dtype int32"):
        _trace_full([2, 3], 1, "int32")


def test_full_validation_rejects_bad_attr_shape_and_dtype():
    spec = _trace_full([2, 3], 1.0, "float32")
    spec.ir["nodes"][0]["attrs"]["shape"] = [2, 0]
    with pytest.raises(ValidationError, match="positive integer shape"):
        validate_ir(spec.ir)

    spec = _trace_full([2, 3], 1.0, "float32")
    spec.ir["nodes"][0]["attrs"]["dtype"] = "int64"
    spec.ir["outputs"][0]["dtype"] = "int64"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "int64"
    with pytest.raises(ValidationError, match="full does not support dtype int64"):
        validate_ir(spec.ir)
