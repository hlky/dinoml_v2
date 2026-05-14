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


class ArgmaxModule(dml.Module):
    def __init__(self, dim=-1, keepdim=False):
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return dml.ops.output(dml.ops.argmax(x, dim=self.dim, keepdim=self.keepdim), "out")


def _trace(dtype="float32", shape=(2, 3, 4), dim=-1, keepdim=False):
    return dml.trace(
        ArgmaxModule(dim=dim, keepdim=keepdim),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name=f"argmax_{dtype}",
    )


def _input(dtype, shape=(2, 3, 4)):
    values = np.array(
        [
            [[1.0, 5.0, 5.0, -1.0], [0.0, -2.0, 3.0, 3.0], [7.0, 1.0, 7.0, 0.0]],
            [[4.0, 4.0, 2.0, 1.0], [-1.0, -1.0, -1.0, -2.0], [0.0, 9.0, 8.0, 9.0]],
        ],
        dtype=np.float32,
    )
    values = values.reshape(shape)
    if dtype == "bool":
        return (values.astype(np.int64) % 3) == 0
    if dtype == "int32":
        return values.astype(np.int32)
    if dtype == "int64":
        return values.astype(np.int64)
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(values, dtype), dtype)
    return values


def _expected(x, dim=-1, keepdim=False):
    result = np.argmax(x, axis=dim)
    if keepdim:
        if dim < 0:
            dim += x.ndim
        result = np.expand_dims(result, axis=dim)
    if result.shape == ():
        result = np.reshape(result, [1])
    return np.asarray(result, dtype=np.int64)


def test_argmax_frontend_ir_normalizes_attrs_shape_and_dtype():
    spec = _trace("float32", dim=-1)

    assert spec.ir["outputs"][0]["shape"] == [2, 3]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 3]
    assert spec.ir["outputs"][0]["dtype"] == "int64"
    node = spec.ir["nodes"][0]
    assert node["op"] == "argmax"
    assert node["inputs"] == ["x"]
    assert node["attrs"] == {"dim": 2, "keepdim": False}


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "bool", "int32", "int64"])
def test_cpu_reference_argmax(dtype):
    spec = _trace(dtype)
    x = _input(dtype)

    actual = execute_cpu(spec, {"x": x})["out"]

    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, _expected(x))


def test_argmax_ties_return_first_index():
    spec = _trace("float32", shape=(2, 4))
    x = np.array([[1.0, 3.0, 3.0, 2.0], [5.0, 5.0, 4.0, 5.0]], dtype=np.float32)

    actual = execute_cpu(spec, {"x": x})["out"]

    np.testing.assert_array_equal(actual, np.array([1, 0], dtype=np.int64))


def test_argmax_keepdim_and_scalar_fallback_shape():
    keepdim_spec = _trace("float32", shape=(2, 4), keepdim=True)
    keepdim_x = np.array([[1.0, 3.0, 2.0, 0.0], [4.0, -1.0, 5.0, 5.0]], dtype=np.float32)
    scalar_spec = _trace("float32", shape=(4,))
    scalar_x = np.array([1.0, 7.0, 7.0, 2.0], dtype=np.float32)

    assert keepdim_spec.ir["outputs"][0]["shape"] == [2, 1]
    assert scalar_spec.ir["outputs"][0]["shape"] == [1]
    np.testing.assert_array_equal(execute_cpu(keepdim_spec, {"x": keepdim_x})["out"], [[1], [2]])
    np.testing.assert_array_equal(execute_cpu(scalar_spec, {"x": scalar_x})["out"], [1])


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "bool", "int32", "int64"])
def test_argmax_generated_cpu_source_and_runtime(tmp_path, dtype):
    spec = _trace(dtype)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["argmax"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int argmax_" in cpu_source
    assert "int64_t* DINO_RESTRICT y" in cpu_source
    assert "best_index = 0" in cpu_source
    assert "value > best_value" in cpu_source
    assert "y[row] = best_index" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x" in cpu_source
        assert "dinoml::math::cast<float>(x[base + col])" in cpu_source
        assert "std::isnan(value)" in cpu_source
    if dtype == "bool":
        assert "const bool* DINO_RESTRICT x" in cpu_source
        assert "bool best_value = x[base]" in cpu_source
    if dtype == "int32":
        assert "const int32_t* DINO_RESTRICT x" in cpu_source
        assert "int32_t best_value = x[base]" in cpu_source
        assert "dinoml::math::cast<float>" not in cpu_source
        assert "std::isnan(value)" not in cpu_source
    if dtype == "int64":
        assert "const int64_t* DINO_RESTRICT x" in cpu_source
        assert "int64_t best_value = x[base]" in cpu_source
        assert "dinoml::math::cast<float>" not in cpu_source
        assert "std::isnan(value)" not in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"argmax_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input(dtype)
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, _expected(x))


def test_argmax_generated_cuda_source_supports_reduced_precision_bool_integer_inputs_and_int64_output():
    for dtype, pointer_type in (
        ("float16", "const half* DINO_RESTRICT x"),
        ("bfloat16", "const __nv_bfloat16* DINO_RESTRICT x"),
        ("bool", "const bool* DINO_RESTRICT x"),
        ("int32", "const int32_t* DINO_RESTRICT x"),
        ("int64", "const int64_t* DINO_RESTRICT x"),
    ):
        spec = _trace(dtype)
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert "int64_t* DINO_RESTRICT y" in cuda_source
        assert "argmax_" in cuda_source
        assert "value > best_value" in cuda_source
        assert "y[row] = best_index" in cuda_source
        if dtype in {"int32", "int64"}:
            assert "dinoml::math::cast<float>" not in cuda_source
            assert "isnan(value)" not in cuda_source


def test_argmax_frontend_rejects_dynamic_non_last_dim_and_bad_dtype():
    class DynamicShapeArgmax(dml.Module):
        def forward(self, x):
            return dml.ops.argmax(x)

    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(DynamicShapeArgmax(), inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3])})
    with pytest.raises(ValueError, match="dim must be an integer"):
        _trace("float32", dim=True)
    with pytest.raises(ValueError, match="out of range"):
        _trace("float32", dim=3)
    with pytest.raises(NotImplementedError, match="only the last dimension"):
        _trace("float32", dim=1)
    with pytest.raises(ValueError, match="does not support dtype float8_e4m3"):
        _trace("float8_e4m3")


def test_argmax_validation_rejects_dynamic_shape_specs_bad_attrs_shape_and_dtype():
    spec = _trace("float32")
    spec.ir["inputs"][0]["shape_spec"] = [Dim("n", 1, 2).to_json(), 3, 4]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [Dim("n", 1, 2).to_json(), 3, 4]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["nodes"][0]["attrs"]["dim"] = True
    with pytest.raises(ValidationError, match="dim must be an integer"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["nodes"][0]["attrs"]["dim"] = 1
    with pytest.raises(ValidationError, match="only the last dimension"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["outputs"][0]["shape"] = [2, 2]
    spec.ir["outputs"][0]["shape_spec"] = [2, 2]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 2]
    output_tensor["shape_spec"] = [2, 2]
    output_tensor["layout"]["strides"] = [2, 1]
    with pytest.raises(ValidationError, match=r"expected \[2, 3\]"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["outputs"][0]["dtype"] = "float32"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "float32"
    with pytest.raises(ValidationError, match="expected int64"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["dtype"] = "float8_e4m3"
    with pytest.raises(ValidationError, match="argmax does not support dtype float8_e4m3"):
        validate_ir(spec.ir)


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_argmax_frontend_ir_and_runtime_allow_bounded_integer_inputs(dtype, tmp_path):
    spec = _trace(dtype, shape=(2, 5))
    assert spec.ir["inputs"][0]["dtype"] == dtype
    assert spec.ir["outputs"][0]["dtype"] == "int64"

    values = np.array([[1, 7, 7, 3, 0], [4, 4, 2, 9, 9]], dtype=np.int64).astype(dtype)
    np.testing.assert_array_equal(execute_cpu(spec, {"x": values})["out"], np.array([1, 3], dtype=np.int64))

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"argmax_{dtype}_bounded_cpu.dinoml")
    session = load(artifact.path).create_session()
    try:
        actual = session.run_numpy({"x": values})["out"]
    finally:
        session.close()
    np.testing.assert_array_equal(actual, np.array([1, 3], dtype=np.int64))


def test_argmax_clip_legacy_eot_pooling_regression_with_int64_ids():
    spec = _trace("int64", shape=(3, 6))
    input_ids = np.array(
        [
            [49406, 120, 49407, 17, 0, 0],
            [49406, 49407, 42, 49407, 12, 0],
            [49406, 1, 2, 3, 4, 5],
        ],
        dtype=np.int64,
    )

    actual = execute_cpu(spec, {"x": input_ids})["out"]

    np.testing.assert_array_equal(actual, np.array([2, 1, 0], dtype=np.int64))


def test_argmax_runtime_contract_keeps_unrelated_integer_tensors_rejected(tmp_path):
    class PassthroughModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(x, "out")

    spec = dml.trace(PassthroughModule(), inputs={"x": dml.TensorSpec([2, 3], "int64")}, name="passthrough_int64")

    with pytest.raises(NotImplementedError, match="unsupported compiled dtypes: \\['int64'\\]"):
        dml.compile(spec, dml.Target("cpu"), tmp_path / "passthrough_int64_cpu.dinoml")
