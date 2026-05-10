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


class FlipModule(dml.Module):
    def __init__(self, dims):
        self.dims = dims

    def forward(self, x):
        return dml.ops.output(dml.ops.flip(x, self.dims), "out")


def _trace_flip(dtype="float32", dims=(-1, 0), shape=(2, 3, 4)):
    return dml.trace(FlipModule(dims), inputs={"x": dml.TensorSpec(shape, dtype)}, name=f"flip_{dtype}")


def _storage_roundtrip(value, dtype):
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return np.asarray(value, dtype=np.bool_ if dtype == "bool" else np.float32)


def _input(dtype):
    if dtype == "bool":
        return np.array(
            [[[True, False], [False, True], [True, True]], [[False, False], [True, False], [False, True]]],
            dtype=np.bool_,
        )
    return np.arange(12, dtype=np.float32).reshape(2, 3, 2)


def test_flip_frontend_ir_normalizes_negative_dims_and_int_dim():
    spec = _trace_flip("float32", dims=(-1, 0))

    assert spec.ir["outputs"][0]["shape"] == [2, 3, 4]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 3, 4]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "flip"
    assert node["inputs"] == ["x"]
    assert node["attrs"] == {"dims": [2, 0]}

    scalar_dim_spec = _trace_flip("float32", dims=-2)
    assert scalar_dim_spec.ir["nodes"][0]["attrs"] == {"dims": [1]}


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("float32", np.float32),
        ("float16", np.float16),
        ("bfloat16", np.float32),
        ("bool", np.bool_),
    ],
)
def test_cpu_reference_flip(dtype, expected_dtype):
    spec = _trace_flip(dtype, dims=(0, 2), shape=(2, 3, 2))
    x = _input(dtype)

    actual = execute_cpu(spec, {"x": x})["out"]

    expected = _storage_roundtrip(np.flip(x, axis=(0, 2)).copy(), dtype)
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "bool"])
def test_flip_generated_cpu_source_and_runtime(tmp_path, dtype):
    spec = _trace_flip(dtype, dims=(0, 2), shape=(2, 3, 2))
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["flip"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int flip_" in cpu_source
    assert "coord = 1 - coord" in cpu_source
    assert "y[idx] = x[input_idx]" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x" in cpu_source
        assert "float* DINO_RESTRICT y" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"flip_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input(dtype)
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    expected = _storage_roundtrip(np.flip(x, axis=(0, 2)).copy(), dtype)
    np.testing.assert_array_equal(actual, expected)


def test_flip_generated_cuda_source_supports_reduced_precision_and_bool():
    for dtype, pointer_type in (
        ("float16", "const half* DINO_RESTRICT x"),
        ("bfloat16", "const __nv_bfloat16* DINO_RESTRICT x"),
        ("bool", "const bool* DINO_RESTRICT x"),
    ):
        spec = _trace_flip(dtype, dims=(0, 2), shape=(2, 3, 2))
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert "flip_" in cuda_source
        assert "y[idx] = x[input_idx]" in cuda_source


def test_flip_frontend_rejects_dynamic_bad_dims_and_unsupported_dtype():
    class DynamicFlip(dml.Module):
        def forward(self, x):
            return dml.ops.flip(x, dims=0)

    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(DynamicFlip(), inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3])})
    with pytest.raises(ValueError, match="non-empty"):
        _trace_flip("float32", dims=())
    with pytest.raises(ValueError, match="out of range"):
        _trace_flip("float32", dims=3)
    with pytest.raises(ValueError, match="duplicates"):
        _trace_flip("float32", dims=(0, -3))
    with pytest.raises(ValueError, match="integers"):
        _trace_flip("float32", dims=(0, "1"))
    with pytest.raises(ValueError, match="does not support dtype int64"):
        _trace_flip("int64", dims=0)


def test_flip_validation_rejects_dynamic_shape_spec_bad_dims_shape_and_dtype():
    spec = _trace_flip("float32", dims=(0, 2))
    spec.ir["inputs"][0]["shape_spec"] = [Dim("n", 1, 2).to_json(), 3, 4]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [Dim("n", 1, 2).to_json(), 3, 4]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace_flip("float32", dims=(0, 2))
    spec.ir["nodes"][0]["attrs"]["dims"] = [0, 3]
    with pytest.raises(ValidationError, match="out of range"):
        validate_ir(spec.ir)

    spec = _trace_flip("float32", dims=(0, 2))
    spec.ir["nodes"][0]["attrs"]["dims"] = [0, 0]
    with pytest.raises(ValidationError, match="duplicates"):
        validate_ir(spec.ir)

    spec = _trace_flip("float32", dims=(0, 2))
    spec.ir["outputs"][0]["shape"] = [2, 3, 5]
    spec.ir["outputs"][0]["shape_spec"] = [2, 3, 5]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 3, 5]
    output_tensor["shape_spec"] = [2, 3, 5]
    output_tensor["layout"]["strides"] = [15, 5, 1]
    with pytest.raises(ValidationError, match=r"expected \[2, 3, 4\]"):
        validate_ir(spec.ir)

    spec = _trace_flip("float32", dims=(0, 2))
    spec.ir["outputs"][0]["dtype"] = "int64"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "int64"
    with pytest.raises(ValidationError, match="expected float32"):
        validate_ir(spec.ir)
