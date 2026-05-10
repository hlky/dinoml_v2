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


class BatchGatherModule(dml.Module):
    def forward(self, x, indices):
        return dml.ops.output(dml.ops.batch_gather(x, indices), "out")


def _trace(dtype="float32", index_dtype="int64", x_shape=(2, 4, 3), index_shape=(2, 3)):
    return dml.trace(
        BatchGatherModule(),
        inputs={"x": dml.TensorSpec(x_shape, dtype), "indices": dml.TensorSpec(index_shape, index_dtype)},
        name=f"batch_gather_{dtype}_{index_dtype}",
    )


def _storage_roundtrip(value, dtype):
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return np.asarray(value, dtype=np.bool_ if dtype == "bool" else np.float32)


def _input(dtype):
    if dtype == "bool":
        return (np.arange(24).reshape(2, 4, 3) % 3) == 0
    return np.arange(24, dtype=np.float32).reshape(2, 4, 3)


def _indices(index_dtype="int64"):
    return np.array([[3, 0, 1], [1, 2, 0]], dtype=np.int64 if index_dtype == "int64" else np.int32)


def _expected_batch_gather(x, indices, dtype):
    result = np.empty((indices.shape[0], indices.shape[1], *x.shape[2:]), dtype=x.dtype)
    for b in range(indices.shape[0]):
        for k in range(indices.shape[1]):
            result[b, k] = x[b, int(indices[b, k])]
    return _storage_roundtrip(result, dtype)


def test_batch_gather_frontend_ir_shape_and_dtype():
    spec = _trace("float32", "int64")

    assert spec.ir["outputs"][0]["shape"] == [2, 3, 3]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 3, 3]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "batch_gather"
    assert node["inputs"] == ["x", "indices"]
    assert node["attrs"] == {}


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("float32", np.float32),
        ("float16", np.float16),
        ("bfloat16", np.float32),
        ("bool", np.bool_),
    ],
)
def test_cpu_reference_batch_gather(dtype, expected_dtype):
    spec = _trace(dtype)
    x = _input(dtype)
    indices = _indices()

    actual = execute_cpu(spec, {"x": x, "indices": indices})["out"]

    expected = _expected_batch_gather(x, indices, dtype)
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


def test_cpu_reference_batch_gather_accepts_int32_indices():
    spec = _trace("float32", "int32")
    x = _input("float32")
    indices = _indices("int32")

    actual = execute_cpu(spec, {"x": x, "indices": indices})["out"]

    np.testing.assert_array_equal(actual, _expected_batch_gather(x, indices, "float32"))


def test_batch_gather_rank2_input_runtime(tmp_path):
    spec = _trace("float32", x_shape=(2, 4), index_shape=(2, 3))
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "batch_gather_rank2_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    indices = _indices()
    try:
        actual = session.run_numpy({"x": x, "indices": indices})["out"]
    finally:
        session.close()

    expected = _expected_batch_gather(x, indices, "float32")
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "bool"])
def test_batch_gather_generated_cpu_source_and_runtime(tmp_path, dtype):
    spec = _trace(dtype)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["batch_gather"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int batch_gather_" in cpu_source
    assert "const int64_t* DINO_RESTRICT index" in cpu_source
    assert "selected_index = static_cast<int64_t>(index[batch * 3 + k]);" in cpu_source
    assert 'return dino_runtime_fail("batch_gather index out of bounds");' in cpu_source
    assert "const int64_t input_idx = batch * 12 + selected_index * 3 + slice_offset;" in cpu_source
    assert "y[idx] = x[input_idx]" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x" in cpu_source
        assert "float* DINO_RESTRICT y" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"batch_gather_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input(dtype)
    indices = _indices()
    try:
        actual = session.run_numpy({"x": x, "indices": indices})["out"]
    finally:
        session.close()

    expected = _expected_batch_gather(x, indices, dtype)
    np.testing.assert_array_equal(actual, expected)


def test_batch_gather_generated_cpu_runtime_rejects_oob_index(tmp_path):
    spec = _trace("float32")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "batch_gather_oob_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input("float32")
    indices = _indices()
    indices[0, 0] = 4
    try:
        with pytest.raises(RuntimeError, match="batch_gather index out of bounds"):
            session.run_numpy({"x": x, "indices": indices})
    finally:
        session.close()


def test_batch_gather_generated_sources_accept_int32_indices_and_cuda_asserts():
    spec = _trace("float32", "int32")
    lowered, _ = PassManager().run(spec.ir)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]
    cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

    assert "const int32_t* DINO_RESTRICT index" in cpu_source
    assert "const int32_t* DINO_RESTRICT index" in cuda_source
    assert "#include <assert.h>" in cuda_source
    assert "assert(batch >= 0 && batch < 2 && selected_index >= 0 && selected_index < 4);" in cuda_source
    assert "const int64_t input_idx = batch * 12 + selected_index * 3 + slice_offset;" in cuda_source


def test_batch_gather_frontend_rejects_dynamic_bad_shapes_and_dtypes():
    with pytest.raises(ValueError, match="only static input and index shapes"):
        dml.trace(
            BatchGatherModule(),
            inputs={"x": dml.TensorSpec([Dim("b", 1, 4), 4, 3]), "indices": dml.TensorSpec([2, 3], "int64")},
        )
    with pytest.raises(ValueError, match="only static input and index shapes"):
        dml.trace(
            BatchGatherModule(),
            inputs={"x": dml.TensorSpec([2, 4, 3]), "indices": dml.TensorSpec([Dim("b", 1, 4), 3], "int64")},
        )
    with pytest.raises(ValueError, match="input rank 1 must be at least 2"):
        _trace("float32", x_shape=(4,), index_shape=(4, 2))
    with pytest.raises(ValueError, match="indices rank 3 must be 2"):
        _trace("float32", x_shape=(2, 4, 3), index_shape=(2, 3, 1))
    with pytest.raises(ValueError, match="batch size mismatch"):
        _trace("float32", x_shape=(2, 4, 3), index_shape=(3, 2))
    with pytest.raises(ValueError, match="batch_gather does not support dtype int64"):
        _trace("int64")
    with pytest.raises(ValueError, match="indices must have dtype int64 or int32, got bool"):
        _trace("float32", "bool")
    with pytest.raises(ValueError, match="indices must have dtype int64 or int32, got float32"):
        _trace("float32", "float32")


def test_batch_gather_validation_rejects_dynamic_shape_specs_bad_shape_and_dtype():
    spec = _trace("float32")
    spec.ir["inputs"][0]["shape_spec"] = [Dim("b", 1, 2).to_json(), 4, 3]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [Dim("b", 1, 2).to_json(), 4, 3]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    index_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "indices")
    index_tensor["shape"] = [3, 3]
    index_tensor["shape_spec"] = [3, 3]
    index_tensor["layout"]["strides"] = [3, 1]
    with pytest.raises(ValidationError, match="batch size mismatch"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["outputs"][0]["shape"] = [2, 2, 3]
    spec.ir["outputs"][0]["shape_spec"] = [2, 2, 3]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 2, 3]
    output_tensor["shape_spec"] = [2, 2, 3]
    output_tensor["layout"]["strides"] = [6, 3, 1]
    with pytest.raises(ValidationError, match=r"expected \[2, 3, 3\]"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["outputs"][0]["dtype"] = "bool"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "bool"
    with pytest.raises(ValidationError, match="expected float32"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    index_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "indices")
    index_tensor["dtype"] = "float32"
    with pytest.raises(ValidationError, match="indices must have dtype int64 or int32"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["dtype"] = "int64"
    with pytest.raises(ValidationError, match="batch_gather does not support dtype int64"):
        validate_ir(spec.ir)
