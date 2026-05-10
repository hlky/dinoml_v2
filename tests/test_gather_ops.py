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


class GatherModule(dml.Module):
    def __init__(self, dim=-2):
        self.dim = dim

    def forward(self, x, index):
        return dml.ops.output(dml.ops.gather(x, self.dim, index), "out")


def _trace(dtype="float32", index_dtype="int64", dim=-2, x_shape=(2, 4, 3), index_shape=(2, 3, 3)):
    return dml.trace(
        GatherModule(dim),
        inputs={"x": dml.TensorSpec(x_shape, dtype), "index": dml.TensorSpec(index_shape, index_dtype)},
        name=f"gather_{dtype}_{index_dtype}",
    )


def _storage_roundtrip(value, dtype):
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return np.asarray(value, dtype=np.bool_ if dtype == "bool" else np.float32)


def _input(dtype):
    if dtype == "bool":
        return (np.arange(24).reshape(2, 4, 3) % 3) == 0
    return np.arange(24, dtype=np.float32).reshape(2, 4, 3)


def _index(index_dtype="int64"):
    return np.array(
        [
            [[3, 0, 1], [2, 1, 0], [0, 3, 2]],
            [[1, 2, 3], [0, 0, 1], [3, 2, 0]],
        ],
        dtype=np.int64 if index_dtype == "int64" else np.int32,
    )


def _expected_gather(x, index, dtype, dim=1):
    result = np.empty(index.shape, dtype=x.dtype)
    for output_coord in np.ndindex(index.shape):
        input_coord = list(output_coord)
        input_coord[dim] = int(index[output_coord])
        result[output_coord] = x[tuple(input_coord)]
    return _storage_roundtrip(result, dtype)


def test_gather_frontend_ir_normalizes_attrs_shape_and_dtype():
    spec = _trace("float32", "int64", dim=-2)

    assert spec.ir["outputs"][0]["shape"] == [2, 3, 3]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 3, 3]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "gather"
    assert node["inputs"] == ["x", "index"]
    assert node["attrs"] == {"dim": 1}


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("float32", np.float32),
        ("float16", np.float16),
        ("bfloat16", np.float32),
        ("bool", np.bool_),
    ],
)
def test_cpu_reference_gather(dtype, expected_dtype):
    spec = _trace(dtype)
    x = _input(dtype)
    index = _index()

    actual = execute_cpu(spec, {"x": x, "index": index})["out"]

    expected = _expected_gather(x, index, dtype)
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


def test_cpu_reference_gather_accepts_int32_index():
    spec = _trace("float32", "int32")
    x = _input("float32")
    index = _index("int32")

    actual = execute_cpu(spec, {"x": x, "index": index})["out"]

    np.testing.assert_array_equal(actual, _expected_gather(x, index, "float32"))


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "bool"])
def test_gather_generated_cpu_source_and_runtime(tmp_path, dtype):
    spec = _trace(dtype)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["gather"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int gather_" in cpu_source
    assert "const int64_t* DINO_RESTRICT index" in cpu_source
    assert "selected_index = static_cast<int64_t>(index[idx]);" in cpu_source
    assert "return dino_runtime_fail(\"gather index out of bounds\");" in cpu_source
    assert "input_idx += selected_index * 3;" in cpu_source
    assert "y[idx] = x[input_idx]" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x" in cpu_source
        assert "float* DINO_RESTRICT y" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"gather_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input(dtype)
    index = _index()
    try:
        actual = session.run_numpy({"x": x, "index": index})["out"]
    finally:
        session.close()

    expected = _expected_gather(x, index, dtype)
    np.testing.assert_array_equal(actual, expected)


def test_gather_generated_cpu_runtime_rejects_oob_index(tmp_path):
    spec = _trace("float32")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "gather_oob_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input("float32")
    index = _index()
    index[0, 0, 0] = 4
    try:
        with pytest.raises(RuntimeError, match="gather index out of bounds"):
            session.run_numpy({"x": x, "index": index})
    finally:
        session.close()


def test_gather_generated_cpu_source_accepts_int32_index():
    spec = _trace("float32", "int32")
    lowered, _ = PassManager().run(spec.ir)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "const int32_t* DINO_RESTRICT index" in cpu_source


def test_gather_generated_cuda_source_supports_reduced_precision_bool_and_index_types():
    for dtype, index_dtype, pointer_type, index_pointer_type in (
        ("float16", "int64", "const half* DINO_RESTRICT x", "const int64_t* DINO_RESTRICT index"),
        ("bfloat16", "int32", "const __nv_bfloat16* DINO_RESTRICT x", "const int32_t* DINO_RESTRICT index"),
        ("bool", "int64", "const bool* DINO_RESTRICT x", "const int64_t* DINO_RESTRICT index"),
    ):
        spec = _trace(dtype, index_dtype)
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert index_pointer_type in cuda_source
        assert "gather_" in cuda_source
        assert "#include <assert.h>" in cuda_source
        assert "assert(selected_index >= 0 && selected_index < 4);" in cuda_source
        assert "input_idx += selected_index * 3;" in cuda_source
        assert "y[idx] = x[input_idx]" in cuda_source


def test_gather_frontend_rejects_dynamic_bad_attrs_shape_and_dtype():
    class DynamicShapeGather(dml.Module):
        def forward(self, x, index):
            return dml.ops.gather(x, 1, index)

    with pytest.raises(ValueError, match="only static input and index shapes"):
        dml.trace(
            DynamicShapeGather(),
            inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3]), "index": dml.TensorSpec([2, 3], "int64")},
        )
    with pytest.raises(ValueError, match="only static input and index shapes"):
        dml.trace(
            DynamicShapeGather(),
            inputs={"x": dml.TensorSpec([2, 3]), "index": dml.TensorSpec([Dim("n", 1, 2), 3], "int64")},
        )
    with pytest.raises(ValueError, match="dim must be an integer"):
        _trace("float32", dim=True)
    with pytest.raises(ValueError, match="out of range"):
        _trace("float32", dim=3)
    with pytest.raises(ValueError, match="index rank 2 must match input rank 3"):
        _trace("float32", index_shape=(2, 3))
    with pytest.raises(ValueError, match="index dim 0 size 3 exceeds input dim 2"):
        _trace("float32", dim=1, index_shape=(3, 2, 3))
    with pytest.raises(ValueError, match="does not support dtype int64"):
        _trace("int64")
    with pytest.raises(ValueError, match="index must have dtype int64 or int32, got bool"):
        _trace("float32", "bool")
    with pytest.raises(ValueError, match="index must have dtype int64 or int32, got float32"):
        _trace("float32", "float32")


def test_gather_validation_rejects_dynamic_shape_specs_bad_attrs_shape_and_dtype():
    spec = _trace("float32")
    spec.ir["inputs"][0]["shape_spec"] = [Dim("n", 1, 2).to_json(), 4, 3]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [Dim("n", 1, 2).to_json(), 4, 3]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["nodes"][0]["attrs"]["dim"] = True
    with pytest.raises(ValidationError, match="dim must be an integer"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    index_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "index")
    index_tensor["shape"] = [2, 3]
    index_tensor["shape_spec"] = [2, 3]
    index_tensor["layout"]["strides"] = [3, 1]
    with pytest.raises(ValidationError, match="index rank 2 must match input rank 3"):
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
    index_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "index")
    index_tensor["dtype"] = "float32"
    with pytest.raises(ValidationError, match="index must have dtype int64 or int32"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["dtype"] = "int64"
    with pytest.raises(ValidationError, match="gather does not support dtype int64"):
        validate_ir(spec.ir)
