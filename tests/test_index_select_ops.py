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


class IndexSelectModule(dml.Module):
    def __init__(self, dim=-2, indices=(3, 1, 1)):
        self.dim = dim
        self.indices = indices

    def forward(self, x):
        return dml.ops.output(dml.ops.index_select(x, self.dim, self.indices), "out")


def _trace(dtype="float32", dim=-2, indices=(3, 1, 1), shape=(2, 4, 3)):
    return dml.trace(
        IndexSelectModule(dim, indices),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name=f"index_select_{dtype}",
    )


def _storage_roundtrip(value, dtype):
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return np.asarray(value, dtype=np.bool_ if dtype == "bool" else np.float32)


def _input(dtype):
    if dtype == "bool":
        return (np.arange(24).reshape(2, 4, 3) % 3) == 0
    return np.arange(24, dtype=np.float32).reshape(2, 4, 3)


def _expected_index_select(x, dtype, dim=1, indices=(3, 1, 1)):
    return _storage_roundtrip(np.take(x, indices, axis=dim).copy(), dtype)


def test_index_select_frontend_ir_normalizes_attrs_shape_and_dtype():
    spec = _trace("float32", dim=-2, indices=(3, 1, 1))

    assert spec.ir["outputs"][0]["shape"] == [2, 3, 3]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 3, 3]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "index_select"
    assert node["inputs"] == ["x"]
    assert node["attrs"] == {"dim": 1, "indices": [3, 1, 1]}


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("float32", np.float32),
        ("float16", np.float16),
        ("bfloat16", np.float32),
        ("bool", np.bool_),
    ],
)
def test_cpu_reference_index_select(dtype, expected_dtype):
    spec = _trace(dtype)
    x = _input(dtype)

    actual = execute_cpu(spec, {"x": x})["out"]

    expected = _expected_index_select(x, dtype)
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "bool"])
def test_index_select_generated_cpu_source_and_runtime(tmp_path, dtype):
    spec = _trace(dtype)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["index_select"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int index_select_" in cpu_source
    assert "selected_index = (coord == 0 ? 3 : (coord == 1 ? 1 : 1));" in cpu_source
    assert "input_idx += selected_index * 3;" in cpu_source
    assert "y[idx] = x[input_idx]" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x" in cpu_source
        assert "float* DINO_RESTRICT y" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"index_select_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input(dtype)
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    expected = _expected_index_select(x, dtype)
    np.testing.assert_array_equal(actual, expected)


def test_index_select_generated_cuda_source_supports_reduced_precision_and_bool():
    for dtype, pointer_type in (
        ("float16", "const half* DINO_RESTRICT x"),
        ("bfloat16", "const __nv_bfloat16* DINO_RESTRICT x"),
        ("bool", "const bool* DINO_RESTRICT x"),
    ):
        spec = _trace(dtype)
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert "index_select_" in cuda_source
        assert "selected_index = (coord == 0 ? 3 : (coord == 1 ? 1 : 1));" in cuda_source
        assert "input_idx += selected_index * 3;" in cuda_source
        assert "y[idx] = x[input_idx]" in cuda_source


def test_index_select_frontend_rejects_dynamic_bad_attrs_and_unsupported_dtype():
    class DynamicShapeIndexSelect(dml.Module):
        def forward(self, x):
            return dml.ops.index_select(x, 1, (0, 1))

    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(DynamicShapeIndexSelect(), inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3])})
    with pytest.raises(ValueError, match="dim must be an integer"):
        _trace("float32", dim=True, indices=(0, 1))
    with pytest.raises(ValueError, match="out of range"):
        _trace("float32", dim=3, indices=(0, 1))
    with pytest.raises(ValueError, match="non-empty sequence"):
        _trace("float32", dim=1, indices=())
    with pytest.raises(ValueError, match="non-empty sequence"):
        _trace("float32", dim=1, indices=2)
    with pytest.raises(ValueError, match="non-bool integers"):
        _trace("float32", dim=1, indices=(0, True))
    with pytest.raises(ValueError, match="non-bool integers"):
        _trace("float32", dim=1, indices=(0, "1"))
    with pytest.raises(ValueError, match="out of bounds"):
        _trace("float32", dim=1, indices=(0, -1))
    with pytest.raises(ValueError, match="out of bounds"):
        _trace("float32", dim=1, indices=(0, 4))
    with pytest.raises(ValueError, match="does not support dtype int64"):
        _trace("int64")


def test_index_select_validation_rejects_dynamic_shape_specs_bad_attrs_shape_and_dtype():
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
    spec.ir["nodes"][0]["attrs"]["indices"] = []
    with pytest.raises(ValidationError, match="non-empty"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["nodes"][0]["attrs"]["indices"] = [0, False]
    with pytest.raises(ValidationError, match="non-bool integers"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["nodes"][0]["attrs"]["indices"] = [0, 4]
    with pytest.raises(ValidationError, match="out of bounds"):
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
