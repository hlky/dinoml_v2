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


class DynamicSliceModule(dml.Module):
    def __init__(self, start_indices=(0, 1, 0), slice_sizes=(2, 2, 2)):
        self.start_indices = start_indices
        self.slice_sizes = slice_sizes

    def forward(self, x):
        return dml.ops.output(dml.ops.dynamic_slice(x, self.start_indices, self.slice_sizes), "out")


def _trace(dtype="float32", start_indices=(0, 1, 0), slice_sizes=(2, 2, 2), shape=(3, 4, 2)):
    return dml.trace(
        DynamicSliceModule(start_indices, slice_sizes),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name=f"dynamic_slice_{dtype}",
    )


def _storage_roundtrip(value, dtype):
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return np.asarray(value, dtype=np.bool_ if dtype == "bool" else np.float32)


def _input(dtype):
    if dtype == "bool":
        return (np.arange(24).reshape(3, 4, 2) % 3) == 0
    return np.arange(24, dtype=np.float32).reshape(3, 4, 2)


def _expected_slice(x, dtype, start_indices=(0, 1, 0), slice_sizes=(2, 2, 2)):
    slices = tuple(slice(start, start + size) for start, size in zip(start_indices, slice_sizes))
    return _storage_roundtrip(x[slices].copy(), dtype)


def test_dynamic_slice_frontend_ir_normalizes_attrs_and_shape():
    spec = _trace("float32", start_indices=(1, 0, 0), slice_sizes=(2, 3, 1))

    assert spec.ir["outputs"][0]["shape"] == [2, 3, 1]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 3, 1]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "dynamic_slice"
    assert node["inputs"] == ["x"]
    assert node["attrs"] == {"start_indices": [1, 0, 0], "slice_sizes": [2, 3, 1]}


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("float32", np.float32),
        ("float16", np.float16),
        ("bfloat16", np.float32),
        ("bool", np.bool_),
    ],
)
def test_cpu_reference_dynamic_slice(dtype, expected_dtype):
    spec = _trace(dtype)
    x = _input(dtype)

    actual = execute_cpu(spec, {"x": x})["out"]

    expected = _expected_slice(x, dtype)
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "bool"])
def test_dynamic_slice_generated_cpu_source_and_runtime(tmp_path, dtype):
    spec = _trace(dtype)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["dynamic_slice"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int dynamic_slice_" in cpu_source
    assert "input_idx += (coord + 0) * 8" in cpu_source
    assert "input_idx += (coord + 1) * 2" in cpu_source
    assert "input_idx += (coord + 0) * 1" in cpu_source
    assert "y[idx] = x[input_idx]" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x" in cpu_source
        assert "float* DINO_RESTRICT y" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"dynamic_slice_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input(dtype)
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    expected = _expected_slice(x, dtype)
    np.testing.assert_array_equal(actual, expected)


def test_dynamic_slice_generated_cuda_source_supports_reduced_precision_and_bool():
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
        assert "dynamic_slice_" in cuda_source
        assert "input_idx += (coord + 0) * 8" in cuda_source
        assert "input_idx += (coord + 1) * 2" in cuda_source
        assert "input_idx += (coord + 0) * 1" in cuda_source
        assert "y[idx] = x[input_idx]" in cuda_source


def test_dynamic_slice_frontend_rejects_dynamic_bad_attrs_and_unsupported_dtype():
    class DynamicShapeSlice(dml.Module):
        def forward(self, x):
            return dml.ops.dynamic_slice(x, start_indices=(0, 0), slice_sizes=(1, 3))

    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(DynamicShapeSlice(), inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3])})
    with pytest.raises(ValueError, match="start_indices length"):
        _trace("float32", start_indices=(0, 1), slice_sizes=(2, 2, 2))
    with pytest.raises(ValueError, match="slice_sizes length"):
        _trace("float32", start_indices=(0, 1, 0), slice_sizes=(2, 2))
    with pytest.raises(ValueError, match="start_indices must contain only integers"):
        _trace("float32", start_indices=(0, True, 0), slice_sizes=(2, 2, 2))
    with pytest.raises(ValueError, match="slice_sizes must contain only integers"):
        _trace("float32", start_indices=(0, 1, 0), slice_sizes=(2, "2", 2))
    with pytest.raises(ValueError, match="non-negative"):
        _trace("float32", start_indices=(0, -1, 0), slice_sizes=(2, 2, 2))
    with pytest.raises(ValueError, match="positive"):
        _trace("float32", start_indices=(0, 1, 0), slice_sizes=(2, 0, 2))
    with pytest.raises(ValueError, match="exceeds input dim"):
        _trace("float32", start_indices=(1, 3, 0), slice_sizes=(2, 2, 2))
    with pytest.raises(ValueError, match="does not support dtype int64"):
        _trace("int64")


def test_dynamic_slice_validation_rejects_dynamic_shape_specs_bad_attrs_shape_and_dtype():
    spec = _trace("float32")
    spec.ir["inputs"][0]["shape_spec"] = [Dim("n", 1, 3).to_json(), 4, 2]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [Dim("n", 1, 3).to_json(), 4, 2]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["nodes"][0]["attrs"]["start_indices"] = [0, 1]
    with pytest.raises(ValidationError, match="start_indices length"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["nodes"][0]["attrs"]["slice_sizes"] = [2, 2]
    with pytest.raises(ValidationError, match="slice_sizes length"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["nodes"][0]["attrs"]["start_indices"] = [0, False, 0]
    with pytest.raises(ValidationError, match="start_indices must contain only integers"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["nodes"][0]["attrs"]["slice_sizes"] = [2, 0, 2]
    with pytest.raises(ValidationError, match="positive"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["nodes"][0]["attrs"]["start_indices"] = [1, 3, 0]
    with pytest.raises(ValidationError, match="exceeds input dim"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["outputs"][0]["shape"] = [2, 3, 2]
    spec.ir["outputs"][0]["shape_spec"] = [2, 3, 2]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 3, 2]
    output_tensor["shape_spec"] = [2, 3, 2]
    output_tensor["layout"]["strides"] = [6, 2, 1]
    with pytest.raises(ValidationError, match=r"expected \[2, 2, 2\]"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["outputs"][0]["dtype"] = "bool"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "bool"
    with pytest.raises(ValidationError, match="expected float32"):
        validate_ir(spec.ir)
