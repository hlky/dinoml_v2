import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.layout import dense_layout
from dinoml.lowering.ops import render_generated_kernels
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError
from dinoml.runtime import load
from dinoml.shapes import Dim


class SliceScatterModule(dml.Module):
    def __init__(self, start_indices=(0, 1, 0)):
        self.start_indices = start_indices

    def forward(self, x, update):
        return dml.ops.output(dml.ops.slice_scatter(x, update, self.start_indices), "out")


class SliceReshapeScatterModule(dml.Module):
    def __init__(self, start_indices=(0, 1, 0), slice_shape=(2, 2, 2)):
        self.start_indices = start_indices
        self.slice_shape = slice_shape

    def forward(self, x, update):
        return dml.ops.output(dml.ops.slice_reshape_scatter(x, update, self.start_indices, self.slice_shape), "out")


def _trace(dtype="float32", start_indices=(0, 1, 0), x_shape=(3, 4, 2), update_shape=(2, 2, 2)):
    return dml.trace(
        SliceScatterModule(start_indices),
        inputs={"x": dml.TensorSpec(x_shape, dtype), "update": dml.TensorSpec(update_shape, dtype)},
        name=f"slice_scatter_{dtype}",
    )


def _trace_slice_reshape_scatter(
    dtype="float32",
    start_indices=(0, 1, 0),
    x_shape=(3, 4, 2),
    update_shape=(8,),
    slice_shape=(2, 2, 2),
):
    return dml.trace(
        SliceReshapeScatterModule(start_indices, slice_shape),
        inputs={"x": dml.TensorSpec(x_shape, dtype), "update": dml.TensorSpec(update_shape, dtype)},
        name=f"slice_reshape_scatter_{dtype}",
    )


def _storage_roundtrip(value, dtype):
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return np.asarray(value, dtype=np.bool_ if dtype == "bool" else np.float32)


def _input(dtype):
    if dtype == "bool":
        return (np.arange(24).reshape(3, 4, 2) % 3) == 0
    return np.arange(24, dtype=np.float32).reshape(3, 4, 2)


def _update(dtype):
    if dtype == "bool":
        return (np.arange(8).reshape(2, 2, 2) % 2) == 0
    return (100 + np.arange(8, dtype=np.float32)).reshape(2, 2, 2)


def _flat_update(dtype):
    return _update(dtype).reshape(8)


def _expected_scatter(x, update, dtype, start_indices=(0, 1, 0)):
    result = x.copy()
    slices = tuple(slice(start, start + size) for start, size in zip(start_indices, update.shape))
    result[slices] = update
    return _storage_roundtrip(result, dtype)


def _set_tensor_shape(spec, name, shape):
    tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == name)
    tensor["shape"] = list(shape)
    tensor["shape_spec"] = list(shape)
    tensor["layout"] = dense_layout(shape)
    return tensor


def test_slice_scatter_frontend_ir_normalizes_attrs_and_shape():
    spec = _trace("float32", start_indices=(1, 0, 0), update_shape=(2, 3, 1))

    assert spec.ir["outputs"][0]["shape"] == [3, 4, 2]
    assert spec.ir["outputs"][0]["shape_spec"] == [3, 4, 2]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "slice_scatter"
    assert node["inputs"] == ["x", "update"]
    assert node["attrs"] == {"start_indices": [1, 0, 0]}


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("float32", np.float32),
        ("bool", np.bool_),
    ],
)
def test_cpu_reference_slice_scatter(dtype, expected_dtype):
    spec = _trace(dtype)
    x = _input(dtype)
    update = _update(dtype)

    actual = execute_cpu(spec, {"x": x, "update": update})["out"]

    expected = _expected_scatter(x, update, dtype)
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["float32", "bool"])
def test_slice_scatter_generated_cpu_source_and_runtime(tmp_path, dtype):
    spec = _trace(dtype)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["slice_scatter"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int slice_scatter_" in cpu_source
    assert "const float* DINO_RESTRICT update" in cpu_source if dtype == "float32" else "const bool* DINO_RESTRICT update" in cpu_source
    assert "in_slice = in_slice && coord >= 0 && coord < 2" in cpu_source
    assert "in_slice = in_slice && coord >= 1 && coord < 3" in cpu_source
    assert "update_idx += (coord - 1) * 2" in cpu_source
    assert "y[idx] = in_slice ? update[update_idx] : x[idx]" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"slice_scatter_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input(dtype)
    update = _update(dtype)
    try:
        actual = session.run_numpy({"x": x, "update": update})["out"]
    finally:
        session.close()

    expected = _expected_scatter(x, update, dtype)
    np.testing.assert_array_equal(actual, expected)


def test_slice_scatter_generated_cuda_source_supports_reduced_precision_and_bool():
    for dtype, pointer_type in (
        ("float16", "const half* DINO_RESTRICT update"),
        ("bfloat16", "const __nv_bfloat16* DINO_RESTRICT update"),
        ("bool", "const bool* DINO_RESTRICT update"),
    ):
        spec = _trace(dtype)
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert "slice_scatter_" in cuda_source
        assert "in_slice = in_slice && coord >= 0 && coord < 2" in cuda_source
        assert "in_slice = in_slice && coord >= 1 && coord < 3" in cuda_source
        assert "y[idx] = in_slice ? update[update_idx] : x[idx]" in cuda_source


def test_slice_reshape_scatter_frontend_ir_emits_reshape_view_and_slice_scatter():
    spec = _trace_slice_reshape_scatter("float32", start_indices=(1, 0, 0), update_shape=(6,), slice_shape=(2, 3, 1))

    assert spec.ir["outputs"][0]["shape"] == [3, 4, 2]
    assert spec.ir["outputs"][0]["shape_spec"] == [3, 4, 2]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    views = spec.ir["metadata"]["views"]["views"]
    assert len(views) == 1
    assert views[0]["source"] == "update"
    assert views[0]["transform"] == "reshape"
    assert views[0]["shape"] == [2, 3, 1]
    node = spec.ir["nodes"][0]
    assert node["op"] == "slice_scatter"
    assert node["inputs"] == ["x", views[0]["tensor"]]
    assert node["attrs"] == {"start_indices": [1, 0, 0]}


@pytest.mark.parametrize("dtype", ["float32", "bool"])
def test_slice_reshape_scatter_generated_cpu_runtime(tmp_path, dtype):
    spec = _trace_slice_reshape_scatter(dtype)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["slice_scatter"]
    assert lowered["metadata"]["memory_plan"]["views"]["views"][0]["transform"] == "reshape"

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"slice_reshape_scatter_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input(dtype)
    update = _flat_update(dtype)
    try:
        actual = session.run_numpy({"x": x, "update": update})["out"]
    finally:
        session.close()

    expected = _expected_scatter(x, update.reshape(2, 2, 2), dtype)
    np.testing.assert_array_equal(actual, expected)


def test_slice_reshape_scatter_frontend_rejects_bad_slice_shape_dynamic_dtype_and_bounds():
    class DynamicInputShapeScatter(dml.Module):
        def forward(self, x, update):
            return dml.ops.slice_reshape_scatter(x, update, start_indices=(0, 0), slice_shape=(1, 3))

    class DynamicUpdateShapeScatter(dml.Module):
        def forward(self, x, update):
            return dml.ops.slice_reshape_scatter(x, update, start_indices=(0, 0), slice_shape=(1, 3))

    with pytest.raises(ValueError, match="slice_shape rank"):
        _trace_slice_reshape_scatter("float32", slice_shape=(2, 4))
    with pytest.raises(ValueError, match="preserve update element count"):
        _trace_slice_reshape_scatter("float32", slice_shape=(2, 2, 3))
    with pytest.raises(ValueError, match="dimensions must be positive"):
        _trace_slice_reshape_scatter("float32", slice_shape=(2, 0, 4))
    with pytest.raises(ValueError, match="dimensions must be positive"):
        _trace_slice_reshape_scatter("float32", slice_shape=(2, -1, 4))
    with pytest.raises(ValueError, match="contain only integers"):
        _trace_slice_reshape_scatter("float32", slice_shape=(2, True, 4))
    with pytest.raises(ValueError, match="must be a sequence"):
        _trace_slice_reshape_scatter("float32", slice_shape=8)
    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(
            DynamicInputShapeScatter(),
            inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3]), "update": dml.TensorSpec([3])},
        )
    with pytest.raises(NotImplementedError, match="reshape currently supports only static input shapes"):
        dml.trace(
            DynamicUpdateShapeScatter(),
            inputs={"x": dml.TensorSpec([2, 3]), "update": dml.TensorSpec([Dim("n", 1, 3)])},
        )
    with pytest.raises(ValueError, match="dtype mismatch"):
        dml.trace(
            SliceReshapeScatterModule(),
            inputs={"x": dml.TensorSpec((3, 4, 2), "float32"), "update": dml.TensorSpec((8,), "bool")},
        )
    with pytest.raises(ValueError, match="exceeds input dim"):
        _trace_slice_reshape_scatter("float32", start_indices=(1, 3, 0))


def test_slice_scatter_frontend_rejects_dynamic_bad_attrs_shape_dtype_and_rank():
    class DynamicShapeScatter(dml.Module):
        def forward(self, x, update):
            return dml.ops.slice_scatter(x, update, start_indices=(0, 0))

    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(
            DynamicShapeScatter(),
            inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3]), "update": dml.TensorSpec([1, 3])},
        )
    with pytest.raises(ValueError, match="dtype mismatch"):
        dml.trace(
            SliceScatterModule(),
            inputs={"x": dml.TensorSpec((3, 4, 2), "float32"), "update": dml.TensorSpec((2, 2, 2), "bool")},
        )
    with pytest.raises(ValueError, match="update rank"):
        _trace("float32", x_shape=(3, 4, 2), update_shape=(2, 2))
    with pytest.raises(ValueError, match="start_indices length"):
        _trace("float32", start_indices=(0, 1))
    with pytest.raises(ValueError, match="start_indices must contain only integers"):
        _trace("float32", start_indices=(0, True, 0))
    with pytest.raises(ValueError, match="start_indices must contain only integers"):
        _trace("float32", start_indices=(0, "1", 0))
    with pytest.raises(ValueError, match="non-negative"):
        _trace("float32", start_indices=(0, -1, 0))
    with pytest.raises(ValueError, match="exceeds input dim"):
        _trace("float32", start_indices=(1, 3, 0))
    with pytest.raises(ValueError, match="does not support dtype int64"):
        _trace("int64")


def test_slice_scatter_validation_rejects_dynamic_shape_specs_bad_attrs_shape_and_dtype():
    spec = _trace("float32")
    spec.ir["inputs"][0]["shape_spec"] = [Dim("n", 1, 3).to_json(), 4, 2]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [Dim("n", 1, 3).to_json(), 4, 2]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    update_tensor = _set_tensor_shape(spec, "update", [2, 2])
    update_tensor["nbytes"] = 16
    with pytest.raises(ValidationError, match="update rank"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["nodes"][0]["attrs"]["start_indices"] = [0, 1]
    with pytest.raises(ValidationError, match="start_indices length"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["nodes"][0]["attrs"]["start_indices"] = [0, False, 0]
    with pytest.raises(ValidationError, match="start_indices must contain only integers"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["nodes"][0]["attrs"]["start_indices"] = [0, -1, 0]
    with pytest.raises(ValidationError, match="non-negative"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["nodes"][0]["attrs"]["start_indices"] = [1, 3, 0]
    with pytest.raises(ValidationError, match="exceeds input dim"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    spec.ir["outputs"][0]["shape"] = [2, 4, 2]
    spec.ir["outputs"][0]["shape_spec"] = [2, 4, 2]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 4, 2]
    output_tensor["shape_spec"] = [2, 4, 2]
    output_tensor["layout"] = dense_layout([2, 4, 2])
    with pytest.raises(ValidationError, match=r"expected \[3, 4, 2\]"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    update_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "update")
    update_tensor["dtype"] = "bool"
    with pytest.raises(ValidationError, match="mismatched input dtypes"):
        validate_ir(spec.ir)

    spec = _trace("float32")
    for tensor in spec.ir["inputs"]:
        tensor["dtype"] = "int64"
    for tensor in spec.ir["outputs"]:
        tensor["dtype"] = "int64"
    for tensor in spec.ir["tensors"]:
        tensor["dtype"] = "int64"
    with pytest.raises(ValidationError, match="does not support dtype int64"):
        validate_ir(spec.ir)
