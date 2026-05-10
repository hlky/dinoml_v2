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


class StackModule(dml.Module):
    def __init__(self, dim=0):
        self.dim = dim

    def forward(self, x, y, z):
        return dml.ops.output(dml.ops.stack([x, y, z], dim=self.dim), "out")


def _trace_stack(dtype="float32", dim=1, shapes=([2, 3, 4], [2, 3, 4], [2, 3, 4])):
    inputs = {
        "x": dml.TensorSpec(shapes[0], dtype),
        "y": dml.TensorSpec(shapes[1], dtype),
        "z": dml.TensorSpec(shapes[2], dtype),
    }
    return dml.trace(StackModule(dim), inputs=inputs, name=f"stack_{dtype}")


def _storage_roundtrip(value, dtype):
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return np.asarray(value, dtype=np.bool_ if dtype == "bool" else np.float32)


def _inputs(dtype):
    if dtype == "bool":
        return {
            "x": np.array([[True, False], [False, True]], dtype=np.bool_),
            "y": np.array([[False, True], [True, False]], dtype=np.bool_),
            "z": np.array([[True, True], [False, False]], dtype=np.bool_),
        }
    return {
        "x": np.arange(4, dtype=np.float32).reshape(2, 2),
        "y": (10 + np.arange(4, dtype=np.float32)).reshape(2, 2),
        "z": (20 + np.arange(4, dtype=np.float32)).reshape(2, 2),
    }


def test_stack_frontend_ir_normalizes_negative_dim():
    spec = _trace_stack("float32", dim=-2)

    assert spec.ir["outputs"][0]["shape"] == [2, 3, 3, 4]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 3, 3, 4]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "stack"
    assert node["inputs"] == ["x", "y", "z"]
    assert node["attrs"] == {"dim": 2}


def test_stack_frontend_allows_insert_at_rank():
    spec = _trace_stack("float32", dim=3)

    assert spec.ir["outputs"][0]["shape"] == [2, 3, 4, 3]
    assert spec.ir["nodes"][0]["attrs"] == {"dim": 3}


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("float32", np.float32),
        ("float16", np.float16),
        ("bfloat16", np.float32),
        ("bool", np.bool_),
    ],
)
def test_cpu_reference_stack(dtype, expected_dtype):
    spec = _trace_stack(dtype, dim=1, shapes=([2, 2], [2, 2], [2, 2]))
    inputs = _inputs(dtype)

    actual = execute_cpu(spec, inputs)["out"]

    expected = _storage_roundtrip(np.stack([inputs["x"], inputs["y"], inputs["z"]], axis=1), dtype)
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "bool"])
def test_stack_generated_cpu_source_and_runtime(tmp_path, dtype):
    spec = _trace_stack(dtype, dim=1, shapes=([2, 2], [2, 2], [2, 2]))
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["stack"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int stack_" in cpu_source
    assert "stack_idx" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x0" in cpu_source
        assert "const float* DINO_RESTRICT x1" in cpu_source
        assert "float* DINO_RESTRICT y" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"stack_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    inputs = _inputs(dtype)
    try:
        actual = session.run_numpy(inputs)["out"]
    finally:
        session.close()

    expected = _storage_roundtrip(np.stack([inputs["x"], inputs["y"], inputs["z"]], axis=1), dtype)
    np.testing.assert_array_equal(actual, expected)


def test_stack_generated_cuda_source_supports_reduced_precision_and_bool():
    for dtype, pointer_type in (
        ("float16", "const half* DINO_RESTRICT x0"),
        ("bfloat16", "const __nv_bfloat16* DINO_RESTRICT x0"),
        ("bool", "const bool* DINO_RESTRICT x0"),
    ):
        spec = _trace_stack(dtype, dim=1, shapes=([2, 2], [2, 2], [2, 2]))
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert "stack_" in cuda_source
        assert "stack_idx" in cuda_source
        assert "y[idx] = x" in cuda_source


def test_stack_frontend_rejects_invalid_inputs_and_dynamic_shapes():
    class DynamicStack(dml.Module):
        def forward(self, x, y):
            return dml.ops.stack([x, y], dim=0)

    with pytest.raises(ValueError, match="non-empty sequence"):
        dml.ops.stack([], dim=0)
    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(
            DynamicStack(),
            inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3]), "y": dml.TensorSpec([2, 3])},
        )
    with pytest.raises(ValueError, match="out of range"):
        _trace_stack("float32", dim=4)
    with pytest.raises(ValueError, match="does not match"):
        _trace_stack("float32", dim=0, shapes=([2, 3], [2, 3, 1], [2, 3]))
    with pytest.raises(ValueError, match="does not match"):
        _trace_stack("float32", dim=1, shapes=([2, 3], [2, 4], [2, 3]))


def test_stack_validation_rejects_dynamic_shape_spec_bad_dim_shape_and_dtype():
    spec = _trace_stack("float32", dim=1)
    spec.ir["inputs"][0]["shape_spec"] = [Dim("n", 1, 2).to_json(), 3, 4]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [Dim("n", 1, 2).to_json(), 3, 4]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace_stack("float32", dim=1)
    spec.ir["nodes"][0]["attrs"]["dim"] = 4
    with pytest.raises(ValidationError, match="out of range"):
        validate_ir(spec.ir)

    spec = _trace_stack("float32", dim=1)
    spec.ir["outputs"][0]["shape"] = [2, 2, 3, 4]
    spec.ir["outputs"][0]["shape_spec"] = [2, 2, 3, 4]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 2, 3, 4]
    output_tensor["shape_spec"] = [2, 2, 3, 4]
    output_tensor["layout"]["strides"] = [24, 12, 4, 1]
    with pytest.raises(ValidationError, match=r"expected \[2, 3, 3, 4\]"):
        validate_ir(spec.ir)

    spec = _trace_stack("float32", dim=1)
    y_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "y")
    y_tensor["dtype"] = "bool"
    with pytest.raises(ValidationError, match="mismatched input dtypes"):
        validate_ir(spec.ir)
