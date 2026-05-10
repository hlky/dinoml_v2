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


class ExpandModule(dml.Module):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return dml.ops.output(dml.ops.expand(x, self.shape), "out")


def _trace_expand(input_shape=(1, 3), output_shape=(2, 3), dtype="float32"):
    return dml.trace(ExpandModule(output_shape), inputs={"x": dml.TensorSpec(input_shape, dtype)}, name=f"expand_{dtype}")


def _storage_roundtrip(value, dtype):
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return np.asarray(value, dtype=np.bool_ if dtype == "bool" else np.float32)


def test_expand_frontend_ir_preserves_shape_spec_and_dtype():
    spec = _trace_expand([1, 3], [2, -1], "float32")

    assert spec.ir["outputs"][0]["shape"] == [2, 3]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 3]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "expand"
    assert node["attrs"] == {"shape": [2, -1]}


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("float32", np.float32),
        ("float16", np.float16),
        ("bfloat16", np.float32),
        ("bool", np.bool_),
    ],
)
def test_cpu_reference_expand(dtype, expected_dtype):
    spec = _trace_expand([1, 3], [2, 3], dtype)
    x = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    if dtype == "bool":
        x = np.array([[False, True, True]], dtype=np.bool_)

    actual = execute_cpu(spec, {"x": x})["out"]

    expected = _storage_roundtrip(np.broadcast_to(x, [2, 3]).copy(), dtype)
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "bool"])
def test_expand_generated_cpu_source_and_runtime(tmp_path, dtype):
    spec = _trace_expand([1, 3], [2, 3], dtype)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["expand"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int expand_" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x" in cpu_source
        assert "float* DINO_RESTRICT y" in cpu_source
        assert "input_idx += coord * input_stride" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"expand_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    if dtype == "bool":
        x = np.array([[False, True, True]], dtype=np.bool_)
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    expected = _storage_roundtrip(np.broadcast_to(x, [2, 3]).copy(), dtype)
    np.testing.assert_array_equal(actual, expected)


def test_expand_generated_cuda_source_supports_reduced_precision_and_bool():
    for dtype, pointer_type in (
        ("float16", "const half* DINO_RESTRICT x"),
        ("bfloat16", "const __nv_bfloat16* DINO_RESTRICT x"),
        ("bool", "const bool* DINO_RESTRICT x"),
    ):
        spec = _trace_expand([1, 3], [2, 3], dtype)
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert "expand_" in cuda_source
        assert "y[idx] = x[" in cuda_source


def test_expand_frontend_rejects_dynamic_bad_shape_and_unbroadcastable():
    class DynamicExpand(dml.Module):
        def forward(self, x):
            return dml.ops.expand(x, [4, 3])

    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(DynamicExpand(), inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3])})
    with pytest.raises(ValueError, match="positive or -1"):
        _trace_expand([1, 3], [2, 0])
    with pytest.raises(ValueError, match="new leading"):
        _trace_expand([3], [-1, 3])
    with pytest.raises(ValueError, match="not broadcastable"):
        _trace_expand([2, 3], [2, 4])


def test_expand_validation_rejects_bad_shape_and_dtype():
    spec = _trace_expand([1, 3], [2, 3], "float32")
    spec.ir["nodes"][0]["attrs"]["shape"] = [2, 0]
    with pytest.raises(ValidationError, match="positive or -1"):
        validate_ir(spec.ir)

    spec = _trace_expand([1, 3], [2, 3], "float32")
    spec.ir["outputs"][0]["dtype"] = "int64"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "int64"
    with pytest.raises(ValidationError, match="expected float32"):
        validate_ir(spec.ir)
