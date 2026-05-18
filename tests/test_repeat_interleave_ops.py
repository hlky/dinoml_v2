import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.lowering.ops import render_generated_kernels
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError
from dinoml.runtime import load
from dinoml.shapes import Dim


class RepeatInterleaveModule(dml.Module):
    def __init__(self, repeats, dim):
        self.repeats = repeats
        self.dim = dim

    def forward(self, x):
        return dml.ops.output(dml.ops.repeat_interleave(x, self.repeats, self.dim), "out")


def _trace_repeat_interleave(dtype="float32", repeats=3, dim=-2, shape=(2, 3, 4)):
    return dml.trace(
        RepeatInterleaveModule(repeats, dim),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name=f"repeat_interleave_{dtype}",
    )


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


def test_repeat_interleave_frontend_ir_normalizes_attrs_and_shape():
    spec = _trace_repeat_interleave("float32", repeats=3, dim=-2)

    assert spec.ir["outputs"][0]["shape"] == [2, 9, 4]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 9, 4]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "repeat_interleave"
    assert node["inputs"] == ["x"]
    assert node["attrs"] == {"repeats": 3, "dim": 1}


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("float32", np.float32),
        ("float16", np.float16),
        ("bfloat16", np.float32),
        ("bool", np.bool_),
    ],
)
def test_cpu_reference_repeat_interleave(dtype, expected_dtype):
    spec = _trace_repeat_interleave(dtype, repeats=2, dim=1, shape=(2, 3, 2))
    x = _input(dtype)

    actual = reference_numpy(spec, {"x": x})["out"]

    expected = _storage_roundtrip(np.repeat(x, 2, axis=1).copy(), dtype)
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "bool"])
def test_repeat_interleave_generated_cpu_source_and_runtime(tmp_path, dtype):
    spec = _trace_repeat_interleave(dtype, repeats=2, dim=1, shape=(2, 3, 2))
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["repeat_interleave"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int repeat_interleave_" in cpu_source
    assert "repeat_interleave_repeats = 2" in cpu_source
    assert "coord = coord / repeat_interleave_repeats" in cpu_source
    assert "y[idx] = x[input_idx]" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x" in cpu_source
        assert "float* DINO_RESTRICT y" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"repeat_interleave_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input(dtype)
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    expected = _storage_roundtrip(np.repeat(x, 2, axis=1).copy(), dtype)
    np.testing.assert_array_equal(actual, expected)


def test_repeat_interleave_generated_cuda_source_supports_reduced_precision_and_bool():
    for dtype, pointer_type in (
        ("float16", "const half* DINO_RESTRICT x"),
        ("bfloat16", "const __nv_bfloat16* DINO_RESTRICT x"),
        ("bool", "const bool* DINO_RESTRICT x"),
    ):
        spec = _trace_repeat_interleave(dtype, repeats=2, dim=1, shape=(2, 3, 2))
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert "repeat_interleave_" in cuda_source
        assert "coord = coord / repeat_interleave_repeats" in cuda_source
        assert "y[idx] = x[input_idx]" in cuda_source


def test_repeat_interleave_frontend_rejects_dynamic_bad_attrs_and_unsupported_dtype():
    class DynamicRepeatInterleave(dml.Module):
        def forward(self, x):
            return dml.ops.repeat_interleave(x, repeats=2, dim=0)

    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(DynamicRepeatInterleave(), inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3])})
    with pytest.raises(ValueError, match="out of range"):
        _trace_repeat_interleave("float32", repeats=2, dim=3)
    with pytest.raises(ValueError, match="positive integer scalar"):
        _trace_repeat_interleave("float32", repeats=[2], dim=0)
    with pytest.raises(ValueError, match="positive"):
        _trace_repeat_interleave("float32", repeats=0, dim=0)
    with pytest.raises(ValueError, match="integer"):
        _trace_repeat_interleave("float32", repeats=2, dim="1")
    with pytest.raises(ValueError, match="does not support dtype int64"):
        _trace_repeat_interleave("int64", repeats=2, dim=0)


def test_repeat_interleave_validation_rejects_dynamic_shape_spec_bad_attrs_shape_and_dtype():
    spec = _trace_repeat_interleave("float32", repeats=3, dim=1)
    spec.ir["inputs"][0]["shape_spec"] = [Dim("n", 1, 2).to_json(), 3, 4]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [Dim("n", 1, 2).to_json(), 3, 4]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace_repeat_interleave("float32", repeats=3, dim=1)
    spec.ir["nodes"][0]["attrs"]["dim"] = 3
    with pytest.raises(ValidationError, match="out of range"):
        validate_ir(spec.ir)

    spec = _trace_repeat_interleave("float32", repeats=3, dim=1)
    spec.ir["nodes"][0]["attrs"]["repeats"] = 0
    with pytest.raises(ValidationError, match="positive"):
        validate_ir(spec.ir)

    spec = _trace_repeat_interleave("float32", repeats=3, dim=1)
    spec.ir["outputs"][0]["shape"] = [2, 8, 4]
    spec.ir["outputs"][0]["shape_spec"] = [2, 8, 4]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 8, 4]
    output_tensor["shape_spec"] = [2, 8, 4]
    output_tensor["layout"]["strides"] = [32, 4, 1]
    with pytest.raises(ValidationError, match=r"expected \[2, 9, 4\]"):
        validate_ir(spec.ir)

    spec = _trace_repeat_interleave("float32", repeats=3, dim=1)
    spec.ir["outputs"][0]["dtype"] = "bool"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "bool"
    with pytest.raises(ValidationError, match="expected float32"):
        validate_ir(spec.ir)
