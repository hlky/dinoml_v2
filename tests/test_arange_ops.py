import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.lowering.ops import render_generated_kernels
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError
from dinoml.runtime import load


class ArangeModule(dml.Module):
    def __init__(self, start, end=None, step=1, dtype="float32"):
        self.start = start
        self.end = end
        self.step = step
        self.dtype = dtype

    def forward(self):
        return dml.ops.output(dml.ops.arange(self.start, self.end, self.step, dtype=self.dtype), "out")


def _trace_arange(start, end=None, step=1, dtype="float32"):
    return dml.trace(ArangeModule(start, end, step, dtype), inputs={}, name=f"arange_{dtype}")


def _expected(start, end=None, step=1, dtype="float32"):
    if end is None:
        start, end = 0, start
    expected = np.arange(float(start), float(end), float(step), dtype=np.float32)
    if dtype in {"float16", "bfloat16"}:
        expected = array_from_storage(array_to_storage(expected, dtype), dtype)
    return expected


def test_arange_frontend_ir_preserves_shape_spec_dtype_and_attrs():
    spec = _trace_arange(1, 7, 2, "float32")

    assert spec.ir["inputs"] == []
    assert spec.ir["outputs"][0]["shape"] == [3]
    assert spec.ir["outputs"][0]["shape_spec"] == [3]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "arange"
    assert node["inputs"] == []
    assert node["attrs"] == {"start": 1.0, "end": 7.0, "step": 2.0, "dtype": "float32"}


@pytest.mark.parametrize(
    ("start", "end", "step"),
    [
        (4, None, 1),
        (5, -1, -2),
        (0.5, 2.0, 0.5),
    ],
)
def test_cpu_reference_arange_semantics(start, end, step):
    spec = _trace_arange(start, end, step, "float32")

    actual = execute_cpu(spec, {})["out"]

    np.testing.assert_array_equal(actual, _expected(start, end, step))


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_arange_generated_cpu_source_and_runtime(tmp_path, dtype):
    spec = _trace_arange(1, 7, 2, dtype)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["arange"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int arange_" in cpu_source
    if dtype == "float32":
        assert "float* DINO_RESTRICT y" in cpu_source
        assert "const float start = 1.0f;" in cpu_source
        assert "const float step = 2.0f;" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"arange_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    try:
        actual = session.run_numpy({})["out"]
    finally:
        session.close()

    np.testing.assert_array_equal(actual, _expected(1, 7, 2, dtype))


def test_arange_generated_cuda_source_supports_reduced_precision():
    for dtype, pointer_type in (
        ("float16", "half* DINO_RESTRICT y"),
        ("bfloat16", "__nv_bfloat16* DINO_RESTRICT y"),
    ):
        spec = _trace_arange(1, 7, 2, dtype)
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert "const float start = 1.0f;" in cuda_source
        assert "const float step = 2.0f;" in cuda_source


def test_arange_frontend_rejects_zero_step_empty_ranges_bool_and_int_dtype():
    with pytest.raises(ValueError, match="step must not be zero"):
        _trace_arange(0, 3, 0)
    with pytest.raises(ValueError, match="non-empty range"):
        _trace_arange(3, 0, 1)
    with pytest.raises(ValueError, match="non-empty range"):
        _trace_arange(0, 3, -1)
    with pytest.raises(ValueError, match="arange does not support dtype bool"):
        _trace_arange(0, 3, 1, "bool")
    with pytest.raises(ValueError, match="arange does not support dtype int32"):
        _trace_arange(0, 3, 1, "int32")


def test_arange_validation_rejects_bad_attrs_shape_and_dtype():
    spec = _trace_arange(0, 3, 1, "float32")
    spec.ir["nodes"][0]["attrs"]["step"] = 0.0
    with pytest.raises(ValidationError, match="step must not be zero"):
        validate_ir(spec.ir)

    spec = _trace_arange(0, 3, 1, "float32")
    spec.ir["nodes"][0]["attrs"]["end"] = 4.0
    with pytest.raises(ValidationError, match="expected \\[4\\]"):
        validate_ir(spec.ir)

    spec = _trace_arange(0, 3, 1, "float32")
    spec.ir["nodes"][0]["attrs"]["dtype"] = "int64"
    spec.ir["outputs"][0]["dtype"] = "int64"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "int64"
    with pytest.raises(ValidationError, match="arange does not support dtype int64"):
        validate_ir(spec.ir)
