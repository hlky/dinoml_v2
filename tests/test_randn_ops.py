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


class RandnModule(dml.Module):
    def __init__(self, shape, dtype="float32", seed=0):
        self.shape = shape
        self.dtype = dtype
        self.seed = seed

    def forward(self):
        return dml.ops.output(dml.ops.randn(self.shape, dtype=self.dtype, seed=self.seed), "out")


def _trace_randn(shape=(2, 3), dtype="float32", seed=17):
    return dml.trace(RandnModule(shape, dtype, seed), inputs={}, name=f"randn_{dtype}")


def _storage_roundtrip(value, dtype):
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return value


def test_randn_frontend_ir_preserves_shape_spec_dtype_and_seed():
    spec = _trace_randn([2, 3], "float32", 17)

    assert spec.ir["inputs"] == []
    assert spec.ir["outputs"][0]["shape"] == [2, 3]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 3]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "randn"
    assert node["inputs"] == []
    assert node["attrs"] == {"shape": [2, 3], "dtype": "float32", "seed": 17}


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        ("float32", np.float32),
        ("float16", np.float16),
        ("bfloat16", np.float32),
    ],
)
def test_cpu_reference_randn_is_seed_deterministic(dtype, expected_dtype):
    spec_a = _trace_randn([2, 3], dtype, 17)
    spec_b = _trace_randn([2, 3], dtype, 17)
    spec_c = _trace_randn([2, 3], dtype, 18)

    actual_a = execute_cpu(spec_a, {})["out"]
    actual_b = execute_cpu(spec_b, {})["out"]
    actual_c = execute_cpu(spec_c, {})["out"]

    assert actual_a.dtype == expected_dtype
    np.testing.assert_array_equal(actual_a, actual_b)
    assert not np.array_equal(actual_a, actual_c)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_randn_generated_cpu_source_and_runtime(tmp_path, dtype):
    spec = _trace_randn([2, 3], dtype, 17)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["randn"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int randn_" in cpu_source
    if dtype == "float32":
        assert "float* DINO_RESTRICT y" in cpu_source
        assert "const unsigned long long seed = 17ull;" in cpu_source
        assert "std::sqrt(-2.0f * std::log(u1))" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"randn_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    try:
        actual = session.run_numpy({})["out"]
    finally:
        session.close()

    expected = _storage_roundtrip(execute_cpu(spec, {})["out"], dtype)
    if dtype == "float32":
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    else:
        np.testing.assert_array_equal(actual, expected)


def test_randn_generated_cuda_source_supports_reduced_precision():
    for dtype, pointer_type in (
        ("float16", "half* DINO_RESTRICT y"),
        ("bfloat16", "__nv_bfloat16* DINO_RESTRICT y"),
    ):
        spec = _trace_randn([2, 3], dtype, 17)
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert "const unsigned long long seed = 17ull;" in cuda_source
        assert "sqrtf(-2.0f * logf(u1))" in cuda_source


def test_randn_frontend_rejects_dynamic_empty_bool_int_and_bad_seed():
    with pytest.raises(ValueError, match="only static shapes"):
        _trace_randn([Dim("n", 1, 4), 3], "float32", 17)
    with pytest.raises(ValueError, match="must not be empty"):
        _trace_randn([], "float32", 17)
    with pytest.raises(ValueError, match="positive"):
        _trace_randn([2, 0], "float32", 17)
    with pytest.raises(ValueError, match="randn does not support dtype bool"):
        _trace_randn([2, 3], "bool", 17)
    with pytest.raises(ValueError, match="randn does not support dtype int32"):
        _trace_randn([2, 3], "int32", 17)
    with pytest.raises(ValueError, match="integer seed"):
        _trace_randn([2, 3], "float32", True)
    with pytest.raises(ValueError, match="uint64"):
        _trace_randn([2, 3], "float32", -1)


def test_randn_validation_rejects_bad_shape_dtype_and_seed():
    spec = _trace_randn([2, 3], "float32", 17)
    spec.ir["nodes"][0]["attrs"]["shape"] = [2, 0]
    with pytest.raises(ValidationError, match="positive integer shape"):
        validate_ir(spec.ir)

    spec = _trace_randn([2, 3], "float32", 17)
    spec.ir["nodes"][0]["attrs"]["dtype"] = "int64"
    spec.ir["outputs"][0]["dtype"] = "int64"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "int64"
    with pytest.raises(ValidationError, match="randn does not support dtype int64"):
        validate_ir(spec.ir)

    spec = _trace_randn([2, 3], "float32", 17)
    spec.ir["nodes"][0]["attrs"]["seed"] = -1
    with pytest.raises(ValidationError, match="uint64"):
        validate_ir(spec.ir)
