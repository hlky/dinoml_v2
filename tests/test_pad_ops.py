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


class PadModule(dml.Module):
    def __init__(self, pad, value=0.0):
        self.pad = pad
        self.value = value

    def forward(self, x):
        return dml.ops.output(dml.ops.pad(x, self.pad, self.value), "out")


class PadLastDimModule(dml.Module):
    def __init__(self, left, right, value=0.0):
        self.left = left
        self.right = right
        self.value = value

    def forward(self, x):
        return dml.ops.output(dml.ops.pad_last_dim(x, self.left, self.right, self.value), "out")


def _trace_pad(dtype="float32", pad=(1, 2, 3, 0), value=-1.5, shape=(2, 3, 4)):
    return dml.trace(PadModule(pad, value), inputs={"x": dml.TensorSpec(shape, dtype)}, name=f"pad_{dtype}")


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


def _np_pad(x, pad, value):
    pad_width = [(0, 0)] * x.ndim
    for pair_index in range(len(pad) // 2):
        axis = x.ndim - 1 - pair_index
        pad_width[axis] = (pad[2 * pair_index], pad[2 * pair_index + 1])
    return np.pad(x, tuple(pad_width), mode="constant", constant_values=value).copy()


def test_pad_frontend_ir_uses_torch_pair_order_and_preserves_dtype():
    spec = _trace_pad("float32", pad=(1, 2, 3, 0), value=-1.5, shape=(2, 3, 4))

    assert spec.ir["outputs"][0]["shape"] == [2, 6, 7]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 6, 7]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "pad"
    assert node["inputs"] == ["x"]
    assert node["attrs"] == {"pad": [1, 2, 3, 0], "value": -1.5}


def test_pad_last_dim_frontend_wraps_pad():
    spec = dml.trace(PadLastDimModule(2, 1, 3.0), inputs={"x": dml.TensorSpec([2, 3], "float32")})

    assert spec.ir["outputs"][0]["shape"] == [2, 6]
    assert spec.ir["nodes"][0]["op"] == "pad"
    assert spec.ir["nodes"][0]["attrs"] == {"pad": [2, 1], "value": 3.0}


@pytest.mark.parametrize(
    ("dtype", "value", "expected_dtype"),
    [
        ("float32", -2.5, np.float32),
        ("float16", 1.125, np.float16),
        ("bfloat16", 1.125, np.float32),
        ("bool", True, np.bool_),
    ],
)
def test_cpu_reference_pad(dtype, value, expected_dtype):
    spec = _trace_pad(dtype, pad=(1, 2, 1, 0), value=value, shape=(2, 3, 2))
    x = _input(dtype)

    actual = execute_cpu(spec, {"x": x})["out"]

    expected = _storage_roundtrip(_np_pad(x, [1, 2, 1, 0], value), dtype)
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        ("float32", -3.5),
        ("float16", 1.125),
        ("bfloat16", 1.125),
        ("bool", True),
    ],
)
def test_pad_generated_cpu_source_and_runtime(tmp_path, dtype, value):
    spec = _trace_pad(dtype, pad=(1, 2, 1, 0), value=value, shape=(2, 3, 2))
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["pad"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int pad_" in cpu_source
    assert "inside ? x[input_idx] : fill_value" in cpu_source
    assert "input_coord = coord - 1" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x" in cpu_source
        assert "float* DINO_RESTRICT y" in cpu_source
        assert "dinoml::math::cast<float>(-3.5f)" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"pad_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input(dtype)
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    expected = _storage_roundtrip(_np_pad(x, [1, 2, 1, 0], value), dtype)
    np.testing.assert_array_equal(actual, expected)


def test_pad_generated_cuda_source_supports_reduced_precision_and_bool():
    for dtype, value, pointer_type, fill_expr in (
        ("float16", 1.25, "const half* DINO_RESTRICT x", "dinoml::math::cast<half>(1.25f)"),
        (
            "bfloat16",
            1.25,
            "const __nv_bfloat16* DINO_RESTRICT x",
            "dinoml::math::cast<__nv_bfloat16>(1.25f)",
        ),
        ("bool", True, "const bool* DINO_RESTRICT x", "dinoml::math::cast<bool>(1.0f)"),
    ):
        spec = _trace_pad(dtype, pad=(1, 2), value=value, shape=(2, 3, 2))
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert fill_expr in cuda_source
        assert "inside ? x[input_idx] : fill_value" in cuda_source


def test_pad_frontend_rejects_dynamic_bad_pad_value_and_unsupported_dtype():
    class DynamicPad(dml.Module):
        def forward(self, x):
            return dml.ops.pad(x, [1, 1])

    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(DynamicPad(), inputs={"x": dml.TensorSpec([Dim("n", 1, 4), 3])})
    with pytest.raises(ValueError, match="non-empty"):
        _trace_pad("float32", pad=())
    with pytest.raises(ValueError, match="even"):
        _trace_pad("float32", pad=(1, 2, 3))
    with pytest.raises(ValueError, match="non-negative"):
        _trace_pad("float32", pad=(1, -1))
    with pytest.raises(ValueError, match="non-bool integers"):
        _trace_pad("float32", pad=(True, 1))
    with pytest.raises(ValueError, match="more dimensions"):
        _trace_pad("float32", pad=(1, 1, 1, 1, 1, 1, 1, 1), shape=(2, 3, 4))
    with pytest.raises(ValueError, match="constant numeric scalar"):
        _trace_pad("float32", pad=(1, 1), value="zero")
    with pytest.raises(ValueError, match="does not support dtype int64"):
        _trace_pad("int64", pad=(1, 1))
    with pytest.raises(ValueError, match="left must be a non-negative integer"):
        dml.trace(PadLastDimModule(True, 1), inputs={"x": dml.TensorSpec([2, 3])})


def test_pad_validation_rejects_dynamic_shape_spec_bad_pad_shape_and_dtype():
    spec = _trace_pad("float32", pad=(1, 2, 1, 0))
    spec.ir["inputs"][0]["shape_spec"] = [Dim("n", 1, 2).to_json(), 3, 4]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [Dim("n", 1, 2).to_json(), 3, 4]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace_pad("float32", pad=(1, 2, 1, 0))
    spec.ir["nodes"][0]["attrs"]["pad"] = [1, -1]
    with pytest.raises(ValidationError, match="non-negative"):
        validate_ir(spec.ir)

    spec = _trace_pad("float32", pad=(1, 2, 1, 0))
    spec.ir["nodes"][0]["attrs"]["value"] = True
    with pytest.raises(ValidationError, match="numeric scalar"):
        validate_ir(spec.ir)

    spec = _trace_pad("float32", pad=(1, 2, 1, 0))
    spec.ir["outputs"][0]["shape"] = [2, 3, 5]
    spec.ir["outputs"][0]["shape_spec"] = [2, 3, 5]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 3, 5]
    output_tensor["shape_spec"] = [2, 3, 5]
    output_tensor["layout"]["strides"] = [15, 5, 1]
    with pytest.raises(ValidationError, match=r"expected \[2, 4, 7\]"):
        validate_ir(spec.ir)

    spec = _trace_pad("float32", pad=(1, 2, 1, 0))
    spec.ir["outputs"][0]["dtype"] = "int64"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "int64"
    with pytest.raises(ValidationError, match="expected float32"):
        validate_ir(spec.ir)
