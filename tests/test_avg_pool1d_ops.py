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


class AvgPool1dModule(dml.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return dml.ops.output(dml.ops.avg_pool1d(x, self.kernel_size, self.stride, self.padding), "out")


def _trace_avg_pool1d(dtype="float32", kernel_size=3, stride=None, padding=1, shape=(2, 3, 8)):
    return dml.trace(
        AvgPool1dModule(kernel_size, stride, padding),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name=f"avg_pool1d_{dtype}",
    )


def _input(shape, dtype):
    value = np.linspace(-3.0, 5.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return value


def _storage_roundtrip(value, dtype):
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return np.asarray(value, dtype=np.float32)


def _np_avg_pool1d(x, kernel_size, stride=None, padding=0):
    kernel = _single(kernel_size)
    normalized_stride = _single(kernel_size if stride is None else stride)
    normalized_padding = _single(padding)
    batch, channels, length = x.shape
    out_length = (length + 2 * normalized_padding - kernel) // normalized_stride + 1
    result = np.empty((batch, channels, out_length), dtype=np.float32)
    source = np.asarray(x, dtype=np.float32)
    divisor = float(kernel)
    for n in range(batch):
        for c in range(channels):
            for ol in range(out_length):
                total = np.float32(0.0)
                for kl in range(kernel):
                    il = ol * normalized_stride + kl - normalized_padding
                    if il < 0 or il >= length:
                        continue
                    total = np.float32(total + source[n, c, il])
                result[n, c, ol] = np.float32(total / divisor)
    return result


def _single(value):
    if isinstance(value, int):
        return int(value)
    return int(value[0])


def test_avg_pool1d_frontend_ir_defaults_stride_to_kernel_and_preserves_dtype():
    spec = _trace_avg_pool1d("float32", kernel_size=(3,), stride=None, padding=(1,), shape=(2, 3, 8))

    assert spec.ir["outputs"][0]["shape"] == [2, 3, 3]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 3, 3]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "avg_pool1d"
    assert node["inputs"] == ["x"]
    assert node["attrs"] == {"kernel_size": [3], "stride": [3], "padding": [1]}


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_cpu_reference_avg_pool1d(dtype):
    shape = (2, 3, 8)
    spec = _trace_avg_pool1d(dtype, kernel_size=3, stride=2, padding=1, shape=shape)
    x = _input(shape, dtype)

    actual = reference_numpy(spec, {"x": x})["out"]

    expected = _storage_roundtrip(_np_avg_pool1d(x, 3, stride=2, padding=1), dtype)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_avg_pool1d_generated_cpu_source_and_runtime(tmp_path, dtype):
    shape = (1, 2, 5)
    spec = _trace_avg_pool1d(dtype, kernel_size=2, stride=1, padding=1, shape=shape)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["avg_pool1d"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int avg_pool1d_" in cpu_source
    assert "sum += dinoml::math::cast<float>(x[input_idx])" in cpu_source
    assert "sum / 2.0f" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x" in cpu_source
        assert "float* DINO_RESTRICT y" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"avg_pool1d_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input(shape, dtype)
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    expected = _storage_roundtrip(_np_avg_pool1d(x, 2, stride=1, padding=1), dtype)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_avg_pool1d_generated_cuda_source_supports_reduced_precision():
    for dtype, pointer_type in (
        ("float16", "const half* DINO_RESTRICT x"),
        ("bfloat16", "const __nv_bfloat16* DINO_RESTRICT x"),
    ):
        spec = _trace_avg_pool1d(dtype, kernel_size=3, stride=2, padding=1, shape=(1, 2, 7))
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert "sum += dinoml::math::cast<float>(x[input_idx])" in cuda_source
        assert "y[idx] = dinoml::math::cast<" in cuda_source
        assert "sum / 3.0f" in cuda_source


def test_avg_pool1d_frontend_rejects_dynamic_rank_dtype_and_bad_attrs():
    class DynamicAvgPool(dml.Module):
        def forward(self, x):
            return dml.ops.avg_pool1d(x, 2)

    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(DynamicAvgPool(), inputs={"x": dml.TensorSpec([1, 3, Dim("l", 4, 8)])})
    with pytest.raises(ValueError, match="rank-3"):
        _trace_avg_pool1d("float32", shape=(2, 3, 4, 5))
    with pytest.raises(ValueError, match="does not support dtype bool"):
        _trace_avg_pool1d("bool")
    with pytest.raises(ValueError, match="positive integers"):
        _trace_avg_pool1d("float32", kernel_size=0)
    with pytest.raises(ValueError, match="positive integers"):
        _trace_avg_pool1d("float32", kernel_size=2, stride=0)
    with pytest.raises(ValueError, match="non-negative integers"):
        _trace_avg_pool1d("float32", kernel_size=2, padding=-1)
    with pytest.raises(ValueError, match="length-1 sequence"):
        _trace_avg_pool1d("float32", kernel_size=(2, 3))
    with pytest.raises(ValueError, match="non-bool integers"):
        _trace_avg_pool1d("float32", kernel_size=(True,))
    with pytest.raises(ValueError, match="output length must be positive"):
        _trace_avg_pool1d("float32", kernel_size=9, shape=(1, 1, 4))


def test_avg_pool1d_validation_rejects_dynamic_shape_spec_bad_attrs_shape_and_dtype():
    spec = _trace_avg_pool1d("float32", kernel_size=2, stride=1, padding=0, shape=(2, 3, 7))
    spec.ir["inputs"][0]["shape_spec"] = [2, 3, Dim("l", 1, 7).to_json()]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [2, 3, Dim("l", 1, 7).to_json()]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace_avg_pool1d("float32", kernel_size=2, stride=1, padding=0, shape=(2, 3, 7))
    spec.ir["nodes"][0]["attrs"]["padding"] = [-1]
    with pytest.raises(ValidationError, match="non-negative"):
        validate_ir(spec.ir)

    spec = _trace_avg_pool1d("float32", kernel_size=2, stride=1, padding=0, shape=(2, 3, 7))
    spec.ir["outputs"][0]["shape"] = [2, 3, 4]
    spec.ir["outputs"][0]["shape_spec"] = [2, 3, 4]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 3, 4]
    output_tensor["shape_spec"] = [2, 3, 4]
    output_tensor["layout"]["strides"] = [12, 4, 1]
    with pytest.raises(ValidationError, match=r"expected \[2, 3, 6\]"):
        validate_ir(spec.ir)

    spec = _trace_avg_pool1d("float32", kernel_size=2, stride=1, padding=0, shape=(2, 3, 7))
    spec.ir["outputs"][0]["dtype"] = "int64"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "int64"
    with pytest.raises(ValidationError, match="expected float32"):
        validate_ir(spec.ir)
