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


class AvgPool2dModule(dml.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return dml.ops.output(dml.ops.avg_pool2d(x, self.kernel_size, self.stride, self.padding), "out")


def _trace_avg_pool2d(dtype="float32", kernel_size=(2, 3), stride=None, padding=(1, 0), shape=(2, 3, 5, 7)):
    return dml.trace(
        AvgPool2dModule(kernel_size, stride, padding),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name=f"avg_pool2d_{dtype}",
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


def _np_avg_pool2d(x, kernel_size, stride=None, padding=0):
    kernel_h, kernel_w = _pair(kernel_size)
    stride_h, stride_w = _pair(kernel_size if stride is None else stride)
    pad_h, pad_w = _pair(padding)
    batch, channels, height, width = x.shape
    out_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
    result = np.empty((batch, channels, out_height, out_width), dtype=np.float32)
    source = np.asarray(x, dtype=np.float32)
    divisor = float(kernel_h * kernel_w)
    for n in range(batch):
        for c in range(channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    total = np.float32(0.0)
                    for kh in range(kernel_h):
                        ih = oh * stride_h + kh - pad_h
                        if ih < 0 or ih >= height:
                            continue
                        for kw in range(kernel_w):
                            iw = ow * stride_w + kw - pad_w
                            if iw < 0 or iw >= width:
                                continue
                            total = np.float32(total + source[n, c, ih, iw])
                    result[n, c, oh, ow] = np.float32(total / divisor)
    return result


def _pair(value):
    if isinstance(value, int):
        return int(value), int(value)
    return int(value[0]), int(value[1])


def test_avg_pool2d_frontend_ir_defaults_stride_to_kernel_and_preserves_dtype():
    spec = _trace_avg_pool2d("float32", kernel_size=(2, 3), stride=None, padding=(1, 2), shape=(2, 3, 5, 7))

    assert spec.ir["outputs"][0]["shape"] == [2, 3, 3, 3]
    assert spec.ir["outputs"][0]["shape_spec"] == [2, 3, 3, 3]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    node = spec.ir["nodes"][0]
    assert node["op"] == "avg_pool2d"
    assert node["inputs"] == ["x"]
    assert node["attrs"] == {"kernel_size": [2, 3], "stride": [2, 3], "padding": [1, 2]}


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_cpu_reference_avg_pool2d(dtype):
    shape = (2, 3, 5, 6)
    spec = _trace_avg_pool2d(dtype, kernel_size=(2, 3), stride=(1, 2), padding=(1, 1), shape=shape)
    x = _input(shape, dtype)

    actual = execute_cpu(spec, {"x": x})["out"]

    expected = _storage_roundtrip(_np_avg_pool2d(x, (2, 3), stride=(1, 2), padding=(1, 1)), dtype)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_avg_pool2d_generated_cpu_source_and_runtime(tmp_path, dtype):
    shape = (1, 2, 4, 5)
    spec = _trace_avg_pool2d(dtype, kernel_size=2, stride=1, padding=1, shape=shape)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["avg_pool2d"]
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int avg_pool2d_" in cpu_source
    assert "sum += dinoml::math::cast<float>(x[input_idx])" in cpu_source
    assert "sum / 4.0f" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x" in cpu_source
        assert "float* DINO_RESTRICT y" in cpu_source

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"avg_pool2d_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input(shape, dtype)
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    expected = _storage_roundtrip(_np_avg_pool2d(x, 2, stride=1, padding=1), dtype)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_avg_pool2d_generated_cuda_source_supports_reduced_precision():
    for dtype, pointer_type in (
        ("float16", "const half* DINO_RESTRICT x"),
        ("bfloat16", "const __nv_bfloat16* DINO_RESTRICT x"),
    ):
        spec = _trace_avg_pool2d(dtype, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0), shape=(1, 2, 5, 4))
        lowered, _ = PassManager().run(spec.ir)
        tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

        cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

        assert pointer_type in cuda_source
        assert "sum += dinoml::math::cast<float>(x[input_idx])" in cuda_source
        assert "y[idx] = dinoml::math::cast<" in cuda_source
        assert "sum / 6.0f" in cuda_source


def test_avg_pool2d_frontend_rejects_dynamic_rank_dtype_and_bad_attrs():
    class DynamicAvgPool(dml.Module):
        def forward(self, x):
            return dml.ops.avg_pool2d(x, 2)

    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(DynamicAvgPool(), inputs={"x": dml.TensorSpec([1, 3, Dim("h", 4, 8), 8])})
    with pytest.raises(ValueError, match="rank-4"):
        _trace_avg_pool2d("float32", shape=(2, 3, 4))
    with pytest.raises(ValueError, match="does not support dtype bool"):
        _trace_avg_pool2d("bool")
    with pytest.raises(ValueError, match="positive integers"):
        _trace_avg_pool2d("float32", kernel_size=0)
    with pytest.raises(ValueError, match="positive integers"):
        _trace_avg_pool2d("float32", kernel_size=2, stride=(1, 0))
    with pytest.raises(ValueError, match="non-negative integers"):
        _trace_avg_pool2d("float32", kernel_size=2, padding=(0, -1))
    with pytest.raises(ValueError, match="pair of integers"):
        _trace_avg_pool2d("float32", kernel_size=(2, 3, 4))
    with pytest.raises(ValueError, match="non-bool integers"):
        _trace_avg_pool2d("float32", kernel_size=(True, 2))
    with pytest.raises(ValueError, match="output height must be positive"):
        _trace_avg_pool2d("float32", kernel_size=(8, 2), shape=(1, 1, 4, 4))


def test_avg_pool2d_validation_rejects_dynamic_shape_spec_bad_attrs_shape_and_dtype():
    spec = _trace_avg_pool2d("float32", kernel_size=2, stride=1, padding=0)
    spec.ir["inputs"][0]["shape_spec"] = [2, 3, Dim("h", 1, 5).to_json(), 7]
    input_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    input_tensor["shape_spec"] = [2, 3, Dim("h", 1, 5).to_json(), 7]
    with pytest.raises(ValidationError, match="only static shapes"):
        validate_ir(spec.ir)

    spec = _trace_avg_pool2d("float32", kernel_size=2, stride=1, padding=0)
    spec.ir["nodes"][0]["attrs"]["padding"] = [0, -1]
    with pytest.raises(ValidationError, match="non-negative"):
        validate_ir(spec.ir)

    spec = _trace_avg_pool2d("float32", kernel_size=2, stride=1, padding=0)
    spec.ir["outputs"][0]["shape"] = [2, 3, 4, 4]
    spec.ir["outputs"][0]["shape_spec"] = [2, 3, 4, 4]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["shape"] = [2, 3, 4, 4]
    output_tensor["shape_spec"] = [2, 3, 4, 4]
    output_tensor["layout"]["strides"] = [48, 16, 4, 1]
    with pytest.raises(ValidationError, match=r"expected \[2, 3, 4, 6\]"):
        validate_ir(spec.ir)

    spec = _trace_avg_pool2d("float32", kernel_size=2, stride=1, padding=0)
    spec.ir["outputs"][0]["dtype"] = "int64"
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    output_tensor["dtype"] = "int64"
    with pytest.raises(ValidationError, match="expected float32"):
        validate_ir(spec.ir)
