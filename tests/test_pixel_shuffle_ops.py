import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from dinoml.lowering.ops import render_generated_kernels
from dinoml.passes import PassManager, validate_ir
from dinoml.runtime import load
from dinoml.shapes import Dim


class PixelShuffleModule(dml.Module):
    def __init__(self, factor):
        self.factor = factor

    def forward(self, x):
        return dml.ops.output(dml.ops.pixel_shuffle(x, self.factor), "out")


class PixelUnshuffleModule(dml.Module):
    def __init__(self, factor):
        self.factor = factor

    def forward(self, x):
        return dml.ops.output(dml.ops.pixel_unshuffle(x, self.factor), "out")


def _trace_shuffle(shape=(1, 8, 2, 3), dtype="float32", factor=2):
    return dml.trace(
        PixelShuffleModule(factor),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name=f"pixel_shuffle_{dtype}",
    )


def _trace_unshuffle(shape=(1, 2, 4, 6), dtype="float32", factor=2):
    return dml.trace(
        PixelUnshuffleModule(factor),
        inputs={"x": dml.TensorSpec(shape, dtype)},
        name=f"pixel_unshuffle_{dtype}",
    )


def _pixel_shuffle_numpy(x, factor):
    n, channels_in, height, width = x.shape
    channels_out = channels_in // (factor * factor)
    return (
        x.reshape(n, channels_out, factor, factor, height, width)
        .transpose(0, 1, 4, 2, 5, 3)
        .reshape(n, channels_out, height * factor, width * factor)
        .copy()
    )


def _pixel_unshuffle_numpy(x, factor):
    n, channels, height_in, width_in = x.shape
    height_out = height_in // factor
    width_out = width_in // factor
    return (
        x.reshape(n, channels, height_out, factor, width_out, factor)
        .transpose(0, 1, 3, 5, 2, 4)
        .reshape(n, channels * factor * factor, height_out, width_out)
        .copy()
    )


def _input(shape, dtype):
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    if dtype == "bool":
        return (values.astype(np.int64) % 3) == 0
    return values


def test_pixel_shuffle_frontend_ir_composes_views_and_permute():
    spec = _trace_shuffle()

    assert spec.ir["outputs"][0]["shape"] == [1, 2, 4, 6]
    assert spec.ir["outputs"][0]["shape_spec"] == [1, 2, 4, 6]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    assert [node["op"] for node in spec.ir["nodes"]] == ["permute"]
    assert spec.ir["nodes"][0]["attrs"] == {"dims": [0, 1, 4, 2, 5, 3]}
    views = spec.ir["metadata"]["views"]["views"]
    assert [(view["transform"], view["shape"]) for view in views] == [
        ("reshape", [1, 2, 2, 2, 2, 3]),
        ("reshape", [1, 2, 4, 6]),
    ]
    assert views[0]["source"] == "x"
    assert views[1]["source"] == spec.ir["nodes"][0]["outputs"][0]


def test_pixel_unshuffle_frontend_ir_composes_views_and_permute():
    spec = _trace_unshuffle()

    assert spec.ir["outputs"][0]["shape"] == [1, 8, 2, 3]
    assert spec.ir["outputs"][0]["shape_spec"] == [1, 8, 2, 3]
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    assert [node["op"] for node in spec.ir["nodes"]] == ["permute"]
    assert spec.ir["nodes"][0]["attrs"] == {"dims": [0, 1, 3, 5, 2, 4]}
    views = spec.ir["metadata"]["views"]["views"]
    assert [(view["transform"], view["shape"]) for view in views] == [
        ("reshape", [1, 2, 2, 2, 3, 2]),
        ("reshape", [1, 8, 2, 3]),
    ]
    assert views[0]["source"] == "x"
    assert views[1]["source"] == spec.ir["nodes"][0]["outputs"][0]


@pytest.mark.parametrize(
    ("op_name", "trace_fn", "shape", "expected_fn"),
    [
        ("pixel_shuffle", _trace_shuffle, (1, 8, 2, 3), _pixel_shuffle_numpy),
        ("pixel_unshuffle", _trace_unshuffle, (1, 2, 4, 6), _pixel_unshuffle_numpy),
    ],
)
@pytest.mark.parametrize(("dtype", "expected_dtype"), [("float32", np.float32), ("bool", np.bool_)])
def test_cpu_reference_pixel_shuffle_ops_match_numpy(op_name, trace_fn, shape, expected_fn, dtype, expected_dtype):
    spec = trace_fn(shape=shape, dtype=dtype)
    x = _input(shape, dtype)

    actual = reference_numpy(spec, {"x": x})["out"]
    expected = expected_fn(x, 2)
    assert actual.dtype == expected_dtype
    assert spec.ir["outputs"][0]["shape"] == list(expected.shape)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("op_name", "trace_fn", "shape", "expected_fn", "dims"),
    [
        ("pixel_shuffle", _trace_shuffle, (1, 8, 2, 3), _pixel_shuffle_numpy, [0, 1, 4, 2, 5, 3]),
        ("pixel_unshuffle", _trace_unshuffle, (1, 2, 4, 6), _pixel_unshuffle_numpy, [0, 1, 3, 5, 2, 4]),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "bool"])
def test_pixel_shuffle_ops_generated_cpu_source_and_runtime(tmp_path, op_name, trace_fn, shape, expected_fn, dims, dtype):
    spec = trace_fn(shape=shape, dtype=dtype)
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [node["op"] for node in lowered["nodes"]] == ["permute"]
    assert lowered["nodes"][0]["attrs"] == {"dims": dims}
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

    cpu_source = render_generated_kernels("cpu", lowered["nodes"], tensor_map)[0]

    assert "static int permute_" in cpu_source
    if dtype == "float32":
        assert "const float* DINO_RESTRICT x" in cpu_source
        assert "float* DINO_RESTRICT y" in cpu_source
    if dtype not in {"float32", "bool"}:
        return

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"{op_name}_{dtype}_cpu.dinoml")
    session = load(artifact.path).create_session()
    x = _input(shape, dtype)
    try:
        actual = session.run_numpy({"x": x})["out"]
    finally:
        session.close()

    expected = expected_fn(x, 2)
    np.testing.assert_array_equal(actual, expected)


def test_pixel_shuffle_ops_generated_cuda_source_supports_reduced_precision_and_bool():
    for trace_fn in (_trace_shuffle, _trace_unshuffle):
        for dtype, pointer_type in (
            ("float16", "const half* DINO_RESTRICT x"),
            ("bfloat16", "const __nv_bfloat16* DINO_RESTRICT x"),
            ("bool", "const bool* DINO_RESTRICT x"),
        ):
            spec = trace_fn(dtype=dtype)
            lowered, _ = PassManager().run(spec.ir)
            tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}

            cuda_source = render_generated_kernels("cuda", lowered["nodes"], tensor_map)[0]

            assert pointer_type in cuda_source
            assert "permute_" in cuda_source
            assert "y[idx] = x[input_idx]" in cuda_source


def test_pixel_shuffle_rejects_bad_inputs():
    class DynamicShuffle(dml.Module):
        def forward(self, x):
            return dml.ops.pixel_shuffle(x, 2)

    with pytest.raises(ValueError, match="rank-4"):
        _trace_shuffle(shape=(8, 2, 3))
    with pytest.raises(ValueError, match="positive integer"):
        _trace_shuffle(factor=True)
    with pytest.raises(ValueError, match="positive integer"):
        _trace_shuffle(factor=0)
    with pytest.raises(ValueError, match="divisible by upscale_factor"):
        _trace_shuffle(shape=(1, 10, 2, 3), factor=2)
    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(DynamicShuffle(), inputs={"x": dml.TensorSpec([1, 8, Dim("h", 1, 4), 3])})
    with pytest.raises(ValueError, match="does not support dtype int64"):
        _trace_shuffle(dtype="int64")


def test_pixel_unshuffle_rejects_bad_inputs():
    class DynamicUnshuffle(dml.Module):
        def forward(self, x):
            return dml.ops.pixel_unshuffle(x, 2)

    with pytest.raises(ValueError, match="rank-4"):
        _trace_unshuffle(shape=(2, 4, 6))
    with pytest.raises(ValueError, match="positive integer"):
        _trace_unshuffle(factor=False)
    with pytest.raises(ValueError, match="positive integer"):
        _trace_unshuffle(factor=-1)
    with pytest.raises(ValueError, match="height .*divisible"):
        _trace_unshuffle(shape=(1, 2, 5, 6), factor=2)
    with pytest.raises(ValueError, match="width .*divisible"):
        _trace_unshuffle(shape=(1, 2, 4, 7), factor=2)
    with pytest.raises(ValueError, match="only static input shapes"):
        dml.trace(DynamicUnshuffle(), inputs={"x": dml.TensorSpec([1, 2, Dim("h", 1, 4), 6])})
    with pytest.raises(ValueError, match="does not support dtype int64"):
        _trace_unshuffle(dtype="int64")
