from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from dinoml.runtime import load


_DTYPES = ("float16", "float32", "bfloat16")
_OPS = ("conv3d", "conv3d_bias", "depthwise_conv3d")
_ATOL_BY_DTYPE = {"float16": 0.005, "float32": 1e-5, "bfloat16": 0.05}
_RTOL_BY_DTYPE = {"float16": 0.003, "float32": 1e-5, "bfloat16": 0.04}
_OP_TAGS = {"conv3d": "c3", "conv3d_bias": "c3b", "depthwise_conv3d": "dw3"}
_DTYPE_TAGS = {"float16": "f16", "float32": "f32", "bfloat16": "bf16"}


class _Conv3dParityModule(dml.Module):
    def __init__(self, op_name: str):
        self._op_name = op_name

    def forward(self, x, weight, bias=None):
        kwargs = {"stride": (2, 1, 1), "padding": (1, 1, 0), "dilation": (1, 1, 1)}
        if self._op_name == "conv3d":
            y = dml.ops.conv3d(x, weight, **kwargs)
        elif self._op_name == "conv3d_bias":
            y = dml.ops.conv3d_bias(x, weight, bias, **kwargs)
        elif self._op_name == "depthwise_conv3d":
            y = dml.ops.depthwise_conv3d(x, weight, **kwargs)
        else:
            raise ValueError(f"Unsupported conv3d parity op {self._op_name!r}")
        return dml.ops.output(y, "y")


def _conv3d_output_shape(
    input_shape: list[int],
    weight_shape: list[int],
    *,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> list[int]:
    batch, _in_channels, in_depth, in_height, in_width = input_shape
    out_channels, _weight_in_channels, kernel_d, kernel_h, kernel_w = weight_shape
    return [
        batch,
        out_channels,
        (in_depth + 2 * padding[0] - dilation[0] * (kernel_d - 1) - 1) // stride[0] + 1,
        (in_height + 2 * padding[1] - dilation[1] * (kernel_h - 1) - 1) // stride[1] + 1,
        (in_width + 2 * padding[2] - dilation[2] * (kernel_w - 1) - 1) // stride[2] + 1,
    ]


def _trace_conv3d_parity_spec(op_name: str, dtype: str):
    attrs = {"stride": (2, 1, 1), "padding": (1, 1, 0), "dilation": (1, 1, 1)}
    if op_name == "depthwise_conv3d":
        input_shape = [2, 4, 5, 6, 7]
        weight_shape = [4, 1, 3, 3, 2]
    else:
        input_shape = [2, 3, 5, 6, 7]
        weight_shape = [5, 3, 3, 3, 2]
    output_shape = _conv3d_output_shape(input_shape, weight_shape, **attrs)
    inputs = {
        "x": dml.TensorSpec(input_shape, dtype),
        "weight": dml.TensorSpec(weight_shape, dtype),
    }
    if op_name == "conv3d_bias":
        inputs["bias"] = dml.TensorSpec([weight_shape[0]], dtype)
    spec = dml.trace(_Conv3dParityModule(op_name), inputs=inputs, name=f"{op_name}_{dtype}_conv3d_parity")
    return spec, input_shape, weight_shape, output_shape, attrs


def _random_inputs(
    dtype: str,
    *,
    input_shape: list[int],
    weight_shape: list[int],
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    inputs = {
        "x": rng.standard_normal(input_shape, dtype=np.float32).astype(np.float32),
        "weight": rng.standard_normal(weight_shape, dtype=np.float32).astype(np.float32),
        "bias": rng.standard_normal([weight_shape[0]], dtype=np.float32).astype(np.float32),
    }
    if dtype == "float16":
        return {name: value.astype(np.float16) for name, value in inputs.items()}
    if dtype == "bfloat16":
        return inputs
    return inputs


def _torch_oracle(
    torch,
    op_name: str,
    inputs: dict[str, np.ndarray],
    *,
    attrs: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    x = torch.from_numpy(inputs["x"]).to(dtype=torch.float32)
    weight = torch.from_numpy(inputs["weight"]).to(dtype=torch.float32)
    bias = None if op_name != "conv3d_bias" else torch.from_numpy(inputs["bias"]).to(dtype=torch.float32)
    groups = int(inputs["x"].shape[1]) if op_name == "depthwise_conv3d" else 1
    result = torch.nn.functional.conv3d(
        x,
        weight,
        bias=bias,
        stride=attrs["stride"],
        padding=attrs["padding"],
        dilation=attrs["dilation"],
        groups=groups,
    )
    return result.cpu().numpy()


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("op_name", _OPS)
def test_conv3d_reference_parity_matches_torch_cpu(op_name: str, dtype: str):
    torch = pytest.importorskip("torch")
    spec, input_shape, weight_shape, _output_shape, attrs = _trace_conv3d_parity_spec(op_name, dtype)
    all_inputs = _random_inputs(dtype, input_shape=input_shape, weight_shape=weight_shape)
    spec_input_names = {item["name"] for item in spec.ir["inputs"]}
    spec_inputs = {name: value for name, value in all_inputs.items() if name in spec_input_names}

    actual = reference_numpy(spec, spec_inputs)["y"]
    expected = _torch_oracle(torch, op_name, all_inputs, attrs=attrs)

    np.testing.assert_allclose(actual, expected, atol=_ATOL_BY_DTYPE[dtype], rtol=_RTOL_BY_DTYPE[dtype])


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("op_name", _OPS)
def test_cpu_conv3d_parity_matches_torch(op_name: str, dtype: str, tmp_path):
    torch = pytest.importorskip("torch")
    spec, input_shape, weight_shape, _output_shape, attrs = _trace_conv3d_parity_spec(op_name, dtype)
    all_inputs = _random_inputs(dtype, input_shape=input_shape, weight_shape=weight_shape)
    spec_input_names = {item["name"] for item in spec.ir["inputs"]}
    spec_inputs = {name: value for name, value in all_inputs.items() if name in spec_input_names}

    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"{_OP_TAGS[op_name]}_{_DTYPE_TAGS[dtype]}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(spec_inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = _torch_oracle(torch, op_name, all_inputs, attrs=attrs)
    np.testing.assert_allclose(actual, expected, atol=_ATOL_BY_DTYPE[dtype], rtol=_RTOL_BY_DTYPE[dtype])
