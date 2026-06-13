from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml


ATOL = 1e-5
RTOL = 1e-5


@dataclass(frozen=True)
class FunctionalConv2dCase:
    name: str
    input_shape: tuple[int, int, int, int]
    weight_shape: tuple[int, int, int, int]
    use_bias: bool
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    dilation: tuple[int, int] = (1, 1)
    dtype: str = "float32"

    @property
    def bias_shape(self) -> tuple[int]:
        return (int(self.weight_shape[0]),)


FUNCTIONAL_CONV2D_CASES = (
    FunctionalConv2dCase(
        name="functional_conv2d_bias_f32",
        input_shape=(2, 3, 7, 8),
        weight_shape=(5, 3, 3, 2),
        use_bias=True,
        stride=(2, 1),
        padding=(1, 0),
    ),
    FunctionalConv2dCase(
        name="functional_conv2d_nobias_f32",
        input_shape=(2, 3, 8, 9),
        weight_shape=(4, 3, 2, 3),
        use_bias=False,
        stride=(1, 2),
        padding=(0, 1),
        dilation=(2, 1),
    ),
)


def _make_weights(shape: tuple[int, ...], *, scale: float) -> np.ndarray:
    values = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return (values / scale - 0.5).astype(np.float32)


def _make_bias(shape: tuple[int, ...], *, scale: float) -> np.ndarray:
    values = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return (values / scale - 0.25).astype(np.float32)


class _FunctionalConv2dCaseModule(dml.Module):
    def __init__(self, case: FunctionalConv2dCase):
        self.case = case

    def forward(self, x, weight, bias=None):
        y = dml.nn.functional.conv2d(
            x,
            weight,
            bias,
            stride=self.case.stride,
            padding=self.case.padding,
            dilation=self.case.dilation,
            groups=1,
        )
        return dml.ops.output(y, "y")


def trace_functional_conv2d_spec(case: FunctionalConv2dCase, *, dtype: str | None = None):
    dtype = dtype or case.dtype
    inputs = {
        "x": dml.TensorSpec(list(case.input_shape), dtype),
        "weight": dml.TensorSpec(list(case.weight_shape), dtype),
    }
    if case.use_bias:
        inputs["bias"] = dml.TensorSpec(list(case.bias_shape), dtype)
    return dml.trace(
        _FunctionalConv2dCaseModule(case),
        inputs=inputs,
        name=f"{case.name}_{dtype}",
    )


def random_inputs(case: FunctionalConv2dCase, *, dtype: str | None = None, seed: int = 7) -> dict[str, np.ndarray]:
    dtype = dtype or case.dtype
    rng = np.random.default_rng(seed)
    inputs = {"x": rng.standard_normal(case.input_shape, dtype=np.float32).astype(dtype)}
    inputs["weight"] = _make_weights(case.weight_shape, scale=37.0).astype(dtype)
    if case.use_bias:
        inputs["bias"] = _make_bias(case.bias_shape, scale=17.0).astype(dtype)
    return inputs


def torch_oracle(case: FunctionalConv2dCase, inputs: dict[str, np.ndarray], *, dtype: str | None = None) -> np.ndarray:
    torch = __import__("torch")
    dtype = dtype or case.dtype
    torch_dtype = getattr(torch, dtype)
    x = torch.from_numpy(inputs["x"]).to(dtype=torch_dtype)
    weight = torch.from_numpy(inputs["weight"]).to(dtype=torch_dtype)
    bias = torch.from_numpy(inputs["bias"]).to(dtype=torch_dtype) if case.use_bias else None
    result = torch.nn.functional.conv2d(
        x,
        weight,
        bias=bias,
        stride=case.stride,
        padding=case.padding,
        dilation=case.dilation,
        groups=1,
    )
    return result.float().cpu().numpy().astype(np.float32, copy=False)
