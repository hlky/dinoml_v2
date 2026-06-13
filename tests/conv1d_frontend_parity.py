from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml


ATOL = 1e-5
RTOL = 1e-5


@dataclass(frozen=True)
class Conv1dFrontendCase:
    name: str
    kind: str
    input_shape: tuple[int, int, int]
    weight_shape: tuple[int, int, int]
    dtype: str = "float32"
    stride: int = 2
    padding: int = 1
    dilation: int = 1

    @property
    def bias_shape(self) -> tuple[int]:
        return (int(self.weight_shape[0]),)


CONV1D_FRONTEND_CASES = (
    Conv1dFrontendCase(
        name="conv1d_module_f32",
        kind="module",
        input_shape=(2, 3, 9),
        weight_shape=(5, 3, 3),
    ),
    Conv1dFrontendCase(
        name="conv1d_functional_f32",
        kind="functional",
        input_shape=(2, 3, 9),
        weight_shape=(5, 3, 3),
    ),
)


def _make_weights(shape: tuple[int, ...], *, scale: float) -> np.ndarray:
    values = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return (values / scale - 0.5).astype(np.float32)


def _make_bias(shape: tuple[int, ...], *, scale: float) -> np.ndarray:
    values = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return (values / scale - 0.25).astype(np.float32)


class _Conv1dModuleCase(dml.Module):
    def __init__(self, case: Conv1dFrontendCase):
        self.conv = dml.nn.Conv1d(
            case.input_shape[1],
            case.weight_shape[0],
            kernel_size=case.weight_shape[2],
            stride=case.stride,
            padding=case.padding,
            dilation=case.dilation,
            bias=True,
            dtype=case.dtype,
        )
        self.conv.weight = dml.Parameter(
            list(case.weight_shape),
            dtype=case.dtype,
            value=_make_weights(case.weight_shape, scale=31.0).astype(case.dtype),
        )
        self.conv.bias = dml.Parameter(
            list(case.bias_shape),
            dtype=case.dtype,
            value=_make_bias(case.bias_shape, scale=17.0).astype(case.dtype),
        )

    def forward(self, x):
        return dml.ops.output(self.conv(x), "y")


class _Conv1dFunctionalCase(dml.Module):
    def __init__(self, case: Conv1dFrontendCase):
        self.case = case

    def forward(self, x, weight, bias):
        y = dml.nn.functional.conv1d(
            x,
            weight,
            bias,
            stride=self.case.stride,
            padding=self.case.padding,
            dilation=self.case.dilation,
            groups=1,
        )
        return dml.ops.output(y, "y")


def trace_conv1d_frontend_spec(case: Conv1dFrontendCase, *, dtype: str | None = None):
    dtype = dtype or case.dtype
    if case.kind == "module":
        module_case = case if dtype == case.dtype else Conv1dFrontendCase(**{**case.__dict__, "dtype": dtype})
        return dml.trace(
            _Conv1dModuleCase(module_case),
            inputs={"x": dml.TensorSpec(list(case.input_shape), dtype)},
            name=case.name,
        )
    return dml.trace(
        _Conv1dFunctionalCase(case),
        inputs={
            "x": dml.TensorSpec(list(case.input_shape), dtype),
            "weight": dml.TensorSpec(list(case.weight_shape), dtype),
            "bias": dml.TensorSpec(list(case.bias_shape), dtype),
        },
        name=case.name,
    )


def random_inputs(case: Conv1dFrontendCase, *, dtype: str | None = None, seed: int = 7) -> dict[str, np.ndarray]:
    dtype = dtype or case.dtype
    rng = np.random.default_rng(seed)
    inputs = {"x": rng.standard_normal(case.input_shape, dtype=np.float32).astype(dtype)}
    if case.kind == "functional":
        inputs["weight"] = _make_weights(case.weight_shape, scale=29.0).astype(dtype)
        inputs["bias"] = _make_bias(case.bias_shape, scale=13.0).astype(dtype)
    return inputs


def torch_oracle(case: Conv1dFrontendCase, inputs: dict[str, np.ndarray], *, dtype: str | None = None) -> np.ndarray:
    torch = __import__("torch")
    dtype = dtype or case.dtype
    torch_dtype = getattr(torch, dtype)
    x = torch.from_numpy(inputs["x"]).to(dtype=torch_dtype)
    if case.kind == "module":
        weight = torch.from_numpy(_make_weights(case.weight_shape, scale=31.0).astype(dtype)).to(dtype=torch_dtype)
        bias = torch.from_numpy(_make_bias(case.bias_shape, scale=17.0).astype(dtype)).to(dtype=torch_dtype)
    else:
        weight = torch.from_numpy(inputs["weight"]).to(dtype=torch_dtype)
        bias = torch.from_numpy(inputs["bias"]).to(dtype=torch_dtype)
    result = torch.nn.functional.conv1d(
        x,
        weight,
        bias=bias,
        stride=case.stride,
        padding=case.padding,
        dilation=case.dilation,
        groups=1,
    )
    return result.float().cpu().numpy().astype(np.float32, copy=False)
