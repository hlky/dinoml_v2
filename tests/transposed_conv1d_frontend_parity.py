from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml


ATOL = 1e-5
RTOL = 1e-5


@dataclass(frozen=True)
class TransposedConv1dFrontendCase:
    name: str
    kind: str
    input_shape: tuple[int, int, int]
    weight_shape: tuple[int, int, int]
    dtype: str = "float32"
    stride: int = 2
    padding: int = 1
    output_padding: int = 1
    dilation: int = 1


TRANSPOSED_CONV1D_FRONTEND_CASES = (
    TransposedConv1dFrontendCase(
        name="conv_transpose1d_module_nobias_f32",
        kind="module",
        input_shape=(1, 8, 4),
        weight_shape=(8, 16, 3),
    ),
    TransposedConv1dFrontendCase(
        name="conv_transpose1d_functional_nobias_f32",
        kind="functional",
        input_shape=(1, 8, 4),
        weight_shape=(8, 16, 3),
    ),
)


def _make_weights(shape: tuple[int, ...], *, scale: float) -> np.ndarray:
    values = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return (values / scale - 0.5).astype(np.float32)


class _ConvTranspose1dModuleCase(dml.Module):
    def __init__(self, case: TransposedConv1dFrontendCase):
        self.conv = dml.nn.ConvTranspose1d(
            case.input_shape[1],
            case.weight_shape[1],
            kernel_size=case.weight_shape[2],
            stride=case.stride,
            padding=case.padding,
            output_padding=case.output_padding,
            dilation=case.dilation,
            bias=False,
            dtype=case.dtype,
        )
        self.conv.weight = dml.Parameter(
            list(case.weight_shape),
            dtype=case.dtype,
            value=_make_weights(case.weight_shape, scale=37.0).astype(case.dtype),
        )

    def forward(self, x):
        return dml.ops.output(self.conv(x), "y")


class _ConvTranspose1dFunctionalCase(dml.Module):
    def __init__(self, case: TransposedConv1dFrontendCase):
        self.case = case

    def forward(self, x, weight):
        y = dml.nn.functional.conv_transpose1d(
            x,
            weight,
            bias=None,
            stride=self.case.stride,
            padding=self.case.padding,
            output_padding=self.case.output_padding,
            dilation=self.case.dilation,
            groups=1,
        )
        return dml.ops.output(y, "y")


def trace_transposed_conv1d_frontend_spec(case: TransposedConv1dFrontendCase, *, dtype: str | None = None):
    dtype = dtype or case.dtype
    case_for_dtype = case if dtype == case.dtype else TransposedConv1dFrontendCase(**{**case.__dict__, "dtype": dtype})
    if case.kind == "module":
        return dml.trace(
            _ConvTranspose1dModuleCase(case_for_dtype),
            inputs={"x": dml.TensorSpec(list(case.input_shape), dtype)},
            name=case.name,
        )
    return dml.trace(
        _ConvTranspose1dFunctionalCase(case_for_dtype),
        inputs={
            "x": dml.TensorSpec(list(case.input_shape), dtype),
            "weight": dml.TensorSpec(list(case.weight_shape), dtype),
        },
        name=case.name,
    )


def random_inputs(case: TransposedConv1dFrontendCase, *, dtype: str | None = None, seed: int = 7) -> dict[str, np.ndarray]:
    dtype = dtype or case.dtype
    rng = np.random.default_rng(seed)
    inputs = {"x": rng.standard_normal(case.input_shape, dtype=np.float32).astype(dtype)}
    if case.kind == "functional":
        inputs["weight"] = _make_weights(case.weight_shape, scale=31.0).astype(dtype)
    return inputs


def torch_oracle(case: TransposedConv1dFrontendCase, inputs: dict[str, np.ndarray], *, dtype: str | None = None) -> np.ndarray:
    torch = __import__("torch")
    dtype = dtype or case.dtype
    torch_dtype = getattr(torch, dtype)
    x = torch.from_numpy(inputs["x"]).to(dtype=torch_dtype)
    if case.kind == "module":
        weight = torch.from_numpy(_make_weights(case.weight_shape, scale=37.0).astype(dtype)).to(dtype=torch_dtype)
    else:
        weight = torch.from_numpy(inputs["weight"]).to(dtype=torch_dtype)
    result = torch.nn.functional.conv_transpose1d(
        x,
        weight,
        bias=None,
        stride=case.stride,
        padding=case.padding,
        output_padding=case.output_padding,
        dilation=case.dilation,
        groups=1,
    )
    return result.float().cpu().numpy().astype(np.float32, copy=False)
