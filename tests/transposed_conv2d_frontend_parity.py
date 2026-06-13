from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml


ATOL = 1e-5
RTOL = 1e-5


@dataclass(frozen=True)
class TransposedConv2dFrontendCase:
    name: str
    kind: str
    input_shape: tuple[int, int, int, int]
    weight_shape: tuple[int, int, int, int]
    dtype: str = "float32"
    stride: tuple[int, int] = (2, 1)
    padding: tuple[int, int] = (1, 0)
    output_padding: tuple[int, int] = (1, 0)
    dilation: tuple[int, int] = (1, 1)


TRANSPOSED_CONV2D_FRONTEND_CASES = (
    TransposedConv2dFrontendCase(
        name="conv_transpose2d_module_nobias_f32",
        kind="module",
        input_shape=(1, 8, 3, 4),
        weight_shape=(8, 16, 3, 2),
    ),
    TransposedConv2dFrontendCase(
        name="conv_transpose2d_functional_nobias_f32",
        kind="functional",
        input_shape=(1, 8, 3, 4),
        weight_shape=(8, 16, 3, 2),
    ),
)


def _make_weights(shape: tuple[int, ...], *, scale: float) -> np.ndarray:
    values = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return (values / scale - 0.5).astype(np.float32)


class _ConvTranspose2dModuleCase(dml.Module):
    def __init__(self, case: TransposedConv2dFrontendCase):
        self.conv = dml.nn.ConvTranspose2d(
            case.input_shape[1],
            case.weight_shape[1],
            kernel_size=case.weight_shape[2:],
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
            value=_make_weights(case.weight_shape, scale=43.0).astype(case.dtype),
        )

    def forward(self, x):
        return dml.ops.output(self.conv(x), "y")


class _ConvTranspose2dFunctionalCase(dml.Module):
    def __init__(self, case: TransposedConv2dFrontendCase):
        self.case = case

    def forward(self, x, weight):
        y = dml.nn.functional.conv_transpose2d(
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


def trace_transposed_conv2d_frontend_spec(case: TransposedConv2dFrontendCase, *, dtype: str | None = None):
    dtype = dtype or case.dtype
    case_for_dtype = case if dtype == case.dtype else TransposedConv2dFrontendCase(**{**case.__dict__, "dtype": dtype})
    if case.kind == "module":
        return dml.trace(
            _ConvTranspose2dModuleCase(case_for_dtype),
            inputs={"x": dml.TensorSpec(list(case.input_shape), dtype)},
            name=case.name,
        )
    inputs = {
        "x": dml.TensorSpec(list(case.input_shape), dtype),
        "weight": dml.TensorSpec(list(case.weight_shape), dtype),
    }
    return dml.trace(_ConvTranspose2dFunctionalCase(case_for_dtype), inputs=inputs, name=case.name)


def random_inputs(case: TransposedConv2dFrontendCase, *, dtype: str | None = None, seed: int = 7) -> dict[str, np.ndarray]:
    dtype = dtype or case.dtype
    rng = np.random.default_rng(seed)
    inputs = {"x": rng.standard_normal(case.input_shape, dtype=np.float32).astype(dtype)}
    if case.kind == "functional":
        inputs["weight"] = _make_weights(case.weight_shape, scale=41.0).astype(dtype)
    return inputs


def torch_oracle(case: TransposedConv2dFrontendCase, inputs: dict[str, np.ndarray], *, dtype: str | None = None) -> np.ndarray:
    torch = __import__("torch")
    dtype = dtype or case.dtype
    torch_dtype = getattr(torch, dtype)
    x = torch.from_numpy(inputs["x"]).to(dtype=torch_dtype)
    if case.kind == "module":
        weight = torch.from_numpy(_make_weights(case.weight_shape, scale=43.0).astype(dtype)).to(dtype=torch_dtype)
    else:
        weight = torch.from_numpy(inputs["weight"]).to(dtype=torch_dtype)
    result = torch.nn.functional.conv_transpose2d(
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
