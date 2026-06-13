from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml


ATOL = 1e-5
RTOL = 1e-5


@dataclass(frozen=True)
class Conv3dFrontendCase:
    name: str
    kind: str
    input_shape: tuple[int, int, int, int, int]
    weight_shape: tuple[int, int, int, int, int]
    use_bias: bool
    dtype: str = "float32"
    stride: tuple[int, int, int] = (2, 1, 1)
    padding: tuple[int, int, int] = (1, 1, 0)
    dilation: tuple[int, int, int] = (1, 1, 1)

    @property
    def bias_shape(self) -> tuple[int]:
        return (int(self.weight_shape[0]),)


CONV3D_FRONTEND_CASES = (
    Conv3dFrontendCase(
        name="conv3d_module_bias_f32",
        kind="module",
        input_shape=(2, 3, 5, 6, 7),
        weight_shape=(5, 3, 3, 3, 2),
        use_bias=True,
    ),
    Conv3dFrontendCase(
        name="conv3d_module_nobias_f32",
        kind="module",
        input_shape=(2, 3, 5, 6, 7),
        weight_shape=(5, 3, 3, 3, 2),
        use_bias=False,
    ),
    Conv3dFrontendCase(
        name="conv3d_functional_bias_f32",
        kind="functional",
        input_shape=(2, 3, 5, 6, 7),
        weight_shape=(5, 3, 3, 3, 2),
        use_bias=True,
    ),
    Conv3dFrontendCase(
        name="conv3d_functional_nobias_f32",
        kind="functional",
        input_shape=(2, 3, 5, 6, 7),
        weight_shape=(5, 3, 3, 3, 2),
        use_bias=False,
    ),
)


def _make_weights(shape: tuple[int, ...], *, scale: float) -> np.ndarray:
    values = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return (values / scale - 0.5).astype(np.float32)


def _make_bias(shape: tuple[int, ...], *, scale: float) -> np.ndarray:
    values = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return (values / scale - 0.25).astype(np.float32)


class _Conv3dModuleCase(dml.Module):
    def __init__(self, case: Conv3dFrontendCase):
        self.conv = dml.nn.Conv3d(
            case.input_shape[1],
            case.weight_shape[0],
            kernel_size=case.weight_shape[2:],
            stride=case.stride,
            padding=case.padding,
            dilation=case.dilation,
            bias=case.use_bias,
            dtype=case.dtype,
        )
        self.conv.weight = dml.Parameter(
            list(case.weight_shape),
            dtype=case.dtype,
            value=_make_weights(case.weight_shape, scale=43.0).astype(case.dtype),
        )
        if case.use_bias:
            self.conv.bias = dml.Parameter(
                list(case.bias_shape),
                dtype=case.dtype,
                value=_make_bias(case.bias_shape, scale=19.0).astype(case.dtype),
            )

    def forward(self, x):
        return dml.ops.output(self.conv(x), "y")


class _Conv3dFunctionalCase(dml.Module):
    def __init__(self, case: Conv3dFrontendCase):
        self.case = case

    def forward(self, x, weight, bias=None):
        y = dml.nn.functional.conv3d(
            x,
            weight,
            bias,
            stride=self.case.stride,
            padding=self.case.padding,
            dilation=self.case.dilation,
            groups=1,
        )
        return dml.ops.output(y, "y")


def trace_conv3d_frontend_spec(case: Conv3dFrontendCase, *, dtype: str | None = None):
    dtype = dtype or case.dtype
    case_for_dtype = case if dtype == case.dtype else Conv3dFrontendCase(**{**case.__dict__, "dtype": dtype})
    if case.kind == "module":
        return dml.trace(
            _Conv3dModuleCase(case_for_dtype),
            inputs={"x": dml.TensorSpec(list(case.input_shape), dtype)},
            name=case.name,
        )
    inputs = {
        "x": dml.TensorSpec(list(case.input_shape), dtype),
        "weight": dml.TensorSpec(list(case.weight_shape), dtype),
    }
    if case.use_bias:
        inputs["bias"] = dml.TensorSpec(list(case.bias_shape), dtype)
    return dml.trace(
        _Conv3dFunctionalCase(case_for_dtype),
        inputs=inputs,
        name=case.name,
    )


def random_inputs(case: Conv3dFrontendCase, *, dtype: str | None = None, seed: int = 7) -> dict[str, np.ndarray]:
    dtype = dtype or case.dtype
    rng = np.random.default_rng(seed)
    inputs = {"x": rng.standard_normal(case.input_shape, dtype=np.float32).astype(dtype)}
    if case.kind == "functional":
        inputs["weight"] = _make_weights(case.weight_shape, scale=41.0).astype(dtype)
        if case.use_bias:
            inputs["bias"] = _make_bias(case.bias_shape, scale=23.0).astype(dtype)
    return inputs


def torch_oracle(case: Conv3dFrontendCase, inputs: dict[str, np.ndarray], *, dtype: str | None = None) -> np.ndarray:
    torch = __import__("torch")
    dtype = dtype or case.dtype
    torch_dtype = getattr(torch, dtype)
    x = torch.from_numpy(inputs["x"]).to(dtype=torch_dtype)
    if case.kind == "module":
        weight = torch.from_numpy(_make_weights(case.weight_shape, scale=43.0).astype(dtype)).to(dtype=torch_dtype)
        bias = (
            torch.from_numpy(_make_bias(case.bias_shape, scale=19.0).astype(dtype)).to(dtype=torch_dtype)
            if case.use_bias
            else None
        )
    else:
        weight = torch.from_numpy(inputs["weight"]).to(dtype=torch_dtype)
        bias = torch.from_numpy(inputs["bias"]).to(dtype=torch_dtype) if case.use_bias else None
    result = torch.nn.functional.conv3d(
        x,
        weight,
        bias=bias,
        stride=case.stride,
        padding=case.padding,
        dilation=case.dilation,
        groups=1,
    )
    return result.float().cpu().numpy().astype(np.float32, copy=False)
