from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.kernels.providers.rocm_tile.common import rocm_tile_fp32_fallback_required
from dinoml.runtime import load


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)

_TRANSPOSED_CONV2D_DTYPES = ("float16", "float32", "bfloat16")
_TRANSPOSED_CONV2D_OPS = (
    "transposed_conv2d",
    "transposed_conv2d_bias",
    "transposed_conv2d_bias_relu",
    "transposed_conv2d_bias_add",
    "transposed_conv2d_bias_add_relu",
)
_ATOL_BY_DTYPE = {"float16": 0.005, "float32": 1e-5, "bfloat16": 0.03}
_RTOL_BY_DTYPE = {"float16": 0.003, "float32": 1e-5, "bfloat16": 0.03}


class _RocmTransposeConv2dParityModule(dml.Module):
    def __init__(self, op_name: str):
        self._op_name = op_name

    def forward(self, x, weight, bias=None, residual=None):
        kwargs = {"stride": (2, 1), "padding": (1, 0), "output_padding": (1, 0), "dilation": (1, 1)}
        if self._op_name == "transposed_conv2d":
            y = dml.ops.transposed_conv2d(x, weight, **kwargs)
        elif self._op_name == "transposed_conv2d_bias":
            y = dml.ops.transposed_conv2d_bias(x, weight, bias, **kwargs)
        elif self._op_name == "transposed_conv2d_bias_relu":
            y = dml.ops.transposed_conv2d_bias_relu(x, weight, bias, **kwargs)
        elif self._op_name == "transposed_conv2d_bias_add":
            y = dml.ops.transposed_conv2d_bias_add(x, weight, bias, residual, **kwargs)
        elif self._op_name == "transposed_conv2d_bias_add_relu":
            y = dml.ops.transposed_conv2d_bias_add_relu(x, weight, bias, residual, **kwargs)
        else:
            raise ValueError(f"Unsupported transposed conv parity op {self._op_name!r}")
        return dml.ops.output(y, "y")


def _transposed_conv2d_output_shape(
    input_shape: list[int],
    weight_shape: list[int],
    *,
    stride: tuple[int, int],
    padding: tuple[int, int],
    output_padding: tuple[int, int],
    dilation: tuple[int, int],
) -> list[int]:
    batch, _in_channels, in_height, in_width = input_shape
    _weight_in_channels, out_channels, kernel_h, kernel_w = weight_shape
    return [
        batch,
        out_channels,
        (in_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_h - 1) + output_padding[0] + 1,
        (in_width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_w - 1) + output_padding[1] + 1,
    ]


def _trace_transposed_conv2d_parity_spec(op_name: str, dtype: str):
    input_shape = [1, 8, 3, 4]
    weight_shape = [8, 4, 3, 2]
    attrs = {"stride": (2, 1), "padding": (1, 0), "output_padding": (1, 0), "dilation": (1, 1)}
    residual_shape = _transposed_conv2d_output_shape(input_shape, weight_shape, **attrs)
    inputs = {
        "x": dml.TensorSpec(input_shape, dtype),
        "weight": dml.TensorSpec(weight_shape, dtype),
    }
    if op_name != "transposed_conv2d":
        inputs["bias"] = dml.TensorSpec([weight_shape[1]], dtype)
    if op_name in {"transposed_conv2d_bias_add", "transposed_conv2d_bias_add_relu"}:
        inputs["residual"] = dml.TensorSpec(residual_shape, dtype)
    spec = dml.trace(
        _RocmTransposeConv2dParityModule(op_name),
        inputs=inputs,
        name=f"rocm_{op_name}_{dtype}_transposed_conv2d_parity",
    )
    return spec, input_shape, weight_shape, residual_shape, attrs


def _random_inputs(dtype: str, *, input_shape: list[int], weight_shape: list[int], residual_shape: list[int]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    inputs = {
        "x": rng.standard_normal(input_shape, dtype=np.float32).astype(np.float32),
        "weight": rng.standard_normal(weight_shape, dtype=np.float32).astype(np.float32),
        "bias": rng.standard_normal([weight_shape[1]], dtype=np.float32).astype(np.float32),
        "residual": rng.standard_normal(residual_shape, dtype=np.float32).astype(np.float32),
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
    dtype: str,
    attrs: dict[str, tuple[int, int]],
) -> np.ndarray:
    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    device = torch.device("cuda")
    x = torch.from_numpy(inputs["x"]).to(device=device, dtype=torch_dtype)
    weight = torch.from_numpy(inputs["weight"]).to(device=device, dtype=torch_dtype)
    result = torch.nn.functional.conv_transpose2d(
        x,
        weight,
        bias=None,
        stride=attrs["stride"],
        padding=attrs["padding"],
        output_padding=attrs["output_padding"],
        dilation=attrs["dilation"],
        groups=1,
    )
    if op_name != "transposed_conv2d":
        bias = torch.from_numpy(inputs["bias"]).to(device=device, dtype=torch_dtype)
        result = result + bias.view(1, -1, 1, 1)
    if op_name in {"transposed_conv2d_bias_add", "transposed_conv2d_bias_add_relu"}:
        residual = torch.from_numpy(inputs["residual"]).to(device=device, dtype=torch_dtype)
        result = result + residual
    if op_name in {"transposed_conv2d_bias_relu", "transposed_conv2d_bias_add_relu"}:
        result = torch.relu(result)
    return result.float().cpu().numpy()


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))


@pytest.mark.parametrize("dtype", _TRANSPOSED_CONV2D_DTYPES)
@pytest.mark.parametrize("op_name", _TRANSPOSED_CONV2D_OPS)
def test_rocm_transposed_conv2d_parity_matches_torch(op_name: str, dtype: str, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")
    if dtype == "float32" and rocm_tile_fp32_fallback_required(dtype, dml.Target("rocm").to_json()):
        pytest.skip("ROCm CK float32 conv is disabled on gfx11/gfx120x and transposed conv has no ROCm Tile fallback")
    if dtype == "bfloat16":
        try:
            torch.zeros((1,), device="cuda", dtype=torch.bfloat16)
        except RuntimeError:
            pytest.skip("Torch bfloat16 ROCm device support is unavailable")

    spec, input_shape, weight_shape, residual_shape, attrs = _trace_transposed_conv2d_parity_spec(op_name, dtype)
    all_inputs = _random_inputs(dtype, input_shape=input_shape, weight_shape=weight_shape, residual_shape=residual_shape)
    spec_inputs = {name: value for name, value in all_inputs.items() if name in {item["name"] for item in spec.ir["inputs"]}}
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{op_name}_{dtype}_transposed_conv2d_parity_rocm.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(spec_inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = _torch_oracle(torch, op_name, all_inputs, dtype=dtype, attrs=attrs)
    np.testing.assert_allclose(actual, expected, atol=_ATOL_BY_DTYPE[dtype], rtol=_RTOL_BY_DTYPE[dtype])
