from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import dinoml as dml
import dinoml.runtime as runtime_mod
from dinoml.backends import rocm as rocm_backend
from dinoml.kernels.providers.rocm_tile.common import rocm_tile_fp32_fallback_required
from dinoml.runtime import load


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)

_CONV1D_DTYPES = ("float16", "float32", "bfloat16")
_CONV1D_OPS = (
    "conv1d_bias",
    "conv1d_bias_relu",
    "conv1d_bias_add",
    "conv1d_bias_add_relu",
)
_ATOL_BY_DTYPE = {"float16": 0.005, "float32": 1e-5, "bfloat16": 0.03}
_RTOL_BY_DTYPE = {"float16": 0.003, "float32": 1e-5, "bfloat16": 0.03}


class _RocmConv1dParityModule(dml.Module):
    def __init__(self, op_name: str):
        self._op_name = op_name

    def forward(self, x, weight, bias, residual=None):
        kwargs = {"stride": 2, "padding": 1, "dilation": 1}
        if self._op_name == "conv1d_bias":
            y = dml.ops.conv1d_bias(x, weight, bias, **kwargs)
        elif self._op_name == "conv1d_bias_relu":
            y = dml.ops.conv1d_bias_relu(x, weight, bias, **kwargs)
        elif self._op_name == "conv1d_bias_add":
            y = dml.ops.conv1d_bias_add(x, weight, bias, residual, **kwargs)
        elif self._op_name == "conv1d_bias_add_relu":
            y = dml.ops.conv1d_bias_add_relu(x, weight, bias, residual, **kwargs)
        else:
            raise ValueError(f"Unsupported conv1d parity op {self._op_name!r}")
        return dml.ops.output(y, "y")


def _conv1d_output_shape(
    input_shape: list[int],
    weight_shape: list[int],
    *,
    stride: int,
    padding: int,
    dilation: int,
) -> list[int]:
    batch, _in_channels, in_width = input_shape
    out_channels, _weight_in_channels, kernel_w = weight_shape
    out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    return [batch, out_channels, out_width]


def _trace_conv1d_parity_spec(op_name: str, dtype: str):
    input_shape = [2, 3, 9]
    weight_shape = [5, 3, 3]
    attrs = {"stride": 2, "padding": 1, "dilation": 1}
    residual_shape = _conv1d_output_shape(input_shape, weight_shape, **attrs)
    inputs = {
        "x": dml.TensorSpec(input_shape, dtype),
        "weight": dml.TensorSpec(weight_shape, dtype),
        "bias": dml.TensorSpec([weight_shape[0]], dtype),
    }
    if op_name in {"conv1d_bias_add", "conv1d_bias_add_relu"}:
        inputs["residual"] = dml.TensorSpec(residual_shape, dtype)
    spec = dml.trace(
        _RocmConv1dParityModule(op_name),
        inputs=inputs,
        name=f"rocm_{op_name}_{dtype}_conv1d_parity",
    )
    return spec, input_shape, weight_shape, residual_shape, attrs


def _random_inputs(dtype: str, *, input_shape: list[int], weight_shape: list[int], residual_shape: list[int]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    inputs = {
        "x": rng.standard_normal(input_shape, dtype=np.float32).astype(np.float32),
        "weight": rng.standard_normal(weight_shape, dtype=np.float32).astype(np.float32),
        "bias": rng.standard_normal([weight_shape[0]], dtype=np.float32).astype(np.float32),
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
    attrs: dict[str, int],
) -> np.ndarray:
    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    device = torch.device("cuda")
    x = torch.from_numpy(inputs["x"]).to(device=device, dtype=torch_dtype)
    weight = torch.from_numpy(inputs["weight"]).to(device=device, dtype=torch_dtype)
    result = torch.nn.functional.conv1d(
        x,
        weight,
        bias=None,
        stride=attrs["stride"],
        padding=attrs["padding"],
        dilation=attrs["dilation"],
        groups=1,
    )
    bias = torch.from_numpy(inputs["bias"]).to(device=device, dtype=torch_dtype)
    result = result + bias.view(1, -1, 1)
    if op_name in {"conv1d_bias_add", "conv1d_bias_add_relu"}:
        residual = torch.from_numpy(inputs["residual"]).to(device=device, dtype=torch_dtype)
        result = result + residual
    if op_name in {"conv1d_bias_relu", "conv1d_bias_add_relu"}:
        result = torch.relu(result)
    return result.float().cpu().numpy()


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))


@pytest.mark.parametrize("dtype", _CONV1D_DTYPES)
@pytest.mark.parametrize("op_name", _CONV1D_OPS)
def test_rocm_conv1d_parity_matches_torch(op_name: str, dtype: str, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")
    if dtype == "float32" and rocm_tile_fp32_fallback_required(dtype, dml.Target("rocm").to_json()):
        pytest.skip("ROCm CK float32 conv is disabled on gfx11/gfx120x and conv1d has no ROCm Tile fallback")
    if dtype == "bfloat16":
        try:
            torch.zeros((1,), device="cuda", dtype=torch.bfloat16)
        except RuntimeError:
            pytest.skip("Torch bfloat16 ROCm device support is unavailable")

    spec, input_shape, weight_shape, residual_shape, attrs = _trace_conv1d_parity_spec(op_name, dtype)
    all_inputs = _random_inputs(dtype, input_shape=input_shape, weight_shape=weight_shape, residual_shape=residual_shape)
    spec_inputs = {name: value for name, value in all_inputs.items() if name in {item["name"] for item in spec.ir["inputs"]}}
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{op_name}_{dtype}_conv1d_parity_rocm.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(spec_inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = _torch_oracle(torch, op_name, all_inputs, dtype=dtype, attrs=attrs)
    np.testing.assert_allclose(actual, expected, atol=_ATOL_BY_DTYPE[dtype], rtol=_RTOL_BY_DTYPE[dtype])


def test_rocm_conv1d_constant_weight_prepacked_native_load_matches_torch(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")

    dtype = "float16"
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec, attrs, all_inputs = _trace_constant_weight_conv1d_spec(dtype)
    expected = _torch_oracle(torch, "conv1d_bias_add_relu", all_inputs, dtype=dtype, attrs=attrs)
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / "conv1d_constant_weight_prepacked_rocm.dinoml")

    def fail_set_constant_numpy(self, name, value):
        raise AssertionError(f"Python constant setter should not run for prepacked CK conv1d constant {name!r}")

    monkeypatch.setattr(runtime_mod.RuntimeModule, "set_constant_numpy", fail_set_constant_numpy)
    module = load(artifact.path)
    session = module.create_session()
    try:
        weight_spec = next(constant for constant in module.metadata["constants"] if constant["name"] == "weight")
        assert weight_spec["storage"]["kind"] == "ck_conv1d_weight"
        assert module._autoloadable_constants_require_native_loader() is True
        actual = session.run_numpy({"x": all_inputs["x"], "residual": all_inputs["residual"]})["y"]
    finally:
        session.close()
        module.close()

    np.testing.assert_allclose(actual, expected, atol=_ATOL_BY_DTYPE[dtype], rtol=_RTOL_BY_DTYPE[dtype])


def _trace_constant_weight_conv1d_spec(dtype: str) -> tuple[dml.ModelSpec, dict[str, int], dict[str, np.ndarray]]:
    weight_value = (np.arange(5 * 3 * 3, dtype=np.float32).reshape(5, 3, 3) / 37.0).astype(np.float16)
    bias_value = (np.arange(5, dtype=np.float32) / 13.0).astype(np.float16)
    attrs = {"stride": 2, "padding": 1, "dilation": 1}
    input_shape = [2, 3, 9]
    residual_shape = _conv1d_output_shape(input_shape, list(weight_value.shape), **attrs)
    rng = np.random.default_rng(11)
    all_inputs = {
        "x": rng.standard_normal(input_shape, dtype=np.float32).astype(np.float16),
        "residual": rng.standard_normal(residual_shape, dtype=np.float32).astype(np.float16),
        "weight": weight_value,
        "bias": bias_value,
    }

    class _ConstWeightConv1dModule(dml.Module):
        def __init__(self):
            self.weight = dml.Parameter(list(weight_value.shape), dtype=dtype, value=weight_value)
            self.bias = dml.Parameter([weight_value.shape[0]], dtype=dtype, value=bias_value)

        def forward(self, x, residual):
            y = dml.ops.conv1d_bias_add_relu(x, self.weight, self.bias, residual, **attrs)
            return dml.ops.output(y, "y")

    spec = dml.trace(
        _ConstWeightConv1dModule(),
        inputs={
            "x": dml.TensorSpec(input_shape, dtype),
            "residual": dml.TensorSpec(residual_shape, dtype),
        },
        name="rocm_conv1d_constant_weight_prepacked_parity",
    )
    return spec, attrs, all_inputs
