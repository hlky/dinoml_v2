from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import dinoml as dml
import dinoml.runtime as runtime_mod
from dinoml.backends import rocm as rocm_backend
from dinoml.runtime import load


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)

_ATOL = 0.005
_RTOL = 0.003


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))


def test_rocm_conv2d_constant_weight_prepacked_native_load_matches_raw_ck_runtime(tmp_path, monkeypatch):
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")

    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec, all_inputs, attrs = _trace_constant_weight_conv2d_spec()
    raw_spec = _trace_raw_weight_conv2d_spec()
    prepacked_artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / "conv2d_constant_weight_prepacked_rocm.dinoml")
    raw_artifact = dml.compile(raw_spec, dml.Target("rocm"), tmp_path / "conv2d_raw_weight_rocm.dinoml")

    def fail_set_constant_numpy(self, name, value):
        raise AssertionError(f"Python constant setter should not run for prepacked CK conv2d constant {name!r}")

    monkeypatch.setattr(runtime_mod.RuntimeModule, "set_constant_numpy", fail_set_constant_numpy)
    module = load(prepacked_artifact.path)
    session = module.create_session()
    try:
        weight_spec = next(constant for constant in module.metadata["constants"] if constant["name"] == "weight")
        assert weight_spec["storage"]["kind"] == "ck_conv2d_weight"
        assert module._autoloadable_constants_require_native_loader() is True
        actual = session.run_numpy({"x": all_inputs["x"], "residual": all_inputs["residual"]})["y"]
    finally:
        session.close()
        module.close()

    raw_module = load(raw_artifact.path)
    raw_session = raw_module.create_session()
    try:
        expected = raw_session.run_numpy(all_inputs)["y"]
    finally:
        raw_session.close()
        raw_module.close()

    np.testing.assert_allclose(actual, expected, atol=_ATOL, rtol=_RTOL)


def test_rocm_conv3d_constant_weight_prepacked_native_load_matches_torch(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")

    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    spec, all_inputs, attrs = _trace_constant_weight_conv3d_spec()
    expected = _torch_conv3d_oracle(torch, all_inputs, attrs=attrs)
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / "conv3d_constant_weight_prepacked_rocm.dinoml")

    def fail_set_constant_numpy(self, name, value):
        raise AssertionError(f"Python constant setter should not run for prepacked CK conv3d constant {name!r}")

    monkeypatch.setattr(runtime_mod.RuntimeModule, "set_constant_numpy", fail_set_constant_numpy)
    module = load(artifact.path)
    session = module.create_session()
    try:
        weight_spec = next(constant for constant in module.metadata["constants"] if constant["name"] == "weight")
        assert weight_spec["storage"]["kind"] == "ck_conv3d_weight"
        assert module._autoloadable_constants_require_native_loader() is True
        actual = session.run_numpy({"x": all_inputs["x"]})["y"]
    finally:
        session.close()
        module.close()

    np.testing.assert_allclose(actual, expected, atol=_ATOL, rtol=_RTOL)


def _trace_constant_weight_conv2d_spec() -> tuple[dml.ModelSpec, dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
    dtype = "float16"
    weight_value = (np.arange(5 * 3 * 3 * 2, dtype=np.float32).reshape(5, 3, 3, 2) / 41.0).astype(np.float16)
    bias_value = (np.arange(5, dtype=np.float32) / 17.0).astype(np.float16)
    attrs = {"stride": (1, 1), "padding": (1, 0), "dilation": (1, 1)}
    rng = np.random.default_rng(23)
    all_inputs = {
        "x": rng.standard_normal((2, 3, 7, 6), dtype=np.float32).astype(np.float16),
        "residual": rng.standard_normal((2, 5, 7, 5), dtype=np.float32).astype(np.float16),
        "weight": weight_value,
        "bias": bias_value,
    }

    class _ConstWeightConv2dModule(dml.Module):
        def __init__(self):
            self.weight = dml.Parameter(list(weight_value.shape), dtype=dtype, value=weight_value)
            self.bias = dml.Parameter([weight_value.shape[0]], dtype=dtype, value=bias_value)

        def forward(self, x, residual):
            y = dml.ops.conv2d_bias_add_relu(x, self.weight, self.bias, residual, **attrs)
            return dml.ops.output(y, "y")

    spec = dml.trace(
        _ConstWeightConv2dModule(),
        inputs={
            "x": dml.TensorSpec([2, 3, 7, 6], dtype),
            "residual": dml.TensorSpec([2, 5, 7, 5], dtype),
        },
        name="rocm_conv2d_constant_weight_prepacked_parity",
    )
    return spec, all_inputs, attrs


def _trace_raw_weight_conv2d_spec() -> dml.ModelSpec:
    dtype = "float16"
    attrs = {"stride": (1, 1), "padding": (1, 0), "dilation": (1, 1)}

    class _RawWeightConv2dModule(dml.Module):
        def forward(self, x, weight, bias, residual):
            y = dml.ops.conv2d_bias_add_relu(x, weight, bias, residual, **attrs)
            return dml.ops.output(y, "y")

    return dml.trace(
        _RawWeightConv2dModule(),
        inputs={
            "x": dml.TensorSpec([2, 3, 7, 6], dtype),
            "weight": dml.TensorSpec([5, 3, 3, 2], dtype),
            "bias": dml.TensorSpec([5], dtype),
            "residual": dml.TensorSpec([2, 5, 7, 5], dtype),
        },
        name="rocm_conv2d_raw_weight_parity_reference",
    )


def _trace_constant_weight_conv3d_spec() -> tuple[dml.ModelSpec, dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
    dtype = "float16"
    weight_value = (np.arange(5 * 3 * 2 * 3 * 2, dtype=np.float32).reshape(5, 3, 2, 3, 2) / 29.0).astype(np.float16)
    bias_value = (np.arange(5, dtype=np.float32) / 19.0).astype(np.float16)
    attrs = {"stride": (1, 1, 1), "padding": (0, 1, 0), "dilation": (1, 1, 1)}
    rng = np.random.default_rng(31)
    all_inputs = {
        "x": rng.standard_normal((2, 3, 4, 6, 5), dtype=np.float32).astype(np.float16),
        "weight": weight_value,
        "bias": bias_value,
    }

    class _ConstWeightConv3dModule(dml.Module):
        def __init__(self):
            self.weight = dml.Parameter(list(weight_value.shape), dtype=dtype, value=weight_value)
            self.bias = dml.Parameter([weight_value.shape[0]], dtype=dtype, value=bias_value)

        def forward(self, x):
            y = dml.ops.conv3d_bias(x, self.weight, self.bias, **attrs)
            return dml.ops.output(y, "y")

    spec = dml.trace(
        _ConstWeightConv3dModule(),
        inputs={"x": dml.TensorSpec([2, 3, 4, 6, 5], dtype)},
        name="rocm_conv3d_constant_weight_prepacked_parity",
    )
    return spec, all_inputs, attrs


def _torch_conv3d_oracle(torch, inputs: dict[str, np.ndarray], *, attrs: dict[str, tuple[int, ...]]) -> np.ndarray:
    x = torch.from_numpy(inputs["x"]).to(dtype=torch.float32)
    weight = torch.from_numpy(inputs["weight"]).to(dtype=torch.float32)
    bias = torch.from_numpy(inputs["bias"]).to(dtype=torch.float32)
    result = torch.nn.functional.conv3d(
        x,
        weight,
        bias=bias,
        stride=attrs["stride"],
        padding=attrs["padding"],
        dilation=attrs["dilation"],
    )
    return result.cpu().numpy()
