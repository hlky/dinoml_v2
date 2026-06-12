from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load


def _discover_nvcc() -> str | None:
    direct = shutil.which("nvcc")
    if direct:
        return direct
    for candidate in (
        os.environ.get("CUDACXX"),
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.8/bin/nvcc",
        "/usr/local/cuda-12.9/bin/nvcc",
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


_NVCC = _discover_nvcc()

pytestmark = pytest.mark.skipif(_NVCC is None, reason="nvcc is required")

_ATOL_BY_DTYPE = {"float32": 1e-2}
_RTOL_BY_DTYPE = {"float32": 1e-2}
_OPS_BY_DTYPE = {
    "float32": ("conv3d", "conv3d_bias", "depthwise_conv3d"),
}


class _CudaConv3dParityModule(dml.Module):
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
            raise ValueError(f"Unsupported cuda conv3d parity op {self._op_name!r}")
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
    spec = dml.trace(_CudaConv3dParityModule(op_name), inputs=inputs, name=f"cuda_{op_name}_{dtype}_conv3d_parity")
    return spec, input_shape, weight_shape, output_shape, attrs


def _random_inputs(dtype: str, *, input_shape: list[int], weight_shape: list[int]) -> dict[str, np.ndarray]:
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
    dtype: str,
    attrs: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    torch_dtype = {"float32": torch.float32}[dtype]
    device = torch.device("cuda")
    x = torch.from_numpy(inputs["x"]).to(device=device, dtype=torch_dtype)
    weight = torch.from_numpy(inputs["weight"]).to(device=device, dtype=torch_dtype)
    bias = None if op_name != "conv3d_bias" else torch.from_numpy(inputs["bias"]).to(device=device, dtype=torch_dtype)
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
    return result.float().cpu().numpy()


@pytest.mark.parametrize("dtype", ("float32",))
def test_cuda_conv3d_family_parity_matches_torch(dtype: str, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is required")
    capability = torch.cuda.get_device_capability()
    if _NVCC is not None:
        os.environ.setdefault("CUDACXX", _NVCC)
        nvcc_parent = str(Path(_NVCC).parent)
        if nvcc_parent not in os.environ.get("PATH", ""):
            os.environ["PATH"] = nvcc_parent + os.pathsep + os.environ.get("PATH", "")

    for op_name in _OPS_BY_DTYPE[dtype]:
        spec, input_shape, weight_shape, _output_shape, attrs = _trace_conv3d_parity_spec(op_name, dtype)
        all_inputs = _random_inputs(dtype, input_shape=input_shape, weight_shape=weight_shape)
        spec_inputs = {name: value for name, value in all_inputs.items() if name in {item["name"] for item in spec.ir["inputs"]}}
        artifact = dml.compile(
            spec,
            dml.Target("cuda", arch=f"sm_{capability[0]}{capability[1]}"),
            tmp_path / f"{op_name}_{dtype}_conv3d_cuda_parity.dinoml",
        )
        module = load(artifact.path)
        session = module.create_session()
        try:
            actual = session.run_numpy(spec_inputs)["y"]
        finally:
            session.close()
            module.close()

        expected = _torch_oracle(torch, op_name, all_inputs, dtype=dtype, attrs=attrs)
        np.testing.assert_allclose(actual, expected, atol=_ATOL_BY_DTYPE[dtype], rtol=_RTOL_BY_DTYPE[dtype])
