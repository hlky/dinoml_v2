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
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


_NVCC = _discover_nvcc()

pytestmark = pytest.mark.skipif(_NVCC is None, reason="nvcc is required")

_TRANSPOSED_CONV1D_CUDA_DTYPES = ("float16", "float32", "bfloat16")
_ATOL_BY_DTYPE = {"float16": 0.005, "float32": 0.01, "bfloat16": 0.03}
_RTOL_BY_DTYPE = {"float16": 0.003, "float32": 0.01, "bfloat16": 0.03}


class _CudaTransposeConv1dParityModule(dml.Module):
    def forward(self, x, weight):
        y = dml.ops.transposed_conv1d(
            x,
            weight,
            stride=2,
            padding=1,
            output_padding=1,
            dilation=1,
        )
        return dml.ops.output(y, "y")


@pytest.mark.parametrize("dtype", _TRANSPOSED_CONV1D_CUDA_DTYPES)
def test_cuda_transposed_conv1d_parity_matches_torch(dtype: str, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is required")
    capability = torch.cuda.get_device_capability()
    if dtype == "bfloat16" and capability < (8, 0):
        pytest.skip("bfloat16 parity requires CUDA sm_80 or newer")

    spec = dml.trace(
        _CudaTransposeConv1dParityModule(),
        inputs={
            "x": dml.TensorSpec([1, 8, 4], dtype),
            "weight": dml.TensorSpec([8, 16, 3], dtype),
        },
        name=f"cuda_transposed_conv1d_{dtype}_parity",
    )
    all_inputs = _random_inputs(dtype)
    if _NVCC is not None:
        os.environ.setdefault("CUDACXX", _NVCC)
        nvcc_parent = str(Path(_NVCC).parent)
        if nvcc_parent not in os.environ.get("PATH", ""):
            os.environ["PATH"] = nvcc_parent + os.pathsep + os.environ.get("PATH", "")
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch=f"sm_{capability[0]}{capability[1]}"),
        tmp_path / f"transposed_conv1d_{dtype}_cuda_parity.dinoml",
    )
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(all_inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = _torch_transposed_conv1d_oracle(torch, all_inputs, dtype=dtype)
    np.testing.assert_allclose(actual, expected, atol=_ATOL_BY_DTYPE[dtype], rtol=_RTOL_BY_DTYPE[dtype])


def _random_inputs(dtype: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    inputs = {
        "x": rng.standard_normal([1, 8, 4], dtype=np.float32).astype(np.float32),
        "weight": rng.standard_normal([8, 16, 3], dtype=np.float32).astype(np.float32),
    }
    if dtype == "float16":
        return {name: value.astype(np.float16) for name, value in inputs.items()}
    if dtype == "bfloat16":
        return inputs
    return inputs


def _torch_transposed_conv1d_oracle(torch, inputs: dict[str, np.ndarray], *, dtype: str) -> np.ndarray:
    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    device = torch.device("cuda")
    x = torch.from_numpy(inputs["x"]).to(device=device, dtype=torch_dtype)
    weight = torch.from_numpy(inputs["weight"]).to(device=device, dtype=torch_dtype)
    result = torch.nn.functional.conv_transpose1d(
        x,
        weight,
        bias=None,
        stride=2,
        padding=1,
        output_padding=1,
        dilation=1,
        groups=1,
    )
    return result.float().cpu().numpy()
