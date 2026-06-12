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

_TRANSPOSED_CONV1D_DTYPES = ("float16", "float32", "bfloat16")
_ATOL_BY_DTYPE = {"float16": 0.005, "float32": 1e-5, "bfloat16": 0.03}
_RTOL_BY_DTYPE = {"float16": 0.003, "float32": 1e-5, "bfloat16": 0.03}


class _RocmTransposeConv1dParityModule(dml.Module):
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


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))


@pytest.mark.parametrize("dtype", _TRANSPOSED_CONV1D_DTYPES)
def test_rocm_transposed_conv1d_parity_matches_torch(dtype: str, tmp_path):
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

    spec = dml.trace(
        _RocmTransposeConv1dParityModule(),
        inputs={
            "x": dml.TensorSpec([1, 8, 4], dtype),
            "weight": dml.TensorSpec([8, 16, 3], dtype),
        },
        name=f"rocm_transposed_conv1d_{dtype}_parity",
    )
    all_inputs = _random_inputs(dtype)
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"transposed_conv1d_{dtype}_parity_rocm.dinoml")
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
