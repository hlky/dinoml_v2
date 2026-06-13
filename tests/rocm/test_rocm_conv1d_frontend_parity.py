from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.kernels.providers.rocm_tile.common import rocm_tile_fp32_fallback_required
from dinoml.runtime import load
from tests.conv1d_frontend_parity import CONV1D_FRONTEND_CASES, ATOL, RTOL, random_inputs, torch_oracle, trace_conv1d_frontend_spec


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))


@pytest.mark.parametrize("case", CONV1D_FRONTEND_CASES, ids=lambda case: case.name)
def test_rocm_conv1d_frontend_parity(case, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")
    dtype = "float16" if rocm_tile_fp32_fallback_required("float32", dml.Target("rocm").to_json()) else "float32"

    spec = trace_conv1d_frontend_spec(case, dtype=dtype)
    inputs = random_inputs(case, dtype=dtype)
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{case.name}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(case, inputs, dtype=dtype)
    atol = 0.005 if dtype == "float16" else ATOL
    rtol = 0.003 if dtype == "float16" else RTOL
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
