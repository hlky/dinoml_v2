from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.runtime import load
from tests.upsample_parity import ATOL, RTOL, UPSAMPLE_CASES, random_inputs, torch_oracle, trace_upsample_spec


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


@pytest.mark.parametrize("case", UPSAMPLE_CASES, ids=lambda case: case.name)
def test_rocm_upsample_parity(case, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")
    spec = trace_upsample_spec(case)
    inputs = random_inputs(case)
    short_name = {
        "upsample_1d_linear_scale_factor": "u1ls",
        "upsample_2d_nearest_size": "u2ns",
        "upsample_3d_nearest_exact_scale_factor": "u3ne",
    }[case.name]
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{short_name}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(case, inputs)
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=ATOL, rtol=RTOL)
