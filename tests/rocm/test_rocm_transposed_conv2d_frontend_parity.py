from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.kernels.providers.rocm_tile.common import rocm_tile_fp32_fallback_required
from dinoml.runtime import load
from tests.transposed_conv2d_frontend_parity import (
    TRANSPOSED_CONV2D_FRONTEND_CASES,
    random_inputs,
    torch_oracle,
    trace_transposed_conv2d_frontend_spec,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)

_ATOL_BY_DTYPE = {"float16": 0.04, "float32": 1e-5}
_RTOL_BY_DTYPE = {"float16": 0.01, "float32": 1e-5}


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))


@pytest.mark.parametrize("case", TRANSPOSED_CONV2D_FRONTEND_CASES, ids=lambda case: case.name)
def test_rocm_transposed_conv2d_frontend_parity(case, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")
    dtype = "float16"
    if dtype == "float32" and rocm_tile_fp32_fallback_required(dtype, dml.Target("rocm").to_json()):
        pytest.skip("ROCm CK float32 conv is disabled on gfx11/gfx120x and transposed conv has no ROCm Tile fallback")

    spec = trace_transposed_conv2d_frontend_spec(case, dtype=dtype)
    inputs = random_inputs(case, dtype=dtype)

    short_name = {
        "conv_transpose2d_module_nobias_f32": "tt2mn",
        "conv_transpose2d_functional_nobias_f32": "tt2fn",
    }[case.name]
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{short_name}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(case, inputs, dtype=dtype)
    np.testing.assert_allclose(actual, expected, atol=_ATOL_BY_DTYPE[dtype], rtol=_RTOL_BY_DTYPE[dtype])
