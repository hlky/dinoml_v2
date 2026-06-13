from __future__ import annotations

from pathlib import Path
import os
import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.transposed_conv1d_frontend_parity import (
    TRANSPOSED_CONV1D_FRONTEND_CASES,
    random_inputs,
    torch_oracle,
    trace_transposed_conv1d_frontend_spec,
)


_ATOL_BY_DTYPE = {"float16": 0.005, "float32": 1e-5}
_RTOL_BY_DTYPE = {"float16": 0.003, "float32": 1e-5}


def _has_rocm_runtime() -> bool:
    return shutil.which("hipcc") is not None and Path("C:/Program Files/AMD/ROCm").exists()


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)


@pytest.mark.parametrize("case", TRANSPOSED_CONV1D_FRONTEND_CASES, ids=lambda case: case.name)
def test_rocm_transposed_conv1d_frontend_parity(case, tmp_path):
    torch = pytest.importorskip("torch")
    if not _has_rocm_runtime():
        pytest.skip("ROCm toolchain/runtime is required")
    if not torch.cuda.is_available():
        pytest.skip("ROCm runtime is required")

    dtype = "float16"
    spec = trace_transposed_conv1d_frontend_spec(case, dtype=dtype)
    inputs = random_inputs(case, dtype=dtype)

    short_name = "tc1m" if case.kind == "module" else "tc1f"
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
