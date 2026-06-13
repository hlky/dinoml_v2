from __future__ import annotations

import os
import shutil
from pathlib import Path

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
            return candidate
    return None


_NVCC = _discover_nvcc()

pytestmark = pytest.mark.skipif(_NVCC is None, reason="nvcc is required")

_ATOL_BY_DTYPE = {"float32": 0.01}
_RTOL_BY_DTYPE = {"float32": 0.01}


@pytest.mark.parametrize("case", TRANSPOSED_CONV1D_FRONTEND_CASES, ids=lambda case: case.name)
def test_cuda_transposed_conv1d_frontend_parity(case, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is required")
    capability = torch.cuda.get_device_capability()
    if _NVCC is not None:
        os.environ.setdefault("CUDACXX", _NVCC)
        nvcc_parent = str(Path(_NVCC).parent)
        if nvcc_parent not in os.environ.get("PATH", ""):
            os.environ["PATH"] = nvcc_parent + os.pathsep + os.environ.get("PATH", "")

    spec = trace_transposed_conv1d_frontend_spec(case)
    inputs = random_inputs(case)

    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch=f"sm_{capability[0]}{capability[1]}"),
        tmp_path / f"{case.name}.dinoml",
    )
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(case, inputs)
    np.testing.assert_allclose(actual, expected, atol=_ATOL_BY_DTYPE[case.dtype], rtol=_RTOL_BY_DTYPE[case.dtype])
