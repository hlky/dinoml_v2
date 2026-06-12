from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.dual_gemm_parity import (
    ATOL_BY_DTYPE,
    DUAL_GEMM_CASES,
    RTOL_BY_DTYPE,
    numpy_oracle,
    random_inputs,
    trace_dual_gemm_spec,
)


def _discover_nvcc() -> str | None:
    direct = shutil.which("nvcc")
    if direct:
        return direct
    for candidate in (os.environ.get("CUDACXX"), "/usr/local/cuda/bin/nvcc", "/usr/local/cuda-12.8/bin/nvcc"):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


_NVCC = _discover_nvcc()

pytestmark = pytest.mark.skipif(_NVCC is None, reason="nvcc is required")

_CUDA_DUAL_GEMM_CASES = tuple(
    case
    for case in DUAL_GEMM_CASES
    if case.name in {"dual_gemm_fast_gelu_f16_broadcast_dynamic", "dual_gemm_bias_fast_gelu_bf16_dynamic"}
)


@pytest.mark.parametrize("case", _CUDA_DUAL_GEMM_CASES, ids=lambda case: f"{case.op_name}_{case.name}")
def test_cuda_dual_gemm_parity(case, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is required")
    capability = torch.cuda.get_device_capability()
    if _NVCC is not None:
        os.environ.setdefault("CUDACXX", _NVCC)
        nvcc_parent = str(Path(_NVCC).parent)
        if nvcc_parent not in os.environ.get("PATH", ""):
            os.environ["PATH"] = nvcc_parent + os.pathsep + os.environ.get("PATH", "")

    spec = trace_dual_gemm_spec(case)
    inputs = random_inputs(case)
    artifact_root = (Path(".pytest_artifacts") / "dual_gemm" / "cuda" / case.name).resolve()
    shutil.rmtree(artifact_root, ignore_errors=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DINOML_CACHE_DIR", str(artifact_root / "cache"))
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch=f"sm_{capability[0]}{capability[1]}"),
        artifact_root / "artifact.dinoml",
    )
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = numpy_oracle(case, inputs)
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected.astype(np.float32),
        atol=ATOL_BY_DTYPE[case.dtype],
        rtol=RTOL_BY_DTYPE[case.dtype],
    )
