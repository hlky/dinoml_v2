from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.runtime import load
from tests.dual_gemm_parity import (
    ATOL_BY_DTYPE,
    DUAL_GEMM_CASES,
    RTOL_BY_DTYPE,
    numpy_oracle,
    random_inputs,
    trace_dual_gemm_spec,
)


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


_ROCM_DUAL_GEMM_CASES = tuple(
    case
    for case in DUAL_GEMM_CASES
    if case.name in {"dual_gemm_fast_gelu_f16_broadcast_dynamic", "dual_gemm_bias_fast_gelu_bf16_dynamic"}
)


@pytest.mark.parametrize("case", _ROCM_DUAL_GEMM_CASES, ids=lambda case: f"{case.op_name}_{case.name}")
def test_rocm_dual_gemm_parity(case, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")

    spec = trace_dual_gemm_spec(case)
    inputs = random_inputs(case)
    artifact_root = (Path(".pytest_artifacts") / "dual_gemm" / "rocm" / case.name).resolve()
    shutil.rmtree(artifact_root, ignore_errors=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DINOML_CACHE_DIR", str(artifact_root / "cache"))
    artifact = dml.compile(spec, dml.Target("rocm"), artifact_root / "artifact.dinoml")
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
