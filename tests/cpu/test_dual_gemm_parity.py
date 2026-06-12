from __future__ import annotations

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


@pytest.mark.parametrize("case", DUAL_GEMM_CASES, ids=lambda case: f"{case.op_name}_{case.name}")
def test_cpu_dual_gemm_parity(case, monkeypatch):
    spec = trace_dual_gemm_spec(case)
    inputs = random_inputs(case)
    artifact_root = (Path(".pytest_artifacts") / "dual_gemm" / "cpu" / case.name).resolve()
    shutil.rmtree(artifact_root, ignore_errors=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DINOML_CACHE_DIR", str(artifact_root / "cache"))

    artifact = dml.compile(spec, dml.Target("cpu"), artifact_root / "artifact.dinoml")
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
