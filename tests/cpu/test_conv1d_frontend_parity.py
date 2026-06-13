from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.conv1d_frontend_parity import CONV1D_FRONTEND_CASES, ATOL, RTOL, random_inputs, torch_oracle, trace_conv1d_frontend_spec


def _artifact_path(case_name: str) -> Path:
    root = Path(__file__).resolve().parents[2] / ".pytest_artifacts" / "conv1d_frontend"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{case_name}.dinoml"


@pytest.mark.parametrize("case", CONV1D_FRONTEND_CASES, ids=lambda case: case.name)
def test_cpu_conv1d_frontend_parity(case):
    torch = pytest.importorskip("torch")
    spec = trace_conv1d_frontend_spec(case)
    inputs = random_inputs(case)
    artifact = dml.compile(spec, dml.Target("cpu"), _artifact_path(case.name))
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(case, inputs)
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
