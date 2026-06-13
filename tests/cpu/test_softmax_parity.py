from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.softmax_parity import SOFTMAX_CASES, random_inputs, torch_oracle, trace_softmax_spec


def _artifact_path(case_name: str) -> Path:
    root = Path(__file__).resolve().parents[2] / ".pytest_artifacts" / "softmax"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{case_name}.dinoml"


@pytest.mark.parametrize("case", SOFTMAX_CASES, ids=lambda case: case.name)
def test_cpu_functional_softmax_parity(case):
    torch = pytest.importorskip("torch")
    spec = trace_softmax_spec(case)
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
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
