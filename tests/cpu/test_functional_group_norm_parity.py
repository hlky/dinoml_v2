from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.functional_group_norm_parity import (
    ATOL_BY_DTYPE,
    FUNCTIONAL_GROUP_NORM_CASES,
    RTOL_BY_DTYPE,
    random_inputs,
    torch_oracle,
    trace_functional_group_norm_spec,
)


def _artifact_path(case_name: str) -> Path:
    root = Path(__file__).resolve().parents[2] / ".pytest_artifacts" / "functional_group_norm"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{case_name}.dinoml"


@pytest.mark.parametrize("case", FUNCTIONAL_GROUP_NORM_CASES[:2], ids=lambda case: case.name)
def test_cpu_functional_group_norm_parity(case):
    torch = pytest.importorskip("torch")
    spec = trace_functional_group_norm_spec(case)
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
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected,
        atol=ATOL_BY_DTYPE[case.dtype],
        rtol=RTOL_BY_DTYPE[case.dtype],
    )
