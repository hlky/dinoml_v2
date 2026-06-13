from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.movedim_parity import MOVEDIM_CASES, random_inputs, torch_oracle, trace_movedim_spec


@pytest.mark.parametrize("case", MOVEDIM_CASES, ids=lambda case: case.name)
def test_cpu_movedim_parity(case, tmp_path):
    spec = trace_movedim_spec(case)
    inputs = random_inputs(case)
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"{case.name}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(case, inputs)
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=1e-6, rtol=1e-6)
