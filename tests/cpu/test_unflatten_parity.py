from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.unflatten_parity import UNFLATTEN_CASES, case_inputs, torch_oracle, trace_unflatten_spec


@pytest.mark.parametrize("case", UNFLATTEN_CASES, ids=lambda case: case.name)
def test_cpu_unflatten_parity(case, tmp_path):
    spec = trace_unflatten_spec(case)
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"{case.name}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        for inputs in case_inputs(case):
            actual = session.run_numpy(inputs)["y"]
            expected = torch_oracle(case, inputs)
            np.testing.assert_array_equal(actual, expected)
    finally:
        session.close()
        module.close()
