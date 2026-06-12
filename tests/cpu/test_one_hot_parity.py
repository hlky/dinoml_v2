from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.one_hot_parity import ONE_HOT_CASES, random_inputs, torch_oracle, trace_one_hot_spec


@pytest.mark.parametrize("case", ONE_HOT_CASES, ids=lambda case: case.name)
def test_cpu_one_hot_parity(case, tmp_path):
    spec = trace_one_hot_spec(case)
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
    np.testing.assert_array_equal(actual, expected)
