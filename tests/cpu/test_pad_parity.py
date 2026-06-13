from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.pad_parity import PAD_CASES, random_inputs, torch_oracle, trace_pad_spec


@pytest.mark.parametrize("case", PAD_CASES, ids=lambda case: case.name)
def test_cpu_pad_parity(case, tmp_path):
    torch = pytest.importorskip("torch")
    spec = trace_pad_spec(case)
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
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
