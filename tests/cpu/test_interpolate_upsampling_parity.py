from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.interpolate_parity import (
    ATOL,
    INTERPOLATE_CASES,
    RTOL,
    random_inputs,
    torch_oracle,
    trace_interpolate_spec,
)


@pytest.mark.parametrize("case", INTERPOLATE_CASES, ids=lambda case: case.name)
def test_cpu_interpolate_upsampling_parity(case, tmp_path):
    pytest.importorskip("torch")
    spec = trace_interpolate_spec(case)
    inputs = random_inputs(case)
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "artifact.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(case, inputs)
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=ATOL, rtol=RTOL)
