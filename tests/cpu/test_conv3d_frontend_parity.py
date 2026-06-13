from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.conv3d_frontend_parity import ATOL, CONV3D_FRONTEND_CASES, RTOL, random_inputs, torch_oracle, trace_conv3d_frontend_spec


@pytest.mark.parametrize("case", CONV3D_FRONTEND_CASES, ids=lambda case: case.name)
def test_cpu_conv3d_frontend_parity(case, tmp_path):
    torch = pytest.importorskip("torch")
    spec = trace_conv3d_frontend_spec(case)
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
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
