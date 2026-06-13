from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.functional_conv2d_parity import (
    ATOL,
    FUNCTIONAL_CONV2D_CASES,
    RTOL,
    random_inputs,
    torch_oracle,
    trace_functional_conv2d_spec,
)


@pytest.mark.parametrize("case", FUNCTIONAL_CONV2D_CASES, ids=lambda case: case.name)
def test_cpu_functional_conv2d_parity(case, tmp_path):
    pytest.importorskip("torch")
    spec = trace_functional_conv2d_spec(case)
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
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=ATOL, rtol=RTOL)
