from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.masked_select_parity import (
    ATOL_BY_DTYPE,
    MASKED_SELECT_CASES,
    RTOL_BY_DTYPE,
    numpy_oracle,
    random_inputs,
    trace_masked_select_spec,
)


@pytest.mark.parametrize("case", MASKED_SELECT_CASES, ids=lambda case: case.name)
def test_cpu_masked_select_parity(case, tmp_path):
    spec = trace_masked_select_spec(case)
    inputs = random_inputs(case)
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"{case.name}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
        reported_shape = session.get_output_shape("y")
    finally:
        session.close()
        module.close()

    expected = numpy_oracle(case, inputs)
    assert tuple(actual.shape) == tuple(expected.shape)
    assert tuple(reported_shape) == tuple(expected.shape)
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected.astype(np.float32),
        atol=ATOL_BY_DTYPE[case.dtype],
        rtol=RTOL_BY_DTYPE[case.dtype],
    )
