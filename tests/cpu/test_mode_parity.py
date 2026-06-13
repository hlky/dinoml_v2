from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.mode_parity import (
    ATOL_BY_DTYPE,
    MODE_CASES,
    RTOL_BY_DTYPE,
    case_inputs,
    torch_oracle,
    trace_mode_spec,
)


@pytest.mark.parametrize("case", MODE_CASES, ids=lambda case: case.name)
def test_cpu_mode_parity(case, tmp_path):
    spec = trace_mode_spec(case)
    inputs = case_inputs(case)
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"{case.name}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)
    finally:
        session.close()
        module.close()

    expected_values, expected_indices = torch_oracle(case, inputs)
    if case.dtype == "bool":
        np.testing.assert_array_equal(actual["values"].astype(np.bool_), expected_values.astype(np.bool_))
    else:
        np.testing.assert_allclose(
            actual["values"].astype(np.float32),
            expected_values.astype(np.float32),
            atol=ATOL_BY_DTYPE[case.dtype],
            rtol=RTOL_BY_DTYPE[case.dtype],
        )
    np.testing.assert_array_equal(actual["indices"].astype(np.int64), expected_indices.astype(np.int64))
