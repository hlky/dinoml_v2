from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.scatter_parity import (
    ATOL_BY_DTYPE,
    RTOL_BY_DTYPE,
    SCATTER_CASES,
    random_inputs,
    torch_oracle,
    trace_scatter_spec,
)


@pytest.mark.parametrize("case", SCATTER_CASES, ids=lambda case: case.name)
def test_cpu_scatter_family_parity(case, tmp_path):
    spec = trace_scatter_spec(case)
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
    if case.dtype == "bool":
        np.testing.assert_array_equal(actual.astype(np.bool_), expected.astype(np.bool_))
    else:
        np.testing.assert_allclose(
            actual.astype(np.float32),
            expected.astype(np.float32),
            atol=ATOL_BY_DTYPE[case.dtype],
            rtol=RTOL_BY_DTYPE[case.dtype],
        )
