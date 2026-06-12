from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.index_add_parity import (
    ATOL_BY_DTYPE,
    INDEX_ADD_CASES,
    RTOL_BY_DTYPE,
    random_inputs,
    torch_oracle,
    trace_index_add_spec,
)


@pytest.mark.parametrize("case", INDEX_ADD_CASES, ids=lambda case: case.name)
def test_cpu_index_add_parity(case, tmp_path):
    spec = trace_index_add_spec(case)
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
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected.astype(np.float32),
        atol=ATOL_BY_DTYPE[case.dtype],
        rtol=RTOL_BY_DTYPE[case.dtype],
    )
