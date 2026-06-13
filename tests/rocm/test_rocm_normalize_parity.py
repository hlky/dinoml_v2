from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.normalize_parity import (
    ATOL_BY_DTYPE,
    NORMALIZE_CASES,
    RTOL_BY_DTYPE,
    random_inputs,
    torch_oracle,
    trace_normalize_spec,
)


@pytest.mark.rocm
@pytest.mark.parametrize("case", NORMALIZE_CASES, ids=lambda case: case.name)
def test_rocm_normalize_parity(case, tmp_path):
    spec = trace_normalize_spec(case)
    inputs = random_inputs(case)
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{case.name}.dinoml")
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
