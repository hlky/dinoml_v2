from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.tensor_filter_helpers_parity import (
    ATOL_BY_DTYPE,
    RTOL_BY_DTYPE,
    TENSOR_FILTER_HELPER_CASES,
    numpy_oracle,
    random_inputs,
    trace_tensor_filter_helper_spec,
)


@pytest.mark.parametrize("case", TENSOR_FILTER_HELPER_CASES, ids=lambda case: case.name)
def test_cpu_tensor_filter_helpers_parity(case, tmp_path):
    spec = trace_tensor_filter_helper_spec(case)
    inputs = random_inputs(case)
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"{case.name}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = numpy_oracle(case, inputs)
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected,
        atol=ATOL_BY_DTYPE[case.dtype],
        rtol=RTOL_BY_DTYPE[case.dtype],
    )
