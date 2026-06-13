from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.transposed_conv2d_frontend_parity import (
    ATOL,
    RTOL,
    TRANSPOSED_CONV2D_FRONTEND_CASES,
    random_inputs,
    torch_oracle,
    trace_transposed_conv2d_frontend_spec,
)


@pytest.mark.parametrize("case", TRANSPOSED_CONV2D_FRONTEND_CASES, ids=lambda case: case.name)
def test_cpu_transposed_conv2d_frontend_parity(case, tmp_path):
    torch = pytest.importorskip("torch")
    del torch
    spec = trace_transposed_conv2d_frontend_spec(case)
    inputs = random_inputs(case)

    short_name = {
        "conv_transpose2d_module_nobias_f32": "tt2mn",
        "conv_transpose2d_functional_nobias_f32": "tt2fn",
    }[case.name]
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"{short_name}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(case, inputs)
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
