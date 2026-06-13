from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.pixel_unshuffle_frontend_parity import (
    ATOL,
    PIXEL_UNSHUFFLE_FRONTEND_CASES,
    RTOL,
    random_inputs,
    torch_oracle,
    trace_pixel_unshuffle_frontend_spec,
)


@pytest.mark.parametrize("case", PIXEL_UNSHUFFLE_FRONTEND_CASES, ids=lambda case: case.name)
def test_cpu_pixel_unshuffle_frontend_parity(case, tmp_path):
    pytest.importorskip("torch")
    spec = trace_pixel_unshuffle_frontend_spec(case)
    inputs = random_inputs(case)
    short_name = {
        "pixel_unshuffle_module_f32": "pusm",
        "pixel_unshuffle_functional_f32": "pusf",
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
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=ATOL, rtol=RTOL)
