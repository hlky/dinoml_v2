from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.upsample_parity import ATOL, RTOL, UPSAMPLE_CASES, random_inputs, torch_oracle, trace_upsample_spec


@pytest.mark.parametrize("case", UPSAMPLE_CASES, ids=lambda case: case.name)
def test_cpu_upsample_parity(case, tmp_path):
    pytest.importorskip("torch")
    spec = trace_upsample_spec(case)
    inputs = random_inputs(case)
    short_name = {
        "upsample_1d_linear_scale_factor": "u1ls",
        "upsample_2d_nearest_size": "u2ns",
        "upsample_3d_nearest_exact_scale_factor": "u3ne",
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
