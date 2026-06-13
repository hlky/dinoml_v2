from __future__ import annotations

import os

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.functional_interpolate_parity import (
    ATOL,
    FUNCTIONAL_INTERPOLATE_CASES,
    RTOL,
    random_inputs,
    torch_oracle,
    trace_functional_interpolate_spec,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)


@pytest.mark.parametrize("case", FUNCTIONAL_INTERPOLATE_CASES, ids=lambda case: case.name)
def test_rocm_functional_interpolate_parity(case, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")

    spec = trace_functional_interpolate_spec(case)
    inputs = random_inputs(case)
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / "artifact.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(case, inputs)
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=ATOL, rtol=RTOL)
