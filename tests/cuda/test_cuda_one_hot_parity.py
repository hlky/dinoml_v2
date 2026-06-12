from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.one_hot_parity import ONE_HOT_CASES, random_inputs, torch_oracle, trace_one_hot_spec


def _discover_nvcc() -> str | None:
    direct = shutil.which("nvcc")
    if direct:
        return direct
    for candidate in (
        os.environ.get("CUDACXX"),
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.9/bin/nvcc",
        "/usr/local/cuda-12.8/bin/nvcc",
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


pytestmark = pytest.mark.skipif(_discover_nvcc() is None, reason="nvcc is required")


@pytest.mark.parametrize("case", ONE_HOT_CASES, ids=lambda case: case.name)
def test_cuda_one_hot_parity(case, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is not available")

    spec = trace_one_hot_spec(case)
    inputs = random_inputs(case)
    artifact = dml.compile(spec, dml.Target("cuda"), tmp_path / f"{case.name}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(case, inputs)
    np.testing.assert_array_equal(actual, expected)
