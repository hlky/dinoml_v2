from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.runtime import load
from tests.scaled_dot_product_attention_frontend_parity import (
    ATOL,
    RTOL,
    SCALED_DOT_PRODUCT_ATTENTION_FRONTEND_CASES,
    random_inputs,
    torch_oracle,
    trace_scaled_dot_product_attention_frontend_spec,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))


@pytest.mark.parametrize("case", SCALED_DOT_PRODUCT_ATTENTION_FRONTEND_CASES, ids=lambda case: case.name)
def test_rocm_scaled_dot_product_attention_frontend_parity(case, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")
    spec = trace_scaled_dot_product_attention_frontend_spec(case)
    inputs = random_inputs(case)
    short_name = {
        "scaled_dot_product_attention_nocausal_f16": "sdpan",
        "scaled_dot_product_attention_causal_f16": "sdpac",
    }[case.name]
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{short_name}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["out"].astype(np.float32)
    finally:
        session.close()
        module.close()

    expected = torch_oracle(case, inputs)
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
