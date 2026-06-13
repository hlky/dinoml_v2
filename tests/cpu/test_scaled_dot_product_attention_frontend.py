from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.scaled_dot_product_attention_frontend_parity import (
    ATOL,
    RTOL,
    SCALED_DOT_PRODUCT_ATTENTION_FRONTEND_CASES,
    random_inputs,
    torch_oracle,
    trace_scaled_dot_product_attention_frontend_spec,
)


@pytest.mark.parametrize("case", SCALED_DOT_PRODUCT_ATTENTION_FRONTEND_CASES, ids=lambda case: case.name)
def test_scaled_dot_product_attention_frontend_reference_parity(case):
    pytest.importorskip("torch")
    spec = trace_scaled_dot_product_attention_frontend_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["out"].astype(np.float32)
    expected = torch_oracle(case, inputs)
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("case", SCALED_DOT_PRODUCT_ATTENTION_FRONTEND_CASES, ids=lambda case: case.name)
def test_cpu_scaled_dot_product_attention_frontend_rejects_cpu_backend(case, tmp_path):
    spec = trace_scaled_dot_product_attention_frontend_spec(case)
    with pytest.raises(NotImplementedError, match="cpu backend does not support op flash_attention"):
        dml.compile(spec, dml.Target("cpu"), tmp_path / f"{case.name}.dinoml")
