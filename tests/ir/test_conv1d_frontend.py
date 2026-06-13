from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.conv1d_frontend_parity import CONV1D_FRONTEND_CASES, ATOL, RTOL, random_inputs, torch_oracle, trace_conv1d_frontend_spec


@pytest.mark.parametrize("case", CONV1D_FRONTEND_CASES, ids=lambda case: case.name)
def test_conv1d_frontend_spellings_trace_to_existing_conv1d_bias(case):
    spec = trace_conv1d_frontend_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == ["conv1d_bias"]
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)


def test_conv1d_module_rejects_bias_false():
    with pytest.raises(NotImplementedError, match="requires bias=True"):
        dml.nn.Conv1d(3, 5, kernel_size=3, bias=False)


def test_functional_conv1d_rejects_missing_bias():
    class BadFunctionalConv1d(dml.Module):
        def forward(self, x, weight):
            return dml.ops.output(dml.nn.functional.conv1d(x, weight, bias=None), "y")

    with pytest.raises(NotImplementedError, match="requires bias tensor input"):
        dml.trace(
            BadFunctionalConv1d(),
            inputs={
                "x": dml.TensorSpec([2, 3, 9], "float32"),
                "weight": dml.TensorSpec([5, 3, 3], "float32"),
            },
            name="functional_conv1d_missing_bias",
        )
