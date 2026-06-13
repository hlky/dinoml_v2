from __future__ import annotations

import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.transposed_conv1d_frontend_parity import (
    TRANSPOSED_CONV1D_FRONTEND_CASES,
    random_inputs,
    torch_oracle,
    trace_transposed_conv1d_frontend_spec,
)


@pytest.mark.parametrize("case", TRANSPOSED_CONV1D_FRONTEND_CASES, ids=lambda case: case.name)
def test_transposed_conv1d_frontend_spellings_trace_to_existing_op(case):
    spec = trace_transposed_conv1d_frontend_spec(case)

    assert [node["op"] for node in spec.ir["nodes"]] == ["transposed_conv1d"]

    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)
    assert actual.shape == expected.shape


def test_conv_transpose1d_module_rejects_bias_true():
    with pytest.raises(NotImplementedError, match="bias=False"):
        dml.nn.ConvTranspose1d(3, 5, 3, bias=True)


def test_conv_transpose1d_functional_rejects_bias_tensor():
    class _BiasModule(dml.Module):
        def forward(self, x, weight, bias):
            return dml.ops.output(dml.nn.functional.conv_transpose1d(x, weight, bias=bias), "y")

    with pytest.raises(NotImplementedError, match="bias=None"):
        dml.trace(
            _BiasModule(),
            inputs={
                "x": dml.TensorSpec([1, 3, 4], "float32"),
                "weight": dml.TensorSpec([3, 5, 3], "float32"),
                "bias": dml.TensorSpec([5], "float32"),
            },
            name="conv_transpose1d_functional_bias_unsupported",
        )
