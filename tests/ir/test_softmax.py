from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.softmax_parity import SOFTMAX_CASES, random_inputs, torch_oracle, trace_softmax_spec


@pytest.mark.parametrize("case", SOFTMAX_CASES, ids=lambda case: case.name)
def test_functional_softmax_traces_to_existing_softmax_op(case):
    spec = trace_softmax_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == ["softmax"]
    assert spec.ir["outputs"][0]["shape"] == list(case.input_shape)
    assert spec.ir["outputs"][0]["dtype"] == case.dtype
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_functional_softmax_rejects_dtype_argument():
    class BadSoftmaxModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.softmax(x, dim=-1, dtype="float16"), "y")

    with pytest.raises(NotImplementedError, match="does not support dtype="):
        dml.trace(
            BadSoftmaxModule(),
            inputs={"x": dml.TensorSpec([2, 3], "float32")},
            name="functional_softmax_bad_dtype",
        )


def test_functional_softmax_rejects_non_last_dim():
    class BadSoftmaxModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.softmax(x, dim=0), "y")

    with pytest.raises(NotImplementedError, match="supports only the last dimension"):
        dml.trace(
            BadSoftmaxModule(),
            inputs={"x": dml.TensorSpec([2, 3], "float32")},
            name="functional_softmax_bad_dim",
        )
