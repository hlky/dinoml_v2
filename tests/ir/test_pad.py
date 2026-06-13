from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.pad_parity import PAD_CASES, random_inputs, torch_oracle, trace_pad_spec


@pytest.mark.parametrize("case", PAD_CASES, ids=lambda case: case.name)
def test_functional_pad_traces_to_existing_pad_op(case):
    spec = trace_pad_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == ["pad"]
    assert spec.ir["outputs"][0]["shape"] == list(case.output_shape)
    assert spec.ir["outputs"][0]["dtype"] == case.dtype
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_functional_pad_default_value_lowers_to_zero():
    case = next(case for case in PAD_CASES if case.value is None)
    spec = trace_pad_spec(case)

    assert spec.ir["nodes"][0]["attrs"]["value"] == 0.0


def test_functional_pad_rejects_non_constant_mode():
    class BadPadModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.pad(x, (1, 1), mode="reflect"), "y")

    with pytest.raises(NotImplementedError, match="supports only mode='constant'"):
        dml.trace(
            BadPadModule(),
            inputs={"x": dml.TensorSpec([2, 3], "float32")},
            name="functional_pad_bad_mode",
        )
