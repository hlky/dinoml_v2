from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.unflatten_parity import UNFLATTEN_CASES, case_inputs, torch_oracle, trace_unflatten_spec


@pytest.mark.parametrize("case", UNFLATTEN_CASES, ids=lambda case: case.name)
def test_unflatten_reference_matches_torch(case):
    spec = trace_unflatten_spec(case)
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == spec.ir["outputs"][0]["tensor"])
    assert spec.ir["nodes"] == []
    assert output_tensor["shape_spec"] == spec.ir["outputs"][0]["shape_spec"]
    assert spec.ir["metadata"]["views"]["views"][0]["transform"] == "reshape"
    for inputs in case_inputs(case):
        actual = reference_numpy(spec, inputs)["y"]
        expected = torch_oracle(case, inputs)
        np.testing.assert_array_equal(actual, expected)


def test_unflatten_rejects_target_product_mismatch():
    class BadProductModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.unflatten(x, 1, (2, 4)), "y")

    with pytest.raises(ValueError, match="must preserve dimension size"):
        dml.trace(
            BadProductModule(),
            inputs={"x": dml.TensorSpec([3, 6], "float32")},
            name="unflatten_bad_product",
        )


def test_unflatten_rejects_invalid_dim():
    class BadDimModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.unflatten(x, 2, (2, 3)), "y")

    with pytest.raises(IndexError, match="axis 2 is out of range for rank 2"):
        dml.trace(
            BadDimModule(),
            inputs={"x": dml.TensorSpec([3, 6], "float32")},
            name="unflatten_bad_dim",
        )
