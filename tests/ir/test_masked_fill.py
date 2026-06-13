from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.masked_fill_parity import (
    ATOL_BY_DTYPE,
    MASKED_FILL_CASES,
    RTOL_BY_DTYPE,
    random_inputs,
    torch_oracle,
    trace_masked_fill_spec,
)


@pytest.mark.parametrize("case", MASKED_FILL_CASES, ids=lambda case: case.name)
def test_masked_fill_rewrites_to_where(case):
    spec = trace_masked_fill_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    counts = Counter(node["op"] for node in spec.ir["nodes"])
    assert counts["where"] == 1
    assert "masked_fill" not in counts
    constants = [tensor for tensor in spec.ir["tensors"] if tensor["kind"] == "constant"]
    assert len(constants) == 1
    assert constants[0]["shape"] == []

    if case.dtype == "bool":
        np.testing.assert_array_equal(actual.astype(np.bool_), expected.astype(np.bool_))
    else:
        np.testing.assert_allclose(
            actual.astype(np.float32),
            expected.astype(np.float32),
            atol=ATOL_BY_DTYPE[case.dtype],
            rtol=RTOL_BY_DTYPE[case.dtype],
        )


def test_masked_fill_dynamic_broadcast_shape_spec_tracks_input():
    rows = dml.Dim("rows", min=2, max=5, typical=4, buckets=(4, 5))
    cols = dml.Dim("cols", min=2, max=4, typical=3, buckets=(3, 4))

    class DynamicMaskedFill(dml.Module):
        def forward(self, x, mask):
            return dml.ops.output(x.masked_fill(mask, -1.0), "y")

    spec = dml.trace(
        DynamicMaskedFill(),
        inputs={
            "x": dml.TensorSpec([rows, cols], "float32"),
            "mask": dml.TensorSpec([1, cols], "bool"),
        },
        name="masked_fill_dynamic",
    )
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    output_tensor_name = spec.ir["outputs"][0]["tensor"]
    assert Counter(node["op"] for node in spec.ir["nodes"])["where"] == 1
    assert tensors[output_tensor_name]["shape_spec"] == tensors["x"]["shape_spec"]


def test_masked_fill_rejects_non_bool_mask():
    class BadMask(dml.Module):
        def forward(self, x, mask):
            return dml.ops.output(x.masked_fill(mask, 0.0), "y")

    with pytest.raises(ValueError, match="mask must have dtype bool"):
        dml.trace(
            BadMask(),
            inputs={
                "x": dml.TensorSpec([2, 3], "float32"),
                "mask": dml.TensorSpec([2, 3], "float32"),
            },
            name="masked_fill_bad_mask_dtype",
        )


def test_masked_fill_rejects_bool_input_dtype():
    class BoolInput(dml.Module):
        def forward(self, x, mask):
            return dml.ops.output(x.masked_fill(mask, True), "y")

    with pytest.raises(ValueError, match="masked_fill does not support dtype bool"):
        dml.trace(
            BoolInput(),
            inputs={
                "x": dml.TensorSpec([2, 3], "bool"),
                "mask": dml.TensorSpec([2, 3], "bool"),
            },
            name="masked_fill_bool_input_rejected",
        )
