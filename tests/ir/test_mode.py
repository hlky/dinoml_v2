from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.mode_parity import (
    ATOL_BY_DTYPE,
    MODE_CASES,
    RTOL_BY_DTYPE,
    case_inputs,
    torch_oracle,
    trace_mode_spec,
)


@pytest.mark.parametrize("case", MODE_CASES, ids=lambda case: case.name)
def test_mode_traces_and_matches_reference(case):
    spec = trace_mode_spec(case)
    inputs = case_inputs(case)
    actual = reference_numpy(spec, inputs)
    expected_values, expected_indices = torch_oracle(case, inputs)

    counts = Counter(node["op"] for node in spec.ir["nodes"])
    assert counts["mode_values"] == 1
    assert counts["mode_indices"] == 1
    values_node, indices_node = spec.ir["nodes"]
    assert values_node["attrs"]["paired_indices_output"] == indices_node["outputs"][0]
    assert indices_node["attrs"]["paired_values_output"] == values_node["outputs"][0]

    if case.dtype == "bool":
        np.testing.assert_array_equal(actual["values"].astype(np.bool_), expected_values.astype(np.bool_))
    else:
        np.testing.assert_allclose(
            actual["values"].astype(np.float32),
            expected_values.astype(np.float32),
            atol=ATOL_BY_DTYPE[case.dtype],
            rtol=RTOL_BY_DTYPE[case.dtype],
        )
    np.testing.assert_array_equal(actual["indices"].astype(np.int64), expected_indices.astype(np.int64))


def test_mode_rejects_dynamic_input_shape():
    rows = dml.Dim("rows", min=2, max=4, typical=3, buckets=(3, 4))

    class DynamicMode(dml.Module):
        def forward(self, x):
            values, indices = x.mode(dim=-1)
            return {
                "values": dml.ops.output(values, "values"),
                "indices": dml.ops.output(indices, "indices"),
            }

    with pytest.raises(ValueError, match="mode currently supports only static input shapes"):
        dml.trace(
            DynamicMode(),
            inputs={"x": dml.TensorSpec([rows, 5], "float32")},
            name="mode_dynamic_rejected",
        )


def test_mode_rejects_non_last_dim():
    class WrongDim(dml.Module):
        def forward(self, x):
            values, indices = x.mode(dim=0)
            return {
                "values": dml.ops.output(values, "values"),
                "indices": dml.ops.output(indices, "indices"),
            }

    with pytest.raises(NotImplementedError, match="mode currently supports only the last dimension"):
        dml.trace(
            WrongDim(),
            inputs={"x": dml.TensorSpec([2, 3], "float32")},
            name="mode_non_last_dim_rejected",
        )
