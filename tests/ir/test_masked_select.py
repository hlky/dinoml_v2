from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

import dinoml as dml
from dinoml.passes.validation import ValidationError, validate_ir
from dinoml.reference import reference_numpy
from dinoml.shapes import symbolic_int_interval
from tests.masked_select_parity import MASKED_SELECT_CASES, numpy_oracle, random_inputs, trace_masked_select_spec


@pytest.mark.parametrize("case", MASKED_SELECT_CASES, ids=lambda case: case.name)
def test_masked_select_reference_matches_oracle(case):
    spec = trace_masked_select_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = numpy_oracle(case, inputs)

    assert Counter(node["op"] for node in spec.ir["nodes"])["masked_select"] == 1
    assert tuple(actual.shape) == tuple(expected.shape)
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=1e-6, rtol=1e-6)


def test_masked_select_trace_records_shape_report_and_dynamic_capacity():
    case = next(case for case in MASKED_SELECT_CASES if case.name == "masked_select_dynamic_bf16")
    spec = trace_masked_select_spec(case)
    output = spec.ir["outputs"][0]
    reports = spec.ir["metadata"]["output_shape_reports"]["reports"]

    assert output["shape"] == [128]
    assert len(output["shape_spec"]) == 1
    assert output["shape_spec"][0]["kind"] == "int_expr"
    assert symbolic_int_interval(output["shape_spec"][0]) == (4, 128)
    assert reports == [{"output": "y", "kind": "shape_buffer"}]


def test_masked_select_rejects_non_bool_mask():
    class BadMaskModule(dml.Module):
        def forward(self, x, mask):
            return dml.ops.output(dml.ops.masked_select(x, mask), "y")

    with pytest.raises(ValueError, match="mask must have dtype bool"):
        dml.trace(
            BadMaskModule(),
            inputs={
                "x": dml.TensorSpec([2, 3], "float32"),
                "mask": dml.TensorSpec([2, 3], "float32"),
            },
            name="masked_select_bad_mask",
        )


def test_masked_select_requires_direct_public_output_shape_contract():
    class DownstreamModule(dml.Module):
        def forward(self, x, mask):
            selected = dml.ops.masked_select(x, mask)
            return dml.ops.output(dml.ops.identity(selected), "y")

    spec = dml.trace(
        DownstreamModule(),
        inputs={
            "x": dml.TensorSpec([2, 3], "float32"),
            "mask": dml.TensorSpec([2, 3], "bool"),
        },
        name="masked_select_downstream_view",
    )

    with pytest.raises(ValidationError, match="requires a public output"):
        validate_ir(spec.ir)
