from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.padding_layout_helpers_parity import (
    PADDING_LAYOUT_HELPER_CASES,
    numpy_oracle,
    random_inputs,
    trace_padding_layout_helper_spec,
)


@pytest.mark.parametrize("case", PADDING_LAYOUT_HELPER_CASES, ids=lambda case: case.name)
def test_padding_layout_helpers_reference_matches_oracle(case):
    spec = trace_padding_layout_helper_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = numpy_oracle(case, inputs)
    assert Counter(node["op"] for node in spec.ir["nodes"])[case.op_name] == 1
    np.testing.assert_allclose(actual.astype(np.float32), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("name", "builder", "input_spec", "message"),
    [
        (
            "nhwc3to4_bad_rank",
            lambda x: dml.ops.nhwc3to4(x),
            dml.TensorSpec([2, 5, 3], "float32"),
            "rank-4",
        ),
        (
            "nhwc3to8_bad_channels",
            lambda x: dml.ops.nhwc3to8(x),
            dml.TensorSpec([2, 5, 7, 4], "float32"),
            "input.shape\\[-1\\] == 3",
        ),
        (
            "ndhwc3to8_bad_dtype",
            lambda x: dml.ops.ndhwc3to8(x),
            dml.TensorSpec([1, 2, 3, 4, 3], "bfloat16"),
            "does not support dtype",
        ),
    ],
)
def test_padding_layout_helpers_validate_inputs(name, builder, input_spec, message):
    class InputModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(builder(x), "y")

    with pytest.raises(ValueError, match=message):
        dml.trace(InputModule(), inputs={"x": input_spec}, name=name)
