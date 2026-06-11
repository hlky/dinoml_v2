from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.tensor_filter_helpers_parity import TENSOR_FILTER_HELPER_CASES, numpy_oracle, random_inputs, trace_tensor_filter_helper_spec


@pytest.mark.parametrize("case", TENSOR_FILTER_HELPER_CASES, ids=lambda case: case.name)
def test_tensor_filter_helpers_reference_matches_oracle(case):
    spec = trace_tensor_filter_helper_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = numpy_oracle(case, inputs)
    assert Counter(node["op"] for node in spec.ir["nodes"])[case.op_name] == 1
    np.testing.assert_allclose(actual.astype(np.float32), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("name", "builder", "inputs", "message"),
    [
        (
            "fir_downsample2d_bad_rank",
            lambda x: dml.ops.fir_downsample2d(x),
            {"x": dml.TensorSpec([2, 5, 7], "float32")},
            "rank-4 NHWC input",
        ),
        (
            "fir_upsample2d_bad_up",
            lambda x: dml.ops.fir_upsample2d(x, up=0),
            {"x": dml.TensorSpec([2, 5, 7, 3], "float32")},
            "positive integer",
        ),
        (
            "kdownsample2d_weight_bad_channels",
            lambda: dml.ops.kdownsample2d_weight(0),
            {},
            "positive integer",
        ),
    ],
)
def test_tensor_filter_helpers_validate_inputs(name, builder, inputs, message):
    if inputs:
        class InputModule(dml.Module):
            def forward(self, x):
                return dml.ops.output(builder(x), "y")

        with pytest.raises(ValueError, match=message):
            dml.trace(InputModule(), inputs=inputs, name=name)
        return

    class NoInputModule(dml.Module):
        def forward(self):
            return dml.ops.output(builder(), "y")

    with pytest.raises(ValueError, match=message):
        dml.trace(NoInputModule(), inputs=inputs, name=name)
