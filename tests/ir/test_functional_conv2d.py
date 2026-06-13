from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.functional_conv2d_parity import (
    ATOL,
    FUNCTIONAL_CONV2D_CASES,
    RTOL,
    random_inputs,
    torch_oracle,
    trace_functional_conv2d_spec,
)


@pytest.mark.parametrize("case", FUNCTIONAL_CONV2D_CASES, ids=lambda case: case.name)
def test_functional_conv2d_routes_to_existing_conv2d_family(case):
    spec = trace_functional_conv2d_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == ["conv2d_bias"]
    node = spec.ir["nodes"][0]
    if case.use_bias:
        assert node["attrs"].get("source_op") is None
        assert node["attrs"].get("bias_mode") is None
    else:
        assert node["attrs"]["source_op"] == "conv2d"
        assert node["attrs"]["bias_mode"] == "explicit_zero_constant"
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=ATOL, rtol=RTOL)


def test_functional_conv2d_rejects_unsupported_groups():
    class _GroupsModule(dml.Module):
        def forward(self, x, weight):
            return dml.ops.output(dml.nn.functional.conv2d(x, weight, groups=2), "y")

    with pytest.raises(NotImplementedError, match="groups=1 only"):
        dml.trace(
            _GroupsModule(),
            inputs={
                "x": dml.TensorSpec([2, 4, 7, 7], "float32"),
                "weight": dml.TensorSpec([6, 4, 3, 3], "float32"),
            },
            name="functional_conv2d_groups_unsupported",
        )
