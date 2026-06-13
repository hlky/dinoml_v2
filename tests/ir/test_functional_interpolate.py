from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.functional_interpolate_parity import (
    ATOL,
    FUNCTIONAL_INTERPOLATE_CASES,
    RTOL,
    random_inputs,
    torch_oracle,
    trace_functional_interpolate_spec,
)


@pytest.mark.parametrize("case", FUNCTIONAL_INTERPOLATE_CASES, ids=lambda case: case.name)
def test_functional_interpolate_rewrites_to_upsampling_family_and_matches_torch(case):
    spec = trace_functional_interpolate_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == ["permute", case.expected_op, "permute"]
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=ATOL, rtol=RTOL)


def test_functional_interpolate_rejects_size_path():
    class _SizeModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.interpolate(x, size=8, mode="nearest"), "y")

    with pytest.raises(NotImplementedError, match="size="):
        dml.trace(
            _SizeModule(),
            inputs={"x": dml.TensorSpec([2, 3, 5], "float32")},
            name="functional_interpolate_size_unsupported",
        )


def test_functional_interpolate_rejects_non_uniform_scale_factor():
    class _NonUniformScaleModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.interpolate(x, scale_factor=(2.0, 3.0), mode="nearest"), "y")

    with pytest.raises(NotImplementedError, match="uniform scale_factor"):
        dml.trace(
            _NonUniformScaleModule(),
            inputs={"x": dml.TensorSpec([2, 3, 4, 5], "float32")},
            name="functional_interpolate_non_uniform_scale",
        )


def test_functional_interpolate_rejects_antialias():
    class _AntialiasModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(
                dml.nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear", antialias=True),
                "y",
            )

    with pytest.raises(NotImplementedError, match="antialias=True"):
        dml.trace(
            _AntialiasModule(),
            inputs={"x": dml.TensorSpec([2, 3, 4, 5], "float32")},
            name="functional_interpolate_antialias_unsupported",
        )
