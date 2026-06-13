from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.upsample_parity import ATOL, RTOL, UPSAMPLE_CASES, random_inputs, torch_oracle, trace_upsample_spec


@pytest.mark.parametrize("case", UPSAMPLE_CASES, ids=lambda case: case.name)
def test_upsample_rewrites_to_upsampling_family_and_matches_torch(case):
    spec = trace_upsample_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == ["permute", case.expected_op, "permute"]
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=ATOL, rtol=RTOL)


def test_upsample_rejects_missing_size_and_scale_factor():
    with pytest.raises(ValueError, match="either size= or scale_factor="):
        dml.nn.Upsample()


def test_upsample_rejects_non_uniform_size():
    class _NonUniformSizeModule(dml.Module):
        def __init__(self):
            self.upsample = dml.nn.Upsample(size=(8, 9), mode="nearest")

        def forward(self, x):
            return dml.ops.output(self.upsample(x), "y")

    with pytest.raises(NotImplementedError, match="uniform scale_factor"):
        dml.trace(
            _NonUniformSizeModule(),
            inputs={"x": dml.TensorSpec([2, 4, 4, 5], "float32")},
            name="upsample_non_uniform_size_unsupported",
        )


def test_upsample_rejects_recompute_scale_factor():
    with pytest.raises(NotImplementedError, match="recompute_scale_factor=True"):
        dml.nn.Upsample(scale_factor=2.0, recompute_scale_factor=True)
