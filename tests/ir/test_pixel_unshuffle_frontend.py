from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.pixel_unshuffle_frontend_parity import (
    ATOL,
    PIXEL_UNSHUFFLE_FRONTEND_CASES,
    RTOL,
    random_inputs,
    torch_oracle,
    trace_pixel_unshuffle_frontend_spec,
)


@pytest.mark.parametrize("case", PIXEL_UNSHUFFLE_FRONTEND_CASES, ids=lambda case: case.name)
def test_pixel_unshuffle_frontend_spellings_route_to_existing_op(case):
    spec = trace_pixel_unshuffle_frontend_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)
    output_tensor_name = spec.ir["outputs"][0]["tensor"]
    output_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == output_tensor_name)

    assert [node["op"] for node in spec.ir["nodes"]] == ["permute"]
    assert output_tensor["shape"] == list(expected.shape)
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=ATOL, rtol=RTOL)


def test_pixel_unshuffle_module_rejects_non_positive_factor():
    with pytest.raises(ValueError, match="positive"):
        dml.nn.PixelUnshuffle(0)


def test_pixel_unshuffle_functional_rejects_non_divisible_spatial_dims():
    class _BadModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.pixel_unshuffle(x, 2), "y")

    with pytest.raises(ValueError, match="must be divisible"):
        dml.trace(
            _BadModule(),
            inputs={"x": dml.TensorSpec([1, 3, 5, 4], "float32")},
            name="pixel_unshuffle_bad_spatial",
        )
