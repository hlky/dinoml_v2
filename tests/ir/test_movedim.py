from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.movedim_parity import MOVEDIM_CASES, random_inputs, torch_oracle, trace_movedim_spec


@pytest.mark.parametrize("case", MOVEDIM_CASES, ids=lambda case: case.name)
def test_movedim_rewrites_to_permute(case):
    spec = trace_movedim_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == ["permute"]
    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=1e-6, rtol=1e-6)


def test_movedim_dynamic_shape_spec_reorders_symbolic_dims():
    batch = dml.Dim("batch", min=1, max=4, typical=2, buckets=(2, 4))
    rows = dml.Dim("rows", min=2, max=5, typical=3, buckets=(3, 5))
    cols = dml.Dim("cols", min=2, max=6, typical=4, buckets=(4, 6))

    class DynamicMovedim(dml.Module):
        def forward(self, x):
            return dml.ops.output(x.movedim(0, -1), "y")

    spec = dml.trace(
        DynamicMovedim(),
        inputs={"x": dml.TensorSpec([batch, rows, cols], "float32")},
        name="movedim_dynamic",
    )
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    output_tensor = tensors[spec.ir["outputs"][0]["tensor"]]
    assert [node["op"] for node in spec.ir["nodes"]] == ["permute"]
    assert output_tensor["shape_spec"] == [tensors["x"]["shape_spec"][1], tensors["x"]["shape_spec"][2], tensors["x"]["shape_spec"][0]]


def test_movedim_rejects_length_mismatch():
    class BadMovedim(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.movedim(x, (0, 1), 2), "y")

    with pytest.raises(ValueError, match="must have the same number of dims"):
        dml.trace(
            BadMovedim(),
            inputs={"x": dml.TensorSpec([2, 3, 4], "float32")},
            name="movedim_length_mismatch",
        )


def test_movedim_rejects_duplicate_source_dim():
    class BadMovedim(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.movedim(x, (0, 0), (1, 2)), "y")

    with pytest.raises(ValueError, match="source dims must not contain duplicates"):
        dml.trace(
            BadMovedim(),
            inputs={"x": dml.TensorSpec([2, 3, 4], "float32")},
            name="movedim_duplicate_source",
        )
