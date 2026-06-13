from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.normalize_parity import (
    ATOL_BY_DTYPE,
    NORMALIZE_CASES,
    RTOL_BY_DTYPE,
    random_inputs,
    torch_oracle,
    trace_normalize_spec,
)


@pytest.mark.parametrize("case", NORMALIZE_CASES, ids=lambda case: case.name)
def test_normalize_reuses_existing_ops(case):
    spec = trace_normalize_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == ["vector_norm", "max", "div"]
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected.astype(np.float32),
        atol=ATOL_BY_DTYPE[case.dtype],
        rtol=RTOL_BY_DTYPE[case.dtype],
    )


def test_normalize_dynamic_shape_spec_tracks_input():
    batch = dml.Dim("batch", min=1, max=4, typical=2, buckets=(2, 4))

    class DynamicNormalize(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.normalize(x, dim=-1), "y")

    spec = dml.trace(
        DynamicNormalize(),
        inputs={"x": dml.TensorSpec([batch, 4], "float32")},
        name="normalize_dynamic",
    )
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    output_tensor = tensors[spec.ir["outputs"][0]["tensor"]]
    assert output_tensor["shape_spec"] == tensors["x"]["shape_spec"]


def test_normalize_rejects_non_l2_p():
    class BadNormalize(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.normalize(x, p=1.0, dim=-1), "y")

    with pytest.raises(NotImplementedError, match="supports only p=2"):
        dml.trace(
            BadNormalize(),
            inputs={"x": dml.TensorSpec([2, 3], "float32")},
            name="normalize_bad_p",
        )


def test_normalize_rejects_non_last_dim():
    class BadNormalize(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.normalize(x, dim=0), "y")

    with pytest.raises(NotImplementedError, match="supports only the last dimension"):
        dml.trace(
            BadNormalize(),
            inputs={"x": dml.TensorSpec([2, 3], "float32")},
            name="normalize_bad_dim",
        )


def test_normalize_rejects_out_argument():
    class BadNormalize(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.normalize(x, out=x), "y")

    with pytest.raises(NotImplementedError, match="does not support out="):
        dml.trace(
            BadNormalize(),
            inputs={"x": dml.TensorSpec([2, 3], "float32")},
            name="normalize_bad_out",
        )
