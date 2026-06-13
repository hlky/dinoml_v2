from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.functional_group_norm_parity import (
    ATOL_BY_DTYPE,
    FUNCTIONAL_GROUP_NORM_CASES,
    RTOL_BY_DTYPE,
    random_inputs,
    torch_oracle,
    trace_functional_group_norm_spec,
)


@pytest.mark.parametrize("case", FUNCTIONAL_GROUP_NORM_CASES[:2], ids=lambda case: case.name)
def test_functional_group_norm_traces_to_existing_group_norm_op(case):
    spec = trace_functional_group_norm_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == ["group_norm"]
    assert spec.ir["outputs"][0]["shape"] == list(case.input_shape)
    assert spec.ir["outputs"][0]["dtype"] == case.dtype
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected,
        atol=ATOL_BY_DTYPE[case.dtype],
        rtol=RTOL_BY_DTYPE[case.dtype],
    )


def test_functional_group_norm_dynamic_shape_spec_tracks_input():
    case = next(case for case in FUNCTIONAL_GROUP_NORM_CASES if case.name == "functional_group_norm_dynamic_batch_f32")
    spec = trace_functional_group_norm_spec(case)
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    output_tensor = tensors[spec.ir["outputs"][0]["tensor"]]

    assert output_tensor["shape_spec"] == tensors["x"]["shape_spec"]


def test_functional_group_norm_rejects_invalid_group_count():
    class BadGroupNormModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.group_norm(x, 3), "y")

    with pytest.raises(ValueError, match="must be divisible by num_groups"):
        dml.trace(
            BadGroupNormModule(),
            inputs={"x": dml.TensorSpec([1, 2, 2, 8], "float32")},
            name="functional_group_norm_bad_groups",
        )
