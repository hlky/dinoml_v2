from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.silu_parity import ATOL_BY_DTYPE, RTOL_BY_DTYPE, SILU_CASES, random_inputs, torch_oracle, trace_silu_spec


@pytest.mark.parametrize("case", SILU_CASES[:2], ids=lambda case: case.name)
def test_functional_silu_traces_to_existing_silu_op(case):
    spec = trace_silu_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == ["silu"]
    assert spec.ir["outputs"][0]["shape"] == list(case.input_shape)
    assert spec.ir["outputs"][0]["dtype"] == case.dtype
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected,
        atol=ATOL_BY_DTYPE[case.dtype],
        rtol=RTOL_BY_DTYPE[case.dtype],
    )


def test_functional_silu_dynamic_shape_spec_tracks_input():
    case = next(case for case in SILU_CASES if case.name == "silu_dynamic_rank2_f32")
    spec = trace_silu_spec(case)
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    output_tensor = tensors[spec.ir["outputs"][0]["tensor"]]

    assert output_tensor["shape_spec"] == tensors["x"]["shape_spec"]


def test_functional_silu_rejects_inplace():
    class BadSiluModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.silu(x, inplace=True), "y")

    with pytest.raises(NotImplementedError, match="does not support inplace=True"):
        dml.trace(
            BadSiluModule(),
            inputs={"x": dml.TensorSpec([2, 3], "float32")},
            name="functional_silu_bad_inplace",
        )
