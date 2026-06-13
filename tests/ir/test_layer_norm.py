from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.layer_norm_parity import (
    ATOL_BY_DTYPE,
    LAYER_NORM_CASES,
    RTOL_BY_DTYPE,
    random_inputs,
    torch_oracle,
    trace_layer_norm_spec,
)


@pytest.mark.parametrize("case", LAYER_NORM_CASES[:2], ids=lambda case: case.name)
def test_functional_layer_norm_traces_to_existing_layer_norm_op(case):
    spec = trace_layer_norm_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == ["layer_norm"]
    assert spec.ir["outputs"][0]["shape"] == list(case.input_shape)
    assert spec.ir["outputs"][0]["dtype"] == case.dtype
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected,
        atol=ATOL_BY_DTYPE[case.dtype],
        rtol=RTOL_BY_DTYPE[case.dtype],
    )


def test_functional_layer_norm_dynamic_shape_spec_tracks_input():
    case = next(case for case in LAYER_NORM_CASES if case.name == "layer_norm_dynamic_batch_f32")
    spec = trace_layer_norm_spec(case)
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    output_tensor = tensors[spec.ir["outputs"][0]["tensor"]]

    assert output_tensor["shape_spec"] == tensors["x"]["shape_spec"]


def test_functional_layer_norm_requires_affine_tensors():
    class BadLayerNormModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.layer_norm(x, 4), "y")

    with pytest.raises(NotImplementedError, match="requires weight and bias tensors"):
        dml.trace(
            BadLayerNormModule(),
            inputs={"x": dml.TensorSpec([2, 4], "float32")},
            name="functional_layer_norm_missing_affine",
        )


def test_functional_layer_norm_rejects_mismatched_normalized_shape():
    class BadLayerNormModule(dml.Module):
        def forward(self, x, weight, bias):
            return dml.ops.output(dml.nn.functional.layer_norm(x, (2, 3), weight, bias), "y")

    with pytest.raises(ValueError, match="normalized_shape must match the input trailing dimensions"):
        dml.trace(
            BadLayerNormModule(),
            inputs={
                "x": dml.TensorSpec([2, 4], "float32"),
                "weight": dml.TensorSpec([2, 3], "float32"),
                "bias": dml.TensorSpec([2, 3], "float32"),
            },
            name="functional_layer_norm_bad_shape",
        )
