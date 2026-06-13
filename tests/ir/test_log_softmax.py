from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.log_softmax_parity import (
    ATOL_BY_DTYPE,
    LOG_SOFTMAX_CASES,
    RTOL_BY_DTYPE,
    random_inputs,
    torch_oracle,
    trace_log_softmax_spec,
)


@pytest.mark.parametrize("case", LOG_SOFTMAX_CASES[:2], ids=lambda case: case.name)
def test_functional_log_softmax_traces_to_existing_ops(case):
    spec = trace_log_softmax_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == ["softmax", "log"]
    assert spec.ir["outputs"][0]["shape"] == list(case.input_shape)
    assert spec.ir["outputs"][0]["dtype"] == case.dtype
    np.testing.assert_allclose(actual, expected, atol=ATOL_BY_DTYPE[case.dtype], rtol=RTOL_BY_DTYPE[case.dtype])


def test_functional_log_softmax_dynamic_shape_spec_tracks_input():
    case = next(case for case in LOG_SOFTMAX_CASES if case.name == "log_softmax_dynamic_rank2_f32")
    spec = trace_log_softmax_spec(case)
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    output_tensor = tensors[spec.ir["outputs"][0]["tensor"]]

    assert output_tensor["shape_spec"] == tensors["x"]["shape_spec"]


def test_functional_log_softmax_rejects_dtype_argument():
    class BadLogSoftmaxModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.log_softmax(x, dim=-1, dtype="float16"), "y")

    with pytest.raises(NotImplementedError, match="does not support dtype="):
        dml.trace(
            BadLogSoftmaxModule(),
            inputs={"x": dml.TensorSpec([2, 3], "float32")},
            name="functional_log_softmax_bad_dtype",
        )


def test_functional_log_softmax_rejects_non_last_dim():
    class BadLogSoftmaxModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.log_softmax(x, dim=0), "y")

    with pytest.raises(NotImplementedError, match="supports only the last dimension"):
        dml.trace(
            BadLogSoftmaxModule(),
            inputs={"x": dml.TensorSpec([2, 3], "float32")},
            name="functional_log_softmax_bad_dim",
        )
