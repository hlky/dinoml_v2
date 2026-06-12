from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.dual_gemm_parity import (
    ATOL_BY_DTYPE,
    DUAL_GEMM_CASES,
    RTOL_BY_DTYPE,
    numpy_oracle,
    random_inputs,
    trace_dual_gemm_spec,
)


@pytest.mark.parametrize("case", DUAL_GEMM_CASES, ids=lambda case: f"{case.op_name}_{case.name}")
def test_dual_gemm_reference_matches_numpy_oracle(case):
    spec = trace_dual_gemm_spec(case)
    inputs = random_inputs(case)
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}

    actual = reference_numpy(spec, inputs)["y"]
    expected = numpy_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == [case.op_name]
    assert list(actual.shape) == [*case.a_shape[:-1], case.b0_shape[0]]
    assert list(expected.shape) == [*case.a_shape[:-1], case.b0_shape[0]]
    assert spec.ir["outputs"][0]["shape_spec"][:-1] == tensors["a"]["shape_spec"][:-1]
    assert spec.ir["outputs"][0]["shape_spec"][-1] == tensors["b0"]["shape_spec"][0]
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected.astype(np.float32),
        atol=ATOL_BY_DTYPE[case.dtype],
        rtol=RTOL_BY_DTYPE[case.dtype],
    )


def test_dual_gemm_dynamic_shape_spec_tracks_a_prefix_and_b0_n():
    case = next(case for case in DUAL_GEMM_CASES if case.name == "dual_gemm_bias_fast_gelu_bf16_dynamic")
    spec = trace_dual_gemm_spec(case)
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    output_tensor = tensors[spec.ir["outputs"][0]["tensor"]]

    assert output_tensor["shape_spec"][:-1] == tensors["a"]["shape_spec"][:-1]
    assert output_tensor["shape_spec"][-1] == tensors["b0"]["shape_spec"][0]
    assert tensors["bias0"]["shape_spec"][0] == tensors["b0"]["shape_spec"][0]
    assert tensors["bias1"]["shape_spec"][0] == tensors["b1"]["shape_spec"][0]


def test_dual_gemm_rejects_b1_shape_mismatch():
    class BadDualGemm(dml.Module):
        def forward(self, a, b0, b1):
            return dml.ops.output(dml.ops.dual_gemm_rcr_silu(a, b0, b1), "y")

    with pytest.raises(ValueError, match="expected B1\\[N,K\\] or B1\\[1,K\\]"):
        dml.trace(
            BadDualGemm(),
            inputs={
                "a": dml.TensorSpec([2, 3, 8], "float32"),
                "b0": dml.TensorSpec([4, 8], "float32"),
                "b1": dml.TensorSpec([2, 8], "float32"),
            },
            name="dual_gemm_bad_b1",
        )


def test_dual_gemm_bias_rejects_bias1_shape_mismatch():
    class BadBiasDualGemm(dml.Module):
        def forward(self, a, b0, b1, bias0, bias1):
            return dml.ops.output(dml.ops.dual_gemm_rcr_bias_fast_gelu(a, b0, b1, bias0, bias1), "y")

    with pytest.raises(ValueError, match="expected bias1 shape \\[N\\] or \\[1, N\\]"):
        dml.trace(
            BadBiasDualGemm(),
            inputs={
                "a": dml.TensorSpec([2, 3, 8], "float32"),
                "b0": dml.TensorSpec([4, 8], "float32"),
                "b1": dml.TensorSpec([1, 8], "float32"),
                "bias0": dml.TensorSpec([4], "float32"),
                "bias1": dml.TensorSpec([4], "float32"),
            },
            name="dual_gemm_bad_bias1",
        )
