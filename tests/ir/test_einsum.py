from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.einsum_parity import ATOL_BY_DTYPE, EINSUM_CASES, RTOL_BY_DTYPE, random_inputs, torch_oracle, trace_einsum_spec


@pytest.mark.parametrize("case", EINSUM_CASES, ids=lambda case: case.name)
def test_einsum_reference_matches_torch_oracle(case):
    spec = trace_einsum_spec(case)
    inputs = random_inputs(case)

    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == list(case.expected_ops)
    assert "einsum" not in {node["op"] for node in spec.ir["nodes"]}
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected.astype(np.float32),
        atol=ATOL_BY_DTYPE[case.dtype],
        rtol=RTOL_BY_DTYPE[case.dtype],
    )


def test_einsum_rejects_dropped_lhs_only_reduction_labels():
    class DroppedLhsReduction(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.einsum("bij,jk->bk", a, b), "y")

    with pytest.raises(NotImplementedError, match="appear only in the lhs operand"):
        dml.trace(
            DroppedLhsReduction(),
            inputs={
                "a": dml.TensorSpec([2, 3, 4], "float32"),
                "b": dml.TensorSpec([4, 5], "float32"),
            },
            name="einsum_drop_lhs_only",
        )


def test_einsum_rejects_outer_product_without_contraction():
    class OuterProduct(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.einsum("bi,bj->bij", a, b), "y")

    with pytest.raises(NotImplementedError, match="requires at least one contraction label"):
        dml.trace(
            OuterProduct(),
            inputs={
                "a": dml.TensorSpec([2, 3], "float32"),
                "b": dml.TensorSpec([2, 4], "float32"),
            },
            name="einsum_outer_product",
        )


def test_einsum_rejects_repeated_labels_within_one_operand():
    class RepeatedLabels(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.einsum("ii,ij->j", a, b), "y")

    with pytest.raises(NotImplementedError, match="repeated labels within one operand"):
        dml.trace(
            RepeatedLabels(),
            inputs={
                "a": dml.TensorSpec([4, 4], "float32"),
                "b": dml.TensorSpec([4, 3], "float32"),
            },
            name="einsum_repeated_labels",
        )


def test_einsum_rejects_dynamic_shapes():
    batch = dml.Dim("batch", min=1, max=4, typical=2, buckets=(2, 4))
    width = dml.Dim("width", min=2, max=8, typical=4, buckets=(4, 8))
    depth = dml.Dim("depth", min=2, max=8, typical=4, buckets=(4, 8))

    class DynamicEinsum(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.einsum("bmk,bkn->bmn", a, b), "y")

    with pytest.raises(ValueError, match="supports only static input shapes"):
        dml.trace(
            DynamicEinsum(),
            inputs={
                "a": dml.TensorSpec([batch, width, depth], "float32"),
                "b": dml.TensorSpec([batch, depth, 5], "float32"),
            },
            name="einsum_dynamic",
        )


def test_einsum_requires_explicit_output():
    class ImplicitOutput(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.einsum("mk,kn", a, b), "y")

    with pytest.raises(NotImplementedError, match="requires an explicit output"):
        dml.trace(
            ImplicitOutput(),
            inputs={
                "a": dml.TensorSpec([3, 4], "float32"),
                "b": dml.TensorSpec([4, 5], "float32"),
            },
            name="einsum_implicit_output",
        )
