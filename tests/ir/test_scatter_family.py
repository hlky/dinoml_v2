from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.scatter_parity import (
    ATOL_BY_DTYPE,
    RTOL_BY_DTYPE,
    SCATTER_CASES,
    random_inputs,
    torch_oracle,
    trace_scatter_spec,
)


@pytest.mark.parametrize("case", SCATTER_CASES, ids=lambda case: case.name)
def test_scatter_family_reference_matches_torch(case):
    spec = trace_scatter_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert Counter(node["op"] for node in spec.ir["nodes"])[case.op] == 1
    if case.dtype == "bool":
        np.testing.assert_array_equal(actual.astype(np.bool_), expected.astype(np.bool_))
    else:
        np.testing.assert_allclose(
            actual.astype(np.float32),
            expected.astype(np.float32),
            atol=ATOL_BY_DTYPE[case.dtype],
            rtol=RTOL_BY_DTYPE[case.dtype],
        )


def test_scatter_rejects_dynamic_shapes():
    rows = dml.Dim("rows", min=2, max=4, typical=3, buckets=(3, 4))

    class DynamicScatter(dml.Module):
        def forward(self, x, index, source):
            return dml.ops.output(x.scatter(1, index, source), "y")

    with pytest.raises(ValueError, match="supports only static input, index, and source shapes"):
        dml.trace(
            DynamicScatter(),
            inputs={
                "x": dml.TensorSpec([2, rows, 3], "float32"),
                "index": dml.TensorSpec([2, 2, 3], "int64"),
                "source": dml.TensorSpec([2, 2, 3], "float32"),
            },
            name="scatter_dynamic_rejected",
        )


def test_scatter_rejects_source_shape_mismatch():
    class BadScatter(dml.Module):
        def forward(self, x, index, source):
            return dml.ops.output(x.scatter(1, index, source), "y")

    with pytest.raises(ValueError, match="source axis 1 size 3 must match index dim 2"):
        dml.trace(
            BadScatter(),
            inputs={
                "x": dml.TensorSpec([2, 4, 3], "float32"),
                "index": dml.TensorSpec([2, 2, 3], "int64"),
                "source": dml.TensorSpec([2, 3, 3], "float32"),
            },
            name="scatter_bad_source_shape",
        )


def test_scatter_add_rejects_bool_dtype():
    class BoolScatterAdd(dml.Module):
        def forward(self, x, index, source):
            return dml.ops.output(x.scatter_add(0, index, source), "y")

    with pytest.raises(ValueError, match="scatter_add does not support dtype bool"):
        dml.trace(
            BoolScatterAdd(),
            inputs={
                "x": dml.TensorSpec([3, 2], "bool"),
                "index": dml.TensorSpec([2, 2], "int64"),
                "source": dml.TensorSpec([2, 2], "bool"),
            },
            name="scatter_add_bool_rejected",
        )


def test_scatter_reduce_rejects_unsupported_reduction():
    class BadReduce(dml.Module):
        def forward(self, x, index, source):
            return dml.ops.output(x.scatter_reduce(1, index, source, reduce="mean"), "y")

    with pytest.raises(ValueError, match="reduce must be one of"):
        dml.trace(
            BadReduce(),
            inputs={
                "x": dml.TensorSpec([2, 4, 3], "float32"),
                "index": dml.TensorSpec([2, 2, 3], "int64"),
                "source": dml.TensorSpec([2, 2, 3], "float32"),
            },
            name="scatter_reduce_bad_reduction",
        )


def test_scatter_reduce_rejects_include_self_false():
    class BadIncludeSelf(dml.Module):
        def forward(self, x, index, source):
            return dml.ops.output(x.scatter_reduce(1, index, source, reduce="sum", include_self=False), "y")

    with pytest.raises(ValueError, match="supports only include_self=True"):
        dml.trace(
            BadIncludeSelf(),
            inputs={
                "x": dml.TensorSpec([2, 4, 3], "float32"),
                "index": dml.TensorSpec([2, 2, 3], "int64"),
                "source": dml.TensorSpec([2, 2, 3], "float32"),
            },
            name="scatter_reduce_include_self_false",
        )
