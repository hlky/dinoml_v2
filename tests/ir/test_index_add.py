from __future__ import annotations

from collections import Counter
import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.index_add_parity import (
    ATOL_BY_DTYPE,
    INDEX_ADD_CASES,
    RTOL_BY_DTYPE,
    random_inputs,
    torch_oracle,
    trace_index_add_spec,
)


@pytest.mark.parametrize("case", INDEX_ADD_CASES, ids=lambda case: case.name)
def test_index_add_reference_matches_torch(case):
    spec = trace_index_add_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert Counter(node["op"] for node in spec.ir["nodes"])["index_add"] == 1
    x_tensor = next(tensor for tensor in spec.ir["tensors"] if tensor["name"] == "x")
    assert spec.ir["outputs"][0]["shape"] == list(x_tensor["shape"])
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected.astype(np.float32),
        atol=ATOL_BY_DTYPE[case.dtype],
        rtol=RTOL_BY_DTYPE[case.dtype],
    )


def test_index_add_rejects_non_integer_index_dtype():
    class BadIndexModule(dml.Module):
        def forward(self, x, index, source):
            return dml.ops.output(dml.ops.index_add(x, 0, index, source), "y")

    with pytest.raises(ValueError, match="index must have dtype int64 or int32"):
        dml.trace(
            BadIndexModule(),
            inputs={
                "x": dml.TensorSpec([4, 3], "float32"),
                "index": dml.TensorSpec([5], "float32"),
                "source": dml.TensorSpec([5, 3], "float32"),
            },
            name="index_add_bad_index_dtype",
        )


def test_index_add_rejects_bool_inputs():
    class BoolModule(dml.Module):
        def forward(self, x, index, source):
            return dml.ops.output(dml.ops.index_add(x, 0, index, source), "y")

    with pytest.raises(ValueError, match="does not support dtype bool"):
        dml.trace(
            BoolModule(),
            inputs={
                "x": dml.TensorSpec([4, 3], "bool"),
                "index": dml.TensorSpec([5], "int64"),
                "source": dml.TensorSpec([5, 3], "bool"),
            },
            name="index_add_bool",
        )


def test_index_add_rejects_source_shape_mismatch():
    class BadShapeModule(dml.Module):
        def forward(self, x, index, source):
            return dml.ops.output(dml.ops.index_add(x, 1, index, source), "y")

    with pytest.raises(ValueError, match="source dim 1 size 3 must match index length 2"):
        dml.trace(
            BadShapeModule(),
            inputs={
                "x": dml.TensorSpec([2, 4, 3], "float32"),
                "index": dml.TensorSpec([2], "int64"),
                "source": dml.TensorSpec([2, 3, 3], "float32"),
            },
            name="index_add_bad_source_shape",
        )


def test_index_add_dynamic_shape_spec_tracks_input_shape():
    rows = dml.Dim("rows", min=2, max=5, typical=4, buckets=(4, 5))
    cols = dml.Dim("cols", min=2, max=4, typical=3, buckets=(3, 4))
    index_len = dml.Dim("index_len", min=2, max=6, typical=4, buckets=(4, 6))

    class DynamicModule(dml.Module):
        def forward(self, x, index, source):
            return dml.ops.output(dml.ops.index_add(x, 0, index, source), "y")

    spec = dml.trace(
        DynamicModule(),
        inputs={
            "x": dml.TensorSpec([rows, cols], "float32"),
            "index": dml.TensorSpec([index_len], "int64"),
            "source": dml.TensorSpec([index_len, cols], "float32"),
        },
        name="index_add_dynamic",
    )
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    output_tensor_name = spec.ir["outputs"][0]["tensor"]
    assert tensors[output_tensor_name]["shape_spec"] == tensors["x"]["shape_spec"]
    assert tensors["source"]["shape_spec"][0] == tensors["index"]["shape_spec"][0]


def test_index_add_rejects_dynamic_symbolic_length_mismatch():
    rows = dml.Dim("rows", min=2, max=5, typical=4, buckets=(4, 5))
    cols = dml.Dim("cols", min=2, max=4, typical=3, buckets=(3, 4))
    index_len = dml.Dim("index_len", min=2, max=6, typical=4, buckets=(4, 6))
    wrong_len = dml.Dim("wrong_len", min=2, max=6, typical=4, buckets=(4, 6))

    class DynamicModule(dml.Module):
        def forward(self, x, index, source):
            return dml.ops.output(dml.ops.index_add(x, 0, index, source), "y")

    with pytest.raises(ValueError, match="must match index length"):
        dml.trace(
            DynamicModule(),
            inputs={
                "x": dml.TensorSpec([rows, cols], "float32"),
                "index": dml.TensorSpec([index_len], "int64"),
                "source": dml.TensorSpec([wrong_len, cols], "float32"),
            },
            name="index_add_dynamic_bad_length",
        )
