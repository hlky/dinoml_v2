from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

import dinoml as dml
from dinoml.lowering.ops import render_generated_kernels
from dinoml.reference import reference_numpy
from tests.one_hot_parity import ONE_HOT_CASES, invalid_inputs, random_inputs, torch_oracle, trace_one_hot_spec


@pytest.mark.parametrize("case", ONE_HOT_CASES, ids=lambda case: case.name)
def test_one_hot_reference_matches_torch(case):
    spec = trace_one_hot_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)
    output_tensor_name = spec.ir["outputs"][0]["tensor"]
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}

    assert Counter(node["op"] for node in spec.ir["nodes"])["one_hot"] == 1
    assert spec.ir["outputs"][0]["shape"] == list(tensors[output_tensor_name]["shape"])
    assert spec.ir["outputs"][0]["dtype"] == "int64"
    np.testing.assert_array_equal(actual, expected)


def test_one_hot_dynamic_shape_spec_appends_num_classes():
    case = next(case for case in ONE_HOT_CASES if case.name == "one_hot_dynamic_rank2_i64")
    spec = trace_one_hot_spec(case)
    output_tensor_name = spec.ir["outputs"][0]["tensor"]
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}

    assert tensors[output_tensor_name]["shape_spec"] == [*tensors["x"]["shape_spec"], case.num_classes]


def test_nn_functional_one_hot_traces_to_explicit_op():
    class Tiny(dml.nn.Module):
        def forward(self, x):
            return dml.ops.output(dml.nn.functional.one_hot(x, 4), "y")

    spec = dml.trace(
        Tiny(),
        inputs={"x": dml.TensorSpec([2, 3], "int64")},
        name="nn_functional_one_hot_trace",
    )

    assert [node["op"] for node in spec.ir["nodes"]] == ["one_hot"]
    assert spec.ir["outputs"][0]["shape"] == [2, 3, 4]
    assert spec.ir["outputs"][0]["dtype"] == "int64"


def test_one_hot_rejects_non_integer_input_dtype():
    class BadInputModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.one_hot(x, 4), "y")

    with pytest.raises(ValueError, match="input must have dtype int64 or int32"):
        dml.trace(
            BadInputModule(),
            inputs={"x": dml.TensorSpec([2, 3], "float32")},
            name="one_hot_bad_input_dtype",
        )


def test_one_hot_rejects_non_positive_num_classes():
    class BadNumClassesModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.one_hot(x, 0), "y")

    with pytest.raises(ValueError, match="num_classes must be positive"):
        dml.trace(
            BadNumClassesModule(),
            inputs={"x": dml.TensorSpec([2, 3], "int64")},
            name="one_hot_bad_num_classes",
        )


def test_one_hot_reference_rejects_out_of_range_inputs():
    case = next(case for case in ONE_HOT_CASES if case.name == "one_hot_rank2_i64")
    spec = trace_one_hot_spec(case)

    with pytest.raises(ValueError, match="out of bounds"):
        reference_numpy(spec, invalid_inputs(case))


def test_one_hot_cpu_generated_source_renders():
    case = next(case for case in ONE_HOT_CASES if case.name == "one_hot_rank2_i64")
    spec = trace_one_hot_spec(case)
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    source = render_generated_kernels("cpu", spec.ir["nodes"], tensor_map)[0]

    assert "static int one_hot_" in source
    assert "one_hot runtime output size mismatch" in source
