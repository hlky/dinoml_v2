from __future__ import annotations

import pytest
import numpy as np

import dinoml as dml
from dinoml.reference import reference_numpy
from dinoml.passes import PassManager, validate_ir
from tests.cases import GraphCase, ir_cases


def _node_ops(spec) -> set[str]:
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    ops = {str(node["op"]) for node in lowered["nodes"]}
    for node in lowered["nodes"]:
        if node["op"] == "fused_elementwise":
            ops.update(str(sub_op["op"]) for sub_op in node.get("attrs", {}).get("sub_ops", ()))
    return ops


@pytest.mark.parametrize("case", ir_cases(), ids=lambda case: case.name)
def test_ir_traces_validates_and_reference_executes(case: GraphCase):
    spec = case.build_spec()
    ops = _node_ops(spec)
    materialized_expected_ops = {
        op
        for op in case.expected_ops
        if op
        not in {
            "reshape",
            "flatten",
            "unsqueeze",
            "squeeze",
            "identity",
            "transpose",
            "split",
            "chunk",
            "meshgrid",
            "pixel_shuffle",
            "pixel_unshuffle",
        }
    }
    assert materialized_expected_ops <= ops

    outputs = reference_numpy(spec, case.inputs())

    assert {output["name"] for output in spec.ir["outputs"]} == set(outputs)
    for name, value in outputs.items():
        assert tuple(value.shape) == tuple(next(output["shape"] for output in spec.ir["outputs"] if output["name"] == name))


def test_add_followed_by_layer_norm_fuses_to_multi_output_op():
    class AddLayerNormCandidate(dml.Module):
        def forward(self, x, residual, weight, bias):
            summed = x + residual
            return {
                "summed": dml.ops.output(summed, "summed"),
                "normalized": dml.ops.output(dml.ops.layer_norm(summed, weight, bias), "normalized"),
            }

    spec = dml.trace(
        AddLayerNormCandidate(),
        inputs={
            "x": dml.TensorSpec([2, 4], "float32"),
            "residual": dml.TensorSpec([2, 4], "float32"),
            "weight": dml.TensorSpec([4], "float32"),
            "bias": dml.TensorSpec([4], "float32"),
        },
        name="add_layer_norm_fusion_contract",
    )

    lowered, _ = PassManager().run(spec.ir)

    assert [node["op"] for node in lowered["nodes"]] == ["add_layer_norm"]
    assert len(lowered["nodes"][0]["outputs"]) == 2


def test_contiguous_dynamic_slice_followed_by_reshape_becomes_offset_view():
    class SliceReshapeCandidate(dml.Module):
        def forward(self, x):
            sliced = dml.ops.dynamic_slice(x, [0, 1, 0], [1, 1, 4])
            return dml.ops.output(dml.ops.reshape(sliced, [1, 4]), "output")

    spec = dml.trace(
        SliceReshapeCandidate(),
        inputs={"x": dml.TensorSpec([1, 3, 4], "float32")},
        name="dynamic_slice_offset_view_contract",
    )

    lowered, _ = PassManager().run(spec.ir)

    assert "dynamic_slice" not in [node["op"] for node in lowered["nodes"]]
    views = lowered["metadata"]["memory_plan"]["views"]["views"]
    output_tensor = lowered["outputs"][0]["tensor"]
    output_view = next(view for view in views if view["tensor"] == output_tensor)
    assert output_view["source"] == "x"
    assert output_view["offset_elements"] == 4

    outputs = reference_numpy(spec, {"x": np.arange(12, dtype=np.float32).reshape(1, 3, 4)})
    assert outputs["output"].tolist() == [[4.0, 5.0, 6.0, 7.0]]


def test_nested_reshape_views_flatten_before_lowering():
    class NestedReshapeCandidate(dml.Module):
        def forward(self, x):
            reshaped = dml.ops.reshape(x, [2, 6])
            return dml.ops.output(dml.ops.reshape(reshaped, [3, 4]), "output")

    spec = dml.trace(
        NestedReshapeCandidate(),
        inputs={"x": dml.TensorSpec([1, 2, 6], "float32")},
        name="nested_reshape_view_contract",
    )

    lowered, _ = PassManager().run(spec.ir)

    views = lowered["metadata"]["memory_plan"]["views"]["views"]
    output_tensor = lowered["outputs"][0]["tensor"]
    output_view = next(view for view in views if view["tensor"] == output_tensor)
    assert output_view["source"] == "x"
    assert output_view["offset_elements"] == 0

    outputs = reference_numpy(spec, {"x": np.arange(12, dtype=np.float32).reshape(1, 2, 6)})
    assert outputs["output"].tolist() == [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]


def test_dynamic_slice_nested_views_flatten_to_original_source():
    class SliceNestedReshapeCandidate(dml.Module):
        def forward(self, x):
            sliced = dml.ops.dynamic_slice(x, [0, 1, 0], [1, 1, 4])
            reshaped = dml.ops.reshape(sliced, [1, 4])
            return dml.ops.output(dml.ops.reshape(reshaped, [2, 2]), "output")

    spec = dml.trace(
        SliceNestedReshapeCandidate(),
        inputs={"x": dml.TensorSpec([1, 3, 4], "float32")},
        name="dynamic_slice_nested_view_contract",
    )

    lowered, _ = PassManager().run(spec.ir)

    assert "dynamic_slice" not in [node["op"] for node in lowered["nodes"]]
    views = lowered["metadata"]["memory_plan"]["views"]["views"]
    output_tensor = lowered["outputs"][0]["tensor"]
    output_view = next(view for view in views if view["tensor"] == output_tensor)
    assert output_view["source"] == "x"
    assert output_view["offset_elements"] == 4

    outputs = reference_numpy(spec, {"x": np.arange(12, dtype=np.float32).reshape(1, 3, 4)})
    assert outputs["output"].tolist() == [[4.0, 5.0], [6.0, 7.0]]


def test_sliced_add_followed_by_layer_norm_fuses_without_full_sequence_add():
    class SlicedAddLayerNormCandidate(dml.Module):
        def forward(self, x, residual, weight, bias):
            summed = x + residual
            pooled = dml.ops.reshape(dml.ops.dynamic_slice(summed, [0, 1, 0], [1, 1, 4]), [1, 4])
            return dml.ops.output(dml.ops.layer_norm(pooled, weight, bias), "output")

    spec = dml.trace(
        SlicedAddLayerNormCandidate(),
        inputs={
            "x": dml.TensorSpec([1, 3, 4], "float32"),
            "residual": dml.TensorSpec([1, 3, 4], "float32"),
            "weight": dml.TensorSpec([4], "float32"),
            "bias": dml.TensorSpec([4], "float32"),
        },
        name="sliced_add_layer_norm_contract",
    )

    lowered, _ = PassManager().run(spec.ir)

    assert [node["op"] for node in lowered["nodes"]] == ["add_layer_norm"]
    assert lowered["nodes"][0]["inputs"][0].endswith("_x_slice")
    assert lowered["nodes"][0]["inputs"][1].endswith("_residual_slice")
    views = lowered["metadata"]["memory_plan"]["views"]["views"]
    sliced_sources = {view["source"]: view["offset_elements"] for view in views}
    assert sliced_sources["x"] == 4
    assert sliced_sources["residual"] == 4
