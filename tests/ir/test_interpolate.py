from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.interpolate_parity import (
    ATOL,
    DYNAMIC_INTERPOLATE_CASE,
    INTERPOLATE_CASES,
    RTOL,
    random_inputs,
    torch_oracle,
    trace_interpolate_spec,
)


@pytest.mark.parametrize("case", INTERPOLATE_CASES, ids=lambda case: case.name)
def test_interpolate_rewrites_to_upsampling_family_and_matches_torch(case):
    spec = trace_interpolate_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == ["permute", case.expected_op, "permute"]
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=ATOL, rtol=RTOL)


def test_interpolate_dynamic_shape_spec_tracks_spatial_dims():
    spec = trace_interpolate_spec(DYNAMIC_INTERPOLATE_CASE)
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    output_tensor = tensors[spec.ir["outputs"][0]["tensor"]]
    height_spec = output_tensor["shape_spec"][2]
    width_spec = output_tensor["shape_spec"][3]

    assert [node["op"] for node in spec.ir["nodes"]] == ["permute", "upsampling2d", "permute"]
    assert output_tensor["shape_spec"][:2] == [2, 3]
    assert height_spec["kind"] == "int_expr"
    assert height_spec["op"] == "mul"
    assert height_spec["lhs"]["kind"] == "dim"
    assert height_spec["lhs"]["name"] == "height"
    assert height_spec["rhs"] == 2
    assert width_spec["kind"] == "int_expr"
    assert width_spec["op"] == "mul"
    assert width_spec["lhs"]["kind"] == "dim"
    assert width_spec["lhs"]["name"] == "width"
    assert width_spec["rhs"] == 2


def test_interpolate_rejects_size_path():
    class SizeModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.interpolate(x, size=8, mode="nearest"), "y")

    with pytest.raises(NotImplementedError, match="size="):
        dml.trace(
            SizeModule(),
            inputs={"x": dml.TensorSpec([2, 3, 5], "float32")},
            name="interpolate_size_unsupported",
        )


def test_interpolate_rejects_non_uniform_scale_factor():
    class NonUniformScaleModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.interpolate(x, scale_factor=(2.0, 3.0), mode="nearest"), "y")

    with pytest.raises(NotImplementedError, match="uniform scale_factor"):
        dml.trace(
            NonUniformScaleModule(),
            inputs={"x": dml.TensorSpec([2, 3, 4, 5], "float32")},
            name="interpolate_non_uniform_scale",
        )


def test_interpolate_rejects_antialias():
    class AntialiasModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.interpolate(x, scale_factor=2.0, mode="bilinear", antialias=True), "y")

    with pytest.raises(NotImplementedError, match="antialias=True"):
        dml.trace(
            AntialiasModule(),
            inputs={"x": dml.TensorSpec([2, 3, 4, 5], "float32")},
            name="interpolate_antialias_unsupported",
        )
