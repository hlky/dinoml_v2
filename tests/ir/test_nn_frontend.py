from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy


def test_dml_nn_linear_layernorm_activation_trace_and_reference():
    class Tiny(dml.nn.Module):
        def __init__(self):
            self.fc = dml.nn.Linear(4, 3)
            self.norm = dml.nn.LayerNorm(3, eps=1e-5)
            self.act = dml.nn.ReLU()

        def forward(self, x):
            return dml.ops.output(self.act(self.norm(self.fc(x))), "y")

    constants = {
        "fc_weight": np.array(
            [
                [0.5, -0.25, 0.125, 0.0],
                [0.25, 0.0, -0.5, 0.75],
                [-0.125, 0.5, 0.25, -0.25],
            ],
            dtype=np.float32,
        ),
        "fc_bias": np.array([0.25, -0.5, 0.125], dtype=np.float32),
        "norm_weight": np.array([1.0, 0.5, 1.5], dtype=np.float32),
        "norm_bias": np.array([0.0, 0.25, -0.125], dtype=np.float32),
    }
    spec = dml.trace(
        Tiny(),
        inputs={"x": dml.TensorSpec([2, 4], "float32")},
        constants=constants,
        name="nn_linear_norm",
    )

    assert [node["op"] for node in spec.ir["nodes"]] == ["gemm_rcr_bias", "layer_norm", "relu"]
    assert [constant["name"] for constant in spec.ir["constants"]] == [
        "fc_weight",
        "fc_bias",
        "norm_weight",
        "norm_bias",
    ]
    assert spec.ir["outputs"][0]["shape"] == [2, 3]

    inputs = {"x": np.array([[1.0, -1.0, 0.5, 2.0], [-0.5, 0.25, 1.5, -1.0]], dtype=np.float32)}
    actual = reference_numpy(spec, inputs)["y"]
    hidden = inputs["x"] @ constants["fc_weight"].T + constants["fc_bias"]
    mean = hidden.mean(axis=-1, keepdims=True)
    var = ((hidden - mean) ** 2).mean(axis=-1, keepdims=True)
    expected = (hidden - mean) / np.sqrt(var + 1e-5)
    expected = np.maximum(expected * constants["norm_weight"] + constants["norm_bias"], 0.0)
    np.testing.assert_allclose(actual, expected.astype(np.float32), atol=1e-5, rtol=1e-5)


def test_dml_nn_conv2d_embedding_and_sequential_exports():
    class TinyVision(dml.nn.Module):
        def __init__(self):
            self.conv = dml.nn.Conv2d(2, 3, kernel_size=3, padding=1, activation="relu")
            self.embedding = dml.nn.Embedding(5, 3)
            self.proj = dml.nn.Sequential(dml.nn.Linear(3, 4), dml.nn.GELU(), dml.nn.Linear(4, 2, bias=False))

        def forward(self, x, indices, features):
            conv = self.conv(x)
            embedded = self.embedding(indices)
            projected = self.proj(features)
            return {
                "conv": dml.ops.output(conv, "conv"),
                "embedded": dml.ops.output(embedded, "embedded"),
                "projected": dml.ops.output(projected, "projected"),
            }

    model = TinyVision()
    named_parameters = dict(model.named_parameters())
    assert list(named_parameters) == [
        "conv.weight",
        "conv.bias",
        "embedding.weight",
        "proj.0.weight",
        "proj.0.bias",
        "proj.2.weight",
    ]
    assert named_parameters["proj.2.weight"].name == "proj_2_weight"

    class TinyList(dml.nn.Module):
        def __init__(self):
            self.layers = dml.nn.ModuleList()
            self.layers.append(dml.nn.Linear(3, 2, bias=False))

    appended = dict(TinyList().named_parameters())
    assert list(appended) == ["layers.0.weight"]
    assert appended["layers.0.weight"].name == "layers_0_weight"

    spec = dml.trace(
        model,
        inputs={
            "x": dml.TensorSpec([1, 2, 4, 4], "float32"),
            "indices": dml.TensorSpec([2, 2], "int64"),
            "features": dml.TensorSpec([2, 3], "float32"),
        },
        name="nn_core_layers",
    )

    assert dml.nn.Linear is not None
    assert "nn" in dml.__all__
    assert [node["op"] for node in spec.ir["nodes"]] == [
        "conv2d_bias_relu",
        "embedding",
        "gemm_rcr_bias",
        "gelu",
        "gemm_rcr",
    ]
    output_shapes = {output["name"]: output["shape"] for output in spec.ir["outputs"]}
    assert output_shapes == {
        "conv": [1, 3, 4, 4],
        "embedded": [2, 2, 3],
        "projected": [2, 2],
    }


def test_dml_nn_transposed_conv2d_trace_and_validation():
    class TinyTranspose(dml.nn.Module):
        def forward(self, x, weight, bias, residual):
            fused = dml.ops.transposed_conv2d_bias_add_relu(
                x,
                weight,
                bias,
                residual,
                stride=2,
                padding=1,
                output_padding=1,
            )
            base = dml.ops.transposed_conv2d(x, weight, stride=2, padding=1, output_padding=1)
            return {
                "fused": dml.ops.output(fused, "fused"),
                "base": dml.ops.output(base, "base"),
            }

    spec = dml.trace(
        TinyTranspose(),
        inputs={
            "x": dml.TensorSpec([1, 2, 3, 4], "float32"),
            "weight": dml.TensorSpec([2, 5, 3, 3], "float32"),
            "bias": dml.TensorSpec([5], "float32"),
            "residual": dml.TensorSpec([1, 5, 6, 8], "float32"),
        },
        name="nn_transposed_conv2d",
    )

    assert [node["op"] for node in spec.ir["nodes"]] == ["transposed_conv2d_bias_add_relu", "transposed_conv2d"]
    output_shapes = {output["name"]: output["shape"] for output in spec.ir["outputs"]}
    assert output_shapes == {"fused": [1, 5, 6, 8], "base": [1, 5, 6, 8]}

    class BadOutputPadding(dml.nn.Module):
        def forward(self, x, weight):
            return dml.ops.output(dml.ops.transposed_conv2d(x, weight, stride=2, output_padding=2), "y")

    with pytest.raises(ValueError, match="output_padding must be smaller than stride"):
        dml.trace(
            BadOutputPadding(),
            inputs={
                "x": dml.TensorSpec([1, 2, 3, 4], "float32"),
                "weight": dml.TensorSpec([2, 5, 3, 3], "float32"),
            },
            name="nn_transposed_conv2d_bad_output_padding",
        )

    class BadGroups(dml.nn.Module):
        def forward(self, x, weight):
            return dml.ops.output(dml.ops.transposed_conv2d(x, weight, groups=2), "y")

    with pytest.raises(NotImplementedError, match="groups=1 only"):
        dml.trace(
            BadGroups(),
            inputs={
                "x": dml.TensorSpec([1, 2, 3, 4], "float32"),
                "weight": dml.TensorSpec([2, 5, 3, 3], "float32"),
            },
            name="nn_transposed_conv2d_bad_groups",
        )
