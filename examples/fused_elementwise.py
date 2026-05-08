from __future__ import annotations

import numpy as np

import dinoml as dml


INPUT_SHAPE = [2, 3, 4]


class FusedElementwise(dml.Module):
    def __init__(self):
        self.scale = dml.Parameter([4], dtype="float32")
        self.bias = dml.Parameter([4], dtype="float32")

    def forward(self, x):
        y = dml.ops.mul(x, self.scale)
        y = dml.ops.add(y, self.bias)
        y = dml.ops.sub(y, dml.ops.sigmoid(x))
        y = dml.ops.relu(y)
        y = dml.ops.mul(y, 0.5)
        return dml.ops.output(y, "y")


def build_constants() -> dict[str, np.ndarray]:
    return {
        "scale": np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float32),
        "bias": np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32),
    }


def build_spec() -> dml.ir.ModelSpec:
    return dml.trace(
        FusedElementwise(),
        inputs={"x": dml.TensorSpec(INPUT_SHAPE, "float32")},
        constants=build_constants(),
        name="fused_elementwise",
    )


def build_validation_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    return {"x": rng.standard_normal(INPUT_SHAPE).astype(np.float32)}


def torch_reference(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import torch

    constants = build_constants()
    x = torch.from_numpy(inputs["x"])
    scale = torch.from_numpy(constants["scale"])
    bias = torch.from_numpy(constants["bias"])
    y = x * scale
    y = y + bias
    y = y - torch.sigmoid(x)
    y = torch.relu(y)
    y = y * 0.5
    return {"y": y.numpy().astype(np.float32)}
