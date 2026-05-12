from __future__ import annotations

import numpy as np

import dinoml as dml


INPUT_SHAPE = [1, 1, 3, 4]
HEIGHT = INPUT_SHAPE[2]
WIDTH = INPUT_SHAPE[3]


class CoordinateRamp(dml.Module):
    def forward(self, x):
        batch_axis = dml.ops.full([1], 0.0, dtype="float32")
        channel_axis = dml.ops.full([1], 0.0, dtype="float32")
        y_axis = dml.ops.arange(HEIGHT, dtype="float32")
        x_axis = dml.ops.arange(WIDTH, dtype="float32")
        batch_grid, channel_grid, y_grid, x_grid = dml.ops.meshgrid(
            (batch_axis, channel_axis, y_axis, x_axis)
        )

        features = (
            x + batch_grid + channel_grid + y_grid * 0.25 + x_grid * 0.125
        )
        return dml.ops.output(dml.ops.relu(features), "features")


def build_spec() -> dml.ir.ModelSpec:
    return dml.trace(
        CoordinateRamp(),
        inputs={"x": dml.TensorSpec(INPUT_SHAPE, "float32")},
        name="coordinate_ramp",
    )


def build_validation_inputs() -> dict[str, np.ndarray]:
    values = np.linspace(-0.75, 0.75, num=int(np.prod(INPUT_SHAPE)), dtype=np.float32)
    return {"x": values.reshape(INPUT_SHAPE)}


def torch_reference(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import torch

    x = torch.from_numpy(inputs["x"])
    batch_axis = torch.full((1,), 0.0, dtype=torch.float32)
    channel_axis = torch.full((1,), 0.0, dtype=torch.float32)
    y_axis = torch.arange(HEIGHT, dtype=torch.float32)
    x_axis = torch.arange(WIDTH, dtype=torch.float32)
    batch_grid, channel_grid, y_grid, x_grid = torch.meshgrid(
        batch_axis, channel_axis, y_axis, x_axis, indexing="ij"
    )
    features = torch.relu(
        x + batch_grid + channel_grid + y_grid * 0.25 + x_grid * 0.125
    )
    return {"features": features.numpy().astype(np.float32)}
