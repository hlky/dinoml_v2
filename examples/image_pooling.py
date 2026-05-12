from __future__ import annotations

import numpy as np

import dinoml as dml


INPUT_SHAPE = [1, 1, 5, 6]


class ImagePooling(dml.Module):
    def forward(self, x):
        x = dml.ops.pad(x, (1, 2, 1, 0), value=-0.25)
        x = dml.ops.avg_pool2d(x, kernel_size=(2, 3), stride=(1, 2), padding=(0, 1))
        x = dml.ops.max_pool2d(x, kernel_size=2, stride=1, padding=0)
        return dml.ops.output(x, "features")


def build_spec() -> dml.ir.ModelSpec:
    return dml.trace(
        ImagePooling(),
        inputs={"x": dml.TensorSpec(INPUT_SHAPE, "float32")},
        name="image_pooling",
    )


def build_validation_inputs() -> dict[str, np.ndarray]:
    values = np.linspace(-1.0, 2.0, num=int(np.prod(INPUT_SHAPE)), dtype=np.float32)
    return {"x": values.reshape(INPUT_SHAPE)}


def torch_reference(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(inputs["x"])
    x = F.pad(x, (1, 2, 1, 0), value=-0.25)
    x = F.avg_pool2d(x, kernel_size=(2, 3), stride=(1, 2), padding=(0, 1))
    x = F.max_pool2d(x, kernel_size=2, stride=1, padding=0)
    return {"features": x.numpy().astype(np.float32)}
