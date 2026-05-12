from __future__ import annotations

import numpy as np

import dinoml as dml


INPUT_SHAPE = [1, 8, 2, 3]
UPSCALE_FACTOR = 2


class SubpixelUpsample(dml.Module):
    def forward(self, x):
        x = dml.ops.pixel_shuffle(x, UPSCALE_FACTOR)
        return dml.ops.output(x, "image")


def build_spec() -> dml.ir.ModelSpec:
    return dml.trace(
        SubpixelUpsample(),
        inputs={"x": dml.TensorSpec(INPUT_SHAPE, "float32")},
        name="subpixel_upsample",
    )


def build_validation_inputs() -> dict[str, np.ndarray]:
    values = np.arange(int(np.prod(INPUT_SHAPE)), dtype=np.float32) / 16.0
    return {"x": values.reshape(INPUT_SHAPE)}


def torch_reference(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(inputs["x"])
    image = F.pixel_shuffle(x, UPSCALE_FACTOR)
    return {"image": image.numpy().astype(np.float32)}
