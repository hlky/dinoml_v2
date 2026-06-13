from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import dinoml as dml


ATOL = 1e-5
RTOL = 1e-5


@dataclass(frozen=True)
class UpsampleCase:
    name: str
    input_shape: tuple[int, ...]
    mode: str
    expected_op: str
    size: int | tuple[int, ...] | None = None
    scale_factor: float | tuple[float, ...] | None = None
    align_corners: bool | None = None
    dtype: str = "float32"


UPSAMPLE_CASES = (
    UpsampleCase(
        name="upsample_1d_linear_scale_factor",
        input_shape=(2, 3, 5),
        scale_factor=2.0,
        mode="linear",
        align_corners=True,
        expected_op="upsampling1d",
    ),
    UpsampleCase(
        name="upsample_2d_nearest_size",
        input_shape=(2, 4, 4, 5),
        size=(8, 10),
        mode="nearest",
        expected_op="upsampling2d",
    ),
    UpsampleCase(
        name="upsample_3d_nearest_exact_scale_factor",
        input_shape=(1, 2, 3, 4, 5),
        scale_factor=(2.0, 2.0, 2.0),
        mode="nearest-exact",
        expected_op="upsampling3d",
    ),
)


class _UpsampleModule(dml.Module):
    def __init__(self, case: UpsampleCase):
        self.upsample = dml.nn.Upsample(
            size=case.size,
            scale_factor=case.scale_factor,
            mode=case.mode,
            align_corners=case.align_corners,
        )

    def forward(self, x):
        return dml.ops.output(self.upsample(x), "y")


def trace_upsample_spec(case: UpsampleCase, dtype: str | None = None):
    dtype = dtype or case.dtype
    return dml.trace(
        _UpsampleModule(case),
        inputs={"x": dml.TensorSpec(list(case.input_shape), dtype)},
        name=case.name,
    )


def random_inputs(case: UpsampleCase, dtype: str | None = None, *, seed: int = 7) -> dict[str, np.ndarray]:
    dtype = dtype or case.dtype
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
    if dtype == "float16":
        x = x.astype(np.float16)
    return {"x": x}


def torch_oracle(case: UpsampleCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    module = torch.nn.Upsample(
        size=case.size,
        scale_factor=case.scale_factor,
        mode=case.mode,
        align_corners=case.align_corners,
    )
    x = torch.from_numpy(np.asarray(inputs["x"], dtype=np.float32))
    result = module(x)
    return result.cpu().numpy().astype(np.float32, copy=False)
