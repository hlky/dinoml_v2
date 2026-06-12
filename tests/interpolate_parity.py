from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import dinoml as dml


ATOL = 1e-5
RTOL = 1e-5


_HEIGHT = dml.Dim("height", min=2, max=5, typical=4, buckets=(4, 5))
_WIDTH = dml.Dim("width", min=3, max=6, typical=5, buckets=(5, 6))


@dataclass(frozen=True)
class InterpolateCase:
    name: str
    input_shape: tuple[int, ...]
    scale_factor: float | tuple[float, ...]
    mode: str
    expected_op: str
    align_corners: bool | None = None
    input_spec_shape: tuple[Any, ...] | None = None

    @property
    def resolved_input_spec_shape(self) -> tuple[Any, ...]:
        return self.input_shape if self.input_spec_shape is None else self.input_spec_shape


INTERPOLATE_CASES = (
    InterpolateCase("interpolate_1d_linear_align_corners", (2, 3, 5), 2.0, "linear", "upsampling1d", True),
    InterpolateCase("interpolate_1d_nearest_exact", (2, 4, 6), (2.0,), "nearest-exact", "upsampling1d"),
    InterpolateCase("interpolate_2d_bilinear", (2, 3, 4, 5), (2.0, 2.0), "bilinear", "upsampling2d", False),
    InterpolateCase("interpolate_2d_nearest", (2, 4, 5, 6), 2.0, "nearest", "upsampling2d"),
    InterpolateCase("interpolate_3d_trilinear_align_corners", (1, 2, 3, 4, 5), 2.0, "trilinear", "upsampling3d", True),
    InterpolateCase("interpolate_3d_nearest_exact", (1, 3, 4, 5, 6), (2.0, 2.0, 2.0), "nearest-exact", "upsampling3d"),
)

DYNAMIC_INTERPOLATE_CASE = InterpolateCase(
    "interpolate_2d_dynamic_nearest",
    (2, 3, 4, 5),
    2.0,
    "nearest",
    "upsampling2d",
    input_spec_shape=(2, 3, _HEIGHT, _WIDTH),
)


class _InterpolateModule(dml.Module):
    def __init__(self, case: InterpolateCase):
        self.case = case

    def forward(self, x):
        return dml.ops.output(
            dml.ops.interpolate(
                x,
                scale_factor=self.case.scale_factor,
                mode=self.case.mode,
                align_corners=self.case.align_corners,
            ),
            "y",
        )


def trace_interpolate_spec(case: InterpolateCase, dtype: str = "float32"):
    return dml.trace(
        _InterpolateModule(case),
        inputs={"x": dml.TensorSpec(list(case.resolved_input_spec_shape), dtype)},
        name=f"{case.name}_{dtype}",
    )


def random_inputs(case: InterpolateCase, dtype: str = "float32", *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
    if dtype == "float16":
        x = x.astype(np.float16)
    return {"x": x}


def torch_oracle(case: InterpolateCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    x = torch.from_numpy(np.asarray(inputs["x"], dtype=np.float32))
    result = torch.nn.functional.interpolate(
        x,
        scale_factor=case.scale_factor,
        mode=case.mode,
        align_corners=case.align_corners,
    )
    return result.cpu().numpy().astype(np.float32, copy=False)
