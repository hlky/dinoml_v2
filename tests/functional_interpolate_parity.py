from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import dinoml as dml


ATOL = 1e-5
RTOL = 1e-5


@dataclass(frozen=True)
class FunctionalInterpolateCase:
    name: str
    input_shape: tuple[int, ...]
    scale_factor: float | tuple[float, ...]
    mode: str
    expected_op: str
    align_corners: bool | None = None


FUNCTIONAL_INTERPOLATE_CASES = (
    FunctionalInterpolateCase("functional_interpolate_1d_linear_align_corners", (2, 3, 5), 2.0, "linear", "upsampling1d", True),
    FunctionalInterpolateCase("functional_interpolate_1d_nearest_exact", (2, 4, 6), (2.0,), "nearest-exact", "upsampling1d"),
    FunctionalInterpolateCase("functional_interpolate_2d_bilinear", (2, 3, 4, 5), (2.0, 2.0), "bilinear", "upsampling2d", False),
    FunctionalInterpolateCase("functional_interpolate_2d_nearest", (2, 4, 5, 6), 2.0, "nearest", "upsampling2d"),
    FunctionalInterpolateCase("functional_interpolate_3d_trilinear_align_corners", (1, 2, 3, 4, 5), 2.0, "trilinear", "upsampling3d", True),
    FunctionalInterpolateCase("functional_interpolate_3d_nearest_exact", (1, 3, 4, 5, 6), (2.0, 2.0, 2.0), "nearest-exact", "upsampling3d"),
)


class _FunctionalInterpolateModule(dml.Module):
    def __init__(self, case: FunctionalInterpolateCase):
        self.case = case

    def forward(self, x):
        return dml.ops.output(
            dml.nn.functional.interpolate(
                x,
                scale_factor=self.case.scale_factor,
                mode=self.case.mode,
                align_corners=self.case.align_corners,
            ),
            "y",
        )


def trace_functional_interpolate_spec(case: FunctionalInterpolateCase, dtype: str = "float32"):
    return dml.trace(
        _FunctionalInterpolateModule(case),
        inputs={"x": dml.TensorSpec(list(case.input_shape), dtype)},
        name=f"{case.name}_{dtype}",
    )


def random_inputs(case: FunctionalInterpolateCase, dtype: str = "float32", *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
    if dtype == "float16":
        x = x.astype(np.float16)
    return {"x": x}


def torch_oracle(case: FunctionalInterpolateCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    x = torch.from_numpy(np.asarray(inputs["x"], dtype=np.float32))
    result = torch.nn.functional.interpolate(
        x,
        scale_factor=case.scale_factor,
        mode=case.mode,
        align_corners=case.align_corners,
    )
    return result.cpu().numpy().astype(np.float32, copy=False)
