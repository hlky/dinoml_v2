from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml


ATOL = 1e-5
RTOL = 1e-5


@dataclass(frozen=True)
class PixelUnshuffleFrontendCase:
    name: str
    kind: str
    input_shape: tuple[int, int, int, int]
    downscale_factor: int
    dtype: str = "float32"


PIXEL_UNSHUFFLE_FRONTEND_CASES = (
    PixelUnshuffleFrontendCase(
        name="pixel_unshuffle_module_f32",
        kind="module",
        input_shape=(2, 3, 6, 4),
        downscale_factor=2,
    ),
    PixelUnshuffleFrontendCase(
        name="pixel_unshuffle_functional_f32",
        kind="functional",
        input_shape=(2, 3, 6, 4),
        downscale_factor=2,
    ),
)


class _PixelUnshuffleModuleCase(dml.Module):
    def __init__(self, case: PixelUnshuffleFrontendCase):
        self.unshuffle = dml.nn.PixelUnshuffle(case.downscale_factor)

    def forward(self, x):
        return dml.ops.output(self.unshuffle(x), "y")


class _PixelUnshuffleFunctionalCase(dml.Module):
    def __init__(self, case: PixelUnshuffleFrontendCase):
        self.case = case

    def forward(self, x):
        return dml.ops.output(dml.nn.functional.pixel_unshuffle(x, self.case.downscale_factor), "y")


def trace_pixel_unshuffle_frontend_spec(case: PixelUnshuffleFrontendCase, *, dtype: str | None = None):
    dtype = dtype or case.dtype
    module = _PixelUnshuffleModuleCase(case) if case.kind == "module" else _PixelUnshuffleFunctionalCase(case)
    return dml.trace(
        module,
        inputs={"x": dml.TensorSpec(list(case.input_shape), dtype)},
        name=case.name,
    )


def random_inputs(case: PixelUnshuffleFrontendCase, *, dtype: str | None = None, seed: int = 7) -> dict[str, np.ndarray]:
    dtype = dtype or case.dtype
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
    if dtype == "float16":
        x = x.astype(np.float16)
    return {"x": x}


def torch_oracle(case: PixelUnshuffleFrontendCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    x = torch.from_numpy(np.asarray(inputs["x"], dtype=np.float32))
    if case.kind == "module":
        result = torch.nn.PixelUnshuffle(case.downscale_factor)(x)
    else:
        result = torch.nn.functional.pixel_unshuffle(x, case.downscale_factor)
    return result.cpu().numpy().astype(np.float32, copy=False)
