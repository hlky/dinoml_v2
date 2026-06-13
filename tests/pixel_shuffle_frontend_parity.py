from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml


ATOL = 1e-5
RTOL = 1e-5


@dataclass(frozen=True)
class PixelShuffleFrontendCase:
    name: str
    kind: str
    input_shape: tuple[int, int, int, int]
    upscale_factor: int
    dtype: str = "float32"


PIXEL_SHUFFLE_FRONTEND_CASES = (
    PixelShuffleFrontendCase(
        name="pixel_shuffle_module_f32",
        kind="module",
        input_shape=(2, 8, 3, 4),
        upscale_factor=2,
    ),
    PixelShuffleFrontendCase(
        name="pixel_shuffle_functional_f32",
        kind="functional",
        input_shape=(2, 8, 3, 4),
        upscale_factor=2,
    ),
)


class _PixelShuffleModuleCase(dml.Module):
    def __init__(self, case: PixelShuffleFrontendCase):
        self.shuffle = dml.nn.PixelShuffle(case.upscale_factor)

    def forward(self, x):
        return dml.ops.output(self.shuffle(x), "y")


class _PixelShuffleFunctionalCase(dml.Module):
    def __init__(self, case: PixelShuffleFrontendCase):
        self.case = case

    def forward(self, x):
        return dml.ops.output(dml.nn.functional.pixel_shuffle(x, self.case.upscale_factor), "y")


def trace_pixel_shuffle_frontend_spec(case: PixelShuffleFrontendCase, *, dtype: str | None = None):
    dtype = dtype or case.dtype
    module = _PixelShuffleModuleCase(case) if case.kind == "module" else _PixelShuffleFunctionalCase(case)
    return dml.trace(
        module,
        inputs={"x": dml.TensorSpec(list(case.input_shape), dtype)},
        name=case.name,
    )


def random_inputs(case: PixelShuffleFrontendCase, *, dtype: str | None = None, seed: int = 7) -> dict[str, np.ndarray]:
    dtype = dtype or case.dtype
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
    if dtype == "float16":
        x = x.astype(np.float16)
    return {"x": x}


def torch_oracle(case: PixelShuffleFrontendCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    x = torch.from_numpy(np.asarray(inputs["x"], dtype=np.float32))
    if case.kind == "module":
        result = torch.nn.PixelShuffle(case.upscale_factor)(x)
    else:
        result = torch.nn.functional.pixel_shuffle(x, case.upscale_factor)
    return result.cpu().numpy().astype(np.float32, copy=False)
