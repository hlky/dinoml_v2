from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml


ATOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6}
RTOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6}


@dataclass(frozen=True)
class PaddingLayoutHelperCase:
    name: str
    op_name: str
    dtype: str
    input_shape: tuple[int, ...]


PADDING_LAYOUT_HELPER_CASES = (
    PaddingLayoutHelperCase("nhwc3to4_f32", "nhwc3to4", "float32", (2, 5, 7, 3)),
    PaddingLayoutHelperCase("nhwc3to4_f16", "nhwc3to4", "float16", (1, 6, 4, 3)),
    PaddingLayoutHelperCase("nhwc3to8_f32", "nhwc3to8", "float32", (2, 4, 6, 3)),
    PaddingLayoutHelperCase("nhwc3to8_f16", "nhwc3to8", "float16", (1, 3, 5, 3)),
    PaddingLayoutHelperCase("ndhwc3to8_f32", "ndhwc3to8", "float32", (2, 3, 4, 5, 3)),
    PaddingLayoutHelperCase("ndhwc3to8_f16", "ndhwc3to8", "float16", (1, 2, 3, 4, 3)),
)


class _PaddingLayoutHelperModule(dml.Module):
    def __init__(self, case: PaddingLayoutHelperCase):
        self.case = case

    def forward(self, x):
        if self.case.op_name == "nhwc3to4":
            y = dml.ops.nhwc3to4(x)
        elif self.case.op_name == "nhwc3to8":
            y = dml.ops.nhwc3to8(x)
        elif self.case.op_name == "ndhwc3to8":
            y = dml.ops.ndhwc3to8(x)
        else:
            raise ValueError(f"Unsupported padding layout helper op {self.case.op_name!r}")
        return dml.ops.output(y, "y")


def trace_padding_layout_helper_spec(case: PaddingLayoutHelperCase):
    return dml.trace(
        _PaddingLayoutHelperModule(case),
        inputs={"x": dml.TensorSpec(list(case.input_shape), case.dtype)},
        name=f"{case.name}_parity",
    )


def random_inputs(case: PaddingLayoutHelperCase, *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    value = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
    if case.dtype == "float16":
        value = value.astype(np.float16)
    return {"x": value}


def numpy_oracle(case: PaddingLayoutHelperCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(inputs["x"])
    padded_channels = 4 if case.op_name == "nhwc3to4" else 8
    output = np.zeros((*x.shape[:-1], padded_channels), dtype=np.float32)
    output[..., :3] = np.asarray(x, dtype=np.float32)
    return _quantize_expected(output, case.dtype)


def _quantize_expected(value: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "float16":
        return np.asarray(value, dtype=np.float16).astype(np.float32)
    if dtype == "float32":
        return np.asarray(value, dtype=np.float32)
    raise ValueError(f"Unsupported padding layout helper dtype {dtype!r}")
