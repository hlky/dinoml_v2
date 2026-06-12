from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import dinoml as dml


_ROWS = dml.Dim("rows", min=2, max=5, typical=4, buckets=(4, 5))
_COLS = dml.Dim("cols", min=2, max=4, typical=3, buckets=(3, 4))


@dataclass(frozen=True)
class OneHotCase:
    name: str
    index_dtype: str
    input_shape: tuple[int, ...]
    num_classes: int
    input_spec_shape: tuple[Any, ...] | None = None

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (*self.input_shape, self.num_classes)

    @property
    def resolved_input_spec_shape(self) -> tuple[Any, ...]:
        return self.input_shape if self.input_spec_shape is None else self.input_spec_shape


ONE_HOT_CASES = (
    OneHotCase(
        name="one_hot_rank2_i64",
        index_dtype="int64",
        input_shape=(3, 4),
        num_classes=5,
    ),
    OneHotCase(
        name="one_hot_rank3_i32",
        index_dtype="int32",
        input_shape=(2, 3, 4),
        num_classes=6,
    ),
    OneHotCase(
        name="one_hot_dynamic_rank2_i64",
        index_dtype="int64",
        input_shape=(4, 3),
        num_classes=7,
        input_spec_shape=(_ROWS, _COLS),
    ),
)


class _OneHotModule(dml.Module):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def forward(self, x):
        return dml.ops.output(dml.ops.one_hot(x, self.num_classes), "y")


def trace_one_hot_spec(case: OneHotCase):
    return dml.trace(
        _OneHotModule(case.num_classes),
        inputs={"x": dml.TensorSpec(list(case.resolved_input_spec_shape), case.index_dtype)},
        name=f"{case.name}_parity",
    )


def random_inputs(case: OneHotCase, *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dtype = np.int64 if case.index_dtype == "int64" else np.int32
    values = rng.integers(0, case.num_classes, size=case.input_shape, endpoint=False, dtype=dtype)
    return {"x": values}


def invalid_inputs(case: OneHotCase) -> dict[str, np.ndarray]:
    dtype = np.int64 if case.index_dtype == "int64" else np.int32
    values = np.zeros(case.input_shape, dtype=dtype)
    values.reshape(-1)[0] = case.num_classes
    return {"x": values}


def torch_oracle(case: OneHotCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    x = torch.tensor(inputs["x"], dtype=torch.int64)
    return torch.nn.functional.one_hot(x, num_classes=case.num_classes).cpu().numpy().astype(np.int64, copy=False)
