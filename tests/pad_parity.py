from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import dinoml as dml


@dataclass(frozen=True)
class PadCase:
    name: str
    dtype: str
    input_shape: tuple[int, ...]
    pad: tuple[int, ...]
    value: float | None = None

    @property
    def output_shape(self) -> tuple[int, ...]:
        output_shape = list(self.input_shape)
        for pair_index in range(len(self.pad) // 2):
            axis = len(output_shape) - 1 - pair_index
            output_shape[axis] += int(self.pad[2 * pair_index]) + int(self.pad[2 * pair_index + 1])
        return tuple(output_shape)


PAD_CASES = (
    PadCase(name="pad_rank2_default_value_f32", dtype="float32", input_shape=(3, 4), pad=(1, 2)),
    PadCase(name="pad_rank4_explicit_value_f32", dtype="float32", input_shape=(1, 2, 3, 4), pad=(1, 0, 2, 1), value=-0.75),
)


class _PadModule(dml.Module):
    def __init__(self, case: PadCase):
        self.case = case

    def forward(self, x):
        y = dml.nn.functional.pad(x, self.case.pad, value=self.case.value)
        return dml.ops.output(y, "y")


def trace_pad_spec(case: PadCase):
    return dml.trace(
        _PadModule(case),
        inputs={"x": dml.TensorSpec(list(case.input_shape), case.dtype)},
        name=f"{case.name}_parity",
    )


def random_inputs(case: PadCase, *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    value = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
    return {"x": value}


def torch_oracle(case: PadCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    x = torch.from_numpy(inputs["x"]).to(dtype=torch.float32)
    result = torch.nn.functional.pad(x, case.pad, mode="constant", value=case.value)
    return result.cpu().numpy().astype(np.float32, copy=False)
