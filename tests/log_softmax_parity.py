from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import dinoml as dml


_BATCH = dml.Dim("batch", min=1, max=4, typical=2, buckets=(2, 4))

ATOL_BY_DTYPE = {"float32": 1e-5}
RTOL_BY_DTYPE = {"float32": 1e-5}


@dataclass(frozen=True)
class LogSoftmaxCase:
    name: str
    dtype: str
    input_shape: tuple[int, ...]
    dim: int | None = None
    input_spec_shape: tuple[Any, ...] | None = None

    @property
    def resolved_input_spec_shape(self) -> tuple[Any, ...]:
        return self.input_shape if self.input_spec_shape is None else self.input_spec_shape


LOG_SOFTMAX_CASES = (
    LogSoftmaxCase(name="log_softmax_rank2_default_dim_f32", dtype="float32", input_shape=(3, 4)),
    LogSoftmaxCase(name="log_softmax_rank3_last_dim_f32", dtype="float32", input_shape=(2, 3, 5), dim=-1),
    LogSoftmaxCase(
        name="log_softmax_dynamic_rank2_f32",
        dtype="float32",
        input_shape=(2, 4),
        dim=-1,
        input_spec_shape=(_BATCH, 4),
    ),
)


class _LogSoftmaxModule(dml.Module):
    def __init__(self, case: LogSoftmaxCase):
        self.case = case

    def forward(self, x):
        y = dml.nn.functional.log_softmax(x, dim=self.case.dim)
        return dml.ops.output(y, "y")


def trace_log_softmax_spec(case: LogSoftmaxCase):
    return dml.trace(
        _LogSoftmaxModule(case),
        inputs={"x": dml.TensorSpec(list(case.resolved_input_spec_shape), case.dtype)},
        name=f"{case.name}_parity",
    )


def random_inputs(case: LogSoftmaxCase, *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    value = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
    return {"x": value}


def torch_oracle(case: LogSoftmaxCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    x = torch.from_numpy(inputs["x"]).to(dtype=torch.float32)
    result = torch.nn.functional.log_softmax(x, dim=-1 if case.dim is None else case.dim)
    return result.cpu().numpy().astype(np.float32, copy=False)
