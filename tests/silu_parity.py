from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import dinoml as dml


_BATCH = dml.Dim("batch", min=1, max=4, typical=2, buckets=(2, 4))

ATOL_BY_DTYPE = {"float16": 1e-3, "float32": 5e-6}
RTOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-5}


@dataclass(frozen=True)
class SiluCase:
    name: str
    dtype: str
    input_shape: tuple[int, ...]
    input_spec_shape: tuple[Any, ...] | None = None

    @property
    def resolved_input_spec_shape(self) -> tuple[Any, ...]:
        return self.input_shape if self.input_spec_shape is None else self.input_spec_shape


SILU_CASES = (
    SiluCase(name="silu_rank2_f32", dtype="float32", input_shape=(3, 4)),
    SiluCase(name="silu_rank3_f16", dtype="float16", input_shape=(2, 3, 5)),
    SiluCase(name="silu_dynamic_rank2_f32", dtype="float32", input_shape=(2, 4), input_spec_shape=(_BATCH, 4)),
)


class _SiluModule(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.nn.functional.silu(x), "y")


def trace_silu_spec(case: SiluCase):
    return dml.trace(
        _SiluModule(),
        inputs={"x": dml.TensorSpec(list(case.resolved_input_spec_shape), case.dtype)},
        name=f"{case.name}_parity",
    )


def random_inputs(case: SiluCase, *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    value = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
    if case.dtype == "float16":
        value = value.astype(np.float16)
    return {"x": value}


def torch_oracle(case: SiluCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[case.dtype]
    x = torch.from_numpy(inputs["x"]).to(dtype=torch_dtype)
    return torch.nn.functional.silu(x).float().cpu().numpy()
