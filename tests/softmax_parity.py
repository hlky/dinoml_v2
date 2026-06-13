from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml


@dataclass(frozen=True)
class SoftmaxCase:
    name: str
    dtype: str
    input_shape: tuple[int, ...]
    dim: int | None = None


SOFTMAX_CASES = (
    SoftmaxCase(name="softmax_rank2_default_dim_f32", dtype="float32", input_shape=(3, 4)),
    SoftmaxCase(name="softmax_rank3_explicit_last_dim_f32", dtype="float32", input_shape=(2, 3, 5), dim=-1),
)


class _SoftmaxModule(dml.Module):
    def __init__(self, case: SoftmaxCase):
        self.case = case

    def forward(self, x):
        y = dml.nn.functional.softmax(x, dim=self.case.dim)
        return dml.ops.output(y, "y")


def trace_softmax_spec(case: SoftmaxCase):
    return dml.trace(
        _SoftmaxModule(case),
        inputs={"x": dml.TensorSpec(list(case.input_shape), case.dtype)},
        name=f"{case.name}_parity",
    )


def random_inputs(case: SoftmaxCase, *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    value = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
    return {"x": value}


def torch_oracle(case: SoftmaxCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    x = torch.from_numpy(inputs["x"]).to(dtype=torch.float32)
    result = torch.nn.functional.softmax(x, dim=-1 if case.dim is None else case.dim)
    return result.cpu().numpy().astype(np.float32, copy=False)
