from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml
from dinoml.ir import array_from_storage, array_to_storage


ATOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6, "bfloat16": 2e-2}
RTOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6, "bfloat16": 2e-2}


@dataclass(frozen=True)
class NormalizeCase:
    name: str
    dtype: str
    shape: tuple[int, ...]
    eps: float


NORMALIZE_CASES = (
    NormalizeCase(name="normalize_float32_rank2", dtype="float32", shape=(3, 5), eps=1e-12),
    NormalizeCase(name="normalize_float16_rank3", dtype="float16", shape=(2, 3, 4), eps=1e-4),
    NormalizeCase(name="normalize_bfloat16_rank3", dtype="bfloat16", shape=(2, 2, 5), eps=1e-3),
)


class _NormalizeModule(dml.Module):
    def __init__(self, case: NormalizeCase):
        self.case = case

    def forward(self, x):
        y = dml.nn.functional.normalize(x, p=2.0, dim=-1, eps=self.case.eps)
        return dml.ops.output(y, "y")


def trace_normalize_spec(case: NormalizeCase):
    return dml.trace(
        _NormalizeModule(case),
        inputs={"x": dml.TensorSpec(list(case.shape), case.dtype)},
        name=f"{case.name}_parity",
    )


def random_inputs(case: NormalizeCase, *, seed: int = 23) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(case.shape, dtype=np.float32).astype(np.float32, copy=False)
    x.reshape(-1)[0] = 0.0
    if case.dtype == "float16":
        x = x.astype(np.float16)
    elif case.dtype == "bfloat16":
        x = array_from_storage(array_to_storage(x, "bfloat16"), "bfloat16")
    return {"x": x}


def torch_oracle(case: NormalizeCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[case.dtype]
    x = torch.tensor(inputs["x"], dtype=dtype)
    y = torch.nn.functional.normalize(x, p=2.0, dim=-1, eps=case.eps)
    result = y.to(torch.float32).cpu().numpy()
    if case.dtype == "float16":
        return result.astype(np.float16).astype(np.float32)
    if case.dtype == "bfloat16":
        return array_from_storage(array_to_storage(result, "bfloat16"), "bfloat16")
    return result.astype(np.float32, copy=False)
