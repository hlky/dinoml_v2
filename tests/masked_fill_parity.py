from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import dinoml as dml
from dinoml.ir import array_from_storage, array_to_storage


ATOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6, "bfloat16": 2e-2}
RTOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6, "bfloat16": 2e-2}


@dataclass(frozen=True)
class MaskedFillCase:
    name: str
    dtype: str
    input_shape: tuple[int, ...]
    mask_shape: tuple[int, ...]
    fill_value: bool | float


MASKED_FILL_CASES = (
    MaskedFillCase(
        name="masked_fill_exact_f32",
        dtype="float32",
        input_shape=(2, 3),
        mask_shape=(2, 3),
        fill_value=-0.5,
    ),
    MaskedFillCase(
        name="masked_fill_broadcast_bf16",
        dtype="bfloat16",
        input_shape=(2, 3, 4),
        mask_shape=(1, 3, 1),
        fill_value=1.25,
    ),
)


class _MaskedFillModule(dml.Module):
    def __init__(self, case: MaskedFillCase):
        self.case = case

    def forward(self, x, mask):
        return dml.ops.output(x.masked_fill(mask, self.case.fill_value), "y")


def trace_masked_fill_spec(case: MaskedFillCase):
    return dml.trace(
        _MaskedFillModule(case),
        inputs={
            "x": dml.TensorSpec(list(case.input_shape), case.dtype),
            "mask": dml.TensorSpec(list(case.mask_shape), "bool"),
        },
        name=f"{case.name}_parity",
    )


def random_inputs(case: MaskedFillCase, *, seed: int = 13) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if case.dtype == "bool":
        x = rng.integers(0, 2, size=case.input_shape, dtype=np.int32).astype(np.bool_)
    else:
        x = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
        if case.dtype == "float16":
            x = x.astype(np.float16)
        elif case.dtype == "bfloat16":
            x = array_from_storage(array_to_storage(x, "bfloat16"), "bfloat16")
    mask = rng.integers(0, 2, size=case.mask_shape, dtype=np.int32).astype(np.bool_)
    return {"x": x, "mask": mask}


def torch_oracle(case: MaskedFillCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    dtype = {"bool": torch.bool, "float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[case.dtype]
    x = torch.tensor(inputs["x"], dtype=dtype)
    mask = torch.tensor(inputs["mask"], dtype=torch.bool)
    result = x.masked_fill(mask, case.fill_value)
    if case.dtype == "bool":
        return result.cpu().numpy().astype(np.bool_)
    expected = result.to(torch.float32).cpu().numpy()
    if case.dtype == "float16":
        return expected.astype(np.float16).astype(np.float32)
    if case.dtype == "bfloat16":
        return array_from_storage(array_to_storage(expected, "bfloat16"), "bfloat16")
    return expected.astype(np.float32, copy=False)
