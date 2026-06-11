from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import dinoml as dml
from dinoml.ir import array_from_storage, array_to_storage


ATOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6, "bfloat16": 2e-2}
RTOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6, "bfloat16": 2e-2}


@dataclass(frozen=True)
class MaskedSelectCase:
    name: str
    dtype: str
    input_spec_shape: tuple[Any, ...]
    mask_spec_shape: tuple[Any, ...]
    input_shape: tuple[int, ...]
    mask_shape: tuple[int, ...]
    zero_mask: bool = False


_ROWS = dml.Dim("rows", min=1, max=4, typical=2, buckets=(2, 4))
_WIDTH = dml.Dim("width", min=1, max=8, typical=3, buckets=(3, 8))

MASKED_SELECT_CASES = (
    MaskedSelectCase(
        name="masked_select_same_shape_f16",
        dtype="float16",
        input_spec_shape=(2, 6),
        mask_spec_shape=(2, 6),
        input_shape=(2, 6),
        mask_shape=(2, 6),
    ),
    MaskedSelectCase(
        name="masked_select_broadcast_f32",
        dtype="float32",
        input_spec_shape=(2, 1, 3),
        mask_spec_shape=(1, 4, 3),
        input_shape=(2, 1, 3),
        mask_shape=(1, 4, 3),
    ),
    MaskedSelectCase(
        name="masked_select_dynamic_bf16",
        dtype="bfloat16",
        input_spec_shape=(_ROWS, 1, _WIDTH),
        mask_spec_shape=(1, 4, _WIDTH),
        input_shape=(2, 1, 3),
        mask_shape=(1, 4, 3),
    ),
    MaskedSelectCase(
        name="masked_select_zero_mask_f32",
        dtype="float32",
        input_spec_shape=(2, 6),
        mask_spec_shape=(2, 6),
        input_shape=(2, 6),
        mask_shape=(2, 6),
        zero_mask=True,
    ),
)


class _MaskedSelectModule(dml.Module):
    def forward(self, x, mask):
        return dml.ops.output(dml.ops.masked_select(x, mask), "y")


def trace_masked_select_spec(case: MaskedSelectCase):
    return dml.trace(
        _MaskedSelectModule(),
        inputs={
            "x": dml.TensorSpec(list(case.input_spec_shape), case.dtype),
            "mask": dml.TensorSpec(list(case.mask_spec_shape), "bool"),
        },
        name=f"{case.name}_parity",
    )


def random_inputs(case: MaskedSelectCase, *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    value = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
    if case.dtype == "float16":
        value = value.astype(np.float16)
    elif case.dtype == "bfloat16":
        value = array_from_storage(array_to_storage(value, "bfloat16"), "bfloat16")
    mask = np.zeros(case.mask_shape, dtype=np.bool_) if case.zero_mask else (rng.random(case.mask_shape) > 0.45)
    return {"x": value, "mask": mask}


def numpy_oracle(case: MaskedSelectCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    source, mask = np.broadcast_arrays(np.asarray(inputs["x"]), np.asarray(inputs["mask"], dtype=np.bool_))
    expected = np.array(source[mask], copy=True)
    if case.dtype == "float16":
        return expected.astype(np.float16).astype(np.float32)
    if case.dtype == "bfloat16":
        return array_from_storage(array_to_storage(expected, "bfloat16"), "bfloat16")
    return expected.astype(np.float32, copy=False)
