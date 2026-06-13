from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml
from dinoml.ir import array_from_storage, array_to_storage


ATOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6, "bfloat16": 2e-2, "bool": 0.0}
RTOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6, "bfloat16": 2e-2, "bool": 0.0}


@dataclass(frozen=True)
class ModeCase:
    name: str
    dtype: str
    shape: tuple[int, ...]
    values: tuple[float | bool, ...]
    keepdim: bool = False


MODE_CASES = (
    ModeCase(
        name="mode_float32_rank1",
        dtype="float32",
        shape=(6,),
        values=(3.0, 3.0, 2.0, 2.0, 2.0, 3.0),
    ),
    ModeCase(
        name="mode_float16_rank2_keepdim",
        dtype="float16",
        shape=(3, 4),
        values=(2.0, 1.0, 1.0, 2.0, 4.0, 4.0, 3.0, 3.0, 6.0, 5.0, 5.0, 6.0),
        keepdim=True,
    ),
    ModeCase(
        name="mode_bfloat16_rank3",
        dtype="bfloat16",
        shape=(2, 2, 5),
        values=(
            1.0, 2.0, 1.0, 3.0, 1.0,
            4.0, 4.0, 2.0, 2.0, 2.0,
            5.0, 5.0, 6.0, 6.0, 5.0,
            7.0, 8.0, 7.0, 8.0, 7.0,
        ),
    ),
    ModeCase(
        name="mode_bool_rank2",
        dtype="bool",
        shape=(3, 5),
        values=(
            True, False, False, True, False,
            True, True, False, True, False,
            False, False, True, False, True,
        ),
    ),
)


class _ModeModule(dml.Module):
    def __init__(self, case: ModeCase):
        self.case = case

    def forward(self, x):
        values, indices = x.mode(dim=-1, keepdim=self.case.keepdim)
        return {
            "values": dml.ops.output(values, "values"),
            "indices": dml.ops.output(indices, "indices"),
        }


def trace_mode_spec(case: ModeCase):
    return dml.trace(
        _ModeModule(case),
        inputs={"x": dml.TensorSpec(list(case.shape), case.dtype)},
        name=f"{case.name}_parity",
    )


def case_inputs(case: ModeCase) -> dict[str, np.ndarray]:
    if case.dtype == "bool":
        x = np.asarray(case.values, dtype=np.bool_).reshape(case.shape)
    else:
        x = np.asarray(case.values, dtype=np.float32).reshape(case.shape)
        if case.dtype == "float16":
            x = x.astype(np.float16)
        elif case.dtype == "bfloat16":
            x = array_from_storage(array_to_storage(x, "bfloat16"), "bfloat16")
    return {"x": x}


def torch_oracle(case: ModeCase, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    torch = __import__("torch")
    dtype = {
        "bool": torch.bool,
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[case.dtype]
    x = torch.tensor(inputs["x"], dtype=dtype)
    values, indices = torch.mode(x, dim=-1, keepdim=case.keepdim)
    if case.dtype == "bool":
        values_np = values.cpu().numpy().astype(np.bool_)
    else:
        values_np = values.to(torch.float32).cpu().numpy()
        if case.dtype == "float16":
            values_np = values_np.astype(np.float16).astype(np.float32)
        elif case.dtype == "bfloat16":
            values_np = array_from_storage(array_to_storage(values_np, "bfloat16"), "bfloat16")
        else:
            values_np = values_np.astype(np.float32, copy=False)
    return values_np, indices.cpu().numpy().astype(np.int64, copy=False)
