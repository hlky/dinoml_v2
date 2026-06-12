from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import dinoml as dml
from dinoml.ir import array_from_storage, array_to_storage


ATOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6, "bfloat16": 2e-2}
RTOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6, "bfloat16": 2e-2}


_ROWS = dml.Dim("rows", min=2, max=5, typical=4, buckets=(4, 5))
_COLS = dml.Dim("cols", min=2, max=4, typical=3, buckets=(3, 4))
_INDEX_LEN = dml.Dim("index_len", min=2, max=6, typical=4, buckets=(4, 6))
_DEPTH = dml.Dim("depth", min=2, max=5, typical=3, buckets=(3, 5))


@dataclass(frozen=True)
class IndexAddCase:
    name: str
    dtype: str
    index_dtype: str
    input_shape: tuple[int, ...]
    index_values: tuple[int, ...]
    dim: int
    input_spec_shape: tuple[Any, ...] | None = None
    index_spec_shape: tuple[Any, ...] | None = None
    source_spec_shape: tuple[Any, ...] | None = None

    @property
    def index_shape(self) -> tuple[int, ...]:
        return (len(self.index_values),)

    @property
    def source_shape(self) -> tuple[int, ...]:
        shape = list(self.input_shape)
        axis = self.normalized_dim
        shape[axis] = len(self.index_values)
        return tuple(shape)

    @property
    def normalized_dim(self) -> int:
        return self.dim if self.dim >= 0 else self.dim + len(self.input_shape)

    @property
    def resolved_input_spec_shape(self) -> tuple[Any, ...]:
        return self.input_shape if self.input_spec_shape is None else self.input_spec_shape

    @property
    def resolved_index_spec_shape(self) -> tuple[Any, ...]:
        return self.index_shape if self.index_spec_shape is None else self.index_spec_shape

    @property
    def resolved_source_spec_shape(self) -> tuple[Any, ...]:
        if self.source_spec_shape is not None:
            return self.source_spec_shape
        shape = list(self.resolved_input_spec_shape)
        shape[self.normalized_dim] = self.resolved_index_spec_shape[0]
        return tuple(shape)


INDEX_ADD_CASES = (
    IndexAddCase(
        name="index_add_dim0_f32",
        dtype="float32",
        index_dtype="int64",
        input_shape=(4, 3),
        index_values=(2, 1, 2, 0, 1),
        dim=0,
    ),
    IndexAddCase(
        name="index_add_dim1_f16_int32",
        dtype="float16",
        index_dtype="int32",
        input_shape=(2, 4, 3),
        index_values=(3, 1, 3, 0, 1),
        dim=-2,
    ),
    IndexAddCase(
        name="index_add_dim2_bf16",
        dtype="bfloat16",
        index_dtype="int64",
        input_shape=(2, 3, 4),
        index_values=(1, 3, 1, 0, 2, 3),
        dim=2,
    ),
    IndexAddCase(
        name="index_add_dynamic_dim0_f32",
        dtype="float32",
        index_dtype="int64",
        input_shape=(4, 3),
        index_values=(3, 1, 3, 0),
        dim=0,
        input_spec_shape=(_ROWS, _COLS),
        index_spec_shape=(_INDEX_LEN,),
        source_spec_shape=(_INDEX_LEN, _COLS),
    ),
    IndexAddCase(
        name="index_add_dynamic_dim1_bf16",
        dtype="bfloat16",
        index_dtype="int32",
        input_shape=(2, 4, 3),
        index_values=(3, 1, 3, 0),
        dim=1,
        input_spec_shape=(2, _ROWS, _DEPTH),
        index_spec_shape=(_INDEX_LEN,),
        source_spec_shape=(2, _INDEX_LEN, _DEPTH),
    ),
)


class _IndexAddModule(dml.Module):
    def __init__(self, case: IndexAddCase):
        self.case = case

    def forward(self, x, index, source):
        return dml.ops.output(dml.ops.index_add(x, self.case.dim, index, source), "y")


def trace_index_add_spec(case: IndexAddCase):
    return dml.trace(
        _IndexAddModule(case),
        inputs={
            "x": dml.TensorSpec(list(case.resolved_input_spec_shape), case.dtype),
            "index": dml.TensorSpec(list(case.resolved_index_spec_shape), case.index_dtype),
            "source": dml.TensorSpec(list(case.resolved_source_spec_shape), case.dtype),
        },
        name=f"{case.name}_parity",
    )


def random_inputs(case: IndexAddCase, *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
    source = rng.standard_normal(case.source_shape, dtype=np.float32).astype(np.float32, copy=False)
    if case.dtype == "float16":
        x = x.astype(np.float16)
        source = source.astype(np.float16)
    elif case.dtype == "bfloat16":
        x = array_from_storage(array_to_storage(x, "bfloat16"), "bfloat16")
        source = array_from_storage(array_to_storage(source, "bfloat16"), "bfloat16")
    index = np.asarray(case.index_values, dtype=np.int64 if case.index_dtype == "int64" else np.int32)
    return {"x": x, "index": index, "source": source}


def torch_oracle(case: IndexAddCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[case.dtype]
    index_dtype = {"int64": torch.int64, "int32": torch.int32}[case.index_dtype]
    x = torch.tensor(inputs["x"], dtype=dtype)
    index = torch.tensor(inputs["index"], dtype=index_dtype)
    source = torch.tensor(inputs["source"], dtype=dtype)
    result = torch.index_add(x, case.dim, index, source)
    expected = result.to(torch.float32).cpu().numpy()
    if case.dtype == "float16":
        return expected.astype(np.float16).astype(np.float32)
    if case.dtype == "bfloat16":
        return array_from_storage(array_to_storage(expected, "bfloat16"), "bfloat16")
    return expected.astype(np.float32, copy=False)
