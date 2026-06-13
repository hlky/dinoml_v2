from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import dinoml as dml
from dinoml.ir import array_from_storage, array_to_storage


ATOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6, "bfloat16": 2e-2}
RTOL_BY_DTYPE = {"float16": 1e-3, "float32": 1e-6, "bfloat16": 2e-2}


@dataclass(frozen=True)
class ScatterCase:
    name: str
    op: str
    dtype: str
    index_dtype: str
    input_shape: tuple[int, ...]
    index_shape: tuple[int, ...]
    index_values: tuple[int, ...]
    dim: int
    reduce: str | None = None
    include_self: bool = True

    @property
    def source_shape(self) -> tuple[int, ...]:
        return self.index_shape


SCATTER_CASES = (
    ScatterCase(
        name="scatter_dim1_last_write_f32",
        op="scatter",
        dtype="float32",
        index_dtype="int64",
        input_shape=(2, 4, 3),
        index_shape=(2, 2, 3),
        index_values=(0, 3, 1, 2, 1, 2, 1, 0, 3, 2, 0, 1),
        dim=1,
    ),
    ScatterCase(
        name="scatter_bool_dim0_int32",
        op="scatter",
        dtype="bool",
        index_dtype="int32",
        input_shape=(3, 2),
        index_shape=(2, 2),
        index_values=(2, 1, 0, 2),
        dim=0,
    ),
    ScatterCase(
        name="scatter_add_dim1_f16",
        op="scatter_add",
        dtype="float16",
        index_dtype="int32",
        input_shape=(2, 4, 3),
        index_shape=(2, 3, 3),
        index_values=(0, 3, 1, 2, 1, 2, 1, 0, 3, 3, 1, 0, 2, 0, 2, 1, 1, 3),
        dim=1,
    ),
    ScatterCase(
        name="scatter_reduce_sum_dim2_bf16",
        op="scatter_reduce",
        dtype="bfloat16",
        index_dtype="int64",
        input_shape=(2, 3, 4),
        index_shape=(2, 3, 3),
        index_values=(0, 1, 3, 2, 2, 0, 1, 3, 1, 2, 0, 1, 3, 1, 2, 0, 2, 3),
        dim=2,
        reduce="sum",
    ),
    ScatterCase(
        name="scatter_reduce_prod_dim1_f32",
        op="scatter_reduce",
        dtype="float32",
        index_dtype="int64",
        input_shape=(2, 4, 2),
        index_shape=(2, 2, 2),
        index_values=(0, 3, 2, 1, 1, 0, 3, 2),
        dim=1,
        reduce="prod",
    ),
    ScatterCase(
        name="scatter_reduce_amax_dim1_f16",
        op="scatter_reduce",
        dtype="float16",
        index_dtype="int32",
        input_shape=(2, 4, 2),
        index_shape=(2, 3, 2),
        index_values=(0, 1, 3, 1, 2, 0, 1, 0, 3, 2, 1, 3),
        dim=1,
        reduce="amax",
    ),
    ScatterCase(
        name="scatter_reduce_amin_dim0_f32",
        op="scatter_reduce",
        dtype="float32",
        index_dtype="int64",
        input_shape=(4, 3),
        index_shape=(3, 3),
        index_values=(0, 1, 3, 2, 1, 0, 1, 3, 2),
        dim=0,
        reduce="amin",
    ),
)


class _ScatterModule(dml.Module):
    def __init__(self, case: ScatterCase):
        self.case = case

    def forward(self, x, index, source):
        if self.case.op == "scatter":
            y = x.scatter(self.case.dim, index, source)
        elif self.case.op == "scatter_add":
            y = x.scatter_add(self.case.dim, index, source)
        elif self.case.op == "scatter_reduce":
            y = x.scatter_reduce(
                self.case.dim,
                index,
                source,
                reduce=self.case.reduce,
                include_self=self.case.include_self,
            )
        else:
            raise ValueError(f"Unsupported scatter case op {self.case.op!r}")
        return dml.ops.output(y, "y")


def trace_scatter_spec(case: ScatterCase):
    return dml.trace(
        _ScatterModule(case),
        inputs={
            "x": dml.TensorSpec(list(case.input_shape), case.dtype),
            "index": dml.TensorSpec(list(case.index_shape), case.index_dtype),
            "source": dml.TensorSpec(list(case.source_shape), case.dtype),
        },
        name=f"{case.name}_parity",
    )


def random_inputs(case: ScatterCase, *, seed: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if case.dtype == "bool":
        x = rng.integers(0, 2, size=case.input_shape, dtype=np.int32).astype(np.bool_)
        source = rng.integers(0, 2, size=case.source_shape, dtype=np.int32).astype(np.bool_)
    else:
        x = rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32, copy=False)
        source = rng.standard_normal(case.source_shape, dtype=np.float32).astype(np.float32, copy=False)
        if case.dtype == "float16":
            x = x.astype(np.float16)
            source = source.astype(np.float16)
        elif case.dtype == "bfloat16":
            x = array_from_storage(array_to_storage(x, "bfloat16"), "bfloat16")
            source = array_from_storage(array_to_storage(source, "bfloat16"), "bfloat16")
    index = np.asarray(case.index_values, dtype=np.int64 if case.index_dtype == "int64" else np.int32).reshape(case.index_shape)
    return {"x": x, "index": index, "source": source}


def torch_oracle(case: ScatterCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    dtype = {
        "bool": torch.bool,
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[case.dtype]
    index_dtype = {"int64": torch.int64, "int32": torch.int32}[case.index_dtype]
    x = torch.tensor(inputs["x"], dtype=dtype)
    index = torch.tensor(inputs["index"], dtype=index_dtype)
    source = torch.tensor(inputs["source"], dtype=dtype)
    if case.op == "scatter":
        result = x.scatter(case.dim, index, source)
    elif case.op == "scatter_add":
        result = x.scatter_add(case.dim, index, source)
    elif case.op == "scatter_reduce":
        result = x.scatter_reduce(case.dim, index, source, reduce=case.reduce, include_self=case.include_self)
    else:
        raise ValueError(f"Unsupported scatter case op {case.op!r}")
    if case.dtype == "bool":
        return result.cpu().numpy().astype(np.bool_)
    expected = result.to(torch.float32).cpu().numpy()
    if case.dtype == "float16":
        return expected.astype(np.float16).astype(np.float32)
    if case.dtype == "bfloat16":
        return array_from_storage(array_to_storage(expected, "bfloat16"), "bfloat16")
    return expected.astype(np.float32, copy=False)
