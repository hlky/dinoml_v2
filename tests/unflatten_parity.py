from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import dinoml as dml
from dinoml.shapes import symbolic_int_expr


OUTER = dml.Dim("outer", min=2, max=4, typical=3, buckets=(2, 4))
OUTER_SPEC = OUTER.to_json()


@dataclass(frozen=True)
class UnflattenCase:
    name: str
    input_spec_shape: tuple[Any, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    dim: int
    sizes_fn: Callable[[Any], tuple[Any, ...]]
    torch_sizes_fn: Callable[[Any], tuple[Any, ...]]


class _UnflattenModule(dml.Module):
    def __init__(self, case: UnflattenCase):
        self.case = case

    def forward(self, x):
        return dml.ops.output(dml.ops.unflatten(x, self.case.dim, self.case.sizes_fn(x)), "y")


UNFLATTEN_CASES = (
    UnflattenCase(
            name="unflatten_dynamic_explicit_sizes",
            input_spec_shape=(OUTER, symbolic_int_expr("mul", OUTER_SPEC, 3)),
            input_shapes=((2, 6), (4, 12)),
            dim=1,
            sizes_fn=lambda x: (dml.ops.size(x, 0), 3),
            torch_sizes_fn=lambda x: (x.shape[0], 3),
        ),
        UnflattenCase(
            name="unflatten_dynamic_inferred_size",
            input_spec_shape=(OUTER, symbolic_int_expr("mul", 2, OUTER_SPEC)),
            input_shapes=((2, 4), (4, 8)),
            dim=1,
            sizes_fn=lambda _x: (2, -1),
            torch_sizes_fn=lambda _x: (2, -1),
        ),
)


def trace_unflatten_spec(case: UnflattenCase):
    return dml.trace(
        _UnflattenModule(case),
        inputs={"x": dml.TensorSpec(list(case.input_spec_shape), "float32")},
        name=case.name,
    )


def case_inputs(case: UnflattenCase) -> tuple[dict[str, np.ndarray], ...]:
    inputs = []
    for shape in case.input_shapes:
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / np.float32(10.0)
        inputs.append({"x": values})
    return tuple(inputs)


def torch_oracle(case: UnflattenCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    x = torch.from_numpy(inputs["x"].copy())
    expected = x.unflatten(case.dim, case.torch_sizes_fn(x))
    return expected.cpu().numpy().astype(np.float32, copy=False)
