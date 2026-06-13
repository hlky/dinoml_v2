from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import dinoml as dml


_BATCH = dml.Dim("batch", min=1, max=4, typical=2, buckets=(2, 4))

ATOL_BY_DTYPE = {"float16": 0.003, "float32": 1e-5}
RTOL_BY_DTYPE = {"float16": 0.002, "float32": 1e-5}


@dataclass(frozen=True)
class LayerNormCase:
    name: str
    dtype: str
    input_shape: tuple[int, ...]
    normalized_shape: tuple[int, ...]
    input_spec_shape: tuple[Any, ...] | None = None

    @property
    def resolved_input_spec_shape(self) -> tuple[Any, ...]:
        return self.input_shape if self.input_spec_shape is None else self.input_spec_shape


LAYER_NORM_CASES = (
    LayerNormCase(name="layer_norm_last_dim_f32", dtype="float32", input_shape=(2, 5, 16), normalized_shape=(16,)),
    LayerNormCase(name="layer_norm_nd_suffix_f16", dtype="float16", input_shape=(2, 3, 4, 8), normalized_shape=(4, 8)),
    LayerNormCase(
        name="layer_norm_dynamic_batch_f32",
        dtype="float32",
        input_shape=(2, 4),
        normalized_shape=(4,),
        input_spec_shape=(_BATCH, 4),
    ),
)


class _LayerNormModule(dml.Module):
    def __init__(self, case: LayerNormCase):
        self.case = case

    def forward(self, x, weight, bias):
        y = dml.nn.functional.layer_norm(
            x,
            self.case.normalized_shape,
            weight=weight,
            bias=bias,
            eps=1e-5,
        )
        return dml.ops.output(y, "y")


def trace_layer_norm_spec(case: LayerNormCase):
    return dml.trace(
        _LayerNormModule(case),
        inputs={
            "x": dml.TensorSpec(list(case.resolved_input_spec_shape), case.dtype),
            "weight": dml.TensorSpec(list(case.normalized_shape), case.dtype),
            "bias": dml.TensorSpec(list(case.normalized_shape), case.dtype),
        },
        name=f"{case.name}_parity",
    )


def random_inputs(case: LayerNormCase, *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs = {
        "x": rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32),
        "weight": rng.standard_normal(case.normalized_shape, dtype=np.float32).astype(np.float32),
        "bias": rng.standard_normal(case.normalized_shape, dtype=np.float32).astype(np.float32),
    }
    if case.dtype == "float16":
        return {name: value.astype(np.float16) for name, value in inputs.items()}
    return inputs


def torch_oracle(case: LayerNormCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[case.dtype]
    x = torch.from_numpy(inputs["x"]).to(dtype=torch_dtype)
    weight = torch.from_numpy(inputs["weight"]).to(dtype=torch_dtype)
    bias = torch.from_numpy(inputs["bias"]).to(dtype=torch_dtype)
    return torch.nn.functional.layer_norm(x, list(case.normalized_shape), weight, bias, eps=1e-5).float().cpu().numpy()
