from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import dinoml as dml


_BATCH = dml.Dim("batch", min=1, max=4, typical=2, buckets=(2, 4))

ATOL_BY_DTYPE = {"float16": 0.003, "float32": 1e-5}
RTOL_BY_DTYPE = {"float16": 0.002, "float32": 1e-5}


@dataclass(frozen=True)
class FunctionalGroupNormCase:
    name: str
    dtype: str
    input_shape: tuple[int, ...]
    num_groups: int
    use_affine: bool
    input_spec_shape: tuple[Any, ...] | None = None

    @property
    def resolved_input_spec_shape(self) -> tuple[Any, ...]:
        return self.input_shape if self.input_spec_shape is None else self.input_spec_shape

    @property
    def channels(self) -> int:
        return int(self.input_shape[-1])


FUNCTIONAL_GROUP_NORM_CASES = (
    FunctionalGroupNormCase(
        name="functional_group_norm_affine_f32",
        dtype="float32",
        input_shape=(2, 3, 2, 4),
        num_groups=2,
        use_affine=True,
    ),
    FunctionalGroupNormCase(
        name="functional_group_norm_default_affine_f16",
        dtype="float16",
        input_shape=(1, 2, 2, 8),
        num_groups=4,
        use_affine=False,
    ),
    FunctionalGroupNormCase(
        name="functional_group_norm_dynamic_batch_f32",
        dtype="float32",
        input_shape=(2, 3, 2, 4),
        num_groups=2,
        use_affine=True,
        input_spec_shape=(_BATCH, 3, 2, 4),
    ),
)


class _FunctionalGroupNormModule(dml.Module):
    def __init__(self, case: FunctionalGroupNormCase):
        self.case = case

    def forward(self, x, weight=None, bias=None):
        y = dml.nn.functional.group_norm(x, self.case.num_groups, weight=weight, bias=bias, eps=1e-5)
        return dml.ops.output(y, "y")


def trace_functional_group_norm_spec(case: FunctionalGroupNormCase):
    inputs: dict[str, Any] = {"x": dml.TensorSpec(list(case.resolved_input_spec_shape), case.dtype)}
    if case.use_affine:
        inputs["weight"] = dml.TensorSpec([case.channels], case.dtype)
        inputs["bias"] = dml.TensorSpec([case.channels], case.dtype)
    return dml.trace(_FunctionalGroupNormModule(case), inputs=inputs, name=f"{case.name}_parity")


def random_inputs(case: FunctionalGroupNormCase, *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs: dict[str, np.ndarray] = {
        "x": rng.standard_normal(case.input_shape, dtype=np.float32).astype(np.float32),
    }
    if case.use_affine:
        inputs["weight"] = rng.standard_normal([case.channels], dtype=np.float32).astype(np.float32)
        inputs["bias"] = rng.standard_normal([case.channels], dtype=np.float32).astype(np.float32)
    if case.dtype == "float16":
        return {name: value.astype(np.float16) for name, value in inputs.items()}
    return inputs


def torch_oracle(case: FunctionalGroupNormCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[case.dtype]
    x = torch.from_numpy(inputs["x"]).to(dtype=torch_dtype)
    x_nchw = x.permute(0, x.ndim - 1, *range(1, x.ndim - 1)).contiguous()
    weight = None
    bias = None
    if case.use_affine:
        weight = torch.from_numpy(inputs["weight"]).to(dtype=torch_dtype)
        bias = torch.from_numpy(inputs["bias"]).to(dtype=torch_dtype)
    result = torch.nn.functional.group_norm(x_nchw, case.num_groups, weight, bias, eps=1e-5)
    return result.permute(0, *range(2, x.ndim), 1).contiguous().float().cpu().numpy()
