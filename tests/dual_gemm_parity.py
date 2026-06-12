from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import dinoml as dml
from dinoml.ir import array_from_storage, array_to_storage


ATOL_BY_DTYPE = {"float16": 3e-3, "float32": 1e-6, "bfloat16": 2e-2}
RTOL_BY_DTYPE = {"float16": 2e-3, "float32": 1e-6, "bfloat16": 2e-2}


_BATCH = dml.Dim("batch", min=1, max=3, typical=2, buckets=(2, 3))
_TOKENS = dml.Dim("tokens", min=2, max=5, typical=4, buckets=(4, 5))
_N = dml.Dim("n", min=2, max=8, typical=6, buckets=(6, 8))
_K = dml.Dim("k", min=4, max=8, typical=8, buckets=(8,))


@dataclass(frozen=True)
class DualGemmCase:
    name: str
    op_name: str
    dtype: str
    a_shape: tuple[int, ...]
    b0_shape: tuple[int, int]
    b1_shape: tuple[int, int]
    a_spec_shape: tuple[Any, ...] | None = None
    b0_spec_shape: tuple[Any, ...] | None = None
    b1_spec_shape: tuple[Any, ...] | None = None

    @property
    def has_bias(self) -> bool:
        return self.op_name == "dual_gemm_rcr_bias_fast_gelu"

    @property
    def bias0_shape(self) -> tuple[int, ...]:
        return (self.b0_shape[0],)

    @property
    def bias1_shape(self) -> tuple[int, ...]:
        return (self.b1_shape[0],)

    @property
    def resolved_a_spec_shape(self) -> tuple[Any, ...]:
        return self.a_shape if self.a_spec_shape is None else self.a_spec_shape

    @property
    def resolved_b0_spec_shape(self) -> tuple[Any, ...]:
        return self.b0_shape if self.b0_spec_shape is None else self.b0_spec_shape

    @property
    def resolved_b1_spec_shape(self) -> tuple[Any, ...]:
        return self.b1_shape if self.b1_spec_shape is None else self.b1_spec_shape

    @property
    def resolved_bias0_spec_shape(self) -> tuple[Any, ...]:
        return (self.resolved_b0_spec_shape[0],)

    @property
    def resolved_bias1_spec_shape(self) -> tuple[Any, ...]:
        return (self.resolved_b1_spec_shape[0],)


DUAL_GEMM_CASES = (
    DualGemmCase(
        name="dual_gemm_silu_f32",
        op_name="dual_gemm_rcr_silu",
        dtype="float32",
        a_shape=(2, 3, 4),
        b0_shape=(5, 4),
        b1_shape=(5, 4),
    ),
    DualGemmCase(
        name="dual_gemm_fast_gelu_f16_broadcast_dynamic",
        op_name="dual_gemm_rcr_fast_gelu",
        dtype="float16",
        a_shape=(2, 4, 8),
        b0_shape=(6, 8),
        b1_shape=(1, 8),
        a_spec_shape=(_BATCH, _TOKENS, _K),
        b0_spec_shape=(_N, _K),
        b1_spec_shape=(1, _K),
    ),
    DualGemmCase(
        name="dual_gemm_bias_fast_gelu_bf16_dynamic",
        op_name="dual_gemm_rcr_bias_fast_gelu",
        dtype="bfloat16",
        a_shape=(2, 4, 8),
        b0_shape=(6, 8),
        b1_shape=(6, 8),
        a_spec_shape=(_BATCH, _TOKENS, _K),
        b0_spec_shape=(_N, _K),
        b1_spec_shape=(_N, _K),
    ),
    DualGemmCase(
        name="dual_gemm_bias_fast_gelu_f32_broadcast",
        op_name="dual_gemm_rcr_bias_fast_gelu",
        dtype="float32",
        a_shape=(2, 3, 8),
        b0_shape=(4, 8),
        b1_shape=(1, 8),
    ),
)


class _DualGemmModule(dml.Module):
    def __init__(self, case: DualGemmCase):
        self.case = case

    def forward(self, a, b0, b1, bias0=None, bias1=None):
        op = getattr(dml.ops, self.case.op_name)
        args = [a, b0, b1]
        if self.case.has_bias:
            args.extend([bias0, bias1])
        return dml.ops.output(op(*args), "y")


def trace_dual_gemm_spec(case: DualGemmCase):
    inputs = {
        "a": dml.TensorSpec(list(case.resolved_a_spec_shape), case.dtype),
        "b0": dml.TensorSpec(list(case.resolved_b0_spec_shape), case.dtype),
        "b1": dml.TensorSpec(list(case.resolved_b1_spec_shape), case.dtype),
    }
    if case.has_bias:
        inputs["bias0"] = dml.TensorSpec(list(case.resolved_bias0_spec_shape), case.dtype)
        inputs["bias1"] = dml.TensorSpec(list(case.resolved_bias1_spec_shape), case.dtype)
    return dml.trace(_DualGemmModule(case), inputs=inputs, name=f"{case.name}_parity")


def random_inputs(case: DualGemmCase, *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs = {
        "a": rng.standard_normal(case.a_shape, dtype=np.float32).astype(np.float32, copy=False),
        "b0": rng.standard_normal(case.b0_shape, dtype=np.float32).astype(np.float32, copy=False),
        "b1": rng.standard_normal(case.b1_shape, dtype=np.float32).astype(np.float32, copy=False),
    }
    if case.has_bias:
        inputs["bias0"] = rng.standard_normal(case.bias0_shape, dtype=np.float32).astype(np.float32, copy=False)
        inputs["bias1"] = rng.standard_normal(case.bias1_shape, dtype=np.float32).astype(np.float32, copy=False)
    if case.dtype == "float16":
        return {name: value.astype(np.float16) for name, value in inputs.items()}
    if case.dtype == "bfloat16":
        return {name: array_from_storage(array_to_storage(value, "bfloat16"), "bfloat16") for name, value in inputs.items()}
    return inputs


def numpy_oracle(case: DualGemmCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    a = np.asarray(inputs["a"], dtype=np.float32)
    b0 = np.asarray(inputs["b0"], dtype=np.float32)
    b1 = np.asarray(inputs["b1"], dtype=np.float32)
    left = np.matmul(a, np.swapaxes(b0, -1, -2))
    right = np.matmul(a, np.swapaxes(b1, -1, -2))
    if case.has_bias:
        left = left + np.reshape(np.asarray(inputs["bias0"], dtype=np.float32), [1] * (left.ndim - 1) + [-1])
        right = right + np.reshape(np.asarray(inputs["bias1"], dtype=np.float32), [1] * (right.ndim - 1) + [-1])
    if case.op_name == "dual_gemm_rcr_silu":
        left = left / (1.0 + np.exp(-left))
    elif case.op_name in {"dual_gemm_rcr_fast_gelu", "dual_gemm_rcr_bias_fast_gelu"}:
        left = 0.5 * left * (1.0 + np.tanh(0.7978845608 * left * (1.0 + 0.044715 * left * left)))
    else:
        raise ValueError(f"Unsupported dual GEMM test op {case.op_name!r}")
    return _cast_expected(case.dtype, left * right)


def _cast_expected(dtype: str, value: np.ndarray) -> np.ndarray:
    expected = np.asarray(value, dtype=np.float32)
    if dtype == "float16":
        return expected.astype(np.float16).astype(np.float32)
    if dtype == "bfloat16":
        return array_from_storage(array_to_storage(expected, "bfloat16"), "bfloat16")
    return expected
