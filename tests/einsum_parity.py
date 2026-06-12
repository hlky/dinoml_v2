from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml
from dinoml.ir import array_from_storage, array_to_storage


ATOL_BY_DTYPE = {"float16": 2e-3, "float32": 1e-6, "bfloat16": 2e-2}
RTOL_BY_DTYPE = {"float16": 2e-3, "float32": 1e-6, "bfloat16": 2e-2}


@dataclass(frozen=True)
class EinsumCase:
    name: str
    equation: str
    dtype: str
    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]
    expected_ops: tuple[str, ...]


EINSUM_CASES = (
    EinsumCase(
        name="einsum_gemm_rrr_f32",
        equation="mk,kn->mn",
        dtype="float32",
        lhs_shape=(3, 4),
        rhs_shape=(4, 5),
        expected_ops=("permute", "gemm_rcr"),
    ),
    EinsumCase(
        name="einsum_bmm_rrr_f16",
        equation="bmk,bkn->bmn",
        dtype="float16",
        lhs_shape=(2, 3, 4),
        rhs_shape=(2, 4, 5),
        expected_ops=("bmm_rrr",),
    ),
    EinsumCase(
        name="einsum_attention_scores_bf16",
        equation="bqhd,bkhd->bhqk",
        dtype="bfloat16",
        lhs_shape=(2, 3, 4, 5),
        rhs_shape=(2, 6, 4, 5),
        expected_ops=("permute", "permute", "bmm_rrr"),
    ),
    EinsumCase(
        name="einsum_attention_context_f32",
        equation="bhqk,bkhd->bqhd",
        dtype="float32",
        lhs_shape=(2, 4, 3, 6),
        rhs_shape=(2, 6, 4, 5),
        expected_ops=("permute", "bmm_rrr", "permute"),
    ),
    EinsumCase(
        name="einsum_batched_dot_f32",
        equation="bhd,bhd->bh",
        dtype="float32",
        lhs_shape=(2, 4, 5),
        rhs_shape=(2, 4, 5),
        expected_ops=("bmm_rrr",),
    ),
)


class _EinsumModule(dml.Module):
    def __init__(self, case: EinsumCase):
        self.case = case

    def forward(self, a, b):
        return dml.ops.output(dml.ops.einsum(self.case.equation, a, b), "y")


def trace_einsum_spec(case: EinsumCase):
    return dml.trace(
        _EinsumModule(case),
        inputs={
            "a": dml.TensorSpec(list(case.lhs_shape), case.dtype),
            "b": dml.TensorSpec(list(case.rhs_shape), case.dtype),
        },
        name=f"{case.name}_parity",
    )


def random_inputs(case: EinsumCase, *, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    lhs = rng.standard_normal(case.lhs_shape, dtype=np.float32).astype(np.float32, copy=False)
    rhs = rng.standard_normal(case.rhs_shape, dtype=np.float32).astype(np.float32, copy=False)
    if case.dtype == "float16":
        return {"a": lhs.astype(np.float16), "b": rhs.astype(np.float16)}
    if case.dtype == "bfloat16":
        return {
            "a": array_from_storage(array_to_storage(lhs, "bfloat16"), "bfloat16"),
            "b": array_from_storage(array_to_storage(rhs, "bfloat16"), "bfloat16"),
        }
    return {"a": lhs, "b": rhs}


def torch_oracle(case: EinsumCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[case.dtype]
    lhs = torch.tensor(inputs["a"], dtype=torch_dtype)
    rhs = torch.tensor(inputs["b"], dtype=torch_dtype)
    result = torch.einsum(case.equation, lhs, rhs).to(torch.float32).cpu().numpy()
    if case.dtype == "float16":
        return result.astype(np.float16).astype(np.float32)
    if case.dtype == "bfloat16":
        return array_from_storage(array_to_storage(result, "bfloat16"), "bfloat16")
    return result.astype(np.float32, copy=False)
