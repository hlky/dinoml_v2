from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from dinoml.kernels.families.gemm import gemm_op_spec


def cutlass_candidate_alignment(candidate: Mapping[str, Any]) -> int:
    cutlass = candidate.get("cutlass", {})
    if isinstance(cutlass, Mapping) and cutlass.get("align") is not None:
        return int(cutlass["align"])
    return 1


def cutlass_gemm_problem_alignment(op_name: str, dtype: str, *, n: int, k: int) -> int:
    spec = gemm_op_spec(op_name)
    if spec.base_layout == "rrr":
        alignment_basis = math.gcd(int(k), int(n))
    elif spec.base_layout == "rcr":
        alignment_basis = int(k)
    else:
        return 1
    return _max_dtype_alignment(dtype, alignment_basis)


def cutlass_gemm_guaranteed_alignment(
    op_name: str,
    dtype: str,
    a_tensor: Mapping[str, Any],
    b_tensor: Mapping[str, Any],
) -> int:
    spec = gemm_op_spec(op_name)
    a_spec = a_tensor.get("shape_spec", a_tensor["shape"])
    b_spec = b_tensor.get("shape_spec", b_tensor["shape"])
    b_k_axis = 0 if spec.base_layout == "rrr" else 1
    k_alignment = math.gcd(_dim_divisible_by(a_spec[-1]), _dim_divisible_by(b_spec[b_k_axis]))
    if spec.base_layout == "rrr":
        alignment_basis = math.gcd(k_alignment, _dim_divisible_by(b_spec[1]))
    elif spec.base_layout == "rcr":
        alignment_basis = k_alignment
    else:
        return 1
    return _max_dtype_alignment(dtype, alignment_basis)


def cutlass_gemm_layout_alignment(
    tensor_names: Sequence[str],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> int | None:
    alignments = []
    for name in tensor_names:
        layout = tensor_map[str(name)].get("layout", {})
        if isinstance(layout, Mapping) and layout.get("alignment") is not None:
            alignments.append(int(layout["alignment"]))
    if len(alignments) != len(tensor_names):
        return None
    return min(alignments)


def combine_alignment_caps(*alignments: int | None) -> int | None:
    values = [int(alignment) for alignment in alignments if alignment is not None]
    return min(values) if values else None


def filter_candidates_by_alignment(
    candidates: Sequence[Mapping[str, Any]],
    max_alignment: int | None,
) -> list[dict[str, Any]]:
    copied = [dict(candidate) for candidate in candidates]
    if max_alignment is None:
        return copied
    return [candidate for candidate in copied if cutlass_candidate_alignment(candidate) <= max_alignment]


def _max_dtype_alignment(dtype: str, number: int) -> int:
    for alignment in _dtype_alignments(dtype):
        if int(number) % alignment == 0:
            return alignment
    return 1


def _dtype_alignments(dtype: str) -> tuple[int, ...]:
    if dtype in {"float16", "bfloat16"}:
        return (8, 4, 2, 1)
    if dtype == "float32":
        return (4, 2, 1)
    return (1,)


def _dim_divisible_by(dim: Any) -> int:
    if isinstance(dim, Mapping):
        return int(dim.get("divisible_by", 1))
    return int(dim)
