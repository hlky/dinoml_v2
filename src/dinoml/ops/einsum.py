from __future__ import annotations

from typing import Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.kernels.families.gemm import GEMM_SUPPORTED_DTYPES
from dinoml.ops.bmm import bmm_rrr
from dinoml.ops.collections import permute
from dinoml.ops.gemm import gemm_rcr
from dinoml.ops.shape_views import reshape


EINSUM_SUPPORTED_DTYPES = GEMM_SUPPORTED_DTYPES


def einsum(equation: object, *operands: object) -> Tensor:
    if not isinstance(equation, str):
        raise ValueError(f"einsum equation must be a string, got {type(equation).__name__}")
    if len(operands) != 2:
        raise NotImplementedError(f"einsum currently supports exactly two input tensors, got {len(operands)}")

    lhs = as_tensor(operands[0], dtype_hint="float32")
    rhs = as_tensor(operands[1], dtype_hint=lhs.dtype)
    if lhs.builder is not rhs.builder:
        raise ValueError("Cannot combine tensors from different DinoML traces")
    if lhs.dtype != rhs.dtype:
        raise ValueError(f"einsum dtype mismatch: {lhs.dtype} vs {rhs.dtype}")
    if lhs.dtype not in EINSUM_SUPPORTED_DTYPES:
        raise ValueError(f"einsum does not support dtype {lhs.dtype}")
    if lhs.dynamic or rhs.dynamic:
        raise ValueError("einsum currently supports only static input shapes")

    input_terms, output_labels = _parse_equation(equation)
    lhs_labels, rhs_labels = input_terms
    if len(lhs_labels) != lhs.rank or len(rhs_labels) != rhs.rank:
        raise ValueError(
            f"einsum operand ranks do not match equation: lhs rank {lhs.rank} vs {len(lhs_labels)}, "
            f"rhs rank {rhs.rank} vs {len(rhs_labels)}"
        )

    lhs_axes = {label: axis for axis, label in enumerate(lhs_labels)}
    rhs_axes = {label: axis for axis, label in enumerate(rhs_labels)}
    lhs_set = set(lhs_labels)
    rhs_set = set(rhs_labels)
    output_set = set(output_labels)

    if not output_labels:
        raise NotImplementedError("einsum currently does not support scalar outputs")

    dropped_lhs_only = [label for label in lhs_labels if label not in rhs_set and label not in output_set]
    if dropped_lhs_only:
        raise NotImplementedError(
            "einsum currently does not support reducing labels that appear only in the lhs operand: "
            + ", ".join(dropped_lhs_only)
        )
    dropped_rhs_only = [label for label in rhs_labels if label not in lhs_set and label not in output_set]
    if dropped_rhs_only:
        raise NotImplementedError(
            "einsum currently does not support reducing labels that appear only in the rhs operand: "
            + ", ".join(dropped_rhs_only)
        )

    shared_labels = lhs_set & rhs_set
    contract_labels = [label for label in lhs_labels if label in shared_labels and label not in output_set]
    if not contract_labels:
        raise NotImplementedError("einsum currently requires at least one contraction label")

    batch_labels = [label for label in output_labels if label in shared_labels]
    lhs_free_labels = [label for label in lhs_labels if label in output_set and label not in rhs_set]
    rhs_free_labels = [label for label in rhs_labels if label in output_set and label not in lhs_set]
    logical_output_labels = [*batch_labels, *lhs_free_labels, *rhs_free_labels]
    if set(logical_output_labels) != output_set or len(logical_output_labels) != len(output_labels):
        raise NotImplementedError(
            "einsum currently supports only outputs composed of shared batch labels plus lhs-only and rhs-only labels"
        )

    for label in shared_labels:
        lhs_dim = int(lhs.shape[lhs_axes[label]])
        rhs_dim = int(rhs.shape[rhs_axes[label]])
        if lhs_dim != rhs_dim:
            raise ValueError(
                f"einsum currently requires matching extents for shared label {label!r}, got {lhs_dim} and {rhs_dim}"
            )

    batch_shape = [int(lhs.shape[lhs_axes[label]]) for label in batch_labels]
    lhs_free_shape = [int(lhs.shape[lhs_axes[label]]) for label in lhs_free_labels]
    rhs_free_shape = [int(rhs.shape[rhs_axes[label]]) for label in rhs_free_labels]
    contract_shape = [int(lhs.shape[lhs_axes[label]]) for label in contract_labels]

    batch_extent = _shape_product(batch_shape) if batch_shape else 1
    m_extent = _shape_product(lhs_free_shape) if lhs_free_shape else 1
    k_extent = _shape_product(contract_shape)
    n_extent = _shape_product(rhs_free_shape) if rhs_free_shape else 1

    logical_output_shape = [*batch_shape, *lhs_free_shape, *rhs_free_shape]
    if batch_shape:
        lhs_prepared = _permute_if_needed(lhs, lhs_labels, [*batch_labels, *lhs_free_labels, *contract_labels])
        rhs_prepared = _permute_if_needed(rhs, rhs_labels, [*batch_labels, *contract_labels, *rhs_free_labels])
        lhs_matrix = _reshape_if_needed(lhs_prepared, [batch_extent, m_extent, k_extent])
        rhs_matrix = _reshape_if_needed(rhs_prepared, [batch_extent, k_extent, n_extent])
        contracted = bmm_rrr(lhs_matrix, rhs_matrix)
        result = _reshape_if_needed(contracted, logical_output_shape)
    else:
        lhs_prepared = _permute_if_needed(lhs, lhs_labels, [*lhs_free_labels, *contract_labels])
        rhs_prepared = _permute_if_needed(rhs, rhs_labels, [*rhs_free_labels, *contract_labels])
        lhs_matrix = _reshape_if_needed(lhs_prepared, [m_extent, k_extent])
        rhs_matrix = _reshape_if_needed(rhs_prepared, [n_extent, k_extent])
        contracted = gemm_rcr(lhs_matrix, rhs_matrix)
        result = _reshape_if_needed(contracted, logical_output_shape)

    return _permute_if_needed(result, logical_output_labels, output_labels)


def _parse_equation(equation: str) -> tuple[tuple[list[str], list[str]], list[str]]:
    normalized = equation.replace(" ", "")
    if "..." in normalized:
        raise NotImplementedError("einsum currently does not support ellipsis")
    if "->" not in normalized:
        raise NotImplementedError("einsum currently requires an explicit output")
    input_expr, output_expr = normalized.split("->", 1)
    input_terms = input_expr.split(",")
    if len(input_terms) != 2:
        raise NotImplementedError(f"einsum currently supports exactly two operands, got {len(input_terms)}")
    lhs_labels = _parse_term(input_terms[0], "lhs")
    rhs_labels = _parse_term(input_terms[1], "rhs")
    output_labels = _parse_term(output_expr, "output", allow_empty=True)
    unknown_output = [label for label in output_labels if label not in set(lhs_labels) | set(rhs_labels)]
    if unknown_output:
        raise ValueError(f"einsum output labels must appear in an input operand, got {unknown_output}")
    return (lhs_labels, rhs_labels), output_labels


def _parse_term(term: str, name: str, *, allow_empty: bool = False) -> list[str]:
    if not term:
        if allow_empty:
            return []
        raise ValueError(f"einsum {name} term must not be empty")
    labels: list[str] = []
    seen: set[str] = set()
    for label in term:
        if not label.isalpha():
            raise ValueError(f"einsum {name} term contains unsupported label {label!r}")
        if label in seen:
            raise NotImplementedError(f"einsum currently does not support repeated labels within one operand: {term!r}")
        seen.add(label)
        labels.append(label)
    return labels


def _permute_if_needed(tensor: Tensor, current_labels: Sequence[str], target_labels: Sequence[str]) -> Tensor:
    if list(current_labels) == list(target_labels):
        return tensor
    dims = [current_labels.index(label) for label in target_labels]
    return permute(tensor, dims)


def _reshape_if_needed(tensor: Tensor, shape: Sequence[int]) -> Tensor:
    normalized = [int(dim) for dim in shape]
    if list(tensor.shape) == normalized:
        return tensor
    return reshape(tensor, normalized)


def _shape_product(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


__all__ = ["EINSUM_SUPPORTED_DTYPES", "einsum"]
