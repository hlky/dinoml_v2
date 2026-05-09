from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from dinoml.kernels.families.bmm import bmm_op_spec
from dinoml.kernels.families.gemm import gemm_op_spec


def cutlass_candidate_alignment(candidate: Mapping[str, Any]) -> int:
    cutlass = candidate.get("cutlass", {})
    if isinstance(cutlass, Mapping) and cutlass.get("align") is not None:
        return int(cutlass["align"])
    return 1


def cutlass_candidate_epilogue_alignment(candidate: Mapping[str, Any]) -> int:
    cutlass = candidate.get("cutlass", {})
    if isinstance(cutlass, Mapping):
        for key in ("align_c", "epilogue_align", "epilogue_alignment"):
            if cutlass.get(key) is not None:
                return int(cutlass[key])
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
        alignment = cutlass_tensor_accessor_alignment(str(name), tensor_map)
        if alignment is not None:
            alignments.append(alignment)
    if len(alignments) != len(tensor_names):
        return None
    return min(alignments)


def cutlass_gemm_static_alignment_context(
    op_name: str,
    dtype: str,
    tensor_map: Mapping[str, Mapping[str, Any]],
    *,
    a_name: str,
    b_name: str,
    c_name: str | None = None,
    epilogue_names: Sequence[str] = (),
) -> dict[str, Any]:
    a_tensor = tensor_map[str(a_name)]
    b_tensor = tensor_map[str(b_name)]
    shape_alignment = cutlass_gemm_guaranteed_alignment(op_name, dtype, a_tensor, b_tensor)
    return _cutlass_gemm_alignment_context(
        op_name,
        dtype,
        tensor_map,
        a_name=a_name,
        b_name=b_name,
        c_name=c_name,
        epilogue_names=epilogue_names,
        shape_alignment=shape_alignment,
        shape_alignment_source="shape_spec_divisibility",
    )


def cutlass_bmm_problem_alignment(op_name: str, dtype: str, *, m: int, n: int, k: int) -> int:
    spec = bmm_op_spec(op_name)
    lda = int(m) if spec.a_layout == "c" else int(k)
    ldb = int(k) if spec.b_layout == "c" else int(n)
    return _max_dtype_alignment(dtype, math.gcd(lda, ldb))


def cutlass_bmm_guaranteed_alignment(
    op_name: str,
    dtype: str,
    a_tensor: Mapping[str, Any],
    b_tensor: Mapping[str, Any],
) -> int:
    del op_name
    a_spec = a_tensor.get("shape_spec", a_tensor["shape"])
    b_spec = b_tensor.get("shape_spec", b_tensor["shape"])
    lda_alignment = _dim_divisible_by(a_spec[2])
    ldb_alignment = _dim_divisible_by(b_spec[2])
    return _max_dtype_alignment(dtype, math.gcd(lda_alignment, ldb_alignment))


def cutlass_bmm_static_alignment_context(
    op_name: str,
    dtype: str,
    tensor_map: Mapping[str, Mapping[str, Any]],
    *,
    a_name: str,
    b_name: str,
    c_name: str | None = None,
    epilogue_names: Sequence[str] = (),
) -> dict[str, Any]:
    a_tensor = tensor_map[str(a_name)]
    b_tensor = tensor_map[str(b_name)]
    shape_alignment = cutlass_bmm_guaranteed_alignment(op_name, dtype, a_tensor, b_tensor)
    context = _cutlass_gemm_alignment_context(
        op_name,
        dtype,
        tensor_map,
        a_name=a_name,
        b_name=b_name,
        c_name=c_name,
        epilogue_names=epilogue_names,
        shape_alignment=shape_alignment,
        shape_alignment_source="bmm_shape_spec_divisibility",
    )
    context["kind"] = "cutlass_bmm_alignment_context"
    return context


def cutlass_bmm_profile_alignment_context(
    op_name: str,
    dtype: str,
    tensor_map: Mapping[str, Mapping[str, Any]],
    *,
    a_name: str,
    b_name: str,
    c_name: str | None = None,
    epilogue_names: Sequence[str] = (),
    m: int,
    n: int,
    k: int,
) -> dict[str, Any]:
    shape_alignment = cutlass_bmm_problem_alignment(op_name, dtype, m=m, n=n, k=k)
    context = _cutlass_gemm_alignment_context(
        op_name,
        dtype,
        tensor_map,
        a_name=a_name,
        b_name=b_name,
        c_name=c_name,
        epilogue_names=epilogue_names,
        shape_alignment=shape_alignment,
        shape_alignment_source="profiled_bmm_problem_shape",
    )
    context["kind"] = "cutlass_bmm_alignment_context"
    return context


def cutlass_gemm_profile_alignment_context(
    op_name: str,
    dtype: str,
    tensor_map: Mapping[str, Mapping[str, Any]],
    *,
    a_name: str,
    b_name: str,
    c_name: str | None = None,
    epilogue_names: Sequence[str] = (),
    n: int,
    k: int,
) -> dict[str, Any]:
    shape_alignment = cutlass_gemm_problem_alignment(op_name, dtype, n=n, k=k)
    return _cutlass_gemm_alignment_context(
        op_name,
        dtype,
        tensor_map,
        a_name=a_name,
        b_name=b_name,
        c_name=c_name,
        epilogue_names=epilogue_names,
        shape_alignment=shape_alignment,
        shape_alignment_source="profiled_problem_shape",
    )


def cutlass_tensor_accessor_alignment(
    tensor_name: str,
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> int | None:
    return cutlass_tensor_accessor_alignment_context(tensor_name, tensor_map)["alignment"]


def cutlass_tensor_accessor_alignment_context(
    tensor_name: str,
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    tensor = tensor_map[str(tensor_name)]
    dtype = str(tensor.get("dtype", ""))
    layout = tensor.get("layout", {})
    caps: list[int] = []
    sources: list[dict[str, Any]] = []
    if isinstance(layout, Mapping):
        kind = str(layout.get("kind", "dense"))
        if kind != "dense":
            caps.append(1)
            sources.append({"source": "layout.kind", "kind": kind, "alignment": 1})
        if layout.get("alignment") is not None:
            alignment = int(layout["alignment"])
            caps.append(alignment)
            sources.append({"source": "layout.alignment", "alignment": alignment})
        if layout.get("storage_offset") is not None:
            offset_alignment = _offset_alignment(int(layout["storage_offset"]), dtype)
            if offset_alignment is not None:
                caps.append(offset_alignment)
                sources.append(
                    {
                        "source": "layout.storage_offset",
                        "offset_elements": int(layout["storage_offset"]),
                        "alignment": offset_alignment,
                    }
                )
    for key in ("storage_offset", "offset_elements"):
        if tensor.get(key) is not None:
            offset_alignment = _offset_alignment(int(tensor[key]), dtype)
            if offset_alignment is not None:
                caps.append(offset_alignment)
                sources.append(
                    {
                        "source": f"tensor.{key}",
                        "offset_elements": int(tensor[key]),
                        "alignment": offset_alignment,
                    }
                )
    alignment = min(caps) if caps else None
    return {
        "tensor": str(tensor_name),
        "alignment": alignment,
        "sources": sources,
    }


def merge_cutlass_alignment_contexts(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    copied = [dict(context) for context in contexts]
    operand_cap = combine_alignment_caps(
        *(
            _candidate_filter_value(context, "max_operand_alignment")
            for context in copied
        )
    )
    epilogue_cap = combine_alignment_caps(
        *(
            _candidate_filter_value(context, "max_epilogue_alignment")
            for context in copied
        )
    )
    return {
        "schema_version": 1,
        "kind": "cutlass_gemm_alignment_context",
        "scope": "merged_kernel_manifest",
        "node_count": len(copied),
        "candidate_filter": {
            "max_operand_alignment": operand_cap,
            "max_epilogue_alignment": epilogue_cap,
        },
        "nodes": copied,
    }


def combine_alignment_caps(*alignments: int | None) -> int | None:
    values = [int(alignment) for alignment in alignments if alignment is not None]
    return min(values) if values else None


def filter_candidates_by_alignment(
    candidates: Sequence[Mapping[str, Any]],
    max_alignment: int | None,
    max_epilogue_alignment: int | None = None,
) -> list[dict[str, Any]]:
    copied = [dict(candidate) for candidate in candidates]
    if max_alignment is None and max_epilogue_alignment is None:
        return copied
    return [
        candidate
        for candidate in copied
        if (max_alignment is None or cutlass_candidate_alignment(candidate) <= max_alignment)
        and (
            max_epilogue_alignment is None
            or cutlass_candidate_epilogue_alignment(candidate) <= max_epilogue_alignment
        )
    ]


def alignment_context_candidate_filter(context: Mapping[str, Any] | None) -> dict[str, int | None]:
    if not isinstance(context, Mapping):
        return {"max_operand_alignment": None, "max_epilogue_alignment": None}
    candidate_filter = context.get("candidate_filter", {})
    if not isinstance(candidate_filter, Mapping):
        return {"max_operand_alignment": None, "max_epilogue_alignment": None}
    return {
        "max_operand_alignment": _optional_int(candidate_filter.get("max_operand_alignment")),
        "max_epilogue_alignment": _optional_int(candidate_filter.get("max_epilogue_alignment")),
    }


def _cutlass_gemm_alignment_context(
    op_name: str,
    dtype: str,
    tensor_map: Mapping[str, Mapping[str, Any]],
    *,
    a_name: str,
    b_name: str,
    c_name: str | None,
    epilogue_names: Sequence[str],
    shape_alignment: int,
    shape_alignment_source: str,
) -> dict[str, Any]:
    a_context = cutlass_tensor_accessor_alignment_context(a_name, tensor_map)
    b_context = cutlass_tensor_accessor_alignment_context(b_name, tensor_map)
    c_context = (
        cutlass_tensor_accessor_alignment_context(c_name, tensor_map)
        if c_name is not None
        else None
    )
    epilogue_contexts = [
        cutlass_tensor_accessor_alignment_context(name, tensor_map)
        for name in epilogue_names
    ]
    operand_cap = combine_alignment_caps(
        int(shape_alignment),
        a_context["alignment"],
        b_context["alignment"],
    )
    epilogue_cap = combine_alignment_caps(
        c_context["alignment"] if c_context is not None else None,
        *(context["alignment"] for context in epilogue_contexts),
    )
    return {
        "schema_version": 1,
        "kind": "cutlass_gemm_alignment_context",
        "scope": shape_alignment_source,
        "op": str(op_name),
        "dtype": str(dtype),
        "shape_alignment": {
            "alignment": int(shape_alignment),
            "source": shape_alignment_source,
        },
        "operands": {
            "a": a_context,
            "b": b_context,
        },
        "epilogue": {
            "c": c_context,
            "inputs": epilogue_contexts,
        },
        "candidate_filter": {
            "max_operand_alignment": operand_cap,
            "max_epilogue_alignment": epilogue_cap,
        },
    }


def _candidate_filter_value(context: Mapping[str, Any], key: str) -> int | None:
    candidate_filter = context.get("candidate_filter", {})
    if not isinstance(candidate_filter, Mapping):
        return None
    return _optional_int(candidate_filter.get(key))


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _offset_alignment(offset_elements: int, dtype: str) -> int | None:
    offset = abs(int(offset_elements))
    if offset == 0:
        return None
    return _max_dtype_alignment(dtype, offset)


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
