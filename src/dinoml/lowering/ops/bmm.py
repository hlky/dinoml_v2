from __future__ import annotations

import re
from typing import Any, Mapping

from dinoml.ir import dtype_nbytes
from dinoml.kernels.bmm import BMM_BASE_OPS, bmm_op_spec
from dinoml.lowering.ops.base import OpLowering
from dinoml.ops.definitions import get_op_def


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> None:
    del target, node, tensor_map
    return None


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    if target != "cuda":
        raise ValueError(f"{node['op']} lowering is currently CUDA-only")
    op_name = str(node["op"])
    spec = bmm_op_spec(op_name)
    if spec.epilogue != "none":
        raise NotImplementedError(f"{op_name} CUDA lowering only supports base BMM")
    a_name, b_name = (str(name) for name in node["inputs"][:2])
    c_name = str(node["outputs"][0])
    a_ident = _c_ident(a_name)
    b_ident = _c_ident(b_name)
    c_ident = _c_ident(c_name)
    _validate_static_contract(op_name, [tensor_map[name] for name in node["inputs"]], tensor_map[c_name])
    dtype = str(tensor_map[c_name]["dtype"])
    manifest_item = _manifest_kernel_item(kernel_manifest, op_name, dtype)
    _validate_cutlass_execution_plan_metadata(op_name, manifest_item)
    symbol = (
        str(manifest_item["kernel_symbol"])
        if manifest_item is not None
        else get_op_def(op_name).backend_kernels[target].resolve(dtype).symbol
    )

    a_batch = f"shape_{a_ident}_0"
    b_batch = f"shape_{b_ident}_0"
    batch_expr = f"(({a_batch} == 1) ? {b_batch} : {a_batch})"
    m_expr = f"shape_{a_ident}_{2 if spec.a_layout == 'c' else 1}"
    k_a_expr = f"shape_{a_ident}_{1 if spec.a_layout == 'c' else 2}"
    n_expr = f"shape_{b_ident}_{1 if spec.b_layout == 'c' else 2}"
    k_b_expr = f"shape_{b_ident}_{2 if spec.b_layout == 'c' else 1}"
    k_expr = k_a_expr
    lda_expr = m_expr if spec.a_layout == "c" else k_expr
    ldb_expr = k_expr if spec.b_layout == "c" else n_expr
    ldc_expr = m_expr if spec.c_layout == "c" else n_expr
    batch_stride_a = f"(({a_batch} == 1 && {batch_expr} != 1) ? 0 : ({m_expr}) * ({k_expr}))"
    batch_stride_b = f"(({b_batch} == 1 && {batch_expr} != 1) ? 0 : ({n_expr}) * ({k_expr}))"
    batch_stride_c = f"({m_expr}) * ({n_expr})"
    output_check = _output_shape_check(c_ident, batch_expr, m_expr, n_expr, spec.c_layout)
    selected_candidate = _selected_cutlass_candidate(manifest_item)
    lines = [
        f'if (!({a_batch} == {b_batch} || {a_batch} == 1 || {b_batch} == 1)) '
        f'return dinoml::module::fail("{op_name} batch dimension mismatch");',
        f'if ({k_a_expr} != {k_b_expr}) return dinoml::module::fail("{op_name} K dimension mismatch");',
        f'if ({output_check}) return dinoml::module::fail("{op_name} output shape mismatch");',
    ]
    lines.extend(
        _cutlass_launch_with_alignment_fallback_lines(
            op_name=op_name,
            symbol=symbol,
            candidate=selected_candidate,
            fallback_candidates=_cutlass_alignment_fallback_candidates(manifest_item, selected_candidate),
            dtype=dtype,
            a_ident=a_ident,
            b_ident=b_ident,
            c_ident=c_ident,
            batch_expr=batch_expr,
            m_expr=m_expr,
            n_expr=n_expr,
            k_expr=k_expr,
            batch_stride_a=batch_stride_a,
            batch_stride_b=batch_stride_b,
            batch_stride_c=batch_stride_c,
            lda_expr=lda_expr,
            ldb_expr=ldb_expr,
            ldc_expr=ldc_expr,
        )
    )
    return "\n".join(lines)


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> None:
    del target, node, tensor_map
    return None


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> None:
    del target, node, tensor_map
    return None


def _output_shape_check(c_ident: str, batch_expr: str, m_expr: str, n_expr: str, c_layout: str) -> str:
    if c_layout == "c":
        return f"shape_{c_ident}_0 != ({batch_expr}) || shape_{c_ident}_1 != ({n_expr}) || shape_{c_ident}_2 != ({m_expr})"
    return f"shape_{c_ident}_0 != ({batch_expr}) || shape_{c_ident}_1 != ({m_expr}) || shape_{c_ident}_2 != ({n_expr})"


def _validate_cutlass_execution_plan_metadata(op_name: str, item: Mapping[str, Any] | None) -> None:
    if not isinstance(item, Mapping):
        return
    if item.get("execution_plan_dispatch"):
        raise NotImplementedError(f"{op_name} BMM guarded execution-plan dispatch is not supported")
    selection = item.get("execution_plan_selection")
    if not isinstance(selection, Mapping):
        return
    split_k = int(selection.get("split_k", 1) or 1)
    workspace_nbytes = int(selection.get("workspace_nbytes", 0) or 0)
    if split_k != 1 or workspace_nbytes != 0:
        raise NotImplementedError(
            f"{op_name} BMM execution-plan selection requires split_k=1 and workspace_nbytes=0"
        )


def _cutlass_launch_lines(
    *,
    op_name: str,
    symbol: str,
    candidate: Mapping[str, Any] | None,
    dtype: str,
    a_ident: str,
    b_ident: str,
    c_ident: str,
    batch_expr: str,
    m_expr: str,
    n_expr: str,
    k_expr: str,
    batch_stride_a: str,
    batch_stride_b: str,
    batch_stride_c: str,
    lda_expr: str,
    ldb_expr: str,
    ldc_expr: str,
    check_alignment: bool = True,
) -> list[str]:
    lines = (
        _cutlass_runtime_alignment_checks(op_name, _candidate_cutlass_alignment(candidate), dtype, a_ident, b_ident)
        if check_alignment
        else []
    )
    lines.append(
        f"if (int err = {symbol}(ptr_{a_ident}, ptr_{b_ident}, ptr_{c_ident}, "
        f"static_cast<int>({batch_expr}), static_cast<int>({m_expr}), static_cast<int>({n_expr}), "
        f"static_cast<int>({k_expr}), static_cast<int64_t>({batch_stride_a}), "
        f"static_cast<int64_t>({batch_stride_b}), static_cast<int64_t>({batch_stride_c}), "
        f"static_cast<int>({lda_expr}), static_cast<int>({ldb_expr}), static_cast<int>({ldc_expr}), "
        f'session->stream)) return dinoml::module::fail("{op_name} CUTLASS BMM launcher failed");'
    )
    return lines


def _cutlass_launch_with_alignment_fallback_lines(
    *,
    op_name: str,
    symbol: str,
    candidate: Mapping[str, Any] | None,
    fallback_candidates: list[Mapping[str, Any]],
    dtype: str,
    a_ident: str,
    b_ident: str,
    c_ident: str,
    batch_expr: str,
    m_expr: str,
    n_expr: str,
    k_expr: str,
    batch_stride_a: str,
    batch_stride_b: str,
    batch_stride_c: str,
    lda_expr: str,
    ldb_expr: str,
    ldc_expr: str,
) -> list[str]:
    selected_alignment = _candidate_cutlass_alignment(candidate)
    if selected_alignment <= 1 or not fallback_candidates:
        return _cutlass_launch_lines(
            op_name=op_name,
            symbol=symbol,
            candidate=candidate,
            dtype=dtype,
            a_ident=a_ident,
            b_ident=b_ident,
            c_ident=c_ident,
            batch_expr=batch_expr,
            m_expr=m_expr,
            n_expr=n_expr,
            k_expr=k_expr,
            batch_stride_a=batch_stride_a,
            batch_stride_b=batch_stride_b,
            batch_stride_c=batch_stride_c,
            lda_expr=lda_expr,
            ldb_expr=ldb_expr,
            ldc_expr=ldc_expr,
        )
    attempts: list[tuple[str, Mapping[str, Any] | None]] = [
        (symbol, candidate),
        *[(str(fallback["kernel_symbol"]), fallback) for fallback in fallback_candidates],
    ]
    lines: list[str] = []
    for index, (attempt_symbol, attempt_candidate) in enumerate(attempts):
        alignment = _candidate_cutlass_alignment(attempt_candidate)
        conditions = _cutlass_runtime_alignment_conditions(alignment, dtype, a_ident, b_ident)
        if index == 0:
            lines.append(f"if ({' && '.join(conditions)}) {{")
        elif index == len(attempts) - 1 or not conditions:
            lines.append("else {")
        else:
            lines.append(f"else if ({' && '.join(conditions)}) {{")
        body = _cutlass_launch_lines(
            op_name=op_name,
            symbol=attempt_symbol,
            candidate=attempt_candidate,
            dtype=dtype,
            a_ident=a_ident,
            b_ident=b_ident,
            c_ident=c_ident,
            batch_expr=batch_expr,
            m_expr=m_expr,
            n_expr=n_expr,
            k_expr=k_expr,
            batch_stride_a=batch_stride_a,
            batch_stride_b=batch_stride_b,
            batch_stride_c=batch_stride_c,
            lda_expr=lda_expr,
            ldb_expr=ldb_expr,
            ldc_expr=ldc_expr,
            check_alignment=index == len(attempts) - 1,
        )
        lines.extend(f"  {line}" for line in body)
        lines.append("}")
    return lines


def _cutlass_runtime_alignment_checks(
    op_name: str,
    align: int,
    dtype: str,
    a_ident: str,
    b_ident: str,
) -> list[str]:
    if align <= 1:
        return []
    byte_alignment = align * dtype_nbytes(dtype)
    return [
        f'if (int err = dinoml::module::check_tensor_pointer_alignment(abi_{a_ident}, ptr_{a_ident}, "{op_name} A", {byte_alignment})) '
        "return err;",
        f'if (int err = dinoml::module::check_tensor_pointer_alignment(abi_{b_ident}, ptr_{b_ident}, "{op_name} B", {byte_alignment})) '
        "return err;",
    ]


def _cutlass_runtime_alignment_conditions(align: int, dtype: str, a_ident: str, b_ident: str) -> list[str]:
    if align <= 1:
        return []
    byte_alignment = align * dtype_nbytes(dtype)
    return [
        f"dinoml::module::is_tensor_pointer_aligned(abi_{a_ident}, ptr_{a_ident}, {byte_alignment})",
        f"dinoml::module::is_tensor_pointer_aligned(abi_{b_ident}, ptr_{b_ident}, {byte_alignment})",
    ]


def _candidate_cutlass_alignment(candidate: Mapping[str, Any] | None) -> int:
    if not isinstance(candidate, Mapping):
        return 1
    cutlass_config = candidate.get("cutlass")
    if not isinstance(cutlass_config, Mapping):
        return 1
    return int(cutlass_config.get("align", 1) or 1)


def _selected_cutlass_candidate(item: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(item, Mapping):
        return None
    selected = item.get("selected_candidate")
    return selected if isinstance(selected, Mapping) else None


def _candidate_by_id(item: Mapping[str, Any] | None, candidate_id: str) -> Mapping[str, Any]:
    if not isinstance(item, Mapping):
        return {}
    for candidate in item.get("candidates", []):
        if isinstance(candidate, Mapping) and str(candidate.get("candidate_id")) == candidate_id:
            return candidate
    return {}


def _cutlass_alignment_fallback_candidates(
    item: Mapping[str, Any] | None,
    selected_candidate: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    if not isinstance(item, Mapping) or not isinstance(selected_candidate, Mapping):
        return []
    candidates = []
    for fallback in item.get("alignment_fallbacks", ()):
        if not isinstance(fallback, Mapping):
            continue
        candidate = _candidate_by_id(item, str(fallback.get("candidate_id", "")))
        if candidate:
            candidates.append(candidate)
    return candidates


def _manifest_kernel_item(kernel_manifest: Mapping[str, Any] | None, op_name: str, dtype: str) -> Mapping[str, Any] | None:
    if kernel_manifest is None:
        return None
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("op") != op_name or item.get("kernel_library") != "cutlass_bmm":
            continue
        selected_id = item.get("selected_candidate_id")
        for candidate in item.get("candidates", []):
            if candidate.get("candidate_id") == selected_id and candidate.get("dtype") == dtype:
                return {
                    **item,
                    "kernel_symbol": str(item.get("kernel_symbol") or candidate.get("kernel_symbol")),
                    "selected_candidate": candidate,
                }
        candidate_set = item.get("candidate_set", {})
        if isinstance(candidate_set, Mapping) and candidate_set.get("dtype") == dtype:
            return item
    return None


def _validate_static_contract(
    op_name: str,
    input_infos: list[Mapping[str, Any]],
    c_info: Mapping[str, Any],
) -> None:
    op_def = get_op_def(op_name)
    if any(input_info["dtype"] != c_info["dtype"] for input_info in input_infos):
        raise NotImplementedError(f"{op_name} CUDA lowering requires matching input/output dtypes")
    if str(c_info["dtype"]) not in op_def.allowed_dtypes:
        raise NotImplementedError(f"{op_name} CUDA lowering does not support dtype {c_info['dtype']}")
    if any(len(input_info["shape"]) != 3 for input_info in input_infos[:2]) or len(c_info["shape"]) != 3:
        raise NotImplementedError(f"{op_name} CUDA lowering expects rank-3 A, B, and C tensors")


def _c_ident(name: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"_{ident}"
    return ident


BMM_LOWERINGS = {
    op_name: OpLowering(
        op_name=op_name,
        render_generated_kernel=render_generated_kernel,
        render_launch=render_launch,
        source_key=source_key,
        generated_function_name=generated_function_name,
    )
    for op_name in BMM_BASE_OPS
}
