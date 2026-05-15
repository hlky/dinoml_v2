from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.ir import dtype_nbytes
from dinoml.lowering.cpp_types import cpu_storage_type
from dinoml.kernels.bmm import BMM_OPS, bmm_op_spec
from dinoml.lowering.ops.base import OpLowering
from dinoml.ops.definitions import get_op_def

_CPU_BMM_OPS = {"bmm_rcr", "bmm_rrr"}
from dinoml.lowering.shape_buffers import c_ident as _c_ident


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    if target == "cpu":
        return _render_cpu_template("bmm_cpu.cpp.j2", _cpu_context(node, tensor_map))
    if target == "cuda":
        return None
    raise ValueError(f"Unsupported BMM lowering target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    op_name = str(node["op"])
    spec = bmm_op_spec(op_name)
    if spec.epilogue not in {"none", "add"}:
        raise NotImplementedError(f"{op_name} {target.upper()} lowering does not support BMM epilogue {spec.epilogue!r}")
    a_name, b_name = (str(name) for name in node["inputs"][:2])
    d0_name = str(node["inputs"][2]) if spec.epilogue == "add" else None
    c_name = str(node["outputs"][0])
    a_ident = _c_ident(a_name)
    b_ident = _c_ident(b_name)
    d0_ident = _c_ident(d0_name) if d0_name is not None else None
    c_ident = _c_ident(c_name)
    _validate_static_contract(op_name, [tensor_map[name] for name in node["inputs"]], tensor_map[c_name], target=target)

    a_batch = f"shape_{a_ident}_0"
    b_batch = f"shape_{b_ident}_0"
    batch_expr = f"(({a_batch} == 1) ? {b_batch} : {a_batch})"
    m_expr = f"shape_{a_ident}_{2 if spec.a_layout == 'c' else 1}"
    k_a_expr = f"shape_{a_ident}_{1 if spec.a_layout == 'c' else 2}"
    n_expr = f"shape_{b_ident}_{1 if spec.b_layout == 'c' else 2}"
    k_b_expr = f"shape_{b_ident}_{2 if spec.b_layout == 'c' else 1}"
    k_expr = k_a_expr
    output_check = _output_shape_check(c_ident, batch_expr, m_expr, n_expr, spec.c_layout)

    if target == "cpu":
        return _render_cpu_launch(
            node=node,
            tensor_map=tensor_map,
            batch_expr=batch_expr,
            a_batch=a_batch,
            b_batch=b_batch,
            m_expr=m_expr,
            n_expr=n_expr,
            k_expr=k_expr,
            k_b_expr=k_b_expr,
            output_check=output_check,
        )
    if target != "cuda":
        raise ValueError(f"{op_name} lowering is only implemented for CPU or CUDA")

    dtype = str(tensor_map[c_name]["dtype"])
    manifest_item = _manifest_kernel_item(kernel_manifest, op_name, dtype)
    _validate_cutlass_execution_plan_metadata(op_name, manifest_item)
    symbol = (
        str(manifest_item["kernel_symbol"])
        if manifest_item is not None
        else get_op_def(op_name).backend_kernels[target].resolve(dtype).symbol
    )

    lda_expr = m_expr if spec.a_layout == "c" else k_expr
    ldb_expr = k_expr if spec.b_layout == "c" else n_expr
    ldc_expr = m_expr if spec.c_layout == "c" else n_expr
    batch_stride_a = f"(({a_batch} == 1 && {batch_expr} != 1) ? 0 : ({m_expr}) * ({k_expr}))"
    batch_stride_b = f"(({b_batch} == 1 && {batch_expr} != 1) ? 0 : ({n_expr}) * ({k_expr}))"
    d0_layout = _d0_layout_context(op_name, spec.c_layout, tensor_map[d0_name], tensor_map[c_name], m_expr, n_expr) if d0_name else None
    batch_stride_d0 = d0_layout["batch_stride"] if d0_layout is not None else f"({m_expr}) * ({n_expr})"
    batch_stride_c = f"({m_expr}) * ({n_expr})"
    selected_candidate = _selected_cutlass_candidate(manifest_item)
    lines = [
        f'if (!({a_batch} == {b_batch} || {a_batch} == 1 || {b_batch} == 1)) '
        f'return dinoml::module::fail("{op_name} batch dimension mismatch");',
        f'if ({k_a_expr} != {k_b_expr}) return dinoml::module::fail("{op_name} K dimension mismatch");',
        f'if ({output_check}) return dinoml::module::fail("{op_name} output shape mismatch");',
    ]
    if d0_ident is not None and d0_layout is not None:
        lines.append(_d0_shape_check(op_name, d0_ident, c_ident, d0_layout))
    default_launch = _cutlass_launch_with_alignment_fallback_lines(
        op_name=op_name,
        symbol=symbol,
        candidate=selected_candidate,
        fallback_candidates=_cutlass_alignment_fallback_candidates(manifest_item, selected_candidate),
        dtype=dtype,
        a_ident=a_ident,
        b_ident=b_ident,
        d0_ident=d0_ident,
        c_ident=c_ident,
        batch_expr=batch_expr,
        m_expr=m_expr,
        n_expr=n_expr,
        k_expr=k_expr,
        batch_stride_a=batch_stride_a,
        batch_stride_b=batch_stride_b,
        batch_stride_d0=batch_stride_d0,
        batch_stride_c=batch_stride_c,
        lda_expr=lda_expr,
        ldb_expr=ldb_expr,
        ldd0_expr=d0_layout["ld"] if d0_layout is not None else ldc_expr,
        ldc_expr=ldc_expr,
    )
    dispatches = _cutlass_dispatch_selections(manifest_item, str(node["id"]))
    if dispatches:
        lines.extend(
            _cutlass_dispatch_lines(
                op_name=op_name,
                item=manifest_item,
                dispatches=dispatches,
                default_launch=default_launch,
                dtype=dtype,
                a_ident=a_ident,
                b_ident=b_ident,
                d0_ident=d0_ident,
                c_ident=c_ident,
                batch_expr=batch_expr,
                m_expr=m_expr,
                n_expr=n_expr,
                k_expr=k_expr,
                batch_stride_a=batch_stride_a,
                batch_stride_b=batch_stride_b,
                batch_stride_d0=batch_stride_d0,
                batch_stride_c=batch_stride_c,
                lda_expr=lda_expr,
                ldb_expr=ldb_expr,
                ldd0_expr=d0_layout["ld"] if d0_layout is not None else ldc_expr,
                ldc_expr=ldc_expr,
            )
        )
    else:
        lines.extend(default_launch)
    return "\n".join(lines)


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    if target == "cpu":
        return f"{target}:{generated_function_name(target, node, tensor_map)}"
    if target == "cuda":
        return None
    raise ValueError(f"Unsupported BMM lowering target: {target}")


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    if target == "cpu":
        return _cpu_function_name(node, tensor_map)
    if target == "cuda":
        return None
    raise ValueError(f"Unsupported BMM lowering target: {target}")


def _render_cpu_launch(
    *,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    batch_expr: str,
    a_batch: str,
    b_batch: str,
    m_expr: str,
    n_expr: str,
    k_expr: str,
    k_b_expr: str,
    output_check: str,
) -> str:
    op_name = str(node["op"])
    func = _cpu_function_name(node, tensor_map)
    a_ident = _c_ident(str(node["inputs"][0]))
    b_ident = _c_ident(str(node["inputs"][1]))
    c_ident = _c_ident(str(node["outputs"][0]))
    lines = [
        f'if (!({a_batch} == {b_batch} || {a_batch} == 1 || {b_batch} == 1)) return dinoml::module::fail("{op_name} batch dimension mismatch");',
        f'if ({k_expr} != {k_b_expr}) return dinoml::module::fail("{op_name} K dimension mismatch");',
        f'if ({output_check}) return dinoml::module::fail("{op_name} output shape mismatch");',
        "if (int err = "
        f"{func}(ptr_{a_ident}, runtime_numel_{a_ident}, ptr_{b_ident}, runtime_numel_{b_ident}, "
        f"ptr_{c_ident}, runtime_numel_{c_ident}, {batch_expr}, {a_batch}, {b_batch}, {m_expr}, {n_expr}, {k_expr})) return err;",
    ]
    return "\n".join(lines)


def _cpu_context(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    op_name = str(node["op"])
    if op_name not in _CPU_BMM_OPS:
        raise ValueError(f"CPU BMM lowering only supports {_CPU_BMM_OPS}, got {op_name}")
    output_tensor = tensor_map[node["outputs"][0]]
    dtype = str(output_tensor["dtype"])
    signature = {
        "op": op_name,
        "dtype": dtype,
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return {
        "func": f"{op_name}_{dtype}_{digest}",
        "storage_type": cpu_storage_type(dtype),
        "b_layout": bmm_op_spec(op_name).b_layout,
    }


def _cpu_function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    return str(_cpu_context(node, tensor_map)["func"])


def _render_cpu_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


def _output_shape_check(c_ident: str, batch_expr: str, m_expr: str, n_expr: str, c_layout: str) -> str:
    if c_layout == "c":
        return f"shape_{c_ident}_0 != ({batch_expr}) || shape_{c_ident}_1 != ({n_expr}) || shape_{c_ident}_2 != ({m_expr})"
    return f"shape_{c_ident}_0 != ({batch_expr}) || shape_{c_ident}_1 != ({m_expr}) || shape_{c_ident}_2 != ({n_expr})"


def _d0_layout_context(
    op_name: str,
    c_layout: str,
    d0_info: Mapping[str, Any],
    c_info: Mapping[str, Any],
    m_expr: str,
    n_expr: str,
) -> dict[str, str]:
    d0_shape = [int(dim) for dim in d0_info["shape"]]
    c_shape = [int(dim) for dim in c_info["shape"]]
    if d0_shape == c_shape:
        ld = m_expr if c_layout == "c" else n_expr
        return {"kind": "full_output", "batch_stride": f"({m_expr}) * ({n_expr})", "ld": ld}
    squeezed = list(d0_shape)
    squeezed_leading = 0
    while len(squeezed) > 1 and squeezed[0] == 1:
        squeezed = squeezed[1:]
        squeezed_leading += 1
    expected_extent = c_shape[-1]
    if len(squeezed) == 1 and int(squeezed[0]) == int(expected_extent):
        return {
            "kind": "trailing_bias",
            "batch_stride": "0",
            "ld": "0",
            "trailing_axis": str(len(d0_shape) - 1),
            "leading_ones": str(squeezed_leading),
        }
    raise NotImplementedError(f"{op_name} CUDA lowering requires full-output or trailing-bias d0 shape")


def _d0_shape_check(op_name: str, d0_ident: str, c_ident: str, d0_layout: Mapping[str, str]) -> str:
    if d0_layout["kind"] == "full_output":
        return (
            f'if (shape_{d0_ident}_0 != shape_{c_ident}_0 || shape_{d0_ident}_1 != shape_{c_ident}_1 || '
            f'shape_{d0_ident}_2 != shape_{c_ident}_2) '
            f'return dinoml::module::fail("{op_name} d0 shape mismatch");'
        )
    leading_checks = [
        f"shape_{d0_ident}_{axis} != 1"
        for axis in range(int(d0_layout.get("leading_ones", "0")))
    ]
    trailing_axis = int(d0_layout["trailing_axis"])
    checks = [*leading_checks, f"shape_{d0_ident}_{trailing_axis} != shape_{c_ident}_2"]
    return f'if ({" || ".join(checks)}) return dinoml::module::fail("{op_name} d0 shape mismatch");'


def _validate_cutlass_execution_plan_metadata(op_name: str, item: Mapping[str, Any] | None) -> None:
    if not isinstance(item, Mapping):
        return
    selection = item.get("execution_plan_selection")
    if isinstance(selection, Mapping):
        _validate_cutlass_execution_plan_selection(op_name, selection)
    for dispatch in item.get("execution_plan_dispatch", ()):
        if isinstance(dispatch, Mapping):
            _validate_cutlass_execution_plan_selection(op_name, dispatch)


def _validate_cutlass_execution_plan_selection(op_name: str, selection: Mapping[str, Any]) -> None:
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
    d0_ident: str | None,
    c_ident: str,
    batch_expr: str,
    m_expr: str,
    n_expr: str,
    k_expr: str,
    batch_stride_a: str,
    batch_stride_b: str,
    batch_stride_d0: str,
    batch_stride_c: str,
    lda_expr: str,
    ldb_expr: str,
    ldd0_expr: str,
    ldc_expr: str,
    check_alignment: bool = True,
) -> list[str]:
    lines = (
        _cutlass_runtime_alignment_checks(
            op_name,
            _candidate_cutlass_alignment(candidate),
            _candidate_cutlass_epilogue_alignment(candidate),
            dtype,
            a_ident,
            b_ident,
            d0_ident,
        )
        if check_alignment
        else []
    )
    if d0_ident is None:
        lines.append(
            f"if (int err = {symbol}(ptr_{a_ident}, ptr_{b_ident}, ptr_{c_ident}, "
            f"static_cast<int>({batch_expr}), static_cast<int>({m_expr}), static_cast<int>({n_expr}), "
            f"static_cast<int>({k_expr}), static_cast<int64_t>({batch_stride_a}), "
            f"static_cast<int64_t>({batch_stride_b}), static_cast<int64_t>({batch_stride_c}), "
            f"static_cast<int>({lda_expr}), static_cast<int>({ldb_expr}), static_cast<int>({ldc_expr}), "
            f'session->stream)) return dinoml::module::fail("{op_name} CUTLASS BMM launcher failed");'
        )
    else:
        lines.append(
            f"if (int err = {symbol}(ptr_{a_ident}, ptr_{b_ident}, ptr_{d0_ident}, ptr_{c_ident}, "
            f"static_cast<int>({batch_expr}), static_cast<int>({m_expr}), static_cast<int>({n_expr}), "
            f"static_cast<int>({k_expr}), static_cast<int64_t>({batch_stride_a}), "
            f"static_cast<int64_t>({batch_stride_b}), static_cast<int64_t>({batch_stride_d0}), "
            f"static_cast<int64_t>({batch_stride_c}), static_cast<int>({lda_expr}), "
            f"static_cast<int>({ldb_expr}), static_cast<int>({ldd0_expr}), static_cast<int>({ldc_expr}), "
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
    d0_ident: str | None,
    c_ident: str,
    batch_expr: str,
    m_expr: str,
    n_expr: str,
    k_expr: str,
    batch_stride_a: str,
    batch_stride_b: str,
    batch_stride_d0: str,
    batch_stride_c: str,
    lda_expr: str,
    ldb_expr: str,
    ldd0_expr: str,
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
            d0_ident=d0_ident,
            c_ident=c_ident,
            batch_expr=batch_expr,
            m_expr=m_expr,
            n_expr=n_expr,
            k_expr=k_expr,
            batch_stride_a=batch_stride_a,
            batch_stride_b=batch_stride_b,
            batch_stride_d0=batch_stride_d0,
            batch_stride_c=batch_stride_c,
            lda_expr=lda_expr,
            ldb_expr=ldb_expr,
            ldd0_expr=ldd0_expr,
            ldc_expr=ldc_expr,
        )
    attempts: list[tuple[str, Mapping[str, Any] | None]] = [
        (symbol, candidate),
        *[(str(fallback["kernel_symbol"]), fallback) for fallback in fallback_candidates],
    ]
    lines: list[str] = []
    for index, (attempt_symbol, attempt_candidate) in enumerate(attempts):
        alignment = _candidate_cutlass_alignment(attempt_candidate)
        conditions = _cutlass_runtime_alignment_conditions(
            alignment,
            _candidate_cutlass_epilogue_alignment(attempt_candidate),
            dtype,
            a_ident,
            b_ident,
            d0_ident,
        )
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
            d0_ident=d0_ident,
            c_ident=c_ident,
            batch_expr=batch_expr,
            m_expr=m_expr,
            n_expr=n_expr,
            k_expr=k_expr,
            batch_stride_a=batch_stride_a,
            batch_stride_b=batch_stride_b,
            batch_stride_d0=batch_stride_d0,
            batch_stride_c=batch_stride_c,
            lda_expr=lda_expr,
            ldb_expr=ldb_expr,
            ldd0_expr=ldd0_expr,
            ldc_expr=ldc_expr,
            check_alignment=index == len(attempts) - 1,
        )
        lines.extend(f"  {line}" for line in body)
        lines.append("}")
    return lines


def _cutlass_dispatch_lines(
    *,
    op_name: str,
    item: Mapping[str, Any] | None,
    dispatches: list[Mapping[str, Any]],
    default_launch: list[str],
    dtype: str,
    a_ident: str,
    b_ident: str,
    d0_ident: str | None,
    c_ident: str,
    batch_expr: str,
    m_expr: str,
    n_expr: str,
    k_expr: str,
    batch_stride_a: str,
    batch_stride_b: str,
    batch_stride_d0: str,
    batch_stride_c: str,
    lda_expr: str,
    ldb_expr: str,
    ldd0_expr: str,
    ldc_expr: str,
) -> list[str]:
    lines = []
    for index, dispatch in enumerate(dispatches):
        candidate = _candidate_by_id(item, str(dispatch.get("selected_candidate_id", "")))
        branch = "if" if index == 0 else "else if"
        lines.append(
            f"{branch} ({_cutlass_dispatch_guard(dispatch, candidate, dtype, a_ident, b_ident, d0_ident, batch_expr, m_expr, n_expr, k_expr)}) {{"
        )
        body = _cutlass_launch_lines(
            op_name=op_name,
            symbol=str(dispatch.get("kernel_symbol") or candidate.get("kernel_symbol")),
            candidate=candidate,
            dtype=dtype,
            a_ident=a_ident,
            b_ident=b_ident,
            d0_ident=d0_ident,
            c_ident=c_ident,
            batch_expr=batch_expr,
            m_expr=m_expr,
            n_expr=n_expr,
            k_expr=k_expr,
            batch_stride_a=batch_stride_a,
            batch_stride_b=batch_stride_b,
            batch_stride_d0=batch_stride_d0,
            batch_stride_c=batch_stride_c,
            lda_expr=lda_expr,
            ldb_expr=ldb_expr,
            ldd0_expr=ldd0_expr,
            ldc_expr=ldc_expr,
            check_alignment=False,
        )
        lines.extend(f"  {line}" for line in body)
        lines.append("}")
    lines.append("else {")
    lines.extend(f"  {line}" for line in default_launch)
    lines.append("}")
    return lines


def _cutlass_dispatch_guard(
    selection: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    dtype: str,
    a_ident: str,
    b_ident: str,
    d0_ident: str | None,
    batch_expr: str,
    m_expr: str,
    n_expr: str,
    k_expr: str,
) -> str:
    shape = selection.get("shape", {})
    if not isinstance(shape, Mapping):
        return "false"
    batch_count = int(shape.get("batch_count", 0) or 0)
    m = int(shape.get("m", 0) or 0)
    n = int(shape.get("n", 0) or 0)
    k = int(shape.get("k", 0) or 0)
    if batch_count <= 0 or m <= 0 or n <= 0 or k <= 0:
        return "false"
    conditions = [
        f"({batch_expr}) == {batch_count}",
        f"({m_expr}) == {m}",
        f"({n_expr}) == {n}",
        f"({k_expr}) == {k}",
    ]
    conditions.extend(
        _cutlass_runtime_alignment_conditions(
            _candidate_cutlass_alignment(candidate),
            _candidate_cutlass_epilogue_alignment(candidate),
            dtype,
            a_ident,
            b_ident,
            d0_ident,
        )
    )
    return " && ".join(conditions)


def _cutlass_runtime_alignment_checks(
    op_name: str,
    align: int,
    epilogue_align: int,
    dtype: str,
    a_ident: str,
    b_ident: str,
    d0_ident: str | None,
) -> list[str]:
    lines = []
    if align > 1:
        byte_alignment = align * dtype_nbytes(dtype)
        lines.extend(
            [
                f'if (int err = dinoml::module::check_tensor_pointer_alignment(abi_{a_ident}, ptr_{a_ident}, "{op_name} A", {byte_alignment})) '
                "return err;",
                f'if (int err = dinoml::module::check_tensor_pointer_alignment(abi_{b_ident}, ptr_{b_ident}, "{op_name} B", {byte_alignment})) '
                "return err;",
            ]
        )
    if d0_ident is not None and epilogue_align > 1:
        byte_alignment = epilogue_align * dtype_nbytes(dtype)
        lines.append(
            f'if (int err = dinoml::module::check_tensor_pointer_alignment(abi_{d0_ident}, ptr_{d0_ident}, "{op_name} d0", {byte_alignment})) '
            "return err;"
        )
    return lines


def _cutlass_runtime_alignment_conditions(
    align: int,
    epilogue_align: int,
    dtype: str,
    a_ident: str,
    b_ident: str,
    d0_ident: str | None,
) -> list[str]:
    conditions = []
    if align > 1:
        byte_alignment = align * dtype_nbytes(dtype)
        conditions.extend(
            [
                f"dinoml::module::is_tensor_pointer_aligned(abi_{a_ident}, ptr_{a_ident}, {byte_alignment})",
                f"dinoml::module::is_tensor_pointer_aligned(abi_{b_ident}, ptr_{b_ident}, {byte_alignment})",
            ]
        )
    if d0_ident is not None and epilogue_align > 1:
        byte_alignment = epilogue_align * dtype_nbytes(dtype)
        conditions.append(f"dinoml::module::is_tensor_pointer_aligned(abi_{d0_ident}, ptr_{d0_ident}, {byte_alignment})")
    return conditions


def _candidate_cutlass_alignment(candidate: Mapping[str, Any] | None) -> int:
    if not isinstance(candidate, Mapping):
        return 1
    cutlass_config = candidate.get("cutlass")
    if not isinstance(cutlass_config, Mapping):
        return 1
    return int(cutlass_config.get("align", 1) or 1)


def _candidate_cutlass_epilogue_alignment(candidate: Mapping[str, Any] | None) -> int:
    if not isinstance(candidate, Mapping):
        return 1
    cutlass_config = candidate.get("cutlass")
    if not isinstance(cutlass_config, Mapping):
        return 1
    for key in ("align_c", "epilogue_align", "epilogue_alignment"):
        if cutlass_config.get(key) is not None:
            return int(cutlass_config[key] or 1)
    return 1


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


def _cutlass_dispatch_selections(item: Mapping[str, Any] | None, node_id: str) -> list[Mapping[str, Any]]:
    if not isinstance(item, Mapping):
        return []
    return [
        selection
        for selection in item.get("execution_plan_dispatch", [])
        if isinstance(selection, Mapping) and str(selection.get("node_id")) == node_id
    ]


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
    *,
    target: str,
) -> None:
    op_def = get_op_def(op_name)
    if any(input_info["dtype"] != c_info["dtype"] for input_info in input_infos):
        raise NotImplementedError(f"{op_name} {target.upper()} lowering requires matching input/output dtypes")
    if str(c_info["dtype"]) not in op_def.allowed_dtypes:
        raise NotImplementedError(f"{op_name} {target.upper()} lowering does not support dtype {c_info['dtype']}")
    if any(len(input_info["shape"]) != 3 for input_info in input_infos[:2]) or len(c_info["shape"]) != 3:
        raise NotImplementedError(f"{op_name} {target.upper()} lowering expects rank-3 A, B, and C tensors")


BMM_LOWERINGS = {
    op_name: OpLowering(
        op_name=op_name,
        render_generated_kernel=render_generated_kernel,
        render_launch=render_launch,
        source_key=source_key,
        generated_function_name=generated_function_name,
    )
    for op_name in BMM_OPS
}
