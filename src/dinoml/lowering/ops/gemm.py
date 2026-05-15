from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from dinoml.ir import dtype_nbytes
from dinoml.lowering.cpp_types import cuda_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.kernels.gemm import GEMM_OPS, gemm_op_spec
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_split_k_supported
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
    spec = gemm_op_spec(op_name)
    a_name, b_name = node["inputs"][:2]
    c_name = node["outputs"][0]
    a_ident = _c_ident(a_name)
    b_ident = _c_ident(b_name)
    c_ident = _c_ident(c_name)
    _validate_static_contract(op_name, [tensor_map[name] for name in node["inputs"]], tensor_map[c_name])
    a_rank = len(tensor_map[a_name]["shape"])
    c_rank = len(tensor_map[c_name]["shape"])
    dtype = str(tensor_map[c_name]["dtype"])
    manifest_item = _manifest_kernel_item(kernel_manifest, op_name, dtype, node_id=str(node["id"]))
    runtime_dequant_plan = _gguf_runtime_dequant_plan(manifest_item, node_id=str(node["id"]))
    launch_b_ident = b_ident
    if runtime_dequant_plan is not None:
        _validate_gguf_runtime_dequant_lowering(op_name, spec, dtype, runtime_dequant_plan)
        launch_b_ident = f"{b_ident}_dequant"
    symbol = (
        str(manifest_item["kernel_symbol"])
        if manifest_item is not None
        else get_op_def(op_name).backend_kernels[target].resolve(dtype).symbol
    )

    m_expr = _product_expr(f"shape_{a_ident}_{axis}" for axis in range(a_rank - 1))
    k_expr = f"shape_{a_ident}_{a_rank - 1}"
    if spec.base_layout == "rrr":
        n_expr = f"shape_{b_ident}_1"
        k_check = f"{k_expr} != shape_{b_ident}_0"
    else:
        n_expr = f"shape_{b_ident}_0"
        k_check = f"{k_expr} != shape_{b_ident}_1"
    output_check = _folded_output_shape_check(c_ident, a_ident, a_rank, n_expr)
    epilogue_checks = []
    epilogue_args = []
    for input_offset, input_name in enumerate(spec.epilogue.inputs, start=2):
        tensor_name = str(node["inputs"][input_offset])
        tensor_ident = _c_ident(tensor_name)
        tensor_rank = len(tensor_map[tensor_name]["shape"])
        if input_name == "bias":
            if tensor_rank == 1:
                epilogue_checks.append(f'if (shape_{tensor_ident}_0 != {n_expr}) return dinoml::module::fail("{op_name} bias shape mismatch");')
            elif tensor_rank == 2:
                epilogue_checks.append(
                    f'if (shape_{tensor_ident}_0 != 1 || shape_{tensor_ident}_1 != {n_expr}) '
                    f'return dinoml::module::fail("{op_name} bias shape mismatch");'
                )
            else:
                raise NotImplementedError(f"{op_name} CUDA lowering supports rank-1 or rank-2 bias only")
        elif input_name.startswith("d"):
            if tensor_rank != c_rank:
                raise NotImplementedError(f"{op_name} CUDA lowering requires residual tensors to match output rank")
            epilogue_checks.append(
                f'if ({_folded_output_shape_check(tensor_ident, a_ident, a_rank, n_expr)}) '
                f'return dinoml::module::fail("{op_name} {input_name} shape mismatch");'
            )
        else:
            raise NotImplementedError(f"{op_name} CUDA lowering does not support epilogue input {input_name!r}")
        epilogue_args.append(f"ptr_{tensor_ident}")

    selection = _cutlass_execution_plan_selection(manifest_item)
    lines = [
        f'if ({k_check}) return dinoml::module::fail("{op_name} K dimension mismatch");',
        f'if ({output_check}) return dinoml::module::fail("{op_name} output shape mismatch");',
    ]
    lines.extend(epilogue_checks)
    epilogue_arg_text = "".join(f"{arg}, " for arg in epilogue_args)
    selected_candidate = _selected_cutlass_candidate(manifest_item)
    default_launch = _cutlass_launch_with_alignment_fallback_lines(
        op_name=op_name,
        symbol=symbol,
        candidate=selected_candidate,
        selection=selection,
        fallback_candidates=_cutlass_alignment_fallback_candidates(manifest_item, selected_candidate),
        dtype=dtype,
        a_ident=a_ident,
        b_ident=launch_b_ident,
        epilogue_arg_text=epilogue_arg_text,
        c_ident=c_ident,
        m_expr=m_expr,
        n_expr=n_expr,
        k_expr=k_expr,
    )
    if runtime_dequant_plan is not None:
        lines.extend(
            _gguf_runtime_dequant_lines(
                op_name=op_name,
                plan=runtime_dequant_plan,
                b_ident=b_ident,
                launch_b_ident=launch_b_ident,
                dtype=dtype,
                direct_linked=bool(kernel_manifest and kernel_manifest.get("gguf_cuda_native_library")),
            )
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
                b_ident=launch_b_ident,
                epilogue_arg_text=epilogue_arg_text,
                c_ident=c_ident,
                m_expr=m_expr,
                n_expr=n_expr,
                k_expr=k_expr,
            )
        )
    else:
        lines.extend(default_launch)
    return "\n".join(lines)


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> None:
    del target, node, tensor_map
    return None


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> None:
    del target, node, tensor_map
    return None


def _product_expr(terms: Iterable[str]) -> str:
    return " * ".join(str(term) for term in terms)


def _folded_output_shape_check(output_ident: str, a_ident: str, a_rank: int, n_expr: str) -> str:
    checks = [f"shape_{output_ident}_{axis} != shape_{a_ident}_{axis}" for axis in range(a_rank - 1)]
    checks.append(f"shape_{output_ident}_{a_rank - 1} != {n_expr}")
    return " || ".join(checks)


def _validate_cutlass_execution_plan_selection(
    op_name: str,
    candidate: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None,
) -> None:
    if candidate is None or selection is None:
        return
    split_k = int(selection.get("split_k", 1) or 1)
    if split_k <= 1:
        return
    if not cutlass_gemm_split_k_supported(candidate):
        launch_abi = str(candidate.get("launch_abi", ""))
        epilogue = str(candidate.get("epilogue", ""))
        raise NotImplementedError(
            f"{op_name} CUDA lowering does not support CUTLASS split-K > 1 for epilogue {epilogue} "
            f"and launch ABI {launch_abi}; "
            f"execution plan requested split_k={split_k}"
        )
    workspace_nbytes = int(selection.get("workspace_nbytes", 0) or 0)
    if workspace_nbytes <= 0:
        raise ValueError(f"{op_name} CUTLASS split-K execution plan requires workspace_nbytes > 0")


def _cutlass_launch_lines(
    *,
    op_name: str,
    symbol: str,
    candidate: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None,
    dtype: str,
    a_ident: str,
    b_ident: str,
    epilogue_arg_text: str,
    c_ident: str,
    m_expr: str,
    n_expr: str,
    k_expr: str,
    check_alignment: bool = True,
) -> list[str]:
    split_k = int(selection.get("split_k", 1) or 1) if selection is not None else 1
    if split_k > 1:
        _validate_cutlass_execution_plan_selection(op_name, candidate, selection)
        symbol = _cutlass_split_k_kernel_symbol(symbol)
        launch_tail = f", {split_k}, session->cutlass_workspace, session->cutlass_workspace_nbytes, session->stream"
    else:
        launch_tail = ", session->stream"
    lines = (
        _cutlass_runtime_alignment_checks(op_name, _candidate_cutlass_alignment(candidate), dtype, a_ident, b_ident)
        if check_alignment
        else []
    )
    lines.append(
        f"if (int err = {symbol}(ptr_{a_ident}, ptr_{b_ident}, {epilogue_arg_text}ptr_{c_ident}, "
        f"static_cast<int>({m_expr}), static_cast<int>({n_expr}), static_cast<int>({k_expr}){launch_tail})) "
        f'return dinoml::module::fail("{op_name} CUTLASS launcher failed");'
    )
    return lines


def _cutlass_launch_with_alignment_fallback_lines(
    *,
    op_name: str,
    symbol: str,
    candidate: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None,
    fallback_candidates: list[Mapping[str, Any]],
    dtype: str,
    a_ident: str,
    b_ident: str,
    epilogue_arg_text: str,
    c_ident: str,
    m_expr: str,
    n_expr: str,
    k_expr: str,
) -> list[str]:
    selected_alignment = _candidate_cutlass_alignment(candidate)
    if selected_alignment <= 1 or not fallback_candidates:
        return _cutlass_launch_lines(
            op_name=op_name,
            symbol=symbol,
            candidate=candidate,
            selection=selection,
            dtype=dtype,
            a_ident=a_ident,
            b_ident=b_ident,
            epilogue_arg_text=epilogue_arg_text,
            c_ident=c_ident,
            m_expr=m_expr,
            n_expr=n_expr,
            k_expr=k_expr,
        )
    attempts: list[tuple[str, Mapping[str, Any] | None, Mapping[str, Any] | None]] = [
        (symbol, candidate, selection),
        *[
            (str(fallback["kernel_symbol"]), fallback, {"split_k": 1})
            for fallback in fallback_candidates
        ],
    ]
    lines: list[str] = []
    for index, (attempt_symbol, attempt_candidate, attempt_selection) in enumerate(attempts):
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
            selection=attempt_selection,
            dtype=dtype,
            a_ident=a_ident,
            b_ident=b_ident,
            epilogue_arg_text=epilogue_arg_text,
            c_ident=c_ident,
            m_expr=m_expr,
            n_expr=n_expr,
            k_expr=k_expr,
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
    epilogue_arg_text: str,
    c_ident: str,
    m_expr: str,
    n_expr: str,
    k_expr: str,
) -> list[str]:
    lines = []
    for index, dispatch in enumerate(dispatches):
        candidate = _candidate_by_id(item, str(dispatch.get("selected_candidate_id", "")))
        branch = "if" if index == 0 else "else if"
        lines.append(
            f"{branch} ({_cutlass_dispatch_guard(dispatch, candidate, dtype, a_ident, b_ident, m_expr, n_expr, k_expr)}) {{"
        )
        body = _cutlass_launch_lines(
            op_name=op_name,
            symbol=str(dispatch.get("kernel_symbol") or candidate.get("kernel_symbol")),
            candidate=candidate,
            selection=dispatch,
            dtype=dtype,
            a_ident=a_ident,
            b_ident=b_ident,
            epilogue_arg_text=epilogue_arg_text,
            c_ident=c_ident,
            m_expr=m_expr,
            n_expr=n_expr,
            k_expr=k_expr,
            check_alignment=False,
        )
        lines.extend(f"  {line}" for line in body)
        lines.append("}")
    lines.append("else {")
    lines.extend(f"  {line}" for line in default_launch)
    lines.append("}")
    return lines


def _validate_gguf_runtime_dequant_lowering(
    op_name: str,
    spec: Any,
    dtype: str,
    plan: Mapping[str, Any],
) -> None:
    if op_name not in {"gemm_rrr", "gemm_rcr", "gemm_rrr_bias", "gemm_rcr_bias"}:
        raise NotImplementedError(
            f"{op_name} GGUF runtime dequant lowering is not supported; supported ops are gemm_rrr, gemm_rcr, "
            "gemm_rrr_bias, and gemm_rcr_bias"
        )
    epilogue_inputs = tuple(spec.epilogue.inputs)
    if epilogue_inputs not in {(), ("bias",)}:
        raise NotImplementedError(
            f"{op_name} GGUF runtime dequant lowering currently supports only base GEMM or a bias epilogue"
        )
    if dtype not in {"float32", "float16"}:
        raise NotImplementedError(f"{op_name} GGUF runtime dequant lowering supports float32 and float16 outputs only")
    if str(plan.get("status")) != "lowered_runtime_dequant_scratch":
        raise NotImplementedError(
            f"{op_name} GGUF runtime dequant plan has unsupported status {plan.get('status')!r}"
        )
    if plan.get("qtype_value") is None:
        raise ValueError(f"{op_name} GGUF runtime dequant plan is missing qtype_value")
    if plan.get("n_per_row") is None:
        raise ValueError(f"{op_name} GGUF runtime dequant plan is missing n_per_row")


def _gguf_runtime_dequant_lines(
    *,
    op_name: str,
    plan: Mapping[str, Any],
    b_ident: str,
    launch_b_ident: str,
    dtype: str,
    direct_linked: bool,
) -> list[str]:
    output_dtype = 0 if dtype == "float32" else 1
    qtype_value = int(plan["qtype_value"])
    n_per_row = int(plan["n_per_row"])
    scratch_nbytes = int(plan["scratch_nbytes"])
    constant_name = str(plan["constant"])
    dequant_call = (
        "libgguf_cuda_dequantize_rows_on_stream"
        if direct_linked
        else "module->libgguf_cuda_dequantize_rows_on_stream"
    )
    lines = []
    if not direct_linked:
        lines.append(
            f'if (module->libgguf_cuda_dequantize_rows_on_stream == nullptr) return dinoml::module::fail("{op_name} GGUF runtime dequant for constant {constant_name} requires native libgguf CUDA dequant launcher");'
        )
    lines.extend(
        [
        f'if (shape_{b_ident}_1 != {n_per_row}) return dinoml::module::fail("{op_name} GGUF runtime dequant n_per_row mismatch for constant {constant_name}");',
        f'if (session->gguf_dequant_scratch_nbytes < {scratch_nbytes}) return dinoml::module::fail("{op_name} GGUF runtime dequant scratch is too small");',
        f"if (int err = {dequant_call}(module->const_{b_ident}, {qtype_value}, shape_{b_ident}_0, {n_per_row}, {output_dtype}, session->gguf_dequant_scratch, session->stream)) "
        f'return dinoml::module::fail("{op_name} GGUF runtime dequant failed for constant {constant_name}");',
        f"const DinoTensor* abi_{launch_b_ident} = nullptr;",
        f"const {cuda_storage_type(dtype)}* ptr_{launch_b_ident} = static_cast<const {cuda_storage_type(dtype)}*>(session->gguf_dequant_scratch);",
        ]
    )
    return lines


def _cutlass_dispatch_guard(
    selection: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    dtype: str,
    a_ident: str,
    b_ident: str,
    m_expr: str,
    n_expr: str,
    k_expr: str,
) -> str:
    shape = selection.get("shape", {})
    if not isinstance(shape, Mapping):
        return "false"
    conditions = [
        f"({m_expr}) == {int(shape.get('m', 0) or 0)}",
        f"({n_expr}) == {int(shape.get('n', 0) or 0)}",
        f"({k_expr}) == {int(shape.get('k', 0) or 0)}",
    ]
    conditions.extend(_cutlass_runtime_alignment_conditions(_candidate_cutlass_alignment(candidate), dtype, a_ident, b_ident))
    return " && ".join(conditions)


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


def _cutlass_dispatch_selections(item: Mapping[str, Any] | None, node_id: str) -> list[Mapping[str, Any]]:
    if not isinstance(item, Mapping):
        return []
    return [
        selection
        for selection in item.get("execution_plan_dispatch", [])
        if isinstance(selection, Mapping) and str(selection.get("node_id")) == node_id
    ]


def _manifest_kernel_symbol(kernel_manifest: Mapping[str, Any] | None, op_name: str, dtype: str) -> str | None:
    item = _manifest_kernel_item(kernel_manifest, op_name, dtype)
    if item is None:
        return None
    return str(item["kernel_symbol"])


def _cutlass_execution_plan_selection(item: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if item is None:
        return None
    selection = item.get("execution_plan_selection")
    return selection if isinstance(selection, Mapping) else None


def _cutlass_split_k_kernel_symbol(symbol: str) -> str:
    prefix = "dinoml_cutlass_"
    if not symbol.startswith(prefix):
        raise ValueError(f"Unsupported CUTLASS kernel symbol for split-K: {symbol!r}")
    return f"dinoml_cutlass_splitk_{symbol[len(prefix):]}"


def _manifest_kernel_item(
    kernel_manifest: Mapping[str, Any] | None,
    op_name: str,
    dtype: str,
    *,
    node_id: str | None = None,
) -> Mapping[str, Any] | None:
    if kernel_manifest is None:
        return None
    matches: list[Mapping[str, Any]] = []
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("op") != op_name or item.get("kernel_library") != "cutlass_gemm":
            continue
        selected_id = item.get("selected_candidate_id")
        for candidate in item.get("candidates", []):
            if candidate.get("candidate_id") == selected_id and candidate.get("dtype") == dtype:
                matches.append(
                    {
                        **item,
                        "kernel_symbol": str(item.get("kernel_symbol") or candidate.get("kernel_symbol")),
                        "selected_candidate": candidate,
                    }
                )
                break
        else:
            candidate_set = item.get("candidate_set", {})
            if isinstance(candidate_set, Mapping) and candidate_set.get("dtype") == dtype:
                matches.append(item)
    if not matches:
        return None
    if node_id is not None:
        for item in matches:
            runtime_dequant = item.get("gguf_runtime_dequant")
            if isinstance(runtime_dequant, Mapping) and str(runtime_dequant.get("node_id", "")) == node_id:
                return item
        for item in matches:
            if not isinstance(item.get("gguf_runtime_dequant"), Mapping):
                return item
    return matches[0]


def _gguf_runtime_dequant_plan(
    item: Mapping[str, Any] | None,
    *,
    node_id: str,
) -> Mapping[str, Any] | None:
    if not isinstance(item, Mapping):
        return None
    runtime_dequant = item.get("gguf_runtime_dequant")
    if not isinstance(runtime_dequant, Mapping):
        return None
    if str(runtime_dequant.get("node_id", "")) != node_id:
        return None
    return runtime_dequant


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
    if (
        len(input_infos[0]["shape"]) < 2
        or len(input_infos[1]["shape"]) != 2
        or len(c_info["shape"]) != len(input_infos[0]["shape"])
    ):
        raise NotImplementedError(f"{op_name} CUDA lowering expects A[...,K], rank-2 B, and C[...,N]")


def _c_ident(name: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"_{ident}"
    return ident


GEMM_LOWERINGS = {
    op_name: OpLowering(
        op_name=op_name,
        render_generated_kernel=render_generated_kernel,
        render_launch=render_launch,
        source_key=source_key,
        generated_function_name=generated_function_name,
    )
    for op_name in GEMM_OPS
}
