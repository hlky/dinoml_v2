from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from dinoml.ir import dtype_nbytes
from dinoml.lowering.ops.base import OpLowering
from dinoml.kernels.gemm import GEMM_OPS, gemm_op_spec
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
    manifest_item = _manifest_kernel_item(kernel_manifest, op_name, dtype)
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
    _validate_cutlass_execution_plan_selection(op_name, manifest_item, selection)
    split_k = int(selection.get("split_k", 1) or 1) if selection is not None else 1
    if split_k > 1:
        symbol = _cutlass_split_k_kernel_symbol(symbol)
        launch_tail = f", {split_k}, session->cutlass_workspace, session->cutlass_workspace_nbytes, session->stream"
    else:
        launch_tail = ", session->stream"
    alignment_checks = _cutlass_runtime_alignment_checks(
        op_name,
        manifest_item,
        dtype,
        a_ident,
        b_ident,
    )
    lines = [
        f'if ({k_check}) return dinoml::module::fail("{op_name} K dimension mismatch");',
        f'if ({output_check}) return dinoml::module::fail("{op_name} output shape mismatch");',
    ]
    lines.extend(alignment_checks)
    lines.extend(epilogue_checks)
    epilogue_arg_text = "".join(f"{arg}, " for arg in epilogue_args)
    lines.append(
        f"if (int err = {symbol}(ptr_{a_ident}, ptr_{b_ident}, {epilogue_arg_text}ptr_{c_ident}, "
        f"static_cast<int>({m_expr}), static_cast<int>({n_expr}), static_cast<int>({k_expr}){launch_tail})) "
        f'return dinoml::module::fail("{op_name} CUTLASS launcher failed");'
    )
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
    item: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None,
) -> None:
    if item is None or selection is None:
        return
    split_k = int(selection.get("split_k", 1) or 1)
    if split_k <= 1:
        return
    launch_abi = _manifest_launch_abi(item)
    if launch_abi not in {"dinoml_cutlass_gemm_v1", "dinoml_cutlass_gemm_bias_v1"}:
        raise NotImplementedError(
            f"{op_name} CUDA lowering does not support CUTLASS split-K > 1 for launch ABI {launch_abi}; "
            f"execution plan requested split_k={split_k}"
        )
    workspace_nbytes = int(selection.get("workspace_nbytes", 0) or 0)
    if workspace_nbytes <= 0:
        raise ValueError(f"{op_name} CUTLASS split-K execution plan requires workspace_nbytes > 0")


def _cutlass_runtime_alignment_checks(
    op_name: str,
    item: Mapping[str, Any] | None,
    dtype: str,
    a_ident: str,
    b_ident: str,
) -> list[str]:
    align = _selected_cutlass_alignment(item)
    if align <= 1:
        return []
    byte_alignment = align * dtype_nbytes(dtype)
    return [
        f'if (int err = dinoml::module::check_pointer_alignment(ptr_{a_ident}, "{op_name} A", {byte_alignment})) '
        "return err;",
        f'if (int err = dinoml::module::check_pointer_alignment(ptr_{b_ident}, "{op_name} B", {byte_alignment})) '
        "return err;",
    ]


def _selected_cutlass_alignment(item: Mapping[str, Any] | None) -> int:
    if item is None:
        return 1
    selected = item.get("selected_candidate")
    if not isinstance(selected, Mapping):
        return 1
    cutlass_config = selected.get("cutlass")
    if not isinstance(cutlass_config, Mapping):
        return 1
    return int(cutlass_config.get("align", 1) or 1)


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


def _manifest_launch_abi(item: Mapping[str, Any]) -> str:
    selected = item.get("selected_candidate")
    if isinstance(selected, Mapping) and selected.get("launch_abi"):
        return str(selected["launch_abi"])
    candidate_set = item.get("candidate_set", {})
    if isinstance(candidate_set, Mapping) and candidate_set.get("launch_abi"):
        return str(candidate_set["launch_abi"])
    return ""


def _cutlass_split_k_kernel_symbol(symbol: str) -> str:
    prefix = "dinoml_cutlass_"
    if not symbol.startswith(prefix):
        raise ValueError(f"Unsupported CUTLASS kernel symbol for split-K: {symbol!r}")
    return f"dinoml_cutlass_splitk_{symbol[len(prefix):]}"


def _manifest_kernel_item(kernel_manifest: Mapping[str, Any] | None, op_name: str, dtype: str) -> Mapping[str, Any] | None:
    if kernel_manifest is None:
        return None
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("op") != op_name or item.get("kernel_library") != "cutlass_gemm":
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
