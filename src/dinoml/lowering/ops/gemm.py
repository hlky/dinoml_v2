from __future__ import annotations

import re
from typing import Any, Mapping

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
    dtype = str(tensor_map[c_name]["dtype"])
    symbol = _manifest_kernel_symbol(kernel_manifest, op_name, dtype) or get_op_def(op_name).backend_kernels[target].resolve(dtype).symbol

    m_expr = f"shape_{a_ident}_0"
    k_expr = f"shape_{a_ident}_1"
    if spec.base_layout == "rrr":
        n_expr = f"shape_{b_ident}_1"
        k_check = f"shape_{a_ident}_1 != shape_{b_ident}_0"
        output_check = f"shape_{c_ident}_0 != shape_{a_ident}_0 || shape_{c_ident}_1 != shape_{b_ident}_1"
    else:
        n_expr = f"shape_{b_ident}_0"
        k_check = f"shape_{a_ident}_1 != shape_{b_ident}_1"
        output_check = f"shape_{c_ident}_0 != shape_{a_ident}_0 || shape_{c_ident}_1 != shape_{b_ident}_0"
    bias_check = None
    bias_arg = ""
    if spec.epilogue.has_bias:
        bias_name = str(node["inputs"][2])
        bias_ident = _c_ident(bias_name)
        bias_rank = len(tensor_map[bias_name]["shape"])
        if bias_rank == 1:
            bias_check = f"shape_{bias_ident}_0 != {n_expr}"
        elif bias_rank == 2:
            bias_check = f"shape_{bias_ident}_0 != 1 || shape_{bias_ident}_1 != {n_expr}"
        else:
            raise NotImplementedError(f"{op_name} CUDA lowering supports rank-1 or rank-2 bias only")
        bias_arg = f"ptr_{bias_ident}, "

    lines = [
        f'if ({k_check}) return dinoml::module::fail("{op_name} K dimension mismatch");',
        f'if ({output_check}) return dinoml::module::fail("{op_name} output shape mismatch");',
    ]
    if bias_check is not None:
        lines.append(f'if ({bias_check}) return dinoml::module::fail("{op_name} bias shape mismatch");')
    lines.append(
        f"if (int err = {symbol}(ptr_{a_ident}, ptr_{b_ident}, {bias_arg}ptr_{c_ident}, "
        f"static_cast<int>({m_expr}), static_cast<int>({n_expr}), static_cast<int>({k_expr}), session->stream)) "
        f'return dinoml::module::fail("{op_name} CUTLASS launcher failed");'
    )
    return "\n".join(lines)


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> None:
    del target, node, tensor_map
    return None


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> None:
    del target, node, tensor_map
    return None


def _manifest_kernel_symbol(kernel_manifest: Mapping[str, Any] | None, op_name: str, dtype: str) -> str | None:
    if kernel_manifest is None:
        return None
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("op") != op_name or item.get("kernel_library") != "cutlass_gemm":
            continue
        selected_id = item.get("selected_candidate_id")
        for candidate in item.get("candidates", []):
            if candidate.get("candidate_id") == selected_id and candidate.get("dtype") == dtype:
                return str(item.get("kernel_symbol") or candidate.get("kernel_symbol"))
        candidate_set = item.get("candidate_set", {})
        if isinstance(candidate_set, Mapping) and candidate_set.get("dtype") == dtype:
            return str(item["kernel_symbol"])
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
    if len(input_infos[0]["shape"]) != 2 or len(input_infos[1]["shape"]) != 2 or len(c_info["shape"]) != 2:
        raise NotImplementedError(f"{op_name} CUDA lowering currently supports rank-2 tensors only")


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
