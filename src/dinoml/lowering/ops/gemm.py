from __future__ import annotations

import re
from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> None:
    del target, node, tensor_map
    return None


def render_launch(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    if target != "cuda":
        raise ValueError(f"{node['op']} lowering is currently CUDA-only")
    op_name = str(node["op"])
    if op_name not in {"gemm_rrr", "gemm_rcr"}:
        raise ValueError(f"Unsupported GEMM op: {op_name}")
    a_name, b_name = node["inputs"]
    c_name = node["outputs"][0]
    a_ident = _c_ident(a_name)
    b_ident = _c_ident(b_name)
    c_ident = _c_ident(c_name)
    _validate_static_contract(op_name, tensor_map[a_name], tensor_map[b_name], tensor_map[c_name])

    m_expr = f"shape_{a_ident}_0"
    k_expr = f"shape_{a_ident}_1"
    if op_name == "gemm_rrr":
        n_expr = f"shape_{b_ident}_1"
        k_check = f"shape_{a_ident}_1 != shape_{b_ident}_0"
        output_check = f"shape_{c_ident}_0 != shape_{a_ident}_0 || shape_{c_ident}_1 != shape_{b_ident}_1"
        symbol = "dinoml_cutlass_gemm_rrr_f32"
    else:
        n_expr = f"shape_{b_ident}_0"
        k_check = f"shape_{a_ident}_1 != shape_{b_ident}_1"
        output_check = f"shape_{c_ident}_0 != shape_{a_ident}_0 || shape_{c_ident}_1 != shape_{b_ident}_0"
        symbol = "dinoml_cutlass_gemm_rcr_f32"

    return "\n".join(
        [
            f'if ({k_check}) return dinoml::module::fail("{op_name} K dimension mismatch");',
            f'if ({output_check}) return dinoml::module::fail("{op_name} output shape mismatch");',
            (
                f"if (int err = {symbol}(ptr_{a_ident}, ptr_{b_ident}, ptr_{c_ident}, "
                f"static_cast<int>({m_expr}), static_cast<int>({n_expr}), static_cast<int>({k_expr}), session->stream)) "
                f'return dinoml::module::fail("{op_name} CUTLASS launcher failed");'
            ),
        ]
    )


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> None:
    del target, node, tensor_map
    return None


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> None:
    del target, node, tensor_map
    return None


def _validate_static_contract(
    op_name: str,
    a_info: Mapping[str, Any],
    b_info: Mapping[str, Any],
    c_info: Mapping[str, Any],
) -> None:
    if a_info["dtype"] != "float32" or b_info["dtype"] != "float32" or c_info["dtype"] != "float32":
        raise NotImplementedError(f"{op_name} CUDA lowering currently supports only float32")
    if len(a_info["shape"]) != 2 or len(b_info["shape"]) != 2 or len(c_info["shape"]) != 2:
        raise NotImplementedError(f"{op_name} CUDA lowering currently supports rank-2 tensors only")


def _c_ident(name: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"_{ident}"
    return ident


GEMM_LOWERINGS = {
    "gemm_rrr": OpLowering(
        op_name="gemm_rrr",
        render_generated_kernel=render_generated_kernel,
        render_launch=render_launch,
        source_key=source_key,
        generated_function_name=generated_function_name,
    ),
    "gemm_rcr": OpLowering(
        op_name="gemm_rcr",
        render_generated_kernel=render_generated_kernel,
        render_launch=render_launch,
        source_key=source_key,
        generated_function_name=generated_function_name,
    ),
}
