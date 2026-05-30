from __future__ import annotations

from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.ops.definitions import get_op_def


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    if target == "rocm":
        return None
    raise ValueError(f"Unsupported FlashAttention lowering target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    if target != "rocm":
        raise ValueError(f"flash_attention lowering is only implemented for ROCm, got {target!r}")
    if str(node["op"]) == "flash_attention_qkv":
        return _render_qkv_launch(target, node, tensor_map, kernel_manifest)
    q_name, k_name, v_name = (str(name) for name in node["inputs"])
    output_name = str(node["outputs"][0])
    q_ident = _c_ident(q_name)
    k_ident = _c_ident(k_name)
    v_ident = _c_ident(v_name)
    output_ident = _c_ident(output_name)
    dtype = str(tensor_map[output_name]["dtype"])
    if dtype != "float16":
        raise ValueError(f"flash_attention ROCm lowering only supports float16, got {dtype}")
    causal = node.get("attrs", {}).get("causal", False)
    if not isinstance(causal, bool):
        raise TypeError("flash_attention causal attr must be a bool")
    symbol = _manifest_symbol(kernel_manifest, str(node["op"]), dtype)
    if symbol is None:
        symbol = get_op_def(str(node["op"])).backend_kernels[target].resolve(dtype).symbol
    causal_arg = 1 if causal else 0
    return "\n".join(
        [
            f'if (shape_{q_ident}_0 != shape_{k_ident}_0 || shape_{q_ident}_0 != shape_{v_ident}_0) '
            'return dinoml::module::fail("flash_attention batch dimension mismatch");',
            f'if (shape_{k_ident}_0 != shape_{v_ident}_0 || shape_{k_ident}_1 != shape_{v_ident}_1 || '
            f'shape_{k_ident}_2 != shape_{v_ident}_2 || shape_{k_ident}_3 != shape_{v_ident}_3) '
            'return dinoml::module::fail("flash_attention key/value shape mismatch");',
            f'if (shape_{q_ident}_3 != shape_{k_ident}_3) '
            'return dinoml::module::fail("flash_attention head_dim mismatch");',
            f'if ((shape_{q_ident}_2 % shape_{k_ident}_2) != 0) '
            'return dinoml::module::fail("flash_attention head grouping mismatch");',
            f'if (shape_{q_ident}_3 != 64) '
            'return dinoml::module::fail("flash_attention unsupported head_dim");',
            f'if (shape_{output_ident}_0 != shape_{q_ident}_0 || shape_{output_ident}_1 != shape_{q_ident}_1 || '
            f'shape_{output_ident}_2 != shape_{q_ident}_2 || shape_{output_ident}_3 != shape_{q_ident}_3) '
            'return dinoml::module::fail("flash_attention output shape mismatch");',
            f"if (int err = {symbol}(ptr_{q_ident}, ptr_{k_ident}, ptr_{v_ident}, ptr_{output_ident}, "
            f"static_cast<int64_t>(shape_{q_ident}_0), static_cast<int64_t>(shape_{q_ident}_1), "
            f"static_cast<int64_t>(shape_{k_ident}_1), static_cast<int64_t>(shape_{q_ident}_2), "
            f"static_cast<int64_t>(shape_{k_ident}_2), static_cast<int64_t>(shape_{q_ident}_3), "
            f'{causal_arg}, session->stream)) return dinoml::module::fail("flash_attention CK launcher failed");',
        ]
    )


def _render_qkv_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None,
) -> str:
    qkv_name = str(node["inputs"][0])
    output_name = str(node["outputs"][0])
    qkv_ident = _c_ident(qkv_name)
    output_ident = _c_ident(output_name)
    dtype = str(tensor_map[output_name]["dtype"])
    if dtype != "float16":
        raise ValueError(f"flash_attention_qkv ROCm lowering only supports float16, got {dtype}")
    causal = node.get("attrs", {}).get("causal", False)
    if not isinstance(causal, bool):
        raise TypeError("flash_attention_qkv causal attr must be a bool")
    symbol = _manifest_symbol(kernel_manifest, str(node["op"]), dtype)
    if symbol is None:
        symbol = get_op_def(str(node["op"])).backend_kernels[target].resolve(dtype).symbol
    causal_arg = 1 if causal else 0
    return "\n".join(
        [
            f'if (shape_{qkv_ident}_2 != 3) '
            'return dinoml::module::fail("flash_attention_qkv expected packed axis size 3");',
            f'if (shape_{qkv_ident}_4 != 64) '
            'return dinoml::module::fail("flash_attention_qkv unsupported head_dim");',
            f'if (shape_{output_ident}_0 != shape_{qkv_ident}_0 || shape_{output_ident}_1 != shape_{qkv_ident}_1 || '
            f'shape_{output_ident}_2 != shape_{qkv_ident}_3 || shape_{output_ident}_3 != shape_{qkv_ident}_4) '
            'return dinoml::module::fail("flash_attention_qkv output shape mismatch");',
            f"if (int err = {symbol}(ptr_{qkv_ident}, ptr_{output_ident}, "
            f"static_cast<int64_t>(shape_{qkv_ident}_0), static_cast<int64_t>(shape_{qkv_ident}_1), "
            f"static_cast<int64_t>(shape_{qkv_ident}_3), static_cast<int64_t>(shape_{qkv_ident}_4), "
            f'{causal_arg}, session->stream)) return dinoml::module::fail("flash_attention_qkv CK launcher failed");',
        ]
    )


def _manifest_symbol(kernel_manifest: Mapping[str, Any] | None, op_name: str, dtype: str) -> str | None:
    if kernel_manifest is None:
        return None
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("op") == op_name and item.get("dtype") == dtype and item.get("kernel_library") == "flash_attn_ck":
            return str(item["kernel_symbol"])
    return None


FLASH_ATTENTION_LOWERING = OpLowering(
    op_name="flash_attention",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
)

FLASH_ATTENTION_QKV_LOWERING = OpLowering(
    op_name="flash_attention_qkv",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
)
