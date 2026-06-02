from __future__ import annotations

from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.ops.definitions import get_op_def


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    if target in {"cuda", "rocm"}:
        return None
    raise ValueError(f"Unsupported FlashAttention lowering target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    if target not in {"cuda", "rocm"}:
        raise ValueError(f"flash_attention lowering is only implemented for CUDA and ROCm, got {target!r}")
    op_name = str(node["op"])
    if op_name == "flash_attention_qkv":
        return _render_qkv_launch(target, node, tensor_map, kernel_manifest)
    if op_name in {"flash_attention_static_kv_cache", "flash_attention_static_kv_cache_bias"}:
        return _render_static_kv_cache_launch(target, node, tensor_map, kernel_manifest)
    if op_name == "flash_attention_bias":
        return _render_bias_launch(target, node, tensor_map, kernel_manifest)
    q_name, k_name, v_name = (str(name) for name in node["inputs"])
    output_name = str(node["outputs"][0])
    q_ident = _c_ident(q_name)
    k_ident = _c_ident(k_name)
    v_ident = _c_ident(v_name)
    output_ident = _c_ident(output_name)
    dtype = str(tensor_map[output_name]["dtype"])
    if dtype not in {"float16", "bfloat16"}:
        raise ValueError(f"flash_attention {target} lowering only supports float16 and bfloat16, got {dtype}")
    causal = node.get("attrs", {}).get("causal", False)
    if not isinstance(causal, bool):
        raise TypeError("flash_attention causal attr must be a bool")
    symbol = _manifest_symbol(kernel_manifest, target, str(node["op"]), dtype)
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
            f'if (shape_{q_ident}_3 <= 0 || shape_{q_ident}_3 > 256) '
            'return dinoml::module::fail("flash_attention unsupported head_dim");',
            f'if (shape_{output_ident}_0 != shape_{q_ident}_0 || shape_{output_ident}_1 != shape_{q_ident}_1 || '
            f'shape_{output_ident}_2 != shape_{q_ident}_2 || shape_{output_ident}_3 != shape_{q_ident}_3) '
            'return dinoml::module::fail("flash_attention output shape mismatch");',
            f"if (int err = {symbol}(ptr_{q_ident}, ptr_{k_ident}, ptr_{v_ident}, ptr_{output_ident}, "
            f"static_cast<int64_t>(shape_{q_ident}_0), static_cast<int64_t>(shape_{q_ident}_1), "
            f"static_cast<int64_t>(shape_{k_ident}_1), static_cast<int64_t>(shape_{q_ident}_2), "
            f"static_cast<int64_t>(shape_{k_ident}_2), static_cast<int64_t>(shape_{q_ident}_3), "
            f'{causal_arg}, session->stream)) return dinoml::module::fail("flash_attention launcher failed");',
        ]
    )


def _render_bias_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None,
) -> str:
    if target != "rocm":
        raise ValueError(f"flash_attention_bias lowering is only implemented for ROCm, got {target!r}")
    q_name, k_name, v_name, bias_name = (str(name) for name in node["inputs"])
    output_name = str(node["outputs"][0])
    q_ident = _c_ident(q_name)
    k_ident = _c_ident(k_name)
    v_ident = _c_ident(v_name)
    bias_ident = _c_ident(bias_name)
    output_ident = _c_ident(output_name)
    dtype = str(tensor_map[output_name]["dtype"])
    if dtype not in {"float16", "bfloat16"}:
        raise ValueError(f"flash_attention_bias {target} lowering only supports float16 and bfloat16, got {dtype}")
    causal = node.get("attrs", {}).get("causal", False)
    if not isinstance(causal, bool):
        raise TypeError("flash_attention_bias causal attr must be a bool")
    symbol = _manifest_symbol(kernel_manifest, target, str(node["op"]), dtype)
    if symbol is None:
        symbol = get_op_def(str(node["op"])).backend_kernels[target].resolve(dtype).symbol
    causal_arg = 1 if causal else 0
    bias_shape = list(tensor_map[bias_name]["shape"])
    if len(bias_shape) == 2:
        bias_batch_dim = "1"
        bias_heads_dim = "1"
        bias_seq_q_dim = f"shape_{bias_ident}_0"
        bias_seq_k_dim = f"shape_{bias_ident}_1"
    elif len(bias_shape) == 3:
        bias_batch_dim = "1"
        bias_heads_dim = f"shape_{bias_ident}_0"
        bias_seq_q_dim = f"shape_{bias_ident}_1"
        bias_seq_k_dim = f"shape_{bias_ident}_2"
    elif len(bias_shape) == 4:
        bias_batch_dim = f"shape_{bias_ident}_0"
        bias_heads_dim = f"shape_{bias_ident}_1"
        bias_seq_q_dim = f"shape_{bias_ident}_2"
        bias_seq_k_dim = f"shape_{bias_ident}_3"
    else:
        raise ValueError("flash_attention_bias bias must have rank 2, 3, or 4")
    return "\n".join(
        [
            f'if (shape_{q_ident}_0 != shape_{k_ident}_0 || shape_{q_ident}_0 != shape_{v_ident}_0) '
            'return dinoml::module::fail("flash_attention_bias batch dimension mismatch");',
            f'if (shape_{k_ident}_0 != shape_{v_ident}_0 || shape_{k_ident}_1 != shape_{v_ident}_1 || '
            f'shape_{k_ident}_2 != shape_{v_ident}_2 || shape_{k_ident}_3 != shape_{v_ident}_3) '
            'return dinoml::module::fail("flash_attention_bias key/value shape mismatch");',
            f'if (shape_{q_ident}_3 != shape_{k_ident}_3) '
            'return dinoml::module::fail("flash_attention_bias head_dim mismatch");',
            f'if ((shape_{q_ident}_2 % shape_{k_ident}_2) != 0) '
            'return dinoml::module::fail("flash_attention_bias head grouping mismatch");',
            f'if (shape_{q_ident}_3 <= 0 || shape_{q_ident}_3 > 256) '
            'return dinoml::module::fail("flash_attention_bias unsupported head_dim");',
            f"if (({bias_batch_dim}) != 1 && ({bias_batch_dim}) != shape_{q_ident}_0) "
            'return dinoml::module::fail("flash_attention_bias bias batch shape mismatch");',
            f"if (({bias_heads_dim}) != 1 && ({bias_heads_dim}) != shape_{q_ident}_2) "
            'return dinoml::module::fail("flash_attention_bias bias head shape mismatch");',
            f"if (({bias_seq_q_dim}) != shape_{q_ident}_1 || ({bias_seq_k_dim}) != shape_{k_ident}_1) "
            'return dinoml::module::fail("flash_attention_bias bias sequence shape mismatch");',
            f'if (shape_{output_ident}_0 != shape_{q_ident}_0 || shape_{output_ident}_1 != shape_{q_ident}_1 || '
            f'shape_{output_ident}_2 != shape_{q_ident}_2 || shape_{output_ident}_3 != shape_{q_ident}_3) '
            'return dinoml::module::fail("flash_attention_bias output shape mismatch");',
            f"if (int err = {symbol}(ptr_{q_ident}, ptr_{k_ident}, ptr_{v_ident}, ptr_{bias_ident}, ptr_{output_ident}, "
            f"static_cast<int64_t>(shape_{q_ident}_0), static_cast<int64_t>(shape_{q_ident}_1), "
            f"static_cast<int64_t>(shape_{k_ident}_1), static_cast<int64_t>(shape_{q_ident}_2), "
            f"static_cast<int64_t>(shape_{k_ident}_2), static_cast<int64_t>(shape_{q_ident}_3), "
            f"static_cast<int64_t>({bias_batch_dim}), static_cast<int64_t>({bias_heads_dim}), "
            f"static_cast<int64_t>({bias_seq_q_dim}), static_cast<int64_t>({bias_seq_k_dim}), "
            f'{causal_arg}, session->stream)) return dinoml::module::fail("flash_attention_bias launcher failed");',
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
    if dtype not in {"float16", "bfloat16"}:
        raise ValueError(f"flash_attention_qkv {target} lowering only supports float16 and bfloat16, got {dtype}")
    causal = node.get("attrs", {}).get("causal", False)
    if not isinstance(causal, bool):
        raise TypeError("flash_attention_qkv causal attr must be a bool")
    symbol = _manifest_symbol(kernel_manifest, target, str(node["op"]), dtype)
    if symbol is None:
        symbol = get_op_def(str(node["op"])).backend_kernels[target].resolve(dtype).symbol
    causal_arg = 1 if causal else 0
    return "\n".join(
        [
            f'if (shape_{qkv_ident}_2 != 3) '
            'return dinoml::module::fail("flash_attention_qkv expected packed axis size 3");',
            f'if (shape_{qkv_ident}_4 <= 0 || shape_{qkv_ident}_4 > 256) '
            'return dinoml::module::fail("flash_attention_qkv unsupported head_dim");',
            f'if (shape_{output_ident}_0 != shape_{qkv_ident}_0 || shape_{output_ident}_1 != shape_{qkv_ident}_1 || '
            f'shape_{output_ident}_2 != shape_{qkv_ident}_3 || shape_{output_ident}_3 != shape_{qkv_ident}_4) '
            'return dinoml::module::fail("flash_attention_qkv output shape mismatch");',
            f"if (int err = {symbol}(ptr_{qkv_ident}, ptr_{output_ident}, "
            f"static_cast<int64_t>(shape_{qkv_ident}_0), static_cast<int64_t>(shape_{qkv_ident}_1), "
            f"static_cast<int64_t>(shape_{qkv_ident}_3), static_cast<int64_t>(shape_{qkv_ident}_4), "
            f'{causal_arg}, session->stream)) return dinoml::module::fail("flash_attention_qkv launcher failed");',
        ]
    )


def _render_static_kv_cache_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None,
) -> str:
    op_name = str(node["op"])
    has_bias = op_name == "flash_attention_static_kv_cache_bias"
    if target not in {"cuda", "rocm"}:
        raise ValueError(f"{op_name} lowering is only implemented for CUDA and ROCm, got {target!r}")
    if has_bias and target != "rocm":
        raise ValueError(f"{op_name} lowering is only implemented for ROCm, got {target!r}")
    if has_bias:
        q_name, past_key_name, past_value_name, new_key_name, new_value_name, cache_seqlens_name, bias_name = (
            str(name) for name in node["inputs"]
        )
    else:
        q_name, past_key_name, past_value_name, new_key_name, new_value_name, cache_seqlens_name = (
            str(name) for name in node["inputs"]
        )
        bias_name = ""
    output_name = str(node["outputs"][0])
    q_ident = _c_ident(q_name)
    past_key_ident = _c_ident(past_key_name)
    past_value_ident = _c_ident(past_value_name)
    new_key_ident = _c_ident(new_key_name)
    new_value_ident = _c_ident(new_value_name)
    cache_seqlens_ident = _c_ident(cache_seqlens_name)
    bias_ident = _c_ident(bias_name) if has_bias else ""
    output_ident = _c_ident(output_name)
    dtype = str(tensor_map[output_name]["dtype"])
    if dtype not in {"float16", "bfloat16"}:
        raise ValueError(f"{op_name} {target} lowering only supports float16 and bfloat16, got {dtype}")
    advance_cache_seqlens = node.get("attrs", {}).get("advance_cache_seqlens", False)
    if not isinstance(advance_cache_seqlens, bool):
        raise TypeError(f"{op_name} advance_cache_seqlens attr must be a bool")
    cache_seqlens_is_state = str(tensor_map[cache_seqlens_name].get("kind", "")) == "state"
    if advance_cache_seqlens and target == "cuda" and not cache_seqlens_is_state:
        raise ValueError(f"{op_name} advance_cache_seqlens on CUDA requires cache_seqlens to be a state tensor")
    symbol = _manifest_symbol(kernel_manifest, target, op_name, dtype)
    if symbol is None:
        symbol = get_op_def(op_name).backend_kernels[target].resolve(dtype).symbol
    rocm_scratch_args = ""
    if target == "rocm":
        rocm_scratch_args = (
            f", {1 if advance_cache_seqlens else 0}, session->flash_attention_static_kv_cache_scratch, "
            "session->flash_attention_static_kv_cache_scratch_nbytes"
        )
    bias_checks: list[str] = []
    bias_call_arg = ""
    bias_dim_args = ""
    if has_bias:
        bias_shape = list(tensor_map[bias_name]["shape"])
        if len(bias_shape) == 2:
            bias_batch_dim = "1"
            bias_heads_dim = "1"
            bias_seq_q_dim = f"shape_{bias_ident}_0"
            bias_seq_k_dim = f"shape_{bias_ident}_1"
        elif len(bias_shape) == 3:
            bias_batch_dim = "1"
            bias_heads_dim = f"shape_{bias_ident}_0"
            bias_seq_q_dim = f"shape_{bias_ident}_1"
            bias_seq_k_dim = f"shape_{bias_ident}_2"
        elif len(bias_shape) == 4:
            bias_batch_dim = f"shape_{bias_ident}_0"
            bias_heads_dim = f"shape_{bias_ident}_1"
            bias_seq_q_dim = f"shape_{bias_ident}_2"
            bias_seq_k_dim = f"shape_{bias_ident}_3"
        else:
            raise ValueError("flash_attention_static_kv_cache_bias bias must have rank 2, 3, or 4")
        bias_checks = [
            f"if (({bias_batch_dim}) != 1 && ({bias_batch_dim}) != shape_{q_ident}_0) "
            f'return dinoml::module::fail("{op_name} bias batch shape mismatch");',
            f"if (({bias_heads_dim}) != 1 && ({bias_heads_dim}) != shape_{q_ident}_2) "
            f'return dinoml::module::fail("{op_name} bias head shape mismatch");',
            f"if (({bias_seq_q_dim}) != shape_{q_ident}_1 || ({bias_seq_k_dim}) != shape_{past_key_ident}_2) "
            f'return dinoml::module::fail("{op_name} bias sequence shape mismatch");',
        ]
        bias_call_arg = f"ptr_{bias_ident}, "
        bias_dim_args = (
            f", static_cast<int64_t>({bias_batch_dim}), static_cast<int64_t>({bias_heads_dim}), "
            f"static_cast<int64_t>({bias_seq_q_dim}), static_cast<int64_t>({bias_seq_k_dim})"
        )
    cuda_cache_seqlens_advance: list[str] = []
    if target == "cuda" and advance_cache_seqlens:
        cuda_cache_seqlens_advance = [
            f"dinoml_cuda_increment_cache_seqlens<<<static_cast<unsigned int>((shape_{cache_seqlens_ident}_0 + 255) / 256), 256, 0, session->stream>>>("
            f"ptr_{cache_seqlens_ident}, static_cast<int64_t>(shape_{cache_seqlens_ident}_0));",
            "if (auto advance_err = cudaGetLastError(); advance_err != cudaSuccess) "
            f'return dinoml::module::fail("{op_name} cache_seqlens advance launch failed");',
        ]
    return "\n".join(
        [
            f'if (shape_{q_ident}_1 != 1) '
            f'return dinoml::module::fail("{op_name} expected q sequence length 1");',
            f'if (shape_{new_key_ident}_2 != 1 || shape_{new_value_ident}_2 != 1) '
            f'return dinoml::module::fail("{op_name} expected new K/V sequence length 1");',
            f'if (shape_{q_ident}_0 != shape_{past_key_ident}_0 || shape_{q_ident}_0 != shape_{past_value_ident}_0 || '
            f'shape_{q_ident}_0 != shape_{new_key_ident}_0 || shape_{q_ident}_0 != shape_{new_value_ident}_0) '
            f'return dinoml::module::fail("{op_name} batch dimension mismatch");',
            f'if (shape_{past_key_ident}_0 != shape_{past_value_ident}_0 || shape_{past_key_ident}_1 != shape_{past_value_ident}_1 || '
            f'shape_{past_key_ident}_2 != shape_{past_value_ident}_2 || shape_{past_key_ident}_3 != shape_{past_value_ident}_3) '
            f'return dinoml::module::fail("{op_name} past key/value shape mismatch");',
            f'if (shape_{new_key_ident}_0 != shape_{new_value_ident}_0 || shape_{new_key_ident}_1 != shape_{new_value_ident}_1 || '
            f'shape_{new_key_ident}_2 != shape_{new_value_ident}_2 || shape_{new_key_ident}_3 != shape_{new_value_ident}_3) '
            f'return dinoml::module::fail("{op_name} new key/value shape mismatch");',
            f'if (shape_{past_key_ident}_1 != shape_{new_key_ident}_1) '
            f'return dinoml::module::fail("{op_name} KV head dimension mismatch");',
            f'if (shape_{q_ident}_3 != shape_{past_key_ident}_3 || shape_{q_ident}_3 != shape_{new_key_ident}_3) '
            f'return dinoml::module::fail("{op_name} head_dim mismatch");',
            f'if ((shape_{q_ident}_2 % shape_{past_key_ident}_1) != 0) '
            f'return dinoml::module::fail("{op_name} head grouping mismatch");',
            f'if (shape_{q_ident}_3 <= 0 || shape_{q_ident}_3 > 256) '
            f'return dinoml::module::fail("{op_name} unsupported head_dim");',
            f'if (shape_{cache_seqlens_ident}_0 != shape_{q_ident}_0) '
            f'return dinoml::module::fail("{op_name} cache_seqlens shape mismatch");',
            *bias_checks,
            f'if (shape_{output_ident}_0 != shape_{q_ident}_0 || shape_{output_ident}_1 != shape_{q_ident}_1 || '
            f'shape_{output_ident}_2 != shape_{q_ident}_2 || shape_{output_ident}_3 != shape_{q_ident}_3) '
            f'return dinoml::module::fail("{op_name} output shape mismatch");',
            f"if (int err = {symbol}(ptr_{q_ident}, ptr_{past_key_ident}, ptr_{past_value_ident}, "
            f"ptr_{new_key_ident}, ptr_{new_value_ident}, ptr_{cache_seqlens_ident}, {bias_call_arg}ptr_{output_ident}, "
            f"static_cast<int64_t>(shape_{q_ident}_0), static_cast<int64_t>(shape_{past_key_ident}_2), "
            f"static_cast<int64_t>(shape_{q_ident}_2), static_cast<int64_t>(shape_{past_key_ident}_1), "
            f"static_cast<int64_t>(shape_{q_ident}_3){bias_dim_args}{rocm_scratch_args}, session->stream)) "
            f'return dinoml::module::fail("{op_name} launcher failed");',
            *cuda_cache_seqlens_advance,
        ]
    )


def _manifest_symbol(kernel_manifest: Mapping[str, Any] | None, target: str, op_name: str, dtype: str) -> str | None:
    if kernel_manifest is None:
        return None
    expected_library = {
        "cuda": "flash_attn_cuda",
        "rocm": "flash_attn_ck",
    }[target]
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("op") == op_name and item.get("dtype") == dtype and item.get("kernel_library") == expected_library:
            return str(item["kernel_symbol"])
    return None


FLASH_ATTENTION_LOWERING = OpLowering(
    op_name="flash_attention",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
)

FLASH_ATTENTION_BIAS_LOWERING = OpLowering(
    op_name="flash_attention_bias",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
)

FLASH_ATTENTION_QKV_LOWERING = OpLowering(
    op_name="flash_attention_qkv",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
)

FLASH_ATTENTION_STATIC_KV_CACHE_LOWERING = OpLowering(
    op_name="flash_attention_static_kv_cache",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
)

FLASH_ATTENTION_STATIC_KV_CACHE_BIAS_LOWERING = OpLowering(
    op_name="flash_attention_static_kv_cache_bias",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
)


def flash_attention_static_kv_cache_scratch_nbytes_for_node(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> int:
    if target != "rocm" or node.get("op") not in {"flash_attention_static_kv_cache", "flash_attention_static_kv_cache_bias"}:
        return 0
    q_name, past_key_name = (str(name) for name in node["inputs"][:2])
    q_tensor = tensor_map[q_name]
    past_key_tensor = tensor_map[past_key_name]
    batch = int(q_tensor["shape"][0])
    seqlen_q = int(q_tensor["shape"][1])
    num_heads_q = int(q_tensor["shape"][2])
    head_dim = int(q_tensor["shape"][3])
    max_cache_len = int(past_key_tensor["shape"][2])
    num_splits = 1
    seqlens_nbytes = _align_nbytes(batch * 4, 16)
    lse_acc_nbytes = _align_nbytes(batch * num_heads_q * num_splits * seqlen_q * 4, 16)
    lse_nbytes = _align_nbytes(batch * num_heads_q * seqlen_q * 4, 16)
    o_acc_nbytes = _align_nbytes(batch * num_heads_q * num_splits * seqlen_q * head_dim * 4, 16)
    # The split-KV API uses runtime max_cache_len for shape checks, but the scratch scales
    # with q/output sizes for num_splits=1. Keep max_cache_len referenced here so invalid
    # cache specs fail early through integer conversion above.
    del max_cache_len
    return seqlens_nbytes + lse_acc_nbytes + lse_nbytes + o_acc_nbytes


def _align_nbytes(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)
