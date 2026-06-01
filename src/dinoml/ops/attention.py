from __future__ import annotations

from typing import Any, Mapping, Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.kernels.providers.ck.flash_attention import (
    flash_attn_ck_qkv_symbol,
    flash_attn_ck_static_kv_cache_symbol,
    flash_attn_ck_symbol,
)
from dinoml.kernels.providers.cuda_flash_attention import (
    flash_attn_cuda_qkv_symbol,
    flash_attn_cuda_static_kv_cache_symbol,
    flash_attn_cuda_symbol,
)
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, KernelVariant, OpDef, OpSchema, op_def


FLASH_ATTENTION_DTYPES = ("float16", "bfloat16")
QKV_SPLIT_DTYPES = ("float16", "float32", "bfloat16")


def flash_attention(q: object, k: object, v: object, *, causal: bool = False) -> Tensor:
    q_tensor = as_tensor(q, dtype_hint=k.dtype if isinstance(k, Tensor) else "float32")
    k_tensor = as_tensor(k, dtype_hint=q_tensor.dtype)
    v_tensor = as_tensor(v, dtype_hint=q_tensor.dtype)
    tensors = (q_tensor, k_tensor, v_tensor)
    for tensor in tensors[1:]:
        if q_tensor.builder is not tensor.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if q_tensor.dtype != tensor.dtype:
            raise ValueError(f"flash_attention dtype mismatch: {q_tensor.dtype} vs {tensor.dtype}")
    if q_tensor.dtype not in FLASH_ATTENTION_DTYPES:
        supported = ", ".join(FLASH_ATTENTION_DTYPES)
        raise ValueError(f"flash_attention supports {supported}, got {q_tensor.dtype}")
    _validate_flash_attention_shapes([tensor.shape for tensor in tensors])
    return q_tensor.builder.emit(
        "flash_attention",
        tensors,
        q_tensor.shape,
        q_tensor.dtype,
        {"causal": bool(causal)},
        shape_spec=q_tensor.shape_spec,
    )


def flash_attention_qkv(qkv: object, *, causal: bool = False) -> Tensor:
    qkv_tensor = as_tensor(qkv)
    if qkv_tensor.dtype not in FLASH_ATTENTION_DTYPES:
        supported = ", ".join(FLASH_ATTENTION_DTYPES)
        raise ValueError(f"flash_attention_qkv supports {supported}, got {qkv_tensor.dtype}")
    _validate_flash_attention_qkv_shape(qkv_tensor.shape)
    return qkv_tensor.builder.emit(
        "flash_attention_qkv",
        [qkv_tensor],
        [qkv_tensor.shape[0], qkv_tensor.shape[1], qkv_tensor.shape[3], qkv_tensor.shape[4]],
        qkv_tensor.dtype,
        {"causal": bool(causal)},
        shape_spec=[qkv_tensor.shape_spec[0], qkv_tensor.shape_spec[1], qkv_tensor.shape_spec[3], qkv_tensor.shape_spec[4]],
    )


def flash_attention_static_kv_cache(
    q: object,
    past_key: object,
    past_value: object,
    new_key: object,
    new_value: object,
    cache_seqlens: object,
) -> Tensor:
    q_tensor = as_tensor(q, dtype_hint="float32")
    tensors = (
        q_tensor,
        as_tensor(past_key, dtype_hint=q_tensor.dtype),
        as_tensor(past_value, dtype_hint=q_tensor.dtype),
        as_tensor(new_key, dtype_hint=q_tensor.dtype),
        as_tensor(new_value, dtype_hint=q_tensor.dtype),
        as_tensor(cache_seqlens, dtype_hint="int32"),
    )
    for tensor in tensors[1:]:
        if q_tensor.builder is not tensor.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
    if q_tensor.dtype not in FLASH_ATTENTION_DTYPES:
        supported = ", ".join(FLASH_ATTENTION_DTYPES)
        raise ValueError(f"flash_attention_static_kv_cache supports {supported}, got {q_tensor.dtype}")
    for tensor in tensors[1:5]:
        if tensor.dtype != q_tensor.dtype:
            raise ValueError(f"flash_attention_static_kv_cache dtype mismatch: {q_tensor.dtype} vs {tensor.dtype}")
    if tensors[5].dtype != "int32":
        raise ValueError(f"flash_attention_static_kv_cache cache_seqlens must be int32, got {tensors[5].dtype}")
    _validate_flash_attention_static_kv_cache_shapes([tensor.shape for tensor in tensors])
    return q_tensor.builder.emit(
        "flash_attention_static_kv_cache",
        tensors,
        q_tensor.shape,
        q_tensor.dtype,
        {},
        shape_spec=q_tensor.shape_spec,
    )


def qkv_split(qkv: object) -> tuple[Tensor, Tensor, Tensor]:
    qkv_tensor = as_tensor(qkv)
    if qkv_tensor.dtype not in QKV_SPLIT_DTYPES:
        supported = ", ".join(QKV_SPLIT_DTYPES)
        raise ValueError(f"qkv_split supports {supported}, got {qkv_tensor.dtype}")
    _validate_qkv_split_shape(qkv_tensor.shape)
    out_shape = [*qkv_tensor.shape[:-1], qkv_tensor.shape[-1] // 3]
    out_shape_spec = [*qkv_tensor.shape_spec[:-1], qkv_tensor.shape[-1] // 3]
    return qkv_tensor.builder.emit_multi(
        "qkv_split",
        [qkv_tensor],
        [
            (out_shape, qkv_tensor.dtype, out_shape_spec),
            (out_shape, qkv_tensor.dtype, out_shape_spec),
            (out_shape, qkv_tensor.dtype, out_shape_spec),
        ],
        {},
    )


def _infer_flash_attention_shape(shapes: Sequence[Sequence[int]]) -> list[int]:
    _validate_flash_attention_shapes(shapes)
    return list(shapes[0])


def _infer_flash_attention_qkv_shape(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 1:
        raise ValueError(f"flash_attention_qkv expects 1 input, got {len(shapes)}")
    _validate_flash_attention_qkv_shape(shapes[0])
    qkv_shape = list(shapes[0])
    return [qkv_shape[0], qkv_shape[1], qkv_shape[3], qkv_shape[4]]


def _infer_flash_attention_static_kv_cache_shape(shapes: Sequence[Sequence[int]]) -> list[int]:
    _validate_flash_attention_static_kv_cache_shapes(shapes)
    return list(shapes[0])


def _validate_flash_attention_shapes(shapes: Sequence[Sequence[int]]) -> None:
    if len(shapes) != 3:
        raise ValueError(f"flash_attention expects 3 inputs, got {len(shapes)}")
    q_shape, k_shape, v_shape = [list(shape) for shape in shapes]
    if any(len(shape) != 4 for shape in (q_shape, k_shape, v_shape)):
        raise ValueError("flash_attention expects q, k, and v with shape [batch, seq, heads, head_dim]")
    if q_shape[0] != k_shape[0] or q_shape[0] != v_shape[0]:
        raise ValueError("flash_attention batch dimension mismatch")
    if k_shape != v_shape:
        raise ValueError("flash_attention key/value shape mismatch")
    if q_shape[3] != k_shape[3]:
        raise ValueError("flash_attention head_dim mismatch")
    if q_shape[2] % k_shape[2] != 0:
        raise ValueError("flash_attention num_heads_q must be divisible by num_heads_k")


def _validate_flash_attention_static_kv_cache_shapes(shapes: Sequence[Sequence[int]]) -> None:
    if len(shapes) != 6:
        raise ValueError(f"flash_attention_static_kv_cache expects 6 inputs, got {len(shapes)}")
    q_shape, past_key_shape, past_value_shape, new_key_shape, new_value_shape, cache_seqlens_shape = [
        list(shape) for shape in shapes
    ]
    if len(q_shape) != 4:
        raise ValueError("flash_attention_static_kv_cache expects q with shape [batch, 1, heads_q, head_dim]")
    if q_shape[1] != 1:
        raise ValueError("flash_attention_static_kv_cache currently expects q sequence length 1")
    if any(len(shape) != 4 for shape in (past_key_shape, past_value_shape, new_key_shape, new_value_shape)):
        raise ValueError(
            "flash_attention_static_kv_cache expects past/new K/V with shape [batch, heads_kv, seq, head_dim]"
        )
    if past_key_shape != past_value_shape:
        raise ValueError("flash_attention_static_kv_cache past key/value shape mismatch")
    if new_key_shape != new_value_shape:
        raise ValueError("flash_attention_static_kv_cache new key/value shape mismatch")
    if q_shape[0] != past_key_shape[0] or q_shape[0] != new_key_shape[0]:
        raise ValueError("flash_attention_static_kv_cache batch dimension mismatch")
    if past_key_shape[1] != new_key_shape[1]:
        raise ValueError("flash_attention_static_kv_cache KV head dimension mismatch")
    if new_key_shape[2] != 1:
        raise ValueError("flash_attention_static_kv_cache expects new K/V sequence length 1")
    if q_shape[3] != past_key_shape[3] or q_shape[3] != new_key_shape[3]:
        raise ValueError("flash_attention_static_kv_cache head_dim mismatch")
    if q_shape[2] % past_key_shape[1] != 0:
        raise ValueError("flash_attention_static_kv_cache num_heads_q must be divisible by num_heads_kv")
    if cache_seqlens_shape != [q_shape[0]]:
        raise ValueError("flash_attention_static_kv_cache cache_seqlens must have shape [batch]")


def _validate_flash_attention_qkv_shape(shape: Sequence[int]) -> None:
    qkv_shape = list(shape)
    if len(qkv_shape) != 5:
        raise ValueError("flash_attention_qkv expects qkv with shape [batch, seq, 3, heads, head_dim]")
    if qkv_shape[2] != 3:
        raise ValueError("flash_attention_qkv expects axis 2 to have size 3")


def _validate_qkv_split_shape(shape: Sequence[int]) -> None:
    qkv_shape = list(shape)
    if len(qkv_shape) < 1:
        raise ValueError("qkv_split expects rank >= 1 input")
    if not isinstance(qkv_shape[-1], int) or qkv_shape[-1] % 3 != 0:
        raise ValueError("qkv_split expects a static last dimension divisible by 3")


def _infer_qkv_split_shape(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 1:
        raise ValueError(f"qkv_split expects 1 input, got {len(shapes)}")
    _validate_qkv_split_shape(shapes[0])
    shape = list(shapes[0])
    shape[-1] //= 3
    return shape


def _infer_flash_attention_shape_with_attrs(
    shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    causal = attrs.get("causal", False)
    if not isinstance(causal, bool):
        raise TypeError("flash_attention causal attr must be a bool")
    return _infer_flash_attention_shape(shapes)


def _infer_flash_attention_qkv_shape_with_attrs(
    shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    causal = attrs.get("causal", False)
    if not isinstance(causal, bool):
        raise TypeError("flash_attention_qkv causal attr must be a bool")
    return _infer_flash_attention_qkv_shape(shapes)


@op_def
class FlashAttention(OpDef):
    name = "flash_attention"
    schema = OpSchema(
        inputs=("q", "k", "v"),
        attrs=(AttrDef("causal", "bool", default=False),),
    )
    infer_shape = _infer_flash_attention_shape
    infer_shape_with_attrs = _infer_flash_attention_shape_with_attrs
    allowed_dtypes = FLASH_ATTENTION_DTYPES
    backend_kernels = {
        "cuda": KernelBinding(
            flash_attn_cuda_symbol("float16"),
            "flash_attn_cuda",
            dtype_variants={
                "float16": KernelVariant(flash_attn_cuda_symbol("float16")),
                "bfloat16": KernelVariant(flash_attn_cuda_symbol("bfloat16")),
            },
        ),
        "rocm": KernelBinding(
            flash_attn_ck_symbol("float16"),
            "flash_attn_ck",
            dtype_variants={
                "float16": KernelVariant(flash_attn_ck_symbol("float16")),
                "bfloat16": KernelVariant(flash_attn_ck_symbol("bfloat16")),
            },
        )
    }
    frontend = FrontendBinding("flash_attention")
    description = "CK FlashAttention forward op for contiguous [batch, seq, heads, head_dim] tensors."


@op_def
class FlashAttentionQKV(OpDef):
    name = "flash_attention_qkv"
    schema = OpSchema(
        inputs=("qkv",),
        attrs=(AttrDef("causal", "bool", default=False),),
    )
    infer_shape = _infer_flash_attention_qkv_shape
    infer_shape_with_attrs = _infer_flash_attention_qkv_shape_with_attrs
    allowed_dtypes = FLASH_ATTENTION_DTYPES
    backend_kernels = {
        "cuda": KernelBinding(
            flash_attn_cuda_qkv_symbol("float16"),
            "flash_attn_cuda",
            dtype_variants={
                "float16": KernelVariant(flash_attn_cuda_qkv_symbol("float16")),
                "bfloat16": KernelVariant(flash_attn_cuda_qkv_symbol("bfloat16")),
            },
        ),
        "rocm": KernelBinding(
            flash_attn_ck_qkv_symbol("float16"),
            "flash_attn_ck",
            dtype_variants={
                "float16": KernelVariant(flash_attn_ck_qkv_symbol("float16")),
                "bfloat16": KernelVariant(flash_attn_ck_qkv_symbol("bfloat16")),
            },
        )
    }
    frontend = FrontendBinding("flash_attention_qkv")
    description = "CK FlashAttention forward op for packed [batch, seq, 3, heads, head_dim] QKV tensors."


@op_def
class FlashAttentionStaticKvCache(OpDef):
    name = "flash_attention_static_kv_cache"
    schema = OpSchema(inputs=("q", "past_key", "past_value", "new_key", "new_value", "cache_seqlens"))
    infer_shape = _infer_flash_attention_static_kv_cache_shape
    allowed_dtypes = FLASH_ATTENTION_DTYPES
    backend_kernels = {
        "cuda": KernelBinding(
            flash_attn_cuda_static_kv_cache_symbol("float16"),
            "flash_attn_cuda",
            dtype_variants={
                "float16": KernelVariant(flash_attn_cuda_static_kv_cache_symbol("float16")),
                "bfloat16": KernelVariant(flash_attn_cuda_static_kv_cache_symbol("bfloat16")),
            },
        ),
        "rocm": KernelBinding(
            flash_attn_ck_static_kv_cache_symbol("float16"),
            "flash_attn_ck",
            dtype_variants={
                "float16": KernelVariant(flash_attn_ck_static_kv_cache_symbol("float16")),
                "bfloat16": KernelVariant(flash_attn_ck_static_kv_cache_symbol("bfloat16")),
            },
        ),
    }
    frontend = FrontendBinding("flash_attention_static_kv_cache")
    description = "FlashAttention decode op for static [batch, kv_heads, max_cache_len, head_dim] KV caches."


@op_def
class QKVSplit(OpDef):
    name = "qkv_split"
    schema = OpSchema(inputs=("qkv",))
    infer_shape = _infer_qkv_split_shape
    allowed_dtypes = QKV_SPLIT_DTYPES
    backend_kernels = {
        "cuda": KernelBinding("generated_qkv_split", "model", source_template="qkv_split_gpu"),
        "rocm": KernelBinding("generated_qkv_split", "model", source_template="qkv_split_gpu"),
        "cpu": KernelBinding("generated_qkv_split", "model", source_template="qkv_split_cpu"),
    }
    frontend = FrontendBinding("qkv_split")
    description = "Split a packed [..., 3 * hidden] QKV tensor into three dense [..., hidden] tensors."
