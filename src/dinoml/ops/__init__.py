from __future__ import annotations

from typing import Any, Mapping

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.bmm import (
    bmm,
    bmm_ccc,
    bmm_ccc_add,
    bmm_ccr,
    bmm_ccr_add,
    bmm_crc,
    bmm_crc_add,
    bmm_crr,
    bmm_crr_add,
    bmm_rcc,
    bmm_rcc_add,
    bmm_rcr,
    bmm_rcr_add,
    bmm_rrc,
    bmm_rrc_add,
    bmm_rrr,
    bmm_rrr_add,
    bmm_xxx,
    bmm_xxx_add,
)
from dinoml.ops.attention import (
    flash_attention,
    flash_attention_bias,
    flash_attention_qkv,
    flash_attention_static_kv_cache,
    flash_attention_static_kv_cache_bias,
    flash_attention_varlen,
    qkv_split,
)
from dinoml.ops.broadcasting import expand, expand_static_shape, meshgrid
from dinoml.ops.cast import cast
from dinoml.ops.collections import (
    batch_gather,
    chunk,
    concatenate,
    concatenate_fast,
    concatenate_tanh,
    dynamic_slice,
    flip,
    gather,
    index_select,
    pad,
    pad_last_dim,
    permute,
    permute021,
    permute0213,
    permute102,
    permute210,
    pixel_shuffle,
    pixel_unshuffle,
    repeat_interleave,
    runtime_index_select,
    slice_reshape_scatter,
    slice_scatter,
    split,
    stack,
    transpose,
)
from dinoml.ops.conv import (
    conv1d_bias,
    conv1d_bias_add,
    conv1d_bias_add_relu,
    conv1d_bias_relu,
    conv2d,
    conv2d_bias,
    conv2d_bias_add,
    conv2d_bias_add_relu,
    conv2d_bias_relu,
    transposed_conv2d,
    transposed_conv2d_bias,
    transposed_conv2d_bias_add,
    transposed_conv2d_bias_add_relu,
    transposed_conv2d_bias_relu,
)
from dinoml.ops.creation import arange, full, randn
from dinoml.ops.elementwise import (
    abs,
    add,
    celu,
    clamp,
    clamp_nan_to_num,
    cos,
    div,
    elu,
    eq,
    exp,
    fast_gelu,
    floor,
    floor_div,
    ge,
    gelu,
    gelu_new,
    gt,
    hardtanh,
    le,
    leaky_relu,
    log,
    log1p,
    lt,
    max,
    min,
    mul,
    nan_to_num,
    ne,
    pow,
    relu,
    sigmoid,
    sign,
    silu,
    sin,
    softplus,
    softsign,
    sqrt,
    sub,
    tanh,
)
from dinoml.ops.embedding import embedding
from dinoml.ops.glm_ocr import glm_ocr_stitch_image_features
from dinoml.ops.gating import swiglu
from dinoml.ops.gemm import (
    gemm_rcr,
    gemm_rcr_bias,
    gemm_rcr_bias_add,
    gemm_rcr_bias_add_add,
    gemm_rcr_bias_add_add_relu,
    gemm_rcr_bias_add_relu,
    gemm_rcr_bias_elup1,
    gemm_rcr_bias_fast_gelu,
    gemm_rcr_bias_gelu,
    gemm_rcr_bias_hardswish,
    gemm_rcr_bias_mul,
    gemm_rcr_bias_mul_add,
    gemm_rcr_bias_mul_tanh,
    gemm_rcr_bias_quick_gelu,
    gemm_rcr_bias_relu,
    gemm_rcr_bias_sigmoid,
    gemm_rcr_bias_sigmoid_mul,
    gemm_rcr_bias_sigmoid_mul_tanh,
    gemm_rcr_bias_swish,
    gemm_rcr_bias_tanh,
    gemm_rrr,
    gemm_rrr_bias,
    gemm_rrr_bias_add,
    gemm_rrr_bias_add_add,
    gemm_rrr_bias_add_add_relu,
    gemm_rrr_bias_add_relu,
    gemm_rrr_bias_elup1,
    gemm_rrr_bias_fast_gelu,
    gemm_rrr_bias_gelu,
    gemm_rrr_bias_hardswish,
    gemm_rrr_bias_mul,
    gemm_rrr_bias_mul_add,
    gemm_rrr_bias_mul_tanh,
    gemm_rrr_bias_relu,
    gemm_rrr_bias_sigmoid,
    gemm_rrr_bias_sigmoid_mul,
    gemm_rrr_bias_sigmoid_mul_tanh,
    gemm_rrr_bias_swish,
    gemm_rrr_bias_tanh,
)
from dinoml.ops.internal import ShapeBufferCountTrue
from dinoml.ops.normalization import add_layer_norm, group_norm, group_norm_swish, layer_norm, rms_norm, t5_layer_norm
from dinoml.ops.pooling import avg_pool1d, avg_pool2d, max_pool2d
from dinoml.ops.positional import (
    get_1d_rotary_pos_embed,
    get_timestep_embedding,
    glm_ocr_text_rope,
    glm_ocr_vision_rope,
)
from dinoml.ops.reductions import (
    argmax,
    reduce_max,
    reduce_mean,
    reduce_min,
    reduce_sum,
    topk,
    var,
    vector_norm,
)
from dinoml.ops.qwen2_5_vl import qwen2_5_vl_stitch_image_features
from dinoml.ops.shape_views import flatten, identity, reshape, squeeze, unsqueeze
from dinoml.ops.softmax import softmax
from dinoml.ops.where import where
from dinoml.shapes import symbolic_int_expr


def size(x: Any, dim: int | None = None) -> tuple[Any, ...] | Any:
    tensor = as_tensor(x)
    shape_spec = [dict(shape_dim) if isinstance(shape_dim, Mapping) else shape_dim for shape_dim in tensor.shape_spec]
    if dim is None:
        return tuple(shape_spec)
    axis = _normalize_symbolic_index(dim, len(shape_spec), "size dim")
    return shape_spec[axis]


def getitem(value: Any, index: Any) -> Any:
    if isinstance(index, bool):
        raise TypeError("getitem index must not be bool")
    return value[index]


def tuple_construct(*values: Any) -> tuple[Any, ...]:
    return tuple(values)


def list_construct(*values: Any) -> list[Any]:
    return list(values)


def int_add(lhs: Any, rhs: Any) -> Any:
    return symbolic_int_expr("add", lhs, rhs)


def int_sub(lhs: Any, rhs: Any) -> Any:
    return symbolic_int_expr("sub", lhs, rhs)


def int_mul(lhs: Any, rhs: Any) -> Any:
    return symbolic_int_expr("mul", lhs, rhs)


def int_div(lhs: Any, rhs: Any) -> Any:
    return symbolic_int_expr("div", lhs, rhs)


def output(x: Any, name: str = "output_0") -> Tensor:
    tensor = as_tensor(x)
    tensor.output_name = name
    return tensor


def _normalize_symbolic_index(index: Any, length: int, name: str) -> int:
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(f"{name} must be an integer, got {type(index).__name__}")
    axis = int(index)
    if axis < 0:
        axis += length
    if axis < 0 or axis >= length:
        raise IndexError(f"{name} {index} is out of range for rank {length}")
    return axis


__all__ = [
    "abs",
    "add_layer_norm",
    "add",
    "arange",
    "argmax",
    "avg_pool1d",
    "avg_pool2d",
    "batch_gather",
    "bmm",
    "bmm_ccc",
    "bmm_ccc_add",
    "bmm_ccr",
    "bmm_ccr_add",
    "bmm_crc",
    "bmm_crc_add",
    "bmm_crr",
    "bmm_crr_add",
    "bmm_rcc",
    "bmm_rcc_add",
    "bmm_rcr",
    "bmm_rcr_add",
    "bmm_rrc",
    "bmm_rrc_add",
    "bmm_rrr",
    "bmm_rrr_add",
    "bmm_xxx",
    "bmm_xxx_add",
    "cast",
    "celu",
    "clamp",
    "chunk",
    "clamp_nan_to_num",
    "concatenate",
    "concatenate_fast",
    "concatenate_tanh",
    "conv1d_bias",
    "conv1d_bias_add",
    "conv1d_bias_add_relu",
    "conv1d_bias_relu",
    "conv2d",
    "conv2d_bias",
    "conv2d_bias_add",
    "conv2d_bias_add_relu",
    "conv2d_bias_relu",
    "cos",
    "div",
    "dynamic_slice",
    "elu",
    "embedding",
    "eq",
    "exp",
    "expand",
    "expand_static_shape",
    "fast_gelu",
    "flatten",
    "flip",
    "floor",
    "floor_div",
    "flash_attention",
    "flash_attention_bias",
    "flash_attention_qkv",
    "flash_attention_static_kv_cache",
    "flash_attention_static_kv_cache_bias",
    "flash_attention_varlen",
    "qkv_split",
    "full",
    "gather",
    "ge",
    "gelu",
    "gelu_new",
    "gemm_rcr",
    "gemm_rcr_bias",
    "gemm_rcr_bias_add",
    "gemm_rcr_bias_add_add",
    "gemm_rcr_bias_add_add_relu",
    "gemm_rcr_bias_add_relu",
    "gemm_rcr_bias_elup1",
    "gemm_rcr_bias_fast_gelu",
    "gemm_rcr_bias_gelu",
    "gemm_rcr_bias_hardswish",
    "gemm_rcr_bias_mul",
    "gemm_rcr_bias_mul_add",
    "gemm_rcr_bias_mul_tanh",
    "gemm_rcr_bias_quick_gelu",
    "gemm_rcr_bias_relu",
    "gemm_rcr_bias_sigmoid",
    "gemm_rcr_bias_sigmoid_mul",
    "gemm_rcr_bias_sigmoid_mul_tanh",
    "gemm_rcr_bias_swish",
    "gemm_rcr_bias_tanh",
    "gemm_rrr",
    "gemm_rrr_bias",
    "gemm_rrr_bias_add",
    "gemm_rrr_bias_add_add",
    "gemm_rrr_bias_add_add_relu",
    "gemm_rrr_bias_add_relu",
    "gemm_rrr_bias_elup1",
    "gemm_rrr_bias_fast_gelu",
    "gemm_rrr_bias_gelu",
    "gemm_rrr_bias_hardswish",
    "gemm_rrr_bias_mul",
    "gemm_rrr_bias_mul_add",
    "gemm_rrr_bias_mul_tanh",
    "gemm_rrr_bias_relu",
    "gemm_rrr_bias_sigmoid",
    "gemm_rrr_bias_sigmoid_mul",
    "gemm_rrr_bias_sigmoid_mul_tanh",
    "gemm_rrr_bias_swish",
    "gemm_rrr_bias_tanh",
    "get_1d_rotary_pos_embed",
    "get_timestep_embedding",
    "glm_ocr_stitch_image_features",
    "glm_ocr_text_rope",
    "glm_ocr_vision_rope",
    "getitem",
    "group_norm",
    "group_norm_swish",
    "gt",
    "hardtanh",
    "identity",
    "index_select",
    "int_add",
    "int_div",
    "int_mul",
    "int_sub",
    "layer_norm",
    "le",
    "leaky_relu",
    "list_construct",
    "log",
    "log1p",
    "lt",
    "max",
    "max_pool2d",
    "meshgrid",
    "min",
    "mul",
    "nan_to_num",
    "ne",
    "output",
    "pad",
    "pad_last_dim",
    "permute",
    "permute021",
    "permute0213",
    "permute102",
    "permute210",
    "pixel_shuffle",
    "pixel_unshuffle",
    "pow",
    "qwen2_5_vl_stitch_image_features",
    "randn",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_sum",
    "relu",
    "repeat_interleave",
    "runtime_index_select",
    "reshape",
    "rms_norm",
    "sigmoid",
    "sign",
    "silu",
    "sin",
    "size",
    "slice_reshape_scatter",
    "slice_scatter",
    "softmax",
    "softplus",
    "softsign",
    "swiglu",
    "split",
    "sqrt",
    "squeeze",
    "stack",
    "sub",
    "t5_layer_norm",
    "tanh",
    "topk",
    "transpose",
    "transposed_conv2d",
    "transposed_conv2d_bias",
    "transposed_conv2d_bias_add",
    "transposed_conv2d_bias_add_relu",
    "transposed_conv2d_bias_relu",
    "tuple_construct",
    "unsqueeze",
    "var",
    "vector_norm",
    "where",
]
