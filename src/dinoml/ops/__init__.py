from __future__ import annotations

from math import isfinite, prod
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dinoml.frontend import GraphBuilder, Parameter, Tensor, as_tensor
from dinoml.ir import normalize_dtype
from dinoml.ops.broadcasting import BROADCAST_DTYPES, resolve_expand_shape
from dinoml.ops.collections import (
    COLLECTION_DTYPES,
    GATHER_INDEX_DTYPES,
    SPECIALIZED_PERMUTE_DIMS,
    chunk_sections,
    infer_batch_gather_shape_with_attrs,
    infer_concatenate_shape_with_attrs,
    infer_dynamic_slice_shape_with_attrs,
    infer_gather_shape_with_attrs,
    infer_index_select_shape_with_attrs,
    infer_pad_shape_with_attrs,
    infer_permute_shape_with_attrs,
    infer_repeat_interleave_shape_with_attrs,
    infer_slice_scatter_shape_with_attrs,
    infer_stack_shape_with_attrs,
    normalize_batch_gather_attrs,
    normalize_chunk_count,
    normalize_concatenate_dim,
    normalize_dynamic_slice_attrs,
    normalize_flip_dims,
    normalize_gather_attrs,
    normalize_index_select_attrs,
    normalize_pad_widths,
    normalize_permute_dims,
    normalize_repeat_interleave_dim,
    normalize_repeat_interleave_repeats,
    normalize_slice_scatter_attrs,
    normalize_split_dim,
    normalize_split_sections,
    normalize_stack_dim,
    normalize_transpose_dims,
)
from dinoml.shapes import Shape, symbolic_int_expr
from dinoml.ops.definitions import OP_REGISTRY, OpDef, get_op_def
from dinoml.ops.creation import ARANGE_DTYPES, CREATION_DTYPES, RANDN_DTYPES
from dinoml.ops.bmm import BMM_FRONTEND_OPS, BMM_HELPER_OPS
from dinoml.ops.conv import (
    CONV2D_BIAS_DTYPES,
    normalize_conv2d_bias_attrs,
    resolve_conv2d_shape,
    resolve_conv2d_bias_add_relu_shape,
    resolve_conv2d_bias_shape,
    resolve_conv2d_bias_add_shape,
    resolve_conv2d_bias_relu_shape,
)
from dinoml.ops.embedding import embedding as _embedding_frontend
from dinoml.ops.elementwise import CAST_ELEMENTWISE_DTYPES, ELEMENTWISE_BY_NAME, elementwise_output_dtype
from dinoml.ops._frontend_utils import infer_shape_spec as _infer_shape_spec
from dinoml.ops.gemm import GEMM_FRONTEND_OPS
from dinoml.ops.normalization import layer_norm as _layer_norm_frontend, t5_layer_norm as _t5_layer_norm_frontend
from dinoml.ops.positional import (
    GET_1D_ROTARY_POS_EMBED_DTYPES,
    emit_get_1d_rotary_pos_embed_component as _emit_get_1d_rotary_pos_embed_component,
    get_timestep_embedding as _get_timestep_embedding_frontend,
    normalize_get_1d_rotary_pos_embed_attrs,
)
from dinoml.ops.pooling import (
    POOLING_DTYPES,
    normalize_avg_pool1d_attrs,
    normalize_avg_pool2d_attrs,
    normalize_max_pool2d_attrs,
    resolve_avg_pool1d_shape,
    resolve_avg_pool2d_shape,
    resolve_max_pool2d_shape,
)
from dinoml.ops.reductions import argmax as _argmax_frontend, reduce_max, reduce_mean, reduce_min, reduce_sum, topk as _topk_frontend, var, vector_norm
from dinoml.ops.shape_views import flatten, identity, reshape, squeeze, unsqueeze
from dinoml.ops.softmax import softmax
from dinoml.ops.where import where as _where_frontend

def emit_registered_op(op_name: str, *args: Any, attrs: Mapping[str, Any] | None = None) -> Tensor:
    op_def = get_op_def(op_name)
    if not op_def.accepts_input_count(len(args)):
        raise ValueError(f"{op_name} expects {op_def.input_count_description()}, got {len(args)}")
    dtype_hint = _dtype_hint(args, op_def)
    tensors = [as_tensor(arg, dtype_hint=dtype_hint) for arg in args]
    builder, dtype = _resolve_builder_and_dtype(op_def, tensors)
    op_attrs = dict(op_def.frontend.default_attrs if op_def.frontend is not None else {})
    if attrs is not None:
        op_attrs.update(attrs)
    out_shape = op_def.infer_shape_for([tensor.shape for tensor in tensors], op_attrs)
    out_shape_spec = _infer_registered_shape_spec(
        op_name,
        [tensor.shape_spec for tensor in tensors],
        out_shape,
        op_attrs,
    )
    out_dtype = elementwise_output_dtype(op_name, dtype, op_attrs) if op_name in ELEMENTWISE_BY_NAME else dtype
    return builder.emit(op_name, tensors, out_shape, out_dtype, op_attrs, shape_spec=out_shape_spec)


def make_frontend_op(op_name: str) -> Callable[..., Tensor]:
    op_def = get_op_def(op_name)
    frontend_name = op_def.frontend.name if op_def.frontend is not None else op_name

    def _frontend(*args: Any, **attrs: Any) -> Tensor:
        return emit_registered_op(op_name, *args, attrs=attrs or None)

    _frontend.__name__ = frontend_name
    _frontend.__qualname__ = frontend_name
    _frontend.__doc__ = op_def.description
    return _frontend


def _cast_frontend(x: Any, dtype: str) -> Tensor:
    dtype = normalize_dtype(dtype)
    if dtype not in CAST_ELEMENTWISE_DTYPES:
        raise ValueError(f"cast does not support dtype {dtype}")
    input_tensor = as_tensor(x)
    if input_tensor.dtype not in CAST_ELEMENTWISE_DTYPES:
        raise ValueError(f"cast does not support input dtype {input_tensor.dtype}")
    return input_tensor.builder.emit(
        "cast",
        [input_tensor],
        input_tensor.shape,
        dtype,
        {"dtype": dtype},
        shape_spec=input_tensor.shape_spec,
    )


def _gelu_new_frontend(x: Any) -> Tensor:
    return emit_registered_op("gelu", x)


def _rms_norm_frontend(x: Any, weight: Any | None = None, eps: float = 1e-6) -> Tensor:
    x_tensor = as_tensor(x, dtype_hint="float32")
    if x_tensor.dtype not in ("float16", "float32", "bfloat16"):
        raise ValueError(f"rms_norm does not support dtype {x_tensor.dtype}")
    if x_tensor.rank < 1:
        raise ValueError("rms_norm requires rank >= 1 input")
    if not isinstance(x_tensor.shape_spec[-1], int):
        raise ValueError("rms_norm currently requires a static last dimension")
    hidden = int(x_tensor.shape[-1])
    if hidden <= 0:
        raise ValueError("rms_norm last dimension must be positive")
    if weight is None:
        weight = as_tensor(
            Parameter([hidden], dtype=x_tensor.dtype, value=np.ones((hidden,), dtype=np.float32)),
            dtype_hint=x_tensor.dtype,
        )
    return _t5_layer_norm_frontend(x_tensor, weight, eps=eps)


def _get_1d_rotary_pos_embed_frontend(
    dim: int,
    pos: Any,
    theta: float = 10000.0,
    use_real: bool = True,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    dtype: str = "float32",
) -> tuple[Tensor, Tensor]:
    output_dtype = normalize_dtype(dtype)
    if output_dtype not in GET_1D_ROTARY_POS_EMBED_DTYPES:
        raise ValueError(f"get_1d_rotary_pos_embed does not support dtype {output_dtype}")
    normalized_attrs = normalize_get_1d_rotary_pos_embed_attrs(
        dim=dim,
        theta=theta,
        use_real=use_real,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
        repeat_interleave_real=repeat_interleave_real,
        output_kind="cos",
    )

    sequence_length: int | None = None
    if isinstance(pos, int) and not isinstance(pos, bool):
        sequence_length = int(pos)
        if sequence_length <= 0:
            raise ValueError("get_1d_rotary_pos_embed integer pos must be a positive sequence length")
        pos_tensor = None
    else:
        pos_tensor = as_tensor(pos, dtype_hint="float32")
        if pos_tensor.dtype not in GET_1D_ROTARY_POS_EMBED_DTYPES:
            raise ValueError(f"get_1d_rotary_pos_embed does not support pos dtype {pos_tensor.dtype}")
        if pos_tensor.rank != 1:
            raise ValueError(f"get_1d_rotary_pos_embed expects rank-1 pos tensor, got rank {pos_tensor.rank}")
        if int(pos_tensor.shape[0]) <= 0:
            raise ValueError("get_1d_rotary_pos_embed pos length must be positive")
        if pos_tensor.dtype != "float32":
            pos_tensor = _cast_frontend(pos_tensor, "float32")

    # IR/runtime lowering is still single-output per node, so the public tuple API
    # is implemented as two generated component ops instead of a fake multi-output node.
    cos_out = _emit_get_1d_rotary_pos_embed_component(
        pos_tensor,
        dim=int(normalized_attrs["dim"]),
        theta=float(normalized_attrs["theta"]),
        use_real=bool(normalized_attrs["use_real"]),
        linear_factor=float(normalized_attrs["linear_factor"]),
        ntk_factor=float(normalized_attrs["ntk_factor"]),
        repeat_interleave_real=bool(normalized_attrs["repeat_interleave_real"]),
        sequence_length=sequence_length if pos_tensor is None else None,
        output_kind="cos",
        dtype=output_dtype,
    )
    sin_out = _emit_get_1d_rotary_pos_embed_component(
        pos_tensor,
        dim=int(normalized_attrs["dim"]),
        theta=float(normalized_attrs["theta"]),
        use_real=bool(normalized_attrs["use_real"]),
        linear_factor=float(normalized_attrs["linear_factor"]),
        ntk_factor=float(normalized_attrs["ntk_factor"]),
        repeat_interleave_real=bool(normalized_attrs["repeat_interleave_real"]),
        sequence_length=sequence_length if pos_tensor is None else None,
        output_kind="sin",
        dtype=output_dtype,
    )
    return cos_out, sin_out


def _full_frontend(shape: Any, fill_value: Any, dtype: str = "float32") -> Tensor:
    dtype = normalize_dtype(dtype)
    if dtype not in CREATION_DTYPES:
        raise ValueError(f"full does not support dtype {dtype}")
    shape_obj = Shape(shape)
    if len(shape_obj) == 0:
        raise ValueError("full shape must not be empty")
    if shape_obj.dynamic:
        raise ValueError("full currently supports only static shapes")
    if dtype == "bool":
        normalized_fill: bool | float = bool(fill_value)
    else:
        normalized_fill = float(fill_value)
    attrs = {"shape": shape_obj.max_shape, "fill_value": normalized_fill, "dtype": dtype}
    return GraphBuilder.current().emit(
        "full",
        [],
        shape_obj.max_shape,
        dtype,
        attrs,
        shape_spec=shape_obj.to_json(),
    )


def _arange_frontend(start: Any, end: Any | None = None, step: Any = 1, dtype: str = "float32") -> Tensor:
    dtype = normalize_dtype(dtype)
    if dtype not in ARANGE_DTYPES:
        raise ValueError(f"arange does not support dtype {dtype}")
    if end is None:
        normalized_start = 0.0
        normalized_end = _creation_number(start, "end")
    else:
        normalized_start = _creation_number(start, "start")
        normalized_end = _creation_number(end, "end")
    normalized_step = _creation_number(step, "step")
    attrs = {"start": normalized_start, "end": normalized_end, "step": normalized_step, "dtype": dtype}
    op_def = get_op_def("arange")
    out_shape = op_def.infer_shape_for([], attrs)
    return GraphBuilder.current().emit(
        "arange",
        [],
        out_shape,
        dtype,
        attrs,
        shape_spec=out_shape,
    )


def _randn_frontend(shape: Any, dtype: str = "float32", seed: int = 0) -> Tensor:
    dtype = normalize_dtype(dtype)
    if dtype not in RANDN_DTYPES:
        raise ValueError(f"randn does not support dtype {dtype}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("randn requires integer seed")
    if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("randn seed must fit in uint64")
    shape_obj = Shape(shape)
    if len(shape_obj) == 0:
        raise ValueError("randn shape must not be empty")
    if shape_obj.dynamic:
        raise ValueError("randn currently supports only static shapes")
    attrs = {"shape": shape_obj.max_shape, "dtype": dtype, "seed": int(seed)}
    return GraphBuilder.current().emit(
        "randn",
        [],
        shape_obj.max_shape,
        dtype,
        attrs,
        shape_spec=shape_obj.to_json(),
    )


def _expand_frontend(x: Any, shape: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in BROADCAST_DTYPES:
        raise ValueError(f"expand does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("expand currently supports only static input shapes")
    out_shape = resolve_expand_shape(tensor.shape, shape)
    return tensor.builder.emit(
        "expand",
        [tensor],
        out_shape,
        tensor.dtype,
        {"shape": list(shape)},
        shape_spec=out_shape,
    )


def _expand_static_shape_frontend(x: Any, shape: Any) -> Tensor:
    return _expand_frontend(x, shape)


def _meshgrid_frontend(inputs: Any, indexing: str = "ij") -> tuple[Tensor, ...]:
    if isinstance(inputs, (Tensor, Parameter)) or not isinstance(inputs, (list, tuple)):
        raise ValueError("meshgrid expects a non-empty sequence of tensors")
    if not inputs:
        raise ValueError("meshgrid expects a non-empty sequence of tensors")
    if indexing != "ij":
        raise NotImplementedError('meshgrid currently supports indexing="ij" only')
    first = as_tensor(inputs[0])
    tensors = [first, *(as_tensor(value, dtype_hint=first.dtype) for value in inputs[1:])]
    for tensor in tensors[1:]:
        if tensor.builder is not first.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if tensor.dtype != first.dtype:
            raise ValueError(f"meshgrid dtype mismatch: {first.dtype} vs {tensor.dtype}")
    if first.dtype not in BROADCAST_DTYPES:
        raise ValueError(f"meshgrid does not support dtype {first.dtype}")
    for tensor in tensors:
        if tensor.rank != 1:
            raise ValueError(f"meshgrid expects rank-1 inputs, got rank {tensor.rank}")
        if tensor.dynamic:
            raise ValueError("meshgrid currently supports only static input shapes")
    grid_shape = [tensor.shape[0] for tensor in tensors]
    outputs = []
    for axis, tensor in enumerate(tensors):
        view_shape = [1] * len(tensors)
        view_shape[axis] = tensor.shape[0]
        outputs.append(_expand_frontend(reshape(tensor, view_shape), grid_shape))
    return tuple(outputs)


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


def _concatenate_frontend(inputs: Any, dim: int = 0) -> Tensor:
    if isinstance(inputs, (Tensor, Parameter)) or not isinstance(inputs, (list, tuple)):
        raise ValueError("concatenate expects a non-empty sequence of tensors")
    if not inputs:
        raise ValueError("concatenate expects a non-empty sequence of tensors")
    first = as_tensor(inputs[0])
    tensors = [first, *(as_tensor(value, dtype_hint=first.dtype) for value in inputs[1:])]
    builder, dtype = _resolve_builder_and_dtype(get_op_def("concatenate"), tensors)
    del builder
    if dtype not in COLLECTION_DTYPES:
        raise ValueError(f"concatenate does not support dtype {dtype}")
    if any(tensor.dynamic for tensor in tensors):
        raise ValueError("concatenate currently supports only static input shapes")
    normalized_dim = normalize_concatenate_dim(dim, first.rank)
    out_shape = infer_concatenate_shape_with_attrs([tensor.shape for tensor in tensors], {"dim": normalized_dim})
    return first.builder.emit(
        "concatenate",
        tensors,
        out_shape,
        dtype,
        {"dim": normalized_dim},
        shape_spec=out_shape,
    )


def _concatenate_fast_frontend(inputs: Any, dim: int = 0) -> Tensor:
    return _concatenate_frontend(inputs, dim=dim)


def _concatenate_tanh_frontend(inputs: Any, dim: int = 0) -> Tensor:
    return tanh(_concatenate_frontend(inputs, dim=dim))


def _stack_frontend(inputs: Any, dim: int = 0) -> Tensor:
    if isinstance(inputs, (Tensor, Parameter)) or not isinstance(inputs, (list, tuple)):
        raise ValueError("stack expects a non-empty sequence of tensors")
    if not inputs:
        raise ValueError("stack expects a non-empty sequence of tensors")
    first = as_tensor(inputs[0])
    tensors = [first, *(as_tensor(value, dtype_hint=first.dtype) for value in inputs[1:])]
    builder, dtype = _resolve_builder_and_dtype(get_op_def("stack"), tensors)
    del builder
    if dtype not in COLLECTION_DTYPES:
        raise ValueError(f"stack does not support dtype {dtype}")
    if any(tensor.dynamic for tensor in tensors):
        raise ValueError("stack currently supports only static input shapes")
    normalized_dim = normalize_stack_dim(dim, first.rank)
    out_shape = infer_stack_shape_with_attrs([tensor.shape for tensor in tensors], {"dim": normalized_dim})
    return first.builder.emit(
        "stack",
        tensors,
        out_shape,
        dtype,
        {"dim": normalized_dim},
        shape_spec=out_shape,
    )


def _flip_frontend(x: Any, dims: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"flip does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("flip currently supports only static input shapes")
    normalized_dims = normalize_flip_dims(dims, tensor.rank)
    return tensor.builder.emit(
        "flip",
        [tensor],
        tensor.shape,
        tensor.dtype,
        {"dims": normalized_dims},
        shape_spec=tensor.shape_spec,
    )


def _repeat_interleave_frontend(x: Any, repeats: Any, dim: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"repeat_interleave does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("repeat_interleave currently supports only static input shapes")
    normalized_dim = normalize_repeat_interleave_dim(dim, tensor.rank)
    normalized_repeats = normalize_repeat_interleave_repeats(repeats)
    out_shape = infer_repeat_interleave_shape_with_attrs(
        [tensor.shape],
        {"repeats": normalized_repeats, "dim": normalized_dim},
    )
    return tensor.builder.emit(
        "repeat_interleave",
        [tensor],
        out_shape,
        tensor.dtype,
        {"repeats": normalized_repeats, "dim": normalized_dim},
        shape_spec=out_shape,
    )


def _permute_frontend(x: Any, dims: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"permute does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("permute currently supports only static input shapes")
    normalized_dims = normalize_permute_dims(dims, tensor.rank)
    out_shape = infer_permute_shape_with_attrs([tensor.shape], {"dims": normalized_dims})
    return tensor.builder.emit(
        "permute",
        [tensor],
        out_shape,
        tensor.dtype,
        {"dims": normalized_dims},
        shape_spec=out_shape,
    )


def _specialized_permute_frontend(op_name: str, x: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"{op_name} does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError(f"{op_name} currently supports only static input shapes")
    fixed_dims = SPECIALIZED_PERMUTE_DIMS[op_name]
    if tensor.rank != len(fixed_dims):
        raise ValueError(f"{op_name} expects rank-{len(fixed_dims)} input, got rank {tensor.rank}")
    normalized_dims = list(fixed_dims)
    out_shape = infer_permute_shape_with_attrs([tensor.shape], {"dims": normalized_dims})
    return tensor.builder.emit(
        op_name,
        [tensor],
        out_shape,
        tensor.dtype,
        {"dims": normalized_dims},
        shape_spec=out_shape,
    )


def _permute021_frontend(x: Any) -> Tensor:
    return _specialized_permute_frontend("permute021", x)


def _permute0213_frontend(x: Any) -> Tensor:
    return _specialized_permute_frontend("permute0213", x)


def _permute102_frontend(x: Any) -> Tensor:
    return _specialized_permute_frontend("permute102", x)


def _permute210_frontend(x: Any) -> Tensor:
    return _specialized_permute_frontend("permute210", x)


def _pixel_shuffle_frontend(x: Any, upscale_factor: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.rank != 4:
        raise ValueError(f"pixel_shuffle expects rank-4 input [N, C, H, W], got rank {tensor.rank}")
    if tensor.dynamic:
        raise ValueError("pixel_shuffle currently supports only static input shapes")
    factor = _normalize_pixel_factor(upscale_factor, "pixel_shuffle upscale_factor")
    batch, channels_in, height, width = tensor.shape
    channel_factor = factor * factor
    if channels_in % channel_factor != 0:
        raise ValueError(
            f"pixel_shuffle input channels {channels_in} must be divisible by upscale_factor^2 ({channel_factor})"
        )
    channels_out = channels_in // channel_factor
    reshaped = reshape(tensor, [batch, channels_out, factor, factor, height, width])
    shuffled = _permute_frontend(reshaped, (0, 1, 4, 2, 5, 3))
    return reshape(shuffled, [batch, channels_out, height * factor, width * factor])


def _pixel_unshuffle_frontend(x: Any, downscale_factor: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.rank != 4:
        raise ValueError(f"pixel_unshuffle expects rank-4 input [N, C, H, W], got rank {tensor.rank}")
    if tensor.dynamic:
        raise ValueError("pixel_unshuffle currently supports only static input shapes")
    factor = _normalize_pixel_factor(downscale_factor, "pixel_unshuffle downscale_factor")
    batch, channels, height_in, width_in = tensor.shape
    if height_in % factor != 0:
        raise ValueError(f"pixel_unshuffle input height {height_in} must be divisible by downscale_factor {factor}")
    if width_in % factor != 0:
        raise ValueError(f"pixel_unshuffle input width {width_in} must be divisible by downscale_factor {factor}")
    height_out = height_in // factor
    width_out = width_in // factor
    reshaped = reshape(tensor, [batch, channels, height_out, factor, width_out, factor])
    unshuffled = _permute_frontend(reshaped, (0, 1, 3, 5, 2, 4))
    return reshape(unshuffled, [batch, channels * factor * factor, height_out, width_out])


def _normalize_pixel_factor(factor: Any, name: str) -> int:
    if not isinstance(factor, int) or isinstance(factor, bool):
        raise ValueError(f"{name} must be a positive integer")
    if factor <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(factor)


def _dynamic_slice_frontend(x: Any, start_indices: Any, slice_sizes: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"dynamic_slice does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("dynamic_slice currently supports only static input shapes")
    normalized_starts, normalized_sizes = normalize_dynamic_slice_attrs(start_indices, slice_sizes, tensor.shape)
    out_shape = infer_dynamic_slice_shape_with_attrs(
        [tensor.shape],
        {"start_indices": normalized_starts, "slice_sizes": normalized_sizes},
    )
    return tensor.builder.emit(
        "dynamic_slice",
        [tensor],
        out_shape,
        tensor.dtype,
        {"start_indices": normalized_starts, "slice_sizes": normalized_sizes},
        shape_spec=out_shape,
    )


def _index_select_frontend(x: Any, dim: Any, indices: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"index_select does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("index_select currently supports only static input shapes")
    normalized_dim, normalized_indices = normalize_index_select_attrs(dim, indices, tensor.shape)
    out_shape = infer_index_select_shape_with_attrs(
        [tensor.shape],
        {"dim": normalized_dim, "indices": normalized_indices},
    )
    return tensor.builder.emit(
        "index_select",
        [tensor],
        out_shape,
        tensor.dtype,
        {"dim": normalized_dim, "indices": normalized_indices},
        shape_spec=out_shape,
    )


def _gather_frontend(x: Any, dim: Any, index: Any) -> Tensor:
    tensor = as_tensor(x)
    index_tensor = as_tensor(index)
    if tensor.builder is not index_tensor.builder:
        raise ValueError("Cannot combine tensors from different DinoML traces")
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"gather does not support dtype {tensor.dtype}")
    if index_tensor.dtype not in GATHER_INDEX_DTYPES:
        raise ValueError(f"gather index must have dtype int64 or int32, got {index_tensor.dtype}")
    if tensor.dynamic or index_tensor.dynamic:
        raise ValueError("gather currently supports only static input and index shapes")
    normalized_dim = normalize_gather_attrs(dim, tensor.shape, index_tensor.shape)
    out_shape = infer_gather_shape_with_attrs(
        [tensor.shape, index_tensor.shape],
        {"dim": normalized_dim},
    )
    return tensor.builder.emit(
        "gather",
        [tensor, index_tensor],
        out_shape,
        tensor.dtype,
        {"dim": normalized_dim},
        shape_spec=out_shape,
    )


def _batch_gather_frontend(x: Any, indices: Any) -> Tensor:
    tensor = as_tensor(x)
    index_tensor = as_tensor(indices)
    if tensor.builder is not index_tensor.builder:
        raise ValueError("Cannot combine tensors from different DinoML traces")
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"batch_gather does not support dtype {tensor.dtype}")
    if index_tensor.dtype not in GATHER_INDEX_DTYPES:
        raise ValueError(f"batch_gather indices must have dtype int64 or int32, got {index_tensor.dtype}")
    if tensor.dynamic or index_tensor.dynamic:
        raise ValueError("batch_gather currently supports only static input and index shapes")
    normalize_batch_gather_attrs(tensor.shape, index_tensor.shape)
    out_shape = infer_batch_gather_shape_with_attrs([tensor.shape, index_tensor.shape], {})
    return tensor.builder.emit(
        "batch_gather",
        [tensor, index_tensor],
        out_shape,
        tensor.dtype,
        {},
        shape_spec=out_shape,
    )


def _slice_scatter_frontend(x: Any, update: Any, start_indices: Any) -> Tensor:
    tensor = as_tensor(x)
    update_tensor = as_tensor(update, dtype_hint=tensor.dtype)
    builder, dtype = _resolve_builder_and_dtype(get_op_def("slice_scatter"), [tensor, update_tensor])
    del builder
    if dtype not in COLLECTION_DTYPES:
        raise ValueError(f"slice_scatter does not support dtype {dtype}")
    if tensor.dynamic or update_tensor.dynamic:
        raise ValueError("slice_scatter currently supports only static input shapes")
    normalized_starts = normalize_slice_scatter_attrs(start_indices, tensor.shape, update_tensor.shape)
    out_shape = infer_slice_scatter_shape_with_attrs(
        [tensor.shape, update_tensor.shape],
        {"start_indices": normalized_starts},
    )
    return tensor.builder.emit(
        "slice_scatter",
        [tensor, update_tensor],
        out_shape,
        dtype,
        {"start_indices": normalized_starts},
        shape_spec=tensor.shape_spec,
    )


def _pad_frontend(x: Any, pad: Any, value: Any = 0.0) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"pad does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("pad currently supports only static input shapes")
    normalize_pad_widths(pad, tensor.rank)
    normalized_pad = [int(value) for value in pad]
    out_shape = infer_pad_shape_with_attrs([tensor.shape], {"pad": normalized_pad})
    if tensor.dtype == "bool":
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"pad value must be a constant scalar, got {value!r}")
        if isinstance(value, float) and not isfinite(float(value)):
            raise ValueError("pad value must be finite")
        normalized_value: bool | float = bool(value)
    else:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"pad value must be a constant numeric scalar, got {value!r}")
        if not isfinite(float(value)):
            raise ValueError("pad value must be finite")
        normalized_value = float(value)
    return tensor.builder.emit(
        "pad",
        [tensor],
        out_shape,
        tensor.dtype,
        {"pad": normalized_pad, "value": normalized_value},
        shape_spec=out_shape,
    )


def _pad_last_dim_frontend(x: Any, left: Any, right: Any, value: Any = 0.0) -> Tensor:
    if not isinstance(left, int) or isinstance(left, bool):
        raise ValueError(f"pad_last_dim left must be a non-negative integer, got {left!r}")
    if not isinstance(right, int) or isinstance(right, bool):
        raise ValueError(f"pad_last_dim right must be a non-negative integer, got {right!r}")
    return _pad_frontend(x, [int(left), int(right)], value=value)


def _avg_pool1d_frontend(x: Any, kernel_size: Any, stride: Any | None = None, padding: Any = 0) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in POOLING_DTYPES:
        raise ValueError(f"avg_pool1d does not support dtype {tensor.dtype}")
    if tensor.rank != 3:
        raise ValueError(f"avg_pool1d expects rank-3 NCL input, got rank {tensor.rank}")
    if tensor.dynamic:
        raise ValueError("avg_pool1d currently supports only static input shapes")
    kernel, normalized_stride, normalized_padding = normalize_avg_pool1d_attrs(kernel_size, stride, padding)
    out_shape = resolve_avg_pool1d_shape(tensor.shape, kernel, normalized_stride, normalized_padding)
    return tensor.builder.emit(
        "avg_pool1d",
        [tensor],
        out_shape,
        tensor.dtype,
        {"kernel_size": kernel, "stride": normalized_stride, "padding": normalized_padding},
        shape_spec=out_shape,
    )


def _avg_pool2d_frontend(x: Any, kernel_size: Any, stride: Any | None = None, padding: Any = 0) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in POOLING_DTYPES:
        raise ValueError(f"avg_pool2d does not support dtype {tensor.dtype}")
    if tensor.rank != 4:
        raise ValueError(f"avg_pool2d expects rank-4 NCHW input, got rank {tensor.rank}")
    if tensor.dynamic:
        raise ValueError("avg_pool2d currently supports only static input shapes")
    kernel, normalized_stride, normalized_padding = normalize_avg_pool2d_attrs(kernel_size, stride, padding)
    out_shape = resolve_avg_pool2d_shape(tensor.shape, kernel, normalized_stride, normalized_padding)
    return tensor.builder.emit(
        "avg_pool2d",
        [tensor],
        out_shape,
        tensor.dtype,
        {"kernel_size": kernel, "stride": normalized_stride, "padding": normalized_padding},
        shape_spec=out_shape,
    )


def _max_pool2d_frontend(x: Any, kernel_size: Any, stride: Any | None = None, padding: Any = 0) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in POOLING_DTYPES:
        raise ValueError(f"max_pool2d does not support dtype {tensor.dtype}")
    if tensor.rank != 4:
        raise ValueError(f"max_pool2d expects rank-4 NCHW input, got rank {tensor.rank}")
    if tensor.dynamic:
        raise ValueError("max_pool2d currently supports only static input shapes")
    kernel, normalized_stride, normalized_padding = normalize_max_pool2d_attrs(kernel_size, stride, padding)
    out_shape = resolve_max_pool2d_shape(tensor.shape, kernel, normalized_stride, normalized_padding)
    return tensor.builder.emit(
        "max_pool2d",
        [tensor],
        out_shape,
        tensor.dtype,
        {"kernel_size": kernel, "stride": normalized_stride, "padding": normalized_padding},
        shape_spec=out_shape,
    )


def _conv2d_bias_frontend(
    x: Any,
    weight: Any,
    bias: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return _conv2d_bias_family_frontend(
        "conv2d_bias",
        resolve_shape=resolve_conv2d_bias_shape,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def _conv2d_bias_relu_frontend(
    x: Any,
    weight: Any,
    bias: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return _conv2d_bias_family_frontend(
        "conv2d_bias_relu",
        resolve_shape=resolve_conv2d_bias_relu_shape,
        x=x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def _conv2d_bias_add_frontend(
    x: Any,
    weight: Any,
    bias: Any,
    residual: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return _conv2d_bias_family_frontend(
        "conv2d_bias_add",
        resolve_shape=resolve_conv2d_bias_add_shape,
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def _conv2d_bias_add_relu_frontend(
    x: Any,
    weight: Any,
    bias: Any,
    residual: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    return _conv2d_bias_family_frontend(
        "conv2d_bias_add_relu",
        resolve_shape=resolve_conv2d_bias_add_relu_shape,
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def _conv2d_bias_family_frontend(
    op_name: str,
    *,
    resolve_shape: Callable[..., list[int]],
    x: Any,
    weight: Any,
    bias: Any,
    residual: Any | None = None,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: int,
) -> Tensor:
    x_tensor = as_tensor(x)
    weight_tensor = as_tensor(weight, dtype_hint=x_tensor.dtype)
    bias_tensor = as_tensor(bias, dtype_hint=x_tensor.dtype)
    tensors = [x_tensor, weight_tensor, bias_tensor]
    residual_tensor = None if residual is None else as_tensor(residual, dtype_hint=x_tensor.dtype)
    if residual_tensor is not None:
        tensors.append(residual_tensor)
    for tensor in tensors[1:]:
        if tensor.builder is not x_tensor.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if tensor.dtype != x_tensor.dtype:
            raise ValueError(f"{op_name} dtype mismatch: {x_tensor.dtype} vs {tensor.dtype}")
    if x_tensor.dtype not in CONV2D_BIAS_DTYPES:
        raise ValueError(f"{op_name} does not support dtype {x_tensor.dtype}")
    if x_tensor.rank != 4:
        raise ValueError(f"{op_name} expects rank-4 NCHW activation, got rank {x_tensor.rank}")
    if weight_tensor.rank != 4:
        raise ValueError(f"{op_name} expects rank-4 OIHW weight, got rank {weight_tensor.rank}")
    if bias_tensor.rank != 1:
        raise ValueError(f"{op_name} expects rank-1 bias, got rank {bias_tensor.rank}")
    if residual_tensor is not None and residual_tensor.rank != 4:
        raise ValueError(f"{op_name} expects rank-4 residual, got rank {residual_tensor.rank}")
    if any(tensor.dynamic for tensor in tensors):
        expected = "activation, weight, bias, and residual" if residual_tensor is not None else "activation, weight, and bias"
        raise ValueError(f"{op_name} currently supports only static {expected} shapes")
    normalized_stride, normalized_padding, normalized_dilation, normalized_groups = normalize_conv2d_bias_attrs(
        stride,
        padding,
        dilation,
        groups,
    )
    if residual_tensor is None:
        out_shape = resolve_shape(
            x_tensor.shape,
            weight_tensor.shape,
            bias_tensor.shape,
            stride=normalized_stride,
            padding=normalized_padding,
            dilation=normalized_dilation,
            groups=normalized_groups,
        )
    else:
        out_shape = resolve_shape(
            x_tensor.shape,
            weight_tensor.shape,
            bias_tensor.shape,
            residual_tensor.shape,
            stride=normalized_stride,
            padding=normalized_padding,
            dilation=normalized_dilation,
            groups=normalized_groups,
        )
    return x_tensor.builder.emit(
        op_name,
        tensors,
        out_shape,
        x_tensor.dtype,
        {
            "stride": normalized_stride,
            "padding": normalized_padding,
            "dilation": normalized_dilation,
            "groups": normalized_groups,
        },
        shape_spec=out_shape,
    )


def _conv2d_frontend(
    x: Any,
    weight: Any,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> Tensor:
    x_tensor = as_tensor(x)
    weight_tensor = as_tensor(weight, dtype_hint=x_tensor.dtype)
    if weight_tensor.builder is not x_tensor.builder:
        raise ValueError("Cannot combine tensors from different DinoML traces")
    if weight_tensor.dtype != x_tensor.dtype:
        raise ValueError(f"conv2d dtype mismatch: {x_tensor.dtype} vs {weight_tensor.dtype}")
    if x_tensor.dtype not in CONV2D_BIAS_DTYPES:
        raise ValueError(f"conv2d does not support dtype {x_tensor.dtype}")
    if x_tensor.rank != 4:
        raise ValueError(f"conv2d expects rank-4 NCHW activation, got rank {x_tensor.rank}")
    if weight_tensor.rank != 4:
        raise ValueError(f"conv2d expects rank-4 OIHW weight, got rank {weight_tensor.rank}")
    if x_tensor.dynamic or weight_tensor.dynamic:
        raise ValueError("conv2d currently supports only static activation and weight shapes")
    normalized_stride, normalized_padding, normalized_dilation, normalized_groups = normalize_conv2d_bias_attrs(
        stride,
        padding,
        dilation,
        groups,
    )
    out_shape = resolve_conv2d_shape(
        x_tensor.shape,
        weight_tensor.shape,
        stride=normalized_stride,
        padding=normalized_padding,
        dilation=normalized_dilation,
        groups=normalized_groups,
    )
    zero_bias = as_tensor(
        Parameter(
            [int(weight_tensor.shape[0])],
            dtype=x_tensor.dtype,
            name="conv2d_zero_bias",
            value=np.zeros((int(weight_tensor.shape[0]),), dtype=np.float32),
        ),
        dtype_hint=x_tensor.dtype,
    )
    return x_tensor.builder.emit(
        "conv2d_bias",
        [x_tensor, weight_tensor, zero_bias],
        out_shape,
        x_tensor.dtype,
        {
            "stride": normalized_stride,
            "padding": normalized_padding,
            "dilation": normalized_dilation,
            "groups": normalized_groups,
            "bias_mode": "explicit_zero_constant",
            "source_op": "conv2d",
        },
        shape_spec=out_shape,
    )


def _slice_reshape_scatter_frontend(x: Any, update: Any, start_indices: Any, slice_shape: Any) -> Tensor:
    tensor = as_tensor(x)
    update_tensor = as_tensor(update, dtype_hint=tensor.dtype)
    normalized_shape = _normalize_slice_reshape_scatter_shape(slice_shape, tensor.rank, update_tensor.numel)
    reshaped_update = reshape(update_tensor, normalized_shape)
    return _slice_scatter_frontend(tensor, reshaped_update, start_indices)


def _normalize_slice_reshape_scatter_shape(slice_shape: Any, rank: int, update_numel: int) -> list[int]:
    if not isinstance(slice_shape, (list, tuple)):
        raise ValueError(f"slice_reshape_scatter slice_shape must be a sequence of integers, got {slice_shape!r}")
    normalized: list[int] = []
    for dim in slice_shape:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"slice_reshape_scatter slice_shape must contain only integers, got {slice_shape!r}")
        if dim <= 0:
            raise ValueError(f"slice_reshape_scatter slice_shape dimensions must be positive, got {dim}")
        normalized.append(int(dim))
    if len(normalized) != rank:
        raise ValueError(f"slice_reshape_scatter slice_shape rank {len(normalized)} must match input rank {rank}")
    if int(prod(normalized)) != int(update_numel):
        raise ValueError(
            f"slice_reshape_scatter slice_shape {normalized} must preserve update element count {int(update_numel)}"
        )
    return normalized


def _split_frontend(x: Any, split_size_or_sections: Any, dim: Any = 0) -> tuple[Tensor, ...]:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"split does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("split currently supports only static input shapes")
    normalized_dim = normalize_split_dim(dim, tensor.rank)
    sections = normalize_split_sections(split_size_or_sections, tensor.shape[normalized_dim])
    return _slice_sections(tensor, sections, normalized_dim)


def _chunk_frontend(x: Any, chunks: Any, dim: Any = 0) -> tuple[Tensor, ...]:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"chunk does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("chunk currently supports only static input shapes")
    normalized_dim = normalize_split_dim(dim, tensor.rank)
    normalized_chunks = normalize_chunk_count(chunks)
    sections = chunk_sections(tensor.shape[normalized_dim], normalized_chunks)
    return _slice_sections(tensor, sections, normalized_dim)


def _slice_sections(tensor: Tensor, sections: Sequence[int], dim: int) -> tuple[Tensor, ...]:
    outputs = []
    start = 0
    for section in sections:
        starts = [0] * tensor.rank
        sizes = list(tensor.shape)
        starts[dim] = start
        sizes[dim] = section
        outputs.append(_dynamic_slice_frontend(tensor, starts, sizes))
        start += section
    return tuple(outputs)


def _transpose_frontend(x: Any, dim0: Any, dim1: Any) -> Tensor:
    tensor = as_tensor(x)
    if tensor.dtype not in COLLECTION_DTYPES:
        raise ValueError(f"transpose does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("transpose currently supports only static input shapes")
    normalized_dim0, normalized_dim1 = normalize_transpose_dims(dim0, dim1, tensor.rank)
    dims = list(range(tensor.rank))
    dims[normalized_dim0], dims[normalized_dim1] = dims[normalized_dim1], dims[normalized_dim0]
    return _permute_frontend(tensor, dims)


def _creation_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"arange requires numeric {name}")
    return float(value)


def output(x: Any, name: str = "output_0") -> Tensor:
    tensor = as_tensor(x)
    tensor.output_name = name
    return tensor


def _resolve_builder_and_dtype(op_def: OpDef, tensors: list[Tensor]) -> tuple[GraphBuilder, str]:
    if not tensors:
        return GraphBuilder.current(), op_def.allowed_dtypes[0]
    first = tensors[0]
    for tensor in tensors[1:]:
        if tensor.builder is not first.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if tensor.dtype != first.dtype:
            raise ValueError(f"{op_def.name} dtype mismatch: {first.dtype} vs {tensor.dtype}")
    if first.dtype not in op_def.allowed_dtypes:
        raise ValueError(f"{op_def.name} does not support dtype {first.dtype}")
    return first.builder, first.dtype


def _dtype_hint(args: tuple[Any, ...], op_def: OpDef) -> str:
    for arg in args:
        if isinstance(arg, (Tensor, Parameter)):
            return arg.dtype
    return op_def.allowed_dtypes[0]


def _infer_registered_shape_spec(
    op_name: str,
    shape_specs: list[list[Any]],
    out_shape: list[int],
    attrs: Mapping[str, Any],
) -> list[Any]:
    if len(shape_specs) == 1 and (op_name == "permute" or op_name in SPECIALIZED_PERMUTE_DIMS):
        dims = attrs.get("dims")
        if dims is None and op_name in SPECIALIZED_PERMUTE_DIMS:
            dims = SPECIALIZED_PERMUTE_DIMS[op_name]
        if dims is not None:
            normalized_dims = normalize_permute_dims(dims, len(shape_specs[0]))
            input_shape_spec = shape_specs[0]
            return [_copy_shape_dim(input_shape_spec[axis]) for axis in normalized_dims]
    return _infer_shape_spec(shape_specs, out_shape)


def _copy_shape_dim(dim: Any) -> Any:
    return dict(dim) if isinstance(dim, Mapping) else dim
def _normalize_symbolic_index(index: Any, length: int, name: str) -> int:
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(f"{name} must be an integer, got {type(index).__name__}")
    axis = int(index)
    if axis < 0:
        axis += length
    if axis < 0 or axis >= length:
        raise IndexError(f"{name} {index} is out of range for rank {length}")
    return axis


for _frontend_name in OP_REGISTRY.frontend_names():
    _op_def = OP_REGISTRY.get_frontend(_frontend_name)
    globals()[_frontend_name] = make_frontend_op(_op_def.name)

globals()["where"] = _where_frontend
globals()["cast"] = _cast_frontend
globals()["gelu_new"] = _gelu_new_frontend
globals()["rms_norm"] = _rms_norm_frontend
globals()["get_1d_rotary_pos_embed"] = _get_1d_rotary_pos_embed_frontend
globals()["get_timestep_embedding"] = _get_timestep_embedding_frontend
globals()["full"] = _full_frontend
globals()["arange"] = _arange_frontend
globals()["randn"] = _randn_frontend
globals()["argmax"] = _argmax_frontend
globals()["topk"] = _topk_frontend
globals()["batch_gather"] = _batch_gather_frontend
globals()["embedding"] = _embedding_frontend
globals()["expand"] = _expand_frontend
globals()["expand_static_shape"] = _expand_static_shape_frontend
globals()["meshgrid"] = _meshgrid_frontend
globals()["concatenate"] = _concatenate_frontend
globals()["concatenate_fast"] = _concatenate_fast_frontend
globals()["concatenate_tanh"] = _concatenate_tanh_frontend
globals()["dynamic_slice"] = _dynamic_slice_frontend
globals()["gather"] = _gather_frontend
globals()["index_select"] = _index_select_frontend
globals()["slice_scatter"] = _slice_scatter_frontend
globals()["slice_reshape_scatter"] = _slice_reshape_scatter_frontend
globals()["split"] = _split_frontend
globals()["chunk"] = _chunk_frontend
globals()["stack"] = _stack_frontend
globals()["pad"] = _pad_frontend
globals()["pad_last_dim"] = _pad_last_dim_frontend
globals()["avg_pool1d"] = _avg_pool1d_frontend
globals()["avg_pool2d"] = _avg_pool2d_frontend
globals()["conv2d"] = _conv2d_frontend
globals()["conv2d_bias"] = _conv2d_bias_frontend
globals()["conv2d_bias_add"] = _conv2d_bias_add_frontend
globals()["conv2d_bias_add_relu"] = _conv2d_bias_add_relu_frontend
globals()["conv2d_bias_relu"] = _conv2d_bias_relu_frontend
globals()["max_pool2d"] = _max_pool2d_frontend
globals()["flip"] = _flip_frontend
globals()["permute"] = _permute_frontend
globals()["layer_norm"] = _layer_norm_frontend
globals()["t5_layer_norm"] = _t5_layer_norm_frontend
globals()["permute021"] = _permute021_frontend
globals()["permute0213"] = _permute0213_frontend
globals()["permute102"] = _permute102_frontend
globals()["permute210"] = _permute210_frontend
globals()["pixel_shuffle"] = _pixel_shuffle_frontend
globals()["pixel_unshuffle"] = _pixel_unshuffle_frontend
globals()["repeat_interleave"] = _repeat_interleave_frontend
globals()["transpose"] = _transpose_frontend
globals().update(GEMM_FRONTEND_OPS)
globals().update(BMM_FRONTEND_OPS)
globals().update(BMM_HELPER_OPS)


__all__ = list(dict.fromkeys([
    *OP_REGISTRY.frontend_names(),
    *BMM_HELPER_OPS,
    "emit_registered_op",
    "expand",
    "expand_static_shape",
    "flip",
    "flatten",
    "gelu_new",
    "get_1d_rotary_pos_embed",
    "get_timestep_embedding",
    "identity",
    "make_frontend_op",
    "meshgrid",
    "output",
    "pad",
    "pad_last_dim",
    "size",
    "int_add",
    "int_sub",
    "int_mul",
    "int_div",
    "getitem",
    "tuple_construct",
    "list_construct",
    "permute",
    "permute021",
    "permute0213",
    "permute102",
    "permute210",
    "pixel_shuffle",
    "pixel_unshuffle",
    "rms_norm",
    "reshape",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_sum",
    "randn",
    "repeat_interleave",
    "softmax",
    "stack",
    "transpose",
    "topk",
    "squeeze",
    "unsqueeze",
    "var",
    "vector_norm",
    "arange",
    "argmax",
    "avg_pool1d",
    "avg_pool2d",
    "batch_gather",
    "cast",
    "chunk",
    "concatenate",
    "concatenate_fast",
    "concatenate_tanh",
    "conv2d",
    "conv2d_bias",
    "conv2d_bias_add",
    "conv2d_bias_add_relu",
    "conv2d_bias_relu",
    "dynamic_slice",
    "full",
    "gather",
    "index_select",
    "max_pool2d",
    "slice_reshape_scatter",
    "slice_scatter",
    "split",
    "where",
]))
