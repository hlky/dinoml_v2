from __future__ import annotations

import builtins
from math import prod
from typing import Any, Callable, Mapping, Sequence

from dinoml.frontend import GraphBuilder, Parameter, Tensor, as_tensor
from dinoml.ir import normalize_dtype
from dinoml.ops.broadcasting import BROADCAST_DTYPES, resolve_expand_shape
from dinoml.ops.collections import (
    COLLECTION_DTYPES,
    chunk_sections,
    infer_concatenate_shape_with_attrs,
    infer_dynamic_slice_shape_with_attrs,
    infer_index_select_shape_with_attrs,
    infer_permute_shape_with_attrs,
    infer_repeat_interleave_shape_with_attrs,
    infer_slice_scatter_shape_with_attrs,
    infer_stack_shape_with_attrs,
    normalize_chunk_count,
    normalize_concatenate_dim,
    normalize_dynamic_slice_attrs,
    normalize_flip_dims,
    normalize_index_select_attrs,
    normalize_permute_dims,
    normalize_repeat_interleave_dim,
    normalize_repeat_interleave_repeats,
    normalize_slice_scatter_attrs,
    normalize_split_dim,
    normalize_split_sections,
    normalize_stack_dim,
    normalize_transpose_dims,
)
from dinoml.shapes import Shape
from dinoml.ops.definitions import OP_REGISTRY, OpDef, get_op_def
from dinoml.ops.creation import ARANGE_DTYPES, CREATION_DTYPES, RANDN_DTYPES
from dinoml.ops.bmm import BMM_FRONTEND_OPS, BMM_HELPER_OPS
from dinoml.ops.elementwise import CAST_ELEMENTWISE_DTYPES, ELEMENTWISE_BY_NAME, ELEMENTWISE_OUTPUT_DTYPES, elementwise_output_dtype
from dinoml.ops.gemm import GEMM_FRONTEND_OPS
from dinoml.ops.reductions import reduce_max, reduce_mean, reduce_min, reduce_sum, var, vector_norm
from dinoml.ops.shape_views import flatten, identity, reshape, squeeze, unsqueeze
from dinoml.ops.softmax import softmax


def emit_registered_op(op_name: str, *args: Any, attrs: Mapping[str, Any] | None = None) -> Tensor:
    op_def = get_op_def(op_name)
    if not op_def.accepts_input_count(len(args)):
        raise ValueError(f"{op_name} expects {op_def.input_count} inputs, got {len(args)}")
    dtype_hint = _dtype_hint(args, op_def)
    tensors = [as_tensor(arg, dtype_hint=dtype_hint) for arg in args]
    builder, dtype = _resolve_builder_and_dtype(op_def, tensors)
    op_attrs = dict(op_def.frontend.default_attrs if op_def.frontend is not None else {})
    if attrs is not None:
        op_attrs.update(attrs)
    out_shape = op_def.infer_shape_for([tensor.shape for tensor in tensors], op_attrs)
    out_shape_spec = _infer_shape_spec([tensor.shape_spec for tensor in tensors], out_shape)
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


def _where_frontend(condition: Any, x: Any, y: Any) -> Tensor:
    op_def = get_op_def("where")
    condition_tensor = as_tensor(condition)
    x_tensor = as_tensor(x)
    y_tensor = as_tensor(y, dtype_hint=x_tensor.dtype)
    tensors = [condition_tensor, x_tensor, y_tensor]
    builder = condition_tensor.builder
    for tensor in tensors[1:]:
        if tensor.builder is not builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
    if condition_tensor.dtype != "bool":
        raise ValueError(f"where condition must have dtype bool, got {condition_tensor.dtype}")
    if x_tensor.dtype != y_tensor.dtype:
        raise ValueError(f"where x/y dtype mismatch: {x_tensor.dtype} vs {y_tensor.dtype}")
    if x_tensor.dtype not in ELEMENTWISE_OUTPUT_DTYPES:
        raise ValueError(f"where does not support dtype {x_tensor.dtype}")
    out_shape = op_def.infer_shape([tensor.shape for tensor in tensors])
    out_shape_spec = _infer_shape_spec([tensor.shape_spec for tensor in tensors], out_shape)
    return builder.emit("where", tensors, out_shape, x_tensor.dtype, {}, shape_spec=out_shape_spec)


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


def _permute021_frontend(x: Any) -> Tensor:
    return _permute_frontend(x, (0, 2, 1))


def _permute0213_frontend(x: Any) -> Tensor:
    return _permute_frontend(x, (0, 2, 1, 3))


def _permute102_frontend(x: Any) -> Tensor:
    return _permute_frontend(x, (1, 0, 2))


def _permute210_frontend(x: Any) -> Tensor:
    return _permute_frontend(x, (2, 1, 0))


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


def _infer_shape_spec(shape_specs: list[list[Any]], out_shape: list[int]) -> list[Any]:
    if not shape_specs:
        return list(out_shape)
    result: list[Any] = []
    max_rank = builtins.max(len(shape) for shape in shape_specs)
    aligned = [[1] * (max_rank - len(shape)) + list(shape) for shape in shape_specs]
    for dims in zip(*aligned):
        chosen = dims[0]
        for dim in dims[1:]:
            if _dim_is_one(chosen):
                chosen = dim
            elif _dim_is_one(dim):
                continue
            elif dim == chosen:
                continue
            else:
                return list(out_shape)
        result.append(chosen)
    return result


def _dim_is_one(dim: Any) -> bool:
    return isinstance(dim, int) and int(dim) == 1


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
globals()["full"] = _full_frontend
globals()["arange"] = _arange_frontend
globals()["randn"] = _randn_frontend
globals()["expand"] = _expand_frontend
globals()["expand_static_shape"] = _expand_static_shape_frontend
globals()["meshgrid"] = _meshgrid_frontend
globals()["concatenate"] = _concatenate_frontend
globals()["concatenate_fast"] = _concatenate_fast_frontend
globals()["concatenate_tanh"] = _concatenate_tanh_frontend
globals()["dynamic_slice"] = _dynamic_slice_frontend
globals()["index_select"] = _index_select_frontend
globals()["slice_scatter"] = _slice_scatter_frontend
globals()["slice_reshape_scatter"] = _slice_reshape_scatter_frontend
globals()["split"] = _split_frontend
globals()["chunk"] = _chunk_frontend
globals()["stack"] = _stack_frontend
globals()["flip"] = _flip_frontend
globals()["permute"] = _permute_frontend
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
    "identity",
    "make_frontend_op",
    "meshgrid",
    "output",
    "size",
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
    "squeeze",
    "unsqueeze",
    "var",
    "vector_norm",
    "arange",
    "cast",
    "chunk",
    "concatenate",
    "concatenate_fast",
    "concatenate_tanh",
    "dynamic_slice",
    "full",
    "index_select",
    "slice_reshape_scatter",
    "slice_scatter",
    "split",
    "where",
]))
