from __future__ import annotations

from typing import Any, Mapping, Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpRegistry, OpSchema


REDUCTION_OPS = ("reduce_sum", "reduce_max", "reduce_min", "reduce_mean", "var", "vector_norm")
BASIC_REDUCTION_OPS = ("reduce_sum", "reduce_max", "reduce_min", "reduce_mean")
REDUCTION_DTYPES = ("float16", "float32", "bfloat16")
ARGMAX_DTYPES = ("float16", "float32", "bfloat16", "bool")


def infer_reduction(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 1:
        raise ValueError("reduction expects exactly one input")
    if not shapes[0]:
        raise ValueError("reduction requires a ranked tensor")
    return list(shapes[0][:-1]) or [1]


def infer_reduction_with_attrs(shape: Sequence[int], keepdim: bool) -> list[int]:
    if not shape:
        raise ValueError("reduction requires a ranked tensor")
    if keepdim:
        out = list(shape)
        out[-1] = 1
        return out
    return list(shape[:-1]) or [1]


def infer_reduction_for_attrs(shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(shapes) != 1:
        raise ValueError("reduction expects exactly one input")
    return infer_reduction_with_attrs(shapes[0], bool(attrs.get("keepdim", False)))


def infer_argmax_shape(shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_argmax_shape_with_attrs(shapes, {"dim": -1, "keepdim": False})


def infer_argmax_shape_with_attrs(shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    if len(shapes) != 1:
        raise ValueError("argmax expects exactly one input")
    return resolve_argmax_shape(shapes[0], attrs.get("dim", -1), bool(attrs.get("keepdim", False)))


def normalize_argmax_dim(dim: Any, rank: int) -> int:
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"argmax dim must be an integer, got {dim!r}")
    if rank <= 0:
        raise ValueError("argmax requires a ranked tensor")
    normalized = int(dim)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"argmax dim {dim} is out of range for rank {rank}")
    return normalized


def resolve_argmax_shape(shape: Sequence[int], dim: Any, keepdim: bool) -> list[int]:
    axis = normalize_argmax_dim(dim, len(shape))
    if axis != len(shape) - 1:
        raise NotImplementedError("argmax currently supports only the last dimension")
    if int(shape[axis]) <= 0:
        raise ValueError("argmax last dimension must be positive")
    out_shape = list(shape)
    if keepdim:
        out_shape[axis] = 1
    else:
        del out_shape[axis]
        if not out_shape:
            out_shape = [1]
    return [int(dim) for dim in out_shape]


def register_reduction_ops(registry: OpRegistry) -> None:
    for op_name in REDUCTION_OPS:
        attrs = [AttrDef("dim", "int", -1), AttrDef("keepdim", "bool", False)]
        allowed_dtypes = REDUCTION_DTYPES
        description = "Dense reduction over a static last dimension with fp32 accumulation."
        if op_name == "var":
            attrs.append(AttrDef("unbiased", "bool", False))
            allowed_dtypes = ("float32",)
            description = "Dense float32 variance reduction over a static last dimension."
        elif op_name == "vector_norm":
            attrs.append(AttrDef("ord", "float", 2.0))
            allowed_dtypes = ("float32",)
            description = "Dense float32 vector norm reduction over a static last dimension."
        registry.register(
            OpDef(
                name=op_name,
                schema=OpSchema(inputs=("x",), attrs=tuple(attrs)),
                infer_shape=infer_reduction,
                infer_shape_with_attrs=infer_reduction_for_attrs,
                backend_kernels={
                    "cuda": KernelBinding("generated_reduction", "model", source_template="reduction_cuda"),
                    "cpu": KernelBinding("generated_reduction", "model", source_template="reduction_cpu"),
                },
                allowed_dtypes=allowed_dtypes,
                description=description,
            )
        )
    registry.register(
        OpDef(
            name="argmax",
            schema=OpSchema(inputs=("x",), attrs=(AttrDef("dim", "int", -1), AttrDef("keepdim", "bool", False))),
            infer_shape=infer_argmax_shape,
            infer_shape_with_attrs=infer_argmax_shape_with_attrs,
            backend_kernels={
                "cuda": KernelBinding("generated_argmax", "model", source_template="argmax_cuda"),
                "cpu": KernelBinding("generated_argmax", "model", source_template="argmax_cpu"),
            },
            frontend=FrontendBinding("argmax"),
            allowed_dtypes=ARGMAX_DTYPES,
            description="Dense argmax over a positive static last dimension, returning int64 indices.",
        )
    )


def reduce_sum(x: object, dim: int = -1, keepdim: bool = False) -> Tensor:
    return _reduction("reduce_sum", x, dim, keepdim)


def reduce_max(x: object, dim: int = -1, keepdim: bool = False) -> Tensor:
    return _reduction("reduce_max", x, dim, keepdim)


def reduce_min(x: object, dim: int = -1, keepdim: bool = False) -> Tensor:
    return _reduction("reduce_min", x, dim, keepdim)


def reduce_mean(x: object, dim: int = -1, keepdim: bool = False) -> Tensor:
    return _reduction("reduce_mean", x, dim, keepdim)


def var(x: object, dim: int = -1, keepdim: bool = False, unbiased: bool = False) -> Tensor:
    return _reduction("var", x, dim, keepdim, {"unbiased": bool(unbiased)})


def vector_norm(x: object, dim: int = -1, keepdim: bool = False, ord: float = 2.0) -> Tensor:
    if float(ord) != 2.0:
        raise NotImplementedError("vector_norm currently supports only ord=2")
    return _reduction("vector_norm", x, dim, keepdim, {"ord": 2.0})


def argmax(x: object, dim: int = -1, keepdim: bool = False) -> Tensor:
    tensor = as_tensor(x, dtype_hint="float32")
    if tensor.dtype not in ARGMAX_DTYPES:
        raise ValueError(f"argmax does not support dtype {tensor.dtype}")
    if tensor.dynamic:
        raise ValueError("argmax currently supports only static input shapes")
    axis = normalize_argmax_dim(dim, tensor.rank)
    if axis != tensor.rank - 1:
        raise NotImplementedError("argmax currently supports only the last dimension")
    if not isinstance(tensor.shape_spec[axis], int):
        raise ValueError("argmax currently requires a static last dimension")
    out_shape = resolve_argmax_shape(tensor.shape, axis, bool(keepdim))
    out_shape_spec = list(tensor.shape_spec)
    if keepdim:
        out_shape_spec[axis] = 1
    else:
        del out_shape_spec[axis]
        if not out_shape_spec:
            out_shape_spec = [1]
    return tensor.builder.emit(
        "argmax",
        [tensor],
        out_shape,
        "int64",
        {"dim": axis, "keepdim": bool(keepdim)},
        shape_spec=out_shape_spec,
    )


def _reduction(op_name: str, x: object, dim: int, keepdim: bool, extra_attrs: dict[str, object] | None = None) -> Tensor:
    tensor = as_tensor(x, dtype_hint="float32")
    allowed_dtypes = REDUCTION_DTYPES if op_name in BASIC_REDUCTION_OPS else ("float32",)
    if tensor.dtype not in allowed_dtypes:
        raise ValueError(f"{op_name} does not support dtype {tensor.dtype}")
    rank = len(tensor.shape)
    if rank == 0:
        raise ValueError(f"{op_name} requires a ranked tensor")
    axis = _normalize_axis(dim, rank)
    if axis != rank - 1:
        raise NotImplementedError(f"{op_name} currently supports only the last dimension")
    if not isinstance(tensor.shape_spec[axis], int):
        raise ValueError(f"{op_name} currently requires a static last dimension")
    if int(tensor.shape[axis]) <= 0:
        raise ValueError(f"{op_name} last dimension must be positive")
    out_shape = list(tensor.shape)
    out_shape_spec = list(tensor.shape_spec)
    if keepdim:
        out_shape[axis] = 1
        out_shape_spec[axis] = 1
    else:
        del out_shape[axis]
        del out_shape_spec[axis]
        if not out_shape:
            out_shape = [1]
            out_shape_spec = [1]
    attrs = {"dim": axis, "keepdim": bool(keepdim)}
    if extra_attrs is not None:
        attrs.update(extra_attrs)
    return tensor.builder.emit(
        op_name,
        [tensor],
        out_shape,
        tensor.dtype,
        attrs,
        shape_spec=out_shape_spec,
    )


def _normalize_axis(axis: int, rank: int) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"reduction dim {axis} is out of range for rank {rank}")
    return normalized
