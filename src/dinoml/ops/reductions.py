from __future__ import annotations

from typing import Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.registry import AttrDef, KernelBinding, OpDef, OpRegistry, OpSchema


REDUCTION_OPS = ("reduce_sum", "reduce_max", "reduce_min", "reduce_mean", "var", "vector_norm")


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


def register_reduction_ops(registry: OpRegistry) -> None:
    for op_name in REDUCTION_OPS:
        attrs = [AttrDef("dim", "int", -1), AttrDef("keepdim", "bool", False)]
        if op_name == "var":
            attrs.append(AttrDef("unbiased", "bool", False))
        elif op_name == "vector_norm":
            attrs.append(AttrDef("ord", "float", 2.0))
        registry.register(
            OpDef(
                name=op_name,
                schema=OpSchema(inputs=("x",), attrs=tuple(attrs)),
                infer_shape=infer_reduction,
                backend_kernels={
                    "cuda": KernelBinding("generated_reduction", "model", source_template="reduction_cuda"),
                    "cpu": KernelBinding("generated_reduction", "model", source_template="reduction_cpu"),
                },
                allowed_dtypes=("float32",),
                description="Dense float32 reduction over a static last dimension.",
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


def _reduction(op_name: str, x: object, dim: int, keepdim: bool, extra_attrs: dict[str, object] | None = None) -> Tensor:
    tensor = as_tensor(x, dtype_hint="float32")
    if tensor.dtype != "float32":
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
