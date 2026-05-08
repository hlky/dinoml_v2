from __future__ import annotations

from typing import Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.registry import AttrDef, KernelBinding, OpDef, OpRegistry, OpSchema


REDUCTION_OPS = ("reduce_sum", "reduce_max", "reduce_min", "reduce_mean")


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
        registry.register(
            OpDef(
                name=op_name,
                schema=OpSchema(inputs=("x",), attrs=(AttrDef("dim", "int", -1), AttrDef("keepdim", "bool", False))),
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


def _reduction(op_name: str, x: object, dim: int, keepdim: bool) -> Tensor:
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
    out = tensor.builder.emit(op_name, [tensor], out_shape, tensor.dtype, {"dim": axis, "keepdim": bool(keepdim)})
    out.shape_spec = out_shape_spec
    tensor.builder.tensors[out.name]["shape_spec"] = out_shape_spec
    return out


def _normalize_axis(axis: int, rank: int) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"reduction dim {axis} is out of range for rank {rank}")
    return normalized
