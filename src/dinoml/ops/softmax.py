from __future__ import annotations

from typing import Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.registry import AttrDef, KernelBinding, OpDef, OpRegistry, OpSchema


def infer_softmax(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 1:
        raise ValueError("softmax expects exactly one input")
    if len(shapes[0]) < 2:
        raise ValueError("softmax requires rank >= 2")
    return list(shapes[0])


def register_softmax_op(registry: OpRegistry) -> None:
    registry.register(
        OpDef(
            name="softmax",
            schema=OpSchema(inputs=("x",), attrs=(AttrDef("dim", "int", -1),)),
            infer_shape=infer_softmax,
            backend_kernels={
                "cuda": KernelBinding("generated_softmax", "model", source_template="softmax_cuda"),
                "cpu": KernelBinding("generated_softmax", "model", source_template="softmax_cpu"),
            },
            allowed_dtypes=("float32",),
            description=(
                "Dense float32 softmax over the last dimension. Initial v2 port "
                "requires a static last-axis reduction extent."
            ),
        )
    )


def softmax(x: object, dim: int = -1) -> Tensor:
    tensor = as_tensor(x, dtype_hint="float32")
    if tensor.dtype != "float32":
        raise ValueError(f"softmax does not support dtype {tensor.dtype}")
    rank = len(tensor.shape)
    if rank < 2:
        raise ValueError("softmax requires rank >= 2")
    axis = _normalize_axis(dim, rank)
    if axis != rank - 1:
        raise NotImplementedError("softmax currently supports only the last dimension")
    if not isinstance(tensor.shape_spec[axis], int):
        raise ValueError("softmax currently requires a static last dimension")
    if int(tensor.shape[axis]) <= 0:
        raise ValueError("softmax last dimension must be positive")
    return tensor.builder.emit("softmax", [tensor], tensor.shape, tensor.dtype, {"dim": axis}, shape_spec=tensor.shape_spec)


def _normalize_axis(axis: int, rank: int) -> int:
    if rank <= 0:
        raise ValueError("softmax requires a ranked tensor")
    normalized = int(axis)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"softmax dim {axis} is out of range for rank {rank}")
    return normalized
