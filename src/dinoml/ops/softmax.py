from __future__ import annotations

from typing import Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpSchema, op_def


def infer_softmax(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 1:
        raise ValueError("softmax expects exactly one input")
    if len(shapes[0]) < 2:
        raise ValueError("softmax requires rank >= 2")
    return list(shapes[0])


@op_def
class Softmax(OpDef):
    name = "softmax"
    schema = OpSchema(inputs=("x",), attrs=(AttrDef("dim", "int", -1),))
    infer_shape = infer_softmax
    backend_kernels = {
        "cuda": KernelBinding("generated_softmax", "model", source_template="softmax_gpu"),
        "rocm": KernelBinding("generated_softmax", "model", source_template="softmax_gpu"),
        "cpu": KernelBinding("generated_softmax", "model", source_template="softmax_cpu"),
    }
    frontend = FrontendBinding("softmax")
    allowed_dtypes = ("float16", "float32", "bfloat16")
    description = (
        "Dense float16, float32, and bfloat16 softmax over the last dimension. "
        "Uses fp32 computation for reduced-precision storage."
    )

    @classmethod
    def forward(cls, x: object, dim: int = -1) -> Tensor:
        tensor = as_tensor(x, dtype_hint="float32")
        if tensor.dtype not in cls.allowed_dtypes:
            raise ValueError(f"softmax does not support dtype {tensor.dtype}")
        rank = len(tensor.shape)
        if rank < 2:
            raise ValueError("softmax requires rank >= 2")
        axis = _normalize_axis(dim, rank)
        if axis != rank - 1:
            raise NotImplementedError("softmax currently supports only the last dimension")
        if int(tensor.shape[axis]) <= 0:
            raise ValueError("softmax last dimension must be positive")
        return tensor.builder.emit("softmax", [tensor], tensor.shape, tensor.dtype, {"dim": axis}, shape_spec=tensor.shape_spec)


def softmax(x: object, dim: int = -1) -> Tensor:
    return Softmax.forward(x, dim)


def _normalize_axis(axis: int, rank: int) -> int:
    if rank <= 0:
        raise ValueError("softmax requires a ranked tensor")
    normalized = int(axis)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"softmax dim {axis} is out of range for rank {rank}")
    return normalized
