from __future__ import annotations

from typing import Any, Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.registry import FrontendBinding, KernelBinding, OpDef, OpSchema, op_def


SWIGLU_DTYPES = ("float16", "float32", "bfloat16")


def infer_swiglu_shape(shapes: Sequence[Sequence[int]]) -> list[int]:
    if len(shapes) != 1:
        raise ValueError(f"swiglu expects one input, got {len(shapes)}")
    shape = list(shapes[0])
    if not shape:
        raise ValueError("swiglu expects rank >= 1 input")
    last_dim = shape[-1]
    if not isinstance(last_dim, int) or last_dim % 2 != 0:
        raise ValueError("swiglu expects a static even last dimension")
    shape[-1] = last_dim // 2
    return shape


@op_def
class SwiGLU(OpDef):
    name = "swiglu"
    schema = OpSchema(inputs=("x",))
    infer_shape = infer_swiglu_shape
    backend_kernels = {
        "cpu": KernelBinding("generated_swiglu", "model", source_template="swiglu_cpu"),
        "cuda": KernelBinding("generated_swiglu", "model", source_template="swiglu_gpu"),
        "rocm": KernelBinding("generated_swiglu", "model", source_template="swiglu_gpu"),
    }
    frontend = FrontendBinding("swiglu")
    allowed_dtypes = SWIGLU_DTYPES
    description = "SwiGLU activation over a [..., 2 * hidden] tensor: silu(x[..., :hidden]) * x[..., hidden:]."

    @classmethod
    def forward(cls, x: Any) -> Tensor:
        tensor = as_tensor(x, dtype_hint="float32")
        if tensor.dtype not in SWIGLU_DTYPES:
            raise ValueError(f"swiglu does not support dtype {tensor.dtype}")
        out_shape = infer_swiglu_shape([tensor.shape])
        out_shape_spec = list(tensor.shape_spec)
        out_shape_spec[-1] = out_shape[-1]
        return tensor.builder.emit(
            "swiglu",
            [tensor],
            out_shape,
            tensor.dtype,
            {},
            shape_spec=out_shape_spec,
        )


def swiglu(x: Any) -> Tensor:
    return SwiGLU.forward(x)


__all__ = ["SWIGLU_DTYPES", "SwiGLU", "infer_swiglu_shape", "swiglu"]
