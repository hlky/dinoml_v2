from __future__ import annotations

from dinoml.ops.registry import KernelBinding, OpDef, OpSchema, op_def


def _flattened_shape(shapes):
    numel = 1
    for dim in shapes[0]:
        numel *= int(dim)
    return [numel]


@op_def
class ShapeBufferCountTrue(OpDef):
    name = "_shape_buffer_count_true"
    schema = OpSchema(inputs=("x",))
    infer_shape = _flattened_shape
    allowed_dtypes = ("bool",)
    backend_kernels = {
        "cpu": KernelBinding(symbol="generated_shape_buffer_count_true", library="model"),
        "cuda": KernelBinding(symbol="generated_shape_buffer_count_true", library="model"),
    }
    description = (
        "Internal fixture op that counts true bool elements and publishes "
        "the count through the output shape buffer."
    )


__all__ = ["ShapeBufferCountTrue"]
