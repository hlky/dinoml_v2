from __future__ import annotations

from typing import Any, Mapping, Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.collections import GATHER_INDEX_DTYPES
from dinoml.ops.registry import FrontendBinding, KernelBinding, OpDef, OpSchema, op_def


EMBEDDING_DTYPES = ("float16", "float32", "bfloat16")


def infer_embedding_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_embedding_shape_with_attrs(input_shapes, {})


def infer_embedding_shape_with_attrs(input_shapes: Sequence[Sequence[int]], attrs: Mapping[str, Any]) -> list[int]:
    del attrs
    if len(input_shapes) != 2:
        raise ValueError("embedding expects exactly two inputs")
    return resolve_embedding_shape(input_shapes[0], input_shapes[1])


def resolve_embedding_shape(table_shape: Sequence[int], index_shape: Sequence[int]) -> list[int]:
    normalize_embedding_shapes(table_shape, index_shape)
    return [*[int(axis) for axis in index_shape], int(table_shape[1])]


def normalize_embedding_shapes(table_shape: Sequence[int], index_shape: Sequence[int]) -> None:
    if len(table_shape) != 2:
        raise ValueError(f"embedding table rank {len(table_shape)} must be 2")
    if len(index_shape) < 1:
        raise ValueError("embedding indices rank must be at least 1")
    vocab = int(table_shape[0])
    hidden = int(table_shape[1])
    if vocab <= 0:
        raise ValueError(f"embedding vocab size must be positive, got {vocab}")
    if hidden <= 0:
        raise ValueError(f"embedding hidden size must be positive, got {hidden}")


@op_def
class Embedding(OpDef):
    name = "embedding"
    schema = OpSchema(inputs=("table", "indices"))
    infer_shape = infer_embedding_shape
    infer_shape_with_attrs = infer_embedding_shape_with_attrs
    allowed_dtypes = EMBEDDING_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(
            symbol="generated_embedding",
            library="model",
            source_template="embedding_cpu.cpp.j2",
        ),
        "cuda": KernelBinding(
            symbol="generated_embedding",
            library="model",
            source_template="embedding_gpu.j2",
        ),
        "rocm": KernelBinding(
            symbol="generated_embedding",
            library="model",
            source_template="embedding_gpu.j2",
        ),
    }
    frontend = FrontendBinding("embedding")
    description = (
        "Bounded learned embedding lookup for a static table [vocab, hidden] "
        "and rank >= 1 int64/int32 indices, preserving dynamic leading index dims "
        "and returning output shaped [..., hidden]."
    )

    @classmethod
    def forward(cls, table: Any, indices: Any) -> Tensor:
        table_tensor = as_tensor(table, dtype_hint="float32")
        index_tensor = as_tensor(indices, dtype_hint="int64")
        if table_tensor.builder is not index_tensor.builder:
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if table_tensor.dtype not in EMBEDDING_DTYPES:
            raise ValueError(f"embedding does not support dtype {table_tensor.dtype}")
        if index_tensor.dtype not in GATHER_INDEX_DTYPES:
            raise ValueError(f"embedding indices must have dtype int64 or int32, got {index_tensor.dtype}")
        if table_tensor.rank != 2:
            raise ValueError(f"embedding table rank {table_tensor.rank} must be 2")
        if index_tensor.rank < 1:
            raise ValueError("embedding indices rank must be at least 1")
        if table_tensor.dynamic or any(not isinstance(dim, int) for dim in table_tensor.shape_spec):
            raise ValueError("embedding currently requires a static table shape [vocab, hidden]")
        normalize_embedding_shapes(table_tensor.shape, index_tensor.shape)
        hidden = int(table_tensor.shape[1])
        output_shape = infer_embedding_shape_with_attrs([table_tensor.shape, index_tensor.shape], {})
        output_shape_spec = [_copy_shape_dim(dim) for dim in index_tensor.shape_spec] + [hidden]
        return table_tensor.builder.emit(
            "embedding",
            [table_tensor, index_tensor],
            output_shape,
            table_tensor.dtype,
            {},
            shape_spec=output_shape_spec,
        )


def embedding(table: Any, indices: Any) -> Tensor:
    return Embedding.forward(table, indices)


def _copy_shape_dim(dim: Any) -> Any:
    return dict(dim) if isinstance(dim, Mapping) else dim
__all__ = [
    "EMBEDDING_DTYPES",
    "Embedding",
    "embedding",
    "infer_embedding_shape",
    "infer_embedding_shape_with_attrs",
    "normalize_embedding_shapes",
    "resolve_embedding_shape",
]
