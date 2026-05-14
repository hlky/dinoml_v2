from __future__ import annotations

from dinoml.ops.bmm import register_bmm_ops
from dinoml.ops.broadcasting import register_broadcasting_ops
from dinoml.ops.collections import register_collection_ops
from dinoml.ops.conv import register_conv_ops
from dinoml.ops.creation import register_creation_ops
from dinoml.ops.elementwise import register_elementwise_ops
from dinoml.ops.gemm import register_gemm_ops
from dinoml.ops.internal import register_internal_ops
from dinoml.ops.normalization import register_normalization_ops
from dinoml.ops.positional import register_positional_ops
from dinoml.ops.pooling import register_pooling_ops
from dinoml.ops.reductions import register_reduction_ops
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, KernelVariant, OpDef, OpRegistry, OpSchema
from dinoml.ops.softmax import register_softmax_op


OP_REGISTRY = OpRegistry()
register_elementwise_ops(OP_REGISTRY)
register_bmm_ops(OP_REGISTRY)
register_gemm_ops(OP_REGISTRY)
register_conv_ops(OP_REGISTRY)
register_normalization_ops(OP_REGISTRY)
register_softmax_op(OP_REGISTRY)
register_reduction_ops(OP_REGISTRY)
register_pooling_ops(OP_REGISTRY)
register_creation_ops(OP_REGISTRY)
register_positional_ops(OP_REGISTRY)
register_broadcasting_ops(OP_REGISTRY)
register_collection_ops(OP_REGISTRY)
register_internal_ops(OP_REGISTRY)
OP_DEFINITIONS = OP_REGISTRY.definitions


def register_op(op_def: OpDef) -> OpDef:
    return OP_REGISTRY.register(op_def)


def get_op_def(op_name: str) -> OpDef:
    return OP_REGISTRY.get(op_name)


__all__ = [
    "AttrDef",
    "FrontendBinding",
    "KernelBinding",
    "KernelVariant",
    "OP_DEFINITIONS",
    "OP_REGISTRY",
    "OpDef",
    "OpRegistry",
    "OpSchema",
    "get_op_def",
    "register_op",
]
