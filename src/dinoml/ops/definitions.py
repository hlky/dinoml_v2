from __future__ import annotations

from dinoml.ops.registry import (
    AttrDef,
    FrontendBinding,
    KernelBinding,
    KernelVariant,
    OpDef,
    OP_REGISTRY,
    OpRegistry,
    OpSchema,
)

OP_DEFINITIONS = OP_REGISTRY.definitions


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
]
