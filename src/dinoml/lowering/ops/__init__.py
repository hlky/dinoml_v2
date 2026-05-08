from __future__ import annotations

from typing import Any, Mapping, Sequence

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.fused_elementwise import FUSED_ELEMENTWISE_LOWERING
from dinoml.ops.elementwise import FUSABLE_ELEMENTWISE_OPS


OP_LOWERINGS: dict[str, OpLowering] = {
    FUSED_ELEMENTWISE_LOWERING.op_name: FUSED_ELEMENTWISE_LOWERING,
}


def render_generated_kernels(
    target: str,
    nodes: Sequence[Mapping[str, Any]],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    kernels = []
    seen_source_keys: set[str] = set()
    for node in nodes:
        lowering = OP_LOWERINGS.get(node["op"])
        if lowering is None:
            continue
        source_key = lowering.source_key(target, node, tensor_map) if lowering.source_key else None
        if source_key is not None and source_key in seen_source_keys:
            continue
        kernel = lowering.render_generated_kernel(target, node, tensor_map)
        if kernel:
            source_key = source_key or kernel
            if source_key in seen_source_keys:
                continue
            seen_source_keys.add(source_key)
            kernels.append(kernel)
    return kernels


def render_launch(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    if node["op"] in FUSABLE_ELEMENTWISE_OPS:
        raise ValueError(f"{node['op']} must be lowered through fused_elementwise before {target} codegen")
    try:
        lowering = OP_LOWERINGS[node["op"]]
    except KeyError as exc:
        raise ValueError(f"Unsupported op for {target} lowering: {node['op']}") from exc
    return lowering.render_launch(target, node, tensor_map)


__all__ = ["OP_LOWERINGS", "render_generated_kernels", "render_launch"]
