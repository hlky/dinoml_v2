from __future__ import annotations

from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> None:
    del target, node, tensor_map
    return None


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del tensor_map, kernel_manifest
    if target != "cuda":
        raise ValueError(f"{node['op']} lowering is currently CUDA-only")
    raise NotImplementedError(
        "conv2d_bias CUDA lowering is not implemented yet; current slice provides "
        "frontend/admission, CPU reference execution, and CUTLASS manifest scaffolding only"
    )


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> None:
    del target, node, tensor_map
    return None


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> None:
    del target, node, tensor_map
    return None


CONV2D_BIAS_LOWERING = OpLowering(
    op_name="conv2d_bias",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


__all__ = ["CONV2D_BIAS_LOWERING"]
