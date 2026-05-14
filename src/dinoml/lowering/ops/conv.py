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


def render_scaffold_wrapper_stage(stage: Mapping[str, Any]) -> str:
    stage_kind = str(stage.get("stage_kind", ""))
    symbol = str(stage.get("symbol", ""))
    if not symbol:
        raise ValueError("CUTLASS Conv scaffold wrapper stage is missing a symbol")
    if stage_kind == "transform_helper":
        source = _descriptor_placeholder(stage.get("source"))
        destination = _descriptor_placeholder(stage.get("destination"))
        shape_args = _shape_placeholders(stage)
        return (
            f"DINO_CUDA_CHECK({symbol}("
            f"{', '.join([source, destination, *shape_args, 'stream'])}));"
        )
    if stage_kind == "provider_launcher":
        inputs = stage.get("inputs")
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("CUTLASS Conv scaffold provider_launcher stage inputs must be a list")
        output = _descriptor_placeholder(stage.get("output"))
        pointer_args = [_descriptor_placeholder(item) for item in inputs]
        shape_args = _shape_placeholders(stage)
        status_name = f"status_{_c_ident(str(stage.get('stage_name', 'cutlass_conv')))}"
        return (
            f"int {status_name} = {symbol}("
            f"{', '.join([*pointer_args, output, *shape_args, 'stream'])});\n"
            f"if ({status_name} != 0) {{\n"
            f"  return {status_name};\n"
            f"}}"
        )
    raise ValueError(f"Unsupported CUTLASS Conv scaffold wrapper stage kind {stage_kind!r}")


def render_scaffold_wrapper_stages(stages: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> list[str]:
    return [render_scaffold_wrapper_stage(stage) for stage in stages]


def _descriptor_placeholder(descriptor: Any) -> str:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"Malformed CUTLASS Conv scaffold descriptor: {descriptor!r}")
    kind = str(descriptor.get("kind", ""))
    if kind == "semantic_tensor":
        role = str(descriptor.get("role", ""))
        if role not in {"activation", "weight", "bias", "output"}:
            raise ValueError(f"Unsupported CUTLASS Conv scaffold semantic tensor role {role!r}")
        return f"ptr_{role}"
    if kind == "temporary_buffer":
        name = str(descriptor.get("name", ""))
        if not name:
            raise ValueError("CUTLASS Conv scaffold temporary buffer descriptor is missing name")
        return f"tmp_{_c_ident(name)}"
    raise ValueError(f"Unsupported CUTLASS Conv scaffold descriptor kind {kind!r}")


def _shape_placeholders(stage: Mapping[str, Any]) -> list[str]:
    raw_args = stage.get("shape_args")
    if not isinstance(raw_args, (list, tuple)):
        raise ValueError("CUTLASS Conv scaffold wrapper stage shape_args must be a list")
    placeholders = []
    for item in raw_args:
        if not isinstance(item, Mapping) or not str(item.get("placeholder", "")):
            raise ValueError(f"Malformed CUTLASS Conv scaffold shape arg descriptor: {item!r}")
        placeholders.append(str(item["placeholder"]))
    return placeholders


def _c_ident(name: str) -> str:
    pieces = []
    for char in str(name):
        pieces.append(char if char.isalnum() else "_")
    ident = "".join(pieces).strip("_")
    return ident or "tmp"


CONV2D_BIAS_LOWERING = OpLowering(
    op_name="conv2d_bias",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


__all__ = [
    "CONV2D_BIAS_LOWERING",
    "render_scaffold_wrapper_stage",
    "render_scaffold_wrapper_stages",
]
