from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.kernels.providers.cutlass.conv import cutlass_conv_wrapper_stages
from dinoml.lowering.cpp_types import cpu_storage_type
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.ops.conv import CONV2D_BIAS_DTYPES, normalize_conv2d_bias_attrs, resolve_conv2d_bias_shape


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    if target == "cpu":
        return _render_cpu_template("conv_cpu.cpp.j2", _cpu_context(node, tensor_map))
    if target == "cuda":
        return None
    raise ValueError(f"Unsupported Conv lowering target: {target}")


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    op_name = str(node["op"])
    if target == "cpu":
        func = _cpu_function_name(node, tensor_map)
        x_ident = _c_ident(str(node["inputs"][0]))
        weight_ident = _c_ident(str(node["inputs"][1]))
        bias_ident = _c_ident(str(node["inputs"][2]))
        out_ident = _c_ident(str(node["outputs"][0]))
        return (
            "if (int err = "
            f"{func}(ptr_{x_ident}, runtime_numel_{x_ident}, ptr_{weight_ident}, runtime_numel_{weight_ident}, "
            f"ptr_{bias_ident}, runtime_numel_{bias_ident}, ptr_{out_ident}, runtime_numel_{out_ident})) return err;"
        )
    if target != "cuda":
        raise ValueError(f"Unsupported Conv lowering target: {target}")
    if op_name != "conv2d_bias":
        raise NotImplementedError(f"{op_name} CUDA Conv lowering is not implemented")
    item = _manifest_kernel_item(kernel_manifest, op_name, node_id=str(node.get("id", "")))
    if item is None:
        raise ValueError(f"{op_name} CUDA lowering requires a CUTLASS Conv manifest entry")
    stages = cutlass_conv_wrapper_stages({"required_kernels": [item]})
    if [str(stage.get("stage_name")) for stage in stages] != [
        "activation_pack",
        "weight_pack",
        "provider_launch",
        "output_unpack",
    ]:
        raise ValueError(f"{op_name} CUTLASS Conv wrapper stages are malformed")

    input_names = [str(name) for name in node.get("inputs", ())]
    if len(input_names) != 3:
        raise ValueError(f"{op_name} CUDA lowering expects activation, weight, and bias inputs")
    output_names = [str(name) for name in node.get("outputs", ())]
    if len(output_names) != 1:
        raise ValueError(f"{op_name} CUDA lowering expects one output")
    roles = {
        "activation": _c_ident(input_names[0]),
        "weight": _c_ident(input_names[1]),
        "bias": _c_ident(input_names[2]),
        "output": _c_ident(output_names[0]),
    }
    _validate_runtime_shape_contract(op_name, tensor_map, input_names, output_names[0], item)

    lines = [
        f'// CUTLASS Conv wrapper lowering for {op_name}: NCHW/OIHW semantic tensors are packed to NHWC/OHWI before the provider call.',
    ]
    for stage in stages:
        lines.extend(_render_runtime_stage(stage, roles=roles))
    return "\n".join(lines)


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    if target == "cpu":
        return f"{target}:{generated_function_name(target, node, tensor_map)}"
    if target == "cuda":
        return None
    raise ValueError(f"Unsupported Conv lowering target: {target}")


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str | None:
    if target == "cpu":
        return _cpu_function_name(node, tensor_map)
    if target == "cuda":
        return None
    raise ValueError(f"Unsupported Conv lowering target: {target}")


def _cpu_context(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    x_name, weight_name, bias_name = (str(name) for name in node["inputs"])
    out_name = str(node["outputs"][0])
    x = tensor_map[x_name]
    weight = tensor_map[weight_name]
    bias = tensor_map[bias_name]
    output = tensor_map[out_name]
    attrs = _validate_cpu_contract(node, x, weight, bias, output)
    batch, in_channels, in_height, in_width = [int(dim) for dim in x["shape"]]
    out_channels, _weight_in_channels, kernel_h, kernel_w = [int(dim) for dim in weight["shape"]]
    out_batch, out_channels_out, out_height, out_width = [int(dim) for dim in output["shape"]]
    return {
        "func": _cpu_function_name(node, tensor_map),
        "storage_type": cpu_storage_type(str(output["dtype"])),
        "input_numel": batch * in_channels * in_height * in_width,
        "weight_numel": out_channels * in_channels * kernel_h * kernel_w,
        "bias_numel": out_channels,
        "output_numel": out_batch * out_channels_out * out_height * out_width,
        "batch": batch,
        "in_channels": in_channels,
        "in_height": in_height,
        "in_width": in_width,
        "out_channels": out_channels,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "out_height": out_height,
        "out_width": out_width,
        "stride_h": attrs["stride"][0],
        "stride_w": attrs["stride"][1],
        "pad_h": attrs["padding"][0],
        "pad_w": attrs["padding"][1],
        "dilation_h": attrs["dilation"][0],
        "dilation_w": attrs["dilation"][1],
    }


def _cpu_function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    x_name, weight_name, bias_name = (str(name) for name in node["inputs"])
    out_name = str(node["outputs"][0])
    x = tensor_map[x_name]
    weight = tensor_map[weight_name]
    bias = tensor_map[bias_name]
    output = tensor_map[out_name]
    attrs = _validate_cpu_contract(node, x, weight, bias, output)
    signature = {
        "op": str(node["op"]),
        "input_shape": [int(dim) for dim in x["shape"]],
        "weight_shape": [int(dim) for dim in weight["shape"]],
        "bias_shape": [int(dim) for dim in bias["shape"]],
        "output_shape": [int(dim) for dim in output["shape"]],
        "stride": attrs["stride"],
        "padding": attrs["padding"],
        "dilation": attrs["dilation"],
        "groups": attrs["groups"],
        "dtype": str(output["dtype"]),
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"conv2d_bias_{digest}"


def _validate_cpu_contract(
    node: Mapping[str, Any],
    x: Mapping[str, Any],
    weight: Mapping[str, Any],
    bias: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any]:
    if str(node["op"]) != "conv2d_bias":
        raise ValueError(f"Unsupported Conv op for CPU lowering: {node['op']}")
    dtype = str(output["dtype"])
    if dtype not in CONV2D_BIAS_DTYPES:
        raise NotImplementedError(f"conv2d_bias CPU lowering does not support dtype {dtype!r}")
    if str(x["dtype"]) != dtype or str(weight["dtype"]) != dtype or str(bias["dtype"]) != dtype:
        raise ValueError("conv2d_bias CPU lowering requires matching activation, weight, bias, and output dtypes")
    attrs = node.get("attrs", {})
    stride, padding, dilation, groups = normalize_conv2d_bias_attrs(
        attrs.get("stride", (1, 1)),
        attrs.get("padding", (0, 0)),
        attrs.get("dilation", (1, 1)),
        attrs.get("groups", 1),
    )
    expected_shape = resolve_conv2d_bias_shape(
        x["shape"],
        weight["shape"],
        bias["shape"],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    if [int(dim) for dim in output["shape"]] != expected_shape:
        raise ValueError("conv2d_bias CPU lowering output shape does not match attrs")
    return {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
    }


def _render_cpu_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


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


def _render_runtime_stage(stage: Mapping[str, Any], *, roles: Mapping[str, str]) -> list[str]:
    stage_kind = str(stage.get("stage_kind", ""))
    symbol = str(stage.get("symbol", ""))
    if not symbol:
        raise ValueError("CUTLASS Conv wrapper stage is missing a symbol")
    if stage_kind == "transform_helper":
        source = _runtime_descriptor_expr(stage.get("source"), roles=roles, node_id=stage.get("node_id"))
        destination = _runtime_descriptor_expr(stage.get("destination"), roles=roles, node_id=stage.get("node_id"))
        shape_args = _runtime_shape_args(stage, roles=roles)
        return [
            f"if (int err = {symbol}({', '.join([source, destination, *shape_args, 'session->stream'])})) "
            f'return dinoml::module::fail("{stage.get("op", "conv2d_bias")} CUTLASS Conv {stage.get("stage_name", "transform")} failed");'
        ]
    if stage_kind == "provider_launcher":
        inputs = stage.get("inputs")
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("CUTLASS Conv provider launcher stage inputs must be a list")
        pointer_args = [_runtime_descriptor_expr(item, roles=roles, node_id=stage.get("node_id")) for item in inputs]
        output = _runtime_descriptor_expr(stage.get("output"), roles=roles, node_id=stage.get("node_id"))
        shape_args = _runtime_shape_args(stage, roles=roles)
        op_name = str(stage.get("op") or "conv2d_bias")
        node_scope = _c_ident(str(stage.get("node_id") or "conv"))
        stage_scope = _c_ident(str(stage.get("stage_name", "provider_launch")))
        status_name = f"status_{node_scope}_{stage_scope}"
        if str(stage.get("status", "")) == "bounded_runtime":
            failure_message = f"{op_name} CUTLASS Conv provider launcher failed"
        else:
            failure_message = f"{op_name} CUTLASS Conv provider launcher is unsupported by the current scaffold"
        return [
            f"int {status_name} = {symbol}({', '.join([*pointer_args, output, *shape_args, 'session->stream'])});",
            f"if ({status_name} != 0) {{",
            f'  return dinoml::module::fail("{failure_message}");',
            "}",
        ]
    raise ValueError(f"Unsupported CUTLASS Conv wrapper stage kind {stage_kind!r}")


def _runtime_descriptor_expr(descriptor: Any, *, roles: Mapping[str, str], node_id: Any = None) -> str:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"Malformed CUTLASS Conv descriptor: {descriptor!r}")
    kind = str(descriptor.get("kind", ""))
    if kind == "semantic_tensor":
        role = str(descriptor.get("role", ""))
        if role not in roles:
            raise ValueError(f"Unsupported CUTLASS Conv semantic tensor role {role!r}")
        return f"ptr_{roles[role]}"
    if kind == "temporary_buffer":
        node_id = str(descriptor.get("node_id") or node_id or "")
        name = str(descriptor.get("name", ""))
        if not node_id or not name:
            raise ValueError("CUTLASS Conv temporary buffer descriptor is missing node_id or name")
        return f"session->cutlass_conv_tmp_{_c_ident(node_id)}_{_c_ident(name)}"
    raise ValueError(f"Unsupported CUTLASS Conv descriptor kind {kind!r}")


def _runtime_shape_args(stage: Mapping[str, Any], *, roles: Mapping[str, str]) -> list[str]:
    args = []
    for item in _shape_arg_items(stage):
        name = str(item.get("name", ""))
        placeholder = str(item.get("placeholder", ""))
        if placeholder == "activation_n":
            args.append(f"static_cast<int>(shape_{roles['activation']}_0)")
        elif placeholder == "activation_c":
            args.append(f"static_cast<int>(shape_{roles['activation']}_1)")
        elif placeholder == "activation_h":
            args.append(f"static_cast<int>(shape_{roles['activation']}_2)")
        elif placeholder == "activation_w":
            args.append(f"static_cast<int>(shape_{roles['activation']}_3)")
        elif placeholder == "output_n":
            args.append(f"static_cast<int>(shape_{roles['output']}_0)")
        elif placeholder == "output_c":
            args.append(f"static_cast<int>(shape_{roles['output']}_1)")
        elif placeholder == "output_h":
            args.append(f"static_cast<int>(shape_{roles['output']}_2)")
        elif placeholder == "output_w":
            args.append(f"static_cast<int>(shape_{roles['output']}_3)")
        elif placeholder == "weight_o":
            args.append(f"static_cast<int>(shape_{roles['weight']}_0)")
        elif placeholder == "weight_i":
            args.append(f"static_cast<int>(shape_{roles['weight']}_1)")
        elif placeholder == "kernel_h":
            args.append(f"static_cast<int>(shape_{roles['weight']}_2)")
        elif placeholder == "kernel_w":
            args.append(f"static_cast<int>(shape_{roles['weight']}_3)")
        elif name in {"stride_h", "stride_w", "pad_h", "pad_w", "dilation_h", "dilation_w"}:
            args.append(str(int(item["value"])))
        else:
            args.append(str(int(item["value"])))
    return args


def _shape_arg_items(stage: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_args = stage.get("shape_args")
    if not isinstance(raw_args, (list, tuple)):
        raise ValueError("CUTLASS Conv wrapper stage shape_args must be a list")
    items = []
    for item in raw_args:
        if not isinstance(item, Mapping):
            raise ValueError(f"Malformed CUTLASS Conv shape arg descriptor: {item!r}")
        items.append(item)
    return items


def _manifest_kernel_item(
    kernel_manifest: Mapping[str, Any] | None,
    op_name: str,
    *,
    node_id: str,
) -> Mapping[str, Any] | None:
    if kernel_manifest is None:
        return None
    matches = []
    for item in kernel_manifest.get("required_kernels", []):
        if not isinstance(item, Mapping):
            continue
        if item.get("op") != op_name or item.get("kernel_library") != "cutlass_conv":
            continue
        conv_plan = item.get("cutlass_conv_plan")
        if isinstance(conv_plan, Mapping) and str(conv_plan.get("node_id", "")) == node_id:
            return item
        matches.append(item)
    return matches[0] if matches else None


def _validate_runtime_shape_contract(
    op_name: str,
    tensor_map: Mapping[str, Mapping[str, Any]],
    input_names: list[str],
    output_name: str,
    item: Mapping[str, Any],
) -> None:
    conv_plan = item.get("cutlass_conv_plan")
    if not isinstance(conv_plan, Mapping):
        raise ValueError(f"{op_name} CUTLASS Conv manifest entry is missing cutlass_conv_plan")
    x_name, weight_name, bias_name = input_names
    x = tensor_map[x_name]
    weight = tensor_map[weight_name]
    bias = tensor_map[bias_name]
    output = tensor_map[output_name]
    expected = {
        "input_shape": x["shape"],
        "weight_shape": weight["shape"],
        "bias_shape": bias["shape"],
        "output_shape": output["shape"],
    }
    for field, shape in expected.items():
        if [int(dim) for dim in conv_plan.get(field, [])] != [int(dim) for dim in shape]:
            raise ValueError(f"{op_name} CUTLASS Conv manifest {field} does not match lowered tensor shape")


def render_scaffold_wrapper_stages(stages: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> list[str]:
    return [render_scaffold_wrapper_stage(stage) for stage in stages]


def render_scaffold_wrapper_source(
    stages: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    op_name: str | None = None,
    node_id: str | None = None,
) -> str:
    if not stages:
        raise ValueError("CUTLASS Conv scaffold wrapper source requires at least one stage")
    first_stage = stages[0]
    op_name = str(op_name or first_stage.get("op") or "conv2d_bias")
    node_id = None if node_id is None and first_stage.get("node_id") is None else str(node_id or first_stage.get("node_id"))
    snippet_lines = []
    for stage in stages:
        snippet_lines.extend(render_scaffold_wrapper_stage(stage).splitlines())
    function_suffix = _c_ident(node_id or op_name)
    lines = [
        "// CUTLASS Conv scaffold only: emitted for artifact/source inspection.",
        "// This debug wrapper snippet is intentionally not compiled into the runtime module.",
        f"// op: {op_name}",
        f"// node_id: {node_id or '<unknown>'}",
        "#if 0",
        f'extern "C" int dinoml_cutlass_conv_wrapper_scaffold_{function_suffix}(cudaStream_t stream) {{',
    ]
    lines.extend(f"  {line}" for line in snippet_lines)
    lines.extend(
        [
            "  return 0;",
            "}",
            "#endif",
            "",
        ]
    )
    return "\n".join(lines)


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


CONV2D_BIAS_LOWERING = OpLowering(
    op_name="conv2d_bias",
    render_generated_kernel=render_generated_kernel,
    render_launch=render_launch,
    source_key=source_key,
    generated_function_name=generated_function_name,
)


__all__ = [
    "CONV2D_BIAS_LOWERING",
    "render_scaffold_wrapper_source",
    "render_scaffold_wrapper_stage",
    "render_scaffold_wrapper_stages",
]
