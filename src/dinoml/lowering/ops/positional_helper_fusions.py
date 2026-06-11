from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.template_rendering import render_op_template, supported_target_spec
from dinoml.lowering.shape_buffers import c_ident as _c_ident
from dinoml.lowering.target_specs import storage_type as target_storage_type
from dinoml.ops.positional import (
    CROPPED_POS_EMBED_DTYPES,
    FOURIER_EMBEDS_FROM_BOUNDINGBOX_DTYPES,
    GAUSSIAN_FOURIER_PROJECTION_DTYPES,
    POSITIONAL_HELPER_FUSION_OPS,
    RELATIVE_ATTENTION_BIAS_DTYPES,
    SINUSOIDAL_POSITIONAL_EMBEDDING_DTYPES,
    infer_cropped_pos_embed_with_attrs,
    infer_gaussian_fourier_projection_with_attrs,
    infer_get_fourier_embeds_from_boundingbox_with_attrs,
    infer_relative_attention_bias_with_attrs,
    infer_sinusoidal_positional_embedding_with_attrs,
    normalize_cropped_pos_embed_attrs,
    normalize_gaussian_fourier_projection_attrs,
    normalize_get_fourier_embeds_from_boundingbox_attrs,
    normalize_relative_attention_bias_attrs,
    normalize_sinusoidal_positional_embedding_attrs,
)


def render_generated_kernel(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    spec = supported_target_spec(target, str(node["op"]))
    context = _context(target, node, tensor_map)
    if spec.is_gpu:
        context.update(spec.gpu_template_context())
    template_name = f"{node['op']}_{'gpu' if spec.is_gpu else 'cpu.cpp'}.j2"
    return render_op_template(template_name, context)


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    del kernel_manifest
    spec = supported_target_spec(target, str(node["op"]))
    func = _function_name(node, tensor_map)
    op_name = str(node["op"])
    if op_name == "cropped_pos_embed":
        out_ident = _c_ident(str(node["outputs"][0]))
        args = f"ptr_{out_ident}, runtime_numel_{out_ident}"
    elif op_name == "gaussian_fourier_projection":
        x_ident = _c_ident(str(node["inputs"][0]))
        weight_ident = _c_ident(str(node["inputs"][1]))
        out_ident = _c_ident(str(node["outputs"][0]))
        args = (
            f"ptr_{x_ident}, ptr_{weight_ident}, ptr_{out_ident}, "
            f"runtime_numel_{x_ident}, runtime_numel_{weight_ident}, runtime_numel_{out_ident}"
        )
    elif op_name == "get_fourier_embeds_from_boundingbox":
        box_ident = _c_ident(str(node["inputs"][0]))
        out_ident = _c_ident(str(node["outputs"][0]))
        args = f"ptr_{box_ident}, ptr_{out_ident}, runtime_numel_{box_ident}, runtime_numel_{out_ident}"
    elif op_name == "relative_attention_bias":
        embedding_ident = _c_ident(str(node["inputs"][0]))
        out_ident = _c_ident(str(node["outputs"][0]))
        args = (
            f"ptr_{embedding_ident}, ptr_{out_ident}, "
            f"shape_{out_ident}_2, shape_{out_ident}_3, shape_{out_ident}_1, "
            f"runtime_numel_{embedding_ident}, runtime_numel_{out_ident}"
        )
    elif op_name == "sinusoidal_positional_embedding":
        x_ident = _c_ident(str(node["inputs"][0]))
        out_ident = _c_ident(str(node["outputs"][0]))
        args = (
            f"ptr_{x_ident}, ptr_{out_ident}, shape_{x_ident}_0, shape_{x_ident}_1, "
            f"runtime_numel_{x_ident}, runtime_numel_{out_ident}"
        )
    else:
        raise ValueError(f"Unsupported positional helper op: {op_name}")
    if not spec.is_gpu:
        return f"if (int err = {func}({args})) return err;"
    return f"if (int err = {func}({args}, {spec.stream_expr})) return err;"


def source_key(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    supported_target_spec(target, str(node["op"]))
    return f"{target}:{_function_name(node, tensor_map)}"


def generated_function_name(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    del target
    return _function_name(node, tensor_map)


def _context(target: str, node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    op_name = str(node["op"])
    validated = _validate_node_contract(node, tensor_map)
    context: dict[str, Any] = {
        "func": _function_name(node, tensor_map),
        "kernel": f"{_function_name(node, tensor_map)}_kernel",
    }
    if op_name == "cropped_pos_embed":
        output_dtype = str(validated["output_dtype"])
        normalized = validated["normalized"]
        context.update(
            {
                "storage_type": target_storage_type(output_dtype, target),
                "embed_dim": int(normalized["embed_dim"]),
                "half_embed_dim": int(normalized["embed_dim"]) // 2,
                "pair_dim": int(normalized["embed_dim"]) // 4,
                "crop_h": int(normalized["crop_h"]),
                "crop_w": int(normalized["crop_w"]),
                "top": int(normalized["top"]),
                "left": int(normalized["left"]),
                "grid_scale_literal": _float_literal(
                    float(normalized["base_size"])
                    / (float(normalized["pos_embed_max_size"]) * float(normalized["interpolation_scale"]))
                ),
                "neg_log_10000_literal": _float_literal(-math.log(10000.0)),
                "block_size": 256,
            }
        )
    elif op_name == "gaussian_fourier_projection":
        output_dtype = str(validated["output_dtype"])
        context.update(
            {
                "storage_type": target_storage_type(output_dtype, target),
                "log_input": bool(validated["normalized"]["log"]),
                "flip_sin_to_cos": bool(validated["normalized"]["flip_sin_to_cos"]),
                "two_pi_literal": _float_literal(2.0 * math.pi),
                "block_size": 256,
            }
        )
    elif op_name == "get_fourier_embeds_from_boundingbox":
        output_dtype = str(validated["output_dtype"])
        embed_dim = int(validated["normalized"]["embed_dim"])
        context.update(
            {
                "storage_type": target_storage_type(output_dtype, target),
                "embed_dim": embed_dim,
                "output_cols": embed_dim * 8,
                "log_100_literal": _float_literal(math.log(100.0)),
                "block_size": 256,
            }
        )
    elif op_name == "relative_attention_bias":
        output_dtype = str(validated["output_dtype"])
        normalized = validated["normalized"]
        context.update(
            {
                "storage_type": target_storage_type(output_dtype, target),
                "num_buckets": int(normalized["num_buckets"]),
                "max_distance": int(normalized["max_distance"]),
                "bidirectional": bool(normalized["bidirectional"]),
                "block_size": 256,
            }
        )
    elif op_name == "sinusoidal_positional_embedding":
        output_dtype = str(validated["output_dtype"])
        normalized = validated["normalized"]
        context.update(
            {
                "storage_type": target_storage_type(output_dtype, target),
                "embed_dim": int(normalized["embed_dim"]),
                "max_seq_len": int(normalized["max_seq_len"]),
                "neg_log_10000_literal": _float_literal(-math.log(10000.0)),
                "block_size": 256,
            }
        )
    else:
        raise ValueError(f"Unsupported positional helper op: {op_name}")
    return context


def _validate_node_contract(
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    op_name = str(node["op"])
    attrs = node.get("attrs", {})
    if op_name == "cropped_pos_embed":
        if node.get("inputs"):
            raise ValueError("cropped_pos_embed expects zero inputs")
        if len(node.get("outputs", ())) != 1:
            raise ValueError("cropped_pos_embed expects one output")
        output_tensor = tensor_map[str(node["outputs"][0])]
        output_dtype = str(attrs.get("dtype", output_tensor["dtype"]))
        if output_dtype not in CROPPED_POS_EMBED_DTYPES:
            raise NotImplementedError(f"cropped_pos_embed does not support dtype {output_dtype}")
        normalized = normalize_cropped_pos_embed_attrs(
            embed_dim=attrs.get("embed_dim"),
            pos_embed_max_size=attrs.get("pos_embed_max_size"),
            base_size=attrs.get("base_size"),
            interpolation_scale=attrs.get("interpolation_scale"),
            patch_size=attrs.get("patch_size"),
            height=attrs.get("height"),
            width=attrs.get("width"),
        )
        expected_shape = infer_cropped_pos_embed_with_attrs([], attrs)
        if list(output_tensor["shape"]) != expected_shape:
            raise ValueError("cropped_pos_embed output shape does not match attrs")
        if str(output_tensor["dtype"]) != output_dtype:
            raise NotImplementedError("cropped_pos_embed output dtype does not match attrs")
        return {"normalized": normalized, "output_dtype": output_dtype}
    if op_name == "gaussian_fourier_projection":
        if len(node.get("inputs", ())) != 2 or len(node.get("outputs", ())) != 1:
            raise ValueError("gaussian_fourier_projection expects two inputs and one output")
        x_tensor = tensor_map[str(node["inputs"][0])]
        weight_tensor = tensor_map[str(node["inputs"][1])]
        output_tensor = tensor_map[str(node["outputs"][0])]
        output_dtype = str(output_tensor["dtype"])
        if output_dtype not in GAUSSIAN_FOURIER_PROJECTION_DTYPES:
            raise NotImplementedError(f"gaussian_fourier_projection does not support dtype {output_dtype}")
        if str(x_tensor["dtype"]) != output_dtype or str(weight_tensor["dtype"]) != output_dtype:
            raise NotImplementedError("gaussian_fourier_projection lowering requires matching input/output dtypes")
        normalized = normalize_gaussian_fourier_projection_attrs(
            log=attrs.get("log", True),
            flip_sin_to_cos=attrs.get("flip_sin_to_cos", False),
        )
        expected_shape = infer_gaussian_fourier_projection_with_attrs(
            [x_tensor["shape"], weight_tensor["shape"]],
            attrs,
        )
        if list(output_tensor["shape"]) != expected_shape:
            raise ValueError("gaussian_fourier_projection output shape does not match inputs")
        return {"normalized": normalized, "output_dtype": output_dtype}
    if op_name == "get_fourier_embeds_from_boundingbox":
        if len(node.get("inputs", ())) != 1 or len(node.get("outputs", ())) != 1:
            raise ValueError("get_fourier_embeds_from_boundingbox expects one input and one output")
        box_tensor = tensor_map[str(node["inputs"][0])]
        output_tensor = tensor_map[str(node["outputs"][0])]
        output_dtype = str(output_tensor["dtype"])
        if output_dtype not in FOURIER_EMBEDS_FROM_BOUNDINGBOX_DTYPES:
            raise NotImplementedError(f"get_fourier_embeds_from_boundingbox does not support dtype {output_dtype}")
        if str(box_tensor["dtype"]) != output_dtype:
            raise NotImplementedError("get_fourier_embeds_from_boundingbox lowering requires matching input/output dtypes")
        normalized = normalize_get_fourier_embeds_from_boundingbox_attrs(embed_dim=attrs.get("embed_dim"))
        expected_shape = infer_get_fourier_embeds_from_boundingbox_with_attrs([box_tensor["shape"]], attrs)
        if list(output_tensor["shape"]) != expected_shape:
            raise ValueError("get_fourier_embeds_from_boundingbox output shape does not match attrs")
        return {"normalized": normalized, "output_dtype": output_dtype}
    if op_name == "relative_attention_bias":
        if len(node.get("inputs", ())) != 1 or len(node.get("outputs", ())) != 1:
            raise ValueError("relative_attention_bias expects one input and one output")
        embedding_tensor = tensor_map[str(node["inputs"][0])]
        output_tensor = tensor_map[str(node["outputs"][0])]
        output_dtype = str(output_tensor["dtype"])
        if output_dtype not in RELATIVE_ATTENTION_BIAS_DTYPES:
            raise NotImplementedError(f"relative_attention_bias does not support dtype {output_dtype}")
        if str(embedding_tensor["dtype"]) != output_dtype:
            raise NotImplementedError("relative_attention_bias lowering requires matching input/output dtypes")
        normalized = normalize_relative_attention_bias_attrs(
            query_length=attrs.get("query_length"),
            key_length=attrs.get("key_length"),
            bidirectional=attrs.get("bidirectional", True),
            num_buckets=attrs.get("num_buckets", 32),
            max_distance=attrs.get("max_distance", 128),
        )
        expected_shape = infer_relative_attention_bias_with_attrs([embedding_tensor["shape"]], attrs)
        if list(output_tensor["shape"]) != expected_shape:
            raise ValueError("relative_attention_bias output shape does not match attrs")
        if list(output_tensor["shape"])[0] != 1:
            raise ValueError("relative_attention_bias output batch dimension must be 1")
        return {"normalized": normalized, "output_dtype": output_dtype}
    if op_name == "sinusoidal_positional_embedding":
        if len(node.get("inputs", ())) != 1 or len(node.get("outputs", ())) != 1:
            raise ValueError("sinusoidal_positional_embedding expects one input and one output")
        x_tensor = tensor_map[str(node["inputs"][0])]
        output_tensor = tensor_map[str(node["outputs"][0])]
        output_dtype = str(output_tensor["dtype"])
        if output_dtype not in SINUSOIDAL_POSITIONAL_EMBEDDING_DTYPES:
            raise NotImplementedError(f"sinusoidal_positional_embedding does not support dtype {output_dtype}")
        if str(x_tensor["dtype"]) != output_dtype:
            raise NotImplementedError("sinusoidal_positional_embedding lowering requires matching input/output dtypes")
        normalized = normalize_sinusoidal_positional_embedding_attrs(
            embed_dim=attrs.get("embed_dim"),
            max_seq_len=attrs.get("max_seq_len"),
        )
        expected_shape = infer_sinusoidal_positional_embedding_with_attrs([x_tensor["shape"]], attrs)
        if list(output_tensor["shape"]) != expected_shape:
            raise ValueError("sinusoidal_positional_embedding output shape does not match input")
        return {"normalized": normalized, "output_dtype": output_dtype}
    raise ValueError(f"Unsupported positional helper op: {op_name}")


def _function_name(node: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> str:
    signature = {
        "op": str(node["op"]),
        "attrs": dict(node.get("attrs", {})),
        "inputs": [(list(tensor_map[str(name)]["shape"]), str(tensor_map[str(name)]["dtype"])) for name in node["inputs"]],
        "outputs": [(list(tensor_map[str(name)]["shape"]), str(tensor_map[str(name)]["dtype"])) for name in node["outputs"]],
    }
    digest = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    return f"{node['op']}_{digest}"


def _float_literal(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("positional helper lowering supports only finite attrs")
    literal = f"{value:.9g}"
    if "." not in literal and "e" not in literal and "E" not in literal:
        literal = f"{literal}.0"
    return f"{literal}f"


POSITIONAL_HELPER_FUSION_LOWERINGS = {
    op_name: OpLowering(
        op_name=op_name,
        render_generated_kernel=render_generated_kernel,
        render_launch=render_launch,
        source_key=source_key,
        generated_function_name=generated_function_name,
    )
    for op_name in POSITIONAL_HELPER_FUSION_OPS
}
