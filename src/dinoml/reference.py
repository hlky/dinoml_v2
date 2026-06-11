from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import numpy as np

from dinoml.ir import ModelSpec, array_from_storage, array_to_storage, dtype_numpy
from dinoml.kernels.bmm import BMM_OPS, bmm_op_spec
from dinoml.kernels.gemm import GEMM_OPS, gemm_op_spec
from dinoml.ops.elementwise import ELEMENTWISE_BY_NAME, FUSABLE_ELEMENTWISE_OPS
from dinoml.ops.collections import SPECIALIZED_PERMUTE_DIMS, normalize_permute_dims
from dinoml.shapes import evaluate_symbolic_int
from dinoml.ops.positional import (
    normalize_cropped_pos_embed_attrs,
    normalize_gaussian_fourier_projection_attrs,
    normalize_get_fourier_embeds_from_boundingbox_attrs,
    normalize_get_1d_rotary_pos_embed_attrs,
    normalize_get_2d_rotary_pos_embed_attrs,
    normalize_get_2d_rotary_pos_embed_lumina_attrs,
    normalize_get_3d_rotary_pos_embed_allegro_attrs,
    normalize_get_3d_rotary_pos_embed_attrs,
    normalize_relative_attention_bias_attrs,
    normalize_sinusoidal_positional_embedding_attrs,
)


def reference_numpy(spec: ModelSpec, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    ir = spec.ir
    values: Dict[str, np.ndarray] = {}
    for input_info in ir["inputs"]:
        name = input_info["name"]
        if name not in inputs:
            raise ValueError(f"Missing input: {name}")
        values[input_info["tensor"]] = _reference_array(inputs[name], str(input_info["dtype"]))
    for constant_info in ir["constants"]:
        name = constant_info["name"]
        if name not in spec.constants:
            raise ValueError(f"Missing constant value: {name}")
        values[constant_info["tensor"]] = _reference_array(spec.constants[name], str(constant_info["dtype"]))

    for node in ir["nodes"]:
        _materialize_available_views(ir, values)
        if node["op"] in FUSABLE_ELEMENTWISE_OPS:
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(_execute_elementwise(node["op"], [values[name] for name in node["inputs"]], node.get("attrs", {})), output_dtype)
        elif node["op"] == "fused_elementwise":
            _execute_fused_elementwise(node, values, ir)
        elif node["op"] == "softmax":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_softmax(values[node["inputs"][0]], node.get("attrs", {})),
                output_dtype,
            )
        elif node["op"] == "t5_layer_norm":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_t5_layer_norm(
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    node.get("attrs", {}),
                ),
                output_dtype,
            )
        elif node["op"] == "layer_norm":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_layer_norm(
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    values[node["inputs"][2]],
                    node.get("attrs", {}),
                ),
                output_dtype,
            )
        elif node["op"] == "add_layer_norm":
            summed_name, normalized_name = node["outputs"]
            summed_dtype = _tensor_dtype(ir, summed_name)
            normalized_dtype = _tensor_dtype(ir, normalized_name)
            summed = _execute_elementwise("add", [values[node["inputs"][0]], values[node["inputs"][1]]], {})
            values[summed_name] = _store_reference(summed, summed_dtype)
            values[normalized_name] = _store_reference(
                _execute_layer_norm(
                    values[summed_name],
                    values[node["inputs"][2]],
                    values[node["inputs"][3]],
                    node.get("attrs", {}),
                ),
                normalized_dtype,
            )
        elif node["op"] in {"group_norm", "group_norm_swish"}:
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_group_norm(
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    values[node["inputs"][2]],
                    node.get("attrs", {}),
                    apply_swish=node["op"] == "group_norm_swish",
                ),
                output_dtype,
            )
        elif node["op"] == "get_timestep_embedding":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_get_timestep_embedding(values[node["inputs"][0]], node.get("attrs", {})),
                output_dtype,
            )
        elif node["op"] == "cropped_pos_embed":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_cropped_pos_embed(node.get("attrs", {})),
                output_dtype,
            )
        elif node["op"] == "gaussian_fourier_projection":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_gaussian_fourier_projection(
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    node.get("attrs", {}),
                ),
                output_dtype,
            )
        elif node["op"] == "get_fourier_embeds_from_boundingbox":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_get_fourier_embeds_from_boundingbox(
                    values[node["inputs"][0]],
                    node.get("attrs", {}),
                ),
                output_dtype,
            )
        elif node["op"] == "relative_attention_bias":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_relative_attention_bias(
                    values[node["inputs"][0]],
                    node.get("attrs", {}),
                ),
                output_dtype,
            )
        elif node["op"] == "sinusoidal_positional_embedding":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_sinusoidal_positional_embedding(
                    values[node["inputs"][0]],
                    node.get("attrs", {}),
                ),
                output_dtype,
            )
        elif node["op"] in {"get_1d_rotary_pos_embed_cos", "get_1d_rotary_pos_embed_sin"}:
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_get_1d_rotary_pos_embed_component(
                    None if not node["inputs"] else values[node["inputs"][0]],
                    node.get("attrs", {}),
                    output_kind="cos" if node["op"] == "get_1d_rotary_pos_embed_cos" else "sin",
                ),
                output_dtype,
            )
        elif node["op"] == "get_2d_rotary_pos_embed":
            cos_name, sin_name = node["outputs"]
            cos_out, sin_out = _execute_get_2d_rotary_pos_embed(node.get("attrs", {}))
            values[cos_name] = _store_reference(cos_out, _tensor_dtype(ir, cos_name))
            values[sin_name] = _store_reference(sin_out, _tensor_dtype(ir, sin_name))
        elif node["op"] == "get_2d_rotary_pos_embed_lumina":
            real_name, imag_name = node["outputs"]
            real_out, imag_out = _execute_get_2d_rotary_pos_embed_lumina(node.get("attrs", {}))
            values[real_name] = _store_reference(real_out, _tensor_dtype(ir, real_name))
            values[imag_name] = _store_reference(imag_out, _tensor_dtype(ir, imag_name))
        elif node["op"] == "get_3d_rotary_pos_embed":
            cos_name, sin_name = node["outputs"]
            cos_out, sin_out = _execute_get_3d_rotary_pos_embed(node.get("attrs", {}))
            values[cos_name] = _store_reference(cos_out, _tensor_dtype(ir, cos_name))
            values[sin_name] = _store_reference(sin_out, _tensor_dtype(ir, sin_name))
        elif node["op"] == "get_3d_rotary_pos_embed_allegro":
            outputs = _execute_get_3d_rotary_pos_embed_allegro(node.get("attrs", {}))
            for output_name, output_value in zip(node["outputs"], outputs):
                values[output_name] = _store_reference(output_value, _tensor_dtype(ir, output_name))
        elif node["op"] == "glm_ocr_text_rope":
            q_name, k_name = node["outputs"]
            q_out, k_out = _execute_glm_ocr_text_rope(
                values[node["inputs"][0]],
                values[node["inputs"][1]],
                values[node["inputs"][2]],
                values[node["inputs"][3]],
                node.get("attrs", {}),
            )
            values[q_name] = _store_reference(q_out, _tensor_dtype(ir, q_name))
            values[k_name] = _store_reference(k_out, _tensor_dtype(ir, k_name))
        elif node["op"] == "glm_ocr_vision_rope":
            q_name, k_name = node["outputs"]
            q_out, k_out = _execute_glm_ocr_vision_rope(
                values[node["inputs"][0]],
                values[node["inputs"][1]],
                values[node["inputs"][2]],
                values[node["inputs"][3]],
            )
            values[q_name] = _store_reference(q_out, _tensor_dtype(ir, q_name))
            values[k_name] = _store_reference(k_out, _tensor_dtype(ir, k_name))
        elif node["op"] == "glm_ocr_stitch_image_features":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_glm_ocr_stitch_image_features(
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    values[node["inputs"][2]],
                    node.get("attrs", {}),
                ),
                output_dtype,
            )
        elif node["op"] == "qwen2_5_vl_stitch_image_features":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_qwen2_5_vl_stitch_image_features(
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    values[node["inputs"][2]],
                    node.get("attrs", {}),
                ),
                output_dtype,
            )
        elif node["op"] in {"reduce_sum", "reduce_max", "reduce_min", "reduce_mean", "var", "vector_norm"}:
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_reduction(node["op"], values[node["inputs"][0]], node.get("attrs", {})),
                output_dtype,
            )
        elif node["op"] == "swiglu":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_swiglu(values[node["inputs"][0]]),
                output_dtype,
            )
        elif node["op"] == "argmax":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_argmax(values[node["inputs"][0]], node.get("attrs", {})),
                output_dtype,
            )
        elif node["op"] in {"topk_values", "topk_indices"}:
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            topk_indices = _execute_topk_indices(values[node["inputs"][0]], node.get("attrs", {}))
            if node["op"] == "topk_indices":
                result = topk_indices
            else:
                result = np.take_along_axis(values[node["inputs"][0]], topk_indices, axis=-1)
            values[output_name] = _store_reference(result, output_dtype)
        elif node["op"] == "full":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            output_shape = _tensor_shape(ir, output_name)
            values[output_name] = _store_reference(
                np.full(output_shape, node.get("attrs", {}).get("fill_value"), dtype=np.float32 if output_dtype != "bool" else np.bool_),
                output_dtype,
            )
        elif node["op"] == "arange":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            output_shape = _tensor_shape(ir, output_name)
            attrs = node.get("attrs", {})
            idx = np.arange(output_shape[0], dtype=np.float32)
            values[output_name] = _store_reference(
                float(attrs["start"]) + idx * float(attrs.get("step", 1.0)),
                output_dtype,
            )
        elif node["op"] == "randn":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            output_shape = _tensor_shape(ir, output_name)
            attrs = node.get("attrs", {})
            values[output_name] = _store_reference(
                _execute_randn(output_shape, int(attrs.get("seed", 0)), str(attrs.get("rng", "dinoml")), output_dtype),
                output_dtype,
            )
        elif node["op"] == "expand":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            output_shape = _tensor_shape(ir, output_name)
            values[output_name] = _store_reference(
                np.broadcast_to(values[node["inputs"][0]], output_shape).copy(),
                output_dtype,
            )
        elif node["op"] == "concatenate":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                np.concatenate([values[name] for name in node["inputs"]], axis=int(node.get("attrs", {}).get("dim", 0))).copy(),
                output_dtype,
            )
        elif node["op"] == "stack":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                np.stack([values[name] for name in node["inputs"]], axis=int(node.get("attrs", {}).get("dim", 0))).copy(),
                output_dtype,
            )
        elif node["op"] == "flip":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                np.flip(values[node["inputs"][0]], axis=tuple(node.get("attrs", {}).get("dims", ()))).copy(),
                output_dtype,
            )
        elif node["op"] == "repeat_interleave":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            attrs = node.get("attrs", {})
            values[output_name] = _store_reference(
                np.repeat(
                    values[node["inputs"][0]],
                    int(attrs["repeats"]),
                    axis=int(attrs["dim"]),
                ).copy(),
                output_dtype,
            )
        elif node["op"] in {"permute", *SPECIALIZED_PERMUTE_DIMS}:
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            axes = _permute_axes(node, len(values[node["inputs"][0]].shape))
            values[output_name] = _store_reference(
                np.transpose(values[node["inputs"][0]], axes=tuple(axes)).copy(),
                output_dtype,
            )
        elif node["op"] == "dynamic_slice":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            attrs = node.get("attrs", {})
            input_name = node["inputs"][0]
            dim_values = _shape_dim_values(ir, input_name, values[input_name].shape)
            slices = tuple(
                slice(
                    evaluate_symbolic_int(start, dim_values),
                    evaluate_symbolic_int(start, dim_values) + evaluate_symbolic_int(size, dim_values),
                )
                for start, size in zip(attrs.get("start_indices", ()), attrs.get("slice_sizes", ()))
            )
            values[output_name] = _store_reference(
                values[input_name][slices].copy(),
                output_dtype,
            )
        elif node["op"] == "index_select":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            attrs = node.get("attrs", {})
            values[output_name] = _store_reference(
                np.take(
                    values[node["inputs"][0]],
                    [int(index) for index in attrs.get("indices", ())],
                    axis=int(attrs.get("dim", 0)),
                ).copy(),
                output_dtype,
            )
        elif node["op"] == "runtime_index_select":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            attrs = node.get("attrs", {})
            values[output_name] = _store_reference(
                np.take(
                    values[node["inputs"][0]],
                    np.asarray(values[node["inputs"][1]], dtype=np.int64),
                    axis=int(attrs.get("dim", 0)),
                ).copy(),
                output_dtype,
            )
        elif node["op"] == "gather":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            attrs = node.get("attrs", {})
            values[output_name] = _store_reference(
                _execute_gather(
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    int(attrs.get("dim", 0)),
                ),
                output_dtype,
            )
        elif node["op"] == "batch_gather":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_batch_gather(values[node["inputs"][0]], values[node["inputs"][1]]),
                output_dtype,
            )
        elif node["op"] == "embedding":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_embedding(values[node["inputs"][0]], values[node["inputs"][1]]),
                output_dtype,
            )
        elif node["op"] == "slice_scatter":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            attrs = node.get("attrs", {})
            update = values[node["inputs"][1]]
            slices = tuple(
                slice(int(start), int(start) + int(size))
                for start, size in zip(attrs.get("start_indices", ()), update.shape)
            )
            result = values[node["inputs"][0]].copy()
            result[slices] = update
            values[output_name] = _store_reference(result, output_dtype)
        elif node["op"] == "pad":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            attrs = node.get("attrs", {})
            pad = [int(value) for value in attrs.get("pad", ())]
            rank = values[node["inputs"][0]].ndim
            pad_width = [(0, 0)] * rank
            for pair_index in range(len(pad) // 2):
                axis = rank - 1 - pair_index
                pad_width[axis] = (pad[2 * pair_index], pad[2 * pair_index + 1])
            values[output_name] = _store_reference(
                np.pad(
                    values[node["inputs"][0]],
                    tuple(pad_width),
                    mode="constant",
                    constant_values=attrs.get("value", 0.0),
                ).copy(),
                output_dtype,
            )
        elif node["op"] == "avg_pool1d":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_avg_pool1d(values[node["inputs"][0]], node.get("attrs", {})),
                output_dtype,
            )
        elif node["op"] == "avg_pool2d":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_avg_pool2d(values[node["inputs"][0]], node.get("attrs", {})),
                output_dtype,
            )
        elif node["op"] == "max_pool2d":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_max_pool2d(values[node["inputs"][0]], node.get("attrs", {})),
                output_dtype,
            )
        elif node["op"] in {"conv1d_bias", "conv1d_bias_relu", "conv1d_bias_add", "conv1d_bias_add_relu"}:
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_conv1d_bias_family(
                    node["op"],
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    values[node["inputs"][2]],
                    None if node["op"] not in {"conv1d_bias_add", "conv1d_bias_add_relu"} else values[node["inputs"][3]],
                    node.get("attrs", {}),
                ),
                output_dtype,
            )
        elif node["op"] in {"conv2d_bias", "conv2d_bias_relu", "conv2d_bias_add", "conv2d_bias_add_relu"}:
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_conv2d_bias_family(
                    node["op"],
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    values[node["inputs"][2]],
                    None if node["op"] not in {"conv2d_bias_add", "conv2d_bias_add_relu"} else values[node["inputs"][3]],
                    node.get("attrs", {}),
                ),
                output_dtype,
            )
        elif node["op"] in {
            "transposed_conv2d",
            "transposed_conv2d_bias",
            "transposed_conv2d_bias_relu",
            "transposed_conv2d_bias_add",
            "transposed_conv2d_bias_add_relu",
        }:
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_transposed_conv2d_family(
                    node["op"],
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    None if node["op"] == "transposed_conv2d" else values[node["inputs"][2]],
                    (
                        None
                        if node["op"] not in {"transposed_conv2d_bias_add", "transposed_conv2d_bias_add_relu"}
                        else values[node["inputs"][3]]
                    ),
                    node.get("attrs", {}),
                ),
                output_dtype,
            )
        elif node["op"] in GEMM_OPS:
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_gemm(node["op"], [values[name] for name in node["inputs"]]),
                output_dtype,
            )
        elif node["op"] in BMM_OPS:
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_bmm(node["op"], [values[name] for name in node["inputs"]]),
                output_dtype,
            )
        elif node["op"] == "flash_attention":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_flash_attention(
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    values[node["inputs"][2]],
                    node.get("attrs", {}),
                ),
                output_dtype,
            )
        elif node["op"] == "flash_attention_bias":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_flash_attention_bias(
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    values[node["inputs"][2]],
                    values[node["inputs"][3]],
                    node.get("attrs", {}),
                ),
                output_dtype,
            )
        elif node["op"] == "flash_attention_qkv":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            qkv = np.asarray(values[node["inputs"][0]])
            values[output_name] = _store_reference(
                _execute_flash_attention(qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :], node.get("attrs", {})),
                output_dtype,
            )
        elif node["op"] == "flash_attention_varlen":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_flash_attention_varlen(
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    values[node["inputs"][2]],
                    values[node["inputs"][3]],
                    node.get("attrs", {}),
                ),
                output_dtype,
            )
        elif node["op"] == "flash_attention_static_kv_cache":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_flash_attention_static_kv_cache(
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    values[node["inputs"][2]],
                    values[node["inputs"][3]],
                    values[node["inputs"][4]],
                    values[node["inputs"][5]],
                ),
                output_dtype,
            )
        elif node["op"] == "flash_attention_static_kv_cache_bias":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_flash_attention_static_kv_cache_bias(
                    values[node["inputs"][0]],
                    values[node["inputs"][1]],
                    values[node["inputs"][2]],
                    values[node["inputs"][3]],
                    values[node["inputs"][4]],
                    values[node["inputs"][5]],
                    values[node["inputs"][6]],
                ),
                output_dtype,
            )
        elif node["op"] == "qkv_split":
            q_name, k_name, v_name = node["outputs"]
            qkv = np.asarray(values[node["inputs"][0]])
            hidden = qkv.shape[-1] // 3
            values[q_name] = _store_reference(qkv[..., :hidden], _tensor_dtype(ir, q_name))
            values[k_name] = _store_reference(qkv[..., hidden : 2 * hidden], _tensor_dtype(ir, k_name))
            values[v_name] = _store_reference(qkv[..., 2 * hidden :], _tensor_dtype(ir, v_name))
        else:
            raise ValueError(f"Unsupported op: {node['op']}")

    _materialize_available_views(ir, values)
    return {output["name"]: values[output["tensor"]] for output in ir["outputs"]}


def _materialize_available_views(ir: Mapping[str, object], values: Dict[str, np.ndarray]) -> None:
    views = ir.get("metadata", {}).get("views", {}).get("views", [])
    while True:
        progressed = False
        for view in views:
            tensor = str(view["tensor"])
            source = str(view["source"])
            if tensor in values or source not in values:
                continue
            offset = int(view.get("offset_elements", 0))
            numel = int(np.prod([int(dim) for dim in view["shape"]], dtype=np.int64))
            values[tensor] = np.reshape(values[source].reshape(-1)[offset : offset + numel], tuple(int(dim) for dim in view["shape"])).copy()
            progressed = True
        if not progressed:
            return


def _execute_gather(x: np.ndarray, index: np.ndarray, dim: int) -> np.ndarray:
    index_values = np.asarray(index, dtype=np.int64)
    result = np.empty(index_values.shape, dtype=x.dtype)
    dim_extent = int(x.shape[dim])
    for output_coord in np.ndindex(index_values.shape):
        selected = int(index_values[output_coord])
        if selected < 0 or selected >= dim_extent:
            raise ValueError(f"gather index {selected} is out of bounds for dim size {dim_extent}")
        input_coord = list(output_coord)
        input_coord[dim] = selected
        result[output_coord] = x[tuple(input_coord)]
    return result


def _execute_batch_gather(x: np.ndarray, indices: np.ndarray) -> np.ndarray:
    index_values = np.asarray(indices, dtype=np.int64)
    result = np.empty((index_values.shape[0], index_values.shape[1], *x.shape[2:]), dtype=x.dtype)
    dim_extent = int(x.shape[1])
    for batch in range(index_values.shape[0]):
        for k in range(index_values.shape[1]):
            selected = int(index_values[batch, k])
            if selected < 0 or selected >= dim_extent:
                raise ValueError(f"batch_gather index {selected} is out of bounds for dim size {dim_extent}")
            result[(batch, k)] = x[(batch, selected)]
    return result


def _execute_embedding(table: np.ndarray, indices: np.ndarray) -> np.ndarray:
    index_values = np.asarray(indices, dtype=np.int64)
    vocab_size = int(table.shape[0])
    bad = np.argwhere((index_values < 0) | (index_values >= vocab_size))
    if bad.size:
        selected = int(index_values[tuple(int(axis) for axis in bad[0])])
        raise ValueError(f"embedding index {selected} is out of bounds for vocab size {vocab_size}")
    return np.array(table[index_values], copy=True)


def _execute_fused_elementwise(node: Mapping[str, object], values: Dict[str, np.ndarray], ir: Mapping[str, object]) -> None:
    for sub_op in node["attrs"]["sub_ops"]:
        op = sub_op["op"]
        inputs = [values[name] for name in sub_op["inputs"]]
        output = sub_op["outputs"][0]
        values[output] = _store_reference(_execute_elementwise(op, inputs, sub_op.get("attrs", {})), _tensor_dtype(ir, output))


def _execute_elementwise(op: str, inputs: list[np.ndarray], attrs: Mapping[str, object]) -> np.ndarray:
    spec = ELEMENTWISE_BY_NAME.get(op)
    if spec is None:
        raise ValueError(f"Unsupported elementwise op: {op}")
    defaults = {name: default for name, default in spec.attr_defaults}
    merged_attrs = {**defaults, **dict(attrs)}
    if op == "add":
        result = inputs[0] + inputs[1]
    elif op == "sub":
        result = inputs[0] - inputs[1]
    elif op == "mul":
        result = inputs[0] * inputs[1]
    elif op == "div":
        result = inputs[0] / inputs[1]
    elif op == "tanh":
        result = np.tanh(inputs[0])
    elif op == "cos":
        result = np.cos(inputs[0])
    elif op == "sin":
        result = np.sin(inputs[0])
    elif op == "sign":
        result = np.sign(inputs[0])
    elif op == "abs":
        result = np.abs(inputs[0])
    elif op == "log":
        result = np.log(inputs[0])
    elif op == "log1p":
        result = np.log1p(inputs[0])
    elif op == "exp":
        result = np.exp(inputs[0])
    elif op == "sqrt":
        result = np.sqrt(inputs[0])
    elif op == "max":
        result = np.maximum(inputs[0], inputs[1])
    elif op == "min":
        result = np.minimum(inputs[0], inputs[1])
    elif op == "sigmoid":
        result = 1.0 / (1.0 + np.exp(-inputs[0]))
    elif op == "leaky_relu":
        negative_slope = float(merged_attrs["negative_slope"])
        result = np.where(inputs[0] > 0.0, inputs[0], inputs[0] * negative_slope)
    elif op == "hardtanh":
        result = np.clip(inputs[0], float(merged_attrs["min_value"]), float(merged_attrs["max_value"]))
    elif op == "relu":
        result = np.maximum(inputs[0], 0.0)
    elif op == "nan_to_num":
        result = np.nan_to_num(
            inputs[0],
            nan=float(merged_attrs["nan_replacement"]),
            posinf=float(merged_attrs["posinf_replacement"]),
            neginf=float(merged_attrs["neginf_replacement"]),
        )
    elif op == "clamp_nan_to_num":
        result = np.where(
            np.isnan(inputs[0]),
            float(merged_attrs["nan_replacement"]),
            np.clip(inputs[0], float(merged_attrs["clamp_min"]), float(merged_attrs["clamp_max"])),
        )
    elif op == "silu":
        x = inputs[0]
        result = x / (1.0 + np.exp(-x))
    elif op == "pow":
        result = np.power(inputs[0], inputs[1])
    elif op == "gelu":
        x = inputs[0]
        result = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))
    elif op == "fast_gelu":
        x = inputs[0]
        result = 0.5 * x * (1.0 + np.tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x)))
    elif op == "softplus":
        result = np.log1p(np.exp(inputs[0]))
    elif op == "elu":
        alpha = float(merged_attrs["alpha"])
        result = np.where(inputs[0] > 0.0, inputs[0], alpha * (np.exp(inputs[0]) - 1.0))
    elif op == "softsign":
        result = inputs[0] / (1.0 + np.abs(inputs[0]))
    elif op == "floor_div":
        result = np.floor(inputs[0] / inputs[1])
    elif op == "celu":
        alpha = float(merged_attrs["alpha"])
        result = np.maximum(0.0, inputs[0]) + np.minimum(0.0, alpha * (np.exp(inputs[0] / alpha) - 1.0))
    elif op == "floor":
        result = np.floor(inputs[0])
    elif op == "eq":
        return np.equal(inputs[0], inputs[1])
    elif op == "ge":
        return np.greater_equal(inputs[0], inputs[1])
    elif op == "gt":
        return np.greater(inputs[0], inputs[1])
    elif op == "le":
        return np.less_equal(inputs[0], inputs[1])
    elif op == "lt":
        return np.less(inputs[0], inputs[1])
    elif op == "ne":
        return np.not_equal(inputs[0], inputs[1])
    elif op == "where":
        result = np.where(inputs[0], inputs[1], inputs[2])
    elif op == "cast":
        result = inputs[0]
    else:
        raise ValueError(f"Unsupported elementwise op: {op}")
    return np.asarray(result, dtype=np.float32)


def _execute_randn(shape: Sequence[int], seed: int, rng: str = "dinoml", dtype: str = "float32") -> np.ndarray:
    rng = rng.lower()
    if rng == "torch":
        return _execute_torch_cpu_randn(shape, seed, dtype)
    if rng not in {"dinoml", "numpy"}:
        raise ValueError(f"Unsupported randn rng: {rng}")
    numel = math.prod(int(dim) for dim in shape)
    idx = np.arange(numel, dtype=np.uint64)
    seed_value = np.uint64(seed)
    u1_bits = _splitmix64(idx + seed_value)
    u2_bits = _splitmix64(idx + seed_value + np.uint64(0x9E3779B97F4A7C15))
    u1 = (((u1_bits >> np.uint64(8)) & np.uint64(0xFFFFFF)).astype(np.float32) + np.float32(0.5)) * np.float32(1.0 / 16777216.0)
    u2 = (((u2_bits >> np.uint64(8)) & np.uint64(0xFFFFFF)).astype(np.float32) + np.float32(0.5)) * np.float32(1.0 / 16777216.0)
    radius = np.sqrt(np.float32(-2.0) * np.log(u1, dtype=np.float32), dtype=np.float32)
    theta = np.float32(6.2831853071795864769) * u2
    return (radius * np.cos(theta, dtype=np.float32)).astype(np.float32, copy=False).reshape(shape)


def _execute_torch_cpu_randn(shape: Sequence[int], seed: int, dtype: str) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("randn with rng='torch' requires torch for the reference implementation") from exc

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        torch_dtype = dtype_map[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch randn dtype: {dtype}") from exc
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensor = torch.randn(tuple(int(dim) for dim in shape), dtype=torch_dtype, device="cpu", generator=generator)
    if dtype == "bfloat16":
        tensor = tensor.float()
    return tensor.numpy()


def _splitmix64(value: np.ndarray) -> np.ndarray:
    value = value + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def _execute_softmax(value: np.ndarray, attrs: Mapping[str, object]) -> np.ndarray:
    dim = int(attrs.get("dim", -1))
    if dim < 0:
        dim += value.ndim
    if dim != value.ndim - 1:
        raise NotImplementedError("CPU reference softmax currently supports only the last dimension")
    shifted = value - np.max(value, axis=dim, keepdims=True)
    exp_value = np.exp(shifted)
    return np.asarray(exp_value / np.sum(exp_value, axis=dim, keepdims=True), dtype=np.float32)


def _execute_t5_layer_norm(value: np.ndarray, weight: np.ndarray, attrs: Mapping[str, object]) -> np.ndarray:
    if value.ndim < 1:
        raise ValueError("CPU reference t5_layer_norm requires rank >= 1 input")
    if weight.ndim != 1:
        raise ValueError("CPU reference t5_layer_norm requires rank-1 weight")
    hidden = int(value.shape[-1])
    if hidden <= 0:
        raise ValueError("CPU reference t5_layer_norm requires a positive last dimension")
    if int(weight.shape[0]) != hidden:
        raise ValueError(
            "CPU reference t5_layer_norm weight length must match input hidden size: "
            f"got hidden={hidden}, weight={weight.shape[0]}"
        )
    eps = float(attrs.get("eps", 1e-6))
    source = np.asarray(value, dtype=np.float32)
    mean_square = np.mean(source * source, axis=-1, keepdims=True)
    normalized = source * (1.0 / np.sqrt(mean_square + eps))
    normalized = np.asarray(normalized, dtype=value.dtype)
    result = np.asarray(normalized, dtype=np.float32) * np.asarray(weight, dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


def _execute_layer_norm(
    value: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    attrs: Mapping[str, object],
) -> np.ndarray:
    if value.ndim < 1:
        raise ValueError("CPU reference layer_norm requires rank >= 1 input")
    if weight.ndim != 1:
        raise ValueError("CPU reference layer_norm requires rank-1 weight")
    if bias.ndim != 1:
        raise ValueError("CPU reference layer_norm requires rank-1 bias")
    hidden = int(value.shape[-1])
    if hidden <= 0:
        raise ValueError("CPU reference layer_norm requires a positive last dimension")
    if int(weight.shape[0]) != hidden:
        raise ValueError(
            "CPU reference layer_norm weight length must match input hidden size: "
            f"got hidden={hidden}, weight={weight.shape[0]}"
        )
    if int(bias.shape[0]) != hidden:
        raise ValueError(
            "CPU reference layer_norm bias length must match input hidden size: "
            f"got hidden={hidden}, bias={bias.shape[0]}"
        )
    eps = float(attrs.get("eps", 1e-5))
    source = np.asarray(value, dtype=np.float32)
    scale = np.asarray(weight, dtype=np.float32)
    shift = np.asarray(bias, dtype=np.float32)
    mean = np.mean(source, axis=-1, keepdims=True)
    variance = np.maximum(np.mean(source * source, axis=-1, keepdims=True) - mean * mean, 0.0)
    normalized = (source - mean) * (1.0 / np.sqrt(variance + eps))
    return np.asarray(normalized * scale + shift, dtype=np.float32)


def _execute_group_norm(
    value: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    attrs: Mapping[str, object],
    *,
    apply_swish: bool,
) -> np.ndarray:
    if value.ndim < 2:
        raise ValueError("CPU reference group_norm requires rank >= 2 input")
    if weight.ndim != 1:
        raise ValueError("CPU reference group_norm requires rank-1 weight")
    if bias.ndim != 1:
        raise ValueError("CPU reference group_norm requires rank-1 bias")
    channels = int(value.shape[-1])
    if channels <= 0:
        raise ValueError("CPU reference group_norm requires a positive last dimension")
    if int(weight.shape[0]) != channels:
        raise ValueError(
            "CPU reference group_norm weight length must match input channels: "
            f"got channels={channels}, weight={weight.shape[0]}"
        )
    if int(bias.shape[0]) != channels:
        raise ValueError(
            "CPU reference group_norm bias length must match input channels: "
            f"got channels={channels}, bias={bias.shape[0]}"
        )
    num_groups = int(attrs["num_groups"])
    if num_groups <= 0:
        raise ValueError("CPU reference group_norm num_groups must be positive")
    if channels % num_groups != 0:
        raise ValueError(f"CPU reference group_norm requires channels divisible by num_groups, got {channels} and {num_groups}")
    eps = float(attrs.get("eps", 1e-5))
    source = np.asarray(value, dtype=np.float32)
    scale = np.asarray(weight, dtype=np.float32)
    shift = np.asarray(bias, dtype=np.float32)
    spatial_shape = source.shape[1:-1]
    channel_first = np.moveaxis(source, -1, 1)
    reshaped = channel_first.reshape(source.shape[0], num_groups, channels // num_groups, *spatial_shape)
    mean = np.mean(reshaped, axis=tuple(range(2, reshaped.ndim)), keepdims=True)
    variance = np.maximum(np.mean(reshaped * reshaped, axis=tuple(range(2, reshaped.ndim)), keepdims=True) - mean * mean, 0.0)
    normalized = (reshaped - mean) * (1.0 / np.sqrt(variance + eps))
    normalized = normalized.reshape(channel_first.shape)
    normalized = np.moveaxis(normalized, 1, -1)
    result = normalized * scale + shift
    if apply_swish:
        result = result / (1.0 + np.exp(-result))
    return np.asarray(result, dtype=np.float32)


def _execute_cropped_pos_embed(attrs: Mapping[str, object]) -> np.ndarray:
    normalized = normalize_cropped_pos_embed_attrs(
        embed_dim=attrs.get("embed_dim"),
        pos_embed_max_size=attrs.get("pos_embed_max_size"),
        base_size=attrs.get("base_size"),
        interpolation_scale=attrs.get("interpolation_scale"),
        patch_size=attrs.get("patch_size"),
        height=attrs.get("height"),
        width=attrs.get("width"),
    )
    embed_dim = int(normalized["embed_dim"])
    max_size = int(normalized["pos_embed_max_size"])
    crop_h = int(normalized["crop_h"])
    crop_w = int(normalized["crop_w"])
    top = int(normalized["top"])
    left = int(normalized["left"])
    grid_h = (
        np.arange(max_size, dtype=np.float32)
        / np.float32(float(max_size) / float(normalized["base_size"]))
        / np.float32(normalized["interpolation_scale"])
    )
    grid_w = (
        np.arange(max_size, dtype=np.float32)
        / np.float32(float(max_size) / float(normalized["base_size"]))
        / np.float32(normalized["interpolation_scale"])
    )
    grid_0, grid_1 = np.meshgrid(grid_w, grid_h, indexing="xy")
    pair_dim = embed_dim // 4
    omega = np.arange(pair_dim, dtype=np.float32) / np.float32(pair_dim)
    omega = np.float32(1.0) / np.power(np.float32(10000.0), omega, dtype=np.float32)
    out_0 = grid_0.reshape(-1, 1) * omega.reshape(1, -1)
    out_1 = grid_1.reshape(-1, 1) * omega.reshape(1, -1)
    pos_embed = np.concatenate(
        [
            np.sin(out_0).astype(np.float32, copy=False),
            np.cos(out_0).astype(np.float32, copy=False),
            np.sin(out_1).astype(np.float32, copy=False),
            np.cos(out_1).astype(np.float32, copy=False),
        ],
        axis=1,
    )
    spatial = pos_embed.reshape(1, max_size, max_size, embed_dim)
    return spatial[:, top : top + crop_h, left : left + crop_w, :].reshape(
        1,
        crop_h * crop_w,
        embed_dim,
    )


def _execute_gaussian_fourier_projection(
    x: np.ndarray,
    weight: np.ndarray,
    attrs: Mapping[str, object],
) -> np.ndarray:
    normalized = normalize_gaussian_fourier_projection_attrs(
        log=attrs.get("log", True),
        flip_sin_to_cos=attrs.get("flip_sin_to_cos", False),
    )
    if x.ndim != 1 or weight.ndim != 1:
        raise ValueError("CPU reference gaussian_fourier_projection expects rank-1 x and weight")
    x_value = np.asarray(x, dtype=np.float32)
    weight_value = np.asarray(weight, dtype=np.float32)
    if bool(normalized["log"]):
        x_value = np.log(x_value)
    x_proj = x_value[:, None] * weight_value[None, :] * np.float32(2.0 * math.pi)
    if bool(normalized["flip_sin_to_cos"]):
        return np.concatenate(
            [
                np.cos(x_proj).astype(np.float32, copy=False),
                np.sin(x_proj).astype(np.float32, copy=False),
            ],
            axis=1,
        )
    return np.concatenate(
        [
            np.sin(x_proj).astype(np.float32, copy=False),
            np.cos(x_proj).astype(np.float32, copy=False),
        ],
        axis=1,
    )


def _execute_get_fourier_embeds_from_boundingbox(
    box: np.ndarray,
    attrs: Mapping[str, object],
) -> np.ndarray:
    normalized = normalize_get_fourier_embeds_from_boundingbox_attrs(embed_dim=attrs.get("embed_dim"))
    if box.ndim != 3 or int(box.shape[2]) != 4:
        raise ValueError("CPU reference get_fourier_embeds_from_boundingbox expects [B, N, 4] input")
    box_value = np.asarray(box, dtype=np.float32)
    embed_dim = int(normalized["embed_dim"])
    emb = np.power(np.float32(100.0), np.arange(embed_dim, dtype=np.float32) / np.float32(embed_dim))
    projected = box_value[..., None] * emb.reshape(1, 1, 1, embed_dim)
    stacked = np.stack(
        (
            np.sin(projected).astype(np.float32, copy=False),
            np.cos(projected).astype(np.float32, copy=False),
        ),
        axis=-1,
    )
    return np.transpose(stacked, (0, 1, 3, 4, 2)).reshape(box.shape[0], box.shape[1], embed_dim * 8)


def _execute_relative_attention_bias(
    embedding: np.ndarray,
    attrs: Mapping[str, object],
) -> np.ndarray:
    normalized = normalize_relative_attention_bias_attrs(
        query_length=attrs.get("query_length"),
        key_length=attrs.get("key_length"),
        bidirectional=attrs.get("bidirectional", True),
        num_buckets=attrs.get("num_buckets", 32),
        max_distance=attrs.get("max_distance", 128),
    )
    if embedding.ndim != 2:
        raise ValueError("CPU reference relative_attention_bias expects [num_buckets, heads] embedding")
    embedding_value = np.asarray(embedding, dtype=np.float32)
    num_buckets, heads = [int(dim) for dim in embedding_value.shape]
    if num_buckets != int(normalized["num_buckets"]):
        raise ValueError("CPU reference relative_attention_bias num_buckets must match embedding.shape[0]")
    query_length = int(normalized["query_length"])
    key_length = int(normalized["key_length"])
    result = np.empty((1, heads, query_length, key_length), dtype=np.float32)
    for query_idx in range(query_length):
        for key_idx in range(key_length):
            bucket = _relative_attention_bucket(
                key_idx - query_idx,
                bidirectional=bool(normalized["bidirectional"]),
                num_buckets=int(normalized["num_buckets"]),
                max_distance=int(normalized["max_distance"]),
            )
            result[0, :, query_idx, key_idx] = embedding_value[bucket]
    return result


def _execute_sinusoidal_positional_embedding(
    x: np.ndarray,
    attrs: Mapping[str, object],
) -> np.ndarray:
    normalized = normalize_sinusoidal_positional_embedding_attrs(
        embed_dim=attrs.get("embed_dim"),
        max_seq_len=attrs.get("max_seq_len"),
    )
    if x.ndim != 3:
        raise ValueError("CPU reference sinusoidal_positional_embedding expects rank-3 x")
    x_value = np.asarray(x, dtype=np.float32)
    batch, seq_len, hidden = [int(dim) for dim in x_value.shape]
    embed_dim = int(normalized["embed_dim"])
    max_seq_len = int(normalized["max_seq_len"])
    if hidden != embed_dim:
        raise ValueError("CPU reference sinusoidal_positional_embedding embed_dim must match x.shape[-1]")
    if seq_len > max_seq_len:
        raise ValueError("CPU reference sinusoidal_positional_embedding seq length exceeds max_seq_len")
    position = np.arange(max_seq_len, dtype=np.float32).reshape(max_seq_len, 1)
    div_term = np.exp(
        np.arange(0, embed_dim, 2, dtype=np.float32) * (-np.log(np.float32(10000.0)) / np.float32(embed_dim))
    ).astype(np.float32, copy=False)
    pe = np.zeros((1, max_seq_len, embed_dim), dtype=np.float32)
    pe[:, :, 0::2] = np.sin(position * div_term).reshape(1, max_seq_len, -1)
    pe[:, :, 1::2] = np.cos(position * div_term[: pe[:, :, 1::2].shape[2]]).reshape(1, max_seq_len, -1)
    return (x_value + pe[:, :seq_len, :]).reshape(batch, seq_len, embed_dim)


def _relative_attention_bucket(
    relative_position: int,
    *,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
) -> int:
    bucket = 0
    buckets_per_direction = int(num_buckets)
    rel = int(relative_position)
    if bidirectional:
        buckets_per_direction //= 2
        if rel > 0:
            bucket += buckets_per_direction
        rel = abs(rel)
    else:
        rel = -min(rel, 0)

    max_exact = buckets_per_direction // 2
    if rel < max_exact:
        return bucket + rel
    scaled = max_exact + int(
        math.log(float(rel) / float(max_exact))
        / math.log(float(max_distance) / float(max_exact))
        * float(buckets_per_direction - max_exact)
    )
    return bucket + min(scaled, buckets_per_direction - 1)


def _execute_get_timestep_embedding(value: np.ndarray, attrs: Mapping[str, object]) -> np.ndarray:
    if value.ndim != 1:
        raise ValueError(f"CPU reference get_timestep_embedding expects rank-1 timesteps, got rank {value.ndim}")
    embedding_dim = int(attrs["embedding_dim"])
    flip_sin_to_cos = bool(attrs.get("flip_sin_to_cos", False))
    downscale_freq_shift = float(attrs.get("downscale_freq_shift", 1.0))
    scale = float(attrs.get("scale", 1.0))
    max_period = float(attrs.get("max_period", 10000.0))
    timesteps = np.asarray(value, dtype=np.float32)
    batch = int(timesteps.shape[0])
    half_dim = embedding_dim // 2
    if half_dim == 0:
        return np.zeros((batch, 1), dtype=np.float32)
    exponent = (
        -np.log(np.float32(max_period))
        * np.arange(half_dim, dtype=np.float32)
        / np.float32(float(half_dim) - downscale_freq_shift)
    )
    frequencies = np.exp(exponent).astype(np.float32, copy=False)
    args = timesteps[:, None] * frequencies[None, :]
    args = args * np.float32(scale)
    sin_part = np.sin(args).astype(np.float32, copy=False)
    cos_part = np.cos(args).astype(np.float32, copy=False)
    pieces = [cos_part, sin_part] if flip_sin_to_cos else [sin_part, cos_part]
    embedding = np.concatenate(pieces, axis=1)
    if embedding_dim % 2 == 1:
        embedding = np.concatenate([embedding, np.zeros((batch, 1), dtype=np.float32)], axis=1)
    return np.asarray(embedding, dtype=np.float32)


def _execute_get_1d_rotary_pos_embed_component(
    value: np.ndarray | None,
    attrs: Mapping[str, object],
    *,
    output_kind: str,
) -> np.ndarray:
    normalized = normalize_get_1d_rotary_pos_embed_attrs(
        dim=attrs.get("dim"),
        theta=attrs.get("theta", 10000.0),
        use_real=attrs.get("use_real", True),
        linear_factor=attrs.get("linear_factor", 1.0),
        ntk_factor=attrs.get("ntk_factor", 1.0),
        repeat_interleave_real=attrs.get("repeat_interleave_real", True),
        output_kind=output_kind,
    )
    if value is None:
        sequence_length = int(attrs.get("sequence_length", 0))
        if sequence_length <= 0:
            raise ValueError("CPU reference get_1d_rotary_pos_embed integer pos must be positive")
        positions = np.arange(sequence_length, dtype=np.float32)
    else:
        if value.ndim != 1:
            raise ValueError(f"CPU reference get_1d_rotary_pos_embed expects rank-1 pos, got rank {value.ndim}")
        positions = np.asarray(value, dtype=np.float32)
    rotary_dim = int(normalized["dim"]) // 2
    scaled_theta = np.float32(float(normalized["theta"]) * float(normalized["ntk_factor"]))
    exponent = -np.log(scaled_theta) * np.arange(rotary_dim, dtype=np.float32) / np.float32(rotary_dim)
    inv_freqs = np.exp(exponent).astype(np.float32, copy=False) / np.float32(float(normalized["linear_factor"]))
    freqs = positions[:, None] * inv_freqs[None, :]
    base = np.cos(freqs).astype(np.float32, copy=False) if output_kind == "cos" else np.sin(freqs).astype(np.float32, copy=False)
    if not bool(normalized["use_real"]):
        return np.asarray(base, dtype=np.float32)
    if bool(normalized["repeat_interleave_real"]):
        return np.repeat(base, 2, axis=1).astype(np.float32, copy=False)
    return np.concatenate([base, base], axis=1).astype(np.float32, copy=False)


def _execute_get_2d_rotary_pos_embed(attrs: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    normalized = normalize_get_2d_rotary_pos_embed_attrs(
        embed_dim=attrs.get("embed_dim"),
        crop_start_h=attrs.get("crop_start_h"),
        crop_start_w=attrs.get("crop_start_w"),
        crop_stop_h=attrs.get("crop_stop_h"),
        crop_stop_w=attrs.get("crop_stop_w"),
        grid_h=attrs.get("grid_h"),
        grid_w=attrs.get("grid_w"),
        theta=attrs.get("theta", 10000.0),
        use_real=attrs.get("use_real", True),
    )
    grid_h = int(normalized["grid_h"])
    grid_w = int(normalized["grid_w"])
    h_positions = np.linspace(
        float(normalized["crop_start_h"]),
        float(normalized["crop_stop_h"]) * float(grid_h - 1) / float(grid_h),
        num=grid_h,
        dtype=np.float32,
    )
    w_positions = np.linspace(
        float(normalized["crop_start_w"]),
        float(normalized["crop_stop_w"]) * float(grid_w - 1) / float(grid_w),
        num=grid_w,
        dtype=np.float32,
    )
    grid_0, grid_1 = np.meshgrid(w_positions, h_positions, indexing="xy")
    rotary_attrs = {
        "dim": int(normalized["embed_dim"]) // 2,
        "theta": float(normalized["theta"]),
        "use_real": True,
        "linear_factor": 1.0,
        "ntk_factor": 1.0,
        "repeat_interleave_real": True,
    }
    cos_h = _execute_get_1d_rotary_pos_embed_component(grid_0.reshape(-1), rotary_attrs, output_kind="cos")
    sin_h = _execute_get_1d_rotary_pos_embed_component(grid_0.reshape(-1), rotary_attrs, output_kind="sin")
    cos_w = _execute_get_1d_rotary_pos_embed_component(grid_1.reshape(-1), rotary_attrs, output_kind="cos")
    sin_w = _execute_get_1d_rotary_pos_embed_component(grid_1.reshape(-1), rotary_attrs, output_kind="sin")
    return (
        np.concatenate([cos_h, cos_w], axis=1).astype(np.float32, copy=False),
        np.concatenate([sin_h, sin_w], axis=1).astype(np.float32, copy=False),
    )


def _execute_get_2d_rotary_pos_embed_lumina(attrs: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    normalized = normalize_get_2d_rotary_pos_embed_lumina_attrs(
        embed_dim=attrs.get("embed_dim"),
        len_h=attrs.get("len_h"),
        len_w=attrs.get("len_w"),
        linear_factor=attrs.get("linear_factor", 1.0),
        ntk_factor=attrs.get("ntk_factor", 1.0),
    )
    embed_dim = int(normalized["embed_dim"])
    len_h = int(normalized["len_h"])
    len_w = int(normalized["len_w"])
    rotary_attrs = {
        "dim": embed_dim // 2,
        "theta": 10000.0,
        "use_real": False,
        "linear_factor": float(normalized["linear_factor"]),
        "ntk_factor": float(normalized["ntk_factor"]),
        "repeat_interleave_real": True,
    }
    pos_h = np.arange(len_h, dtype=np.float32)
    pos_w = np.arange(len_w, dtype=np.float32)
    real_h = _execute_get_1d_rotary_pos_embed_component(pos_h, rotary_attrs, output_kind="cos")
    imag_h = _execute_get_1d_rotary_pos_embed_component(pos_h, rotary_attrs, output_kind="sin")
    real_w = _execute_get_1d_rotary_pos_embed_component(pos_w, rotary_attrs, output_kind="cos")
    imag_w = _execute_get_1d_rotary_pos_embed_component(pos_w, rotary_attrs, output_kind="sin")
    quarter_dim = embed_dim // 4
    real = np.empty((len_h, len_w, embed_dim // 2), dtype=np.float32)
    imag = np.empty_like(real)
    for h_idx in range(len_h):
        for w_idx in range(len_w):
            for dim_idx in range(quarter_dim):
                base = 2 * dim_idx
                real[h_idx, w_idx, base] = real_h[h_idx, dim_idx]
                imag[h_idx, w_idx, base] = imag_h[h_idx, dim_idx]
                real[h_idx, w_idx, base + 1] = real_w[w_idx, dim_idx]
                imag[h_idx, w_idx, base + 1] = imag_w[w_idx, dim_idx]
    return real, imag


def _execute_get_3d_rotary_pos_embed(attrs: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    normalized = normalize_get_3d_rotary_pos_embed_attrs(
        embed_dim=attrs.get("embed_dim"),
        crop_start_h=attrs.get("crop_start_h"),
        crop_start_w=attrs.get("crop_start_w"),
        crop_stop_h=attrs.get("crop_stop_h"),
        crop_stop_w=attrs.get("crop_stop_w"),
        grid_h=attrs.get("grid_h"),
        grid_w=attrs.get("grid_w"),
        temporal_size=attrs.get("temporal_size"),
        theta=attrs.get("theta", 10000.0),
        use_real=attrs.get("use_real", True),
        grid_type=attrs.get("grid_type", "linspace"),
        max_h=attrs.get("max_h", 0),
        max_w=attrs.get("max_w", 0),
    )
    embed_dim = int(normalized["embed_dim"])
    grid_h = int(normalized["grid_h"])
    grid_w = int(normalized["grid_w"])
    temporal_size = int(normalized["temporal_size"])
    if str(normalized["grid_type"]) == "linspace":
        h_positions = np.linspace(
            float(normalized["crop_start_h"]),
            float(normalized["crop_stop_h"]) * float(grid_h - 1) / float(grid_h),
            num=grid_h,
            dtype=np.float32,
        )
        w_positions = np.linspace(
            float(normalized["crop_start_w"]),
            float(normalized["crop_stop_w"]) * float(grid_w - 1) / float(grid_w),
            num=grid_w,
            dtype=np.float32,
        )
        t_positions = np.linspace(
            0.0,
            float(temporal_size) * float(temporal_size - 1) / float(temporal_size),
            num=temporal_size,
            dtype=np.float32,
        )
    else:
        h_positions = np.arange(int(normalized["max_h"]), dtype=np.float32)
        w_positions = np.arange(int(normalized["max_w"]), dtype=np.float32)
        t_positions = np.arange(temporal_size, dtype=np.float32)
    dim_t = embed_dim // 4
    dim_h = (embed_dim // 8) * 3
    dim_w = (embed_dim // 8) * 3
    rotary_t = {
        "dim": dim_t,
        "theta": float(normalized["theta"]),
        "use_real": True,
        "linear_factor": 1.0,
        "ntk_factor": 1.0,
        "repeat_interleave_real": True,
    }
    rotary_h = {
        "dim": dim_h,
        "theta": float(normalized["theta"]),
        "use_real": True,
        "linear_factor": 1.0,
        "ntk_factor": 1.0,
        "repeat_interleave_real": True,
    }
    rotary_w = {
        "dim": dim_w,
        "theta": float(normalized["theta"]),
        "use_real": True,
        "linear_factor": 1.0,
        "ntk_factor": 1.0,
        "repeat_interleave_real": True,
    }
    t_cos = _execute_get_1d_rotary_pos_embed_component(t_positions, rotary_t, output_kind="cos")
    t_sin = _execute_get_1d_rotary_pos_embed_component(t_positions, rotary_t, output_kind="sin")
    h_cos = _execute_get_1d_rotary_pos_embed_component(h_positions, rotary_h, output_kind="cos")
    h_sin = _execute_get_1d_rotary_pos_embed_component(h_positions, rotary_h, output_kind="sin")
    w_cos = _execute_get_1d_rotary_pos_embed_component(w_positions, rotary_w, output_kind="cos")
    w_sin = _execute_get_1d_rotary_pos_embed_component(w_positions, rotary_w, output_kind="sin")
    if str(normalized["grid_type"]) == "slice":
        h_cos = h_cos[:grid_h]
        h_sin = h_sin[:grid_h]
        w_cos = w_cos[:grid_w]
        w_sin = w_sin[:grid_w]

    def _combine(freq_t: np.ndarray, freq_h: np.ndarray, freq_w: np.ndarray) -> np.ndarray:
        freq_t_b = np.broadcast_to(freq_t[:, None, None, :], (temporal_size, grid_h, grid_w, freq_t.shape[1]))
        freq_h_b = np.broadcast_to(freq_h[None, :, None, :], (temporal_size, grid_h, grid_w, freq_h.shape[1]))
        freq_w_b = np.broadcast_to(freq_w[None, None, :, :], (temporal_size, grid_h, grid_w, freq_w.shape[1]))
        return np.concatenate([freq_t_b, freq_h_b, freq_w_b], axis=-1).reshape(temporal_size * grid_h * grid_w, -1)

    return (
        _combine(t_cos, h_cos, w_cos).astype(np.float32, copy=False),
        _combine(t_sin, h_sin, w_sin).astype(np.float32, copy=False),
    )


def _execute_get_3d_rotary_pos_embed_allegro(attrs: Mapping[str, object]) -> tuple[np.ndarray, ...]:
    normalized = normalize_get_3d_rotary_pos_embed_allegro_attrs(
        height=attrs.get("height"),
        width=attrs.get("width"),
        num_frames=attrs.get("num_frames"),
        vae_scale_factor_spatial=attrs.get("vae_scale_factor_spatial", 8),
        patch_size=attrs.get("patch_size", 2),
        interpolation_scale_h=attrs.get("interpolation_scale_h", 2.0),
        interpolation_scale_t=attrs.get("interpolation_scale_t", 2.2),
        interpolation_scale_w=attrs.get("interpolation_scale_w", 2.0),
        attention_head_dim=attrs.get("attention_head_dim", 96),
    )
    num_frames = int(normalized["num_frames"])
    grid_h = int(normalized["grid_h"])
    grid_w = int(normalized["grid_w"])
    dim_axis = int(normalized["attention_head_dim"]) // 3
    t_positions = np.linspace(
        0.0,
        float(num_frames) * float(num_frames - 1) / float(num_frames),
        num=num_frames,
        dtype=np.float32,
    ) / np.float32(normalized["interpolation_scale_t"])
    h_positions = np.linspace(0.0, float(grid_h - 1), num=grid_h, dtype=np.float32) / np.float32(
        normalized["interpolation_scale_h"]
    )
    w_positions = np.linspace(0.0, float(grid_w - 1), num=grid_w, dtype=np.float32) / np.float32(
        normalized["interpolation_scale_w"]
    )
    rotary_attrs = {
        "dim": dim_axis,
        "theta": 10000.0,
        "use_real": True,
        "linear_factor": 1.0,
        "ntk_factor": 1.0,
        "repeat_interleave_real": False,
    }
    t_cos = _execute_get_1d_rotary_pos_embed_component(t_positions, rotary_attrs, output_kind="cos")
    t_sin = _execute_get_1d_rotary_pos_embed_component(t_positions, rotary_attrs, output_kind="sin")
    h_cos = _execute_get_1d_rotary_pos_embed_component(h_positions, rotary_attrs, output_kind="cos")
    h_sin = _execute_get_1d_rotary_pos_embed_component(h_positions, rotary_attrs, output_kind="sin")
    w_cos = _execute_get_1d_rotary_pos_embed_component(w_positions, rotary_attrs, output_kind="cos")
    w_sin = _execute_get_1d_rotary_pos_embed_component(w_positions, rotary_attrs, output_kind="sin")
    grid_t_vals = np.arange(num_frames, dtype=np.int64)
    grid_h_vals = np.arange(grid_h, dtype=np.int64)
    grid_w_vals = np.arange(grid_w, dtype=np.int64)
    grid_t, grid_h_arr, grid_w_arr = np.meshgrid(grid_t_vals, grid_h_vals, grid_w_vals, indexing="ij")
    return (
        t_cos.astype(np.float32, copy=False),
        t_sin.astype(np.float32, copy=False),
        h_cos.astype(np.float32, copy=False),
        h_sin.astype(np.float32, copy=False),
        w_cos.astype(np.float32, copy=False),
        w_sin.astype(np.float32, copy=False),
        grid_t.reshape(1, -1),
        grid_h_arr.reshape(1, -1),
        grid_w_arr.reshape(1, -1),
    )


def _execute_glm_ocr_text_rope(
    q: np.ndarray,
    k: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    attrs: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    rotary_dim = int(attrs["rotary_dim"])
    cos_value = np.asarray(cos, dtype=np.float32)
    sin_value = np.asarray(sin, dtype=np.float32)

    def apply(value: np.ndarray) -> np.ndarray:
        source = np.asarray(value, dtype=np.float32)
        result = source.copy()
        rot = source[..., :rotary_dim]
        c = np.repeat(cos_value[..., : rotary_dim // 2], 2, axis=-1)[..., None, :]
        s = np.repeat(sin_value[..., : rotary_dim // 2], 2, axis=-1)[..., None, :]
        rotated = np.empty_like(rot)
        rotated[..., 0::2] = -rot[..., 1::2]
        rotated[..., 1::2] = rot[..., 0::2]
        result[..., :rotary_dim] = rot * c + rotated * s
        return result

    return apply(q), apply(k)


def _execute_glm_ocr_vision_rope(
    q: np.ndarray,
    k: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cos_value = np.asarray(cos, dtype=np.float32)[:, None, :]
    sin_value = np.asarray(sin, dtype=np.float32)[:, None, :]

    def apply(value: np.ndarray) -> np.ndarray:
        source = np.asarray(value, dtype=np.float32)
        half = source.shape[-1] // 2
        rotated = np.concatenate([-source[..., half:], source[..., :half]], axis=-1)
        return source * cos_value + rotated * sin_value

    return apply(q), apply(k)


def _execute_glm_ocr_stitch_image_features(
    input_ids: np.ndarray,
    inputs_embeds: np.ndarray,
    image_features: np.ndarray,
    attrs: Mapping[str, object],
) -> np.ndarray:
    image_token_id = int(attrs["image_token_id"])
    ids = np.asarray(input_ids)
    embeds = np.asarray(inputs_embeds).copy()
    features = np.asarray(image_features, dtype=embeds.dtype)
    if ids.ndim != 2 or embeds.ndim != 3 or features.ndim != 2:
        raise ValueError("CPU reference glm_ocr_stitch_image_features expects [1, seq], [1, seq, hidden], [image_seq, hidden]")
    if ids.shape[0] != 1 or embeds.shape[0] != 1:
        raise ValueError("CPU reference glm_ocr_stitch_image_features currently supports only batch=1")
    if ids.shape[1] != embeds.shape[1]:
        raise ValueError("CPU reference glm_ocr_stitch_image_features input_ids and inputs_embeds sequence lengths must match")
    if features.shape[1] != embeds.shape[2]:
        raise ValueError("CPU reference glm_ocr_stitch_image_features image_features hidden size must match inputs_embeds")
    positions = np.flatnonzero(ids[0] == image_token_id)
    if positions.size != features.shape[0]:
        raise ValueError(
            "CPU reference glm_ocr_stitch_image_features token count does not match image_features rows: "
            f"tokens={positions.size}, features={features.shape[0]}"
        )
    if positions.size:
        expected = np.arange(int(positions[0]), int(positions[0]) + positions.size, dtype=positions.dtype)
        if not np.array_equal(positions, expected):
            raise ValueError("CPU reference glm_ocr_stitch_image_features expects a contiguous image token block")
        embeds[0, positions[0] : positions[0] + positions.size, :] = features
    return embeds


def _execute_qwen2_5_vl_stitch_image_features(
    input_ids: np.ndarray,
    inputs_embeds: np.ndarray,
    image_features: np.ndarray,
    attrs: Mapping[str, object],
) -> np.ndarray:
    image_token_id = int(attrs["image_token_id"])
    ids = np.asarray(input_ids)
    embeds = np.asarray(inputs_embeds).copy()
    features = np.asarray(image_features, dtype=embeds.dtype)
    if ids.ndim != 2 or embeds.ndim != 3 or features.ndim != 2:
        raise ValueError(
            "CPU reference qwen2_5_vl_stitch_image_features expects [batch, seq], [batch, seq, hidden], [image_seq, hidden]"
        )
    if ids.shape != embeds.shape[:2]:
        raise ValueError(
            "CPU reference qwen2_5_vl_stitch_image_features input_ids shape must match inputs_embeds leading dimensions"
        )
    if features.shape[1] != embeds.shape[2]:
        raise ValueError(
            "CPU reference qwen2_5_vl_stitch_image_features image_features hidden size must match inputs_embeds"
        )
    positions = np.argwhere(ids == image_token_id)
    if positions.shape[0] != features.shape[0]:
        raise ValueError(
            "CPU reference qwen2_5_vl_stitch_image_features token count does not match image_features rows: "
            f"tokens={positions.shape[0]}, features={features.shape[0]}"
        )
    for feature_idx, (batch_idx, seq_idx) in enumerate(positions):
        embeds[int(batch_idx), int(seq_idx), :] = features[feature_idx]
    return embeds


def _execute_swiglu(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    hidden = source.shape[-1] // 2
    gate = source[..., :hidden]
    up = source[..., hidden:]
    return up * (gate / (1.0 + np.exp(-gate)))


def _execute_reduction(op: str, value: np.ndarray, attrs: Mapping[str, object]) -> np.ndarray:
    dim = int(attrs.get("dim", -1))
    if dim < 0:
        dim += value.ndim
    if dim != value.ndim - 1:
        raise NotImplementedError("CPU reference reductions currently support only the last dimension")
    keepdim = bool(attrs.get("keepdim", False))
    if op == "reduce_sum":
        result = np.sum(value, axis=dim, keepdims=keepdim)
    elif op == "reduce_max":
        result = np.max(value, axis=dim, keepdims=keepdim)
    elif op == "reduce_min":
        result = np.min(value, axis=dim, keepdims=keepdim)
    elif op == "reduce_mean":
        result = np.mean(value, axis=dim, keepdims=keepdim)
    elif op == "var":
        ddof = 1 if bool(attrs.get("unbiased", False)) else 0
        result = np.var(value, axis=dim, keepdims=keepdim, ddof=ddof)
    elif op == "vector_norm":
        ord_value = float(attrs.get("ord", 2.0))
        if ord_value != 2.0:
            raise NotImplementedError("CPU reference vector_norm currently supports only ord=2")
        result = np.sqrt(np.sum(value * value, axis=dim, keepdims=keepdim))
    else:
        raise ValueError(f"Unsupported reduction op: {op}")
    if result.shape == ():
        result = np.reshape(result, [1])
    return np.asarray(result, dtype=np.float32)


def _execute_argmax(value: np.ndarray, attrs: Mapping[str, object]) -> np.ndarray:
    dim = int(attrs.get("dim", -1))
    if dim < 0:
        dim += value.ndim
    if dim != value.ndim - 1:
        raise NotImplementedError("CPU reference argmax currently supports only the last dimension")
    result = np.argmax(value, axis=dim)
    if bool(attrs.get("keepdim", False)):
        result = np.expand_dims(result, axis=dim)
    if result.shape == ():
        result = np.reshape(result, [1])
    return np.asarray(result, dtype=np.int64)


def _execute_topk_indices(value: np.ndarray, attrs: Mapping[str, object]) -> np.ndarray:
    dim = int(attrs.get("dim", -1))
    if dim < 0:
        dim += value.ndim
    if dim != value.ndim - 1:
        raise NotImplementedError("CPU reference topk currently supports only the last dimension")
    if not bool(attrs.get("largest", True)):
        raise NotImplementedError("CPU reference topk currently supports only largest=True")
    if not bool(attrs.get("sorted", True)):
        raise NotImplementedError("CPU reference topk currently supports only sorted=True")
    k = int(attrs["k"])
    rows = int(np.prod(value.shape[:-1], dtype=np.int64)) if value.ndim > 1 else 1
    cols = int(value.shape[-1])
    source = np.reshape(value, (rows, cols))
    result = np.empty((rows, k), dtype=np.int64)
    for row in range(rows):
        used = np.zeros((cols,), dtype=np.bool_)
        for out_col in range(k):
            best_index = -1
            for col in range(cols):
                if used[col]:
                    continue
                if best_index < 0 or _topk_is_better(source[row, col], source[row, best_index]):
                    best_index = col
            used[best_index] = True
            result[row, out_col] = best_index
    return np.reshape(result, (*value.shape[:-1], k))


def _topk_is_better(candidate: object, current: object) -> bool:
    if isinstance(candidate, (bool, np.bool_)) or isinstance(current, (bool, np.bool_)):
        return bool(candidate) > bool(current)
    candidate_float = float(candidate)
    current_float = float(current)
    return candidate_float > current_float or (math.isnan(candidate_float) and not math.isnan(current_float))


def _execute_gemm(op: str, inputs: Sequence[np.ndarray]) -> np.ndarray:
    spec = gemm_op_spec(op)
    a = inputs[0]
    b = inputs[1]
    if spec.base_layout == "rrr":
        result = np.matmul(a, b)
    elif spec.base_layout == "rcr":
        result = np.matmul(a, np.swapaxes(b, -1, -2))
    else:
        raise ValueError(f"Unsupported GEMM op: {op}")
    if spec.epilogue.has_bias:
        bias = np.reshape(inputs[2], [-1])
        result = result + bias
    if spec.epilogue.pre_residual_activation is not None:
        result = _execute_gemm_activation(spec.epilogue.pre_residual_activation, result)
    if spec.epilogue.name == "bias_add":
        result = result + inputs[3]
    elif spec.epilogue.name == "bias_add_add":
        result = result + inputs[3] + inputs[4]
    elif spec.epilogue.name == "bias_add_relu":
        result = result + inputs[3]
    elif spec.epilogue.name == "bias_add_add_relu":
        result = result + inputs[3] + inputs[4]
    elif spec.epilogue.name in {"bias_mul", "bias_mul_tanh", "bias_sigmoid_mul", "bias_sigmoid_mul_tanh"}:
        result = result * inputs[3]
    elif spec.epilogue.name == "bias_mul_add":
        result = result * inputs[3] + inputs[4]
    if spec.epilogue.activation is not None:
        result = _execute_gemm_activation(spec.epilogue.activation, result)
    return np.asarray(result, dtype=np.float32)


def _execute_bmm(op: str, inputs: Sequence[np.ndarray]) -> np.ndarray:
    spec = bmm_op_spec(op)
    a = _logical_bmm_a(inputs[0], spec.a_layout)
    b = _logical_bmm_b(inputs[1], spec.b_layout)
    result = np.matmul(a, b)
    if spec.c_layout == "c":
        result = np.swapaxes(result, -1, -2)
    if spec.epilogue == "add":
        result = result + inputs[2]
    return np.asarray(result, dtype=np.float32)


def _execute_flash_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, attrs: Mapping[str, object]) -> np.ndarray:
    q_value = np.asarray(q, dtype=np.float32)
    k_value = np.asarray(k, dtype=np.float32)
    v_value = np.asarray(v, dtype=np.float32)
    if q_value.ndim != 4 or k_value.ndim != 4 or v_value.ndim != 4:
        raise ValueError("CPU reference flash_attention expects rank-4 q, k, and v")
    if k_value.shape != v_value.shape:
        raise ValueError("CPU reference flash_attention key/value shape mismatch")
    batch, seqlen_q, heads_q, head_dim = q_value.shape
    batch_k, seqlen_k, heads_k, head_dim_k = k_value.shape
    if batch != batch_k or head_dim != head_dim_k or heads_q % heads_k != 0:
        raise ValueError("CPU reference flash_attention shape mismatch")
    if heads_q != heads_k:
        repeat = heads_q // heads_k
        k_value = np.repeat(k_value, repeat, axis=2)
        v_value = np.repeat(v_value, repeat, axis=2)
    scores = np.einsum("bqhd,bkhd->bhqk", q_value, k_value) * np.float32(1.0 / math.sqrt(float(head_dim)))
    if bool(attrs.get("causal", False)):
        q_idx = np.arange(seqlen_q)[:, None]
        k_idx = np.arange(seqlen_k)[None, :]
        scores = np.where(k_idx > q_idx, np.float32(-1.0e30), scores)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", probs, v_value).astype(np.float32, copy=False)


def _execute_flash_attention_bias(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    bias: np.ndarray,
    attrs: Mapping[str, object],
) -> np.ndarray:
    q_value = np.asarray(q, dtype=np.float32)
    k_value = np.asarray(k, dtype=np.float32)
    v_value = np.asarray(v, dtype=np.float32)
    bias_value = np.asarray(bias, dtype=np.float32)
    if q_value.ndim != 4 or k_value.ndim != 4 or v_value.ndim != 4:
        raise ValueError("CPU reference flash_attention_bias expects rank-4 q, k, and v")
    if k_value.shape != v_value.shape:
        raise ValueError("CPU reference flash_attention_bias key/value shape mismatch")
    batch, seqlen_q, heads_q, head_dim = q_value.shape
    batch_k, seqlen_k, heads_k, head_dim_k = k_value.shape
    if batch != batch_k or head_dim != head_dim_k or heads_q % heads_k != 0:
        raise ValueError("CPU reference flash_attention_bias shape mismatch")
    if heads_q != heads_k:
        repeat = heads_q // heads_k
        k_value = np.repeat(k_value, repeat, axis=2)
        v_value = np.repeat(v_value, repeat, axis=2)
    scores = np.einsum("bqhd,bkhd->bhqk", q_value, k_value) * np.float32(1.0 / math.sqrt(float(head_dim)))
    scores = scores + _broadcast_flash_attention_bias(bias_value, batch, heads_q, seqlen_q, seqlen_k)
    if bool(attrs.get("causal", False)):
        q_idx = np.arange(seqlen_q)[:, None]
        k_idx = np.arange(seqlen_k)[None, :]
        scores = np.where(k_idx > q_idx, np.float32(-1.0e30), scores)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", probs, v_value).astype(np.float32, copy=False)


def _execute_flash_attention_varlen(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    cu_seqlens: np.ndarray,
    attrs: Mapping[str, object],
) -> np.ndarray:
    q_value = np.asarray(q, dtype=np.float32)
    k_value = np.asarray(k, dtype=np.float32)
    v_value = np.asarray(v, dtype=np.float32)
    starts = np.asarray(cu_seqlens, dtype=np.int64).reshape(-1)
    if q_value.ndim != 3 or k_value.ndim != 3 or v_value.ndim != 3:
        raise ValueError("CPU reference flash_attention_varlen expects rank-3 q, k, and v")
    if starts.ndim != 1 or starts.shape[0] < 2:
        raise ValueError("CPU reference flash_attention_varlen expects rank-1 cu_seqlens length >= 2")
    if k_value.shape != v_value.shape:
        raise ValueError("CPU reference flash_attention_varlen key/value shape mismatch")
    if q_value.shape[0] != k_value.shape[0] or q_value.shape[2] != k_value.shape[2]:
        raise ValueError("CPU reference flash_attention_varlen shape mismatch")
    output = np.empty_like(q_value, dtype=np.float32)
    for start, end in zip(starts[:-1], starts[1:], strict=True):
        start_i = int(start)
        end_i = int(end)
        if start_i < 0 or end_i < start_i or end_i > q_value.shape[0]:
            raise ValueError("CPU reference flash_attention_varlen cu_seqlens out of bounds")
        if end_i == start_i:
            continue
        chunk = _execute_flash_attention(
            q_value[None, start_i:end_i],
            k_value[None, start_i:end_i],
            v_value[None, start_i:end_i],
            attrs,
        )
        output[start_i:end_i] = chunk[0]
    return output


def _broadcast_flash_attention_bias(
    bias: np.ndarray,
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_k: int,
) -> np.ndarray:
    if bias.ndim == 2:
        bias_value = bias.reshape(1, 1, bias.shape[0], bias.shape[1])
    elif bias.ndim == 3:
        bias_value = bias.reshape(1, bias.shape[0], bias.shape[1], bias.shape[2])
    elif bias.ndim == 4:
        bias_value = bias
    else:
        raise ValueError("CPU reference flash_attention_bias bias must have rank 2, 3, or 4")
    if bias_value.shape[0] not in (1, batch) or bias_value.shape[1] not in (1, heads):
        raise ValueError("CPU reference flash_attention_bias bias batch/head shape mismatch")
    if bias_value.shape[2] != seqlen_q or bias_value.shape[3] != seqlen_k:
        raise ValueError("CPU reference flash_attention_bias bias sequence shape mismatch")
    return np.broadcast_to(bias_value, (batch, heads, seqlen_q, seqlen_k))


def _execute_flash_attention_static_kv_cache(
    q: np.ndarray,
    past_key: np.ndarray,
    past_value: np.ndarray,
    new_key: np.ndarray,
    new_value: np.ndarray,
    cache_seqlens: np.ndarray,
) -> np.ndarray:
    return _execute_flash_attention_static_kv_cache_impl(
        q,
        past_key,
        past_value,
        new_key,
        new_value,
        cache_seqlens,
        bias=None,
    )


def _execute_flash_attention_static_kv_cache_bias(
    q: np.ndarray,
    past_key: np.ndarray,
    past_value: np.ndarray,
    new_key: np.ndarray,
    new_value: np.ndarray,
    cache_seqlens: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    return _execute_flash_attention_static_kv_cache_impl(
        q,
        past_key,
        past_value,
        new_key,
        new_value,
        cache_seqlens,
        bias=bias,
    )


def _execute_flash_attention_static_kv_cache_impl(
    q: np.ndarray,
    past_key: np.ndarray,
    past_value: np.ndarray,
    new_key: np.ndarray,
    new_value: np.ndarray,
    cache_seqlens: np.ndarray,
    *,
    bias: np.ndarray | None,
) -> np.ndarray:
    q_value = np.asarray(q, dtype=np.float32)
    past_key_value = np.asarray(past_key, dtype=np.float32)
    past_value_value = np.asarray(past_value, dtype=np.float32)
    new_key_value = np.asarray(new_key, dtype=np.float32)
    new_value_value = np.asarray(new_value, dtype=np.float32)
    cache_lengths = np.asarray(cache_seqlens, dtype=np.int32)
    bias_value = None if bias is None else np.asarray(bias, dtype=np.float32)
    if q_value.ndim != 4 or past_key_value.ndim != 4 or past_value_value.ndim != 4:
        raise ValueError("CPU reference flash_attention_static_kv_cache expects rank-4 q and cache tensors")
    if new_key_value.ndim != 4 or new_value_value.ndim != 4:
        raise ValueError("CPU reference flash_attention_static_kv_cache expects rank-4 new K/V tensors")
    if past_key_value.shape != past_value_value.shape:
        raise ValueError("CPU reference flash_attention_static_kv_cache past key/value shape mismatch")
    if new_key_value.shape != new_value_value.shape:
        raise ValueError("CPU reference flash_attention_static_kv_cache new key/value shape mismatch")
    batch, seqlen_q, heads_q, head_dim = q_value.shape
    if seqlen_q != 1:
        raise ValueError("CPU reference flash_attention_static_kv_cache expects q sequence length 1")
    if cache_lengths.shape != (batch,):
        raise ValueError("CPU reference flash_attention_static_kv_cache cache_seqlens shape mismatch")
    max_cache_len = past_key_value.shape[2]
    broadcast_bias = (
        None
        if bias_value is None
        else _broadcast_flash_attention_bias(bias_value, batch, heads_q, seqlen_q, max_cache_len)
    )
    outputs = []
    for batch_idx in range(batch):
        valid_past = int(cache_lengths[batch_idx])
        if valid_past < 0 or valid_past >= max_cache_len:
            raise ValueError("CPU reference flash_attention_static_kv_cache cache length out of range")
        total_len = valid_past + 1
        k_batch = np.concatenate(
            [
                np.transpose(past_key_value[batch_idx : batch_idx + 1, :, :valid_past, :], (0, 2, 1, 3)),
                np.transpose(new_key_value[batch_idx : batch_idx + 1, :, :, :], (0, 2, 1, 3)),
            ],
            axis=1,
        )
        v_batch = np.concatenate(
            [
                np.transpose(past_value_value[batch_idx : batch_idx + 1, :, :valid_past, :], (0, 2, 1, 3)),
                np.transpose(new_value_value[batch_idx : batch_idx + 1, :, :, :], (0, 2, 1, 3)),
            ],
            axis=1,
        )
        if broadcast_bias is None:
            outputs.append(
                _execute_flash_attention(q_value[batch_idx : batch_idx + 1], k_batch, v_batch, {"causal": False})
            )
        else:
            outputs.append(
                _execute_flash_attention_bias(
                    q_value[batch_idx : batch_idx + 1],
                    k_batch,
                    v_batch,
                    broadcast_bias[batch_idx : batch_idx + 1, :, :, :total_len],
                    {"causal": False},
                )
            )
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _logical_bmm_a(value: np.ndarray, layout: str) -> np.ndarray:
    if layout == "r":
        return value
    if layout == "c":
        return np.swapaxes(value, -1, -2)
    raise ValueError(f"Unsupported BMM A layout: {layout}")


def _logical_bmm_b(value: np.ndarray, layout: str) -> np.ndarray:
    if layout == "r":
        return value
    if layout == "c":
        return np.swapaxes(value, -1, -2)
    raise ValueError(f"Unsupported BMM B layout: {layout}")


def _execute_gemm_activation(activation: str, value: np.ndarray) -> np.ndarray:
    if activation == "relu":
        return np.maximum(value, 0.0)
    if activation == "gelu":
        return 0.5 * value * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (value + 0.044715 * value * value * value)))
    if activation == "fast_gelu":
        return 0.5 * value * (1.0 + np.tanh(0.7978845608 * value * (1.0 + 0.044715 * value * value)))
    if activation == "quick_gelu":
        return value / (1.0 + np.exp(-1.702 * value))
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-value))
    if activation == "tanh":
        return np.tanh(value)
    if activation == "swish":
        return value / (1.0 + np.exp(-value))
    if activation == "hardswish":
        return value * np.clip(value + 3.0, 0.0, 6.0) / 6.0
    if activation == "elup1":
        return np.where(value >= 0.0, value + 1.0, np.exp(value))
    raise ValueError(f"Unsupported GEMM activation: {activation}")


def _execute_avg_pool1d(value: np.ndarray, attrs: Mapping[str, object]) -> np.ndarray:
    kernel = int(attrs["kernel_size"][0])
    stride_values = attrs.get("stride", attrs["kernel_size"])
    stride = int(stride_values[0])
    padding = int(attrs.get("padding", (0,))[0])
    batch, channels, length = value.shape
    out_length = (length + 2 * padding - kernel) // stride + 1
    result = np.empty((batch, channels, out_length), dtype=np.float32)
    divisor = float(kernel)
    source = np.asarray(value, dtype=np.float32)
    for n in range(batch):
        for c in range(channels):
            for ol in range(out_length):
                l_start = ol * stride - padding
                total = 0.0
                for kl in range(kernel):
                    il = l_start + kl
                    if il < 0 or il >= length:
                        continue
                    total += float(source[n, c, il])
                result[n, c, ol] = total / divisor
    return result


def _execute_avg_pool2d(value: np.ndarray, attrs: Mapping[str, object]) -> np.ndarray:
    kernel_h, kernel_w = [int(item) for item in attrs["kernel_size"]]
    stride_values = attrs.get("stride", attrs["kernel_size"])
    stride_h, stride_w = [int(item) for item in stride_values]
    pad_h, pad_w = [int(item) for item in attrs.get("padding", (0, 0))]
    batch, channels, height, width = value.shape
    out_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
    result = np.empty((batch, channels, out_height, out_width), dtype=np.float32)
    divisor = float(kernel_h * kernel_w)
    source = np.asarray(value, dtype=np.float32)
    for n in range(batch):
        for c in range(channels):
            for oh in range(out_height):
                h_start = oh * stride_h - pad_h
                for ow in range(out_width):
                    w_start = ow * stride_w - pad_w
                    total = 0.0
                    for kh in range(kernel_h):
                        ih = h_start + kh
                        if ih < 0 or ih >= height:
                            continue
                        for kw in range(kernel_w):
                            iw = w_start + kw
                            if iw < 0 or iw >= width:
                                continue
                            total += float(source[n, c, ih, iw])
                    result[n, c, oh, ow] = total / divisor
    return result


def _execute_max_pool2d(value: np.ndarray, attrs: Mapping[str, object]) -> np.ndarray:
    kernel_h, kernel_w = [int(item) for item in attrs["kernel_size"]]
    stride_values = attrs.get("stride", attrs["kernel_size"])
    stride_h, stride_w = [int(item) for item in stride_values]
    pad_h, pad_w = [int(item) for item in attrs.get("padding", (0, 0))]
    batch, channels, height, width = value.shape
    out_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
    result = np.empty((batch, channels, out_height, out_width), dtype=np.float32)
    source = np.asarray(value, dtype=np.float32)
    for n in range(batch):
        for c in range(channels):
            for oh in range(out_height):
                h_start = oh * stride_h - pad_h
                for ow in range(out_width):
                    w_start = ow * stride_w - pad_w
                    max_value = -np.inf
                    for kh in range(kernel_h):
                        ih = h_start + kh
                        if ih < 0 or ih >= height:
                            continue
                        for kw in range(kernel_w):
                            iw = w_start + kw
                            if iw < 0 or iw >= width:
                                continue
                            max_value = max(max_value, float(source[n, c, ih, iw]))
                    result[n, c, oh, ow] = max_value
    return result


def _execute_conv1d_bias_family(
    op_name: str,
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    residual: np.ndarray | None,
    attrs: Mapping[str, object],
) -> np.ndarray:
    stride = int(list(attrs.get("stride", (1,)))[0])
    padding = int(list(attrs.get("padding", (0,)))[0])
    dilation = int(list(attrs.get("dilation", (1,)))[0])
    groups = int(attrs.get("groups", 1))
    if groups != 1:
        raise NotImplementedError(f"{op_name} CPU reference currently supports groups=1 only, got {groups}")
    batch, in_channels, in_width = [int(dim) for dim in x.shape]
    out_channels, weight_in_channels, kernel_w = [int(dim) for dim in weight.shape]
    if weight_in_channels != in_channels:
        raise ValueError(
            f"{op_name} CPU reference weight input channels must match activation channels for groups=1: "
            f"got activation C={in_channels}, weight I={weight_in_channels}"
        )
    if bias.shape != (out_channels,):
        raise ValueError(
            f"{op_name} CPU reference bias shape must be ({out_channels},), got {tuple(int(dim) for dim in bias.shape)}"
        )
    out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    if op_name in {"conv1d_bias_add", "conv1d_bias_add_relu"}:
        if residual is None:
            raise ValueError(f"{op_name} CPU reference requires a residual tensor")
        if tuple(int(dim) for dim in residual.shape) != (batch, out_channels, out_width):
            raise ValueError(
                f"{op_name} CPU reference residual shape must match the output shape "
                f"({batch}, {out_channels}, {out_width}), got {tuple(int(dim) for dim in residual.shape)}"
            )
    result = np.empty((batch, out_channels, out_width), dtype=np.float32)
    source = np.asarray(x, dtype=np.float32)
    filters = np.asarray(weight, dtype=np.float32)
    bias_values = np.asarray(bias, dtype=np.float32)
    residual_values = None if residual is None else np.asarray(residual, dtype=np.float32)
    for n in range(batch):
        for oc in range(out_channels):
            for ow in range(out_width):
                w_start = ow * stride - padding
                total = float(bias_values[oc])
                for ic in range(in_channels):
                    for kw in range(kernel_w):
                        iw = w_start + kw * dilation
                        if iw < 0 or iw >= in_width:
                            continue
                        total += float(source[n, ic, iw] * filters[oc, ic, kw])
                if op_name in {"conv1d_bias_add", "conv1d_bias_add_relu"}:
                    total += float(residual_values[n, oc, ow])
                if op_name in {"conv1d_bias_relu", "conv1d_bias_add_relu"}:
                    total = max(total, 0.0)
                result[n, oc, ow] = total
    return result


def _execute_conv2d_bias(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    attrs: Mapping[str, object],
) -> np.ndarray:
    return _execute_conv2d_bias_family("conv2d_bias", x, weight, bias, None, attrs)


def _execute_conv2d_bias_relu(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    attrs: Mapping[str, object],
) -> np.ndarray:
    return _execute_conv2d_bias_family("conv2d_bias_relu", x, weight, bias, None, attrs)


def _execute_conv2d_bias_add(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    residual: np.ndarray,
    attrs: Mapping[str, object],
) -> np.ndarray:
    return _execute_conv2d_bias_family("conv2d_bias_add", x, weight, bias, residual, attrs)


def _execute_conv2d_bias_add_relu(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    residual: np.ndarray,
    attrs: Mapping[str, object],
) -> np.ndarray:
    return _execute_conv2d_bias_family("conv2d_bias_add_relu", x, weight, bias, residual, attrs)


def _execute_conv2d_bias_family(
    op_name: str,
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    residual: np.ndarray | None,
    attrs: Mapping[str, object],
) -> np.ndarray:
    stride_h, stride_w = [int(item) for item in attrs.get("stride", (1, 1))]
    pad_h, pad_w = [int(item) for item in attrs.get("padding", (0, 0))]
    dilation_h, dilation_w = [int(item) for item in attrs.get("dilation", (1, 1))]
    groups = int(attrs.get("groups", 1))
    if groups != 1:
        raise NotImplementedError(f"{op_name} CPU reference currently supports groups=1 only, got {groups}")
    batch, in_channels, in_height, in_width = [int(dim) for dim in x.shape]
    out_channels, weight_in_channels, kernel_h, kernel_w = [int(dim) for dim in weight.shape]
    if weight_in_channels != in_channels:
        raise ValueError(
            f"{op_name} CPU reference weight input channels must match activation channels for groups=1: "
            f"got activation C={in_channels}, weight C={weight_in_channels}"
        )
    if bias.shape != (out_channels,):
        raise ValueError(
            f"{op_name} CPU reference bias shape must be ({out_channels},), got {tuple(int(dim) for dim in bias.shape)}"
        )
    out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    if op_name in {"conv2d_bias_add", "conv2d_bias_add_relu"}:
        if residual is None:
            raise ValueError(f"{op_name} CPU reference requires a residual tensor")
        if tuple(int(dim) for dim in residual.shape) != (batch, out_channels, out_height, out_width):
            raise ValueError(
                f"{op_name} CPU reference residual shape must match the output shape "
                f"({batch}, {out_channels}, {out_height}, {out_width}), got {tuple(int(dim) for dim in residual.shape)}"
            )
    result = np.empty((batch, out_channels, out_height, out_width), dtype=np.float32)
    source = np.asarray(x, dtype=np.float32)
    filters = np.asarray(weight, dtype=np.float32)
    bias_values = np.asarray(bias, dtype=np.float32)
    residual_values = None if residual is None else np.asarray(residual, dtype=np.float32)
    for n in range(batch):
        for oc in range(out_channels):
            for oh in range(out_height):
                h_start = oh * stride_h - pad_h
                for ow in range(out_width):
                    w_start = ow * stride_w - pad_w
                    total = float(bias_values[oc])
                    for ic in range(in_channels):
                        for kh in range(kernel_h):
                            ih = h_start + kh * dilation_h
                            if ih < 0 or ih >= in_height:
                                continue
                            for kw in range(kernel_w):
                                iw = w_start + kw * dilation_w
                                if iw < 0 or iw >= in_width:
                                    continue
                                total += float(source[n, ic, ih, iw] * filters[oc, ic, kh, kw])
                    if op_name in {"conv2d_bias_add", "conv2d_bias_add_relu"}:
                        total += float(residual_values[n, oc, oh, ow])
                    if op_name in {"conv2d_bias_relu", "conv2d_bias_add_relu"}:
                        total = max(total, 0.0)
                    result[n, oc, oh, ow] = total
    return result


def _execute_transposed_conv2d_family(
    op_name: str,
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    residual: np.ndarray | None,
    attrs: Mapping[str, object],
) -> np.ndarray:
    stride_h, stride_w = [int(item) for item in attrs.get("stride", (1, 1))]
    pad_h, pad_w = [int(item) for item in attrs.get("padding", (0, 0))]
    output_pad_h, output_pad_w = [int(item) for item in attrs.get("output_padding", (0, 0))]
    dilation_h, dilation_w = [int(item) for item in attrs.get("dilation", (1, 1))]
    groups = int(attrs.get("groups", 1))
    if groups != 1:
        raise NotImplementedError(f"{op_name} CPU reference currently supports groups=1 only, got {groups}")
    batch, in_channels, in_height, in_width = [int(dim) for dim in x.shape]
    weight_in_channels, out_channels, kernel_h, kernel_w = [int(dim) for dim in weight.shape]
    if weight_in_channels != in_channels:
        raise ValueError(
            f"{op_name} CPU reference weight input channels must match activation channels for groups=1: "
            f"got activation C={in_channels}, weight I={weight_in_channels}"
        )
    if output_pad_h >= stride_h or output_pad_w >= stride_w:
        raise ValueError(
            f"{op_name} CPU reference output_padding must be smaller than stride, "
            f"got output_padding=({output_pad_h}, {output_pad_w}) and stride=({stride_h}, {stride_w})"
        )
    if bias is not None and bias.shape != (out_channels,):
        raise ValueError(
            f"{op_name} CPU reference bias shape must be ({out_channels},), got {tuple(int(dim) for dim in bias.shape)}"
        )
    out_height = (in_height - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + output_pad_h + 1
    out_width = (in_width - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + output_pad_w + 1
    if op_name in {"transposed_conv2d_bias_add", "transposed_conv2d_bias_add_relu"}:
        if residual is None:
            raise ValueError(f"{op_name} CPU reference requires a residual tensor")
        if tuple(int(dim) for dim in residual.shape) != (batch, out_channels, out_height, out_width):
            raise ValueError(
                f"{op_name} CPU reference residual shape must match the output shape "
                f"({batch}, {out_channels}, {out_height}, {out_width}), got {tuple(int(dim) for dim in residual.shape)}"
            )
    result = np.empty((batch, out_channels, out_height, out_width), dtype=np.float32)
    source = np.asarray(x, dtype=np.float32)
    filters = np.asarray(weight, dtype=np.float32)
    bias_values = None if bias is None else np.asarray(bias, dtype=np.float32)
    residual_values = None if residual is None else np.asarray(residual, dtype=np.float32)
    for n in range(batch):
        for oc in range(out_channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    total = 0.0 if bias_values is None else float(bias_values[oc])
                    for ic in range(in_channels):
                        for kh in range(kernel_h):
                            h_numerator = oh + pad_h - kh * dilation_h
                            if h_numerator < 0 or h_numerator % stride_h != 0:
                                continue
                            ih = h_numerator // stride_h
                            if ih < 0 or ih >= in_height:
                                continue
                            for kw in range(kernel_w):
                                w_numerator = ow + pad_w - kw * dilation_w
                                if w_numerator < 0 or w_numerator % stride_w != 0:
                                    continue
                                iw = w_numerator // stride_w
                                if iw < 0 or iw >= in_width:
                                    continue
                                total += float(source[n, ic, ih, iw] * filters[ic, oc, kh, kw])
                    if op_name in {"transposed_conv2d_bias_add", "transposed_conv2d_bias_add_relu"}:
                        total += float(residual_values[n, oc, oh, ow])
                    if op_name in {"transposed_conv2d_bias_relu", "transposed_conv2d_bias_add_relu"}:
                        total = max(total, 0.0)
                    result[n, oc, oh, ow] = total
    return result


def _reference_array(value: object, dtype: str) -> np.ndarray:
    storage = array_to_storage(value, dtype)
    array = array_from_storage(storage, dtype)
    if dtype in {"float16", "bfloat16"}:
        array = np.asarray(array, dtype=np.float32)
    return array


def _store_reference(value: object, dtype: str) -> np.ndarray:
    if dtype == "float16":
        return array_from_storage(array_to_storage(value, dtype), dtype)
    if dtype == "bfloat16":
        return array_from_storage(array_to_storage(value, dtype), dtype)
    return np.asarray(value, dtype=dtype_numpy(dtype))


def _tensor_dtype(ir: Mapping[str, object], tensor_name: str) -> str:
    for tensor in ir["tensors"]:
        if tensor["name"] == tensor_name:
            return str(tensor["dtype"])
    raise KeyError(tensor_name)


def _permute_axes(node: Mapping[str, object], rank: int) -> list[int]:
    op_name = str(node["op"])
    if op_name in SPECIALIZED_PERMUTE_DIMS:
        attrs_dims = node.get("attrs", {}).get("dims")
        fixed_dims = list(SPECIALIZED_PERMUTE_DIMS[op_name])
        if attrs_dims is None:
            return fixed_dims
        normalized_dims = normalize_permute_dims(attrs_dims, rank)
        if tuple(normalized_dims) != SPECIALIZED_PERMUTE_DIMS[op_name]:
            raise ValueError(f"{op_name} uses fixed dims {fixed_dims}, got {list(normalized_dims)}")
        return normalized_dims
    return normalize_permute_dims(node.get("attrs", {}).get("dims"), rank)


def _tensor_shape(ir: Mapping[str, object], tensor_name: str) -> list[int]:
    for tensor in ir["tensors"]:
        if tensor["name"] == tensor_name:
            return [int(dim) for dim in tensor["shape"]]
    raise KeyError(tensor_name)


def _shape_dim_values(ir: Mapping[str, object], tensor_name: str, runtime_shape: Sequence[int]) -> dict[str, int]:
    for tensor in ir["tensors"]:
        if tensor["name"] != tensor_name:
            continue
        shape_spec = tensor.get("shape_spec", tensor["shape"])
        dim_values: dict[str, int] = {}
        for actual, dim_spec in zip(runtime_shape, shape_spec):
            if not isinstance(dim_spec, Mapping) or dim_spec.get("kind") != "dim":
                continue
            dim_values.setdefault(str(dim_spec["name"]), int(actual))
        return dim_values
    raise KeyError(tensor_name)
