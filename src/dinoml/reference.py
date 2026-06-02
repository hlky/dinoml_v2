from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import numpy as np

from dinoml.ir import ModelSpec, array_from_storage, array_to_storage, dtype_numpy
from dinoml.kernels.bmm import BMM_OPS, bmm_op_spec
from dinoml.kernels.gemm import GEMM_OPS, gemm_op_spec
from dinoml.ops.elementwise import ELEMENTWISE_BY_NAME, FUSABLE_ELEMENTWISE_OPS
from dinoml.ops.collections import SPECIALIZED_PERMUTE_DIMS, normalize_permute_dims
from dinoml.ops.positional import normalize_get_1d_rotary_pos_embed_attrs


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
        elif node["op"] == "get_timestep_embedding":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_get_timestep_embedding(values[node["inputs"][0]], node.get("attrs", {})),
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
            slices = tuple(
                slice(int(start), int(start) + int(size))
                for start, size in zip(attrs.get("start_indices", ()), attrs.get("slice_sizes", ()))
            )
            values[output_name] = _store_reference(
                values[node["inputs"][0]][slices].copy(),
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
    scale = np.asarray(weight, dtype=np.float32)
    mean_square = np.mean(source * source, axis=-1, keepdims=True)
    return np.asarray(source * (1.0 / np.sqrt(mean_square + eps)) * scale, dtype=np.float32)


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
