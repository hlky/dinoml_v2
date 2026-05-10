from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from dinoml.ir import ModelSpec, array_from_storage, array_to_storage, dtype_numpy, write_json
from dinoml.kernels.bmm import BMM_OPS, bmm_op_spec
from dinoml.kernels.gemm import GEMM_OPS, gemm_op_spec
from dinoml.kernels.manifest import build_support_manifest
from dinoml.lowering.cpu import render_cpu_module, render_template
from dinoml.lowering.ops import collect_generated_sources
from dinoml.ops.elementwise import ELEMENTWISE_BY_NAME, FUSABLE_ELEMENTWISE_OPS


def execute_cpu(spec: ModelSpec, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
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
        elif node["op"] in {"reduce_sum", "reduce_max", "reduce_min", "reduce_mean", "var", "vector_norm"}:
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_reduction(node["op"], values[node["inputs"][0]], node.get("attrs", {})),
                output_dtype,
            )
        elif node["op"] == "argmax":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                _execute_argmax(values[node["inputs"][0]], node.get("attrs", {})),
                output_dtype,
            )
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
                _execute_randn(output_shape, int(attrs.get("seed", 0))),
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
        elif node["op"] == "permute":
            output_name = node["outputs"][0]
            output_dtype = _tensor_dtype(ir, output_name)
            values[output_name] = _store_reference(
                np.transpose(values[node["inputs"][0]], axes=tuple(node.get("attrs", {}).get("dims", ()))).copy(),
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
            values[tensor] = np.reshape(values[source], tuple(int(dim) for dim in view["shape"])).copy()
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
        result = x / (1.0 + np.exp(-1.702 * x))
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


def _execute_randn(shape: Sequence[int], seed: int) -> np.ndarray:
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


def _tensor_shape(ir: Mapping[str, object], tensor_name: str) -> list[int]:
    for tensor in ir["tensors"]:
        if tensor["name"] == tensor_name:
            return [int(dim) for dim in tensor["shape"]]
    raise KeyError(tensor_name)


@dataclass(frozen=True)
class CpuSupportLibs:
    runtime_lib: Path
    kernels_lib: Path
    runtime_include: Path
    common_include: Path
    kernels_include: Path


def build_cpu_module(
    ir: Mapping,
    *,
    target,
    artifact_dir: Path,
    generated_src_dir: Path,
    kernel_manifest: Mapping[str, object],
) -> None:
    support_libs = ensure_cpu_support_libs(kernel_manifest=kernel_manifest)
    artifact_lib_dir = artifact_dir / "lib"
    artifact_lib_dir.mkdir(parents=True, exist_ok=True)
    runtime_lib = artifact_lib_dir / support_libs.runtime_lib.name
    kernels_lib = artifact_lib_dir / support_libs.kernels_lib.name
    shutil.copy2(support_libs.runtime_lib, runtime_lib)
    shutil.copy2(support_libs.kernels_lib, kernels_lib)

    generated_src_dir.mkdir(parents=True, exist_ok=True)
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
    generated_sources = collect_generated_sources(
        "cpu",
        ir["nodes"],
        tensor_map,
        generated_src_dir=generated_src_dir,
    )
    (generated_src_dir / "module.cpp").write_text(
        render_cpu_module(ir, generated_kernels=generated_sources["kernels"]),
        encoding="utf-8",
    )
    (generated_src_dir / "CMakeLists.txt").write_text(
        render_template(
            "cpu_module_cmake.txt.j2",
            {
                "runtime_lib": str(runtime_lib),
                "kernels_lib": str(kernels_lib),
                "runtime_include": str(support_libs.runtime_include),
                "common_include": str(support_libs.common_include),
                "kernels_include": str(support_libs.kernels_include),
            },
        ),
        encoding="utf-8",
    )
    build_dir = generated_src_dir / "build"
    _run_cmake(
        [
            "cmake",
            "-S",
            str(generated_src_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={artifact_dir}",
            *([f"-DDINOML_ENABLE_OPENMP={os.environ['DINOML_ENABLE_OPENMP']}"] if "DINOML_ENABLE_OPENMP" in os.environ else []),
        ],
        cwd=artifact_dir,
    )
    _run_cmake(["cmake", "--build", str(build_dir), "--target", "module", "--parallel"], cwd=artifact_dir)


def ensure_cpu_support_libs(*, kernel_manifest: Mapping[str, object] | None = None) -> CpuSupportLibs:
    repo_root = Path(__file__).resolve().parents[3]
    cache_root = Path(os.environ.get("DINOML_CACHE_DIR", Path.home() / ".cache" / "dinoml_v2"))
    manifest_key = "full" if kernel_manifest is None else str(kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"]))[:16]
    support_root = cache_root / "support" / "cpu" / manifest_key
    build_dir = support_root / "build"
    lib_dir = support_root / "lib"
    runtime_lib = lib_dir / "libdinoml_runtime.so"
    kernels_lib = lib_dir / "libdinoml_cpu_kernels.so"
    lib_dir.mkdir(parents=True, exist_ok=True)
    configure_cmd = [
        "cmake",
        "-S",
        str(repo_root),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DDINOML_ENABLE_CUDA=OFF",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
    ]
    if "DINOML_ENABLE_OPENMP" in os.environ:
        configure_cmd.append(f"-DDINOML_ENABLE_OPENMP={os.environ['DINOML_ENABLE_OPENMP']}")
    _run_cmake(configure_cmd, cwd=repo_root)
    _run_cmake(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "dinoml_runtime",
            "dinoml_cpu_kernels",
            "--parallel",
        ],
        cwd=repo_root,
    )
    if not runtime_lib.exists() or not kernels_lib.exists():
        raise RuntimeError(f"Expected CPU support libraries under {lib_dir}, but they were not produced")
    write_json(
        lib_dir / "support_manifest.json",
        build_support_manifest(
            target={"name": "cpu", "arch": "native"},
            libraries={"runtime": runtime_lib.name, "kernels": kernels_lib.name},
            required_kernel_cache_key=None if kernel_manifest is None else str(kernel_manifest.get("support_cache_key", kernel_manifest["cache_key"])),
        ),
    )
    return CpuSupportLibs(
        runtime_lib=runtime_lib,
        kernels_lib=kernels_lib,
        runtime_include=repo_root / "runtime" / "include",
        common_include=repo_root / "kernels" / "common" / "include",
        kernels_include=repo_root / "kernels" / "cpu" / "include",
    )


def _run_cmake(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            "CMake command failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
