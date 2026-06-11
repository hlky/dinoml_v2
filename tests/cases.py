from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import dinoml as dml
from dinoml.kernels.bmm import BMM_OPS, bmm_op_spec
from dinoml.kernels.gemm import GEMM_OPS, gemm_op_spec
from dinoml.ir import ModelSpec, array_from_storage, array_to_storage
from dinoml.ops.conv import CONV2D_BIAS_DTYPES, CONV2D_BIAS_FAMILY_OPS


@dataclass(frozen=True)
class GraphCase:
    name: str
    build_spec: Callable[[], ModelSpec]
    inputs: Callable[[], dict[str, np.ndarray]]
    expected_ops: frozenset[str]
    cpu: bool = True
    cuda: bool = True
    rocm: bool = True
    atol: float = 1e-5
    rtol: float = 1e-5
    reason: str = ""


def _roundtrip(value, dtype: str):
    if dtype in {"float16", "bfloat16"}:
        return array_from_storage(array_to_storage(value, dtype), dtype)
    if dtype == "bool":
        return np.asarray(value, dtype=np.bool_)
    if dtype == "int32":
        return np.asarray(value, dtype=np.int32)
    if dtype == "int64":
        return np.asarray(value, dtype=np.int64)
    return np.asarray(value, dtype=np.float32)


class ElementwiseModule(dml.Module):
    def forward(self, x, y, condition):
        positive = dml.ops.abs(x) + 0.25
        outputs = {
            "add": x + y,
            "sub": x - y,
            "mul": x * y,
            "div": x / (y + 2.0),
            "tanh": dml.ops.tanh(x),
            "cos": dml.ops.cos(x),
            "sin": dml.ops.sin(x),
            "sign": dml.ops.sign(x),
            "abs": dml.ops.abs(x),
            "log": dml.ops.log(positive),
            "log1p": dml.ops.log1p(positive),
            "exp": dml.ops.exp(x * 0.1),
            "sqrt": dml.ops.sqrt(positive),
            "max": dml.ops.max(x, y),
            "min": dml.ops.min(x, y),
            "sigmoid": dml.ops.sigmoid(x),
            "leaky_relu": dml.ops.leaky_relu(x, negative_slope=0.2),
            "hardtanh": dml.ops.hardtanh(x, min_value=-0.5, max_value=0.75),
            "clamp": dml.ops.clamp(x, min=-0.5, max=0.75),
            "relu": dml.ops.relu(x),
            "nan_to_num": dml.ops.nan_to_num(x, nan_replacement=0.0, posinf_replacement=0.0, neginf_replacement=0.0),
            "clamp_nan_to_num": dml.ops.clamp_nan_to_num(x, clamp_min=-1.0, clamp_max=1.0, nan_replacement=0.0),
            "silu": dml.ops.silu(x),
            "pow": dml.ops.pow(positive, y + 2.0),
            "gelu": dml.ops.gelu(x),
            "gelu_new": dml.ops.gelu_new(x),
            "fast_gelu": dml.ops.fast_gelu(x),
            "softplus": dml.ops.softplus(x),
            "elu": dml.ops.elu(x, alpha=1.25),
            "softsign": dml.ops.softsign(x),
            "floor_div": dml.ops.floor_div(x + 4.0, y + 2.0),
            "celu": dml.ops.celu(x, alpha=1.1),
            "floor": dml.ops.floor(x),
            "eq": dml.ops.eq(x, y),
            "ge": dml.ops.ge(x, y),
            "gt": dml.ops.gt(x, y),
            "le": dml.ops.le(x, y),
            "lt": dml.ops.lt(x, y),
            "ne": dml.ops.ne(x, y),
            "where": dml.ops.where(condition, x, y),
            "cast_bool": dml.ops.cast(condition, "float32"),
        }
        return {name: dml.ops.output(value, name) for name, value in outputs.items()}


def elementwise_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            ElementwiseModule(),
            inputs={
                "x": dml.TensorSpec([2, 3], "float32"),
                "y": dml.TensorSpec([2, 3], "float32"),
                "condition": dml.TensorSpec([2, 3], "bool"),
            },
            name="fresh_elementwise_contract",
        )

    def inputs():
        return {
            "x": np.array([[-1.25, -0.5, 0.0], [0.25, 1.0, 2.0]], dtype=np.float32),
            "y": np.array([[0.5, 1.0, 1.5], [2.0, 0.25, 0.75]], dtype=np.float32),
            "condition": np.array([[True, False, True], [False, True, False]], dtype=np.bool_),
        }

    return GraphCase(
        "elementwise",
        build_spec,
        inputs,
        frozenset(
            {
                "fused_elementwise",
                "add",
                "sub",
                "mul",
                "div",
                "tanh",
                "cos",
                "sin",
                "sign",
                "abs",
                "log",
                "log1p",
                "exp",
                "sqrt",
                "max",
                "min",
                "sigmoid",
                "leaky_relu",
                "hardtanh",
                "relu",
                "nan_to_num",
                "clamp_nan_to_num",
                "silu",
                "pow",
                "gelu",
                "fast_gelu",
                "softplus",
                "elu",
                "softsign",
                "floor_div",
                "celu",
                "floor",
                "eq",
                "ge",
                "gt",
                "le",
                "lt",
                "ne",
                "where",
                "cast",
            }
        ),
        atol=2e-4,
        rtol=2e-4,
    )


class ReductionModule(dml.Module):
    def forward(self, x):
        return {
            "softmax": dml.ops.output(dml.ops.softmax(x, dim=-1), "softmax"),
            "reduce_sum": dml.ops.output(dml.ops.reduce_sum(x, dim=-1), "reduce_sum"),
            "reduce_max": dml.ops.output(dml.ops.reduce_max(x, dim=-1), "reduce_max"),
            "reduce_min": dml.ops.output(dml.ops.reduce_min(x, dim=-1), "reduce_min"),
            "reduce_mean": dml.ops.output(dml.ops.reduce_mean(x, dim=-1), "reduce_mean"),
            "var": dml.ops.output(dml.ops.var(x, dim=-1, unbiased=False), "var"),
            "vector_norm": dml.ops.output(dml.ops.vector_norm(x, dim=-1), "vector_norm"),
        }


def reduction_case() -> GraphCase:
    def build_spec():
        return dml.trace(ReductionModule(), inputs={"x": dml.TensorSpec([2, 3, 4], "float32")}, name="fresh_reductions")

    def inputs():
        return {"x": np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(2, 3, 4)}

    return GraphCase(
        "reductions",
        build_spec,
        inputs,
        frozenset({"softmax", "reduce_sum", "reduce_max", "reduce_min", "reduce_mean", "var", "vector_norm"}),
        atol=1e-4,
        rtol=1e-4,
    )


class CreationModule(dml.Module):
    def forward(self):
        return {
            "full": dml.ops.output(dml.ops.full([2, 3], 1.25, dtype="float32"), "full"),
            "arange": dml.ops.output(dml.ops.arange(1, 7, 2, dtype="float32"), "arange"),
            "randn": dml.ops.output(dml.ops.randn([2, 3], dtype="float32", seed=17), "randn"),
        }


def creation_case() -> GraphCase:
    return GraphCase("creation", lambda: dml.trace(CreationModule(), inputs={}, name="fresh_creation"), lambda: {}, frozenset({"full", "arange", "randn"}))


def _dtype_elementwise_outputs(prefix: str, x, y, condition, dtype: str):
    positive = dml.ops.abs(x) + 0.25
    return {
        f"{prefix}_add": x + y,
        f"{prefix}_sub": x - y,
        f"{prefix}_mul": x * y,
        f"{prefix}_div": x / (y + 2.0),
        f"{prefix}_tanh": dml.ops.tanh(x),
        f"{prefix}_cos": dml.ops.cos(x),
        f"{prefix}_sin": dml.ops.sin(x),
        f"{prefix}_sign": dml.ops.sign(x),
        f"{prefix}_abs": dml.ops.abs(x),
        f"{prefix}_log": dml.ops.log(positive),
        f"{prefix}_log1p": dml.ops.log1p(positive),
        f"{prefix}_exp": dml.ops.exp(x * 0.1),
        f"{prefix}_sqrt": dml.ops.sqrt(positive),
        f"{prefix}_max": dml.ops.max(x, y),
        f"{prefix}_min": dml.ops.min(x, y),
        f"{prefix}_sigmoid": dml.ops.sigmoid(x),
        f"{prefix}_leaky_relu": dml.ops.leaky_relu(x, negative_slope=0.2),
        f"{prefix}_hardtanh": dml.ops.hardtanh(x, min_value=-0.5, max_value=0.75),
        f"{prefix}_clamp": dml.ops.clamp(x, min=-0.5, max=0.75),
        f"{prefix}_relu": dml.ops.relu(x),
        f"{prefix}_nan_to_num": dml.ops.nan_to_num(x, nan_replacement=0.0, posinf_replacement=0.0, neginf_replacement=0.0),
        f"{prefix}_clamp_nan_to_num": dml.ops.clamp_nan_to_num(x, clamp_min=-1.0, clamp_max=1.0, nan_replacement=0.0),
        f"{prefix}_silu": dml.ops.silu(x),
        f"{prefix}_pow": dml.ops.pow(positive, y + 2.0),
        f"{prefix}_gelu": dml.ops.gelu(x),
        f"{prefix}_gelu_new": dml.ops.gelu_new(x),
        f"{prefix}_fast_gelu": dml.ops.fast_gelu(x),
        f"{prefix}_softplus": dml.ops.softplus(x),
        f"{prefix}_elu": dml.ops.elu(x, alpha=1.25),
        f"{prefix}_softsign": dml.ops.softsign(x),
        f"{prefix}_floor_div": dml.ops.floor_div(x + 4.0, y + 2.0),
        f"{prefix}_celu": dml.ops.celu(x, alpha=1.1),
        f"{prefix}_floor": dml.ops.floor(x),
        f"{prefix}_eq": dml.ops.eq(x, y),
        f"{prefix}_ge": dml.ops.ge(x, y),
        f"{prefix}_gt": dml.ops.gt(x, y),
        f"{prefix}_le": dml.ops.le(x, y),
        f"{prefix}_lt": dml.ops.lt(x, y),
        f"{prefix}_ne": dml.ops.ne(x, y),
        f"{prefix}_where": dml.ops.where(condition, x, y),
        f"{prefix}_cast": dml.ops.cast(condition, dtype),
    }


class DtypeElementwiseModule(dml.Module):
    def forward(self, x16, y16, xbf, ybf, condition):
        outputs = {
            **_dtype_elementwise_outputs("float16", x16, y16, condition, "float16"),
            **_dtype_elementwise_outputs("bfloat16", xbf, ybf, condition, "bfloat16"),
        }
        return {name: dml.ops.output(value, name) for name, value in outputs.items()}


def dtype_elementwise_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            DtypeElementwiseModule(),
            inputs={
                "x16": dml.TensorSpec([2, 3], "float16"),
                "y16": dml.TensorSpec([2, 3], "float16"),
                "xbf": dml.TensorSpec([2, 3], "bfloat16"),
                "ybf": dml.TensorSpec([2, 3], "bfloat16"),
                "condition": dml.TensorSpec([2, 3], "bool"),
            },
            name="fresh_dtype_elementwise",
        )

    def inputs():
        return {
            "x16": _roundtrip([[-1.25, -0.5, 0.0], [0.25, 1.0, 2.0]], "float16"),
            "y16": _roundtrip([[0.5, 1.0, 1.5], [2.0, 0.25, 0.75]], "float16"),
            "xbf": _roundtrip([[-1.25, -0.5, 0.0], [0.25, 1.0, 2.0]], "bfloat16"),
            "ybf": _roundtrip([[0.5, 1.0, 1.5], [2.0, 0.25, 0.75]], "bfloat16"),
            "condition": np.array([[True, False, True], [False, True, False]], dtype=np.bool_),
        }

    return GraphCase(
        "dtype_elementwise",
        build_spec,
        inputs,
        frozenset(
            {
                "fused_elementwise",
                "add",
                "sub",
                "mul",
                "div",
                "tanh",
                "cos",
                "sin",
                "sign",
                "abs",
                "log",
                "log1p",
                "exp",
                "sqrt",
                "max",
                "min",
                "sigmoid",
                "leaky_relu",
                "hardtanh",
                "relu",
                "nan_to_num",
                "clamp_nan_to_num",
                "silu",
                "pow",
                "gelu",
                "fast_gelu",
                "softplus",
                "elu",
                "softsign",
                "floor_div",
                "celu",
                "floor",
                "eq",
                "ge",
                "gt",
                "le",
                "lt",
                "ne",
                "where",
                "cast",
            }
        ),
        atol=2e-2,
        rtol=2e-2,
    )


class DtypeGeneratedModule(dml.Module):
    def forward(self, x16, xbf, xbool, xint64, gather_index32, batch_index32, embedding_table, embedding_index32):
        topk_bool_values, topk_bool_indices = dml.ops.topk(xbool, 2, dim=-1)
        return {
            "full_float16": dml.ops.output(dml.ops.full([2, 3], 1.25, dtype="float16"), "full_float16"),
            "full_bool": dml.ops.output(dml.ops.full([2, 3], True, dtype="bool"), "full_bool"),
            "full_bfloat16": dml.ops.output(dml.ops.full([2, 3], 1.25, dtype="bfloat16"), "full_bfloat16"),
            "arange_float16": dml.ops.output(dml.ops.arange(0, 6, 1, dtype="float16"), "arange_float16"),
            "arange_bfloat16": dml.ops.output(dml.ops.arange(0, 6, 1, dtype="bfloat16"), "arange_bfloat16"),
            "randn_float16": dml.ops.output(dml.ops.randn([2, 3], dtype="float16", seed=7), "randn_float16"),
            "randn_bfloat16": dml.ops.output(dml.ops.randn([2, 3], dtype="bfloat16", seed=7), "randn_bfloat16"),
            "argmax_bool": dml.ops.output(dml.ops.argmax(xbool, dim=-1), "argmax_bool"),
            "argmax_int64": dml.ops.output(dml.ops.argmax(xint64, dim=-1), "argmax_int64"),
            "topk_bool_values": dml.ops.output(topk_bool_values, "topk_bool_values"),
            "topk_bool_indices": dml.ops.output(topk_bool_indices, "topk_bool_indices"),
            "gather_float16_int32": dml.ops.output(dml.ops.gather(x16, 1, gather_index32), "gather_float16_int32"),
            "batch_gather_bfloat16_int32": dml.ops.output(dml.ops.batch_gather(xbf, batch_index32), "batch_gather_bfloat16_int32"),
            "embedding_bfloat16_int32": dml.ops.output(dml.ops.embedding(embedding_table, embedding_index32), "embedding_bfloat16_int32"),
            "where_bfloat16": dml.ops.output(dml.ops.where(xbool, xbf, dml.ops.full([2, 3], -0.5, dtype="bfloat16")), "where_bfloat16"),
            "reduce_sum_float16": dml.ops.output(dml.ops.reduce_sum(x16, dim=-1), "reduce_sum_float16"),
            "softmax_bfloat16": dml.ops.output(dml.ops.softmax(xbf, dim=-1), "softmax_bfloat16"),
        }


def dtype_generated_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            DtypeGeneratedModule(),
            inputs={
                "x16": dml.TensorSpec([2, 3], "float16"),
                "xbf": dml.TensorSpec([2, 3], "bfloat16"),
                "xbool": dml.TensorSpec([2, 3], "bool"),
                "xint64": dml.TensorSpec([2, 3], "int64"),
                "gather_index32": dml.TensorSpec([2, 2], "int32"),
                "batch_index32": dml.TensorSpec([2, 2], "int32"),
                "embedding_table": dml.TensorSpec([5, 2], "bfloat16"),
                "embedding_index32": dml.TensorSpec([2, 2], "int32"),
            },
            name="fresh_dtype_generated",
        )

    def inputs():
        return {
            "x16": _roundtrip([[-1.5, 0.25, 2.0], [3.5, -0.75, 1.25]], "float16"),
            "xbf": _roundtrip([[-0.5, 0.75, 1.5], [2.25, -1.0, 0.5]], "bfloat16"),
            "xbool": np.array([[True, False, True], [False, True, False]], dtype=np.bool_),
            "xint64": np.array([[5, -1, 5], [0, 10, 9]], dtype=np.int64),
            "gather_index32": np.array([[2, 0], [1, 2]], dtype=np.int32),
            "batch_index32": np.array([[0, 2], [1, 0]], dtype=np.int32),
            "embedding_table": _roundtrip(np.linspace(-1.0, 1.25, num=10, dtype=np.float32).reshape(5, 2), "bfloat16"),
            "embedding_index32": np.array([[0, 3], [4, 1]], dtype=np.int32),
        }

    return GraphCase(
        "dtype_generated",
        build_spec,
        inputs,
        frozenset(
            {
                "full",
                "arange",
                "randn",
                "argmax",
                "topk_values",
                "topk_indices",
                "gather",
                "batch_gather",
                "embedding",
                "where",
                "reduce_sum",
                "softmax",
            }
        ),
        atol=2e-2,
        rtol=2e-2,
    )


class DtypeReductionModule(dml.Module):
    def forward(self, x16, xbf):
        return {
            "softmax_float16": dml.ops.output(dml.ops.softmax(x16, dim=-1), "softmax_float16"),
            "reduce_sum_float16": dml.ops.output(dml.ops.reduce_sum(x16, dim=-1), "reduce_sum_float16"),
            "reduce_max_float16": dml.ops.output(dml.ops.reduce_max(x16, dim=-1), "reduce_max_float16"),
            "reduce_min_float16": dml.ops.output(dml.ops.reduce_min(x16, dim=-1), "reduce_min_float16"),
            "reduce_mean_float16": dml.ops.output(dml.ops.reduce_mean(x16, dim=-1), "reduce_mean_float16"),
            "softmax_bfloat16": dml.ops.output(dml.ops.softmax(xbf, dim=-1), "softmax_bfloat16"),
            "reduce_sum_bfloat16": dml.ops.output(dml.ops.reduce_sum(xbf, dim=-1), "reduce_sum_bfloat16"),
            "reduce_max_bfloat16": dml.ops.output(dml.ops.reduce_max(xbf, dim=-1), "reduce_max_bfloat16"),
            "reduce_min_bfloat16": dml.ops.output(dml.ops.reduce_min(xbf, dim=-1), "reduce_min_bfloat16"),
            "reduce_mean_bfloat16": dml.ops.output(dml.ops.reduce_mean(xbf, dim=-1), "reduce_mean_bfloat16"),
        }


def dtype_reduction_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            DtypeReductionModule(),
            inputs={
                "x16": dml.TensorSpec([2, 3, 4], "float16"),
                "xbf": dml.TensorSpec([2, 3, 4], "bfloat16"),
            },
            name="fresh_dtype_reductions",
        )

    def inputs():
        values = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(2, 3, 4)
        return {"x16": _roundtrip(values, "float16"), "xbf": _roundtrip(values, "bfloat16")}

    return GraphCase(
        "dtype_reductions",
        build_spec,
        inputs,
        frozenset({"softmax", "reduce_sum", "reduce_max", "reduce_min", "reduce_mean"}),
        atol=2e-2,
        rtol=2e-2,
    )


class ShapeViewModule(dml.Module):
    def forward(self, x, x1):
        return {
            "reshape": dml.ops.output(dml.ops.reshape(x, [3, 2]), "reshape"),
            "flatten": dml.ops.output(dml.ops.flatten(x, 0, 1), "flatten"),
            "unsqueeze": dml.ops.output(dml.ops.unsqueeze(x, 0), "unsqueeze"),
            "squeeze": dml.ops.output(dml.ops.squeeze(x1, 0), "squeeze"),
            "identity": dml.ops.output(dml.ops.identity(x), "identity"),
            "transpose": dml.ops.output(dml.ops.transpose(x, 0, 1), "transpose"),
        }


def shape_view_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            ShapeViewModule(),
            inputs={"x": dml.TensorSpec([2, 3], "float32"), "x1": dml.TensorSpec([1, 2, 3], "float32")},
            name="fresh_shape_views",
        )

    def inputs():
        return {"x": np.arange(6, dtype=np.float32).reshape(2, 3), "x1": np.arange(6, dtype=np.float32).reshape(1, 2, 3)}

    return GraphCase("shape_views", build_spec, inputs, frozenset({"identity", "reshape", "flatten", "unsqueeze", "squeeze", "transpose", "permute"}))


class CollectionModule(dml.Module):
    def forward(self, x, y, z, x4, index, batch_index, update):
        concat = dml.ops.concatenate([x, y], dim=1)
        stacked = dml.ops.stack([x, y], dim=0)
        permuted = dml.ops.permute(stacked, [1, 0, 2, 3])
        return {
            "expand": dml.ops.output(dml.ops.expand(z, [2, 3]), "expand"),
            "concatenate": dml.ops.output(concat, "concatenate"),
            "stack": dml.ops.output(stacked, "stack"),
            "flip": dml.ops.output(dml.ops.flip(x, dims=(-1,)), "flip"),
            "repeat_interleave": dml.ops.output(dml.ops.repeat_interleave(x, 2, dim=1), "repeat_interleave"),
            "permute": dml.ops.output(permuted, "permute"),
            "permute021": dml.ops.output(dml.ops.permute021(x), "permute021"),
            "permute102": dml.ops.output(dml.ops.permute102(x), "permute102"),
            "permute210": dml.ops.output(dml.ops.permute210(x), "permute210"),
            "permute0213": dml.ops.output(dml.ops.permute0213(x4), "permute0213"),
            "dynamic_slice": dml.ops.output(dml.ops.dynamic_slice(concat, [0, 1, 0], [2, 2, 2]), "dynamic_slice"),
            "index_select": dml.ops.output(dml.ops.index_select(x, dim=1, indices=[2, 0]), "index_select"),
            "gather": dml.ops.output(dml.ops.gather(x, 1, index), "gather"),
            "batch_gather": dml.ops.output(dml.ops.batch_gather(x, batch_index), "batch_gather"),
            "slice_scatter": dml.ops.output(dml.ops.slice_scatter(x, update, [0, 1, 0]), "slice_scatter"),
            "pad": dml.ops.output(dml.ops.pad(x, [1, 0, 0, 1], value=-1.0), "pad"),
        }


def collection_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            CollectionModule(),
            inputs={
                "x": dml.TensorSpec([2, 3, 2], "float32"),
                "y": dml.TensorSpec([2, 3, 2], "float32"),
                "z": dml.TensorSpec([1, 3], "float32"),
                "x4": dml.TensorSpec([1, 2, 3, 2], "float32"),
                "index": dml.TensorSpec([2, 2, 2], "int64"),
                "batch_index": dml.TensorSpec([2, 2], "int64"),
                "update": dml.TensorSpec([2, 1, 2], "float32"),
            },
            name="fresh_collections",
        )

    def inputs():
        return {
            "x": np.arange(12, dtype=np.float32).reshape(2, 3, 2),
            "y": (np.arange(12, dtype=np.float32).reshape(2, 3, 2) + 20.0),
            "z": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "x4": np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2),
            "index": np.array([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=np.int64),
            "batch_index": np.array([[0, 2], [1, 0]], dtype=np.int64),
            "update": np.array([[[100.0, 101.0]], [[102.0, 103.0]]], dtype=np.float32),
        }

    return GraphCase(
        "collections",
        build_spec,
        inputs,
        frozenset(
            {
                "expand",
                "concatenate",
                "stack",
                "flip",
                "repeat_interleave",
                "permute",
                "permute021",
                "permute102",
                "permute210",
                "permute0213",
                "dynamic_slice",
                "index_select",
                "gather",
                "batch_gather",
                "slice_scatter",
                "pad",
            }
        ),
    )


def _dtype_collection_outputs(prefix: str, x, y, z, x4, index, batch_index, update):
    concat = dml.ops.concatenate([x, y], dim=1)
    stacked = dml.ops.stack([x, y], dim=0)
    permuted = dml.ops.permute(stacked, [1, 0, 2, 3])
    return {
        f"{prefix}_expand": dml.ops.expand(z, [2, 3]),
        f"{prefix}_concatenate": concat,
        f"{prefix}_stack": stacked,
        f"{prefix}_flip": dml.ops.flip(x, dims=(-1,)),
        f"{prefix}_repeat_interleave": dml.ops.repeat_interleave(x, 2, dim=1),
        f"{prefix}_permute": permuted,
        f"{prefix}_permute021": dml.ops.permute021(x),
        f"{prefix}_permute102": dml.ops.permute102(x),
        f"{prefix}_permute210": dml.ops.permute210(x),
        f"{prefix}_permute0213": dml.ops.permute0213(x4),
        f"{prefix}_dynamic_slice": dml.ops.dynamic_slice(concat, [0, 1, 0], [2, 2, 2]),
        f"{prefix}_index_select": dml.ops.index_select(x, dim=1, indices=[2, 0]),
        f"{prefix}_gather": dml.ops.gather(x, 1, index),
        f"{prefix}_batch_gather": dml.ops.batch_gather(x, batch_index),
        f"{prefix}_slice_scatter": dml.ops.slice_scatter(x, update, [0, 1, 0]),
        f"{prefix}_pad": dml.ops.pad(x, [1, 0, 0, 1], value=-1.0),
    }


class DtypeCollectionModule(dml.Module):
    def forward(
        self,
        x16,
        y16,
        z16,
        x4_16,
        update16,
        xbf,
        ybf,
        zbf,
        x4_bf,
        updatebf,
        index32,
        batch_index32,
    ):
        outputs = {
            **_dtype_collection_outputs("float16", x16, y16, z16, x4_16, index32, batch_index32, update16),
            **_dtype_collection_outputs("bfloat16", xbf, ybf, zbf, x4_bf, index32, batch_index32, updatebf),
        }
        return {name: dml.ops.output(value, name) for name, value in outputs.items()}


def dtype_collection_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            DtypeCollectionModule(),
            inputs={
                "x16": dml.TensorSpec([2, 3, 2], "float16"),
                "y16": dml.TensorSpec([2, 3, 2], "float16"),
                "z16": dml.TensorSpec([1, 3], "float16"),
                "x4_16": dml.TensorSpec([1, 2, 3, 2], "float16"),
                "update16": dml.TensorSpec([2, 1, 2], "float16"),
                "xbf": dml.TensorSpec([2, 3, 2], "bfloat16"),
                "ybf": dml.TensorSpec([2, 3, 2], "bfloat16"),
                "zbf": dml.TensorSpec([1, 3], "bfloat16"),
                "x4_bf": dml.TensorSpec([1, 2, 3, 2], "bfloat16"),
                "updatebf": dml.TensorSpec([2, 1, 2], "bfloat16"),
                "index32": dml.TensorSpec([2, 2, 2], "int32"),
                "batch_index32": dml.TensorSpec([2, 2], "int32"),
            },
            name="fresh_dtype_collections",
        )

    def inputs():
        x = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        y = x + 20.0
        z = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        x4 = np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2)
        update = np.array([[[100.0, 101.0]], [[102.0, 103.0]]], dtype=np.float32)
        return {
            "x16": _roundtrip(x, "float16"),
            "y16": _roundtrip(y, "float16"),
            "z16": _roundtrip(z, "float16"),
            "x4_16": _roundtrip(x4, "float16"),
            "update16": _roundtrip(update, "float16"),
            "xbf": _roundtrip(x, "bfloat16"),
            "ybf": _roundtrip(y, "bfloat16"),
            "zbf": _roundtrip(z, "bfloat16"),
            "x4_bf": _roundtrip(x4, "bfloat16"),
            "updatebf": _roundtrip(update, "bfloat16"),
            "index32": np.array([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=np.int32),
            "batch_index32": np.array([[0, 2], [1, 0]], dtype=np.int32),
        }

    return GraphCase(
        "dtype_collections",
        build_spec,
        inputs,
        frozenset(
            {
                "expand",
                "concatenate",
                "stack",
                "flip",
                "repeat_interleave",
                "permute",
                "permute021",
                "permute102",
                "permute210",
                "permute0213",
                "dynamic_slice",
                "index_select",
                "gather",
                "batch_gather",
                "slice_scatter",
                "pad",
            }
        ),
        atol=2e-2,
        rtol=2e-2,
    )


class PoolingModule(dml.Module):
    def forward(self, x1, x2):
        return {
            "avg_pool1d": dml.ops.output(dml.ops.avg_pool1d(x1, kernel_size=2, stride=1, padding=1), "avg_pool1d"),
            "avg_pool2d": dml.ops.output(dml.ops.avg_pool2d(x2, kernel_size=(2, 2), stride=1, padding=(1, 0)), "avg_pool2d"),
            "max_pool2d": dml.ops.output(dml.ops.max_pool2d(x2, kernel_size=(2, 2), stride=1, padding=(1, 0)), "max_pool2d"),
        }


def pooling_case() -> GraphCase:
    def build_spec():
        return dml.trace(PoolingModule(), inputs={"x1": dml.TensorSpec([1, 2, 5], "float32"), "x2": dml.TensorSpec([1, 2, 4, 5], "float32")}, name="fresh_pooling")

    def inputs():
        return {
            "x1": np.linspace(-1.0, 2.0, num=10, dtype=np.float32).reshape(1, 2, 5),
            "x2": np.linspace(-2.0, 3.0, num=40, dtype=np.float32).reshape(1, 2, 4, 5),
        }

    return GraphCase("pooling", build_spec, inputs, frozenset({"avg_pool1d", "avg_pool2d", "max_pool2d"}))


class DtypePoolingModule(dml.Module):
    def forward(self, x1_16, x2_16, x1_bf, x2_bf):
        return {
            "avg_pool1d_float16": dml.ops.output(dml.ops.avg_pool1d(x1_16, kernel_size=2, stride=1, padding=1), "avg_pool1d_float16"),
            "avg_pool2d_float16": dml.ops.output(dml.ops.avg_pool2d(x2_16, kernel_size=(2, 2), stride=1, padding=(1, 0)), "avg_pool2d_float16"),
            "max_pool2d_float16": dml.ops.output(dml.ops.max_pool2d(x2_16, kernel_size=(2, 2), stride=1, padding=(1, 0)), "max_pool2d_float16"),
            "avg_pool1d_bfloat16": dml.ops.output(dml.ops.avg_pool1d(x1_bf, kernel_size=2, stride=1, padding=1), "avg_pool1d_bfloat16"),
            "avg_pool2d_bfloat16": dml.ops.output(dml.ops.avg_pool2d(x2_bf, kernel_size=(2, 2), stride=1, padding=(1, 0)), "avg_pool2d_bfloat16"),
            "max_pool2d_bfloat16": dml.ops.output(dml.ops.max_pool2d(x2_bf, kernel_size=(2, 2), stride=1, padding=(1, 0)), "max_pool2d_bfloat16"),
        }


def dtype_pooling_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            DtypePoolingModule(),
            inputs={
                "x1_16": dml.TensorSpec([1, 2, 5], "float16"),
                "x2_16": dml.TensorSpec([1, 2, 4, 5], "float16"),
                "x1_bf": dml.TensorSpec([1, 2, 5], "bfloat16"),
                "x2_bf": dml.TensorSpec([1, 2, 4, 5], "bfloat16"),
            },
            name="fresh_dtype_pooling",
        )

    def inputs():
        x1 = np.linspace(-1.0, 2.0, num=10, dtype=np.float32).reshape(1, 2, 5)
        x2 = np.linspace(-2.0, 3.0, num=40, dtype=np.float32).reshape(1, 2, 4, 5)
        return {
            "x1_16": _roundtrip(x1, "float16"),
            "x2_16": _roundtrip(x2, "float16"),
            "x1_bf": _roundtrip(x1, "bfloat16"),
            "x2_bf": _roundtrip(x2, "bfloat16"),
        }

    return GraphCase(
        "dtype_pooling",
        build_spec,
        inputs,
        frozenset({"avg_pool1d", "avg_pool2d", "max_pool2d"}),
        atol=2e-2,
        rtol=2e-2,
    )


class SelectionModule(dml.Module):
    def forward(self, x):
        values, indices = dml.ops.topk(x, 2, dim=-1)
        return {
            "argmax": dml.ops.output(dml.ops.argmax(x, dim=-1), "argmax"),
            "topk_values": dml.ops.output(values, "topk_values"),
            "topk_indices": dml.ops.output(indices, "topk_indices"),
        }


def selection_case() -> GraphCase:
    def build_spec():
        return dml.trace(SelectionModule(), inputs={"x": dml.TensorSpec([2, 3, 5], "float32")}, name="fresh_selection")

    def inputs():
        return {"x": np.array([[[1, 5, 5, -1, 2], [0, -2, 3, 3, 1], [7, 1, 7, 0, 8]], [[4, 4, 2, 1, 0], [-1, -1, -1, -2, 9], [0, 9, 8, 9, 7]]], dtype=np.float32)}

    return GraphCase("selection", build_spec, inputs, frozenset({"argmax", "topk_values", "topk_indices"}))


class DtypeSelectionModule(dml.Module):
    def forward(self, x16, xbf):
        values16, indices16 = dml.ops.topk(x16, 2, dim=-1)
        valuesbf, indicesbf = dml.ops.topk(xbf, 2, dim=-1)
        return {
            "argmax_float16": dml.ops.output(dml.ops.argmax(x16, dim=-1), "argmax_float16"),
            "topk_values_float16": dml.ops.output(values16, "topk_values_float16"),
            "topk_indices_float16": dml.ops.output(indices16, "topk_indices_float16"),
            "argmax_bfloat16": dml.ops.output(dml.ops.argmax(xbf, dim=-1), "argmax_bfloat16"),
            "topk_values_bfloat16": dml.ops.output(valuesbf, "topk_values_bfloat16"),
            "topk_indices_bfloat16": dml.ops.output(indicesbf, "topk_indices_bfloat16"),
        }


def dtype_selection_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            DtypeSelectionModule(),
            inputs={
                "x16": dml.TensorSpec([2, 3, 5], "float16"),
                "xbf": dml.TensorSpec([2, 3, 5], "bfloat16"),
            },
            name="fresh_dtype_selection",
        )

    def inputs():
        values = np.array(
            [
                [[1, 5, 5, -1, 2], [0, -2, 3, 3, 1], [7, 1, 7, 0, 8]],
                [[4, 4, 2, 1, 0], [-1, -1, -1, -2, 9], [0, 9, 8, 9, 7]],
            ],
            dtype=np.float32,
        )
        return {"x16": _roundtrip(values, "float16"), "xbf": _roundtrip(values, "bfloat16")}

    return GraphCase(
        "dtype_selection",
        build_spec,
        inputs,
        frozenset({"argmax", "topk_values", "topk_indices"}),
        atol=2e-2,
        rtol=2e-2,
    )


class NormalizationModule(dml.Module):
    def forward(self, x, weight, bias):
        return {
            "layer_norm": dml.ops.output(dml.ops.layer_norm(x, weight, bias, eps=1e-5), "layer_norm"),
            "t5_layer_norm": dml.ops.output(dml.ops.t5_layer_norm(x, weight, eps=1e-6), "t5_layer_norm"),
            "rms_norm": dml.ops.output(dml.ops.rms_norm(x, weight, eps=1e-6), "rms_norm"),
        }


def normalization_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            NormalizationModule(),
            inputs={
                "x": dml.TensorSpec([2, 3, 4], "float32"),
                "weight": dml.TensorSpec([4], "float32"),
                "bias": dml.TensorSpec([4], "float32"),
            },
            name="fresh_normalization",
        )

    def inputs():
        return {
            "x": np.linspace(-2.0, 2.0, num=24, dtype=np.float32).reshape(2, 3, 4),
            "weight": np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
            "bias": np.array([-0.25, 0.0, 0.25, 0.5], dtype=np.float32),
        }

    return GraphCase("normalization", build_spec, inputs, frozenset({"layer_norm", "t5_layer_norm"}), atol=1e-4, rtol=1e-4)


class DtypeNormalizationModule(dml.Module):
    def forward(self, x16, weight16, bias16, xbf, weightbf, biasbf):
        return {
            "layer_norm_float16": dml.ops.output(dml.ops.layer_norm(x16, weight16, bias16, eps=1e-5), "layer_norm_float16"),
            "t5_layer_norm_float16": dml.ops.output(dml.ops.t5_layer_norm(x16, weight16, eps=1e-6), "t5_layer_norm_float16"),
            "rms_norm_float16": dml.ops.output(dml.ops.rms_norm(x16, weight16, eps=1e-6), "rms_norm_float16"),
            "layer_norm_bfloat16": dml.ops.output(dml.ops.layer_norm(xbf, weightbf, biasbf, eps=1e-5), "layer_norm_bfloat16"),
            "t5_layer_norm_bfloat16": dml.ops.output(dml.ops.t5_layer_norm(xbf, weightbf, eps=1e-6), "t5_layer_norm_bfloat16"),
            "rms_norm_bfloat16": dml.ops.output(dml.ops.rms_norm(xbf, weightbf, eps=1e-6), "rms_norm_bfloat16"),
        }


def dtype_normalization_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            DtypeNormalizationModule(),
            inputs={
                "x16": dml.TensorSpec([2, 3, 4], "float16"),
                "weight16": dml.TensorSpec([4], "float16"),
                "bias16": dml.TensorSpec([4], "float16"),
                "xbf": dml.TensorSpec([2, 3, 4], "bfloat16"),
                "weightbf": dml.TensorSpec([4], "bfloat16"),
                "biasbf": dml.TensorSpec([4], "bfloat16"),
            },
            name="fresh_dtype_normalization",
        )

    def inputs():
        x = np.linspace(-2.0, 2.0, num=24, dtype=np.float32).reshape(2, 3, 4)
        weight = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        bias = np.array([-0.25, 0.0, 0.25, 0.5], dtype=np.float32)
        return {
            "x16": _roundtrip(x, "float16"),
            "weight16": _roundtrip(weight, "float16"),
            "bias16": _roundtrip(bias, "float16"),
            "xbf": _roundtrip(x, "bfloat16"),
            "weightbf": _roundtrip(weight, "bfloat16"),
            "biasbf": _roundtrip(bias, "bfloat16"),
        }

    return GraphCase(
        "dtype_normalization",
        build_spec,
        inputs,
        frozenset({"layer_norm", "t5_layer_norm"}),
        atol=3e-2,
        rtol=3e-2,
    )


class GroupNormModule(dml.Module):
    def forward(self, x, weight, bias):
        return {
            "group_norm": dml.ops.output(dml.ops.group_norm(x, 4, weight, bias, eps=1e-5), "group_norm"),
            "group_norm_swish": dml.ops.output(dml.ops.group_norm_swish(x, 4, weight, bias, eps=1e-5), "group_norm_swish"),
        }


def group_norm_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            GroupNormModule(),
            inputs={
                "x": dml.TensorSpec([2, 4, 3, 8], "float32"),
                "weight": dml.TensorSpec([8], "float32"),
                "bias": dml.TensorSpec([8], "float32"),
            },
            name="fresh_group_norm",
        )

    def inputs():
        return {
            "x": np.linspace(-1.5, 1.5, num=2 * 4 * 3 * 8, dtype=np.float32).reshape(2, 4, 3, 8),
            "weight": np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25], dtype=np.float32),
            "bias": np.array([-0.5, -0.25, 0.0, 0.25, 0.5, -0.125, 0.125, 0.375], dtype=np.float32),
        }

    return GraphCase(
        "group_norm",
        build_spec,
        inputs,
        frozenset({"group_norm", "group_norm_swish"}),
        atol=1e-4,
        rtol=1e-4,
    )


class DtypeGroupNormModule(dml.Module):
    def forward(self, x16, weight16, bias16, xbf, weightbf, biasbf):
        return {
            "group_norm_float16": dml.ops.output(dml.ops.group_norm(x16, 4, weight16, bias16, eps=1e-5), "group_norm_float16"),
            "group_norm_swish_float16": dml.ops.output(
                dml.ops.group_norm_swish(x16, 4, weight16, bias16, eps=1e-5),
                "group_norm_swish_float16",
            ),
            "group_norm_bfloat16": dml.ops.output(dml.ops.group_norm(xbf, 4, weightbf, biasbf, eps=1e-5), "group_norm_bfloat16"),
            "group_norm_swish_bfloat16": dml.ops.output(
                dml.ops.group_norm_swish(xbf, 4, weightbf, biasbf, eps=1e-5),
                "group_norm_swish_bfloat16",
            ),
        }


def dtype_group_norm_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            DtypeGroupNormModule(),
            inputs={
                "x16": dml.TensorSpec([2, 4, 3, 8], "float16"),
                "weight16": dml.TensorSpec([8], "float16"),
                "bias16": dml.TensorSpec([8], "float16"),
                "xbf": dml.TensorSpec([2, 4, 3, 8], "bfloat16"),
                "weightbf": dml.TensorSpec([8], "bfloat16"),
                "biasbf": dml.TensorSpec([8], "bfloat16"),
            },
            name="fresh_dtype_group_norm",
        )

    def inputs():
        x = np.linspace(-1.5, 1.5, num=2 * 4 * 3 * 8, dtype=np.float32).reshape(2, 4, 3, 8)
        weight = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25], dtype=np.float32)
        bias = np.array([-0.5, -0.25, 0.0, 0.25, 0.5, -0.125, 0.125, 0.375], dtype=np.float32)
        return {
            "x16": _roundtrip(x, "float16"),
            "weight16": _roundtrip(weight, "float16"),
            "bias16": _roundtrip(bias, "float16"),
            "xbf": _roundtrip(x, "bfloat16"),
            "weightbf": _roundtrip(weight, "bfloat16"),
            "biasbf": _roundtrip(bias, "bfloat16"),
        }

    return GraphCase(
        "dtype_group_norm",
        build_spec,
        inputs,
        frozenset({"group_norm", "group_norm_swish"}),
        atol=3e-2,
        rtol=3e-2,
    )


class PositionalModule(dml.Module):
    def forward(self, timesteps, positions):
        cos, sin = dml.ops.get_1d_rotary_pos_embed(6, positions)
        return {
            "timestep": dml.ops.output(dml.ops.get_timestep_embedding(timesteps, embedding_dim=6), "timestep"),
            "rotary_cos": dml.ops.output(cos, "rotary_cos"),
            "rotary_sin": dml.ops.output(sin, "rotary_sin"),
        }


def positional_case() -> GraphCase:
    def build_spec():
        return dml.trace(PositionalModule(), inputs={"timesteps": dml.TensorSpec([3], "float32"), "positions": dml.TensorSpec([3], "float32")}, name="fresh_positional")

    def inputs():
        return {"timesteps": np.array([0.0, 1.0, 10.0], dtype=np.float32), "positions": np.array([0.0, 1.0, 2.0], dtype=np.float32)}

    return GraphCase("positional", build_spec, inputs, frozenset({"get_timestep_embedding", "get_1d_rotary_pos_embed_cos", "get_1d_rotary_pos_embed_sin"}), atol=1e-4, rtol=1e-4)


class DtypePositionalModule(dml.Module):
    def forward(self, timesteps16, positions16, timestepsbf, positionsbf):
        cos16, sin16 = dml.ops.get_1d_rotary_pos_embed(6, positions16, dtype="float16")
        cosbf, sinbf = dml.ops.get_1d_rotary_pos_embed(6, positionsbf, dtype="bfloat16")
        return {
            "timestep_float16": dml.ops.output(dml.ops.get_timestep_embedding(timesteps16, embedding_dim=6), "timestep_float16"),
            "rotary_cos_float16": dml.ops.output(cos16, "rotary_cos_float16"),
            "rotary_sin_float16": dml.ops.output(sin16, "rotary_sin_float16"),
            "timestep_bfloat16": dml.ops.output(dml.ops.get_timestep_embedding(timestepsbf, embedding_dim=6), "timestep_bfloat16"),
            "rotary_cos_bfloat16": dml.ops.output(cosbf, "rotary_cos_bfloat16"),
            "rotary_sin_bfloat16": dml.ops.output(sinbf, "rotary_sin_bfloat16"),
        }


def dtype_positional_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            DtypePositionalModule(),
            inputs={
                "timesteps16": dml.TensorSpec([3], "float16"),
                "positions16": dml.TensorSpec([3], "float16"),
                "timestepsbf": dml.TensorSpec([3], "bfloat16"),
                "positionsbf": dml.TensorSpec([3], "bfloat16"),
            },
            name="fresh_dtype_positional",
        )

    def inputs():
        timesteps = np.array([0.0, 1.0, 10.0], dtype=np.float32)
        positions = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        return {
            "timesteps16": _roundtrip(timesteps, "float16"),
            "positions16": _roundtrip(positions, "float16"),
            "timestepsbf": _roundtrip(timesteps, "bfloat16"),
            "positionsbf": _roundtrip(positions, "bfloat16"),
        }

    return GraphCase(
        "dtype_positional",
        build_spec,
        inputs,
        frozenset({"get_timestep_embedding", "get_1d_rotary_pos_embed_cos", "get_1d_rotary_pos_embed_sin"}),
        atol=3e-2,
        rtol=3e-2,
    )


class RotaryPositionalFusionModule(dml.Module):
    def forward(self):
        cos2, sin2 = dml.ops.get_2d_rotary_pos_embed(
            embed_dim=64,
            crops_coords=((0.0, 0.0), (1.5, 2.25)),
            grid_size=(4, 5),
        )
        real2, imag2 = dml.ops.get_2d_rotary_pos_embed_lumina(
            embed_dim=64,
            len_h=4,
            len_w=5,
            linear_factor=1.25,
            ntk_factor=1.5,
        )
        cos3, sin3 = dml.ops.get_3d_rotary_pos_embed(
            embed_dim=64,
            crops_coords=((0.0, 0.0), (1.25, 2.0)),
            grid_size=(3, 4),
            temporal_size=2,
            grid_type="slice",
            max_size=(6, 7),
        )
        (freqs, grids) = dml.ops.get_3d_rotary_pos_embed_allegro(
            height=96,
            width=128,
            num_frames=3,
            vae_scale_factor_spatial=8,
            patch_size=2,
            interpolation_scale_h=2.0,
            interpolation_scale_t=2.2,
            interpolation_scale_w=2.0,
            attention_head_dim=96,
        )
        (t_freqs, h_freqs, w_freqs) = freqs
        t_cos, t_sin = t_freqs
        h_cos, h_sin = h_freqs
        w_cos, w_sin = w_freqs
        grid_t, grid_h, grid_w = grids
        return {
            "rotary2d_cos": dml.ops.output(cos2, "rotary2d_cos"),
            "rotary2d_sin": dml.ops.output(sin2, "rotary2d_sin"),
            "rotary2d_lumina_real": dml.ops.output(real2, "rotary2d_lumina_real"),
            "rotary2d_lumina_imag": dml.ops.output(imag2, "rotary2d_lumina_imag"),
            "rotary3d_cos": dml.ops.output(cos3, "rotary3d_cos"),
            "rotary3d_sin": dml.ops.output(sin3, "rotary3d_sin"),
            "allegro_t_cos": dml.ops.output(t_cos, "allegro_t_cos"),
            "allegro_t_sin": dml.ops.output(t_sin, "allegro_t_sin"),
            "allegro_h_cos": dml.ops.output(h_cos, "allegro_h_cos"),
            "allegro_h_sin": dml.ops.output(h_sin, "allegro_h_sin"),
            "allegro_w_cos": dml.ops.output(w_cos, "allegro_w_cos"),
            "allegro_w_sin": dml.ops.output(w_sin, "allegro_w_sin"),
            "allegro_grid_t": dml.ops.output(grid_t, "allegro_grid_t"),
            "allegro_grid_h": dml.ops.output(grid_h, "allegro_grid_h"),
            "allegro_grid_w": dml.ops.output(grid_w, "allegro_grid_w"),
        }


def rotary_positional_fusions_case() -> GraphCase:
    return GraphCase(
        "rotary_positional_fusions",
        lambda: dml.trace(RotaryPositionalFusionModule(), inputs={}, name="fresh_rotary_positional_fusions"),
        lambda: {},
        frozenset(
            {
                "get_2d_rotary_pos_embed",
                "get_2d_rotary_pos_embed_lumina",
                "get_3d_rotary_pos_embed",
                "get_3d_rotary_pos_embed_allegro",
            }
        ),
        rocm=True,
        cuda=True,
        atol=1e-4,
        rtol=1e-4,
    )


class PositionalHelperFusionModule(dml.Module):
    def forward(self, x, weight, box, rel_embedding, seq):
        return {
            "cropped_pos_embed": dml.ops.output(
                dml.ops.cropped_pos_embed(
                    embed_dim=16,
                    pos_embed_max_size=8,
                    base_size=4,
                    interpolation_scale=1.0,
                    patch_size=2,
                    height=8,
                    width=12,
                ),
                "cropped_pos_embed",
            ),
            "gaussian_fourier_projection": dml.ops.output(
                dml.ops.gaussian_fourier_projection(x, weight, log=True, flip_sin_to_cos=True),
                "gaussian_fourier_projection",
            ),
            "get_fourier_embeds_from_boundingbox": dml.ops.output(
                dml.ops.get_fourier_embeds_from_boundingbox(4, box),
                "get_fourier_embeds_from_boundingbox",
            ),
            "relative_attention_bias": dml.ops.output(
                dml.ops.relative_attention_bias(
                    rel_embedding,
                    3,
                    5,
                    bidirectional=True,
                    num_buckets=16,
                    max_distance=32,
                ),
                "relative_attention_bias",
            ),
            "sinusoidal_positional_embedding": dml.ops.output(
                dml.ops.sinusoidal_positional_embedding(seq, 8, 6),
                "sinusoidal_positional_embedding",
            ),
        }


def positional_helper_fusions_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            PositionalHelperFusionModule(),
            inputs={
                "x": dml.TensorSpec([4], "float32"),
                "weight": dml.TensorSpec([6], "float32"),
                "box": dml.TensorSpec([2, 3, 4], "float32"),
                "rel_embedding": dml.TensorSpec([16, 2], "float32"),
                "seq": dml.TensorSpec([2, 6, 8], "float32"),
            },
            name="fresh_positional_helper_fusions",
        )

    def inputs():
        return {
            "x": np.array([0.125, 0.5, 1.25, 3.0], dtype=np.float32),
            "weight": np.linspace(-0.75, 0.5, num=6, dtype=np.float32),
            "box": np.linspace(-0.5, 0.75, num=24, dtype=np.float32).reshape(2, 3, 4),
            "rel_embedding": np.linspace(-1.0, 1.0, num=32, dtype=np.float32).reshape(16, 2),
            "seq": np.linspace(-1.25, 1.5, num=96, dtype=np.float32).reshape(2, 6, 8),
        }

    return GraphCase(
        "positional_helper_fusions",
        build_spec,
        inputs,
        frozenset(
            {
                "cropped_pos_embed",
                "gaussian_fourier_projection",
                "get_fourier_embeds_from_boundingbox",
                "relative_attention_bias",
                "sinusoidal_positional_embedding",
            }
        ),
        atol=1e-4,
        rtol=1e-4,
    )


class DtypePositionalHelperFusionModule(dml.Module):
    def forward(
        self,
        x16,
        weight16,
        box16,
        rel_embedding16,
        seq16,
        xbf,
        weightbf,
        boxbf,
        rel_embeddingbf,
        seqbf,
    ):
        return {
            "cropped_pos_embed_float16": dml.ops.output(
                dml.ops.cropped_pos_embed(
                    embed_dim=16,
                    pos_embed_max_size=8,
                    base_size=4,
                    interpolation_scale=1.0,
                    patch_size=2,
                    height=8,
                    width=12,
                    dtype="float16",
                ),
                "cropped_pos_embed_float16",
            ),
            "cropped_pos_embed_bfloat16": dml.ops.output(
                dml.ops.cropped_pos_embed(
                    embed_dim=16,
                    pos_embed_max_size=8,
                    base_size=4,
                    interpolation_scale=1.0,
                    patch_size=2,
                    height=8,
                    width=12,
                    dtype="bfloat16",
                ),
                "cropped_pos_embed_bfloat16",
            ),
            "gaussian_fourier_projection_float16": dml.ops.output(
                dml.ops.gaussian_fourier_projection(x16, weight16, log=False, flip_sin_to_cos=False),
                "gaussian_fourier_projection_float16",
            ),
            "gaussian_fourier_projection_bfloat16": dml.ops.output(
                dml.ops.gaussian_fourier_projection(xbf, weightbf, log=False, flip_sin_to_cos=True),
                "gaussian_fourier_projection_bfloat16",
            ),
            "get_fourier_embeds_from_boundingbox_float16": dml.ops.output(
                dml.ops.get_fourier_embeds_from_boundingbox(4, box16),
                "get_fourier_embeds_from_boundingbox_float16",
            ),
            "get_fourier_embeds_from_boundingbox_bfloat16": dml.ops.output(
                dml.ops.get_fourier_embeds_from_boundingbox(4, boxbf),
                "get_fourier_embeds_from_boundingbox_bfloat16",
            ),
            "relative_attention_bias_float16": dml.ops.output(
                dml.ops.relative_attention_bias(
                    rel_embedding16,
                    3,
                    5,
                    bidirectional=False,
                    num_buckets=16,
                    max_distance=32,
                ),
                "relative_attention_bias_float16",
            ),
            "relative_attention_bias_bfloat16": dml.ops.output(
                dml.ops.relative_attention_bias(
                    rel_embeddingbf,
                    3,
                    5,
                    bidirectional=True,
                    num_buckets=16,
                    max_distance=32,
                ),
                "relative_attention_bias_bfloat16",
            ),
            "sinusoidal_positional_embedding_float16": dml.ops.output(
                dml.ops.sinusoidal_positional_embedding(seq16, 8, 6),
                "sinusoidal_positional_embedding_float16",
            ),
            "sinusoidal_positional_embedding_bfloat16": dml.ops.output(
                dml.ops.sinusoidal_positional_embedding(seqbf, 8, 6),
                "sinusoidal_positional_embedding_bfloat16",
            ),
        }


def dtype_positional_helper_fusions_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            DtypePositionalHelperFusionModule(),
            inputs={
                "x16": dml.TensorSpec([4], "float16"),
                "weight16": dml.TensorSpec([6], "float16"),
                "box16": dml.TensorSpec([2, 3, 4], "float16"),
                "rel_embedding16": dml.TensorSpec([16, 2], "float16"),
                "seq16": dml.TensorSpec([2, 6, 8], "float16"),
                "xbf": dml.TensorSpec([4], "bfloat16"),
                "weightbf": dml.TensorSpec([6], "bfloat16"),
                "boxbf": dml.TensorSpec([2, 3, 4], "bfloat16"),
                "rel_embeddingbf": dml.TensorSpec([16, 2], "bfloat16"),
                "seqbf": dml.TensorSpec([2, 6, 8], "bfloat16"),
            },
            name="fresh_dtype_pos_helpers",
        )

    def inputs():
        x = np.array([0.125, 0.5, 1.25, 3.0], dtype=np.float32)
        weight = np.linspace(-0.75, 0.5, num=6, dtype=np.float32)
        box = np.linspace(-0.5, 0.75, num=24, dtype=np.float32).reshape(2, 3, 4)
        rel_embedding = np.linspace(-1.0, 1.0, num=32, dtype=np.float32).reshape(16, 2)
        seq = np.linspace(-1.25, 1.5, num=96, dtype=np.float32).reshape(2, 6, 8)
        return {
            "x16": _roundtrip(x, "float16"),
            "weight16": _roundtrip(weight, "float16"),
            "box16": _roundtrip(box, "float16"),
            "rel_embedding16": _roundtrip(rel_embedding, "float16"),
            "seq16": _roundtrip(seq, "float16"),
            "xbf": _roundtrip(x, "bfloat16"),
            "weightbf": _roundtrip(weight, "bfloat16"),
            "boxbf": _roundtrip(box, "bfloat16"),
            "rel_embeddingbf": _roundtrip(rel_embedding, "bfloat16"),
            "seqbf": _roundtrip(seq, "bfloat16"),
        }

    return GraphCase(
        "dtype_pos_helpers",
        build_spec,
        inputs,
        frozenset(
            {
                "cropped_pos_embed",
                "gaussian_fourier_projection",
                "get_fourier_embeds_from_boundingbox",
                "relative_attention_bias",
                "sinusoidal_positional_embedding",
            }
        ),
        atol=3e-2,
        rtol=3e-2,
    )


class VisionLayoutModule(dml.Module):
    def forward(self, x_shuffle, x_unshuffle):
        return {
            "pixel_shuffle": dml.ops.output(dml.ops.pixel_shuffle(x_shuffle, 2), "pixel_shuffle"),
            "pixel_unshuffle": dml.ops.output(dml.ops.pixel_unshuffle(x_unshuffle, 2), "pixel_unshuffle"),
        }


def vision_layout_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            VisionLayoutModule(),
            inputs={"x_shuffle": dml.TensorSpec([1, 8, 2, 3], "float32"), "x_unshuffle": dml.TensorSpec([1, 2, 4, 6], "float32")},
            name="fresh_vision_layout",
        )

    def inputs():
        return {
            "x_shuffle": np.arange(1 * 8 * 2 * 3, dtype=np.float32).reshape(1, 8, 2, 3),
            "x_unshuffle": np.arange(1 * 2 * 4 * 6, dtype=np.float32).reshape(1, 2, 4, 6),
        }

    return GraphCase("vision_layout", build_spec, inputs, frozenset({"reshape", "permute", "pixel_shuffle", "pixel_unshuffle"}))


class EmbeddingModule(dml.Module):
    def forward(self, table, indices):
        return dml.ops.output(dml.ops.embedding(table, indices), "embedding")


def embedding_case() -> GraphCase:
    def build_spec():
        return dml.trace(EmbeddingModule(), inputs={"table": dml.TensorSpec([5, 3], "float32"), "indices": dml.TensorSpec([2, 2], "int64")}, name="fresh_embedding")

    def inputs():
        return {"table": np.arange(15, dtype=np.float32).reshape(5, 3), "indices": np.array([[0, 3], [4, 1]], dtype=np.int64)}

    return GraphCase("embedding", build_spec, inputs, frozenset({"embedding"}))


class DtypeEmbeddingModule(dml.Module):
    def forward(self, table16, tablebf, indices32):
        return {
            "embedding_float16": dml.ops.output(dml.ops.embedding(table16, indices32), "embedding_float16"),
            "embedding_bfloat16": dml.ops.output(dml.ops.embedding(tablebf, indices32), "embedding_bfloat16"),
        }


def dtype_embedding_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            DtypeEmbeddingModule(),
            inputs={
                "table16": dml.TensorSpec([5, 3], "float16"),
                "tablebf": dml.TensorSpec([5, 3], "bfloat16"),
                "indices32": dml.TensorSpec([2, 2], "int32"),
            },
            name="fresh_dtype_embedding",
        )

    def inputs():
        table = np.arange(15, dtype=np.float32).reshape(5, 3)
        return {
            "table16": _roundtrip(table, "float16"),
            "tablebf": _roundtrip(table, "bfloat16"),
            "indices32": np.array([[0, 3], [4, 1]], dtype=np.int32),
        }

    return GraphCase(
        "dtype_embedding",
        build_spec,
        inputs,
        frozenset({"embedding"}),
        atol=2e-2,
        rtol=2e-2,
    )


class SplitChunkModule(dml.Module):
    def forward(self, x):
        split_a, split_b, split_c = dml.ops.split(x, [1, 2, 1], dim=1)
        chunk_a, chunk_b = dml.ops.chunk(x, 2, dim=1)
        return {
            "split_a": dml.ops.output(split_a, "split_a"),
            "split_b": dml.ops.output(split_b, "split_b"),
            "split_c": dml.ops.output(split_c, "split_c"),
            "chunk_a": dml.ops.output(chunk_a, "chunk_a"),
            "chunk_b": dml.ops.output(chunk_b, "chunk_b"),
        }


def split_chunk_case() -> GraphCase:
    def build_spec():
        return dml.trace(SplitChunkModule(), inputs={"x": dml.TensorSpec([2, 4, 3], "float32")}, name="fresh_split_chunk")

    def inputs():
        return {"x": np.arange(24, dtype=np.float32).reshape(2, 4, 3)}

    return GraphCase("split_chunk", build_spec, inputs, frozenset({"dynamic_slice", "split", "chunk"}))


class MeshgridModule(dml.Module):
    def forward(self, x, y):
        xx, yy = dml.ops.meshgrid([x, y])
        return {"xx": dml.ops.output(xx, "xx"), "yy": dml.ops.output(yy, "yy")}


def meshgrid_case() -> GraphCase:
    def build_spec():
        return dml.trace(MeshgridModule(), inputs={"x": dml.TensorSpec([2], "float32"), "y": dml.TensorSpec([3], "float32")}, name="fresh_meshgrid")

    def inputs():
        return {"x": np.array([1.0, 2.0], dtype=np.float32), "y": np.array([3.0, 4.0, 5.0], dtype=np.float32)}

    return GraphCase("meshgrid", build_spec, inputs, frozenset({"reshape", "expand", "meshgrid"}))


class GemmBmmConvModule(dml.Module):
    def forward(self, a, b, bias, batch_a, batch_b, x, weight, conv_bias):
        return {
            "gemm_rcr_bias": dml.ops.output(dml.ops.gemm_rcr_bias(a, b, bias), "gemm_rcr_bias"),
            "bmm_rrr": dml.ops.output(dml.ops.bmm_rrr(batch_a, batch_b), "bmm_rrr"),
            "conv2d_bias": dml.ops.output(dml.ops.conv2d_bias(x, weight, conv_bias, padding=1), "conv2d_bias"),
        }


def provider_ops_case() -> GraphCase:
    def build_spec():
        return dml.trace(
            GemmBmmConvModule(),
            inputs={
                "a": dml.TensorSpec([2, 3], "float32"),
                "b": dml.TensorSpec([4, 3], "float32"),
                "bias": dml.TensorSpec([4], "float32"),
                "batch_a": dml.TensorSpec([2, 3, 4], "float32"),
                "batch_b": dml.TensorSpec([2, 4, 5], "float32"),
                "x": dml.TensorSpec([1, 2, 4, 4], "float32"),
                "weight": dml.TensorSpec([3, 2, 3, 3], "float32"),
                "conv_bias": dml.TensorSpec([3], "float32"),
            },
            name="fresh_provider_ops",
        )

    def inputs():
        return {
            "a": np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0,
            "b": np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0,
            "bias": np.array([0.0, 0.5, -0.25, 1.0], dtype=np.float32),
            "batch_a": np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10.0,
            "batch_b": np.arange(40, dtype=np.float32).reshape(2, 4, 5) / 10.0,
            "x": np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4) / 10.0,
            "weight": np.arange(54, dtype=np.float32).reshape(3, 2, 3, 3) / 100.0,
            "conv_bias": np.array([0.0, 0.25, -0.5], dtype=np.float32),
        }

    return GraphCase(
        "provider_ops",
        build_spec,
        inputs,
        frozenset({"gemm_rcr_bias", "bmm_rrr", "conv2d_bias"}),
        rocm=False,
        atol=1e-3,
        rtol=1e-3,
    )


PROVIDER_COVERAGE_DTYPES = ("float16", "float32", "bfloat16")


def _provider_value(shape: tuple[int, ...], dtype: str, start: float = -0.5, step: float = 0.05) -> np.ndarray:
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * np.float32(step) + np.float32(start)
    return _roundtrip(values, dtype)


def _provider_gemm_shapes(op_name: str) -> dict[str, tuple[int, ...]]:
    spec = gemm_op_spec(op_name)
    m, n, k = 2, 4, 3
    output_shape = (m, n)
    shapes = {
        "a": (m, k),
        "b": (k, n) if spec.base_layout == "rrr" else (n, k),
        "bias": (n,),
        "d0": output_shape,
        "d1": output_shape,
    }
    return {name: shapes[name] for name in ("a", "b", *spec.epilogue.inputs)}


def _provider_bmm_shapes(op_name: str) -> dict[str, tuple[int, ...]]:
    spec = bmm_op_spec(op_name)
    batch, m, n, k = 2, 2, 4, 3
    a_shape = (batch, k, m) if spec.a_layout == "c" else (batch, m, k)
    b_shape = (batch, n, k) if spec.b_layout == "c" else (batch, k, n)
    output_shape = (batch, n, m) if spec.c_layout == "c" else (batch, m, n)
    shapes = {"a": a_shape, "b": b_shape, "d0": output_shape}
    return {name: shapes[name] for name in ("a", "b", *spec.inputs)}


def _provider_conv_shapes(op_name: str) -> dict[str, tuple[int, ...]]:
    shapes = {
        "x": (1, 2, 4, 4),
        "weight": (3, 2, 3, 3),
        "bias": (3,),
        "residual": (1, 3, 4, 4),
    }
    inputs = ("x", "weight", "bias", "residual") if op_name.endswith("_add") or op_name.endswith("_add_relu") else ("x", "weight", "bias")
    return {name: shapes[name] for name in inputs}


def _provider_specs_and_inputs() -> tuple[dict[str, dml.TensorSpec], dict[str, np.ndarray]]:
    specs: dict[str, dml.TensorSpec] = {}
    inputs: dict[str, np.ndarray] = {}

    def add_inputs(prefix: str, shapes: dict[str, tuple[int, ...]], dtype: str) -> None:
        for input_name, shape in shapes.items():
            name = f"{prefix}_{input_name}"
            specs[name] = dml.TensorSpec(list(shape), dtype)
            inputs[name] = _provider_value(shape, dtype, start=0.125 if input_name == "b" else -0.25)

    for dtype in PROVIDER_COVERAGE_DTYPES:
        for op_name in GEMM_OPS:
            add_inputs(f"{op_name}_{dtype}", _provider_gemm_shapes(op_name), dtype)
        for op_name in BMM_OPS:
            add_inputs(f"{op_name}_{dtype}", _provider_bmm_shapes(op_name), dtype)
    for dtype in CONV2D_BIAS_DTYPES:
        for op_name in CONV2D_BIAS_FAMILY_OPS:
            add_inputs(f"{op_name}_{dtype}", _provider_conv_shapes(op_name), dtype)
    return specs, inputs


class ProviderCoverageModule(dml.Module):
    def forward(self, **inputs):
        outputs = {}
        for dtype in PROVIDER_COVERAGE_DTYPES:
            for op_name in GEMM_OPS:
                spec = gemm_op_spec(op_name)
                prefix = f"{op_name}_{dtype}"
                args = [inputs[f"{prefix}_a"], inputs[f"{prefix}_b"]]
                args.extend(inputs[f"{prefix}_{name}"] for name in spec.epilogue.inputs)
                outputs[f"{prefix}_out"] = getattr(dml.ops, op_name)(*args)
            for op_name in BMM_OPS:
                spec = bmm_op_spec(op_name)
                prefix = f"{op_name}_{dtype}"
                args = [inputs[f"{prefix}_a"], inputs[f"{prefix}_b"]]
                args.extend(inputs[f"{prefix}_{name}"] for name in spec.inputs)
                outputs[f"{prefix}_out"] = getattr(dml.ops, op_name)(*args)
        for dtype in CONV2D_BIAS_DTYPES:
            for op_name in CONV2D_BIAS_FAMILY_OPS:
                prefix = f"{op_name}_{dtype}"
                args = [inputs[f"{prefix}_x"], inputs[f"{prefix}_weight"], inputs[f"{prefix}_bias"]]
                if op_name.endswith("_add") or op_name.endswith("_add_relu"):
                    args.append(inputs[f"{prefix}_residual"])
                outputs[f"{prefix}_out"] = getattr(dml.ops, op_name)(*args, padding=1)
        return {name: dml.ops.output(value, name) for name, value in outputs.items()}


def provider_coverage_case() -> GraphCase:
    def build_spec():
        specs, _ = _provider_specs_and_inputs()
        return dml.trace(ProviderCoverageModule(), inputs=specs, name="fresh_provider_coverage")

    def inputs():
        _, values = _provider_specs_and_inputs()
        return values

    return GraphCase(
        "provider_coverage",
        build_spec,
        inputs,
        frozenset({*GEMM_OPS, *BMM_OPS, *CONV2D_BIAS_FAMILY_OPS}),
        cpu=False,
        cuda=False,
        rocm=False,
        atol=3e-2,
        rtol=3e-2,
        reason="metadata/reference coverage for the full provider op/dtype surface; runtime compiles remain targeted",
    )


def standard_cases() -> list[GraphCase]:
    return [
        elementwise_case(),
        reduction_case(),
        creation_case(),
        dtype_elementwise_case(),
        dtype_generated_case(),
        dtype_reduction_case(),
        shape_view_case(),
        collection_case(),
        dtype_collection_case(),
        pooling_case(),
        dtype_pooling_case(),
        selection_case(),
        dtype_selection_case(),
        normalization_case(),
        dtype_normalization_case(),
        group_norm_case(),
        dtype_group_norm_case(),
        positional_case(),
        dtype_positional_case(),
        rotary_positional_fusions_case(),
        positional_helper_fusions_case(),
        dtype_positional_helper_fusions_case(),
        vision_layout_case(),
        embedding_case(),
        dtype_embedding_case(),
        split_chunk_case(),
        meshgrid_case(),
        provider_ops_case(),
        provider_coverage_case(),
    ]


def ir_cases() -> list[GraphCase]:
    return standard_cases()
