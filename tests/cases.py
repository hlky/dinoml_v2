from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import dinoml as dml
from dinoml.ir import ModelSpec, array_from_storage, array_to_storage


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


class DtypeGeneratedModule(dml.Module):
    def forward(self, x16, xbf, xbool, xint64, gather_index32, batch_index32, embedding_table, embedding_index32):
        topk_bool_values, topk_bool_indices = dml.ops.topk(xbool, 2, dim=-1)
        return {
            "full_bool": dml.ops.output(dml.ops.full([2, 3], True, dtype="bool"), "full_bool"),
            "full_bfloat16": dml.ops.output(dml.ops.full([2, 3], 1.25, dtype="bfloat16"), "full_bfloat16"),
            "arange_float16": dml.ops.output(dml.ops.arange(0, 6, 1, dtype="float16"), "arange_float16"),
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
    def forward(self, x, y, z, index, batch_index, update):
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
                "dynamic_slice",
                "index_select",
                "gather",
                "batch_gather",
                "slice_scatter",
                "pad",
            }
        ),
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


def standard_cases() -> list[GraphCase]:
    return [
        elementwise_case(),
        reduction_case(),
        creation_case(),
        dtype_generated_case(),
        shape_view_case(),
        collection_case(),
        pooling_case(),
        selection_case(),
        normalization_case(),
        positional_case(),
        vision_layout_case(),
        embedding_case(),
        split_chunk_case(),
        meshgrid_case(),
        provider_ops_case(),
    ]


def ir_cases() -> list[GraphCase]:
    return standard_cases()
