from __future__ import annotations

import ctypes
import hashlib
import importlib.machinery
import importlib.util
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dinoml.ir import array_to_storage, canonical_json, dtype_nbytes, read_json, write_json
from dinoml.kernels.bmm import BMM_OPS, bmm_op_spec, bmm_problem
from dinoml.kernels.gemm import GEMM_OPS, gemm_op_spec, gemm_problem
from dinoml.kernels.manifest import PROFILE_CACHE_SCHEMA_VERSION
from dinoml.kernels.profile_cache import (
    ProfileCacheLookup,
    ProfileCacheWrite,
    default_profile_cache_path,
    open_profile_cache_backend,
)
from dinoml.kernels.providers.cutlass.alignment import (
    alignment_context_candidate_filter,
    cutlass_bmm_profile_alignment_context,
    cutlass_candidate_alignment,
    cutlass_gemm_profile_alignment_context,
    filter_candidates_by_alignment,
)
from dinoml.kernels.providers.cutlass.conv import (
    CONV_OPS,
    cutlass_conv_candidate_compatible_with_plan,
    validate_cutlass_conv_plan,
)
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_split_k_supported
from dinoml.ops.conv import normalize_conv2d_bias_attrs
from dinoml.ops.definitions import get_op_def
from dinoml.shapes import evaluate_symbolic_int, validate_runtime_shape


PROFILE_REPORT_SCHEMA_VERSION = 7
EXECUTION_PLAN_SCHEMA_VERSION = 1
PROFILE_STATISTICS_SCHEMA_VERSION = 1
PROFILE_CONFIDENCE_LEVEL = 0.95
PROFILE_CONFIDENCE_Z_SCORE = 1.96
PROFILE_CONFIDENCE_MIN_REPEATS = 3
PROFILE_CONFIDENCE_MIN_ABSOLUTE_MARGIN_MS = 0.002
PROFILE_CONFIDENCE_MIN_RELATIVE_SPEEDUP = 0.02
PROFILE_ADAPTIVE_MIN_TOTAL_SAMPLE_MS = 0.5
PROFILE_ADAPTIVE_MAX_ITERATIONS = 1024


@dataclass(frozen=True)
class ProfileShapeScenario:
    source: str
    case_id: str
    dim_values: Mapping[str, int]
    dim_sources: Mapping[str, str]
    overrides: Mapping[str, Sequence[int]]


@dataclass(frozen=True)
class PreparedProfileWorkload:
    workload: GemmProfileWorkload | ConvProfileWorkload
    cache_lookup: ProfileCacheLookup
    key_payload: Mapping[str, Any]
    profile_key: str
    resolution: str
    representative: bool
    cache_entry: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class GemmProfileWorkload:
    node_id: str
    op: str
    dtype: str
    kernel_symbol: str
    profiler_symbol: str
    candidate_set_id: str | None
    candidate_set_key: str | None
    candidate_id: str
    candidate_config_key: str | None
    candidate: Mapping[str, Any]
    a_tensor: str
    b_tensor: str
    bias_tensor: str | None
    residual_tensors: tuple[str, ...]
    output_tensor: str
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    bias_shape: tuple[int, ...] | None
    residual_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    m: int
    n: int
    k: int
    split_k: int
    workspace_nbytes: int
    shape_source: str
    shape_case_id: str
    dim_values: Mapping[str, int]
    dim_sources: Mapping[str, str]
    alignment_context: Mapping[str, Any]
    kernel_library: str = "cutlass_gemm"
    batch_count: int | None = None
    batch_stride_a: int | None = None
    batch_stride_b: int | None = None
    batch_stride_d0: int | None = None
    batch_stride_c: int | None = None
    lda: int | None = None
    ldb: int | None = None
    ldd0: int | None = None
    ldc: int | None = None

    def to_json(self) -> dict[str, Any]:
        payload = {
            "node_id": self.node_id,
            "op": self.op,
            "dtype": self.dtype,
            "kernel_library": self.kernel_library,
            "kernel_symbol": self.kernel_symbol,
            "profiler_symbol": self.profiler_symbol,
            "candidate_set_id": self.candidate_set_id,
            "candidate_set_key": self.candidate_set_key,
            "candidate_id": self.candidate_id,
            "candidate_config_key": self.candidate_config_key,
            "candidate": dict(self.candidate),
            "inputs": {
                self.a_tensor: list(self.a_shape),
                self.b_tensor: list(self.b_shape),
                **({self.bias_tensor: list(self.bias_shape or ())} if self.bias_tensor is not None else {}),
                **{name: list(shape) for name, shape in zip(self.residual_tensors, self.residual_shapes)},
            },
            "output": {
                self.output_tensor: list(self.output_shape),
            },
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "split_k": self.split_k,
            "workspace_nbytes": self.workspace_nbytes,
            "profile_variant": {
                "split_k": self.split_k,
            },
            "shape_case": {
                "source": self.shape_source,
                "case_id": self.shape_case_id,
                "dims": dict(self.dim_values),
                "dim_sources": dict(self.dim_sources),
            },
            "alignment_context": dict(self.alignment_context),
        }
        if self.batch_count is not None:
            payload["batch_count"] = int(self.batch_count)
            payload["batch_strides"] = {
                "a": int(self.batch_stride_a or 0),
                "b": int(self.batch_stride_b or 0),
                "c": int(self.batch_stride_c or 0),
            }
            payload["leading_dimensions"] = {
                "a": int(self.lda or 0),
                "b": int(self.ldb or 0),
                "c": int(self.ldc or 0),
            }
            if self.residual_tensors:
                payload["batch_strides"]["d0"] = int(self.batch_stride_d0 or 0)
                payload["leading_dimensions"]["d0"] = int(self.ldd0 or 0)
        return payload


@dataclass(frozen=True)
class ConvProfileWorkload:
    node_id: str
    op: str
    dtype: str
    kernel_symbol: str
    profiler_symbol: str
    candidate_set_id: str | None
    candidate_set_key: str | None
    candidate_id: str
    candidate_config_key: str | None
    candidate: Mapping[str, Any]
    x_tensor: str
    weight_tensor: str
    bias_tensor: str
    residual_tensor: str | None
    output_tensor: str
    x_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    bias_shape: tuple[int, ...]
    residual_shape: tuple[int, ...] | None
    output_shape: tuple[int, ...]
    conv_config: Mapping[str, Any]
    semantic_layout: Mapping[str, str]
    provider_layout: Mapping[str, str]
    layout_translation: Mapping[str, Any]
    weight_transform: Mapping[str, Any]
    temporary_buffers: tuple[Mapping[str, Any], ...]
    workspace_nbytes: int
    source_op: str | None
    bias_mode: str | None
    shape_source: str
    shape_case_id: str
    dim_values: Mapping[str, int]
    dim_sources: Mapping[str, str]
    kernel_library: str = "cutlass_conv"

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op": self.op,
            "dtype": self.dtype,
            "kernel_library": self.kernel_library,
            "kernel_symbol": self.kernel_symbol,
            "profiler_symbol": self.profiler_symbol,
            "candidate_set_id": self.candidate_set_id,
            "candidate_set_key": self.candidate_set_key,
            "candidate_id": self.candidate_id,
            "candidate_config_key": self.candidate_config_key,
            "candidate": dict(self.candidate),
            "inputs": {
                self.x_tensor: list(self.x_shape),
                self.weight_tensor: list(self.weight_shape),
                self.bias_tensor: list(self.bias_shape),
                **({self.residual_tensor: list(self.residual_shape or ())} if self.residual_tensor is not None else {}),
            },
            "output": {self.output_tensor: list(self.output_shape)},
            "conv": dict(self.conv_config),
            "semantic_layout": dict(self.semantic_layout),
            "provider_layout": dict(self.provider_layout),
            "layout_translation": dict(self.layout_translation),
            "weight_transform": dict(self.weight_transform),
            "temporary_buffers": [dict(buffer) for buffer in self.temporary_buffers],
            "workspace_nbytes": self.workspace_nbytes,
            **(
                {"source_op": self.source_op, "bias_mode": self.bias_mode}
                if self.source_op is not None
                else {}
            ),
            "profile_variant": {
                "kind": str(self.candidate.get("status", "runtime")),
                "profiler_status": str(self.candidate.get("profiler_status", "runtime_profiler")),
            },
            "shape_case": {
                "source": self.shape_source,
                "case_id": self.shape_case_id,
                "dims": dict(self.dim_values),
                "dim_sources": dict(self.dim_sources),
            },
        }


def parse_shape_overrides(items: Sequence[str] | None) -> dict[str, tuple[int, ...]]:
    overrides: dict[str, tuple[int, ...]] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"Expected shape override like name=1,128,768, got {item!r}")
        name, raw_shape = item.split("=", 1)
        dims = tuple(int(part) for part in raw_shape.split(",") if part)
        if not name or not dims or any(dim <= 0 for dim in dims):
            raise ValueError(f"Invalid shape override: {item!r}")
        overrides[name] = dims
    return overrides


def build_profile_workloads(
    graph: Mapping[str, Any],
    kernel_manifest: Mapping[str, Any],
    *,
    input_shapes: Mapping[str, Sequence[int]] | None = None,
) -> list[GemmProfileWorkload | ConvProfileWorkload]:
    target_name = str(kernel_manifest.get("target", {}).get("name", ""))
    if target_name not in {"cuda", "rocm"}:
        return []
    tensor_map = {str(tensor["name"]): tensor for tensor in graph["tensors"]}
    has_runtime_overrides = bool(input_shapes)
    metadata_shape_scenarios = _metadata_profile_shape_scenarios(graph, tensor_map) if not has_runtime_overrides else None
    required_by_op: dict[str, list[Mapping[str, Any]]] = {}
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("profiler_symbol"):
            required_by_op.setdefault(str(item["op"]), []).append(item)
    overrides = {name: tuple(int(dim) for dim in shape) for name, shape in (input_shapes or {}).items()}
    workloads = []
    for node in graph["nodes"]:
        op_name = str(node["op"])
        if op_name in BMM_OPS:
            _append_bmm_profile_workloads(
                workloads,
                node,
                tensor_map,
                required_by_op.get(op_name, ()),
                overrides,
                metadata_shape_scenarios,
                backend=target_name,
            )
            continue
        if op_name in CONV_OPS:
            if target_name == "rocm":
                _append_ck_conv_profile_workloads(
                    workloads,
                    node,
                    tensor_map,
                    required_by_op.get(op_name, ()),
                    overrides,
                    metadata_shape_scenarios,
                )
            else:
                _append_conv_profile_workloads(
                    workloads,
                    node,
                    tensor_map,
                    required_by_op.get(op_name, ()),
                    overrides,
                    metadata_shape_scenarios,
                )
            continue
        if op_name not in GEMM_OPS:
            continue
        output_name = str(node["outputs"][0])
        output_info = tensor_map[output_name]
        dtype = str(output_info["dtype"])
        binding = get_op_def(op_name).backend_kernels[target_name].resolve(dtype)
        required_item = _required_profile_item(
            required_by_op.get(op_name, ()),
            dtype,
            binding.symbol,
            node_id=str(node["id"]),
        )
        if required_item is None:
            continue
        spec = gemm_op_spec(op_name)
        a_name, b_name = (str(name) for name in node["inputs"][:2])
        epilogue_tensor_names = {
            input_name: str(node["inputs"][input_offset])
            for input_offset, input_name in enumerate(spec.epilogue.inputs, start=2)
        }
        bias_name = epilogue_tensor_names.get("bias")
        residual_names = tuple(epilogue_tensor_names[name] for name in spec.epilogue.inputs if name.startswith("d"))
        for scenario in _profile_shape_scenarios(node, tensor_map, overrides, metadata_shape_scenarios):
            a_shape = _runtime_tensor_shape(a_name, tensor_map[a_name], scenario.overrides, scenario.dim_values)
            b_shape = _runtime_tensor_shape(b_name, tensor_map[b_name], scenario.overrides, scenario.dim_values)
            bias_shape = (
                _runtime_tensor_shape(bias_name, tensor_map[bias_name], scenario.overrides, scenario.dim_values)
                if bias_name is not None
                else None
            )
            residual_shapes = tuple(
                _runtime_tensor_shape(name, tensor_map[name], scenario.overrides, scenario.dim_values)
                for name in residual_names
            )
            problem_shapes = [a_shape, b_shape, *(shape for shape in (bias_shape,) if shape is not None), *residual_shapes]
            m, n, k, output_shape = gemm_problem(op_name, problem_shapes)
            if target_name == "rocm":
                spec_problem = _ck_gemm_profile_problem(op_name, m=m, n=n, k=k)
                alignment_context = _ck_profile_alignment_context(
                    kind="ck_gemm_profile_alignment_context",
                    problem=spec_problem,
                )
                profile_candidates = _ck_profile_candidates(required_item, spec_problem)
            else:
                alignment_context = cutlass_gemm_profile_alignment_context(
                    op_name,
                    dtype,
                    tensor_map,
                    a_name=a_name,
                    b_name=b_name,
                    c_name=output_name,
                    epilogue_names=tuple(name for name in (bias_name, *residual_names) if name is not None),
                    n=n,
                    k=k,
                )
                profile_candidates = _profile_candidates(required_item, alignment_context=alignment_context)
            for candidate in profile_candidates:
                for split_k in _candidate_profile_split_k_values(candidate, m=m, n=n, k=k):
                    workloads.append(
                        GemmProfileWorkload(
                            node_id=str(node["id"]),
                            op=op_name,
                            dtype=dtype,
                            kernel_symbol=str(candidate.get("kernel_symbol") or binding.symbol),
                            profiler_symbol=str(candidate.get("profiler_symbol") or required_item["profiler_symbol"]),
                            candidate_set_id=(
                                str(required_item["candidate_set_id"])
                                if required_item.get("candidate_set_id") is not None
                                else None
                            ),
                            candidate_set_key=(
                                str(required_item["candidate_set_key"])
                                if required_item.get("candidate_set_key") is not None
                                else None
                            ),
                            candidate_id=str(candidate["candidate_id"]),
                            candidate_config_key=(
                                str(candidate["candidate_config_key"]) if candidate.get("candidate_config_key") is not None else None
                            ),
                            candidate=candidate,
                            a_tensor=a_name,
                            b_tensor=b_name,
                            bias_tensor=bias_name,
                            residual_tensors=residual_names,
                            output_tensor=output_name,
                            a_shape=tuple(a_shape),
                            b_shape=tuple(b_shape),
                            bias_shape=bias_shape,
                            residual_shapes=tuple(tuple(shape) for shape in residual_shapes),
                            output_shape=tuple(output_shape),
                            m=m,
                            n=n,
                            k=k,
                            split_k=split_k,
                            workspace_nbytes=_candidate_profile_workspace_nbytes(candidate, m=m, n=n, split_k=split_k),
                            shape_source=scenario.source,
                            shape_case_id=scenario.case_id,
                            dim_values=scenario.dim_values,
                            dim_sources=scenario.dim_sources,
                            alignment_context=alignment_context,
                            kernel_library=binding.library,
                        )
                    )
    return workloads


def _unsupported_profile_workload_reason(workload: GemmProfileWorkload | ConvProfileWorkload) -> str | None:
    if workload.kernel_library != "cutlass_conv":
        return None
    profiler_status = str(workload.candidate.get("profiler_status", ""))
    if profiler_status in {"runtime_profiler", "bounded_runtime_profiler"}:
        return None
    return f"CUTLASS Conv profile workload for {workload.op} candidate {workload.candidate_id} has unsupported profiler_status={profiler_status!r}."


def _require_supported_profile_workload(
    workload: GemmProfileWorkload | ConvProfileWorkload,
    *,
    context: str,
) -> None:
    reason = _unsupported_profile_workload_reason(workload)
    if reason is None:
        return
    raise NotImplementedError(f"{reason} Unsupported context: {context}.")


def _reject_unsupported_profile_workloads(
    workloads: Sequence[GemmProfileWorkload | ConvProfileWorkload],
    *,
    context: str,
) -> None:
    for workload in workloads:
        _require_supported_profile_workload(workload, context=context)


def _append_conv_profile_workloads(
    workloads: list[GemmProfileWorkload | ConvProfileWorkload],
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    required_items: Sequence[Mapping[str, Any]],
    overrides: Mapping[str, Sequence[int]],
    metadata_shape_scenarios: Sequence[ProfileShapeScenario] | None,
) -> None:
    op_name = str(node["op"])
    output_name = str(node["outputs"][0])
    output_info = tensor_map[output_name]
    dtype = str(output_info["dtype"])
    binding = get_op_def(op_name).backend_kernels["cuda"].resolve(dtype)
    required_item = _required_profile_item(required_items, dtype, binding.symbol, node_id=str(node["id"]))
    if required_item is None:
        return
    conv_plan = required_item.get("cutlass_conv_plan")
    x_name, weight_name, bias_name = (str(name) for name in node["inputs"][:3])
    residual_name = str(node["inputs"][3]) if op_name in {"conv2d_bias_add", "conv2d_bias_add_relu"} else None
    normalized_conv_plan = validate_cutlass_conv_plan(
        conv_plan,
        node_id=str(node["id"]),
    )
    profile_candidates = [
        candidate
        for candidate in _profile_candidates(required_item)
        if cutlass_conv_candidate_compatible_with_plan(candidate, normalized_conv_plan)
    ]
    if not profile_candidates:
        raise ValueError(
            "CUTLASS Conv profile workload construction found no candidate compatible with "
            f"node {node.get('id')!r} shape/layout/dtype contract"
        )
    for scenario in _profile_shape_scenarios(node, tensor_map, overrides, metadata_shape_scenarios):
        x_shape = _runtime_tensor_shape(x_name, tensor_map[x_name], scenario.overrides, scenario.dim_values)
        weight_shape = _runtime_tensor_shape(weight_name, tensor_map[weight_name], scenario.overrides, scenario.dim_values)
        bias_shape = _runtime_tensor_shape(bias_name, tensor_map[bias_name], scenario.overrides, scenario.dim_values)
        residual_shape = (
            _runtime_tensor_shape(residual_name, tensor_map[residual_name], scenario.overrides, scenario.dim_values)
            if residual_name is not None
            else None
        )
        output_shape = _runtime_tensor_shape(output_name, output_info, scenario.overrides, scenario.dim_values)
        for candidate in profile_candidates:
            workloads.append(
                ConvProfileWorkload(
                    node_id=str(node["id"]),
                    op=op_name,
                    dtype=dtype,
                    kernel_symbol=str(candidate.get("kernel_symbol") or binding.symbol),
                    profiler_symbol=str(candidate.get("profiler_symbol") or required_item["profiler_symbol"]),
                    candidate_set_id=(
                        str(required_item["candidate_set_id"])
                        if required_item.get("candidate_set_id") is not None
                        else None
                    ),
                    candidate_set_key=(
                        str(required_item["candidate_set_key"])
                        if required_item.get("candidate_set_key") is not None
                        else None
                    ),
                    candidate_id=str(candidate["candidate_id"]),
                    candidate_config_key=(
                        str(candidate["candidate_config_key"]) if candidate.get("candidate_config_key") is not None else None
                    ),
                    candidate=candidate,
                    x_tensor=x_name,
                    weight_tensor=weight_name,
                    bias_tensor=bias_name,
                    residual_tensor=residual_name,
                    output_tensor=output_name,
                    x_shape=tuple(x_shape),
                    weight_shape=tuple(weight_shape),
                    bias_shape=tuple(bias_shape),
                    residual_shape=None if residual_shape is None else tuple(residual_shape),
                    output_shape=tuple(output_shape),
                    conv_config=dict(normalized_conv_plan.get("conv_config", {})),
                    semantic_layout=dict(normalized_conv_plan.get("semantic_layout", {})),
                    provider_layout=dict(normalized_conv_plan.get("provider_layout", {})),
                    layout_translation=dict(normalized_conv_plan.get("layout_translation", {})),
                    weight_transform=dict(normalized_conv_plan.get("weight_transform", {})),
                    temporary_buffers=tuple(
                        dict(buffer)
                        for buffer in normalized_conv_plan.get("temporary_buffers", ())
                        if isinstance(buffer, Mapping)
                    ),
                    workspace_nbytes=int(normalized_conv_plan.get("workspace_nbytes", 0) or 0),
                    source_op=(
                        str(normalized_conv_plan["source_op"])
                        if normalized_conv_plan.get("source_op") is not None
                        else None
                    ),
                    bias_mode=(
                        str(normalized_conv_plan["bias_mode"])
                        if normalized_conv_plan.get("bias_mode") is not None
                        else None
                    ),
                    shape_source=scenario.source,
                    shape_case_id=scenario.case_id,
                    dim_values=scenario.dim_values,
                    dim_sources=scenario.dim_sources,
                )
            )


def _append_ck_conv_profile_workloads(
    workloads: list[GemmProfileWorkload | ConvProfileWorkload],
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    required_items: Sequence[Mapping[str, Any]],
    overrides: Mapping[str, Sequence[int]],
    metadata_shape_scenarios: Sequence[ProfileShapeScenario] | None,
) -> None:
    op_name = str(node["op"])
    if op_name not in {"conv2d_bias", "conv2d_bias_relu", "conv2d_bias_add", "conv2d_bias_add_relu"}:
        return
    output_name = str(node["outputs"][0])
    output_info = tensor_map[output_name]
    dtype = str(output_info["dtype"])
    binding = get_op_def(op_name).backend_kernels["rocm"].resolve(dtype)
    required_item = _required_profile_item(required_items, dtype, binding.symbol, node_id=str(node["id"]))
    if required_item is None:
        return
    x_name, weight_name, bias_name = (str(name) for name in node["inputs"][:3])
    residual_name = str(node["inputs"][3]) if op_name in {"conv2d_bias_add", "conv2d_bias_add_relu"} else None
    attrs = dict(node.get("attrs", {}))
    try:
        stride, padding, dilation, groups = normalize_conv2d_bias_attrs(
            attrs.get("stride", (1, 1)),
            attrs.get("padding", (0, 0)),
            attrs.get("dilation", (1, 1)),
            attrs.get("groups", 1),
        )
    except (NotImplementedError, ValueError):
        return
    if int(groups) != 1:
        return
    for scenario in _profile_shape_scenarios(node, tensor_map, overrides, metadata_shape_scenarios):
        x_shape = _runtime_tensor_shape(x_name, tensor_map[x_name], scenario.overrides, scenario.dim_values)
        weight_shape = _runtime_tensor_shape(weight_name, tensor_map[weight_name], scenario.overrides, scenario.dim_values)
        bias_shape = _runtime_tensor_shape(bias_name, tensor_map[bias_name], scenario.overrides, scenario.dim_values)
        residual_shape = (
            _runtime_tensor_shape(residual_name, tensor_map[residual_name], scenario.overrides, scenario.dim_values)
            if residual_name is not None
            else None
        )
        output_shape = _runtime_tensor_shape(output_name, output_info, scenario.overrides, scenario.dim_values)
        if len(x_shape) != 4 or len(weight_shape) != 4 or len(output_shape) != 4:
            continue
        if residual_shape is not None and tuple(int(dim) for dim in residual_shape) != tuple(int(dim) for dim in output_shape):
            continue
        batch, in_channels, _in_h, _in_w = (int(dim) for dim in x_shape)
        out_channels, _weight_c, kernel_h, kernel_w = (int(dim) for dim in weight_shape)
        out_h, out_w = int(output_shape[2]), int(output_shape[3])
        problem = {
            "batch": batch,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_h": kernel_h,
            "kernel_w": kernel_w,
            "out_h": out_h,
            "out_w": out_w,
            "groups": int(groups),
            "gemm_m": batch * out_h * out_w,
            "gemm_n": out_channels,
            "gemm_k": in_channels * kernel_h * kernel_w,
        }
        profile_candidates = _ck_profile_candidates(required_item, problem)
        conv_config = {"stride": list(stride), "padding": list(padding), "dilation": list(dilation), "groups": int(groups)}
        for candidate in profile_candidates:
            workloads.append(
                ConvProfileWorkload(
                    node_id=str(node["id"]),
                    op=op_name,
                    dtype=dtype,
                    kernel_symbol=str(candidate.get("kernel_symbol") or binding.symbol),
                    profiler_symbol=str(candidate.get("profiler_symbol") or required_item["profiler_symbol"]),
                    candidate_set_id=(
                        str(required_item["candidate_set_id"])
                        if required_item.get("candidate_set_id") is not None
                        else None
                    ),
                    candidate_set_key=(
                        str(required_item["candidate_set_key"])
                        if required_item.get("candidate_set_key") is not None
                        else None
                    ),
                    candidate_id=str(candidate["candidate_id"]),
                    candidate_config_key=(
                        str(candidate["candidate_config_key"]) if candidate.get("candidate_config_key") is not None else None
                    ),
                    candidate=candidate,
                    x_tensor=x_name,
                    weight_tensor=weight_name,
                    bias_tensor=bias_name,
                    residual_tensor=residual_name,
                    output_tensor=output_name,
                    x_shape=tuple(x_shape),
                    weight_shape=tuple(weight_shape),
                    bias_shape=tuple(bias_shape),
                    residual_shape=None if residual_shape is None else tuple(residual_shape),
                    output_shape=tuple(output_shape),
                    conv_config=conv_config,
                    semantic_layout=dict(candidate.get("semantic_layout", {})),
                    provider_layout=dict(candidate.get("provider_layout", {})),
                    layout_translation={},
                    weight_transform={},
                    temporary_buffers=(),
                    workspace_nbytes=int(candidate.get("workspace_nbytes", 0) or 0),
                    source_op=None,
                    bias_mode=None,
                    shape_source=scenario.source,
                    shape_case_id=scenario.case_id,
                    dim_values=scenario.dim_values,
                    dim_sources=scenario.dim_sources,
                    kernel_library=binding.library,
                )
            )


def _append_bmm_profile_workloads(
    workloads: list[GemmProfileWorkload],
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    required_items: Sequence[Mapping[str, Any]],
    overrides: Mapping[str, Sequence[int]],
    metadata_shape_scenarios: Sequence[ProfileShapeScenario] | None,
    *,
    backend: str,
) -> None:
    op_name = str(node["op"])
    output_name = str(node["outputs"][0])
    output_info = tensor_map[output_name]
    dtype = str(output_info["dtype"])
    binding = get_op_def(op_name).backend_kernels[backend].resolve(dtype)
    required_item = _required_profile_item(required_items, dtype, binding.symbol, node_id=str(node["id"]))
    if required_item is None:
        return
    spec = bmm_op_spec(op_name)
    a_name, b_name = (str(name) for name in node["inputs"][:2])
    residual_names = tuple(str(node["inputs"][input_offset]) for input_offset, _name in enumerate(spec.inputs, start=2))
    for scenario in _profile_shape_scenarios(node, tensor_map, overrides, metadata_shape_scenarios):
        a_shape = _runtime_tensor_shape(a_name, tensor_map[a_name], scenario.overrides, scenario.dim_values)
        b_shape = _runtime_tensor_shape(b_name, tensor_map[b_name], scenario.overrides, scenario.dim_values)
        residual_shapes = tuple(
            _runtime_tensor_shape(name, tensor_map[name], scenario.overrides, scenario.dim_values)
            for name in residual_names
        )
        batch_count, m, n, k, output_shape = bmm_problem(op_name, [a_shape, b_shape, *residual_shapes])
        if backend == "rocm":
            problem = _ck_bmm_profile_problem(op_name, batch=batch_count, m=m, n=n, k=k)
            alignment_context = _ck_profile_alignment_context(
                kind="ck_bmm_profile_alignment_context",
                problem=problem,
            )
            profile_candidates = _ck_profile_candidates(required_item, problem)
        else:
            alignment_context = cutlass_bmm_profile_alignment_context(
                op_name,
                dtype,
                tensor_map,
                a_name=a_name,
                b_name=b_name,
                c_name=output_name,
                epilogue_names=residual_names,
                m=m,
                n=n,
                k=k,
            )
            profile_candidates = _profile_candidates(required_item, alignment_context=alignment_context)
        lda = m if spec.a_layout == "c" else k
        ldb = k if spec.b_layout == "c" else n
        ldc = m if spec.c_layout == "c" else n
        batch_stride_a = 0 if int(a_shape[0]) == 1 and batch_count != 1 else m * k
        batch_stride_b = 0 if int(b_shape[0]) == 1 and batch_count != 1 else n * k
        batch_stride_c = m * n
        batch_stride_d0, ldd0 = _bmm_d0_profile_layout(spec.c_layout, residual_shapes, output_shape, m=m, n=n)
        for candidate in profile_candidates:
            for split_k in _candidate_profile_split_k_values(candidate, m=m, n=n, k=k):
                workloads.append(
                    GemmProfileWorkload(
                        node_id=str(node["id"]),
                        op=op_name,
                        dtype=dtype,
                        kernel_symbol=str(candidate.get("kernel_symbol") or binding.symbol),
                        profiler_symbol=str(candidate.get("profiler_symbol") or required_item["profiler_symbol"]),
                        candidate_set_id=(
                            str(required_item["candidate_set_id"])
                            if required_item.get("candidate_set_id") is not None
                            else None
                        ),
                        candidate_set_key=(
                            str(required_item["candidate_set_key"])
                            if required_item.get("candidate_set_key") is not None
                            else None
                        ),
                        candidate_id=str(candidate["candidate_id"]),
                        candidate_config_key=(
                            str(candidate["candidate_config_key"]) if candidate.get("candidate_config_key") is not None else None
                        ),
                        candidate=candidate,
                        a_tensor=a_name,
                        b_tensor=b_name,
                        bias_tensor=None,
                        residual_tensors=residual_names,
                        output_tensor=output_name,
                        a_shape=tuple(a_shape),
                        b_shape=tuple(b_shape),
                        bias_shape=None,
                        residual_shapes=tuple(tuple(shape) for shape in residual_shapes),
                        output_shape=tuple(output_shape),
                        m=m,
                        n=n,
                        k=k,
                        split_k=split_k,
                        workspace_nbytes=_candidate_profile_workspace_nbytes(candidate, m=m, n=n, split_k=split_k),
                        shape_source=scenario.source,
                        shape_case_id=scenario.case_id,
                        dim_values=scenario.dim_values,
                        dim_sources=scenario.dim_sources,
                        alignment_context=alignment_context,
                        kernel_library=binding.library,
                        batch_count=batch_count,
                        batch_stride_a=batch_stride_a,
                        batch_stride_b=batch_stride_b,
                        batch_stride_d0=batch_stride_d0,
                        batch_stride_c=batch_stride_c,
                        lda=lda,
                        ldb=ldb,
                        ldd0=ldd0,
                        ldc=ldc,
                    )
                )


def _bmm_d0_profile_layout(
    c_layout: str,
    residual_shapes: Sequence[Sequence[int]],
    output_shape: Sequence[int],
    *,
    m: int,
    n: int,
) -> tuple[int | None, int | None]:
    if not residual_shapes:
        return None, None
    d0_shape = tuple(int(dim) for dim in residual_shapes[0])
    output = tuple(int(dim) for dim in output_shape)
    if d0_shape == output:
        return int(m * n), int(m if c_layout == "c" else n)
    squeezed = list(d0_shape)
    while len(squeezed) > 1 and squeezed[0] == 1:
        squeezed = squeezed[1:]
    if len(squeezed) == 1 and int(squeezed[0]) == int(output[-1]):
        return 0, 0
    raise RuntimeError(f"CUTLASS BMM profiler only supports full-output or trailing-bias add epilogue, got {d0_shape}")


def _ck_gemm_profile_problem(op_name: str, *, m: int, n: int, k: int) -> dict[str, int | str]:
    spec = gemm_op_spec(op_name)
    return {
        "m": int(m),
        "n": int(n),
        "k": int(k),
        "a_k": int(k),
        "b_k": int(k),
        "b_n": int(n),
        "output_n": int(n),
        "base_layout": spec.base_layout,
    }


def _ck_bmm_profile_problem(op_name: str, *, batch: int, m: int, n: int, k: int) -> dict[str, int | str]:
    spec = bmm_op_spec(op_name)
    return {
        "batch": int(batch),
        "m": int(m),
        "n": int(n),
        "k": int(k),
        "a_m": int(m),
        "a_k": int(k),
        "b_n": int(n),
        "b_k": int(k),
        "output_n": int(n),
        "output_layout": spec.c_layout,
        "base_layout": spec.base_layout,
    }


def _ck_profile_alignment_context(*, kind: str, problem: Mapping[str, int | str]) -> dict[str, Any]:
    return {
        "kind": kind,
        "candidate_filter": {
            "provider": "ck",
            "policy": "selection_predicate",
        },
        "problem": dict(problem),
    }


def _ck_profile_candidates(required_item: Mapping[str, Any], problem: Mapping[str, int | str]) -> list[dict[str, Any]]:
    candidates = [
        dict(candidate)
        for candidate in required_item.get("candidates", [])
        if _ck_profile_candidate_compatible(candidate, problem)
    ]
    if candidates:
        return candidates
    if required_item.get("candidates"):
        fallback = _selected_profile_candidate(required_item)
        return [fallback] if _ck_profile_candidate_compatible(fallback, problem) else []
    return [_selected_profile_candidate(required_item)]


def _ck_profile_candidate_compatible(candidate: Mapping[str, Any], problem: Mapping[str, int | str]) -> bool:
    predicate = candidate.get("selection_predicate")
    if not isinstance(predicate, Mapping):
        return True
    required_output_layout = predicate.get("requires_output_layout")
    if required_output_layout is not None and problem.get("output_layout") != required_output_layout:
        return False
    exact = predicate.get("exact", {})
    if isinstance(exact, Mapping):
        for key, expected in exact.items():
            if problem.get(str(key)) != expected:
                return False
    min_problem = predicate.get("min_problem", {})
    if isinstance(min_problem, Mapping):
        for key, minimum in min_problem.items():
            value = problem.get(str(key))
            if not isinstance(value, int) or value < int(minimum):
                return False
    alignment = predicate.get("alignment", {})
    if isinstance(alignment, Mapping):
        for key, divisor in alignment.items():
            width = int(divisor)
            value = problem.get(str(key))
            if width > 1 and (not isinstance(value, int) or value % width != 0):
                return False
    padded_block_loop_multiple = predicate.get("padded_block_loop_multiple", {})
    if isinstance(padded_block_loop_multiple, Mapping):
        for key, rule in padded_block_loop_multiple.items():
            if not isinstance(rule, Mapping):
                return False
            value = problem.get(str(key))
            block = int(rule.get("block", 0) or 0)
            multiple = int(rule.get("multiple", 0) or 0)
            if not isinstance(value, int) or block <= 0 or multiple <= 0:
                return False
            padded_loop_count = (value + block - 1) // block
            if padded_loop_count % multiple != 0:
                return False
    padded_block_loop_minimum = predicate.get("padded_block_loop_minimum", {})
    if isinstance(padded_block_loop_minimum, Mapping):
        for key, rule in padded_block_loop_minimum.items():
            if not isinstance(rule, Mapping):
                return False
            value = problem.get(str(key))
            block = int(rule.get("block", 0) or 0)
            minimum = int(rule.get("minimum", 0) or 0)
            if not isinstance(value, int) or block <= 0 or minimum <= 0:
                return False
            padded_loop_count = (value + block - 1) // block
            if padded_loop_count < minimum:
                return False
    return True


def _profile_candidates(
    required_item: Mapping[str, Any],
    *,
    alignment_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    candidate_filter = alignment_context_candidate_filter(alignment_context)
    candidates = filter_candidates_by_alignment(
        required_item.get("candidates", []),
        candidate_filter["max_operand_alignment"],
        candidate_filter["max_epilogue_alignment"],
    )
    if candidates:
        return _primary_alignment_profile_candidates(candidates)
    if required_item.get("candidates"):
        raise ValueError(
            "CUTLASS GEMM profiling alignment filter removed all candidates "
            f"for {required_item.get('op')} with filter {candidate_filter}"
        )
    fallback = _selected_profile_candidate(required_item)
    if (
        candidate_filter["max_operand_alignment"] is not None
        and cutlass_candidate_alignment(fallback) > candidate_filter["max_operand_alignment"]
    ):
        raise ValueError(
            "CUTLASS GEMM profiling alignment filter removed manifest default "
            f"for {required_item.get('op')} with filter {candidate_filter}"
        )
    return [fallback]


def _primary_alignment_profile_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best_by_policy: dict[str, Mapping[str, Any]] = {}
    policy_order: list[str] = []
    for candidate in candidates:
        policy_key = _candidate_alignment_policy_key(candidate)
        if policy_key not in best_by_policy:
            policy_order.append(policy_key)
            best_by_policy[policy_key] = candidate
            continue
        if cutlass_candidate_alignment(candidate) > cutlass_candidate_alignment(best_by_policy[policy_key]):
            best_by_policy[policy_key] = candidate
    return [dict(best_by_policy[policy_key]) for policy_key in policy_order]


def _candidate_alignment_policy_key(candidate: Mapping[str, Any]) -> str:
    cutlass = dict(candidate.get("cutlass", {})) if isinstance(candidate.get("cutlass"), Mapping) else {}
    cutlass.pop("align", None)
    payload = {
        "provider": candidate.get("provider"),
        "family": candidate.get("family"),
        "op": candidate.get("op"),
        "dtype": candidate.get("dtype"),
        "accumulator_dtype": candidate.get("accumulator_dtype"),
        "layouts": candidate.get("layouts"),
        "epilogue": candidate.get("epilogue"),
        "epilogue_config": candidate.get("epilogue_config"),
        "launch_abi": candidate.get("launch_abi"),
        "supports_split_k": candidate.get("supports_split_k"),
        "split_k_search": candidate.get("split_k_search"),
        "cutlass": cutlass,
    }
    return canonical_json(payload)


def _candidate_profile_split_k_values(candidate: Mapping[str, Any], *, m: int, n: int, k: int) -> tuple[int, ...]:
    raw_values = candidate.get("split_k_values")
    if raw_values is None:
        raw_values = (candidate.get("split_k_default", 1),)
    elif isinstance(raw_values, int):
        raw_values = (raw_values,)
    values = {int(value) for value in raw_values if int(value) > 0}
    if candidate.get("supports_split_k"):
        search = candidate.get("split_k_search", {})
        if not isinstance(search, Mapping) or search.get("strategy") == "v1_gemm_factor":
            max_split_k = int(search.get("max_split_k", 32) or 32) if isinstance(search, Mapping) else 32
            values.update(_v1_split_k_values(m=m, n=n, k=k, max_split_k=max_split_k))
    values = tuple(sorted(values))
    return values or (1,)


def _v1_split_k_values(*, m: int, n: int, k: int, max_split_k: int = 32) -> tuple[int, ...]:
    values = {1}
    largest_mn = max(int(m), int(n))
    if largest_mn <= 0:
        return (1,)
    factor = int(k) // largest_mn
    if factor <= 1:
        return (1,)
    low = max(1, factor // 4)
    high = min(factor, int(max_split_k))
    if low == 1:
        low += 1
    if low < high:
        values.update(range(low, high, 2))
    return tuple(sorted(value for value in values if value > 0))


def _candidate_profile_workspace_nbytes(candidate: Mapping[str, Any], *, m: int, n: int, split_k: int) -> int:
    if int(split_k) <= 1:
        return int(candidate.get("workspace_nbytes", 0) or 0)
    cutlass = candidate.get("cutlass", {})
    if not isinstance(cutlass, Mapping):
        return int(candidate.get("workspace_nbytes", 0) or 0)
    threadblock = cutlass.get("threadblock", ())
    try:
        tb_m = int(threadblock[0])
        tb_n = int(threadblock[1])
    except (IndexError, TypeError, ValueError):
        return int(candidate.get("workspace_nbytes", 0) or 0)
    if tb_m <= 0 or tb_n <= 0:
        return int(candidate.get("workspace_nbytes", 0) or 0)
    tiles_m = (int(m) + tb_m - 1) // tb_m
    tiles_n = (int(n) + tb_n - 1) // tb_n
    return max(int(candidate.get("workspace_nbytes", 0) or 0), int(tiles_m * tiles_n * 4))


def _required_profile_item(
    required_items: Sequence[Mapping[str, Any]],
    dtype: str,
    fallback_symbol: str,
    *,
    node_id: str | None = None,
) -> Mapping[str, Any] | None:
    if node_id is not None:
        for item in required_items:
            gguf_runtime_dequant = item.get("gguf_runtime_dequant")
            if not isinstance(gguf_runtime_dequant, Mapping):
                continue
            if str(gguf_runtime_dequant.get("node_id")) != str(node_id):
                continue
            if str(gguf_runtime_dequant.get("status")) != "planned_not_lowered":
                continue
            raise NotImplementedError(
                "CUTLASS profiling does not support planned_not_lowered "
                "gguf_runtime_dequant GEMM nodes; generated CUDA lowering "
                "rejects them before profile workload generation"
            )
        node_scoped_items = False
        for item in required_items:
            item_node_ids = _required_profile_item_node_ids(item)
            if not item_node_ids:
                continue
            node_scoped_items = True
            if str(node_id) in item_node_ids and _required_profile_item_matches(item, dtype, fallback_symbol):
                return item
        if node_scoped_items:
            return None
    for item in required_items:
        if _required_profile_item_matches(item, dtype, fallback_symbol, include_fallback=False):
            return item
    for item in required_items:
        if _required_profile_item_matches(item, dtype, fallback_symbol, include_dtype=False):
            return item
    return None


def _required_profile_item_node_ids(item: Mapping[str, Any]) -> set[str]:
    node_ids = {str(item["node_id"])} if item.get("node_id") is not None else set()
    cutlass_conv_plan = item.get("cutlass_conv_plan")
    if isinstance(cutlass_conv_plan, Mapping) and cutlass_conv_plan.get("node_id") is not None:
        node_ids.add(str(cutlass_conv_plan["node_id"]))
    alignment_context = item.get("cutlass_alignment")
    if isinstance(alignment_context, Mapping):
        nodes = alignment_context.get("nodes")
        if isinstance(nodes, Sequence) and not isinstance(nodes, (str, bytes)):
            for node in nodes:
                if isinstance(node, Mapping) and node.get("node_id") is not None:
                    node_ids.add(str(node["node_id"]))
    return node_ids


def _required_profile_item_matches(
    item: Mapping[str, Any],
    dtype: str,
    fallback_symbol: str,
    *,
    include_dtype: bool = True,
    include_fallback: bool = True,
) -> bool:
    if include_dtype:
        candidates = item.get("candidates", [])
        if any(str(candidate.get("dtype")) == dtype for candidate in candidates):
            return True
        candidate_set = item.get("candidate_set", {})
        if isinstance(candidate_set, Mapping) and str(candidate_set.get("dtype")) == dtype:
            return True
    return include_fallback and str(item.get("kernel_symbol")) == fallback_symbol


def _selected_profile_candidate(required_item: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [dict(candidate) for candidate in required_item.get("candidates", [])]
    if candidates:
        selected_id = required_item.get("selected_candidate_id")
        for candidate in candidates:
            if candidate.get("candidate_id") == selected_id:
                return candidate
        return candidates[0]
    return {
        "candidate_id": "manifest_default",
        "symbol_id": "manifest_default",
        "provider": "manifest",
        "family": "unknown",
        "op": required_item.get("op"),
        "kernel_symbol": required_item.get("kernel_symbol"),
        "profiler_symbol": required_item.get("profiler_symbol"),
        "candidate_config_key": None,
    }


def _blocked_profile_items(kernel_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    blocked = []
    for item in kernel_manifest.get("required_kernels", []):
        reason = item.get("profile_blocked_reason")
        if not reason:
            continue
        details = item.get("profile_blocked_details")
        blocked.append(
            {
                "op": item.get("op"),
                "dtype": item.get("dtype"),
                "kernel_library": item.get("kernel_library"),
                "kernel_symbol": item.get("kernel_symbol"),
                "profiler_symbol": item.get("profiler_symbol"),
                "candidate_set_id": item.get("candidate_set_id"),
                "candidate_set_key": item.get("candidate_set_key"),
                "selected_candidate_id": item.get("selected_candidate_id"),
                "reason": str(reason),
                "details": dict(details) if isinstance(details, Mapping) else {},
            }
        )
    return blocked


def _prepare_profile_workloads(
    workloads: Sequence[GemmProfileWorkload | ConvProfileWorkload],
    manifest: Mapping[str, Any],
    kernel_manifest: Mapping[str, Any],
    codegen_plan: Mapping[str, Any],
    cache_backend: Any,
    *,
    iterations: int,
    repeats: int,
    refresh: bool,
    context: Mapping[str, Any],
) -> tuple[list[PreparedProfileWorkload], list[PreparedProfileWorkload]]:
    prepared: list[PreparedProfileWorkload] = []
    unique: list[PreparedProfileWorkload] = []
    representatives: dict[str, ProfileCacheLookup] = {}
    for workload in workloads:
        cache_lookup = _profile_cache_lookup(workload, manifest, kernel_manifest, codegen_plan, context=context)
        key_payload = cache_lookup.key_payload
        profile_key = cache_lookup.profile_key
        representative = profile_key not in representatives
        if representative:
            representatives[profile_key] = cache_lookup
            item = PreparedProfileWorkload(
                workload=workload,
                cache_lookup=cache_lookup,
                key_payload=key_payload,
                profile_key=profile_key,
                resolution="profile",
                representative=True,
            )
            unique.append(item)
        else:
            item = PreparedProfileWorkload(
                workload=workload,
                cache_lookup=cache_lookup,
                key_payload=key_payload,
                profile_key=profile_key,
                resolution="duplicate",
                representative=False,
            )
        prepared.append(item)
    cache_entries = cache_backend.lookup_many([item.cache_lookup for item in unique]) if unique and not refresh else {}
    for index, item in enumerate(unique):
        cached = cache_entries.get(item.profile_key)
        cache_entry = cached if (
            not refresh
            and cached is not None
            and _cache_entry_satisfies(cached, key_payload=item.key_payload, iterations=iterations, repeats=repeats)
        ) else None
        if cache_entry is None:
            continue
        unique[index] = PreparedProfileWorkload(
            workload=item.workload,
            cache_lookup=item.cache_lookup,
            key_payload=item.key_payload,
            profile_key=item.profile_key,
            resolution="cache",
            representative=True,
            cache_entry=cache_entry,
        )
    return prepared, unique


def profile_artifact(
    artifact: str | Path,
    *,
    input_shapes: Mapping[str, Sequence[int]] | None = None,
    iterations: int = 20,
    repeats: int = PROFILE_CONFIDENCE_MIN_REPEATS,
    output: str | Path | None = None,
    execution_plan_output: str | Path | None = None,
    seed: int = 2027,
    refresh: bool = False,
) -> dict[str, Any]:
    iterations = _positive_int(iterations, "iterations")
    repeats = _positive_int(repeats, "repeats")
    artifact_dir = Path(artifact)
    manifest = read_json(artifact_dir / "manifest.json")
    target_name = str(manifest.get("target", {}).get("name", ""))
    if target_name not in {"cuda", "rocm"}:
        raise ValueError("Profiler runner currently supports CUDA and ROCm artifacts only")
    graph = read_json(artifact_dir / manifest["files"]["graph"])
    _validate_symbolic_int_expr_profile_graph(graph)
    kernel_manifest = read_json(artifact_dir / manifest["files"]["kernel_manifest"])
    codegen_plan = read_json(artifact_dir / manifest["files"]["kernel_codegen_plan"])
    workloads = build_profile_workloads(graph, kernel_manifest, input_shapes=input_shapes)
    _reject_unsupported_profile_workloads(workloads, context="profile execution")
    cache_backend = open_profile_cache_backend(codegen_plan)
    context = _profile_context(artifact_dir, manifest, codegen_plan)
    blocked_profile_items = _blocked_profile_items(kernel_manifest)
    prepared_workloads, unique_workloads = _prepare_profile_workloads(
        workloads,
        manifest,
        kernel_manifest,
        codegen_plan,
        cache_backend,
        iterations=iterations,
        repeats=repeats,
        refresh=refresh,
        context=context,
    )
    summary = {
        "profiled": 0,
        "cached": 0,
        "skipped": 0,
        "failed": 0,
        "blocked": len(blocked_profile_items),
    }
    raw_workload_count = len(workloads)
    total_workloads = len(unique_workloads)
    timed_workloads = sum(1 for item in unique_workloads if item.resolution == "profile")
    progress_summary = {
        "profiled": 0,
        "cached": total_workloads - timed_workloads,
        "skipped": 0,
        "failed": 0,
    }
    print(
        "[dml.profile] Starting profile run: "
        f"artifact={artifact_dir}, "
        f"target={target_name}, "
        f"raw_workloads={raw_workload_count}, "
        f"unique_profile_tasks={total_workloads}, "
        f"timed_profile_tasks={timed_workloads}, "
        f"blocked={summary['blocked']}, "
        f"iterations={iterations}, "
        f"repeats={repeats}, "
        f"refresh={refresh}"
    )
    if not workloads:
        report = _profile_report(
            artifact_dir,
            manifest,
            kernel_manifest,
            codegen_plan,
            iterations,
            repeats,
            [],
            summary,
            context=context,
            blocked_profile_items=blocked_profile_items,
        )
        execution_plan = build_execution_plan(report)
        execution_plan_path = _write_execution_plan(execution_plan, artifact_dir, execution_plan_output)
        report["execution_plan"] = _execution_plan_summary(execution_plan, execution_plan_path)
        _write_profile_report(report, artifact_dir, output)
        print("[dml.profile] No profileable workloads were found")
        cache_backend.close()
        return report

    rng = np.random.default_rng(seed)
    profiler = None
    started_at = time.monotonic()
    last_progress_at = started_at
    last_progress_count = int(progress_summary["cached"])
    resolved_results: dict[str, tuple[str, Any, Any]] = {}
    cache_writes: list[ProfileCacheWrite] = []
    for item in unique_workloads:
        if item.resolution == "cache" and item.cache_entry is not None:
            resolved_results[item.profile_key] = ("cache", item.cache_entry, None)
    try:
        profile_workloads = [item for item in unique_workloads if item.resolution == "profile"]
        index = 0
        while index < len(profile_workloads):
            prepared = profile_workloads[index]
            workload = prepared.workload
            if workload.kernel_library in {"cutlass_gemm", "cutlass_bmm", "cutlass_conv"}:
                group = [prepared]
                index += 1
                while index < len(profile_workloads) and _same_cutlass_native_profile_problem(
                    workload,
                    profile_workloads[index].workload,
                ):
                    group.append(profile_workloads[index])
                    index += 1
                if not group:
                    processed = _profile_processed_count(progress_summary)
                    now = time.monotonic()
                    if _should_print_profile_progress(
                        processed,
                        total_workloads,
                        last_progress_count=last_progress_count,
                        elapsed_s=now - started_at,
                        since_last_s=now - last_progress_at,
                    ):
                        print(
                            _format_profile_progress(
                                progress_summary,
                                processed=processed,
                                total_workloads=total_workloads,
                                elapsed_s=now - started_at,
                                current_workload=workload,
                            )
                        )
                        last_progress_at = now
                        last_progress_count = processed
                    continue
                if profiler is None:
                    profiler = _profiler_for_target(artifact_dir, manifest, codegen_plan)
                if workload.kernel_library == "cutlass_gemm":
                    profiled_rows = profiler.profile_cutlass_gemm_problem(
                        workload,
                        iterations=iterations,
                        repeats=repeats,
                    )
                elif workload.kernel_library == "cutlass_bmm":
                    profiled_rows = profiler.profile_cutlass_bmm_problem(
                        workload,
                        iterations=iterations,
                        repeats=repeats,
                    )
                else:
                    profiled_rows = profiler.profile_cutlass_conv_problem(
                        workload,
                        iterations=iterations,
                        repeats=repeats,
                    )
                rows_by_candidate = {str(row["candidate"].get("candidate_id")): row for row in profiled_rows}
                for candidate_item in group:
                    candidate_workload = candidate_item.workload
                    row = rows_by_candidate.get(candidate_workload.candidate_id)
                    if row is None:
                        continue
                    samples_ms = list(row["samples_ms"])
                    timing = _profile_timing(samples_ms, iterations=iterations)
                    workspace_nbytes = max(int(candidate_workload.workspace_nbytes), int(row["workspace_nbytes"]))
                    result = _profile_result(
                        candidate_workload,
                        timing["median_ms"],
                        iterations,
                        profile_key=candidate_item.profile_key,
                        status="ok",
                        workspace_nbytes=workspace_nbytes,
                        timing=timing,
                    )
                    cache_entry = _cache_entry(candidate_workload, result, candidate_item.key_payload)
                    resolved_results[candidate_item.profile_key] = ("result", result, cache_entry)
                    cache_writes.append(ProfileCacheWrite(candidate_item.cache_lookup, cache_entry))
                    progress_summary["profiled"] += 1
                processed = _profile_processed_count(progress_summary)
                now = time.monotonic()
                if _should_print_profile_progress(
                    processed,
                    total_workloads,
                    last_progress_count=last_progress_count,
                    elapsed_s=now - started_at,
                    since_last_s=now - last_progress_at,
                ):
                    print(
                        _format_profile_progress(
                            progress_summary,
                            processed=processed,
                            total_workloads=total_workloads,
                            elapsed_s=now - started_at,
                            current_workload=workload,
                        )
                    )
                    last_progress_at = now
                    last_progress_count = processed
                continue
            index += 1
            if profiler is None:
                profiler = _profiler_for_target(artifact_dir, manifest, codegen_plan)
            try:
                samples_ms, workspace_nbytes, effective_iterations = _profile_workload_samples(
                    profiler,
                    workload,
                    iterations=iterations,
                    repeats=repeats,
                    rng=rng,
                )
            except RuntimeError as exc:
                resolved_results[prepared.profile_key] = ("failure", str(exc), int(iterations))
                progress_summary["failed"] += 1
                processed = _profile_processed_count(progress_summary)
                now = time.monotonic()
                if _should_print_profile_progress(
                    processed,
                    total_workloads,
                    last_progress_count=last_progress_count,
                    elapsed_s=now - started_at,
                    since_last_s=now - last_progress_at,
                ):
                    print(
                        _format_profile_progress(
                            progress_summary,
                            processed=processed,
                            total_workloads=total_workloads,
                            elapsed_s=now - started_at,
                            current_workload=workload,
                        )
                    )
                    last_progress_at = now
                    last_progress_count = processed
                continue
            timing = _profile_timing(samples_ms, iterations=effective_iterations)
            result = _profile_result(
                workload,
                timing["median_ms"],
                effective_iterations,
                profile_key=prepared.profile_key,
                status="ok",
                workspace_nbytes=workspace_nbytes,
                timing=timing,
            )
            if effective_iterations != iterations:
                result["requested_iterations"] = int(iterations)
                result["adaptive_iterations"] = _adaptive_profile_iterations_payload(
                    requested_iterations=iterations,
                    effective_iterations=effective_iterations,
                )
            cache_entry = _cache_entry(workload, result, prepared.key_payload)
            resolved_results[prepared.profile_key] = ("result", result, cache_entry)
            cache_writes.append(ProfileCacheWrite(prepared.cache_lookup, cache_entry))
            progress_summary["profiled"] += 1
            processed = _profile_processed_count(progress_summary)
            now = time.monotonic()
            if _should_print_profile_progress(
                processed,
                total_workloads,
                last_progress_count=last_progress_count,
                elapsed_s=now - started_at,
                since_last_s=now - last_progress_at,
            ):
                print(
                    _format_profile_progress(
                        progress_summary,
                        processed=processed,
                        total_workloads=total_workloads,
                        elapsed_s=now - started_at,
                        current_workload=workload,
                    )
                )
                last_progress_at = now
                last_progress_count = processed
    finally:
        if profiler is not None:
            profiler.close()
    results = []
    for item in prepared_workloads:
        resolved = resolved_results.get(item.profile_key)
        if resolved is None:
            raise RuntimeError(f"Missing resolved profile result for key {item.profile_key}")
        kind = str(resolved[0])
        if kind == "cache":
            result = _profile_result_from_cache(item.workload, resolved[1])
            summary["cached"] += 1
        elif kind == "result":
            if item.representative:
                result = dict(resolved[1])
                summary["profiled"] += 1
            else:
                result = _profile_result_from_cache(item.workload, resolved[2])
                summary["cached"] += 1
        elif kind == "failure":
            result = _profile_failure_result(
                item.workload,
                int(resolved[2]),
                profile_key=item.profile_key,
                error=str(resolved[1]),
            )
            summary["failed"] += 1
        else:
            raise RuntimeError(f"Unsupported resolved profile result kind {kind!r}")
        results.append(result)
    if progress_summary["profiled"]:
        cache_backend.upsert_many(cache_writes)

    report = _profile_report(
        artifact_dir,
        manifest,
        kernel_manifest,
        codegen_plan,
        iterations,
        repeats,
        results,
        summary,
        context=context,
        blocked_profile_items=blocked_profile_items,
    )
    execution_plan = build_execution_plan(report)
    execution_plan_path = _write_execution_plan(execution_plan, artifact_dir, execution_plan_output)
    report["execution_plan"] = _execution_plan_summary(execution_plan, execution_plan_path)
    _write_profile_report(report, artifact_dir, output)
    cache_backend.close()
    print(
        "[dml.profile] Completed profile run: "
        f"raw_workloads={raw_workload_count}, "
        f"unique_profile_tasks={total_workloads}, "
        f"profiled={summary['profiled']}, "
        f"cached={summary['cached']}, "
        f"failed={summary['failed']}, "
        f"blocked={summary['blocked']}, "
        f"elapsed={time.monotonic() - started_at:.1f}s"
    )
    return report


def profile_cache_path(codegen_plan: Mapping[str, Any]) -> Path:
    return default_profile_cache_path(codegen_plan)


def _validate_symbolic_int_expr_profile_graph(graph: Mapping[str, Any]) -> None:
    sources = _profile_graph_direct_dim_sources(graph)
    for section in ("inputs", "outputs", "constants", "tensors"):
        for item in graph.get(section, []):
            for axis, dim in enumerate(item.get("shape_spec", item.get("shape", []))):
                missing = _missing_symbolic_int_sources(dim, sources)
                if missing:
                    raise NotImplementedError(
                        "Symbolic integer shape expressions in shape_spec require direct runtime sources for profiling "
                        f"({section} entry {item.get('name', item.get('tensor'))!r}, axis {axis}, missing {missing})."
                    )
    for view in graph.get("metadata", {}).get("memory_plan", {}).get("views", {}).get("views", []):
        for axis, dim in enumerate(view.get("shape_spec", view.get("shape", []))):
            missing = _missing_symbolic_int_sources(dim, sources)
            if missing:
                raise NotImplementedError(
                    "Symbolic integer shape expressions in shape_spec require direct runtime sources for profiling "
                    f"(view tensor {view.get('tensor')!r}, axis {axis}, missing {missing})."
                )


def _profile_graph_direct_dim_sources(graph: Mapping[str, Any]) -> set[str]:
    sources: set[str] = set()
    for section in ("inputs", "constants"):
        for item in graph.get(section, []):
            for dim in item.get("shape_spec", item.get("shape", [])):
                if isinstance(dim, Mapping) and dim.get("kind") == "dim":
                    sources.add(str(dim["name"]))
    return sources


def _missing_symbolic_int_sources(value: Any, sources: set[str]) -> list[str]:
    if not isinstance(value, Mapping) or value.get("kind") != "int_expr":
        return []
    return sorted({str(leaf["name"]) for leaf in _named_dim_leaves(value) if str(leaf["name"]) not in sources})


def _positive_int(value: int, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _profile_processed_count(summary: Mapping[str, Any]) -> int:
    return sum(int(summary.get(name, 0) or 0) for name in ("profiled", "cached", "skipped", "failed"))


def _should_print_profile_progress(
    processed: int,
    total_workloads: int,
    *,
    last_progress_count: int,
    elapsed_s: float,
    since_last_s: float,
) -> bool:
    if total_workloads <= 0:
        return False
    if processed >= total_workloads:
        return True
    count_interval = max(1, min(1000, total_workloads // 100))
    if processed - last_progress_count >= count_interval:
        return True
    return elapsed_s >= 5.0 and since_last_s >= 5.0 and processed > last_progress_count


def _format_profile_progress(
    summary: Mapping[str, Any],
    *,
    processed: int,
    total_workloads: int,
    elapsed_s: float,
    current_workload: GemmProfileWorkload | ConvProfileWorkload | None,
) -> str:
    fraction = processed / total_workloads if total_workloads > 0 else 1.0
    eta_s = ((elapsed_s / fraction) - elapsed_s) if fraction > 0.0 and processed < total_workloads else 0.0
    current = ""
    if current_workload is not None:
        current = (
            f", current={current_workload.kernel_library}:{current_workload.op}:{current_workload.dtype}"
            f":{current_workload.shape_case_id}"
        )
    return (
        "[dml.profile] Progress: "
        f"{processed}/{total_workloads} ({fraction * 100.0:.1f}%), "
        f"profiled={int(summary.get('profiled', 0) or 0)}, "
        f"cached={int(summary.get('cached', 0) or 0)}, "
        f"failed={int(summary.get('failed', 0) or 0)}, "
        f"elapsed={elapsed_s:.1f}s, "
        f"eta={max(0.0, eta_s):.1f}s"
        f"{current}"
    )


def _runtime_tensor_shape(
    name: str,
    tensor: Mapping[str, Any],
    overrides: Mapping[str, Sequence[int]],
    dim_values: Mapping[str, int] | None = None,
) -> tuple[int, ...]:
    if name in overrides:
        if dim_values:
            return _validate_profile_runtime_shape(tensor, overrides[name], dim_values)
        return tuple(validate_runtime_shape(name, overrides[name], tensor))
    if dim_values:
        return _shape_from_dim_assignments(tensor, dim_values)
    return tuple(int(dim) for dim in tensor["shape"])


def _profile_shape_scenarios(
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Sequence[int]],
    metadata_shape_scenarios: Sequence[ProfileShapeScenario] | None = None,
) -> list[ProfileShapeScenario]:
    if overrides:
        dim_values, dim_sources = _profile_dim_values_from_overrides(tensor_map, overrides)
        return [
            ProfileShapeScenario(
                source="runtime_override",
                case_id="runtime_override",
                dim_values=dim_values,
                dim_sources=dim_sources,
                overrides=overrides,
            )
        ]
    if metadata_shape_scenarios is not None:
        return list(metadata_shape_scenarios)
    dynamic_values = _profile_dim_values(node, tensor_map)
    if not dynamic_values or not any(info["source"] != "max" or len(info["values"]) > 1 for info in dynamic_values.values()):
        return [
            ProfileShapeScenario(
                source="graph_max_shape",
                case_id="max",
                dim_values={},
                dim_sources={},
                overrides={},
            )
        ]

    scenarios = []
    dim_names = sorted(dynamic_values)
    scenario_source = "dim_buckets" if any(info["source"] == "bucket" for info in dynamic_values.values()) else "dim_typical"
    case_prefix = "bucket" if scenario_source == "dim_buckets" else "shape"
    for values in itertools.product(*(dynamic_values[name]["values"] for name in dim_names)):
        assignments = dict(zip(dim_names, values))
        scenario_overrides = {
            str(tensor_name): _shape_from_dim_assignments(tensor_map[str(tensor_name)], assignments)
            for tensor_name in node.get("inputs", [])
        }
        scenarios.append(
            ProfileShapeScenario(
                source=scenario_source,
                case_id=f"{case_prefix}_" + "_".join(f"{name}={assignments[name]}" for name in dim_names),
                dim_values=assignments,
                dim_sources={name: str(dynamic_values[name]["source"]) for name in dim_names},
                overrides=scenario_overrides,
            )
        )
    return scenarios


def _metadata_profile_shape_scenarios(
    graph: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> list[ProfileShapeScenario] | None:
    metadata = graph.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    profiling = metadata.get("profiling")
    if not isinstance(profiling, Mapping):
        return None
    raw_scenarios = profiling.get("shape_scenarios")
    if not isinstance(raw_scenarios, Sequence) or isinstance(raw_scenarios, (str, bytes, bytearray)):
        return None

    scenarios: list[ProfileShapeScenario] = []
    for index, raw_scenario in enumerate(raw_scenarios):
        if not isinstance(raw_scenario, Mapping):
            raise ValueError(
                f"Graph profiling metadata shape_scenarios[{index}] must be a mapping, "
                f"got {type(raw_scenario).__name__}"
            )
        raw_dim_values = raw_scenario.get("dim_values", {})
        if not isinstance(raw_dim_values, Mapping):
            raise ValueError(f"Graph profiling metadata shape_scenarios[{index}].dim_values must be a mapping")
        dim_values = {
            str(name): _positive_int(value, f"shape_scenarios[{index}].dim_values[{name!r}]")
            for name, value in raw_dim_values.items()
        }

        raw_overrides = raw_scenario.get("overrides", {})
        if not isinstance(raw_overrides, Mapping):
            raise ValueError(f"Graph profiling metadata shape_scenarios[{index}].overrides must be a mapping")
        overrides = {
            str(name): tuple(_positive_int(dim, f"shape_scenarios[{index}].overrides[{name!r}]") for dim in shape)
            for name, shape in raw_overrides.items()
        }
        for name, shape in overrides.items():
            tensor = tensor_map.get(name)
            if tensor is None:
                continue
            _runtime_tensor_shape(name, tensor, overrides, dim_values)
        scenarios.append(
            ProfileShapeScenario(
                source=str(raw_scenario.get("source", "graph_metadata")),
                case_id=str(raw_scenario.get("case_id", f"scenario_{index}")),
                dim_values=dim_values,
                dim_sources={name: "metadata" for name in dim_values},
                overrides=overrides,
            )
        )
    return scenarios or None


def _profile_dim_values(
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for tensor_name in node.get("inputs", []):
        tensor = tensor_map[str(tensor_name)]
        for dim in tensor.get("shape_spec", tensor["shape"]):
            for leaf in _named_dim_leaves(dim):
                _record_profile_dim_value(by_name, leaf)
    return {name: dict(info) for name, info in sorted(by_name.items())}


def _profile_dim_values_from_overrides(
    tensor_map: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Sequence[int]],
) -> tuple[dict[str, int], dict[str, str]]:
    assignments: dict[str, int] = {}
    sources: dict[str, str] = {}
    for tensor_name, shape in sorted(overrides.items()):
        tensor = tensor_map.get(str(tensor_name))
        if tensor is None:
            continue
        actual_shape = tuple(validate_runtime_shape(str(tensor_name), shape, tensor))
        shape_spec = tensor.get("shape_spec", tensor["shape"])
        for axis, (actual, dim_spec) in enumerate(zip(actual_shape, shape_spec)):
            if not isinstance(dim_spec, Mapping) or dim_spec.get("kind") != "dim":
                continue
            dim_name = str(dim_spec["name"])
            existing = assignments.get(dim_name)
            if existing is not None and existing != int(actual):
                raise ValueError(
                    f"Runtime profile override for tensor {tensor_name!r} axis {axis} assigns "
                    f"dynamic dimension {dim_name!r}={actual}, conflicting with earlier {existing}"
                )
            assignments[dim_name] = int(actual)
            sources[dim_name] = "runtime_override"
    return assignments, sources


def _record_profile_dim_value(by_name: dict[str, dict[str, Any]], dim: Mapping[str, Any]) -> None:
    name = str(dim["name"])
    buckets = tuple(int(bucket) for bucket in dim.get("buckets", ()))
    typical = None if dim.get("typical") is None else int(dim["typical"])
    values = _profile_dim_candidate_values(
        max_dim=int(dim["max"]),
        typical=typical,
        buckets=buckets,
    )
    signature = (
        int(dim["min"]),
        int(dim["max"]),
        int(dim.get("divisible_by", 1)),
        typical,
        buckets,
    )
    info = by_name.setdefault(
        name,
        {
            "signature": signature,
            "values": values,
            "source": "bucket" if buckets else ("typical" if typical is not None else "max"),
        },
    )
    if info["signature"] != signature:
        raise ValueError(f"Inconsistent profiling bucket metadata for dynamic dimension {name!r}")


def _profile_dim_candidate_values(
    *,
    max_dim: int,
    typical: int | None,
    buckets: Sequence[int],
) -> tuple[int, ...]:
    values = list(int(bucket) for bucket in buckets)
    if not values and typical is not None:
        values.append(int(typical))
    values.append(int(max_dim))
    return tuple(dict.fromkeys(values))


def _shape_from_dim_assignments(
    tensor: Mapping[str, Any],
    assignments: Mapping[str, int],
) -> tuple[int, ...]:
    shape = []
    for dim in tensor.get("shape_spec", tensor["shape"]):
        if isinstance(dim, Mapping) and dim.get("kind") == "dim":
            shape.append(int(assignments.get(str(dim["name"]), int(dim["max"]))))
        elif isinstance(dim, Mapping) and dim.get("kind") == "int_expr":
            shape.append(evaluate_symbolic_int(dim, assignments))
        elif isinstance(dim, Mapping):
            raise ValueError(f"Unsupported profiling shape dimension kind: {dim.get('kind')!r}")
        else:
            shape.append(int(dim))
    return _validate_profile_runtime_shape(tensor, shape, assignments)


def _validate_profile_runtime_shape(
    tensor: Mapping[str, Any],
    shape: Sequence[int],
    assignments: Mapping[str, int],
) -> tuple[int, ...]:
    actual_shape = tuple(int(dim) for dim in shape)
    shape_spec = tensor.get("shape_spec", tensor["shape"])
    if len(actual_shape) != len(shape_spec):
        raise ValueError(f"{tensor['name']} rank mismatch: got {len(actual_shape)}, expected {len(shape_spec)}")
    for axis, (actual, dim_spec) in enumerate(zip(actual_shape, shape_spec)):
        if isinstance(dim_spec, int):
            if actual != int(dim_spec):
                raise ValueError(f"{tensor['name']} axis {axis} has dim {actual}, expected static dim {dim_spec}")
            continue
        if dim_spec.get("kind") == "int_expr":
            expected = evaluate_symbolic_int(dim_spec, assignments)
            if expected <= 0:
                raise ValueError(f"{tensor['name']} axis {axis} symbolic expression evaluated to non-positive dim {expected}")
            if actual != expected:
                raise ValueError(f"{tensor['name']} axis {axis} has dim {actual}, expected symbolic dim {expected}")
            continue
        dim_name = str(dim_spec["name"])
        expected = assignments.get(dim_name)
        if expected is not None and actual != int(expected):
            raise ValueError(f"{tensor['name']} axis {axis} ({dim_name}) has dim {actual}, expected assigned dim {expected}")
        min_dim = int(dim_spec["min"])
        max_dim = int(dim_spec["max"])
        divisible_by = int(dim_spec.get("divisible_by", 1))
        if actual < min_dim or actual > max_dim:
            raise ValueError(f"{tensor['name']} axis {axis} ({dim_name}) has dim {actual}, expected [{min_dim}, {max_dim}]")
        if actual % divisible_by != 0:
            raise ValueError(f"{tensor['name']} axis {axis} ({dim_name}) has dim {actual}, expected divisible by {divisible_by}")
    return actual_shape


def _same_cutlass_native_profile_problem(
    first: GemmProfileWorkload | ConvProfileWorkload,
    second: GemmProfileWorkload | ConvProfileWorkload,
) -> bool:
    if first.kernel_library != second.kernel_library or first.kernel_library not in {"cutlass_gemm", "cutlass_bmm", "cutlass_conv"}:
        return False
    if first.kernel_library == "cutlass_conv":
        if not isinstance(first, ConvProfileWorkload) or not isinstance(second, ConvProfileWorkload):
            return False
        return (
            first.node_id == second.node_id
            and first.op == second.op
            and first.dtype == second.dtype
            and first.candidate_set_key == second.candidate_set_key
            and first.x_shape == second.x_shape
            and first.weight_shape == second.weight_shape
            and first.bias_shape == second.bias_shape
            and first.residual_shape == second.residual_shape
            and first.output_shape == second.output_shape
            and first.conv_config == second.conv_config
        )
    same = (
        first.node_id == second.node_id
        and first.op == second.op
        and first.dtype == second.dtype
        and first.candidate_set_key == second.candidate_set_key
        and first.m == second.m
        and first.n == second.n
        and first.k == second.k
        and first.split_k == second.split_k
        and first.shape_case_id == second.shape_case_id
        and first.alignment_context == second.alignment_context
        and first.bias_shape == second.bias_shape
        and first.residual_shapes == second.residual_shapes
    )
    if not same or first.kernel_library != "cutlass_bmm":
        return same
    return (
        first.batch_count == second.batch_count
        and first.batch_stride_a == second.batch_stride_a
        and first.batch_stride_b == second.batch_stride_b
        and first.batch_stride_d0 == second.batch_stride_d0
        and first.batch_stride_c == second.batch_stride_c
        and first.lda == second.lda
        and first.ldb == second.ldb
        and first.ldd0 == second.ldd0
        and first.ldc == second.ldc
    )


def _same_cutlass_gemm_profile_problem(
    first: GemmProfileWorkload | ConvProfileWorkload,
    second: GemmProfileWorkload | ConvProfileWorkload,
) -> bool:
    return _same_cutlass_native_profile_problem(first, second) and first.kernel_library == "cutlass_gemm"


def _named_dim_leaves(dim: Any) -> list[Mapping[str, Any]]:
    if not isinstance(dim, Mapping):
        return []
    kind = dim.get("kind")
    if kind == "dim":
        return [dim]
    if kind == "int_expr":
        return [*_named_dim_leaves(dim["lhs"]), *_named_dim_leaves(dim["rhs"])]
    raise ValueError(f"Unsupported profiling shape dimension kind: {kind!r}")


def _profile_result(
    workload: GemmProfileWorkload | ConvProfileWorkload,
    elapsed_ms: float,
    iterations: int,
    *,
    profile_key: str,
    status: str,
    reason: str = "only_candidate",
    workspace_nbytes: int | None = None,
    timing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_supported_profile_workload(workload, context="profile result")
    if isinstance(workload, ConvProfileWorkload):
        return _conv_profile_result(
            workload,
            elapsed_ms,
            iterations,
            profile_key=profile_key,
            status=status,
            reason=reason,
            workspace_nbytes=workspace_nbytes,
            timing=timing,
        )
    actual_workspace_nbytes = int(workspace_nbytes if workspace_nbytes is not None else workload.workspace_nbytes)
    timing_payload = _profile_timing([elapsed_ms], iterations=iterations) if timing is None else dict(timing)
    batch_count = int(workload.batch_count or 1)
    flops = 2 * batch_count * workload.m * workload.n * workload.k
    if workload.batch_count is not None:
        bytes_moved = dtype_nbytes(workload.dtype) * (
            int(np.prod(workload.a_shape, dtype=np.int64))
            + int(np.prod(workload.b_shape, dtype=np.int64))
            + int(np.prod(workload.output_shape, dtype=np.int64))
        )
    else:
        bytes_moved = dtype_nbytes(workload.dtype) * (
            workload.m * workload.k + workload.n * workload.k + workload.m * workload.n
        )
        if workload.bias_shape is not None:
            bytes_moved += dtype_nbytes(workload.dtype) * int(np.prod(workload.bias_shape, dtype=np.int64))
        for residual_shape in workload.residual_shapes:
            bytes_moved += dtype_nbytes(workload.dtype) * int(np.prod(residual_shape, dtype=np.int64))
    seconds = max(float(elapsed_ms) / 1000.0, 1e-12)
    tflops = float(flops / seconds / 1.0e12)
    gbps = float(bytes_moved / seconds / 1.0e9)
    payload = workload.to_json()
    candidate_result = dict(workload.candidate)
    candidate_result.update(
        {
            "candidate_id": workload.candidate_id,
            "split_k": workload.split_k,
            "workspace_nbytes": actual_workspace_nbytes,
            "avg_ms": float(timing_payload["mean_ms"]),
            "median_ms": float(timing_payload["median_ms"]),
            "mean_ms": float(timing_payload["mean_ms"]),
            "min_ms": float(timing_payload["min_ms"]),
            "max_ms": float(timing_payload["max_ms"]),
            "stddev_ms": float(timing_payload["stddev_ms"]),
            "standard_error_ms": float(timing_payload["standard_error_ms"]),
            "mean_ci95_ms": dict(timing_payload["mean_ci95_ms"]),
            "relative_stddev": float(timing_payload["relative_stddev"]),
            "gflops": float(tflops * 1000.0),
            "iterations": int(iterations),
            "repeats": int(timing_payload["repeats"]),
            "statistics_schema_version": int(timing_payload["statistics_schema_version"]),
        }
    )
    payload.update(
        {
            "profile_key": profile_key,
            "status": status,
            "shape": {
                "m": workload.m,
                "n": workload.n,
                "k": workload.k,
                "source": workload.shape_source,
                "case_id": workload.shape_case_id,
                "dims": dict(workload.dim_values),
                "dim_sources": dict(workload.dim_sources),
            },
            "tensors": {
                "a": workload.a_tensor,
                "b": workload.b_tensor,
                "bias": workload.bias_tensor,
                "c": workload.output_tensor,
            },
            "kernel_library": workload.kernel_library,
            "elapsed_ms": float(elapsed_ms),
            "iterations": int(iterations),
            "repeats": int(timing_payload["repeats"]),
            "timing": timing_payload,
            "split_k": workload.split_k,
            "workspace_nbytes": actual_workspace_nbytes,
            "flops": int(flops),
            "bytes": int(bytes_moved),
            "gflops": float(tflops * 1000.0),
            "tflops": tflops,
            "gbps": gbps,
            "candidates": [candidate_result],
            "selected": {
                "candidate_id": workload.candidate_id,
                "split_k": workload.split_k,
                "reason": reason,
            },
        }
    )
    if workload.batch_count is not None:
        payload["shape"]["batch_count"] = int(workload.batch_count)
        payload["batch_count"] = int(workload.batch_count)
        payload["batch_strides"] = {
            "a": int(workload.batch_stride_a or 0),
            "b": int(workload.batch_stride_b or 0),
            "c": int(workload.batch_stride_c or 0),
        }
        payload["leading_dimensions"] = {
            "a": int(workload.lda or 0),
            "b": int(workload.ldb or 0),
            "c": int(workload.ldc or 0),
        }
    return payload


def _conv_profile_result(
    workload: ConvProfileWorkload,
    elapsed_ms: float,
    iterations: int,
    *,
    profile_key: str,
    status: str,
    reason: str,
    workspace_nbytes: int | None,
    timing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    actual_workspace_nbytes = int(workspace_nbytes if workspace_nbytes is not None else workload.workspace_nbytes)
    timing_payload = _profile_timing([elapsed_ms], iterations=iterations) if timing is None else dict(timing)
    n, c, h, w = (int(dim) for dim in workload.x_shape)
    out_n, out_c, out_h, out_w = (int(dim) for dim in workload.output_shape)
    weight_o, _weight_i, kernel_h, kernel_w = (int(dim) for dim in workload.weight_shape)
    groups = int(workload.conv_config.get("groups", 1) or 1)
    flops = 2 * out_n * out_h * out_w * out_c * kernel_h * kernel_w * (c // groups)
    bytes_moved = dtype_nbytes(workload.dtype) * (
        int(np.prod(workload.x_shape, dtype=np.int64))
        + int(np.prod(workload.weight_shape, dtype=np.int64))
        + int(np.prod(workload.bias_shape, dtype=np.int64))
        + int(np.prod(workload.output_shape, dtype=np.int64))
    )
    seconds = max(float(elapsed_ms) / 1000.0, 1e-12)
    tflops = float(flops / seconds / 1.0e12)
    gbps = float(bytes_moved / seconds / 1.0e9)
    payload = workload.to_json()
    candidate_result = dict(workload.candidate)
    candidate_result.update(
        {
            "candidate_id": workload.candidate_id,
            "workspace_nbytes": actual_workspace_nbytes,
            "avg_ms": float(timing_payload["mean_ms"]),
            "median_ms": float(timing_payload["median_ms"]),
            "mean_ms": float(timing_payload["mean_ms"]),
            "min_ms": float(timing_payload["min_ms"]),
            "max_ms": float(timing_payload["max_ms"]),
            "stddev_ms": float(timing_payload["stddev_ms"]),
            "standard_error_ms": float(timing_payload["standard_error_ms"]),
            "mean_ci95_ms": dict(timing_payload["mean_ci95_ms"]),
            "relative_stddev": float(timing_payload["relative_stddev"]),
            "gflops": float(tflops * 1000.0),
            "iterations": int(iterations),
            "repeats": int(timing_payload["repeats"]),
            "statistics_schema_version": int(timing_payload["statistics_schema_version"]),
        }
    )
    shape_payload = {
        "source": workload.shape_source,
        "case_id": workload.shape_case_id,
        "dims": dict(workload.dim_values),
        "dim_sources": dict(workload.dim_sources),
        "n": n,
        "c": c,
        "h": h,
        "w": w,
        "out_n": out_n,
        "out_c": out_c,
        "out_h": out_h,
        "out_w": out_w,
        "weight_o": weight_o,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
    }
    payload.update(
        {
            "profile_key": profile_key,
            "status": status,
            "shape": shape_payload,
            "tensors": {
                "x": workload.x_tensor,
                "weight": workload.weight_tensor,
                "bias": workload.bias_tensor,
                "output": workload.output_tensor,
            },
            "kernel_library": workload.kernel_library,
            "elapsed_ms": float(elapsed_ms),
            "iterations": int(iterations),
            "repeats": int(timing_payload["repeats"]),
            "timing": timing_payload,
            "workspace_nbytes": actual_workspace_nbytes,
            "flops": int(flops),
            "bytes": int(bytes_moved),
            "gflops": float(tflops * 1000.0),
            "tflops": tflops,
            "gbps": gbps,
            "candidates": [candidate_result],
            "selected": {
                "candidate_id": workload.candidate_id,
                "reason": reason,
            },
        }
    )
    return payload


_CK_PROFILER_RETURN_CODE_REASONS = {
    1: "invalid pointer or profiler setup failure",
    2: "invalid problem dimensions, strides, or launch parameters",
    3: "HIP launch failed",
    4: "CK IsSupportedArgument rejected the problem for this candidate",
}


def _ck_profiler_return_code(elapsed_ms: float | None) -> int | None:
    if elapsed_ms is None or not math.isfinite(float(elapsed_ms)) or float(elapsed_ms) >= 0.0:
        return None
    magnitude = abs(float(elapsed_ms))
    rounded = int(round(magnitude))
    if rounded > 0 and math.isclose(magnitude, float(rounded), rel_tol=0.0, abs_tol=1.0e-3):
        return rounded
    return None


def _profile_failure_diagnostics(
    workload: GemmProfileWorkload | ConvProfileWorkload,
    *,
    elapsed_ms: float | None = None,
) -> dict[str, Any]:
    return_code = _ck_profiler_return_code(elapsed_ms)
    diagnostics: dict[str, Any] = {
        "kernel_library": workload.kernel_library,
        "op": workload.op,
        "dtype": workload.dtype,
        "candidate_id": workload.candidate_id,
        "candidate_config_key": workload.candidate_config_key,
        "candidate_set_id": workload.candidate_set_id,
        "profiler_symbol": workload.profiler_symbol,
        "kernel_symbol": workload.kernel_symbol,
        "launch_abi": str(workload.candidate.get("launch_abi", "")),
        "selection_predicate": dict(workload.candidate.get("selection_predicate", {})),
        "shape_case": {
            "source": workload.shape_source,
            "case_id": workload.shape_case_id,
            "dims": dict(workload.dim_values),
            "dim_sources": dict(workload.dim_sources),
        },
    }
    if return_code is not None:
        diagnostics["return_code"] = return_code
        diagnostics["return_reason"] = _CK_PROFILER_RETURN_CODE_REASONS.get(return_code, "unknown CK profiler failure")
        diagnostics["elapsed_ms"] = float(elapsed_ms if elapsed_ms is not None else 0.0)
    ck_config = workload.candidate.get("ck")
    if isinstance(ck_config, Mapping):
        diagnostics["ck"] = dict(ck_config)
    if isinstance(workload, ConvProfileWorkload):
        n, c, h, w = (int(dim) for dim in workload.x_shape)
        out_n, out_c, out_h, out_w = (int(dim) for dim in workload.output_shape)
        weight_o, weight_i, kernel_h, kernel_w = (int(dim) for dim in workload.weight_shape)
        diagnostics["problem"] = {
            "n": n,
            "c": c,
            "h": h,
            "w": w,
            "out_n": out_n,
            "out_c": out_c,
            "out_h": out_h,
            "out_w": out_w,
            "weight_o": weight_o,
            "weight_i": weight_i,
            "kernel_h": kernel_h,
            "kernel_w": kernel_w,
            "groups": int(workload.conv_config.get("groups", 1) or 1),
            "gemm_m": out_n * out_h * out_w,
            "gemm_n": out_c,
            "gemm_k": c * kernel_h * kernel_w,
        }
        diagnostics["conv_config"] = dict(workload.conv_config)
    else:
        problem = {
            "m": int(workload.m),
            "n": int(workload.n),
            "k": int(workload.k),
            "split_k": int(workload.split_k),
        }
        if workload.batch_count is not None:
            problem.update(
                {
                    "batch_count": int(workload.batch_count),
                    "batch_stride_a": int(workload.batch_stride_a or 0),
                    "batch_stride_b": int(workload.batch_stride_b or 0),
                    "batch_stride_c": int(workload.batch_stride_c or 0),
                    "lda": int(workload.lda or 0),
                    "ldb": int(workload.ldb or 0),
                    "ldc": int(workload.ldc or 0),
                }
            )
            if workload.residual_tensors:
                problem["batch_stride_d0"] = int(workload.batch_stride_d0 or 0)
                problem["ldd0"] = int(workload.ldd0 or 0)
        diagnostics["problem"] = problem
    return diagnostics


def _format_profile_failure_diagnostics(diagnostics: Mapping[str, Any]) -> str:
    fields = []
    if diagnostics.get("return_code") is not None:
        fields.append(f"return_code={diagnostics['return_code']} ({diagnostics.get('return_reason', 'unknown')})")
    problem = diagnostics.get("problem")
    if isinstance(problem, Mapping):
        fields.append(f"problem={json.dumps(dict(problem), sort_keys=True)}")
    ck_config = diagnostics.get("ck")
    if isinstance(ck_config, Mapping):
        config = ck_config.get("config")
        if isinstance(config, Mapping):
            fields.append(f"ck_config={json.dumps(dict(config), sort_keys=True)}")
    predicate = diagnostics.get("selection_predicate")
    if isinstance(predicate, Mapping) and predicate:
        fields.append(f"selection_predicate={json.dumps(dict(predicate), sort_keys=True)}")
    return "; ".join(fields) if fields else "no CK diagnostics available"


def _profile_failure_result(
    workload: GemmProfileWorkload | ConvProfileWorkload,
    iterations: int,
    *,
    profile_key: str,
    error: str,
) -> dict[str, Any]:
    payload = workload.to_json()
    diagnostics = _profile_failure_diagnostics(workload)
    candidate_result = dict(workload.candidate)
    candidate_result.update(
        {
            "candidate_id": workload.candidate_id,
            "workspace_nbytes": int(workload.workspace_nbytes),
            "iterations": int(iterations),
            "status": "failed",
            "error": str(error),
            "diagnostics": dict(diagnostics),
        }
    )
    payload.update(
        {
            "profile_key": profile_key,
            "status": "failed",
            "reason": "profiler_error",
            "error": str(error),
            "diagnostics": diagnostics,
            "kernel_library": workload.kernel_library,
            "iterations": int(iterations),
            "repeats": 0,
            "workspace_nbytes": int(workload.workspace_nbytes),
            "candidates": [candidate_result],
            "selected": {
                "candidate_id": workload.candidate_id,
                "split_k": getattr(workload, "split_k", 1),
                "reason": "profiler_error",
            },
        }
    )
    return payload


def _profile_report(
    artifact_dir: Path,
    manifest: Mapping[str, Any],
    kernel_manifest: Mapping[str, Any],
    codegen_plan: Mapping[str, Any],
    iterations: int,
    repeats: int,
    problems: Sequence[Mapping[str, Any]],
    summary: Mapping[str, int],
    *,
    context: Mapping[str, Any] | None = None,
    blocked_profile_items: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    problem_payloads = [dict(item) for item in problems]
    profile_context = dict(context or _profile_context(artifact_dir, manifest, codegen_plan))
    return {
        "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "artifact": str(artifact_dir.resolve()),
        "target": manifest["target"],
        "kernel_manifest_cache_key": kernel_manifest["cache_key"],
        "codegen_plan_cache_key": codegen_plan["cache_key"],
        "iterations": int(iterations),
        "fingerprint": profile_context["fingerprint"],
        "hardware": profile_context["fingerprint"]["hardware"],
        "hardware_cache_key": profile_context["fingerprint"]["hardware_key"],
        "libraries": profile_context["fingerprint"]["support_libraries"],
        "support_libraries_cache_key": profile_context["fingerprint"]["support_libraries_key"],
        "problems": problem_payloads,
        "workloads": problem_payloads,
        "blocked_profile_items": [dict(item) for item in blocked_profile_items],
        "repeats": int(repeats),
        "summary": dict(summary),
    }


def _profile_workload_samples(
    profiler: Any,
    workload: GemmProfileWorkload | ConvProfileWorkload,
    *,
    iterations: int,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[list[float], int, int]:
    effective_iterations = int(iterations)
    workspace_nbytes = int(workload.workspace_nbytes)
    profile_samples = getattr(profiler, "profile_samples", None)
    can_batch_repeats = callable(profile_samples) and int(repeats) > 1
    if can_batch_repeats:
        while True:
            pilot_samples_ms, pilot_workspace_nbytes = _profile_workload_sample_batch(
                profiler,
                workload,
                iterations=effective_iterations,
                repeats=1,
                rng=rng,
            )
            pilot_elapsed_ms = float(pilot_samples_ms[0])
            workspace_nbytes = max(workspace_nbytes, int(pilot_workspace_nbytes))
            adapted_iterations = _adaptive_profile_iterations(
                workload,
                requested_iterations=effective_iterations,
                elapsed_ms=pilot_elapsed_ms,
            )
            if adapted_iterations > effective_iterations:
                effective_iterations = adapted_iterations
                continue
            batch_samples_ms, batch_workspace_nbytes = _profile_workload_sample_batch(
                profiler,
                workload,
                iterations=effective_iterations,
                repeats=repeats,
                rng=rng,
            )
            workspace_nbytes = max(workspace_nbytes, int(batch_workspace_nbytes))
            restart = False
            for elapsed_ms in batch_samples_ms:
                adapted_iterations = _adaptive_profile_iterations(
                    workload,
                    requested_iterations=effective_iterations,
                    elapsed_ms=elapsed_ms,
                )
                if adapted_iterations > effective_iterations:
                    effective_iterations = adapted_iterations
                    restart = True
                    break
            if restart:
                continue
            return list(batch_samples_ms), workspace_nbytes, effective_iterations
    samples_ms: list[float] = []
    while len(samples_ms) < int(repeats):
        batch_samples_ms, sample_workspace_nbytes = _profile_workload_sample_batch(
            profiler,
            workload,
            iterations=effective_iterations,
            repeats=1,
            rng=rng,
        )
        elapsed_ms = float(batch_samples_ms[0])
        workspace_nbytes = max(workspace_nbytes, int(sample_workspace_nbytes))
        adapted_iterations = _adaptive_profile_iterations(
            workload,
            requested_iterations=effective_iterations,
            elapsed_ms=elapsed_ms,
        )
        if adapted_iterations > effective_iterations:
            effective_iterations = adapted_iterations
            samples_ms.clear()
            continue
        samples_ms.append(elapsed_ms)
    return samples_ms, workspace_nbytes, effective_iterations


def _profile_workload_sample_batch(
    profiler: Any,
    workload: GemmProfileWorkload | ConvProfileWorkload,
    *,
    iterations: int,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[list[float], int]:
    batch_samples = getattr(profiler, "profile_samples", None)
    if callable(batch_samples):
        samples_ms, workspace_nbytes = batch_samples(
            workload,
            iterations=int(iterations),
            repeats=int(repeats),
            rng=rng,
        )
        normalized = [float(sample) for sample in samples_ms]
        if len(normalized) != int(repeats):
            raise RuntimeError(
                f"Profiler returned {len(normalized)} samples for requested repeats={int(repeats)}"
            )
        return normalized, int(workspace_nbytes)
    if int(repeats) != 1:
        raise RuntimeError("Profiler does not support batched repeats")
    elapsed_ms, workspace_nbytes = profiler.profile(workload, iterations=int(iterations), rng=rng)
    return [float(elapsed_ms)], int(workspace_nbytes)


def _adaptive_profile_iterations(
    workload: GemmProfileWorkload | ConvProfileWorkload,
    *,
    requested_iterations: int,
    elapsed_ms: float,
) -> int:
    requested = int(requested_iterations)
    if str(workload.kernel_library) not in {"ck_gemm", "ck_bmm", "ck_conv"}:
        return requested
    elapsed = float(elapsed_ms)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        return requested
    total_sample_ms = elapsed * requested
    if total_sample_ms >= PROFILE_ADAPTIVE_MIN_TOTAL_SAMPLE_MS:
        return requested
    target_iterations = int(math.ceil(PROFILE_ADAPTIVE_MIN_TOTAL_SAMPLE_MS / elapsed))
    return max(requested, min(PROFILE_ADAPTIVE_MAX_ITERATIONS, target_iterations))


def _adaptive_profile_iterations_payload(
    *,
    requested_iterations: int,
    effective_iterations: int,
) -> dict[str, Any]:
    return {
        "policy": "min_total_sample_ms_v1",
        "requested_iterations": int(requested_iterations),
        "effective_iterations": int(effective_iterations),
        "min_total_sample_ms": PROFILE_ADAPTIVE_MIN_TOTAL_SAMPLE_MS,
        "max_iterations": PROFILE_ADAPTIVE_MAX_ITERATIONS,
    }


def _write_profile_report(report: Mapping[str, Any], artifact_dir: Path, output: str | Path | None) -> None:
    report_path = Path(output) if output is not None else artifact_dir / "debug" / "profile_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, dict(report))


def _profile_libraries(
    artifact_dir: Path,
    manifest: Mapping[str, Any],
    codegen_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    files = manifest["files"]
    by_name: dict[str, dict[str, Any]] = {}

    def merge(name: str, **fields: Any) -> None:
        entry = by_name.setdefault(name, {"name": name})
        for key, value in fields.items():
            if value not in (None, ""):
                entry[key] = value

    for item in codegen_plan.get("external_support_libraries", []):
        name = str(item["name"])
        cache_dir = Path(str(item.get("cache_dir", ""))) if item.get("cache_dir") else None
        cache_library = _cache_library_path(item, cache_dir)
        support_manifest = _support_library_manifest_path(name, cache_dir)
        support_payload = _read_optional_json(support_manifest) if support_manifest else {}
        manifest_fields = _support_manifest_fields(support_payload)
        manifest_fields.setdefault("build_mode", item.get("build_mode"))
        manifest_fields.setdefault("used_candidate_plan_key", item.get("used_candidate_plan_key"))
        merge(
            name,
            cache_dir=str(cache_dir) if cache_dir else None,
            cache_library=str(cache_library) if cache_library else None,
            cache_library_sha256=_file_sha256(cache_library) if cache_library else None,
            cache_modules=_cache_modules(item, cache_dir),
            manifest=str(support_manifest) if support_manifest and support_manifest.exists() else None,
            **manifest_fields,
        )
    return [by_name[name] for name in sorted(by_name)]


def _profile_context(
    artifact_dir: Path,
    manifest: Mapping[str, Any],
    codegen_plan: Mapping[str, Any],
) -> dict[str, Any]:
    libraries = _profile_libraries(artifact_dir, manifest, codegen_plan)
    hardware = _hardware_fingerprint(manifest["target"])
    hardware_cache_payload = _hardware_cache_payload(hardware)
    support_libraries_cache_payload = _support_libraries_cache_payload(libraries)
    hardware_key = _fingerprint_key(hardware_cache_payload)
    support_libraries_key = _fingerprint_key(support_libraries_cache_payload)
    fingerprint_key = _fingerprint_key(
        {
            "hardware_key": hardware_key,
            "support_libraries_key": support_libraries_key,
        }
    )
    return {
        "fingerprint": {
            "schema_version": 1,
            "key": fingerprint_key,
            "hardware_key": hardware_key,
            "support_libraries_key": support_libraries_key,
            "hardware": hardware,
            "support_libraries": libraries,
        },
        "hardware_cache_payload": hardware_cache_payload,
        "support_libraries_cache_payload": support_libraries_cache_payload,
        "support_libraries_by_name": {str(item.get("name", "")): dict(item) for item in libraries if item.get("name")},
    }


def _hardware_fingerprint(target: Mapping[str, Any]) -> dict[str, Any]:
    backend = str(target.get("name", "cuda"))
    if backend == "rocm":
        return _rocm_hardware_fingerprint(target)
    return _cuda_hardware_fingerprint(target)


def _cache_library_path(item: Mapping[str, Any], cache_dir: Path | None) -> Path | None:
    if cache_dir is None:
        return None
    library = str(item.get("library", ""))
    if not library:
        return None
    candidate = cache_dir / library
    return candidate if candidate.exists() else None


def _cache_modules(item: Mapping[str, Any], cache_dir: Path | None) -> list[dict[str, Any]] | None:
    if cache_dir is None:
        return None
    modules = item.get("modules")
    if not isinstance(modules, Sequence) or isinstance(modules, (str, bytes)):
        return None
    result = []
    for module in modules:
        if not isinstance(module, Mapping):
            continue
        relative = str(module.get("library") or module.get("archive") or "")
        if not relative:
            continue
        path = cache_dir / relative
        if path.exists():
            entry = {**dict(module), "sha256": _file_sha256(path)}
            profiler_stem = str(module.get("profiler_stem", ""))
            if profiler_stem:
                bind_path = _first_matching_path(cache_dir / "lib", f"{profiler_stem}_bind")
                exe_path = _first_matching_path(cache_dir / "lib", profiler_stem, exclude_substring="_bind")
                if bind_path is not None:
                    entry["profiler_bind"] = bind_path.name
                    entry["profiler_bind_sha256"] = _file_sha256(bind_path)
                if exe_path is not None:
                    entry["profiler_executable"] = exe_path.name
                    entry["profiler_executable_sha256"] = _file_sha256(exe_path)
            result.append(entry)
    return result


def _first_matching_path(directory: Path, stem: str, *, exclude_substring: str | None = None) -> Path | None:
    patterns = [f"{stem}{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES]
    patterns.append(f"{stem}*")
    seen: set[Path] = set()
    for pattern in patterns:
        for candidate in sorted(directory.glob(pattern)):
            if candidate in seen:
                continue
            seen.add(candidate)
            if exclude_substring is not None and exclude_substring in candidate.name:
                continue
            if candidate.is_file():
                return candidate
    return None


def _support_library_manifest_path(name: str, cache_dir: Path | None) -> Path | None:
    if cache_dir is None:
        return None
    if name == "cutlass_gemm":
        return cache_dir / "lib" / "cutlass_gemm_manifest.json"
    if name == "cutlass_bmm":
        return cache_dir / "lib" / "cutlass_bmm_manifest.json"
    return cache_dir / "lib" / f"{name}_manifest.json"


def _support_manifest_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in (
        "schema_version",
        "provider",
        "source_sha256",
        "library_sha256",
        "modules",
        "source_manifest",
        "cache_key",
        "provenance_key",
        "build_fingerprint",
        "family_cache_key",
        "external_kernel_plan_cache_key",
    ):
        if key in payload:
            fields[f"manifest_{key}" if key in {"schema_version", "cache_key"} else key] = payload[key]
    target = payload.get("target")
    if isinstance(target, Mapping):
        fields["manifest_target"] = dict(target)
    for key in ("compile", "provenance"):
        if isinstance(payload.get(key), Mapping):
            fields[key] = dict(payload[key])
    if "used_candidate_plan_key" in payload:
        fields["used_candidate_plan_key"] = payload["used_candidate_plan_key"]
    return fields


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cuda_hardware_fingerprint(target: Mapping[str, Any]) -> dict[str, Any]:
    devices = _query_nvidia_smi_devices()
    return {
        "backend": "cuda",
        "target_arch": str(target.get("arch", "")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": "available" if devices else "unavailable",
        "devices": devices,
        "nvcc": _query_nvcc_version(),
    }


def _rocm_hardware_fingerprint(target: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "backend": "rocm",
        "target_arch": str(target.get("arch", "")),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
        "gpu_device_ordinal": os.environ.get("GPU_DEVICE_ORDINAL", ""),
        "devices": _query_rocm_smi_devices(),
        "hipconfig": _query_tool_version("hipconfig", "--version"),
        "rocminfo": _query_tool_version("rocminfo", "--version"),
    }


def _query_rocm_smi_devices() -> list[dict[str, Any]]:
    if shutil.which("rocm-smi") is None:
        return []
    proc = _run_capture(["rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "vram"], timeout=2.0)
    if proc is None or proc.returncode != 0:
        return []
    devices = []
    current: dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        if "GPU[" in line:
            if current:
                devices.append(current)
            index_match = re.search(r"GPU\[(\d+)\]", line)
            current = {"index": _parse_int(index_match.group(1)) if index_match else None, "raw": line.strip()}
        elif current and line.strip():
            current.setdefault("raw_extra", []).append(line.strip())
    if current:
        devices.append(current)
    return devices


def _query_tool_version(tool: str, *args: str) -> dict[str, str]:
    if shutil.which(tool) is None:
        return {"available": "false"}
    proc = _run_capture([tool, *args], timeout=2.0)
    if proc is None or proc.returncode != 0:
        return {"available": "false"}
    first_line = next((line.strip() for line in proc.stdout.splitlines() if line.strip()), "")
    payload = {"available": "true"}
    if first_line:
        payload["version"] = first_line
    return payload


def _query_nvidia_smi_devices() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    proc = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout=2.0,
    )
    if proc is None or proc.returncode != 0:
        return []
    devices = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        devices.append(
            {
                "index": _parse_int(parts[0]),
                "name": parts[1],
                "compute_capability": parts[2],
                "driver_version": parts[3],
                "memory_total_mib": _parse_int(parts[4]),
            }
        )
    return devices


def _query_nvcc_version() -> dict[str, str]:
    if shutil.which("nvcc") is None:
        return {"available": "false"}
    proc = _run_capture(["nvcc", "--version"], timeout=2.0)
    if proc is None or proc.returncode != 0:
        return {"available": "false"}
    release_match = re.search(r"release\s+([0-9.]+)", proc.stdout)
    build_match = re.search(r"V([0-9.]+)", proc.stdout)
    payload = {"available": "true"}
    if release_match:
        payload["release"] = release_match.group(1)
    if build_match:
        payload["build"] = build_match.group(1)
    return payload


def _run_capture(cmd: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _hardware_cache_payload(hardware: Mapping[str, Any]) -> dict[str, Any]:
    devices = []
    for item in hardware.get("devices", []):
        if not isinstance(item, Mapping):
            continue
        devices.append(
            {
                "name": item.get("name"),
                "compute_capability": item.get("compute_capability"),
                "driver_version": item.get("driver_version"),
                "memory_total_mib": item.get("memory_total_mib"),
                "raw": item.get("raw"),
            }
        )
    payload = {
        "backend": hardware.get("backend"),
        "target_arch": hardware.get("target_arch"),
        "devices": devices,
    }
    if hardware.get("backend") == "rocm":
        payload.update(
            {
                "hip_visible_devices": hardware.get("hip_visible_devices", ""),
                "rocr_visible_devices": hardware.get("rocr_visible_devices", ""),
                "gpu_device_ordinal": hardware.get("gpu_device_ordinal", ""),
                "hipconfig": dict(hardware.get("hipconfig", {})) if isinstance(hardware.get("hipconfig"), Mapping) else {},
                "rocminfo": dict(hardware.get("rocminfo", {})) if isinstance(hardware.get("rocminfo"), Mapping) else {},
            }
        )
    else:
        payload.update(
            {
                "cuda_visible_devices": hardware.get("cuda_visible_devices", ""),
                "nvcc": dict(hardware.get("nvcc", {})) if isinstance(hardware.get("nvcc"), Mapping) else {},
            }
        )
    return payload


def _support_libraries_cache_payload(libraries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payloads = []
    for library in libraries:
        payloads.append(
            {
                "name": library.get("name"),
                "build_mode": library.get("build_mode"),
                "artifact_sha256": library.get("artifact_sha256"),
                "artifact_modules": library.get("artifact_modules"),
                "modules": library.get("modules"),
                "cache_modules": library.get("cache_modules"),
                "cache_library_sha256": library.get("cache_library_sha256"),
                "source_sha256": library.get("source_sha256"),
                "library_sha256": library.get("library_sha256"),
                "source_manifest": library.get("source_manifest"),
                "provenance_key": library.get("provenance_key"),
                "build_fingerprint": library.get("build_fingerprint"),
                "family_cache_key": library.get("family_cache_key"),
                "used_candidate_plan_key": library.get("used_candidate_plan_key"),
                "manifest_cache_key": library.get("manifest_cache_key"),
                "manifest_target": library.get("manifest_target"),
                "external_kernel_plan_cache_key": library.get("external_kernel_plan_cache_key"),
            }
        )
    return sorted(payloads, key=lambda item: str(item.get("name", "")))


def _fingerprint_key(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _profile_cache_lookup(
    workload: GemmProfileWorkload | ConvProfileWorkload,
    manifest: Mapping[str, Any],
    kernel_manifest: Mapping[str, Any],
    codegen_plan: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> ProfileCacheLookup:
    del kernel_manifest
    _require_supported_profile_workload(workload, context="profile cache key")
    profile_context = context or _profile_context(Path("."), manifest, codegen_plan)
    target_payload = dict(manifest["target"])
    support_payload = _profile_support_library_fingerprint_payload(workload, profile_context)
    provider_problem_payload = _profile_provider_problem_payload(workload)
    candidate_payload = _profile_candidate_identity_payload(workload)
    variant_payload = _profile_variant_payload(workload)
    key_payload = {
        "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": target_payload,
        "hardware_fingerprint_key": profile_context["fingerprint"]["hardware_key"],
        "support_library": support_payload,
        "provider_problem": provider_problem_payload,
        "candidate": candidate_payload,
        "profile_variant": variant_payload,
    }
    profile_key = _profile_key(key_payload)
    provider_problem_key = _fingerprint_key(provider_problem_payload)
    problem_key = _fingerprint_key(provider_problem_payload["problem"])
    semantics_key = _fingerprint_key(provider_problem_payload["semantics"])
    variant_key = _fingerprint_key(variant_payload)
    metadata = {
        "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target_name": str(target_payload.get("name", "")),
        "target_arch": str(target_payload.get("arch", "")),
        "hardware_fingerprint_key": profile_context["fingerprint"]["hardware_key"],
        "support_library_name": str(support_payload.get("name", workload.kernel_library)),
        "support_fingerprint_key": _fingerprint_key(support_payload),
        "kernel_library": workload.kernel_library,
        "op_family": _profile_op_family(workload),
        "op": workload.op,
        "dtype": workload.dtype,
        "candidate_set_key": workload.candidate_set_key,
        "candidate_id": workload.candidate_id,
        "candidate_config_key": workload.candidate_config_key,
        "provider_problem_key": provider_problem_key,
        "problem_key": problem_key,
        "semantics_key": semantics_key,
        "variant_key": variant_key,
    }
    return ProfileCacheLookup(profile_key=profile_key, key_payload=key_payload, metadata=metadata)


def _profile_op_family(workload: GemmProfileWorkload | ConvProfileWorkload) -> str:
    if isinstance(workload, ConvProfileWorkload):
        return "conv"
    if workload.batch_count is not None:
        return "bmm"
    return "gemm"


def _profile_support_library_fingerprint_payload(
    workload: GemmProfileWorkload | ConvProfileWorkload,
    profile_context: Mapping[str, Any],
) -> dict[str, Any]:
    libraries_by_name = profile_context.get("support_libraries_by_name", {})
    if not isinstance(libraries_by_name, Mapping):
        libraries_by_name = {}
    library = libraries_by_name.get(workload.kernel_library)
    if not isinstance(library, Mapping):
        return {
            "name": workload.kernel_library,
            "support_libraries_fingerprint_key": profile_context["fingerprint"]["support_libraries_key"],
        }
    payload: dict[str, Any] = {
        "name": library.get("name"),
        "build_mode": library.get("build_mode"),
        "source_sha256": library.get("source_sha256"),
        "library_sha256": library.get("library_sha256"),
        "source_manifest": library.get("source_manifest"),
        "provenance_key": library.get("provenance_key"),
        "build_fingerprint": library.get("build_fingerprint"),
        "family_cache_key": library.get("family_cache_key"),
        "external_kernel_plan_cache_key": library.get("external_kernel_plan_cache_key"),
        "manifest_cache_key": library.get("manifest_cache_key"),
        "manifest_target": library.get("manifest_target"),
        "compile": dict(library.get("compile", {})) if isinstance(library.get("compile"), Mapping) else {},
        "provenance": dict(library.get("provenance", {})) if isinstance(library.get("provenance"), Mapping) else {},
    }
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _profile_provider_problem_payload(workload: GemmProfileWorkload | ConvProfileWorkload) -> dict[str, Any]:
    if isinstance(workload, ConvProfileWorkload):
        semantics: dict[str, Any] = {
            "layouts": dict(workload.candidate.get("layouts", {})),
            "semantic_layout": dict(workload.semantic_layout),
            "provider_layout": dict(workload.provider_layout),
            "layout_translation": dict(workload.layout_translation),
            "weight_transform": dict(workload.weight_transform),
            "conv_config": dict(workload.conv_config),
        }
        if workload.source_op is not None:
            semantics["source_op"] = workload.source_op
        if workload.bias_mode is not None:
            semantics["bias_mode"] = workload.bias_mode
        return {
            "provider": {
                "kernel_library": workload.kernel_library,
                "op_family": "conv",
                "op": workload.op,
                "dtype": workload.dtype,
            },
            "semantics": semantics,
            "problem": {
                "input": list(workload.x_shape),
                "weight": list(workload.weight_shape),
                "bias": list(workload.bias_shape),
                "output": list(workload.output_shape),
            },
        }
    semantics = {
        "layouts": dict(workload.candidate.get("layouts", {})),
        "epilogue": workload.candidate.get("epilogue"),
        "epilogue_config": workload.candidate.get("epilogue_config"),
        "alignment_context": dict(workload.alignment_context),
    }
    problem: dict[str, Any] = {
        "m": int(workload.m),
        "n": int(workload.n),
        "k": int(workload.k),
    }
    if workload.batch_count is not None:
        problem["batch_count"] = int(workload.batch_count)
    if workload.lda is not None or workload.ldb is not None or workload.ldc is not None:
        problem["leading_dimensions"] = {
            "a": int(workload.lda or 0),
            "b": int(workload.ldb or 0),
            "c": int(workload.ldc or 0),
        }
    if workload.batch_count is not None:
        problem["batch_strides"] = {
            "a": int(workload.batch_stride_a or 0),
            "b": int(workload.batch_stride_b or 0),
            "c": int(workload.batch_stride_c or 0),
        }
        if workload.residual_tensors:
            problem["leading_dimensions"]["d0"] = int(workload.ldd0 or 0)
            problem["batch_strides"]["d0"] = int(workload.batch_stride_d0 or 0)
    return {
        "provider": {
            "kernel_library": workload.kernel_library,
            "op_family": _profile_op_family(workload),
            "op": workload.op,
            "dtype": workload.dtype,
        },
        "semantics": semantics,
        "problem": problem,
    }


def _profile_candidate_identity_payload(workload: GemmProfileWorkload | ConvProfileWorkload) -> dict[str, Any]:
    return {
        "candidate_set_key": workload.candidate_set_key,
        "candidate_id": workload.candidate_id,
        "candidate_config_key": workload.candidate_config_key,
    }


def _profile_variant_payload(workload: GemmProfileWorkload | ConvProfileWorkload) -> dict[str, Any]:
    if isinstance(workload, ConvProfileWorkload):
        return {
            "kind": str(workload.candidate.get("status", "")),
            "profiler_status": str(workload.candidate.get("profiler_status", "")),
        }
    return {"split_k": int(workload.split_k)}


def _profile_key(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _cache_entry_satisfies(
    entry: Mapping[str, Any],
    *,
    iterations: int,
    repeats: int,
    key_payload: Mapping[str, Any] | None = None,
) -> bool:
    if not isinstance(entry, Mapping):
        return False
    if key_payload is not None:
        cached_key = entry.get("key")
        if not isinstance(cached_key, Mapping) or dict(cached_key) != dict(key_payload):
            return False
    entry_iterations = _cache_positive_int(entry.get("iterations"), default=0)
    entry_repeats = _cache_positive_int(entry.get("repeats"), default=1)
    timing = entry.get("timing")
    if not isinstance(timing, Mapping):
        return False
    timing_repeats = _cache_positive_int(timing.get("repeats", timing.get("sample_count")), default=0)
    sample_count = _cache_positive_int(timing.get("sample_count", timing_repeats), default=0)
    if entry_iterations < 0 or entry_repeats < 0 or timing_repeats < 0 or sample_count < 0:
        return False
    return (
        entry.get("statistics_schema_version") == PROFILE_STATISTICS_SCHEMA_VERSION
        and timing.get("statistics_schema_version") == PROFILE_STATISTICS_SCHEMA_VERSION
        and entry_iterations >= int(iterations)
        and entry_repeats >= int(repeats)
        and timing_repeats >= int(repeats)
        and sample_count >= int(repeats)
    )


def _cache_positive_int(value: Any, *, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _cache_entry(
    workload: GemmProfileWorkload | ConvProfileWorkload,
    result: Mapping[str, Any],
    key_payload: Mapping[str, Any],
) -> dict[str, Any]:
    _require_supported_profile_workload(workload, context="profile cache write")
    candidate = result["candidates"][0]
    timing = dict(result.get("timing", {})) if isinstance(result.get("timing"), Mapping) else {}
    payload = {
        "profile_key": result["profile_key"],
        "key": dict(key_payload),
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "op": workload.op,
        "dtype": workload.dtype,
        "kernel_library": workload.kernel_library,
        "shape": dict(result["shape"]),
        "kernel_symbol": workload.kernel_symbol,
        "profiler_symbol": workload.profiler_symbol,
        "best_candidate_id": workload.candidate_id,
        "candidate_set_id": workload.candidate_set_id,
        "candidate_set_key": workload.candidate_set_key,
        "candidate_config_key": workload.candidate_config_key,
        "launch_abi": str(candidate.get("launch_abi") or workload.candidate.get("launch_abi") or ""),
        "symbol_id": candidate.get("symbol_id") or workload.candidate.get("symbol_id"),
        "workspace_nbytes": int(result.get("workspace_nbytes", workload.workspace_nbytes)),
        "elapsed_ms": float(result.get("elapsed_ms", candidate["avg_ms"])),
        "avg_ms": float(candidate["avg_ms"]),
        "timing": timing,
        "statistics_schema_version": int(
            candidate.get("statistics_schema_version", timing.get("statistics_schema_version", 0)) or 0
        ),
        "gflops": float(candidate["gflops"]),
        "iterations": int(candidate["iterations"]),
        "repeats": int(candidate.get("repeats", timing.get("repeats", 1)) or 1),
    }
    if isinstance(workload, ConvProfileWorkload):
        payload.update(
            {
                "layout_translation": dict(workload.layout_translation),
                "weight_transform": dict(workload.weight_transform),
                "conv_config": dict(workload.conv_config),
            }
        )
    else:
        payload["alignment_context"] = dict(result.get("alignment_context", workload.alignment_context))
        payload["split_k"] = int(result.get("split_k", workload.split_k))
    return payload


def _profile_result_from_cache(
    workload: GemmProfileWorkload | ConvProfileWorkload,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    _require_supported_profile_workload(workload, context="profile cache read")
    timing = dict(entry.get("timing", {})) if isinstance(entry.get("timing"), Mapping) and entry.get("timing") else None
    return _profile_result(
        workload,
        float(entry.get("elapsed_ms", entry["avg_ms"])),
        int(entry["iterations"]),
        profile_key=str(entry["profile_key"]),
        status="cached",
        reason="cache_hit",
        workspace_nbytes=int(entry.get("workspace_nbytes", workload.workspace_nbytes) or 0),
        timing=timing,
    )


def _profile_timing(samples_ms: Sequence[float], *, iterations: int) -> dict[str, Any]:
    samples = [float(sample) for sample in samples_ms]
    if not samples:
        raise ValueError("profile timing requires at least one sample")
    values = np.asarray(samples, dtype=np.float64)
    mean_ms = float(np.mean(values))
    stddev_ms = float(np.std(values, ddof=1)) if len(samples) > 1 else 0.0
    standard_error_ms = float(stddev_ms / math.sqrt(len(samples))) if samples else 0.0
    ci_half_width = float(PROFILE_CONFIDENCE_Z_SCORE * standard_error_ms)
    return {
        "statistics_schema_version": PROFILE_STATISTICS_SCHEMA_VERSION,
        "samples_ms": samples,
        "sample_count": len(samples),
        "repeats": len(samples),
        "iterations_per_sample": int(iterations),
        "median_ms": float(np.median(values)),
        "mean_ms": mean_ms,
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "stddev_ms": stddev_ms,
        "standard_error_ms": standard_error_ms,
        "mean_ci95_ms": {
            "low": float(mean_ms - ci_half_width),
            "high": float(mean_ms + ci_half_width),
            "half_width": ci_half_width,
            "confidence_level": PROFILE_CONFIDENCE_LEVEL,
            "z_score": PROFILE_CONFIDENCE_Z_SCORE,
        },
        "relative_stddev": float(stddev_ms / mean_ms) if mean_ms > 0.0 else 0.0,
    }


def build_execution_plan(report: Mapping[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = {}
    for problem in report.get("problems", []):
        if not isinstance(problem, Mapping) or problem.get("status") not in {"ok", "cached"}:
            continue
        candidate_set_key = problem.get("candidate_set_key")
        if not candidate_set_key:
            continue
        shape_payload = _execution_plan_problem_shape(problem)
        if shape_payload is None:
            continue
        key = (
            str(problem.get("node_id", "")),
            str(problem.get("op", "")),
            str(problem.get("dtype", "")),
            str(candidate_set_key),
            canonical_json(shape_payload),
        )
        if key[0] and key[1] and key[2] and key[3]:
            groups.setdefault(key, []).append(problem)

    candidate_selections = [_selection_from_group(key, entries) for key, entries in sorted(groups.items())]
    selections = [
        selection
        for selection in candidate_selections
        if bool(selection.get("confidence", {}).get("confident", True))
    ]
    low_confidence_selections = [
        selection
        for selection in candidate_selections
        if not bool(selection.get("confidence", {}).get("confident", True))
    ]
    static_selections, conflicts = _static_execution_selections(selections)
    plan = {
        "schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
        "kind": "dinoml.execution_plan",
        "profile_report_schema_version": report.get("schema_version"),
        "profile_cache_schema_version": report.get("profile_cache_schema_version"),
        "target": dict(report.get("target", {})) if isinstance(report.get("target"), Mapping) else {},
        "artifact": report.get("artifact"),
        "kernel_manifest_cache_key": report.get("kernel_manifest_cache_key"),
        "codegen_plan_cache_key": report.get("codegen_plan_cache_key"),
        "fingerprint": dict(report.get("fingerprint", {})) if isinstance(report.get("fingerprint"), Mapping) else {},
        "hardware_cache_key": report.get("hardware_cache_key"),
        "support_libraries_cache_key": report.get("support_libraries_cache_key"),
        "selection_policy": "lowest_median_elapsed_ms_per_node_shape",
        "selection_confidence_policy": _selection_confidence_policy(),
        "static_selection_policy": "unique_selected_candidate_per_op_dtype_candidate_set",
        "selections": selections,
        "low_confidence_selections": low_confidence_selections,
        "static_selections": static_selections,
        "conflicts": conflicts,
        "summary": {
            "selection_count": len(selections),
            "low_confidence_count": len(low_confidence_selections),
            "static_selection_count": len(static_selections),
            "conflict_count": len(conflicts),
        },
    }
    plan["execution_plan_key"] = hashlib.sha256(
        canonical_json({key: value for key, value in plan.items() if key != "execution_plan_key"}).encode("utf-8")
    ).hexdigest()
    return plan


def _selection_from_group(
    key: tuple[str, str, str, str, str],
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ranked = sorted(
        entries,
        key=lambda item: (
            _problem_elapsed_ms(item),
            str(item.get("candidate_id", "")),
            _problem_split_k(item),
        ),
    )
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    candidate = _result_candidate(best)
    shape = dict(best.get("shape", {})) if isinstance(best.get("shape"), Mapping) else {}
    split_k = _problem_split_k(best)
    selection_shape = _execution_plan_problem_shape(best) or shape
    profile_variant = (
        dict(best.get("profile_variant", {}))
        if isinstance(best.get("profile_variant"), Mapping)
        else {"split_k": split_k}
    )
    if best.get("kernel_library") not in {"cutlass_conv", "ck_conv"}:
        profile_variant["split_k"] = split_k
    return {
        "selection_key": hashlib.sha256(
            canonical_json(
                {
                    "node_id": key[0],
                    "op": key[1],
                    "dtype": key[2],
                    "candidate_set_key": key[3],
                    "shape": selection_shape,
                }
            ).encode("utf-8")
        ).hexdigest(),
        "node_id": key[0],
        "op": key[1],
        "dtype": key[2],
        "candidate_set_id": best.get("candidate_set_id"),
        "candidate_set_key": key[3],
        "selected_candidate_id": best.get("candidate_id"),
        "candidate_config_key": best.get("candidate_config_key") or candidate.get("candidate_config_key"),
        "kernel_symbol": best.get("kernel_symbol") or candidate.get("kernel_symbol"),
        "profiler_symbol": best.get("profiler_symbol") or candidate.get("profiler_symbol"),
        "shape": shape,
        "workspace_nbytes": _problem_workspace_nbytes(best),
        "split_k": split_k,
        "profile_variant": profile_variant,
        "kernel_library": best.get("kernel_library"),
        "avg_ms": _problem_elapsed_ms(best),
        "gflops": float(best.get("gflops", 0.0)),
        "iterations": int(best.get("iterations", 0) or 0),
        "profile_key": best.get("profile_key"),
        "status": best.get("status"),
        **(
            {"source_op": best.get("source_op"), "bias_mode": best.get("bias_mode")}
            if best.get("source_op") is not None
            else {}
        ),
        "confidence": _selection_confidence(best, runner_up),
    }


def _execution_plan_problem_shape(problem: Mapping[str, Any]) -> dict[str, Any] | None:
    if problem.get("kernel_library") in {"cutlass_conv", "ck_conv"}:
        shape = problem.get("shape")
        if not isinstance(shape, Mapping):
            return None
        required = ("n", "c", "h", "w", "out_n", "out_c", "out_h", "out_w", "kernel_h", "kernel_w")
        payload = {}
        for field in required:
            value = shape.get(field)
            if type(value) is not int or value <= 0:
                return None
            payload[field] = int(value)
        conv_config = problem.get("conv")
        if isinstance(conv_config, Mapping):
            payload["conv_config"] = {
                "stride": list(conv_config.get("stride", ())),
                "padding": list(conv_config.get("padding", ())),
                "dilation": list(conv_config.get("dilation", ())),
                "groups": conv_config.get("groups"),
            }
        return payload
    shape = problem.get("shape", {})
    batch_count = 0
    if isinstance(shape, Mapping) and shape.get("batch_count") is not None:
        batch_count = int(shape.get("batch_count", 0) or 0)
    elif problem.get("batch_count") is not None:
        batch_count = int(problem.get("batch_count", 0) or 0)
    m = int(problem.get("m", 0))
    n = int(problem.get("n", 0))
    k = int(problem.get("k", 0))
    if m <= 0 or n <= 0 or k <= 0:
        return None
    payload = {"m": m, "n": n, "k": k}
    if batch_count > 0:
        payload["batch_count"] = batch_count
    return payload


def _selection_confidence_policy() -> dict[str, Any]:
    return {
        "name": "confidence_interval_margin_v1",
        "statistics_schema_version": PROFILE_STATISTICS_SCHEMA_VERSION,
        "confidence_level": PROFILE_CONFIDENCE_LEVEL,
        "z_score": PROFILE_CONFIDENCE_Z_SCORE,
        "min_repeats": PROFILE_CONFIDENCE_MIN_REPEATS,
        "min_absolute_margin_ms": PROFILE_CONFIDENCE_MIN_ABSOLUTE_MARGIN_MS,
        "min_relative_speedup": PROFILE_CONFIDENCE_MIN_RELATIVE_SPEEDUP,
    }


def _selection_confidence(
    best: Mapping[str, Any],
    runner_up: Mapping[str, Any] | None,
) -> dict[str, Any]:
    best_elapsed_ms = _problem_elapsed_ms(best)
    best_timing = _problem_timing(best)
    best_sample_count = int(best_timing.get("sample_count", best.get("repeats", 1)) or 1)
    best_standard_error_ms = float(best_timing.get("standard_error_ms", 0.0) or 0.0)
    payload = {
        **_selection_confidence_policy(),
        "selection_metric_ms": best_elapsed_ms,
        "best_candidate_id": best.get("candidate_id"),
        "best_split_k": _problem_split_k(best),
        "best_standard_error_ms": best_standard_error_ms,
        "runner_up_elapsed_ms": None,
        "runner_up_candidate_id": None,
        "runner_up_split_k": None,
        "runner_up_standard_error_ms": None,
        "sample_counts": {"best": best_sample_count, "runner_up": None},
        "margin_ms": None,
        "required_margin_ms": None,
        "combined_standard_error_ms": None,
        "relative_speedup_over_runner_up": None,
        "reasons": [],
    }
    if runner_up is None:
        payload.update({"level": "single_candidate", "confident": True})
        return payload

    runner_elapsed_ms = _problem_elapsed_ms(runner_up)
    runner_timing = _problem_timing(runner_up)
    runner_sample_count = int(runner_timing.get("sample_count", runner_up.get("repeats", 1)) or 1)
    runner_standard_error_ms = float(runner_timing.get("standard_error_ms", 0.0) or 0.0)
    margin_ms = float(runner_elapsed_ms - best_elapsed_ms)
    relative_speedup = max(margin_ms / best_elapsed_ms, 0.0) if best_elapsed_ms > 0.0 else 0.0
    combined_standard_error_ms = math.sqrt(
        best_standard_error_ms * best_standard_error_ms
        + runner_standard_error_ms * runner_standard_error_ms
    )
    required_margin_ms = max(
        PROFILE_CONFIDENCE_MIN_ABSOLUTE_MARGIN_MS,
        PROFILE_CONFIDENCE_MIN_RELATIVE_SPEEDUP * best_elapsed_ms,
        PROFILE_CONFIDENCE_Z_SCORE * combined_standard_error_ms,
    )
    reasons = []
    if best_sample_count < PROFILE_CONFIDENCE_MIN_REPEATS:
        reasons.append("best_insufficient_repeats")
    if runner_sample_count < PROFILE_CONFIDENCE_MIN_REPEATS:
        reasons.append("runner_up_insufficient_repeats")
    if margin_ms < required_margin_ms:
        reasons.append("margin_below_required_threshold")
    payload.update(
        {
            "runner_up_candidate_id": runner_up.get("candidate_id"),
            "runner_up_split_k": _problem_split_k(runner_up),
            "runner_up_elapsed_ms": runner_elapsed_ms,
            "runner_up_standard_error_ms": runner_standard_error_ms,
            "sample_counts": {"best": best_sample_count, "runner_up": runner_sample_count},
            "margin_ms": margin_ms,
            "required_margin_ms": required_margin_ms,
            "combined_standard_error_ms": combined_standard_error_ms,
            "relative_speedup_over_runner_up": relative_speedup,
            "reasons": reasons,
            "level": "low" if reasons else "high",
            "confident": not reasons,
        }
    )
    return payload


def _problem_elapsed_ms(problem: Mapping[str, Any]) -> float:
    return float(problem.get("elapsed_ms", problem.get("avg_ms", float("inf"))) or float("inf"))


def _problem_timing(problem: Mapping[str, Any]) -> Mapping[str, Any]:
    timing = problem.get("timing")
    return timing if isinstance(timing, Mapping) else {}


def _problem_split_k(problem: Mapping[str, Any]) -> int:
    if problem.get("split_k") is not None:
        return int(problem["split_k"])
    variant = problem.get("profile_variant")
    if isinstance(variant, Mapping) and variant.get("split_k") is not None:
        return int(variant["split_k"])
    selected = problem.get("selected")
    if isinstance(selected, Mapping) and selected.get("split_k") is not None:
        return int(selected["split_k"])
    candidate = _result_candidate(problem)
    return int(candidate.get("split_k_default", 1) or 1)


def _problem_workspace_nbytes(problem: Mapping[str, Any]) -> int:
    if problem.get("workspace_nbytes") is not None:
        return int(problem["workspace_nbytes"])
    candidate = _result_candidate(problem)
    return int(candidate.get("workspace_nbytes", 0) or 0)


def _result_candidate(problem: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = problem.get("candidates", [])
    if isinstance(candidates, (list, tuple)) and candidates:
        first = candidates[0]
        if isinstance(first, Mapping):
            return first
    return {}


def _static_execution_selections(
    selections: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_candidate_set: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for selection in selections:
        key = (
            str(selection.get("op", "")),
            str(selection.get("dtype", "")),
            str(selection.get("candidate_set_key", "")),
        )
        by_candidate_set.setdefault(key, []).append(selection)

    static_selections: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for key, group in sorted(by_candidate_set.items()):
        selected_signatures = {
            (str(item.get("selected_candidate_id", "")), int(item.get("split_k", 1) or 1))
            for item in group
        }
        if len(selected_signatures) == 1:
            selection = dict(group[0])
            selection["workspace_nbytes"] = max(int(item.get("workspace_nbytes", 0) or 0) for item in group)
            selection["selection_key"] = hashlib.sha256(
                canonical_json(
                    {
                        "op": key[0],
                        "dtype": key[1],
                        "candidate_set_key": key[2],
                        "selected_candidate_id": selection.get("selected_candidate_id"),
                        "split_k": int(selection.get("split_k", 1) or 1),
                    }
                ).encode("utf-8")
            ).hexdigest()
            selection["node_id"] = None
            selection["shape"] = {
                "source": "static_overlay_from_consistent_profiled_shapes",
                "profiled_shapes": [dict(item.get("shape", {})) for item in group],
            }
            selection["confidence"] = _static_selection_confidence(group)
            static_selections.append(selection)
        else:
            conflicts.append(
                {
                    "op": key[0],
                    "dtype": key[1],
                    "candidate_set_key": key[2],
                    "reason": "profiled_shapes_selected_different_candidate_or_split_k",
                    "selected_candidate_ids": sorted({item[0] for item in selected_signatures}),
                    "selected_split_k": sorted({item[1] for item in selected_signatures}),
                    "profiled_shapes": [dict(item.get("shape", {})) for item in group],
                }
            )
    return static_selections, conflicts


def _static_selection_confidence(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    confidence_items = [
        dict(item.get("confidence", {}))
        for item in group
        if isinstance(item.get("confidence"), Mapping)
    ]
    low_count = sum(1 for item in confidence_items if not bool(item.get("confident", False)))
    return {
        "name": "all_profiled_shapes_confident_v1",
        "level": "low" if low_count else "high",
        "confident": low_count == 0,
        "profiled_shape_count": len(group),
        "low_confidence_selection_count": low_count,
        "profiled_shape_confidences": confidence_items,
    }


def _write_execution_plan(
    execution_plan: Mapping[str, Any],
    artifact_dir: Path,
    output: str | Path | None,
) -> Path:
    plan_path = Path(output) if output is not None else artifact_dir / "debug" / "execution_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, dict(execution_plan))
    return plan_path


def _execution_plan_summary(execution_plan: Mapping[str, Any], path: Path) -> dict[str, Any]:
    summary = dict(execution_plan.get("summary", {})) if isinstance(execution_plan.get("summary"), Mapping) else {}
    return {
        "path": str(path.resolve()),
        "schema_version": execution_plan.get("schema_version"),
        "execution_plan_key": execution_plan.get("execution_plan_key"),
        **summary,
    }


def _cutlass_split_k_supported(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("supports_split_k")) and cutlass_gemm_split_k_supported(candidate)


def _cutlass_split_k_profiler_symbol(symbol: str) -> str:
    prefix = "dinoml_profile_cutlass_"
    if not symbol.startswith(prefix):
        raise ValueError(f"Unsupported CUTLASS profiler symbol for split-K: {symbol!r}")
    return f"dinoml_profile_cutlass_splitk_{symbol[len(prefix):]}"


def _cutlass_workspace_symbol(symbol: str) -> str:
    prefix = "dinoml_cutlass_"
    if not symbol.startswith(prefix):
        raise ValueError(f"Unsupported CUTLASS kernel symbol for workspace query: {symbol!r}")
    return f"dinoml_cutlass_workspace_{symbol[len(prefix):]}"


class _CutlassGemmProfiler:
    def __init__(self, modules: Mapping[tuple[str, str], Any], candidates: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]):
        self._modules = dict(modules)
        self._candidates = {
            key: {str(candidate.get("profiler_symbol")): dict(candidate) for candidate in value}
            for key, value in candidates.items()
        }

    @classmethod
    def from_codegen_plan(cls, codegen_plan: Mapping[str, Any]) -> "_CutlassGemmProfiler | None":
        item = _external_support_library(codegen_plan, "cutlass_gemm")
        if item is None:
            return None
        cache_dir = Path(str(item["cache_dir"]))
        modules = item.get("modules")
        if not isinstance(modules, Sequence) or isinstance(modules, (str, bytes)) or not modules:
            raise RuntimeError("CUTLASS GEMM profiler support entry has no op/dtype modules")
        loaded = {}
        candidates_by_key = {}
        for module in modules:
            if not isinstance(module, Mapping):
                raise RuntimeError("CUTLASS GEMM profiler support entry has malformed op/dtype module metadata")
            op = str(module.get("op", ""))
            dtype = str(module.get("dtype", ""))
            if not op or not dtype:
                raise RuntimeError("CUTLASS GEMM profiler support entry is missing op/dtype metadata")
            _cutlass_gemm_profiler_extension(cache_dir, op=op, dtype=dtype)
            module_path = _cutlass_gemm_profiler_extension(cache_dir, op=op, dtype=dtype)
            loaded[(op, dtype)] = _load_python_extension(module_path, f"dinoml_cutlass_gemm_profiler_{op}_{dtype}_bind")
        plan_entries = []
        for support_item in codegen_plan.get("external_support_libraries", []):
            if isinstance(support_item, Mapping) and support_item.get("name") == "cutlass_gemm":
                raw_entries = support_item.get("entries", [])
                if isinstance(raw_entries, Sequence) and not isinstance(raw_entries, (str, bytes)):
                    plan_entries = list(raw_entries)
                break
        if isinstance(plan_entries, Sequence) and not isinstance(plan_entries, (str, bytes)):
            for entry in plan_entries:
                if isinstance(entry, Mapping):
                    op = str(entry.get("op", ""))
                    dtype = str(entry.get("dtype", ""))
                    candidates = entry.get("candidates", [])
                    if op and dtype and isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
                        candidates_by_key[(op, dtype)] = [dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)]
        return cls(loaded, candidates_by_key)

    def profile(self, workload: GemmProfileWorkload, *, iterations: int, repeats: int) -> list[dict[str, Any]]:
        split_k = int(workload.split_k)
        module = self._modules.get((workload.op, workload.dtype))
        if module is None:
            raise RuntimeError(f"CUTLASS GEMM profiler for {workload.op}/{workload.dtype} is not loaded")
        raw_results = module.profile_gemm(
            dtype=str(workload.dtype),
            m=int(workload.m),
            n=int(workload.n),
            k=int(workload.k),
            split_k=split_k,
            iterations=int(iterations),
            repeats=int(repeats),
            max_operand_alignment=int(workload.alignment_context.get("candidate_filter", {}).get("max_operand_alignment") or 0),
            has_bias=workload.bias_shape is not None,
            residual_count=len(workload.residual_shapes),
            seed=_stable_u32_seed(workload.op, workload.dtype, workload.shape_case_id, str(workload.m), str(workload.n), str(workload.k)),
        )
        candidates = self._candidates.get((workload.op, workload.dtype), {})
        results = []
        for raw in raw_results:
            profiler_symbol = str(raw["profiler_symbol"])
            candidate = candidates.get(profiler_symbol)
            if candidate is None:
                continue
            if split_k > 1 and not _cutlass_split_k_supported(candidate):
                continue
            results.append(
                {
                    "candidate": candidate,
                    "samples_ms": [float(sample) for sample in raw["samples_ms"]],
                    "workspace_nbytes": int(raw["workspace_nbytes"]),
                }
            )
        if not results:
            raise RuntimeError(f"CUTLASS GEMM profiler for {workload.op}/{workload.dtype} returned no usable candidate timings")
        return results


class _CutlassBmmProfiler:
    def __init__(self, modules: Mapping[tuple[str, str], Any], candidates: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]):
        self._modules = dict(modules)
        self._candidates = {
            key: {str(candidate.get("profiler_symbol")): dict(candidate) for candidate in value}
            for key, value in candidates.items()
        }

    @classmethod
    def from_codegen_plan(cls, codegen_plan: Mapping[str, Any]) -> "_CutlassBmmProfiler | None":
        item = _external_support_library(codegen_plan, "cutlass_bmm")
        if item is None:
            return None
        cache_dir = Path(str(item["cache_dir"]))
        modules = item.get("modules")
        if not isinstance(modules, Sequence) or isinstance(modules, (str, bytes)) or not modules:
            raise RuntimeError("CUTLASS BMM profiler support entry has no op/dtype modules")
        loaded = {}
        for module in modules:
            if not isinstance(module, Mapping):
                raise RuntimeError("CUTLASS BMM profiler support entry has malformed op/dtype module metadata")
            op = str(module.get("op", ""))
            dtype = str(module.get("dtype", ""))
            if not op or not dtype:
                raise RuntimeError("CUTLASS BMM profiler support entry is missing op/dtype metadata")
            module_path = _cutlass_bmm_profiler_extension(cache_dir, op=op, dtype=dtype)
            loaded[(op, dtype)] = _load_python_extension(module_path, f"dinoml_cutlass_bmm_profiler_{op}_{dtype}_bind")
        candidates_by_key = {}
        raw_entries = item.get("entries", [])
        if isinstance(raw_entries, Sequence) and not isinstance(raw_entries, (str, bytes)):
            for entry in raw_entries:
                if isinstance(entry, Mapping):
                    op = str(entry.get("op", ""))
                    dtype = str(entry.get("dtype", ""))
                    candidates = entry.get("candidates", [])
                    if op and dtype and isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
                        candidates_by_key[(op, dtype)] = [dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)]
        return cls(loaded, candidates_by_key)

    def profile(self, workload: GemmProfileWorkload, *, iterations: int, repeats: int) -> list[dict[str, Any]]:
        module = self._modules.get((workload.op, workload.dtype))
        if module is None:
            raise RuntimeError(f"CUTLASS BMM profiler for {workload.op}/{workload.dtype} is not loaded")
        raw_results = module.profile_bmm(
            dtype=str(workload.dtype),
            batch_count=int(workload.batch_count or 1),
            m=int(workload.m),
            n=int(workload.n),
            k=int(workload.k),
            batch_stride_a=int(workload.batch_stride_a or 0),
            batch_stride_b=int(workload.batch_stride_b or 0),
            batch_stride_d0=int(workload.batch_stride_d0 or 0),
            batch_stride_c=int(workload.batch_stride_c or 0),
            lda=int(workload.lda or 0),
            ldb=int(workload.ldb or 0),
            ldd0=int(workload.ldd0 or 0),
            ldc=int(workload.ldc or 0),
            iterations=int(iterations),
            repeats=int(repeats),
            max_operand_alignment=int(workload.alignment_context.get("candidate_filter", {}).get("max_operand_alignment") or 0),
            residual_count=len(workload.residual_shapes),
            a_elements=int(np.prod(workload.a_shape, dtype=np.int64)),
            b_elements=int(np.prod(workload.b_shape, dtype=np.int64)),
            d0_elements=(int(np.prod(workload.residual_shapes[0], dtype=np.int64)) if workload.residual_shapes else 0),
            c_elements=int(np.prod(workload.output_shape, dtype=np.int64)),
            seed=_stable_u32_seed(workload.op, workload.dtype, workload.shape_case_id, str(workload.m), str(workload.n), str(workload.k)),
        )
        candidates = self._candidates.get((workload.op, workload.dtype), {})
        results = []
        for raw in raw_results:
            profiler_symbol = str(raw["profiler_symbol"])
            candidate = candidates.get(profiler_symbol)
            if candidate is None:
                continue
            results.append(
                {
                    "candidate": candidate,
                    "samples_ms": [float(sample) for sample in raw["samples_ms"]],
                    "workspace_nbytes": int(raw["workspace_nbytes"]),
                }
            )
        if not results:
            raise RuntimeError(f"CUTLASS BMM profiler for {workload.op}/{workload.dtype} returned no usable candidate timings")
        return results


class _CutlassConvProfiler:
    def __init__(self, modules: Mapping[tuple[str, str], Any], candidates: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]):
        self._modules = dict(modules)
        self._candidates = {
            key: {str(candidate.get("profiler_symbol")): dict(candidate) for candidate in value}
            for key, value in candidates.items()
        }

    @classmethod
    def from_codegen_plan(cls, codegen_plan: Mapping[str, Any]) -> "_CutlassConvProfiler | None":
        item = _external_support_library(codegen_plan, "cutlass_conv")
        if item is None:
            return None
        cache_dir = Path(str(item["cache_dir"]))
        modules = item.get("modules")
        if not isinstance(modules, Sequence) or isinstance(modules, (str, bytes)) or not modules:
            raise RuntimeError("CUTLASS Conv profiler support entry has no op/dtype modules")
        loaded = {}
        for module in modules:
            if not isinstance(module, Mapping):
                raise RuntimeError("CUTLASS Conv profiler support entry has malformed op/dtype module metadata")
            op = str(module.get("op", ""))
            dtype = str(module.get("dtype", ""))
            if not op or not dtype:
                raise RuntimeError("CUTLASS Conv profiler support entry is missing op/dtype metadata")
            module_path = _cutlass_conv_profiler_extension(cache_dir, op=op, dtype=dtype)
            loaded[(op, dtype)] = _load_python_extension(module_path, f"dinoml_cutlass_conv_profiler_{op}_{dtype}_bind")
        candidates_by_key = {}
        raw_entries = item.get("entries", [])
        if isinstance(raw_entries, Sequence) and not isinstance(raw_entries, (str, bytes)):
            for entry in raw_entries:
                if isinstance(entry, Mapping):
                    op = str(entry.get("op", ""))
                    candidate_set = entry.get("candidate_set", {})
                    dtype = str(candidate_set.get("dtype", ""))
                    candidates = entry.get("candidates", [])
                    if op and dtype and isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
                        candidates_by_key[(op, dtype)] = [dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)]
        return cls(loaded, candidates_by_key)

    def profile(self, workload: ConvProfileWorkload, *, iterations: int, repeats: int) -> list[dict[str, Any]]:
        module = self._modules.get((workload.op, workload.dtype))
        if module is None:
            raise RuntimeError(f"CUTLASS Conv profiler for {workload.op}/{workload.dtype} is not loaded")
        n, c, h, w = (int(dim) for dim in workload.x_shape)
        out_n, out_c, out_h, out_w = (int(dim) for dim in workload.output_shape)
        weight_o, weight_i, kernel_h, kernel_w = (int(dim) for dim in workload.weight_shape)
        if out_n != n or weight_i != c or weight_o != out_c:
            raise RuntimeError(f"CUTLASS Conv profile workload has inconsistent shapes for {workload.node_id}")
        stride = [int(value) for value in workload.conv_config.get("stride", ())]
        padding = [int(value) for value in workload.conv_config.get("padding", ())]
        dilation = [int(value) for value in workload.conv_config.get("dilation", ())]
        if len(stride) != 2 or len(padding) != 2 or len(dilation) != 2:
            raise RuntimeError(f"CUTLASS Conv profile workload has malformed conv_config for {workload.node_id}")
        raw_results = module.profile_conv(
            dtype=str(workload.dtype),
            n=n,
            h=h,
            w=w,
            c=c,
            out_h=out_h,
            out_w=out_w,
            out_c=out_c,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride[0],
            stride_w=stride[1],
            pad_h=padding[0],
            pad_w=padding[1],
            dilation_h=dilation[0],
            dilation_w=dilation[1],
            iterations=int(iterations),
            repeats=int(repeats),
            residual_count=(1 if workload.residual_shape is not None else 0),
            seed=_stable_u32_seed(workload.op, workload.dtype, workload.shape_case_id, str(n), str(out_h), str(out_w), str(out_c)),
        )
        candidates = self._candidates.get((workload.op, workload.dtype), {})
        results = []
        for raw in raw_results:
            profiler_symbol = str(raw["profiler_symbol"])
            candidate = candidates.get(profiler_symbol)
            if candidate is None:
                continue
            results.append(
                {
                    "candidate": candidate,
                    "samples_ms": [float(sample) for sample in raw["samples_ms"]],
                    "workspace_nbytes": int(raw["workspace_nbytes"]),
                }
            )
        if not results:
            raise RuntimeError(f"CUTLASS Conv profiler for {workload.op}/{workload.dtype} returned no usable candidate timings")
        return results


def _external_support_library(codegen_plan: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for item in codegen_plan.get("external_support_libraries", []):
        if isinstance(item, Mapping) and item.get("name") == name:
            return item
    return None


def _stable_u32_seed(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _cutlass_gemm_profiler_extension(cache_dir: Path, *, op: str, dtype: str) -> Path:
    stem = f"dinoml_cutlass_gemm_profiler_{op}_{dtype}"
    return _python_extension_path(
        cache_dir / "lib",
        stem=f"{stem}_bind",
        error_label=(
            f"CUTLASS GEMM profiler binding is missing from {cache_dir / 'lib'}. "
            f"Build the CMake target `{stem}_bind` for this CUDA architecture."
        ),
    )


def _cutlass_bmm_profiler_extension(cache_dir: Path, *, op: str, dtype: str) -> Path:
    stem = f"dinoml_cutlass_bmm_profiler_{op}_{dtype}"
    return _python_extension_path(
        cache_dir / "lib",
        stem=f"{stem}_bind",
        error_label=(
            f"CUTLASS BMM profiler binding is missing from {cache_dir / 'lib'}. "
            f"Build the CMake target `{stem}_bind` for this CUDA architecture."
        ),
    )


def _cutlass_conv_profiler_extension(cache_dir: Path, *, op: str, dtype: str) -> Path:
    stem = f"dinoml_cutlass_conv_profiler_{op}_{dtype}"
    return _python_extension_path(
        cache_dir / "lib",
        stem=f"{stem}_bind",
        error_label=(
            f"CUTLASS Conv profiler binding is missing from {cache_dir / 'lib'}. "
            f"Build the CMake target `{stem}_bind` for this CUDA architecture."
        ),
    )


def _ck_gemm_profiler_extension(cache_dir: Path, *, op: str, dtype: str) -> Path:
    stem = f"dinoml_ck_gemm_profiler_{op}_{dtype}"
    return _python_extension_path(
        cache_dir / "lib",
        stem=f"{stem}_bind",
        error_label=(
            f"CK GEMM profiler binding is missing from {cache_dir / 'lib'}. "
            f"Build the CMake target `{stem}_bind` for this ROCm architecture."
        ),
    )


def _ck_bmm_profiler_extension(cache_dir: Path, *, op: str, dtype: str) -> Path:
    stem = f"dinoml_ck_bmm_profiler_{op}_{dtype}"
    return _python_extension_path(
        cache_dir / "lib",
        stem=f"{stem}_bind",
        error_label=(
            f"CK BMM profiler binding is missing from {cache_dir / 'lib'}. "
            f"Build the CMake target `{stem}_bind` for this ROCm architecture."
        ),
    )


def _ck_conv_profiler_extension(cache_dir: Path, *, op: str, dtype: str) -> Path:
    stem = f"dinoml_ck_conv_profiler_{op}_{dtype}"
    return _python_extension_path(
        cache_dir / "lib",
        stem=f"{stem}_bind",
        error_label=(
            f"CK Conv profiler binding is missing from {cache_dir / 'lib'}. "
            f"Build the CMake target `{stem}_bind` for this ROCm architecture."
        ),
    )


def _python_extension_path(out_dir: Path, *, stem: str, error_label: str) -> Path:
    patterns = [f"{stem}{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES]
    patterns.append(f"{stem}*.so")
    seen: set[Path] = set()
    for pattern in patterns:
        for candidate in sorted(out_dir.glob(pattern)):
            if candidate in seen:
                continue
            seen.add(candidate)
            return candidate
    raise RuntimeError(error_label)


def _load_python_extension(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python extension from {path}")
    module = importlib.util.module_from_spec(spec)
    getdlopenflags = getattr(sys, "getdlopenflags", None)
    setdlopenflags = getattr(sys, "setdlopenflags", None)
    old_flags = None
    if callable(getdlopenflags) and callable(setdlopenflags):
        old_flags = getdlopenflags()
        flags = old_flags
        for name in ("RTLD_NOW", "RTLD_GLOBAL", "RTLD_NODELETE"):
            flags |= int(getattr(os, name, 0))
        setdlopenflags(flags)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if old_flags is not None:
            setdlopenflags(old_flags)
    return module


def _profiler_for_target(artifact_dir: Path, manifest: Mapping[str, Any], codegen_plan: Mapping[str, Any]) -> Any:
    target_name = str(manifest.get("target", {}).get("name", ""))
    if target_name == "rocm":
        return _RocmProfiler(artifact_dir, manifest, codegen_plan)
    return _CudaProfiler(artifact_dir, manifest, codegen_plan)


class _CkGemmProfiler:
    def __init__(self, modules: Mapping[tuple[str, str], Any]):
        self._modules = dict(modules)

    @classmethod
    def from_codegen_plan(cls, codegen_plan: Mapping[str, Any]) -> "_CkGemmProfiler | None":
        item = _external_support_library(codegen_plan, "ck_gemm")
        if item is None:
            return None
        cache_dir = Path(str(item["cache_dir"]))
        modules = item.get("modules")
        if not isinstance(modules, Sequence) or isinstance(modules, (str, bytes)) or not modules:
            raise RuntimeError("CK GEMM profiler support entry has no op/dtype modules")
        loaded = {}
        for module in modules:
            if not isinstance(module, Mapping):
                raise RuntimeError("CK GEMM profiler support entry has malformed op/dtype module metadata")
            op = str(module.get("op", ""))
            dtype = str(module.get("dtype", ""))
            if not op or not dtype:
                raise RuntimeError("CK GEMM profiler support entry is missing op/dtype metadata")
            module_path = _ck_gemm_profiler_extension(cache_dir, op=op, dtype=dtype)
            loaded[(op, dtype)] = _load_python_extension(module_path, f"dinoml_ck_gemm_profiler_{op}_{dtype}_bind")
        return cls(loaded)

    def profile(self, workload: GemmProfileWorkload, *, iterations: int, repeats: int) -> list[dict[str, Any]]:
        module = self._modules.get((workload.op, workload.dtype))
        if module is None:
            raise RuntimeError(f"CK GEMM profiler for {workload.op}/{workload.dtype} is not loaded")
        return list(
            module.profile_gemm(
                profiler_symbol=str(workload.profiler_symbol),
                dtype=str(workload.dtype),
                m=int(workload.m),
                n=int(workload.n),
                k=int(workload.k),
                iterations=int(iterations),
                repeats=int(repeats),
                has_bias=workload.bias_shape is not None,
                residual_count=len(workload.residual_shapes),
                seed=_stable_u32_seed(
                    workload.op,
                    workload.dtype,
                    workload.shape_case_id,
                    str(workload.m),
                    str(workload.n),
                    str(workload.k),
                    str(workload.profiler_symbol),
                ),
            )
        )


class _CkBmmProfiler:
    def __init__(self, modules: Mapping[tuple[str, str], Any]):
        self._modules = dict(modules)

    @classmethod
    def from_codegen_plan(cls, codegen_plan: Mapping[str, Any]) -> "_CkBmmProfiler | None":
        item = _external_support_library(codegen_plan, "ck_bmm")
        if item is None:
            return None
        cache_dir = Path(str(item["cache_dir"]))
        modules = item.get("modules")
        if not isinstance(modules, Sequence) or isinstance(modules, (str, bytes)) or not modules:
            raise RuntimeError("CK BMM profiler support entry has no op/dtype modules")
        loaded = {}
        for module in modules:
            if not isinstance(module, Mapping):
                raise RuntimeError("CK BMM profiler support entry has malformed op/dtype module metadata")
            op = str(module.get("op", ""))
            dtype = str(module.get("dtype", ""))
            if not op or not dtype:
                raise RuntimeError("CK BMM profiler support entry is missing op/dtype metadata")
            module_path = _ck_bmm_profiler_extension(cache_dir, op=op, dtype=dtype)
            loaded[(op, dtype)] = _load_python_extension(module_path, f"dinoml_ck_bmm_profiler_{op}_{dtype}_bind")
        return cls(loaded)

    def profile(self, workload: GemmProfileWorkload, *, iterations: int, repeats: int) -> list[dict[str, Any]]:
        module = self._modules.get((workload.op, workload.dtype))
        if module is None:
            raise RuntimeError(f"CK BMM profiler for {workload.op}/{workload.dtype} is not loaded")
        return list(
            module.profile_bmm(
                profiler_symbol=str(workload.profiler_symbol),
                dtype=str(workload.dtype),
                batch_count=int(workload.batch_count or 1),
                m=int(workload.m),
                n=int(workload.n),
                k=int(workload.k),
                batch_stride_a=int(workload.batch_stride_a or 0),
                batch_stride_b=int(workload.batch_stride_b or 0),
                batch_stride_d0=int(workload.batch_stride_d0 or 0),
                batch_stride_c=int(workload.batch_stride_c or 0),
                lda=int(workload.lda or 0),
                ldb=int(workload.ldb or 0),
                ldd0=int(workload.ldd0 or 0),
                ldc=int(workload.ldc or 0),
                iterations=int(iterations),
                repeats=int(repeats),
                residual_count=len(workload.residual_shapes),
                a_elements=int(np.prod(workload.a_shape, dtype=np.int64)),
                b_elements=int(np.prod(workload.b_shape, dtype=np.int64)),
                d0_elements=(int(np.prod(workload.residual_shapes[0], dtype=np.int64)) if workload.residual_shapes else 0),
                c_elements=int(np.prod(workload.output_shape, dtype=np.int64)),
                seed=_stable_u32_seed(
                    workload.op,
                    workload.dtype,
                    workload.shape_case_id,
                    str(workload.batch_count or 1),
                    str(workload.m),
                    str(workload.n),
                    str(workload.k),
                    str(workload.profiler_symbol),
                ),
            )
        )


class _CkConvProfiler:
    def __init__(self, modules: Mapping[tuple[str, str], Any]):
        self._modules = dict(modules)

    @classmethod
    def from_codegen_plan(cls, codegen_plan: Mapping[str, Any]) -> "_CkConvProfiler | None":
        item = _external_support_library(codegen_plan, "ck_conv")
        if item is None:
            return None
        cache_dir = Path(str(item["cache_dir"]))
        modules = item.get("modules")
        if not isinstance(modules, Sequence) or isinstance(modules, (str, bytes)) or not modules:
            raise RuntimeError("CK Conv profiler support entry has no op/dtype modules")
        loaded = {}
        for module in modules:
            if not isinstance(module, Mapping):
                raise RuntimeError("CK Conv profiler support entry has malformed op/dtype module metadata")
            op = str(module.get("op", ""))
            dtype = str(module.get("dtype", ""))
            if not op or not dtype:
                raise RuntimeError("CK Conv profiler support entry is missing op/dtype metadata")
            module_path = _ck_conv_profiler_extension(cache_dir, op=op, dtype=dtype)
            loaded[(op, dtype)] = _load_python_extension(module_path, f"dinoml_ck_conv_profiler_{op}_{dtype}_bind")
        return cls(loaded)

    def profile(self, workload: ConvProfileWorkload, *, iterations: int, repeats: int) -> list[dict[str, Any]]:
        module = self._modules.get((workload.op, workload.dtype))
        if module is None:
            raise RuntimeError(f"CK Conv profiler for {workload.op}/{workload.dtype} is not loaded")
        stride = list(workload.conv_config.get("stride", (1, 1)))
        padding = list(workload.conv_config.get("padding", (0, 0)))
        dilation = list(workload.conv_config.get("dilation", (1, 1)))
        return list(
            module.profile_conv(
                profiler_symbol=str(workload.profiler_symbol),
                dtype=str(workload.dtype),
                batch=int(workload.x_shape[0]),
                in_channels=int(workload.x_shape[1]),
                in_height=int(workload.x_shape[2]),
                in_width=int(workload.x_shape[3]),
                out_channels=int(workload.weight_shape[0]),
                kernel_h=int(workload.weight_shape[2]),
                kernel_w=int(workload.weight_shape[3]),
                out_height=int(workload.output_shape[2]),
                out_width=int(workload.output_shape[3]),
                stride_h=int(stride[0]),
                stride_w=int(stride[1]),
                pad_h=int(padding[0]),
                pad_w=int(padding[1]),
                dilation_h=int(dilation[0]),
                dilation_w=int(dilation[1]),
                iterations=int(iterations),
                repeats=int(repeats),
                has_residual=workload.residual_shape is not None,
                x_elements=int(np.prod(workload.x_shape, dtype=np.int64)),
                weight_elements=int(np.prod(workload.weight_shape, dtype=np.int64)),
                bias_elements=int(np.prod(workload.bias_shape, dtype=np.int64)),
                residual_elements=(
                    int(np.prod(workload.residual_shape, dtype=np.int64))
                    if workload.residual_shape is not None
                    else 0
                ),
                output_elements=int(np.prod(workload.output_shape, dtype=np.int64)),
                seed=_stable_u32_seed(
                    workload.op,
                    workload.dtype,
                    workload.shape_case_id,
                    *(str(dim) for dim in workload.x_shape),
                    *(str(dim) for dim in workload.weight_shape),
                    *(str(dim) for dim in workload.output_shape),
                    str(workload.profiler_symbol),
                ),
            )
        )


class _RocmProfiler:
    def __init__(self, artifact_dir: Path, manifest: Mapping[str, Any], codegen_plan: Mapping[str, Any]):
        del artifact_dir, manifest
        self._ck_gemm_profiler = _CkGemmProfiler.from_codegen_plan(codegen_plan)
        self._ck_bmm_profiler = _CkBmmProfiler.from_codegen_plan(codegen_plan)
        self._ck_conv_profiler = _CkConvProfiler.from_codegen_plan(codegen_plan)

    def close(self) -> None:
        return None

    def profile(
        self,
        workload: GemmProfileWorkload | ConvProfileWorkload,
        *,
        iterations: int,
        rng: np.random.Generator,
    ) -> tuple[float, int]:
        samples_ms, workspace_nbytes = self.profile_samples(
            workload,
            iterations=iterations,
            repeats=1,
            rng=rng,
        )
        return float(samples_ms[0]), int(workspace_nbytes)

    def profile_samples(
        self,
        workload: GemmProfileWorkload | ConvProfileWorkload,
        *,
        iterations: int,
        repeats: int,
        rng: np.random.Generator,
    ) -> tuple[list[float], int]:
        if workload.kernel_library == "ck_gemm":
            if self._ck_gemm_profiler is None:
                raise RuntimeError("CK GEMM profiler requested but codegen plan has no ck_gemm support entry")
            rows = self._ck_gemm_profiler.profile(workload, iterations=iterations, repeats=repeats)
            first = rows[0]
            return [float(sample) for sample in first["samples_ms"]], int(first["workspace_nbytes"])
        if workload.kernel_library == "ck_bmm":
            if self._ck_bmm_profiler is None:
                raise RuntimeError("CK BMM profiler requested but codegen plan has no ck_bmm support entry")
            rows = self._ck_bmm_profiler.profile(workload, iterations=iterations, repeats=repeats)
            first = rows[0]
            return [float(sample) for sample in first["samples_ms"]], int(first["workspace_nbytes"])
        if workload.kernel_library == "ck_conv" and isinstance(workload, ConvProfileWorkload):
            if self._ck_conv_profiler is None:
                raise RuntimeError("CK Conv profiler requested but codegen plan has no ck_conv support entry")
            rows = self._ck_conv_profiler.profile(workload, iterations=iterations, repeats=repeats)
            first = rows[0]
            return [float(sample) for sample in first["samples_ms"]], int(first["workspace_nbytes"])
        raise RuntimeError(f"Unsupported ROCm profiler library {workload.kernel_library!r}")


class _CkRocmProfiler:
    def __init__(self, artifact_dir: Path, manifest: Mapping[str, Any]):
        files = manifest["files"]
        global_mode = getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(ctypes, "RTLD_NOW", 0)
        self._dll_dirs: list[Any] = []
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            for directory in (artifact_dir, artifact_dir / "lib"):
                if directory.exists():
                    self._dll_dirs.append(os.add_dll_directory(str(directory)))
        self._runtime = ctypes.CDLL(str(artifact_dir / files["runtime_library"]), mode=global_mode)
        self._rocm_runtime = ctypes.CDLL(str(artifact_dir / files["rocm_runtime_library"]), mode=global_mode)
        module_path = artifact_dir / str(files.get("module", "module.so"))
        if not module_path.exists():
            raise RuntimeError(f"ROCm profiler requires a compiled module with CK profiler symbols: {module_path}")
        self._module = ctypes.CDLL(str(module_path), mode=global_mode)
        self._buffers: list[ctypes.c_void_p] = []
        self._runtime.dino_get_last_error.restype = ctypes.c_char_p
        if hasattr(self._rocm_runtime, "dino_get_last_error"):
            self._rocm_runtime.dino_get_last_error.restype = ctypes.c_char_p
        self._rocm_runtime.dino_device_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._rocm_runtime.dino_device_malloc.restype = ctypes.c_int
        self._rocm_runtime.dino_device_free.argtypes = [ctypes.c_void_p]
        self._rocm_runtime.dino_device_free.restype = ctypes.c_int
        self._rocm_runtime.dino_copy_host_to_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self._rocm_runtime.dino_copy_host_to_device.restype = ctypes.c_int

    def close(self) -> None:
        self._release_buffers_since(0)
        for handle in reversed(self._dll_dirs):
            close = getattr(handle, "close", None)
            if close is not None:
                close()
        self._dll_dirs.clear()

    def profile(
        self,
        workload: GemmProfileWorkload | ConvProfileWorkload,
        *,
        iterations: int,
        rng: np.random.Generator,
    ) -> tuple[float, int]:
        if workload.kernel_library == "ck_gemm":
            return self.profile_ck_gemm(workload, iterations=iterations, rng=rng)
        if workload.kernel_library == "ck_bmm":
            return self.profile_ck_bmm(workload, iterations=iterations, rng=rng)
        if workload.kernel_library == "ck_conv" and isinstance(workload, ConvProfileWorkload):
            return self.profile_ck_conv(workload, iterations=iterations, rng=rng)
        raise RuntimeError(f"Unsupported ROCm profiler library {workload.kernel_library!r}")

    def profile_ck_gemm(
        self,
        workload: GemmProfileWorkload,
        *,
        iterations: int,
        rng: np.random.Generator,
    ) -> tuple[float, int]:
        start_index = len(getattr(self, "_buffers", ()))
        fn = self._profiler_function(workload.profiler_symbol)
        abi = str(workload.candidate.get("launch_abi", ""))
        ptr = ctypes.c_void_p
        if abi == "dinoml_ck_gemm_v1":
            fn.argtypes = [ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr]
        elif abi == "dinoml_ck_gemm_bias_v1":
            fn.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr]
        elif abi == "dinoml_ck_gemm_bias_residual_v1":
            fn.argtypes = [ptr, ptr, ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr]
        elif abi == "dinoml_ck_gemm_bias_residual2_v1":
            fn.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr]
        else:
            raise RuntimeError(f"Unsupported CK GEMM profiler ABI {abi!r} for {workload.candidate_id}")
        try:
            a = self._device_array(_random_storage(workload.a_shape, workload.dtype, rng))
            b = self._device_array(_random_storage(workload.b_shape, workload.dtype, rng))
            c = self._device_array(_zero_storage(workload.output_shape, workload.dtype))
            args: list[Any] = [a, b]
            if abi != "dinoml_ck_gemm_v1":
                if workload.bias_shape is None:
                    raise RuntimeError(f"CK GEMM profiler ABI {abi!r} requires a bias tensor")
                args.append(self._device_array(_random_storage(workload.bias_shape, workload.dtype, rng)))
            if abi in {"dinoml_ck_gemm_bias_residual_v1", "dinoml_ck_gemm_bias_residual2_v1"}:
                if not workload.residual_shapes:
                    raise RuntimeError(f"CK GEMM profiler ABI {abi!r} requires residual tensor d0")
                args.append(self._device_array(_random_storage(workload.residual_shapes[0], workload.dtype, rng)))
            if abi == "dinoml_ck_gemm_bias_residual2_v1":
                if len(workload.residual_shapes) < 2:
                    raise RuntimeError(f"CK GEMM profiler ABI {abi!r} requires residual tensor d1")
                args.append(self._device_array(_random_storage(workload.residual_shapes[1], workload.dtype, rng)))
            args.extend([c, workload.m, workload.n, workload.k, int(iterations), ctypes.c_void_p()])
            elapsed_ms = float(fn(*args))
            self._check_ck_elapsed(elapsed_ms, workload)
            return elapsed_ms, int(workload.workspace_nbytes)
        finally:
            self._release_buffers_since(start_index)

    def profile_ck_bmm(
        self,
        workload: GemmProfileWorkload,
        *,
        iterations: int,
        rng: np.random.Generator,
    ) -> tuple[float, int]:
        start_index = len(getattr(self, "_buffers", ()))
        fn = self._profiler_function(workload.profiler_symbol)
        abi = str(workload.candidate.get("launch_abi", ""))
        ptr = ctypes.c_void_p
        i64 = ctypes.c_longlong
        if abi == "dinoml_ck_bmm_v1":
            fn.argtypes = [
                ptr,
                ptr,
                ptr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                i64,
                i64,
                i64,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ptr,
            ]
        elif abi == "dinoml_ck_bmm_add_v1":
            fn.argtypes = [
                ptr,
                ptr,
                ptr,
                ptr,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                i64,
                i64,
                i64,
                i64,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ptr,
            ]
        else:
            raise RuntimeError(f"Unsupported CK BMM profiler ABI {abi!r} for {workload.candidate_id}")
        try:
            a = self._device_array(_random_storage(workload.a_shape, workload.dtype, rng))
            b = self._device_array(_random_storage(workload.b_shape, workload.dtype, rng))
            c = self._device_array(_zero_storage(workload.output_shape, workload.dtype))
            common = [
                int(workload.batch_count or 1),
                workload.m,
                workload.n,
                workload.k,
                int(workload.batch_stride_a or 0),
                int(workload.batch_stride_b or 0),
            ]
            if abi == "dinoml_ck_bmm_add_v1":
                if not workload.residual_shapes:
                    raise RuntimeError("CK BMM add profiler requires residual tensor d0")
                d0 = self._device_array(_random_storage(workload.residual_shapes[0], workload.dtype, rng))
                args = [
                    a,
                    b,
                    d0,
                    c,
                    *common,
                    int(workload.batch_stride_d0 or 0),
                    int(workload.batch_stride_c or 0),
                    int(workload.lda or 0),
                    int(workload.ldb or 0),
                    int(workload.ldd0 or 0),
                    int(workload.ldc or 0),
                    int(iterations),
                    ctypes.c_void_p(),
                ]
            else:
                args = [
                    a,
                    b,
                    c,
                    *common,
                    int(workload.batch_stride_c or 0),
                    int(workload.lda or 0),
                    int(workload.ldb or 0),
                    int(workload.ldc or 0),
                    int(iterations),
                    ctypes.c_void_p(),
                ]
            elapsed_ms = float(fn(*args))
            self._check_ck_elapsed(elapsed_ms, workload)
            return elapsed_ms, int(workload.workspace_nbytes)
        finally:
            self._release_buffers_since(start_index)

    def profile_ck_conv(
        self,
        workload: ConvProfileWorkload,
        *,
        iterations: int,
        rng: np.random.Generator,
    ) -> tuple[float, int]:
        start_index = len(getattr(self, "_buffers", ()))
        fn = self._profiler_function(workload.profiler_symbol)
        ptr = ctypes.c_void_p
        has_residual = workload.residual_shape is not None
        fn.argtypes = [
            ptr,
            ptr,
            ptr,
            *( [ptr] if has_residual else [] ),
            ptr,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ptr,
        ]
        try:
            x = self._device_array(_random_storage(workload.x_shape, workload.dtype, rng))
            weight = self._device_array(_random_storage(workload.weight_shape, workload.dtype, rng))
            bias = self._device_array(_random_storage(workload.bias_shape, workload.dtype, rng))
            residual = (
                self._device_array(_random_storage(workload.residual_shape, workload.dtype, rng))
                if has_residual
                else None
            )
            output = self._device_array(_zero_storage(workload.output_shape, workload.dtype))
            batch, in_channels, in_height, in_width = (int(dim) for dim in workload.x_shape)
            out_channels, _weight_channels, kernel_h, kernel_w = (int(dim) for dim in workload.weight_shape)
            out_height, out_width = int(workload.output_shape[2]), int(workload.output_shape[3])
            stride = list(workload.conv_config.get("stride", (1, 1)))
            padding = list(workload.conv_config.get("padding", (0, 0)))
            dilation = list(workload.conv_config.get("dilation", (1, 1)))
            pointer_args = [x, weight, bias, *([residual] if residual is not None else []), output]
            elapsed_ms = float(
                fn(
                    *pointer_args,
                    batch,
                    in_channels,
                    in_height,
                    in_width,
                    out_channels,
                    kernel_h,
                    kernel_w,
                    out_height,
                    out_width,
                    int(stride[0]),
                    int(stride[1]),
                    int(padding[0]),
                    int(padding[1]),
                    int(dilation[0]),
                    int(dilation[1]),
                    int(iterations),
                    ctypes.c_void_p(),
                )
            )
            self._check_ck_elapsed(elapsed_ms, workload)
            return elapsed_ms, int(workload.workspace_nbytes)
        finally:
            self._release_buffers_since(start_index)

    def _profiler_function(self, symbol: str) -> Any:
        try:
            fn = getattr(self._module, symbol)
        except AttributeError as exc:
            raise RuntimeError(f"Compiled ROCm module does not export CK profiler symbol {symbol!r}") from exc
        fn.restype = ctypes.c_float
        return fn

    def _device_array(self, array: np.ndarray) -> ctypes.c_void_p:
        contiguous = np.ascontiguousarray(array)
        ptr = ctypes.c_void_p()
        self._check(self._rocm_runtime.dino_device_malloc(ctypes.byref(ptr), ctypes.c_size_t(contiguous.nbytes)))
        self._buffers.append(ptr)
        self._check(
            self._rocm_runtime.dino_copy_host_to_device(
                ptr,
                ctypes.c_void_p(contiguous.ctypes.data),
                ctypes.c_size_t(contiguous.nbytes),
            )
        )
        return ptr

    def _release_buffers_since(self, start_index: int) -> None:
        if not hasattr(self, "_buffers"):
            return
        for index in range(len(self._buffers) - 1, start_index - 1, -1):
            ptr = self._buffers[index]
            self._check(self._rocm_runtime.dino_device_free(ptr))
            del self._buffers[index]

    def _check(self, code: int) -> None:
        if code == 0:
            return
        error = None
        getter = getattr(self._rocm_runtime, "dino_get_last_error", None)
        if getter is not None:
            error = getter()
        if not error:
            error = self._runtime.dino_get_last_error()
        message = error.decode("utf-8") if error else f"ROCm profiler helper failed with code {code}"
        raise RuntimeError(message)

    def _check_ck_elapsed(
        self,
        elapsed_ms: float,
        workload: GemmProfileWorkload | ConvProfileWorkload,
    ) -> None:
        if math.isfinite(elapsed_ms) and elapsed_ms >= 0.0:
            return
        diagnostics = _profile_failure_diagnostics(workload, elapsed_ms=elapsed_ms)
        raise RuntimeError(
            f"CK ROCm profiler rejected {workload.op} candidate {workload.candidate_id} "
            f"for profiler symbol {workload.profiler_symbol!r}; "
            f"{_format_profile_failure_diagnostics(diagnostics)}"
        )


class _CudaProfiler:
    def __init__(self, artifact_dir: Path, manifest: Mapping[str, Any], codegen_plan: Mapping[str, Any]):
        files = manifest["files"]
        global_mode = getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(ctypes, "RTLD_NOW", 0)
        needs_ctypes_runtime = False
        self._runtime = (
            ctypes.CDLL(str(artifact_dir / files["runtime_library"]), mode=global_mode)
            if needs_ctypes_runtime
            else None
        )
        self._cuda_runtime = (
            ctypes.CDLL(str(artifact_dir / files["cuda_runtime_library"]), mode=global_mode)
            if needs_ctypes_runtime
            else None
        )
        self._cutlass_gemm_profiler = _CutlassGemmProfiler.from_codegen_plan(codegen_plan)
        self._cutlass_bmm_profiler = _CutlassBmmProfiler.from_codegen_plan(codegen_plan)
        self._cutlass_conv_profiler = _CutlassConvProfiler.from_codegen_plan(codegen_plan)
        self._buffers: list[ctypes.c_void_p] = []
        if self._runtime is not None and self._cuda_runtime is not None:
            self._runtime.dino_get_last_error.restype = ctypes.c_char_p
            if hasattr(self._cuda_runtime, "dino_get_last_error"):
                self._cuda_runtime.dino_get_last_error.restype = ctypes.c_char_p
            self._cuda_runtime.dino_device_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            self._cuda_runtime.dino_device_malloc.restype = ctypes.c_int
            self._cuda_runtime.dino_device_free.argtypes = [ctypes.c_void_p]
            self._cuda_runtime.dino_device_free.restype = ctypes.c_int
            self._cuda_runtime.dino_copy_host_to_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            self._cuda_runtime.dino_copy_host_to_device.restype = ctypes.c_int

    def close(self) -> None:
        if self._cuda_runtime is None:
            return
        for index in range(len(self._buffers) - 1, -1, -1):
            ptr = self._buffers[index]
            self._check(self._cuda_runtime.dino_device_free(ptr))
            del self._buffers[index]

    def profile(
        self,
        workload: GemmProfileWorkload | ConvProfileWorkload,
        *,
        iterations: int,
        rng: np.random.Generator,
    ) -> tuple[float, int]:
        if workload.kernel_library == "cutlass_bmm":
            del rng
            rows = self.profile_cutlass_bmm_problem(workload, iterations=iterations, repeats=1)
            first = rows[0]
            return float(first["samples_ms"][0]), int(first["workspace_nbytes"])
        if workload.kernel_library == "cutlass_conv":
            del rng
            rows = self.profile_cutlass_conv_problem(workload, iterations=iterations, repeats=1)
            first = rows[0]
            return float(first["samples_ms"][0]), int(first["workspace_nbytes"])
        if workload.kernel_library != "cutlass_gemm":
            raise RuntimeError(f"Unsupported profiler library {workload.kernel_library!r}")
        del rng
        if self._cutlass_gemm_profiler is None:
            raise RuntimeError("CUTLASS GEMM profiler requested but codegen plan has no cutlass_gemm support entry")
        rows = self._cutlass_gemm_profiler.profile(workload, iterations=iterations, repeats=1)
        first = rows[0]
        return float(first["samples_ms"][0]), int(first["workspace_nbytes"])

    def profile_gemm(self, workload: GemmProfileWorkload, *, iterations: int, rng: np.random.Generator) -> tuple[float, int]:
        del rng
        if self._cutlass_gemm_profiler is None:
            raise RuntimeError("CUTLASS GEMM profiler requested but codegen plan has no cutlass_gemm support entry")
        rows = self._cutlass_gemm_profiler.profile(workload, iterations=iterations, repeats=1)
        first = rows[0]
        return float(first["samples_ms"][0]), int(first["workspace_nbytes"])

    def profile_cutlass_gemm_problem(
        self,
        workload: GemmProfileWorkload,
        *,
        iterations: int,
        repeats: int,
    ) -> list[dict[str, Any]]:
        if self._cutlass_gemm_profiler is None:
            raise RuntimeError("CUTLASS GEMM profiler requested but codegen plan has no cutlass_gemm support entry")
        return self._cutlass_gemm_profiler.profile(workload, iterations=iterations, repeats=repeats)

    def profile_cutlass_bmm_problem(
        self,
        workload: GemmProfileWorkload,
        *,
        iterations: int,
        repeats: int,
    ) -> list[dict[str, Any]]:
        if self._cutlass_bmm_profiler is None:
            raise RuntimeError("CUTLASS BMM profiler requested but codegen plan has no cutlass_bmm support entry")
        return self._cutlass_bmm_profiler.profile(workload, iterations=iterations, repeats=repeats)

    def profile_cutlass_conv_problem(
        self,
        workload: ConvProfileWorkload,
        *,
        iterations: int,
        repeats: int,
    ) -> list[dict[str, Any]]:
        if self._cutlass_conv_profiler is None:
            raise RuntimeError("CUTLASS Conv profiler requested but codegen plan has no cutlass_conv support entry")
        return self._cutlass_conv_profiler.profile(workload, iterations=iterations, repeats=repeats)

    def _cutlass_library(self, library: str, *, workload: GemmProfileWorkload | ConvProfileWorkload | None = None) -> ctypes.CDLL:
        raise RuntimeError(f"Artifact does not contain required {library} support library")

    def _cutlass_workspace_nbytes(self, workload: GemmProfileWorkload, cutlass: ctypes.CDLL) -> int:
        symbol = _cutlass_workspace_symbol(workload.kernel_symbol)
        fn = getattr(cutlass, symbol)
        fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        fn.restype = ctypes.c_size_t
        return int(
            fn(
                ctypes.c_int(workload.m),
                ctypes.c_int(workload.n),
                ctypes.c_int(workload.k),
                ctypes.c_int(workload.split_k),
            )
        )

    def _device_buffer(self, nbytes: int) -> ctypes.c_void_p:
        if self._cuda_runtime is None:
            raise RuntimeError("CUDA runtime library is not loaded for this profiler")
        ptr = ctypes.c_void_p()
        self._check(self._cuda_runtime.dino_device_malloc(ctypes.byref(ptr), ctypes.c_size_t(int(nbytes))))
        self._buffers.append(ptr)
        return ptr

    def _device_array(self, array: np.ndarray) -> ctypes.c_void_p:
        if self._cuda_runtime is None:
            raise RuntimeError("CUDA runtime library is not loaded for this profiler")
        contiguous = np.ascontiguousarray(array)
        ptr = ctypes.c_void_p()
        self._check(self._cuda_runtime.dino_device_malloc(ctypes.byref(ptr), ctypes.c_size_t(contiguous.nbytes)))
        self._buffers.append(ptr)
        self._check(
            self._cuda_runtime.dino_copy_host_to_device(
                ptr,
                ctypes.c_void_p(contiguous.ctypes.data),
                ctypes.c_size_t(contiguous.nbytes),
            )
        )
        return ptr

    def _check(self, code: int) -> None:
        if code == 0:
            return
        error = None
        getter = getattr(self._cuda_runtime, "dino_get_last_error", None) if self._cuda_runtime is not None else None
        if getter is not None:
            error = getter()
        if not error and self._runtime is not None:
            error = self._runtime.dino_get_last_error()
        message = error.decode("utf-8") if error else f"CUDA profiler helper failed with code {code}"
        raise RuntimeError(message)


def _random_storage(shape: Sequence[int], dtype: str, rng: np.random.Generator) -> np.ndarray:
    values = (rng.standard_normal(tuple(shape)).astype(np.float32) * 0.125)
    return np.ascontiguousarray(array_to_storage(values, dtype))


def _zero_storage(shape: Sequence[int], dtype: str) -> np.ndarray:
    return np.ascontiguousarray(array_to_storage(np.zeros(tuple(shape), dtype=np.float32), dtype))
