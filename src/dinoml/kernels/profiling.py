from __future__ import annotations

import ctypes
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dinoml.ir import array_to_storage, canonical_json, dtype_nbytes, read_json, write_json
from dinoml.kernels.bmm import BMM_OPS, bmm_op_spec, bmm_problem
from dinoml.kernels.gemm import GEMM_OPS, gemm_op_spec, gemm_problem
from dinoml.kernels.manifest import PROFILE_CACHE_SCHEMA_VERSION
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
    validate_cutlass_conv_scaffold_plan,
)
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_split_k_supported
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


@dataclass(frozen=True)
class ProfileShapeScenario:
    source: str
    case_id: str
    dim_values: Mapping[str, int]
    dim_sources: Mapping[str, str]
    overrides: Mapping[str, Sequence[int]]


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
            "profile_variant": {
                "kind": str(self.candidate.get("status", "manifest_scaffold_only")),
                "profiler_status": str(self.candidate.get("profiler_status", "unsupported_stub")),
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
    if kernel_manifest.get("target", {}).get("name") != "cuda":
        return []
    tensor_map = {str(tensor["name"]): tensor for tensor in graph["tensors"]}
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
            )
            continue
        if op_name in CONV_OPS:
            _append_conv_profile_workloads(
                workloads,
                node,
                tensor_map,
                required_by_op.get(op_name, ()),
                overrides,
            )
            continue
        if op_name not in GEMM_OPS:
            continue
        output_name = str(node["outputs"][0])
        output_info = tensor_map[output_name]
        dtype = str(output_info["dtype"])
        binding = get_op_def(op_name).backend_kernels["cuda"].resolve(dtype)
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
        for scenario in _profile_shape_scenarios(node, tensor_map, overrides):
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
                        )
                    )
    return workloads


def _unsupported_profile_workload_reason(workload: GemmProfileWorkload | ConvProfileWorkload) -> str | None:
    if workload.kernel_library != "cutlass_conv":
        return None
    profiler_status = str(workload.candidate.get("profiler_status", ""))
    if profiler_status == "bounded_runtime_profiler":
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
    normalized_conv_plan = validate_cutlass_conv_scaffold_plan(
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
    for scenario in _profile_shape_scenarios(node, tensor_map, overrides):
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
                    shape_source=scenario.source,
                    shape_case_id=scenario.case_id,
                    dim_values=scenario.dim_values,
                    dim_sources=scenario.dim_sources,
                )
            )


def _append_bmm_profile_workloads(
    workloads: list[GemmProfileWorkload],
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    required_items: Sequence[Mapping[str, Any]],
    overrides: Mapping[str, Sequence[int]],
) -> None:
    op_name = str(node["op"])
    output_name = str(node["outputs"][0])
    output_info = tensor_map[output_name]
    dtype = str(output_info["dtype"])
    binding = get_op_def(op_name).backend_kernels["cuda"].resolve(dtype)
    required_item = _required_profile_item(required_items, dtype, binding.symbol, node_id=str(node["id"]))
    if required_item is None:
        return
    spec = bmm_op_spec(op_name)
    a_name, b_name = (str(name) for name in node["inputs"][:2])
    residual_names = tuple(str(node["inputs"][input_offset]) for input_offset, _name in enumerate(spec.inputs, start=2))
    for scenario in _profile_shape_scenarios(node, tensor_map, overrides):
        a_shape = _runtime_tensor_shape(a_name, tensor_map[a_name], scenario.overrides, scenario.dim_values)
        b_shape = _runtime_tensor_shape(b_name, tensor_map[b_name], scenario.overrides, scenario.dim_values)
        residual_shapes = tuple(
            _runtime_tensor_shape(name, tensor_map[name], scenario.overrides, scenario.dim_values)
            for name in residual_names
        )
        batch_count, m, n, k, output_shape = bmm_problem(op_name, [a_shape, b_shape, *residual_shapes])
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
                        kernel_library="cutlass_bmm",
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
        return candidates
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
    if manifest.get("target", {}).get("name") != "cuda":
        raise ValueError("Profiler runner currently supports CUDA artifacts only")
    graph = read_json(artifact_dir / manifest["files"]["graph"])
    _validate_symbolic_int_expr_profile_graph(graph)
    kernel_manifest = read_json(artifact_dir / manifest["files"]["kernel_manifest"])
    codegen_plan = read_json(artifact_dir / manifest["files"]["kernel_codegen_plan"])
    workloads = build_profile_workloads(graph, kernel_manifest, input_shapes=input_shapes)
    _reject_unsupported_profile_workloads(workloads, context="profile execution")
    cache_path = profile_cache_path(codegen_plan)
    cache = _read_profile_cache(cache_path, manifest["target"])
    context = _profile_context(artifact_dir, manifest, codegen_plan)
    summary = {"profiled": 0, "cached": 0, "skipped": 0, "failed": 0}
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
        )
        execution_plan = build_execution_plan(report)
        execution_plan_path = _write_execution_plan(execution_plan, artifact_dir, execution_plan_output)
        report["execution_plan"] = _execution_plan_summary(execution_plan, execution_plan_path)
        _write_profile_report(report, artifact_dir, output)
        return report

    rng = np.random.default_rng(seed)
    results = []
    profiler = None
    try:
        for workload in workloads:
            key_payload = _profile_key_payload(workload, manifest, kernel_manifest, codegen_plan, context=context)
            profile_key = _profile_key(key_payload)
            cached = cache["entries"].get(profile_key)
            if (
                cached is not None
                and not refresh
                and _cache_entry_satisfies(cached, key_payload=key_payload, iterations=iterations, repeats=repeats)
            ):
                results.append(_profile_result_from_cache(workload, cached))
                summary["cached"] += 1
            else:
                if profiler is None:
                    profiler = _CudaProfiler(artifact_dir, manifest)
                samples_ms = []
                workspace_nbytes = int(workload.workspace_nbytes)
                for _ in range(repeats):
                    elapsed_ms, sample_workspace_nbytes = profiler.profile(workload, iterations=iterations, rng=rng)
                    samples_ms.append(elapsed_ms)
                    workspace_nbytes = max(workspace_nbytes, int(sample_workspace_nbytes))
                timing = _profile_timing(samples_ms, iterations=iterations)
                result = _profile_result(
                    workload,
                    timing["median_ms"],
                    iterations,
                    profile_key=profile_key,
                    status="ok",
                    workspace_nbytes=workspace_nbytes,
                    timing=timing,
                )
                results.append(result)
                cache["entries"][profile_key] = _cache_entry(workload, result, key_payload)
                summary["profiled"] += 1
    finally:
        if profiler is not None:
            profiler.close()
    if summary["profiled"]:
        _write_profile_cache(cache_path, cache)

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
    )
    execution_plan = build_execution_plan(report)
    execution_plan_path = _write_execution_plan(execution_plan, artifact_dir, execution_plan_output)
    report["execution_plan"] = _execution_plan_summary(execution_plan, execution_plan_path)
    _write_profile_report(report, artifact_dir, output)
    return report


def profile_cache_path(codegen_plan: Mapping[str, Any]) -> Path:
    return Path(str(codegen_plan["support_cache_dir"])) / f"profile_cache.v{PROFILE_CACHE_SCHEMA_VERSION}.json"


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
    return tuple(int(dim) for dim in tensor["shape"])


def _profile_shape_scenarios(
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Sequence[int]],
) -> list[ProfileShapeScenario]:
    if overrides:
        return [
            ProfileShapeScenario(
                source="runtime_override",
                case_id="runtime_override",
                dim_values={},
                dim_sources={},
                overrides=overrides,
            )
        ]
    dynamic_values = _profile_dim_values(node, tensor_map)
    if not dynamic_values or not any(info["source"] == "bucket" for info in dynamic_values.values()):
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
    for values in itertools.product(*(dynamic_values[name]["values"] for name in dim_names)):
        assignments = dict(zip(dim_names, values))
        scenario_overrides = {
            str(tensor_name): _shape_from_dim_assignments(tensor_map[str(tensor_name)], assignments)
            for tensor_name in node.get("inputs", [])
        }
        scenarios.append(
            ProfileShapeScenario(
                source="dim_buckets",
                case_id="bucket_" + "_".join(f"{name}={assignments[name]}" for name in dim_names),
                dim_values=assignments,
                dim_sources={name: str(dynamic_values[name]["source"]) for name in dim_names},
                overrides=scenario_overrides,
            )
        )
    return scenarios


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


def _record_profile_dim_value(by_name: dict[str, dict[str, Any]], dim: Mapping[str, Any]) -> None:
    name = str(dim["name"])
    buckets = tuple(int(bucket) for bucket in dim.get("buckets", ()))
    signature = (
        int(dim["min"]),
        int(dim["max"]),
        int(dim.get("divisible_by", 1)),
        buckets,
    )
    info = by_name.setdefault(
        name,
        {
            "signature": signature,
            "values": buckets or (int(dim["max"]),),
            "source": "bucket" if buckets else "max",
        },
    )
    if info["signature"] != signature:
        raise ValueError(f"Inconsistent profiling bucket metadata for dynamic dimension {name!r}")


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
    if workload.kernel_library == "cutlass_conv":
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
    if workload.kernel_library == "cutlass_bmm":
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
        "repeats": int(repeats),
        "summary": dict(summary),
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

    if "cutlass_gemm_library" in files:
        path = artifact_dir / files["cutlass_gemm_library"]
        merge(
            "cutlass_gemm",
            path=str(path.resolve()),
            artifact_path=str(path.resolve()),
            artifact_sha256=_file_sha256(path),
        )
    if "cutlass_bmm_library" in files:
        path = artifact_dir / files["cutlass_bmm_library"]
        merge(
            "cutlass_bmm",
            path=str(path.resolve()),
            artifact_path=str(path.resolve()),
            artifact_sha256=_file_sha256(path),
        )
    for item in codegen_plan.get("external_support_libraries", []):
        name = str(item["name"])
        cache_dir = Path(str(item.get("cache_dir", ""))) if item.get("cache_dir") else None
        cache_library = _cache_library_path(item, cache_dir)
        support_manifest = _support_library_manifest_path(name, cache_dir)
        support_payload = _read_optional_json(support_manifest) if support_manifest else {}
        manifest_fields = _support_manifest_fields(support_payload)
        merge(
            name,
            cache_dir=str(cache_dir) if cache_dir else None,
            cache_library=str(cache_library) if cache_library else None,
            cache_library_sha256=_file_sha256(cache_library) if cache_library else None,
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
    hardware = _cuda_hardware_fingerprint(manifest["target"])
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
    }


def _cache_library_path(item: Mapping[str, Any], cache_dir: Path | None) -> Path | None:
    if cache_dir is None:
        return None
    library = str(item.get("library", ""))
    if not library:
        return None
    candidate = cache_dir / library
    return candidate if candidate.exists() else None


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
            }
        )
    return {
        "backend": hardware.get("backend"),
        "target_arch": hardware.get("target_arch"),
        "cuda_visible_devices": hardware.get("cuda_visible_devices", ""),
        "devices": devices,
        "nvcc": dict(hardware.get("nvcc", {})) if isinstance(hardware.get("nvcc"), Mapping) else {},
    }


def _support_libraries_cache_payload(libraries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payloads = []
    for library in libraries:
        payloads.append(
            {
                "name": library.get("name"),
                "artifact_sha256": library.get("artifact_sha256"),
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


def _profile_key_payload(
    workload: GemmProfileWorkload | ConvProfileWorkload,
    manifest: Mapping[str, Any],
    kernel_manifest: Mapping[str, Any],
    codegen_plan: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_supported_profile_workload(workload, context="profile cache key")
    profile_context = context or _profile_context(Path("."), manifest, codegen_plan)
    if workload.kernel_library == "cutlass_conv":
        return {
            "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
            "target": manifest["target"],
            "hardware_fingerprint_key": profile_context["fingerprint"]["hardware_key"],
            "support_libraries_fingerprint_key": profile_context["fingerprint"]["support_libraries_key"],
            "support_cache_key": kernel_manifest.get("support_cache_key"),
            "codegen_plan_cache_key": codegen_plan["cache_key"],
            "kernel_library": workload.kernel_library,
            "op": workload.op,
            "dtype": workload.dtype,
            "layouts": dict(workload.candidate.get("layouts", {})),
            "semantic_layout": dict(workload.semantic_layout),
            "provider_layout": dict(workload.provider_layout),
            "layout_translation": dict(workload.layout_translation),
            "weight_transform": dict(workload.weight_transform),
            "conv_config": dict(workload.conv_config),
            "shape": {
                "input": list(workload.x_shape),
                "weight": list(workload.weight_shape),
                "bias": list(workload.bias_shape),
                "output": list(workload.output_shape),
                "source": workload.shape_source,
                "case_id": workload.shape_case_id,
                "dims": dict(workload.dim_values),
            },
            "profile_variant": {
                "kind": str(workload.candidate.get("status", "")),
                "profiler_status": str(workload.candidate.get("profiler_status", "")),
            },
            "kernel_symbol": workload.kernel_symbol,
            "profiler_symbol": workload.profiler_symbol,
            "candidate_set_id": workload.candidate_set_id,
            "candidate_set_key": workload.candidate_set_key,
            "candidate_id": workload.candidate_id,
            "candidate_config_key": workload.candidate_config_key,
        }
    shape = {"m": workload.m, "n": workload.n, "k": workload.k}
    if workload.batch_count is not None:
        shape["batch_count"] = int(workload.batch_count)
    return {
        "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": manifest["target"],
        "hardware_fingerprint_key": profile_context["fingerprint"]["hardware_key"],
        "support_libraries_fingerprint_key": profile_context["fingerprint"]["support_libraries_key"],
        "support_cache_key": kernel_manifest.get("support_cache_key"),
        "codegen_plan_cache_key": codegen_plan["cache_key"],
        "kernel_library": workload.kernel_library,
        "op": workload.op,
        "dtype": workload.dtype,
        "layouts": dict(workload.candidate.get("layouts", {})),
        "epilogue": workload.candidate.get("epilogue"),
        "epilogue_config": workload.candidate.get("epilogue_config"),
        "alignment_context": dict(workload.alignment_context),
        "shape": shape,
        "profile_variant": {"split_k": workload.split_k},
        "split_k": workload.split_k,
        "kernel_symbol": workload.kernel_symbol,
        "profiler_symbol": workload.profiler_symbol,
        "candidate_set_id": workload.candidate_set_id,
        "candidate_set_key": workload.candidate_set_key,
        "candidate_id": workload.candidate_id,
        "candidate_config_key": workload.candidate_config_key,
    }


def _profile_key(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _read_profile_cache(path: Path, target: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": PROFILE_CACHE_SCHEMA_VERSION, "target": dict(target), "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": PROFILE_CACHE_SCHEMA_VERSION, "target": dict(target), "entries": {}}
    if payload.get("schema_version") != PROFILE_CACHE_SCHEMA_VERSION:
        return {"schema_version": PROFILE_CACHE_SCHEMA_VERSION, "target": dict(target), "entries": {}}
    if payload.get("target") != dict(target):
        return {"schema_version": PROFILE_CACHE_SCHEMA_VERSION, "target": dict(target), "entries": {}}
    entries = payload.get("entries", {})
    if not isinstance(entries, Mapping):
        return {"schema_version": PROFILE_CACHE_SCHEMA_VERSION, "target": dict(target), "entries": {}}
    payload["entries"] = _valid_profile_cache_entries(entries, target=target)
    return payload


def _write_profile_cache(path: Path, cache: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = cache.get("target")
    payload = {
        "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": dict(target) if isinstance(target, Mapping) else {},
        "entries": {},
    }
    merged_entries: dict[str, Any] = {}
    if isinstance(target, Mapping):
        existing = _read_profile_cache(path, target)
        merged_entries.update(existing.get("entries", {}))
    entries = cache.get("entries", {})
    if isinstance(entries, Mapping):
        merged_entries.update(_valid_profile_cache_entries(entries, target=target))
    payload["entries"] = merged_entries
    write_json(path, payload)


def _valid_profile_cache_entries(
    entries: Mapping[str, Any],
    *,
    target: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    valid: dict[str, dict[str, Any]] = {}
    expected_target = dict(target) if isinstance(target, Mapping) else None
    for key, value in entries.items():
        if not isinstance(value, Mapping):
            continue
        entry = dict(value)
        entry_key = entry.get("profile_key")
        normalized_key = str(key)
        if not isinstance(entry_key, str) or entry_key != normalized_key:
            continue
        key_payload = entry.get("key")
        if key_payload is not None:
            if not isinstance(key_payload, Mapping):
                continue
            if key_payload.get("schema_version") != PROFILE_CACHE_SCHEMA_VERSION:
                continue
            if expected_target is not None and key_payload.get("target") != expected_target:
                continue
            if _profile_key(key_payload) != normalized_key:
                continue
        valid[normalized_key] = entry
    return valid


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
    if workload.kernel_library == "cutlass_conv":
        payload.update(
            {
                "layout_translation": dict(workload.layout_translation),
                "weight_transform": dict(workload.weight_transform),
                "conv_config": dict(workload.conv_config),
                "candidate_config_key": workload.candidate_config_key,
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
    if best.get("kernel_library") != "cutlass_conv":
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
        "confidence": _selection_confidence(best, runner_up),
    }


def _execution_plan_problem_shape(problem: Mapping[str, Any]) -> dict[str, Any] | None:
    if problem.get("kernel_library") == "cutlass_conv":
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


class _CudaProfiler:
    def __init__(self, artifact_dir: Path, manifest: Mapping[str, Any]):
        files = manifest["files"]
        global_mode = getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(ctypes, "RTLD_NOW", 0)
        self._runtime = ctypes.CDLL(str(artifact_dir / files["runtime_library"]), mode=global_mode)
        self._cuda_runtime = ctypes.CDLL(str(artifact_dir / files["cuda_runtime_library"]), mode=global_mode)
        self._cutlass_gemm = (
            ctypes.CDLL(str(artifact_dir / files["cutlass_gemm_library"]), mode=global_mode)
            if "cutlass_gemm_library" in files
            else None
        )
        self._cutlass_bmm = (
            ctypes.CDLL(str(artifact_dir / files["cutlass_bmm_library"]), mode=global_mode)
            if "cutlass_bmm_library" in files
            else None
        )
        self._cutlass_conv = (
            ctypes.CDLL(str(artifact_dir / files["cutlass_conv_library"]), mode=global_mode)
            if "cutlass_conv_library" in files
            else None
        )
        self._buffers: list[ctypes.c_void_p] = []
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
            return self._profile_bmm(workload, iterations=iterations, rng=rng)
        if workload.kernel_library == "cutlass_conv":
            return self._profile_conv(workload, iterations=iterations, rng=rng)
        if workload.kernel_library != "cutlass_gemm":
            raise RuntimeError(f"Unsupported profiler library {workload.kernel_library!r}")
        return self.profile_gemm(workload, iterations=iterations, rng=rng)

    def profile_gemm(self, workload: GemmProfileWorkload, *, iterations: int, rng: np.random.Generator) -> tuple[float, int]:
        a = self._device_array(_random_storage(workload.a_shape, workload.dtype, rng))
        b = self._device_array(_random_storage(workload.b_shape, workload.dtype, rng))
        bias = (
            self._device_array(_random_storage(workload.bias_shape, workload.dtype, rng))
            if workload.bias_shape is not None
            else None
        )
        residuals = [
            self._device_array(_random_storage(shape, workload.dtype, rng))
            for shape in workload.residual_shapes
        ]
        c = self._device_array(_zero_storage(workload.output_shape, workload.dtype))
        split_k = int(workload.split_k)
        workspace_nbytes = int(workload.workspace_nbytes)
        workspace = ctypes.c_void_p(0)
        profiler_symbol = workload.profiler_symbol
        cutlass = self._cutlass_library("cutlass_gemm")
        if split_k > 1:
            if not _cutlass_split_k_supported(workload.candidate):
                raise RuntimeError(f"CUTLASS candidate {workload.candidate_id} does not support split-K profiling")
            profiler_symbol = _cutlass_split_k_profiler_symbol(workload.profiler_symbol)
            workspace_nbytes = max(workspace_nbytes, self._cutlass_workspace_nbytes(workload, cutlass))
            if workspace_nbytes > 0:
                workspace = self._device_buffer(workspace_nbytes)
        fn = getattr(cutlass, profiler_symbol)
        pointer_args = [ctypes.c_void_p, ctypes.c_void_p]
        call_args = [a, b]
        if bias is not None:
            pointer_args.append(ctypes.c_void_p)
            call_args.append(bias)
        for residual in residuals:
            pointer_args.append(ctypes.c_void_p)
            call_args.append(residual)
        pointer_args.append(ctypes.c_void_p)
        call_args.append(c)
        fn.restype = ctypes.c_float
        if split_k > 1:
            fn.argtypes = [
                *pointer_args,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            elapsed_ms = float(
                fn(
                    *call_args,
                    ctypes.c_int(workload.m),
                    ctypes.c_int(workload.n),
                    ctypes.c_int(workload.k),
                    ctypes.c_int(split_k),
                    workspace,
                    ctypes.c_size_t(workspace_nbytes),
                    ctypes.c_int(iterations),
                    ctypes.c_void_p(0),
                )
            )
        else:
            fn.argtypes = [
                *pointer_args,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            elapsed_ms = float(
                fn(
                    *call_args,
                    ctypes.c_int(workload.m),
                    ctypes.c_int(workload.n),
                    ctypes.c_int(workload.k),
                    ctypes.c_int(iterations),
                    ctypes.c_void_p(0),
                )
            )
        if elapsed_ms < 0.0:
            raise RuntimeError(f"CUTLASS profiler {profiler_symbol} failed")
        return elapsed_ms, workspace_nbytes

    def _profile_bmm(self, workload: GemmProfileWorkload, *, iterations: int, rng: np.random.Generator) -> tuple[float, int]:
        a = self._device_array(_random_storage(workload.a_shape, workload.dtype, rng))
        b = self._device_array(_random_storage(workload.b_shape, workload.dtype, rng))
        residuals = [
            self._device_array(_random_storage(shape, workload.dtype, rng))
            for shape in workload.residual_shapes
        ]
        c = self._device_array(_zero_storage(workload.output_shape, workload.dtype))
        fn = getattr(self._cutlass_library("cutlass_bmm"), workload.profiler_symbol)
        fn.restype = ctypes.c_float
        if residuals:
            if len(residuals) != 1:
                raise RuntimeError(f"CUTLASS BMM profiler only supports one add epilogue input for {workload.op}")
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            elapsed_ms = float(
                fn(
                    a,
                    b,
                    residuals[0],
                    c,
                    ctypes.c_int(int(workload.batch_count or 1)),
                    ctypes.c_int(workload.m),
                    ctypes.c_int(workload.n),
                    ctypes.c_int(workload.k),
                    ctypes.c_int64(int(workload.batch_stride_a or 0)),
                    ctypes.c_int64(int(workload.batch_stride_b or 0)),
                    ctypes.c_int64(int(workload.batch_stride_d0 or 0)),
                    ctypes.c_int64(int(workload.batch_stride_c or 0)),
                    ctypes.c_int(int(workload.lda or 0)),
                    ctypes.c_int(int(workload.ldb or 0)),
                    ctypes.c_int(int(workload.ldd0 or 0)),
                    ctypes.c_int(int(workload.ldc or 0)),
                    ctypes.c_int(iterations),
                    ctypes.c_void_p(0),
                )
            )
        else:
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            elapsed_ms = float(
                fn(
                    a,
                    b,
                    c,
                    ctypes.c_int(int(workload.batch_count or 1)),
                    ctypes.c_int(workload.m),
                    ctypes.c_int(workload.n),
                    ctypes.c_int(workload.k),
                    ctypes.c_int64(int(workload.batch_stride_a or 0)),
                    ctypes.c_int64(int(workload.batch_stride_b or 0)),
                    ctypes.c_int64(int(workload.batch_stride_c or 0)),
                    ctypes.c_int(int(workload.lda or 0)),
                    ctypes.c_int(int(workload.ldb or 0)),
                    ctypes.c_int(int(workload.ldc or 0)),
                    ctypes.c_int(iterations),
                    ctypes.c_void_p(0),
                )
            )
        if elapsed_ms < 0.0:
            raise RuntimeError(f"CUTLASS profiler {workload.profiler_symbol} failed")
        return elapsed_ms, int(workload.workspace_nbytes)

    def _profile_conv(self, workload: ConvProfileWorkload, *, iterations: int, rng: np.random.Generator) -> tuple[float, int]:
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
        activation = self._device_array(_random_storage((n, h, w, c), workload.dtype, rng))
        weight = self._device_array(_random_storage((weight_o, kernel_h, kernel_w, weight_i), workload.dtype, rng))
        bias = self._device_array(_random_storage(workload.bias_shape, workload.dtype, rng))
        residual = (
            self._device_array(_random_storage((out_n, out_h, out_w, out_c), workload.dtype, rng))
            if workload.residual_shape is not None
            else None
        )
        output = self._device_array(_zero_storage((out_n, out_h, out_w, out_c), workload.dtype))
        fn = getattr(self._cutlass_library("cutlass_conv"), workload.profiler_symbol)
        fn.restype = ctypes.c_float
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            *( [ctypes.c_void_p] if residual is not None else [] ),
            ctypes.c_void_p,
            *([ctypes.c_int] * 16),
            ctypes.c_void_p,
        ]
        elapsed_ms = float(
            fn(
                activation,
                weight,
                bias,
                *( [residual] if residual is not None else [] ),
                output,
                ctypes.c_int(n),
                ctypes.c_int(h),
                ctypes.c_int(w),
                ctypes.c_int(c),
                ctypes.c_int(out_h),
                ctypes.c_int(out_w),
                ctypes.c_int(out_c),
                ctypes.c_int(kernel_h),
                ctypes.c_int(kernel_w),
                ctypes.c_int(stride[0]),
                ctypes.c_int(stride[1]),
                ctypes.c_int(padding[0]),
                ctypes.c_int(padding[1]),
                ctypes.c_int(dilation[0]),
                ctypes.c_int(dilation[1]),
                ctypes.c_int(iterations),
                ctypes.c_void_p(0),
            )
        )
        if elapsed_ms < 0.0:
            raise RuntimeError(f"CUTLASS profiler {workload.profiler_symbol} failed")
        return elapsed_ms, int(workload.workspace_nbytes)

    def _cutlass_library(self, library: str) -> ctypes.CDLL:
        if library == "cutlass_gemm" and self._cutlass_gemm is not None:
            return self._cutlass_gemm
        if library == "cutlass_bmm" and self._cutlass_bmm is not None:
            return self._cutlass_bmm
        if library == "cutlass_conv" and self._cutlass_conv is not None:
            return self._cutlass_conv
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
        ptr = ctypes.c_void_p()
        self._check(self._cuda_runtime.dino_device_malloc(ctypes.byref(ptr), ctypes.c_size_t(int(nbytes))))
        self._buffers.append(ptr)
        return ptr

    def _device_array(self, array: np.ndarray) -> ctypes.c_void_p:
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
        getter = getattr(self._cuda_runtime, "dino_get_last_error", None)
        if getter is not None:
            error = getter()
        if not error:
            error = self._runtime.dino_get_last_error()
        message = error.decode("utf-8") if error else f"CUDA profiler helper failed with code {code}"
        raise RuntimeError(message)


def _random_storage(shape: Sequence[int], dtype: str, rng: np.random.Generator) -> np.ndarray:
    values = (rng.standard_normal(tuple(shape)).astype(np.float32) * 0.125)
    return np.ascontiguousarray(array_to_storage(values, dtype))


def _zero_storage(shape: Sequence[int], dtype: str) -> np.ndarray:
    return np.ascontiguousarray(array_to_storage(np.zeros(tuple(shape), dtype=np.float32), dtype))
