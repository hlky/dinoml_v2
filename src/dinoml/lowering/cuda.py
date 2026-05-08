from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.ir import RUNTIME_ABI_VERSION, canonical_json, dtype_runtime_enum
from dinoml.lowering.ops import render_generated_kernels, render_launch
from dinoml.lowering.shape_buffers import (
    dynamic_dim_sources,
    numel_expr,
    shape_buffer_context,
    shape_dim_expr,
    shape_literal,
    shape_vars_literal,
)


def render_cuda_module(ir: Mapping[str, Any]) -> str:
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
    input_map = {item["tensor"]: idx for idx, item in enumerate(ir["inputs"])}
    output_map = {item["tensor"]: idx for idx, item in enumerate(ir["outputs"])}
    constant_tensors = {item["tensor"]: item for item in ir["constants"]}
    temporaries = ir.get("metadata", {}).get("memory_plan", {}).get("temporaries", [])
    metadata = {
        "runtime_abi_version": RUNTIME_ABI_VERSION,
        "name": ir["name"],
        "inputs": ir["inputs"],
        "outputs": ir["outputs"],
        "constants": ir["constants"],
    }

    return render_template(
        "cuda_module.cu.j2",
        {
            "metadata_json": canonical_json(metadata),
            "input_count": len(ir["inputs"]),
            "output_count": len(ir["outputs"]),
            "inputs": [
                _io_context(idx, item["name"], item["shape"], item["dtype"], item.get("shape_spec", item["shape"]))
                for idx, item in enumerate(ir["inputs"])
            ],
            "outputs": [
                _io_context(idx, item["name"], item["shape"], item["dtype"], item.get("shape_spec", item["shape"]))
                for idx, item in enumerate(ir["outputs"])
            ],
            "constants": [_constant_context(item) for item in ir["constants"]],
            "temporaries": [_temporary_context(item) for item in temporaries],
            "shape_buffers": [shape_buffer_context(item) for item in ir["tensors"]],
            "shape_equal_checks": _shape_equal_checks(ir["inputs"], ir["outputs"]),
            "generated_kernels": render_generated_kernels("cuda", ir["nodes"], tensor_map),
            "pointer_decls": list(
                _pointer_decls(
                    input_map=input_map,
                    output_map=output_map,
                    constant_tensors=constant_tensors,
                    temporaries=temporaries,
                    tensor_map=tensor_map,
                )
            ),
            "launches": [render_launch("cuda", node, tensor_map) for node in ir["nodes"]],
        },
    )


def render_template(name: str, context: Mapping[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parents[1] / "templates")),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(name).render(**context)


def _io_context(index: int, name: str, shape: Iterable[int], dtype: str, shape_spec: Iterable[Any]) -> dict[str, Any]:
    dims = _dim_ranges(shape_spec)
    return {
        "index": index,
        "name": name,
        "shape_literal": shape_literal(shape),
        "min_shape_literal": shape_literal(dim["min"] for dim in dims),
        "max_shape_literal": shape_literal(dim["max"] for dim in dims),
        "divisible_by_literal": shape_literal(dim["divisible_by"] for dim in dims),
        "dtype": dtype,
        "dtype_enum": dtype_runtime_enum(dtype),
    }


def _constant_context(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": item["name"],
        "ident": _c_ident(item["tensor"]),
        "offset": int(item["offset"]),
        "nbytes": int(item["nbytes"]),
        "shape_literal": shape_literal(item["shape"]),
        "dtype": item["dtype"],
        "dtype_enum": dtype_runtime_enum(item["dtype"]),
    }


def _temporary_context(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ident": _c_ident(item["tensor"]),
        "nbytes": int(item["nbytes"]),
    }


def _pointer_decls(
    *,
    input_map: Mapping[str, int],
    output_map: Mapping[str, int],
    constant_tensors: Mapping[str, Mapping[str, Any]],
    temporaries: Iterable[Mapping[str, Any]],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> Iterable[str]:
    dynamic_dims = dynamic_dim_sources(input_map=input_map, output_map=output_map, tensor_map=tensor_map)
    for tensor_name, idx in input_map.items():
        ident = _c_ident(tensor_name)
        cpp_type = _cpp_storage_type(str(tensor_map[tensor_name]["dtype"]))
        yield f"const {cpp_type}* ptr_{ident} = static_cast<const {cpp_type}*>(inputs[{idx}].data);"
        for axis in range(len(tensor_map[tensor_name]["shape"])):
            yield f"const int64_t shape_{ident}_{axis} = inputs[{idx}].shape[{axis}];"
        yield f"DINO_CUDA_CHECK(cudaMemcpy(session->shape_{ident}, inputs[{idx}].shape, sizeof(int64_t) * {len(tensor_map[tensor_name]['shape'])}, cudaMemcpyHostToDevice));"
        yield f"const int64_t* shape_{ident} = session->shape_{ident};"
        yield f"const int64_t runtime_numel_{ident} = dinoml::module::tensor_numel(inputs[{idx}]);"
    for tensor_name in constant_tensors:
        ident = _c_ident(tensor_name)
        cpp_type = _cpp_storage_type(str(tensor_map[tensor_name]["dtype"]))
        for axis, dim in enumerate(tensor_map[tensor_name]["shape"]):
            yield f"static constexpr int64_t shape_{ident}_{axis} = {int(dim)};"
        yield f"const int64_t* shape_{ident} = session->shape_{ident};"
        yield (
            f"const {cpp_type}* ptr_{ident} = "
            f"static_cast<const {cpp_type}*>(module->const_{ident});"
        )
    for item in temporaries:
        tensor_name = item["tensor"]
        ident = _c_ident(tensor_name)
        cpp_type = _cpp_storage_type(str(tensor_map[tensor_name]["dtype"]))
        for axis in range(len(tensor_map[tensor_name]["shape"])):
            yield f"const int64_t shape_{ident}_{axis} = {shape_dim_expr(tensor_map[tensor_name], axis, dynamic_dims)};"
        yield f"const int64_t host_shape_{ident}[] = {{ {shape_vars_literal(ident, len(tensor_map[tensor_name]['shape']))} }};"
        yield f"DINO_CUDA_CHECK(cudaMemcpy(session->shape_{ident}, host_shape_{ident}, sizeof(int64_t) * {len(tensor_map[tensor_name]['shape'])}, cudaMemcpyHostToDevice));"
        yield f"const int64_t* shape_{ident} = session->shape_{ident};"
        yield f"const int64_t runtime_numel_{ident} = {numel_expr(ident, len(tensor_map[tensor_name]['shape']))};"
        yield f"{cpp_type}* ptr_{ident} = static_cast<{cpp_type}*>(session->tmp_{ident});"
    for tensor_name, idx in output_map.items():
        ident = _c_ident(tensor_name)
        cpp_type = _cpp_storage_type(str(tensor_map[tensor_name]["dtype"]))
        yield f"{cpp_type}* ptr_{ident} = static_cast<{cpp_type}*>(outputs[{idx}].data);"
        for axis in range(len(tensor_map[tensor_name]["shape"])):
            yield f"const int64_t shape_{ident}_{axis} = outputs[{idx}].shape[{axis}];"
        yield f"DINO_CUDA_CHECK(cudaMemcpy(session->shape_{ident}, outputs[{idx}].shape, sizeof(int64_t) * {len(tensor_map[tensor_name]['shape'])}, cudaMemcpyHostToDevice));"
        yield f"const int64_t* shape_{ident} = session->shape_{ident};"
        yield f"const int64_t runtime_numel_{ident} = dinoml::module::tensor_numel(outputs[{idx}]);"


def _cpp_storage_type(dtype: str) -> str:
    if dtype == "float32":
        return "float"
    if dtype == "float16":
        return "half"
    if dtype == "bfloat16":
        return "__nv_bfloat16"
    raise NotImplementedError(f"CUDA module lowering does not support dtype {dtype!r}")


def _dim_ranges(shape_spec: Iterable[Any]) -> list[dict[str, int]]:
    dims = []
    for dim in shape_spec:
        if isinstance(dim, int):
            dims.append({"min": int(dim), "max": int(dim), "divisible_by": 1})
        else:
            dims.append(
                {
                    "min": int(dim["min"]),
                    "max": int(dim["max"]),
                    "divisible_by": int(dim.get("divisible_by", 1)),
                }
            )
    return dims


def _shape_equal_checks(inputs: Iterable[Mapping[str, Any]], outputs: Iterable[Mapping[str, Any]]) -> list[str]:
    dim_sources: dict[str, tuple[str, int, str]] = {}
    checks: list[str] = []
    indexed_inputs = list(enumerate(list(inputs)))
    indexed_outputs = list(enumerate(list(outputs)))
    for array_name, indexed in (("inputs", indexed_inputs), ("outputs", indexed_outputs)):
        for tensor_idx, item in indexed:
            for axis, dim in enumerate(item.get("shape_spec", item["shape"])):
                if isinstance(dim, int):
                    continue
                name = str(dim["name"])
                expr = f"{array_name}[{tensor_idx}].shape[{axis}]"
                if name not in dim_sources:
                    dim_sources[name] = (expr, tensor_idx, item["name"])
                else:
                    source_expr, _source_idx, source_name = dim_sources[name]
                    checks.append(
                        f'  if ({expr} != {source_expr}) return dinoml::module::fail("Dynamic dimension {name} mismatch between {source_name} and {item["name"]}");'
                    )
    return checks


def _numel(shape: Iterable[int]) -> int:
    return int(np.prod(list(shape), dtype=np.int64))


def _c_ident(name: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"_{ident}"
    return ident
