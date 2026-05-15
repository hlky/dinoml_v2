from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.ir import dtype_nbytes, dtype_runtime_enum
from dinoml.lowering.cpp_types import cpu_storage_type
from dinoml.lowering.ops import render_generated_kernels, render_launch
from dinoml.lowering.shape_buffers import (
    constant_expression_axis_checks,
    dynamic_dim_sources,
    expression_axis_checks,
    named_dim_leaves,
    numel_expr,
    shape_buffer_context,
    shape_dim_expr,
    shape_dim_range,
    shape_literal,
    shape_vars_literal,
    validate_symbolic_int_sources,
)


def render_cpu_module(ir: Mapping[str, Any], *, generated_kernels: Iterable[str] | None = None) -> str:
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
    input_map = {item["tensor"]: idx for idx, item in enumerate(ir["inputs"])}
    output_map = {item["tensor"]: idx for idx, item in enumerate(ir["outputs"])}
    constant_tensors = {item["tensor"]: item for item in ir["constants"]}
    temporaries = ir.get("metadata", {}).get("memory_plan", {}).get("temporaries", [])
    views = _view_contexts(ir, output_map=output_map, tensor_map=tensor_map)
    dynamic_dims = dynamic_dim_sources(input_map=input_map, output_map=output_map, tensor_map=tensor_map)
    validate_symbolic_int_sources(items=ir["inputs"], dynamic_dims=dynamic_dims, context="input")
    validate_symbolic_int_sources(items=ir["outputs"], dynamic_dims=dynamic_dims, context="output")
    validate_symbolic_int_sources(items=ir["constants"], dynamic_dims=dynamic_dims, context="constant")
    validate_symbolic_int_sources(items=tensor_map.values(), dynamic_dims=dynamic_dims, context="tensor")
    return render_template(
        "cpu_module.cpp.j2",
        {
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
            "temporaries": [_temporary_context(item, tensor_map) for item in temporaries],
            "shape_buffers": [shape_buffer_context(item) for item in ir["tensors"]],
            "shape_equal_checks": _shape_equal_checks(ir["inputs"], ir["outputs"], ir["constants"]),
            "generated_kernels": list(generated_kernels)
            if generated_kernels is not None
            else render_generated_kernels("cpu", ir["nodes"], tensor_map),
            "pointer_decls": list(
                _pointer_decls(
                    input_map=input_map,
                    output_map=output_map,
                    constant_tensors=constant_tensors,
                    temporaries=temporaries,
                    views=views,
                    tensor_map=tensor_map,
                )
            ),
            "launches": [render_launch("cpu", node, tensor_map) for node in ir["nodes"]],
            "output_materializations": _output_materializations(views),
            "output_shape_reports": _output_shape_report_contexts(ir, tensor_map=tensor_map),
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
    dims = _dim_ranges(item.get("shape_spec", item["shape"]))
    storage = item.get("storage")
    autoload_from_constants_bin = not (
        isinstance(storage, Mapping)
        and storage.get("kind") == "gguf"
        and str(storage.get("residency", "eager_dense_device")) == "manual_runtime_load"
    )
    return {
        "name": item["name"],
        "ident": _c_ident(item["tensor"]),
        "offset": int(item["offset"]),
        "nbytes": int(item["nbytes"]),
        "numel": _numel(item["shape"]),
        "shape_literal": shape_literal(item["shape"]),
        "min_shape_literal": shape_literal(dim["min"] for dim in dims),
        "max_shape_literal": shape_literal(dim["max"] for dim in dims),
        "divisible_by_literal": shape_literal(dim["divisible_by"] for dim in dims),
        "dtype": item["dtype"],
        "dtype_enum": dtype_runtime_enum(item["dtype"]),
        "dtype_nbytes": dtype_nbytes(item["dtype"]),
        "storage_type": cpu_storage_type(str(item["dtype"])),
        "autoload_from_constants_bin": autoload_from_constants_bin,
    }


def _temporary_context(item: Mapping[str, Any], tensor_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    nbytes = int(item["nbytes"])
    dtype = str(tensor_map[item["tensor"]]["dtype"])
    return {
        "ident": _c_ident(item["tensor"]),
        "nbytes": nbytes,
        "numel": nbytes // dtype_nbytes(dtype),
        "storage_type": _temporary_storage_type(dtype),
    }


def _pointer_decls(
    *,
    input_map: Mapping[str, int],
    output_map: Mapping[str, int],
    constant_tensors: Mapping[str, Mapping[str, Any]],
    temporaries: Iterable[Mapping[str, Any]],
    views: Iterable[Mapping[str, Any]],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> Iterable[str]:
    dynamic_dims = dynamic_dim_sources(input_map=input_map, output_map=output_map, tensor_map=tensor_map)
    view_by_tensor = {str(view["tensor"]): view for view in views}
    for tensor_name, idx in input_map.items():
        ident = _c_ident(tensor_name)
        cpp_type = cpu_storage_type(str(tensor_map[tensor_name]["dtype"]))
        yield f"const {cpp_type}* ptr_{ident} = static_cast<const {cpp_type}*>(dinoml::module::tensor_data(inputs[{idx}]));"
        for axis in range(len(tensor_map[tensor_name]["shape"])):
            yield f"const int64_t shape_{ident}_{axis} = inputs[{idx}].shape[{axis}];"
        yield f"session->shape_{ident}.assign(inputs[{idx}].shape, inputs[{idx}].shape + {len(tensor_map[tensor_name]['shape'])});"
        yield f"const int64_t* shape_{ident} = session->shape_{ident}.data();"
        yield f"const int64_t runtime_numel_{ident} = dinoml::module::tensor_numel(inputs[{idx}]);"
    for tensor_name in constant_tensors:
        ident = _c_ident(tensor_name)
        cpp_type = cpu_storage_type(str(tensor_map[tensor_name]["dtype"]))
        rank = len(tensor_map[tensor_name]["shape"])
        for axis in range(rank):
            yield f"const int64_t shape_{ident}_{axis} = module->const_shape_{ident}[{axis}];"
        yield f"session->shape_{ident} = module->const_shape_{ident};"
        yield f"const int64_t* shape_{ident} = session->shape_{ident}.data();"
        yield f"const int64_t runtime_numel_{ident} = {numel_expr(ident, rank)};"
        yield f"const {cpp_type}* ptr_{ident} = module->const_{ident}.data();"
    for item in temporaries:
        tensor_name = item["tensor"]
        ident = _c_ident(tensor_name)
        cpp_type = cpu_storage_type(str(tensor_map[tensor_name]["dtype"]))
        for axis in range(len(tensor_map[tensor_name]["shape"])):
            yield f"const int64_t shape_{ident}_{axis} = {shape_dim_expr(tensor_map[tensor_name], axis, dynamic_dims)};"
        yield f"session->shape_{ident} = std::vector<int64_t>{{ {shape_vars_literal(ident, len(tensor_map[tensor_name]['shape']))} }};"
        yield f"const int64_t* shape_{ident} = session->shape_{ident}.data();"
        yield f"const int64_t runtime_numel_{ident} = {numel_expr(ident, len(tensor_map[tensor_name]['shape']))};"
        if str(tensor_map[tensor_name]["dtype"]) == "bool":
            yield f"{cpp_type}* ptr_{ident} = reinterpret_cast<{cpp_type}*>(session->tmp_{ident}.data());"
        else:
            yield f"{cpp_type}* ptr_{ident} = session->tmp_{ident}.data();"
    for tensor_name, idx in output_map.items():
        if tensor_name in view_by_tensor:
            continue
        ident = _c_ident(tensor_name)
        cpp_type = cpu_storage_type(str(tensor_map[tensor_name]["dtype"]))
        yield f"{cpp_type}* ptr_{ident} = static_cast<{cpp_type}*>(dinoml::module::tensor_data(outputs[{idx}]));"
        for axis in range(len(tensor_map[tensor_name]["shape"])):
            yield f"const int64_t shape_{ident}_{axis} = outputs[{idx}].shape[{axis}];"
        yield f"session->shape_{ident}.assign(outputs[{idx}].shape, outputs[{idx}].shape + {len(tensor_map[tensor_name]['shape'])});"
        yield f"const int64_t* shape_{ident} = session->shape_{ident}.data();"
        yield f"const int64_t runtime_numel_{ident} = dinoml::module::tensor_numel(outputs[{idx}]);"
    for view in views:
        tensor_name = str(view["tensor"])
        source_name = str(view["source"])
        ident = _c_ident(tensor_name)
        source_ident = _c_ident(source_name)
        cpp_type = cpu_storage_type(str(tensor_map[tensor_name]["dtype"]))
        output_idx = view.get("output_index")
        if output_idx is None:
            for axis in range(len(tensor_map[tensor_name]["shape"])):
                yield f"const int64_t shape_{ident}_{axis} = {shape_dim_expr(tensor_map[tensor_name], axis, dynamic_dims)};"
            yield f"session->shape_{ident} = std::vector<int64_t>{{ {shape_vars_literal(ident, len(tensor_map[tensor_name]['shape']))} }};"
        else:
            for axis in range(len(tensor_map[tensor_name]["shape"])):
                yield f"const int64_t shape_{ident}_{axis} = outputs[{int(output_idx)}].shape[{axis}];"
            yield f"session->shape_{ident}.assign(outputs[{int(output_idx)}].shape, outputs[{int(output_idx)}].shape + {len(tensor_map[tensor_name]['shape'])});"
        yield f"const int64_t* shape_{ident} = session->shape_{ident}.data();"
        yield f"const int64_t runtime_numel_{ident} = {numel_expr(ident, len(tensor_map[tensor_name]['shape']))};"
        yield f"const {cpp_type}* ptr_{ident} = ptr_{source_ident};"


def _view_contexts(
    ir: Mapping[str, Any],
    *,
    output_map: Mapping[str, int],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw_views = ir.get("metadata", {}).get("memory_plan", {}).get("views", {}).get("views", [])
    view_tensors = {view["tensor"] for view in raw_views}
    view_sources = {view["source"] for view in raw_views}
    view_of_view = sorted(view_tensors & view_sources)
    if view_of_view:
        raise NotImplementedError(
            "View-of-view aliases are not supported by CPU lowering; "
            f"view tensors used as view sources: {view_of_view}"
        )
    node_view_outputs = sorted(
        output_name
        for node in ir["nodes"]
        for output_name in node["outputs"]
        if output_name in view_tensors
    )
    if node_view_outputs:
        raise NotImplementedError(f"View alias tensors cannot be kernel outputs: {node_view_outputs}")
    contexts = []
    for view in raw_views:
        tensor_name = str(view["tensor"])
        cpp_type = cpu_storage_type(str(tensor_map[tensor_name]["dtype"]))
        contexts.append(
            {
                "tensor": tensor_name,
                "ident": _c_ident(tensor_name),
                "source": str(view["source"]),
                "source_ident": _c_ident(str(view["source"])),
                "output_index": output_map.get(tensor_name),
                "nbytes_expr": f"runtime_numel_{_c_ident(tensor_name)} * sizeof({cpp_type})",
            }
        )
    return contexts


def _output_materializations(views: Iterable[Mapping[str, Any]]) -> list[str]:
    materializations = []
    for view in views:
        output_idx = view.get("output_index")
        if output_idx is None:
            continue
        materializations.append(
            f"std::memcpy(dinoml::module::tensor_data(outputs[{int(output_idx)}]), ptr_{view['ident']}, {view['nbytes_expr']});"
        )
    return materializations


def _output_shape_report_contexts(
    ir: Mapping[str, Any],
    *,
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    shape_buffer_outputs = {
        str(report["output"])
        for report in ir.get("metadata", {}).get("output_shape_reports", {}).get("reports", [])
        if isinstance(report, Mapping) and report.get("kind") == "shape_buffer"
    }
    reports = []
    for idx, output in enumerate(ir["outputs"]):
        tensor_name = str(output["tensor"])
        reports.append(
            {
                "index": idx,
                "name": str(output["name"]),
                "ident": _c_ident(tensor_name),
                "rank": len(tensor_map[tensor_name]["shape"]),
                "source": "shape_buffer" if str(output["name"]) in shape_buffer_outputs else "caller",
            }
        )
    return reports


def _temporary_storage_type(dtype: str) -> str:
    if dtype == "bool":
        return "uint8_t"
    return cpu_storage_type(dtype)


def _dim_ranges(shape_spec: Iterable[Any]) -> list[dict[str, int]]:
    return [shape_dim_range(dim) for dim in shape_spec]


def _shape_equal_checks(
    inputs: Iterable[Mapping[str, Any]],
    outputs: Iterable[Mapping[str, Any]],
    constants: Iterable[Mapping[str, Any]],
) -> list[str]:
    dim_sources: dict[str, tuple[str, int, str]] = {}
    checks: list[str] = []
    indexed_inputs = list(enumerate(list(inputs)))
    indexed_outputs = list(enumerate(list(outputs)))
    for array_name, indexed in (("inputs", indexed_inputs), ("outputs", indexed_outputs)):
        for tensor_idx, item in indexed:
            for axis, dim in enumerate(item.get("shape_spec", item["shape"])):
                if isinstance(dim, Mapping) and dim.get("kind") == "dim":
                    _append_named_dim_check(
                        dim_sources,
                        checks,
                        name=str(dim["name"]),
                        expr=f"{array_name}[{tensor_idx}].shape[{axis}]",
                        item_name=str(item["name"]),
                    )
                elif isinstance(dim, Mapping) and dim.get("kind") == "int_expr":
                    for leaf in named_dim_leaves(dim):
                        if str(leaf["name"]) in dim_sources:
                            continue
    for item in constants:
        ident = _c_ident(str(item["tensor"]))
        for axis, dim in enumerate(item.get("shape_spec", item["shape"])):
            if isinstance(dim, Mapping) and dim.get("kind") == "dim":
                _append_named_dim_check(
                    dim_sources,
                    checks,
                    name=str(dim["name"]),
                    expr=f"session->module->const_shape_{ident}[{axis}]",
                    item_name=str(item["name"]),
                )
            elif isinstance(dim, Mapping) and dim.get("kind") == "int_expr":
                for leaf in named_dim_leaves(dim):
                    if str(leaf["name"]) in dim_sources:
                        continue
    dynamic_dims = {name: source[0] for name, source in dim_sources.items()}
    checks.extend(expression_axis_checks(items=inputs, array_name="inputs", dynamic_dims=dynamic_dims))
    checks.extend(expression_axis_checks(items=outputs, array_name="outputs", dynamic_dims=dynamic_dims))
    checks.extend(
        constant_expression_axis_checks(constants=constants, dynamic_dims=dynamic_dims, c_ident_fn=_c_ident)
    )
    return checks


def _append_named_dim_check(
    dim_sources: dict[str, tuple[str, int, str]],
    checks: list[str],
    *,
    name: str,
    expr: str,
    item_name: str,
) -> None:
    if name not in dim_sources:
        dim_sources[name] = (expr, -1, item_name)
        return
    source_expr, _source_idx, source_name = dim_sources[name]
    checks.append(
        f'  if ({expr} != {source_expr}) return dinoml::module::fail("Dynamic dimension {name} mismatch between {source_name} and {item_name}");'
    )


def _numel(shape: Iterable[int]) -> int:
    return int(np.prod(list(shape), dtype=np.int64))


def _c_ident(name: str) -> str:
    ident = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not ident or ident[0].isdigit():
        ident = f"_{ident}"
    ident = re.sub(r"_(\d+)$", r"__\1", ident)
    return ident
