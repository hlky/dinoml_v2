from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from dinoml.ir import dtype_nbytes, dtype_runtime_enum
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_split_k_supported
from dinoml.lowering.cpp_types import cuda_storage_type
from dinoml.lowering.ops import render_generated_kernels, render_launch
from dinoml.lowering.shape_buffers import (
    dynamic_dim_sources,
    numel_expr,
    shape_buffer_context,
    shape_dim_expr,
    shape_literal,
    shape_vars_literal,
)


def render_cuda_module(
    ir: Mapping[str, Any],
    *,
    generated_kernels: Iterable[str] | None = None,
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}
    input_map = {item["tensor"]: idx for idx, item in enumerate(ir["inputs"])}
    output_map = {item["tensor"]: idx for idx, item in enumerate(ir["outputs"])}
    constant_tensors = {item["tensor"]: item for item in ir["constants"]}
    temporaries = ir.get("metadata", {}).get("memory_plan", {}).get("temporaries", [])
    views = _view_contexts(ir, output_map=output_map, tensor_map=tensor_map)
    return render_template(
        "cuda_module.cu.j2",
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
            "temporaries": [_temporary_context(item) for item in temporaries],
            "shape_buffers": [shape_buffer_context(item) for item in ir["tensors"]],
            "shape_equal_checks": _shape_equal_checks(ir["inputs"], ir["outputs"], ir["constants"]),
            "generated_kernels": list(generated_kernels)
            if generated_kernels is not None
            else render_generated_kernels("cuda", ir["nodes"], tensor_map),
            "external_kernel_declarations": _external_kernel_declarations(kernel_manifest),
            "cutlass_workspace": _cutlass_workspace_context(kernel_manifest),
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
            "launches": [render_launch("cuda", node, tensor_map, kernel_manifest=kernel_manifest) for node in ir["nodes"]],
            "output_materializations": _output_materializations(views),
        },
    )


def _external_kernel_declarations(kernel_manifest: Mapping[str, Any] | None) -> list[str]:
    if kernel_manifest is None:
        return []
    declarations = []
    seen = set()
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") not in {"cutlass_gemm", "cutlass_bmm"}:
            continue
        for declaration in _cutlass_item_declarations(item):
            symbol = declaration["symbol"]
            if symbol in seen:
                continue
            cpp_type = cuda_storage_type(str(declaration["dtype"]))
            if item.get("kernel_library") == "cutlass_bmm":
                declarations.append(
                    _cutlass_bmm_declaration(
                        symbol,
                        cpp_type,
                        str(declaration["launch_abi"]),
                    )
                )
            else:
                declarations.append(
                    _cutlass_gemm_declaration(
                        symbol,
                        cpp_type,
                        str(declaration["launch_abi"]),
                        str(declaration["epilogue"]),
                        split_k=bool(declaration["split_k"]),
                    )
                )
            seen.add(symbol)
    return declarations


def _selected_candidate(item: Mapping[str, Any]) -> Mapping[str, Any]:
    selected_id = item.get("selected_candidate_id")
    candidates = [candidate for candidate in item.get("candidates", []) if isinstance(candidate, Mapping)]
    for candidate in candidates:
        if candidate.get("candidate_id") == selected_id:
            return candidate
    if candidates:
        return candidates[0]
    return {}


def _candidate_by_id(item: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    for candidate in item.get("candidates", []):
        if isinstance(candidate, Mapping) and str(candidate.get("candidate_id")) == candidate_id:
            return candidate
    return {}


def _cutlass_item_declarations(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    declarations = []
    selected = _selected_candidate(item)
    declarations.append(
        _cutlass_declaration_context(
            item,
            selected,
            str(item["kernel_symbol"]),
            split_k=_cutlass_item_split_k(item),
        )
    )
    for selection in item.get("execution_plan_dispatch", ()):
        if not isinstance(selection, Mapping):
            continue
        candidate = _candidate_by_id(item, str(selection.get("selected_candidate_id", "")))
        declarations.append(
            _cutlass_declaration_context(
                item,
                candidate,
                str(selection.get("kernel_symbol") or candidate.get("kernel_symbol")),
                split_k=int(selection.get("split_k", 1) or 1),
            )
        )
    for fallback in item.get("alignment_fallbacks", ()):
        if not isinstance(fallback, Mapping):
            continue
        candidate = _candidate_by_id(item, str(fallback.get("candidate_id", "")))
        declarations.append(
            _cutlass_declaration_context(
                item,
                candidate,
                str(fallback.get("kernel_symbol") or candidate.get("kernel_symbol")),
                split_k=1,
            )
        )
    return declarations


def _cutlass_declaration_context(
    item: Mapping[str, Any],
    candidate: Mapping[str, Any],
    symbol: str,
    *,
    split_k: int,
) -> dict[str, Any]:
    launch_abi = str(candidate.get("launch_abi") or item.get("candidate_set", {}).get("launch_abi"))
    epilogue = str(candidate.get("epilogue") or item.get("candidate_set", {}).get("epilogue"))
    dtype = str(candidate.get("dtype") or item.get("candidate_set", {}).get("dtype"))
    return {
        "symbol": _cutlass_split_k_kernel_symbol(symbol) if split_k > 1 else symbol,
        "dtype": dtype,
        "launch_abi": launch_abi,
        "epilogue": epilogue,
        "split_k": split_k > 1,
    }


def _cutlass_gemm_declaration(
    symbol: str,
    cpp_type: str,
    launch_abi: str,
    epilogue: str,
    *,
    split_k: bool = False,
) -> str:
    if split_k and not cutlass_gemm_split_k_supported({"launch_abi": launch_abi, "epilogue": epilogue}):
        raise ValueError(f"Unsupported CUTLASS split-K epilogue/launch ABI: {epilogue!r} / {launch_abi!r}")
    extra_args = ""
    if launch_abi == "dinoml_cutlass_gemm_bias_v1":
        extra_args = f"    const {cpp_type}* bias,\n"
    elif launch_abi == "dinoml_cutlass_gemm_bias_residual_v1":
        extra_args = f"    const {cpp_type}* bias,\n" f"    const {cpp_type}* d0,\n"
    elif launch_abi == "dinoml_cutlass_gemm_bias_residual2_v1":
        extra_args = f"    const {cpp_type}* bias,\n" f"    const {cpp_type}* d0,\n" f"    const {cpp_type}* d1,\n"
    elif launch_abi != "dinoml_cutlass_gemm_v1":
        raise ValueError(f"Unsupported CUTLASS GEMM launch ABI: {launch_abi!r}")
    launch_args = (
        "    int split_k,\n"
        "    void* workspace,\n"
        "    size_t workspace_nbytes,\n"
        "    cudaStream_t stream);"
        if split_k
        else "    cudaStream_t stream);"
    )
    return (
        f"extern \"C\" int {symbol}(\n"
        f"    const {cpp_type}* a,\n"
        f"    const {cpp_type}* b,\n"
        f"{extra_args}"
        f"    {cpp_type}* c,\n"
        "    int m,\n"
        "    int n,\n"
        "    int k,\n"
        f"{launch_args}"
    )


def _cutlass_bmm_declaration(
    symbol: str,
    cpp_type: str,
    launch_abi: str,
) -> str:
    if launch_abi == "dinoml_cutlass_bmm_add_v1":
        return (
            f"extern \"C\" int {symbol}(\n"
            f"    const {cpp_type}* a,\n"
            f"    const {cpp_type}* b,\n"
            f"    const {cpp_type}* d0,\n"
            f"    {cpp_type}* c,\n"
            "    int batch_count,\n"
            "    int m,\n"
            "    int n,\n"
            "    int k,\n"
            "    int64_t batch_stride_a,\n"
            "    int64_t batch_stride_b,\n"
            "    int64_t batch_stride_d0,\n"
            "    int64_t batch_stride_c,\n"
            "    int lda,\n"
            "    int ldb,\n"
            "    int ldd0,\n"
            "    int ldc,\n"
            "    cudaStream_t stream);"
        )
    if launch_abi != "dinoml_cutlass_bmm_v1":
        raise ValueError(f"Unsupported CUTLASS BMM launch ABI: {launch_abi!r}")
    return (
        f"extern \"C\" int {symbol}(\n"
        f"    const {cpp_type}* a,\n"
        f"    const {cpp_type}* b,\n"
        f"    {cpp_type}* c,\n"
        "    int batch_count,\n"
        "    int m,\n"
        "    int n,\n"
        "    int k,\n"
        "    int64_t batch_stride_a,\n"
        "    int64_t batch_stride_b,\n"
        "    int64_t batch_stride_c,\n"
        "    int lda,\n"
        "    int ldb,\n"
        "    int ldc,\n"
        "    cudaStream_t stream);"
    )


def _cutlass_workspace_context(kernel_manifest: Mapping[str, Any] | None) -> dict[str, int] | None:
    if kernel_manifest is None:
        return None
    max_workspace = 0
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "cutlass_gemm":
            continue
        for selection in _cutlass_workspace_selections(item):
            if int(selection.get("split_k", 1) or 1) <= 1:
                continue
            max_workspace = max(max_workspace, int(selection.get("workspace_nbytes", 0) or 0))
    if max_workspace <= 0:
        return None
    return {"nbytes": max_workspace}


def _cutlass_workspace_selections(item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    selections = []
    selection = item.get("execution_plan_selection")
    if isinstance(selection, Mapping):
        selections.append(selection)
    selections.extend(selection for selection in item.get("execution_plan_dispatch", ()) if isinstance(selection, Mapping))
    return selections


def _cutlass_declaration_symbol(item: Mapping[str, Any]) -> str:
    symbol = str(item["kernel_symbol"])
    if _cutlass_item_split_k(item) > 1:
        return _cutlass_split_k_kernel_symbol(symbol)
    return symbol


def _cutlass_item_split_k(item: Mapping[str, Any]) -> int:
    selection = item.get("execution_plan_selection")
    if not isinstance(selection, Mapping):
        return 1
    return int(selection.get("split_k", 1) or 1)


def _cutlass_split_k_kernel_symbol(symbol: str) -> str:
    prefix = "dinoml_cutlass_"
    if not symbol.startswith(prefix):
        raise ValueError(f"Unsupported CUTLASS kernel symbol for split-K: {symbol!r}")
    return f"dinoml_cutlass_splitk_{symbol[len(prefix):]}"


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
    return {
        "name": item["name"],
        "ident": _c_ident(item["tensor"]),
        "offset": int(item["offset"]),
        "nbytes": int(item["nbytes"]),
        "shape_literal": shape_literal(item["shape"]),
        "min_shape_literal": shape_literal(dim["min"] for dim in dims),
        "max_shape_literal": shape_literal(dim["max"] for dim in dims),
        "divisible_by_literal": shape_literal(dim["divisible_by"] for dim in dims),
        "dtype": item["dtype"],
        "dtype_enum": dtype_runtime_enum(item["dtype"]),
        "dtype_nbytes": dtype_nbytes(item["dtype"]),
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
    views: Iterable[Mapping[str, Any]],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> Iterable[str]:
    dynamic_dims = dynamic_dim_sources(input_map=input_map, output_map=output_map, tensor_map=tensor_map)
    view_by_tensor = {str(view["tensor"]): view for view in views}
    for tensor_name, idx in input_map.items():
        ident = _c_ident(tensor_name)
        cpp_type = cuda_storage_type(str(tensor_map[tensor_name]["dtype"]))
        yield f"const DinoTensor* abi_{ident} = &inputs[{idx}];"
        yield f"const {cpp_type}* ptr_{ident} = static_cast<const {cpp_type}*>(dinoml::module::tensor_data(inputs[{idx}]));"
        for axis in range(len(tensor_map[tensor_name]["shape"])):
            yield f"const int64_t shape_{ident}_{axis} = inputs[{idx}].shape[{axis}];"
        yield f"DINO_CUDA_CHECK(cudaMemcpy(session->shape_{ident}, inputs[{idx}].shape, sizeof(int64_t) * {len(tensor_map[tensor_name]['shape'])}, cudaMemcpyHostToDevice));"
        yield f"const int64_t* shape_{ident} = session->shape_{ident};"
        yield f"const int64_t runtime_numel_{ident} = dinoml::module::tensor_numel(inputs[{idx}]);"
    for tensor_name in constant_tensors:
        ident = _c_ident(tensor_name)
        cpp_type = cuda_storage_type(str(tensor_map[tensor_name]["dtype"]))
        rank = len(tensor_map[tensor_name]["shape"])
        yield f"const DinoTensor* abi_{ident} = nullptr;"
        for axis in range(rank):
            yield f"const int64_t shape_{ident}_{axis} = module->const_shape_{ident}[{axis}];"
        yield f"DINO_CUDA_CHECK(cudaMemcpy(session->shape_{ident}, module->const_shape_{ident}.data(), sizeof(int64_t) * {rank}, cudaMemcpyHostToDevice));"
        yield f"const int64_t* shape_{ident} = session->shape_{ident};"
        yield f"const int64_t runtime_numel_{ident} = {numel_expr(ident, rank)};"
        yield (
            f"const {cpp_type}* ptr_{ident} = "
            f"static_cast<const {cpp_type}*>(module->const_{ident});"
        )
    for item in temporaries:
        tensor_name = item["tensor"]
        ident = _c_ident(tensor_name)
        cpp_type = cuda_storage_type(str(tensor_map[tensor_name]["dtype"]))
        yield f"const DinoTensor* abi_{ident} = nullptr;"
        for axis in range(len(tensor_map[tensor_name]["shape"])):
            yield f"const int64_t shape_{ident}_{axis} = {shape_dim_expr(tensor_map[tensor_name], axis, dynamic_dims)};"
        yield f"const int64_t host_shape_{ident}[] = {{ {shape_vars_literal(ident, len(tensor_map[tensor_name]['shape']))} }};"
        yield f"DINO_CUDA_CHECK(cudaMemcpy(session->shape_{ident}, host_shape_{ident}, sizeof(int64_t) * {len(tensor_map[tensor_name]['shape'])}, cudaMemcpyHostToDevice));"
        yield f"const int64_t* shape_{ident} = session->shape_{ident};"
        yield f"const int64_t runtime_numel_{ident} = {numel_expr(ident, len(tensor_map[tensor_name]['shape']))};"
        yield f"{cpp_type}* ptr_{ident} = static_cast<{cpp_type}*>(session->tmp_{ident});"
    for tensor_name, idx in output_map.items():
        if tensor_name in view_by_tensor:
            continue
        ident = _c_ident(tensor_name)
        cpp_type = cuda_storage_type(str(tensor_map[tensor_name]["dtype"]))
        yield f"const DinoTensor* abi_{ident} = &outputs[{idx}];"
        yield f"{cpp_type}* ptr_{ident} = static_cast<{cpp_type}*>(dinoml::module::tensor_data(outputs[{idx}]));"
        for axis in range(len(tensor_map[tensor_name]["shape"])):
            yield f"const int64_t shape_{ident}_{axis} = outputs[{idx}].shape[{axis}];"
        yield f"DINO_CUDA_CHECK(cudaMemcpy(session->shape_{ident}, outputs[{idx}].shape, sizeof(int64_t) * {len(tensor_map[tensor_name]['shape'])}, cudaMemcpyHostToDevice));"
        yield f"const int64_t* shape_{ident} = session->shape_{ident};"
        yield f"const int64_t runtime_numel_{ident} = dinoml::module::tensor_numel(outputs[{idx}]);"
    for view in views:
        tensor_name = str(view["tensor"])
        source_name = str(view["source"])
        ident = _c_ident(tensor_name)
        source_ident = _c_ident(source_name)
        cpp_type = cuda_storage_type(str(tensor_map[tensor_name]["dtype"]))
        output_idx = view.get("output_index")
        yield f"const DinoTensor* abi_{ident} = abi_{source_ident};"
        if output_idx is None:
            for axis in range(len(tensor_map[tensor_name]["shape"])):
                yield f"const int64_t shape_{ident}_{axis} = {shape_dim_expr(tensor_map[tensor_name], axis, dynamic_dims)};"
            yield f"const int64_t host_shape_{ident}[] = {{ {shape_vars_literal(ident, len(tensor_map[tensor_name]['shape']))} }};"
            yield f"DINO_CUDA_CHECK(cudaMemcpy(session->shape_{ident}, host_shape_{ident}, sizeof(int64_t) * {len(tensor_map[tensor_name]['shape'])}, cudaMemcpyHostToDevice));"
        else:
            for axis in range(len(tensor_map[tensor_name]["shape"])):
                yield f"const int64_t shape_{ident}_{axis} = outputs[{int(output_idx)}].shape[{axis}];"
            yield f"DINO_CUDA_CHECK(cudaMemcpy(session->shape_{ident}, outputs[{int(output_idx)}].shape, sizeof(int64_t) * {len(tensor_map[tensor_name]['shape'])}, cudaMemcpyHostToDevice));"
        yield f"const int64_t* shape_{ident} = session->shape_{ident};"
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
            "View-of-view aliases are not supported by CUDA lowering; "
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
        cpp_type = cuda_storage_type(str(tensor_map[tensor_name]["dtype"]))
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
            "DINO_CUDA_CHECK(cudaMemcpyAsync("
            f"dinoml::module::tensor_data(outputs[{int(output_idx)}]), ptr_{view['ident']}, {view['nbytes_expr']}, "
            "cudaMemcpyDeviceToDevice, session->stream));"
        )
    return materializations


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
    for item in constants:
        ident = _c_ident(str(item["tensor"]))
        for axis, dim in enumerate(item.get("shape_spec", item["shape"])):
            if isinstance(dim, int):
                continue
            name = str(dim["name"])
            expr = f"session->module->const_shape_{ident}[{axis}]"
            if name not in dim_sources:
                dim_sources[name] = (expr, -1, item["name"])
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
