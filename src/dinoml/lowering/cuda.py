from __future__ import annotations

from typing import Any, Iterable, Mapping


from dinoml.kernels.providers.cutlass.conv import cutlass_conv_wrapper_stages
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_split_k_supported
from dinoml.lowering.cpp_types import cuda_storage_type
from dinoml.lowering.gpu import (
    _candidate_by_id,
    _selected_candidate,
    _shape_args,
    render_gpu_module,
)
from dinoml.lowering.shape_buffers import (
    c_ident as _c_ident,
)


def render_cuda_module(
    ir: Mapping[str, Any],
    *,
    generated_kernels: Iterable[str] | None = None,
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    return render_gpu_module(
        "cuda",
        ir,
        generated_kernels=generated_kernels,
        kernel_manifest=kernel_manifest,
        cutlass_conv_temporaries=_cutlass_conv_temporary_contexts(kernel_manifest),
        external_kernel_declarations=_external_kernel_declarations(kernel_manifest),
        cutlass_workspace=_cutlass_workspace_context(kernel_manifest),
    )


def _external_kernel_declarations(
    kernel_manifest: Mapping[str, Any] | None,
) -> list[str]:
    if kernel_manifest is None:
        return []
    declarations = []
    seen = set()
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") not in {
            "cutlass_gemm",
            "cutlass_bmm",
            "cutlass_conv",
        }:
            continue
        if item.get("kernel_library") == "cutlass_conv":
            conv_manifest = {"required_kernels": [item]}
            for declaration in _cutlass_conv_declarations(
                cutlass_conv_wrapper_stages(conv_manifest)
            ):
                symbol = declaration["symbol"]
                if symbol in seen:
                    continue
                declarations.append(declaration["source"])
                seen.add(symbol)
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


def _cutlass_conv_declarations(
    stages: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    declarations = []
    for stage in stages:
        symbol = str(stage.get("symbol", ""))
        if not symbol:
            continue
        stage_kind = str(stage.get("stage_kind", ""))
        if stage_kind == "transform_helper":
            args = [
                "    const void* src,",
                "    void* dst,",
                *["    int," for _ in _shape_args(stage)],
                "    cudaStream_t stream);",
            ]
        elif stage_kind == "provider_launcher":
            inputs = stage.get("inputs")
            if not isinstance(inputs, (list, tuple)) or len(inputs) < 3:
                continue
            args = [
                "    const void* activation_nhwc,",
                "    const void* weight_ohwi,",
                "    const void* bias,",
                *(["    const void* residual_nhwc,"] if len(inputs) > 3 else []),
                "    void* output_nhwc,",
                *["    int," for _ in _shape_args(stage)],
                "    cudaStream_t stream);",
            ]
        else:
            continue
        declarations.append(
            {
                "symbol": symbol,
                "source": f'extern "C" int {symbol}(\n' + "\n".join(args),
            }
        )
    return declarations


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
        candidate = _candidate_by_id(
            item, str(selection.get("selected_candidate_id", ""))
        )
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
    launch_abi = str(
        candidate.get("launch_abi") or item.get("candidate_set", {}).get("launch_abi")
    )
    epilogue = str(
        candidate.get("epilogue") or item.get("candidate_set", {}).get("epilogue")
    )
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
    if split_k and not cutlass_gemm_split_k_supported(
        {"launch_abi": launch_abi, "epilogue": epilogue}
    ):
        raise ValueError(
            f"Unsupported CUTLASS split-K epilogue/launch ABI: {epilogue!r} / {launch_abi!r}"
        )
    extra_args = ""
    if launch_abi == "dinoml_cutlass_gemm_bias_v1":
        extra_args = f"    const {cpp_type}* bias,\n"
    elif launch_abi == "dinoml_cutlass_gemm_bias_residual_v1":
        extra_args = f"    const {cpp_type}* bias,\n" f"    const {cpp_type}* d0,\n"
    elif launch_abi == "dinoml_cutlass_gemm_bias_residual2_v1":
        extra_args = (
            f"    const {cpp_type}* bias,\n"
            f"    const {cpp_type}* d0,\n"
            f"    const {cpp_type}* d1,\n"
        )
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
        f'extern "C" int {symbol}(\n'
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
            f'extern "C" int {symbol}(\n'
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
        f'extern "C" int {symbol}(\n'
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


def _cutlass_workspace_context(
    kernel_manifest: Mapping[str, Any] | None,
) -> dict[str, int] | None:
    if kernel_manifest is None:
        return None
    max_workspace = 0
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") != "cutlass_gemm":
            continue
        for selection in _cutlass_workspace_selections(item):
            if int(selection.get("split_k", 1) or 1) <= 1:
                continue
            max_workspace = max(
                max_workspace, int(selection.get("workspace_nbytes", 0) or 0)
            )
    if max_workspace <= 0:
        return None
    return {"nbytes": max_workspace}


def _cutlass_conv_temporary_contexts(
    kernel_manifest: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if kernel_manifest is None:
        return []
    contexts = []
    seen = set()
    for item in kernel_manifest.get("required_kernels", []):
        if (
            not isinstance(item, Mapping)
            or item.get("kernel_library") != "cutlass_conv"
        ):
            continue
        plan = item.get("cutlass_conv_plan")
        if not isinstance(plan, Mapping):
            continue
        node_id = str(
            plan.get("node_id") or item.get("node_id") or item.get("op") or "conv"
        )
        for buffer in plan.get("temporary_buffers", ()):
            if not isinstance(buffer, Mapping):
                continue
            name = str(buffer.get("name", ""))
            nbytes = int(buffer.get("nbytes", 0) or 0)
            if not name or nbytes <= 0:
                continue
            ident = f"{_c_ident(node_id)}_{_c_ident(name)}"
            if ident in seen:
                continue
            seen.add(ident)
            contexts.append(
                {
                    "ident": ident,
                    "node_id": node_id,
                    "name": name,
                    "layout": str(buffer.get("layout", "")),
                    "nbytes": nbytes,
                }
            )
    return contexts


def _cutlass_workspace_selections(item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    selections = []
    selection = item.get("execution_plan_selection")
    if isinstance(selection, Mapping):
        selections.append(selection)
    selections.extend(
        selection
        for selection in item.get("execution_plan_dispatch", ())
        if isinstance(selection, Mapping)
    )
    return selections


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
