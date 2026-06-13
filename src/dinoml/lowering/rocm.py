from __future__ import annotations

from typing import Any, Iterable, Mapping

from dinoml.lowering.cpp_types import rocm_storage_type
from dinoml.lowering.gpu import _candidate_by_id, _selected_candidate, render_gpu_module


def render_rocm_module(
    ir: Mapping[str, Any],
    *,
    generated_kernels: Iterable[str] | None = None,
    generated_kernel_declarations: Iterable[str] | None = None,
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    return render_gpu_module(
        "rocm",
        ir,
        generated_kernels=generated_kernels,
        kernel_manifest=kernel_manifest,
        external_kernel_declarations=[
            *(list(generated_kernel_declarations) if generated_kernel_declarations is not None else []),
            *_external_kernel_declarations(kernel_manifest),
        ],
    )


def _external_kernel_declarations(kernel_manifest: Mapping[str, Any] | None) -> list[str]:
    if kernel_manifest is None:
        return []
    declarations = []
    seen = set()
    for item in kernel_manifest.get("required_kernels", []):
        if item.get("kernel_library") == "ck_gemm":
            item_declarations = _ck_gemm_item_declarations(item)
            declaration_fn = _ck_gemm_declaration
        elif item.get("kernel_library") == "ck_bmm":
            item_declarations = _ck_bmm_item_declarations(item)
            declaration_fn = _ck_bmm_declaration
        elif item.get("kernel_library") == "ck_conv":
            item_declarations = _ck_conv_item_declarations(item)
            declaration_fn = _ck_conv_declaration
        else:
            continue
        for declaration in item_declarations:
            symbol = declaration["symbol"]
            if symbol in seen:
                continue
            declarations.append(
                declaration_fn(
                    symbol,
                    rocm_storage_type(str(declaration["dtype"])),
                    str(declaration["launch_abi"]),
                )
            )
            seen.add(symbol)
    return declarations


def _ck_bmm_item_declarations(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    declarations = []
    selected = _selected_candidate(item)
    declarations.append(_ck_bmm_declaration_context(item, selected, str(item["kernel_symbol"])))
    for selection in item.get("execution_plan_dispatch", ()):
        if not isinstance(selection, Mapping):
            continue
        candidate = _candidate_by_id(item, str(selection.get("selected_candidate_id", "")))
        declarations.append(
            _ck_bmm_declaration_context(
                item,
                candidate,
                str(selection.get("kernel_symbol") or candidate.get("kernel_symbol")),
            )
        )
    return declarations


def _ck_conv_item_declarations(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    declarations = []
    selected = _selected_candidate(item)
    declarations.append(_ck_conv_declaration_context(item, selected, str(item["kernel_symbol"])))
    for selection in item.get("execution_plan_dispatch", ()):
        if not isinstance(selection, Mapping):
            continue
        candidate = _candidate_by_id(item, str(selection.get("selected_candidate_id", "")))
        declarations.append(
            _ck_conv_declaration_context(
                item,
                candidate,
                str(selection.get("kernel_symbol") or candidate.get("kernel_symbol")),
            )
        )
    return declarations


def _ck_conv_declaration_context(
    item: Mapping[str, Any],
    candidate: Mapping[str, Any],
    symbol: str,
) -> dict[str, Any]:
    launch_abi = str(candidate.get("launch_abi") or item.get("candidate_set", {}).get("launch_abi"))
    dtype = str(candidate.get("dtype") or item.get("candidate_set", {}).get("dtype"))
    return {
        "symbol": symbol,
        "dtype": dtype,
        "launch_abi": launch_abi,
    }


def _ck_bmm_declaration_context(
    item: Mapping[str, Any],
    candidate: Mapping[str, Any],
    symbol: str,
) -> dict[str, Any]:
    launch_abi = str(candidate.get("launch_abi") or item.get("candidate_set", {}).get("launch_abi"))
    dtype = str(candidate.get("dtype") or item.get("candidate_set", {}).get("dtype"))
    return {
        "symbol": symbol,
        "dtype": dtype,
        "launch_abi": launch_abi,
    }


def _ck_gemm_item_declarations(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    declarations = []
    selected = _selected_candidate(item)
    declarations.append(_ck_gemm_declaration_context(item, selected, str(item["kernel_symbol"])))
    for selection in item.get("execution_plan_dispatch", ()):
        if not isinstance(selection, Mapping):
            continue
        candidate = _candidate_by_id(item, str(selection.get("selected_candidate_id", "")))
        declarations.append(
            _ck_gemm_declaration_context(
                item,
                candidate,
                str(selection.get("kernel_symbol") or candidate.get("kernel_symbol")),
            )
        )
    for fallback in item.get("alignment_fallbacks", ()):
        if not isinstance(fallback, Mapping):
            continue
        candidate = _candidate_by_id(item, str(fallback.get("candidate_id", "")))
        declarations.append(
            _ck_gemm_declaration_context(
                item,
                candidate,
                str(fallback.get("kernel_symbol") or candidate.get("kernel_symbol")),
            )
        )
    return declarations


def _ck_gemm_declaration_context(
    item: Mapping[str, Any],
    candidate: Mapping[str, Any],
    symbol: str,
) -> dict[str, Any]:
    launch_abi = str(candidate.get("launch_abi") or item.get("candidate_set", {}).get("launch_abi"))
    dtype = str(candidate.get("dtype") or item.get("candidate_set", {}).get("dtype"))
    return {
        "symbol": symbol,
        "dtype": dtype,
        "launch_abi": launch_abi,
    }


def _ck_gemm_declaration(symbol: str, cpp_type: str, launch_abi: str) -> str:
    extra_args = ""
    shape_args = ""
    if launch_abi == "dinoml_ck_dual_gemm_v1":
        extra_args = f"    const {cpp_type}* b1,\n"
        shape_args = "    int b1_n,\n"
    elif launch_abi == "dinoml_ck_dual_gemm_bias_v1":
        extra_args = (
            f"    const {cpp_type}* b1,\n"
            f"    const {cpp_type}* bias0,\n"
            f"    const {cpp_type}* bias1,\n"
        )
        shape_args = "    int b1_n,\n"
    elif launch_abi == "dinoml_ck_gemm_bias_v1":
        extra_args = f"    const {cpp_type}* bias,\n"
    elif launch_abi == "dinoml_ck_gemm_bias_residual_v1":
        extra_args = f"    const {cpp_type}* bias,\n" f"    const {cpp_type}* d0,\n"
    elif launch_abi == "dinoml_ck_gemm_bias_residual2_v1":
        extra_args = (
            f"    const {cpp_type}* bias,\n"
            f"    const {cpp_type}* d0,\n"
            f"    const {cpp_type}* d1,\n"
        )
    elif launch_abi != "dinoml_ck_gemm_v1":
        raise ValueError(f"Unsupported CK GEMM launch ABI: {launch_abi!r}")
    return (
        f'extern "C" int {symbol}(\n'
        f"    const {cpp_type}* a,\n"
        f"    const {cpp_type}* b,\n"
        f"{extra_args}"
        f"    {cpp_type}* c,\n"
        "    int m,\n"
        "    int n,\n"
        "    int k,\n"
        f"{shape_args}"
        "    hipStream_t stream);"
    )


def _ck_bmm_declaration(symbol: str, cpp_type: str, launch_abi: str) -> str:
    extra_args = ""
    extra_strides = ""
    extra_lds = ""
    if launch_abi == "dinoml_ck_bmm_add_v1":
        extra_args = f"    const {cpp_type}* d0,\n"
        extra_strides = "    int64_t batch_stride_d0,\n"
        extra_lds = "    int ldd0,\n"
    elif launch_abi != "dinoml_ck_bmm_v1":
        raise ValueError(f"Unsupported CK BMM launch ABI: {launch_abi!r}")
    return (
        f'extern "C" int {symbol}(\n'
        f"    const {cpp_type}* a,\n"
        f"    const {cpp_type}* b,\n"
        f"{extra_args}"
        f"    {cpp_type}* c,\n"
        "    int batch_count,\n"
        "    int m,\n"
        "    int n,\n"
        "    int k,\n"
        "    int64_t batch_stride_a,\n"
        "    int64_t batch_stride_b,\n"
        f"{extra_strides}"
        "    int64_t batch_stride_c,\n"
        "    int lda,\n"
        "    int ldb,\n"
        f"{extra_lds}"
        "    int ldc,\n"
        "    hipStream_t stream);"
    )


def _ck_conv_declaration(symbol: str, cpp_type: str, launch_abi: str) -> str:
    bias_arg = ""
    residual_arg = ""
    extra_shape_args = ""
    weight_pack_arg = ""
    if launch_abi in {"dinoml_ck_conv1d_bias_v1", "dinoml_ck_conv1d_bias_relu_v1"}:
        bias_arg = f"    const {cpp_type}* bias,\n"
        return (
            f'extern "C" int {symbol}(\n'
            f"    const {cpp_type}* x,\n"
            f"    const {cpp_type}* weight,\n"
            f"{bias_arg}"
            f"    {cpp_type}* output,\n"
            "    int batch,\n"
            "    int in_channels,\n"
            "    int in_width,\n"
            "    int out_channels,\n"
            "    int kernel_w,\n"
            "    int out_width,\n"
            "    int stride_w,\n"
            "    int pad_w,\n"
            "    int dilation_w,\n"
            "    int weight_is_kxc,\n"
            "    hipStream_t stream);"
        )
    elif launch_abi in {"dinoml_ck_conv1d_bias_add_v1", "dinoml_ck_conv1d_bias_add_relu_v1"}:
        bias_arg = f"    const {cpp_type}* bias,\n"
        residual_arg = f"    const {cpp_type}* residual,\n"
        return (
            f'extern "C" int {symbol}(\n'
            f"    const {cpp_type}* x,\n"
            f"    const {cpp_type}* weight,\n"
            f"{bias_arg}"
            f"{residual_arg}"
            f"    {cpp_type}* output,\n"
            "    int batch,\n"
            "    int in_channels,\n"
            "    int in_width,\n"
            "    int out_channels,\n"
            "    int kernel_w,\n"
            "    int out_width,\n"
            "    int stride_w,\n"
            "    int pad_w,\n"
            "    int dilation_w,\n"
            "    int weight_is_kxc,\n"
            "    hipStream_t stream);"
        )
    elif launch_abi in {"dinoml_ck_conv2d_bias_v1", "dinoml_ck_conv2d_bias_relu_v1"}:
        bias_arg = f"    const {cpp_type}* bias,\n"
        weight_pack_arg = "    int weight_is_kyxc,\n"
    elif launch_abi == "dinoml_ck_conv3d_bias_v1":
        bias_arg = f"    const {cpp_type}* bias,\n"
        return (
            f'extern "C" int {symbol}(\n'
            f"    const {cpp_type}* x,\n"
            f"    const {cpp_type}* weight,\n"
            f"{bias_arg}"
            f"    {cpp_type}* output,\n"
            "    int batch,\n"
            "    int in_channels,\n"
            "    int in_depth,\n"
            "    int in_height,\n"
            "    int in_width,\n"
            "    int out_channels,\n"
            "    int kernel_d,\n"
            "    int kernel_h,\n"
            "    int kernel_w,\n"
            "    int out_depth,\n"
            "    int out_height,\n"
            "    int out_width,\n"
            "    int stride_d,\n"
            "    int stride_h,\n"
            "    int stride_w,\n"
            "    int pad_d,\n"
            "    int pad_h,\n"
            "    int pad_w,\n"
            "    int dilation_d,\n"
            "    int dilation_h,\n"
            "    int dilation_w,\n"
            "    int weight_is_kzyxc,\n"
            "    int groups,\n"
            "    hipStream_t stream);"
        )
    elif launch_abi in {"dinoml_ck_conv2d_bias_add_v1", "dinoml_ck_conv2d_bias_add_relu_v1"}:
        bias_arg = f"    const {cpp_type}* bias,\n"
        residual_arg = f"    const {cpp_type}* residual,\n"
        weight_pack_arg = "    int weight_is_kyxc,\n"
    elif launch_abi in {
        "dinoml_ck_transposed_conv2d_v1",
        "dinoml_ck_transposed_conv2d_bias_v1",
        "dinoml_ck_transposed_conv2d_bias_relu_v1",
        "dinoml_ck_transposed_conv2d_bias_add_v1",
        "dinoml_ck_transposed_conv2d_bias_add_relu_v1",
    }:
        if launch_abi != "dinoml_ck_transposed_conv2d_v1":
            bias_arg = f"    const {cpp_type}* bias,\n"
        if launch_abi in {"dinoml_ck_transposed_conv2d_bias_add_v1", "dinoml_ck_transposed_conv2d_bias_add_relu_v1"}:
            residual_arg = f"    const {cpp_type}* residual,\n"
        extra_shape_args = "    int output_pad_h,\n    int output_pad_w,\n"
    elif launch_abi not in {
        "dinoml_ck_conv2d_bias_v1",
        "dinoml_ck_conv2d_bias_relu_v1",
        "dinoml_ck_conv2d_bias_add_v1",
        "dinoml_ck_conv2d_bias_add_relu_v1",
        "dinoml_ck_transposed_conv2d_v1",
        "dinoml_ck_transposed_conv2d_bias_v1",
        "dinoml_ck_transposed_conv2d_bias_relu_v1",
        "dinoml_ck_transposed_conv2d_bias_add_v1",
        "dinoml_ck_transposed_conv2d_bias_add_relu_v1",
    }:
        raise ValueError(f"Unsupported CK Conv launch ABI: {launch_abi!r}")
    return (
        f'extern "C" int {symbol}(\n'
        f"    const {cpp_type}* x,\n"
        f"    const {cpp_type}* weight,\n"
        f"{bias_arg}"
        f"{residual_arg}"
        f"    {cpp_type}* output,\n"
        "    int batch,\n"
        "    int in_channels,\n"
        "    int in_height,\n"
        "    int in_width,\n"
        "    int out_channels,\n"
        "    int kernel_h,\n"
        "    int kernel_w,\n"
        "    int out_height,\n"
        "    int out_width,\n"
        "    int stride_h,\n"
        "    int stride_w,\n"
        "    int pad_h,\n"
        "    int pad_w,\n"
        f"{extra_shape_args}"
        "    int dilation_h,\n"
        "    int dilation_w,\n"
        f"{weight_pack_arg}"
        "    hipStream_t stream);"
    )
