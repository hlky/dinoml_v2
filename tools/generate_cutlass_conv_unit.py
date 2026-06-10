from __future__ import annotations

import argparse
from pathlib import Path

from dinoml.kernels.providers.cutlass.conv import (
    CONV_OPS,
    CONV_SUPPORTED_DTYPES,
    CUTLASS_CONV1D_OPS,
    CUTLASS_TRANSPOSED_CONV_OPS,
    cutlass_conv_candidates,
    cutlass_conv1d_input_pack_symbol,
    cutlass_conv1d_output_unpack_symbol,
    cutlass_conv1d_weight_pack_symbol,
    cutlass_conv_input_pack_symbol,
    cutlass_conv_output_unpack_symbol,
    cutlass_conv_weight_pack_symbol,
    cutlass_transposed_conv_candidates,
    cutlass_transposed_conv_weight_pack_symbol,
    render_cutlass_conv_source,
)


def _conv1d_transform_helpers(dtype: str) -> list[dict[str, str]]:
    return [
        {
            "symbol": cutlass_conv1d_input_pack_symbol(dtype),
            "dtype": dtype,
            "tensor_role": "activation",
            "transform": "ncw_to_nwc_temporary",
            "layout_from": "ncw",
            "layout_to": "nwc",
            "shape_order": ["n", "c", "w"],
            "helper_abi": "dinoml_cutlass_layout_transform_v1",
        },
        {
            "symbol": cutlass_conv1d_weight_pack_symbol(dtype),
            "dtype": dtype,
            "tensor_role": "weight",
            "transform": "oiw_to_owi_temporary",
            "layout_from": "oiw",
            "layout_to": "owi",
            "shape_order": ["o", "i", "w"],
            "helper_abi": "dinoml_cutlass_layout_transform_v1",
        },
        {
            "symbol": cutlass_conv1d_output_unpack_symbol(dtype),
            "dtype": dtype,
            "tensor_role": "output",
            "transform": "nwc_to_ncw_temporary",
            "layout_from": "nwc",
            "layout_to": "ncw",
            "shape_order": ["n", "c", "w"],
            "helper_abi": "dinoml_cutlass_layout_transform_v1",
        },
    ]


def _transform_helpers(dtype: str) -> list[dict[str, str]]:
    return [
        {
            "symbol": cutlass_conv_input_pack_symbol(dtype),
            "dtype": dtype,
            "tensor_role": "activation",
            "transform": "nchw_to_nhwc_temporary",
            "layout_from": "nchw",
            "layout_to": "nhwc",
            "shape_order": ["n", "c", "h", "w"],
            "helper_abi": "dinoml_cutlass_layout_transform_v1",
        },
        {
            "symbol": cutlass_conv_weight_pack_symbol(dtype),
            "dtype": dtype,
            "tensor_role": "weight",
            "transform": "oihw_to_ohwi_temporary",
            "layout_from": "oihw",
            "layout_to": "ohwi",
            "shape_order": ["o", "i", "h", "w"],
            "helper_abi": "dinoml_cutlass_layout_transform_v1",
        },
        {
            "symbol": cutlass_conv_output_unpack_symbol(dtype),
            "dtype": dtype,
            "tensor_role": "output",
            "transform": "nhwc_to_nchw_temporary",
            "layout_from": "nhwc",
            "layout_to": "nchw",
            "shape_order": ["n", "c", "h", "w"],
            "helper_abi": "dinoml_cutlass_layout_transform_v1",
        },
    ]


def _transposed_transform_helpers(dtype: str) -> list[dict[str, str]]:
    return [
        {
            "symbol": cutlass_conv_input_pack_symbol(dtype),
            "dtype": dtype,
            "tensor_role": "activation",
            "transform": "nchw_to_nhwc_temporary",
            "layout_from": "nchw",
            "layout_to": "nhwc",
            "shape_order": ["n", "c", "h", "w"],
            "helper_abi": "dinoml_cutlass_layout_transform_v1",
        },
        {
            "symbol": cutlass_transposed_conv_weight_pack_symbol(dtype),
            "dtype": dtype,
            "tensor_role": "weight",
            "transform": "iohw_to_ihwo_temporary",
            "layout_from": "iohw",
            "layout_to": "ihwo",
            "shape_order": ["i", "o", "h", "w"],
            "helper_abi": "dinoml_cutlass_layout_transform_v1",
        },
        {
            "symbol": cutlass_conv_output_unpack_symbol(dtype),
            "dtype": dtype,
            "tensor_role": "output",
            "transform": "nhwc_to_nchw_temporary",
            "layout_from": "nhwc",
            "layout_to": "nchw",
            "shape_order": ["n", "c", "h", "w"],
            "helper_abi": "dinoml_cutlass_layout_transform_v1",
        },
    ]


def render_cutlass_conv_unit(op: str, dtype: str, source: Path) -> str:
    if op not in {*CONV_OPS, *CUTLASS_TRANSPOSED_CONV_OPS}:
        raise ValueError(f"Unsupported CUTLASS Conv op {op!r}")
    if dtype not in CONV_SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported CUTLASS Conv dtype {dtype!r}")
    if op in CUTLASS_TRANSPOSED_CONV_OPS:
        candidates = cutlass_transposed_conv_candidates(op, dtype, target={"name": "cuda", "arch": "sm_80"})
        transform_helpers = _transposed_transform_helpers(dtype)
    elif op in CUTLASS_CONV1D_OPS:
        candidates = cutlass_conv_candidates(op, dtype, target={"name": "cuda", "arch": "sm_80"})
        transform_helpers = _conv1d_transform_helpers(dtype)
    else:
        candidates = cutlass_conv_candidates(op, dtype, target={"name": "cuda", "arch": "sm_80"})
        transform_helpers = _transform_helpers(dtype)
    used_plan = {
        "transform_helpers": transform_helpers,
        "candidates": candidates,
        "profiler_symbols": [str(candidate["profiler_symbol"]) for candidate in candidates],
    }
    return render_cutlass_conv_source(source.read_text(encoding="utf-8"), used_plan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rendered = render_cutlass_conv_unit(args.op, args.dtype, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
