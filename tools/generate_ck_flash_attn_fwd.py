#!/usr/bin/env python3
"""Generate a DinoML-selected subset of CK FMHA forward kernels.

This wraps CK's forward codegen without modifying third_party sources. We keep
using CK's own kernel objects and API emitter, but apply DinoML's exact forward
selection locally instead of relying on CK's single-pattern fnmatch filter.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path
from typing import Iterable, List


def _matches_extra_filter(name: str, extra_filter: str) -> bool:
    if extra_filter == "":
        return True
    return any(
        fnmatch.fnmatch(name, pattern.strip())
        for pattern in extra_filter.split("|")
        if pattern.strip()
    )


def _is_masked(mask_impl: str, mask_value: str) -> bool:
    if mask_impl == "simplified":
        return mask_value != "s_no"
    if mask_impl == "generic":
        return mask_value != "no"
    raise ValueError(f"Unsupported mask implementation: {mask_impl}")


def _is_unmasked(mask_impl: str, mask_value: str) -> bool:
    return not _is_masked(mask_impl, mask_value)


def _want_kernel(kernel, mask_impl: str, extra_filter: str) -> bool:
    if kernel.F_dtype not in {"fp16", "bf16"}:
        return False

    pipeline = kernel.F_pipeline
    if pipeline.F_logits != "f" or pipeline.F_lse != "f" or pipeline.F_dropout != "f":
        return False

    if kernel.F_mode not in {"batch", "group"}:
        return False

    if pipeline.F_bias == "bias":
        if not _is_masked(mask_impl, pipeline.F_mask):
            return False
    elif pipeline.F_bias == "no":
        pass
    else:
        return False

    return _matches_extra_filter(kernel.name, extra_filter)


def _parse_optdims(optdim: str) -> List[int]:
    return [int(hdim) for hdim in optdim.split(",")]


def _load_ck_modules(generator_dir: Path):
    sys.path.insert(0, str(generator_dir))

    from codegen.cmake_config import GEN_DIR
    from codegen.ops.fmha_fwd import (
        FMHA_FWD_API_FILENAME,
        FmhaFwdApiPool,
        get_fwd_blobs,
        write_fwd_api,
        write_single_fwd_kernel,
    )

    return {
        "GEN_DIR": GEN_DIR,
        "FMHA_FWD_API_FILENAME": FMHA_FWD_API_FILENAME,
        "FmhaFwdApiPool": FmhaFwdApiPool,
        "get_fwd_blobs": get_fwd_blobs,
        "write_fwd_api": write_fwd_api,
        "write_single_fwd_kernel": write_single_fwd_kernel,
    }


def _select_kernels(
    ck,
    generator_dir: Path,
    targets: List[str],
    receipt: int,
    optdim_list: List[int],
    mask_impl: str,
    extra_filter: str,
):
    del generator_dir
    _, kernels = ck["get_fwd_blobs"](targets, "", receipt, optdim_list, mask_impl)
    selected = [k for k in kernels if _want_kernel(k, mask_impl, extra_filter)]
    api_pool = ck["FmhaFwdApiPool"](mask_impl)
    for kernel in selected:
        api_pool.register_traits(kernel.api_trait())
    return api_pool, selected


def _list_blobs(
    ck,
    file_path: Path,
    kernels: Iterable,
) -> None:
    with file_path.open("w", encoding="utf-8") as handle:
        for kernel in kernels:
            handle.write(str(file_path.parent / ck["GEN_DIR"] / kernel.filename) + "\n")
        handle.write(str(file_path.parent / ck["GEN_DIR"] / ck["FMHA_FWD_API_FILENAME"]) + "\n")


def _write_blobs(
    ck,
    output_dir: Path,
    api_pool,
    kernels: Iterable,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for kernel in kernels:
        ck["write_single_fwd_kernel"](kernel, output_dir)
    ck["write_fwd_api"](api_pool, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="generate_ck_flash_attn_fwd",
        description="Generate DinoML-selected CK FMHA forward kernels",
    )
    parser.add_argument(
        "--generator_dir",
        required=True,
        help="Path to CK example/ck_tile/01_fmha directory",
    )
    parser.add_argument(
        "--targets",
        default="gfx12",
        help="Comma-separated GPU targets",
    )
    parser.add_argument(
        "--output_dir",
        help="Write generated blobs into a directory",
    )
    parser.add_argument(
        "--list_blobs",
        help="Write the generated blob list to a file",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Optional additional glob filter applied after DinoML selection",
    )
    parser.add_argument(
        "--mask",
        default="simplified",
        help="Mask implementation: simplified or generic",
    )
    parser.add_argument(
        "--receipt",
        type=int,
        default=600,
        help="CK codegen receipt",
    )
    parser.add_argument(
        "--optdim",
        default="32,64,128,192,256",
        help="Comma-separated head dimensions to generate, or -1 for all",
    )
    args = parser.parse_args()

    generator_dir = Path(args.generator_dir).resolve()
    ck = _load_ck_modules(generator_dir)
    targets = args.targets.split(",")
    optdim_list = _parse_optdims(args.optdim)

    api_pool, kernels = _select_kernels(
        ck,
        generator_dir,
        targets,
        args.receipt,
        optdim_list,
        args.mask,
        args.filter,
    )

    if args.list_blobs:
        _list_blobs(ck, Path(args.list_blobs), kernels)
        return 0

    output_dir = generator_dir if args.output_dir is None else Path(args.output_dir)
    _write_blobs(ck, output_dir, api_pool, kernels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
