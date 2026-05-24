#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dinoml.kernels.providers.ck.gemm import (  # noqa: E402
    ck_gemm_candidate_set,
    ck_gemm_candidates,
    render_ck_gemm_source,
)


def ck_gemm_unit_plan(op: str, dtype: str) -> dict:
    target = {"name": "rocm", "arch": "gfx1201"}
    candidate_set = ck_gemm_candidate_set(op, dtype, target=target)
    candidates = [dict(candidate) for candidate in ck_gemm_candidates(op, dtype, target=target)]
    return {
        "entries": [
            {
                "op": op,
                "dtype": dtype,
                "candidate_set": candidate_set,
                "candidate_set_id": candidate_set["candidate_set_id"],
                "candidate_set_key": candidate_set["candidate_set_key"],
                "candidates": candidates,
                "kernel_symbols": sorted({str(candidate["kernel_symbol"]) for candidate in candidates}),
                "profiler_symbols": sorted({str(candidate["profiler_symbol"]) for candidate in candidates}),
            }
        ],
        "candidates": candidates,
        "kernel_symbols": sorted({str(candidate["kernel_symbol"]) for candidate in candidates}),
        "profiler_symbols": sorted({str(candidate["profiler_symbol"]) for candidate in candidates}),
    }


def render_ck_gemm_unit(op: str, dtype: str, source: Path) -> str:
    return render_ck_gemm_source(source.read_text(encoding="utf-8"), ck_gemm_unit_plan(op, dtype))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rendered = render_ck_gemm_unit(args.op, args.dtype, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
