#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dinoml.kernels.providers.cutlass.bmm import cutlass_bmm_candidate_set, cutlass_bmm_candidates, render_cutlass_bmm_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source).read_text(encoding="utf-8")
    candidate_set = cutlass_bmm_candidate_set(args.op, args.dtype)
    candidates = [dict(candidate) for candidate in cutlass_bmm_candidates(args.op, args.dtype)]
    plan = {
        "entries": [
            {
                "op": args.op,
                "dtype": args.dtype,
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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_cutlass_bmm_source(source, plan), encoding="utf-8")


if __name__ == "__main__":
    main()
