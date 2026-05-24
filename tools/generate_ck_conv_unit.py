from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dinoml.kernels.providers.ck.conv import ck_conv_candidate_set, ck_conv_candidates, render_ck_conv_source  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a CK Conv op/dtype translation unit")
    parser.add_argument("--op", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    target = {"name": "rocm", "arch": "gfx1201"}
    candidate_set = ck_conv_candidate_set(args.op, args.dtype, target=target)
    candidates = [dict(candidate) for candidate in ck_conv_candidates(args.op, args.dtype, target=target)]
    plan = {
        "schema_version": 1,
        "provider": "ck",
        "library": "ck_conv",
        "target": target,
        "entries": [
            {
                "op": args.op,
                "dtype": args.dtype,
                "candidate_set_id": candidate_set["candidate_set_id"],
                "candidate_set_key": candidate_set["candidate_set_key"],
                "candidate_set": candidate_set,
                "candidates": candidates,
                "kernel_symbols": sorted({str(candidate["kernel_symbol"]) for candidate in candidates}),
                "profiler_symbols": sorted({str(candidate["profiler_symbol"]) for candidate in candidates}),
            }
        ],
        "candidate_sets": [candidate_set],
        "candidates": candidates,
        "kernel_symbols": sorted({str(candidate["kernel_symbol"]) for candidate in candidates}),
        "profiler_symbols": sorted({str(candidate["profiler_symbol"]) for candidate in candidates}),
    }

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_ck_conv_source(source, plan), encoding="utf-8")


if __name__ == "__main__":
    main()
