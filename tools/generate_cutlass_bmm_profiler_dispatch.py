#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dinoml.kernels.providers.cutlass.bmm import cutlass_bmm_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidates = cutlass_bmm_candidates(args.op, args.dtype)
    profiler_symbols = sorted({candidate["profiler_symbol"] for candidate in candidates})
    lines = [
        '#include "cutlass_bmm_profiler_core.cuh"',
        "",
        "#include <string>",
        "#include <vector>",
        "",
    ]
    for symbol in profiler_symbols:
        lines.append(f'extern "C" float {symbol}();')
    lines.extend(
        [
            "",
            "namespace dinoml::cutlass_bmm_profiler {",
            "",
            "void* resolve_profile_symbol(const std::string& symbol) {",
        ]
    )
    for symbol in profiler_symbols:
        lines.append(f'  if (symbol == "{symbol}") return reinterpret_cast<void*>(&{symbol});')
    lines.extend(
        [
            '  throw std::runtime_error("CUTLASS BMM profiler symbol not found: " + symbol);',
            "}",
            "",
            "const std::vector<BmmCandidate>& profiler_candidates() {",
            "  static const std::vector<BmmCandidate> candidates = {",
        ]
    )
    for candidate in candidates:
        policy = {
            "provider": candidate.get("provider"),
            "family": candidate.get("family"),
            "op": candidate.get("op"),
            "dtype": candidate.get("dtype"),
            "accumulator_dtype": candidate.get("accumulator_dtype"),
            "layouts": candidate.get("layouts"),
            "epilogue": candidate.get("epilogue"),
            "epilogue_config": candidate.get("epilogue_config"),
            "launch_abi": candidate.get("launch_abi"),
            "cutlass": {key: value for key, value in dict(candidate.get("cutlass", {})).items() if key != "align"},
        }
        lines.append(
            "    {"
            f"\"{candidate['profiler_symbol']}\", "
            f"{int(candidate['cutlass']['align'])}, "
            f"{json.dumps(json.dumps(policy, sort_keys=True, separators=(',', ':')))}"
            "},"
        )
    lines.extend(
        [
            "  };",
            "  return candidates;",
            "}",
            "",
            "}  // namespace dinoml::cutlass_bmm_profiler",
            "",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
