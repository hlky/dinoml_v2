#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dinoml.kernels.providers.ck.bmm import ck_bmm_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    profiler_symbols = sorted({candidate["profiler_symbol"] for candidate in ck_bmm_candidates(args.op, args.dtype)})
    lines = [
        '#include "ck_bmm_profiler_core.hpp"',
        "",
        "#include <string>",
        "",
    ]
    for symbol in profiler_symbols:
        lines.append(f'extern "C" float {symbol}();')
    lines.extend(
        [
            "",
            "namespace dinoml::ck_bmm_profiler {",
            "",
            "void* resolve_profile_symbol(const std::string& symbol) {",
        ]
    )
    for symbol in profiler_symbols:
        lines.append(f'  if (symbol == "{symbol}") return reinterpret_cast<void*>(&{symbol});')
    lines.extend(
        [
            '  throw std::runtime_error("CK BMM profiler symbol not found: " + symbol);',
            "}",
            "",
            "}  // namespace dinoml::ck_bmm_profiler",
            "",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
