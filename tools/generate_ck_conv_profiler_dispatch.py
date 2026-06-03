#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dinoml.kernels.providers.ck.conv import ck_conv_candidates  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    profiler_symbols = sorted({candidate["profiler_symbol"] for candidate in ck_conv_candidates(args.op, args.dtype)})
    lines = [
        '#include "ck_conv_profiler_core.hpp"',
        "",
        "#include <string>",
        "",
    ]
    for symbol in profiler_symbols:
        lines.append(f'extern "C" float {symbol}();')
    lines.extend(
        [
            "",
            "namespace dinoml::ck_conv_profiler {",
            "",
            "void* resolve_profile_symbol(const std::string& symbol) {",
        ]
    )
    for symbol in profiler_symbols:
        lines.append(f'  if (symbol == "{symbol}") return reinterpret_cast<void*>(&{symbol});')
    lines.extend(
        [
            '  throw std::runtime_error("CK Conv profiler symbol not found: " + symbol);',
            "}",
            "",
            "}  // namespace dinoml::ck_conv_profiler",
            "",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
