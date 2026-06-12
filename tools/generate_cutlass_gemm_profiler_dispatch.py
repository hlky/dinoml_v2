#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_candidates


def workspace_symbol(kernel_symbol: str) -> str:
    prefix = "dinoml_cutlass_"
    if not kernel_symbol.startswith(prefix):
        raise ValueError(f"unexpected CUTLASS kernel symbol: {kernel_symbol}")
    return f"dinoml_cutlass_workspace_{kernel_symbol[len(prefix):]}"


def splitk_symbol(profiler_symbol: str) -> str:
    prefix = "dinoml_profile_cutlass_"
    if not profiler_symbol.startswith(prefix):
        raise ValueError(f"unexpected CUTLASS profiler symbol: {profiler_symbol}")
    return f"dinoml_profile_cutlass_splitk_{profiler_symbol[len(prefix):]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidates = cutlass_gemm_candidates(args.op, args.dtype)
    profiler_symbols = sorted({candidate["profiler_symbol"] for candidate in candidates})
    splitk_symbols = sorted(
        {
            splitk_symbol(str(candidate["profiler_symbol"]))
            for candidate in candidates
            if candidate.get("supports_split_k")
        }
    )
    workspace_symbols = sorted(
        {
            workspace_symbol(str(candidate["kernel_symbol"]))
            for candidate in candidates
            if candidate.get("supports_split_k")
        }
    )

    lines = [
        '#include "cutlass_gemm_profiler_core.cuh"',
        "",
        "#include <string>",
        "#include <vector>",
        "",
    ]
    for symbol in [*profiler_symbols, *splitk_symbols]:
        lines.append(f'extern "C" float {symbol}();')
    for symbol in workspace_symbols:
        lines.append(f'extern "C" size_t {symbol}();')
    lines.extend(
        [
            "",
            "namespace dinoml::cutlass_gemm_profiler {",
            "",
            "void* resolve_profile_symbol(const std::string& symbol) {",
        ]
    )
    for symbol in [*profiler_symbols, *splitk_symbols]:
        lines.append(f'  if (symbol == "{symbol}") return reinterpret_cast<void*>(&{symbol});')
    lines.extend(
        [
            '  throw std::runtime_error("CUTLASS GEMM profiler symbol not found: " + symbol);',
            "}",
            "",
            "void* resolve_workspace_symbol(const std::string& symbol) {",
        ]
    )
    for symbol in workspace_symbols:
        lines.append(f'  if (symbol == "{symbol}") return reinterpret_cast<void*>(&{symbol});')
    lines.extend(
        [
            '  throw std::runtime_error("CUTLASS GEMM workspace symbol not found: " + symbol);',
            "}",
            "",
            "const std::vector<GemmCandidate>& profiler_candidates() {",
            "  static const std::vector<GemmCandidate> candidates = {",
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
            "supports_split_k": candidate.get("supports_split_k"),
            "split_k_search": candidate.get("split_k_search"),
            "cutlass": {key: value for key, value in dict(candidate.get("cutlass", {})).items() if key != "align"},
        }
        lines.append(
            "    {"
            f"\"{candidate['profiler_symbol']}\", "
            f"\"{workspace_symbol(candidate['kernel_symbol']) if candidate.get('supports_split_k') else ''}\", "
            f"{int(candidate['cutlass']['align'])}, "
            f"{'true' if candidate.get('supports_split_k') else 'false'}, "
            f"{json.dumps(json.dumps(policy, sort_keys=True, separators=(',', ':')))}"
            "},"
        )
    lines.extend(
        [
            "  };",
            "  return candidates;",
            "}",
            "",
            "}  // namespace dinoml::cutlass_gemm_profiler",
            "",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
