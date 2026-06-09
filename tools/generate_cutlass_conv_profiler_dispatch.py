from __future__ import annotations

import argparse
from pathlib import Path

from dinoml.kernels.providers.cutlass.conv import (
    CONV_OPS,
    CONV_SUPPORTED_DTYPES,
    CUTLASS_TRANSPOSED_CONV_OPS,
    cutlass_conv_candidates,
    cutlass_transposed_conv_candidates,
)


def _candidate_row(candidate: dict) -> str:
    predicate = dict(candidate.get("selection_predicate", {}))
    kind = str(predicate.get("kind", ""))
    input_channels = int(predicate.get("input_channels", 0) or 0)
    min_input_channels = int(predicate.get("min_input_channels", 0) or 0)
    input_multiple = int(predicate.get("input_channels_multiple", 1) or 1)
    output_multiple = int(predicate.get("output_channels_multiple", 1) or 1)
    return (
        "    {"
        f"\"{candidate['profiler_symbol']}\", "
        f"\"{candidate['kernel_symbol']}\", "
        f"\"{kind}\", "
        f"{input_channels}, {min_input_channels}, {input_multiple}, {output_multiple}"
        "},"
    )


def render_dispatch(op: str, dtype: str) -> str:
    if op not in {*CONV_OPS, *CUTLASS_TRANSPOSED_CONV_OPS}:
        raise ValueError(f"Unsupported CUTLASS Conv op {op!r}")
    if dtype not in CONV_SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported CUTLASS Conv dtype {dtype!r}")
    if op in CUTLASS_TRANSPOSED_CONV_OPS:
        candidates = cutlass_transposed_conv_candidates(op, dtype, target={"name": "cuda", "arch": "sm_80"})
    else:
        candidates = cutlass_conv_candidates(op, dtype, target={"name": "cuda", "arch": "sm_80"})
    rows = "\n".join(_candidate_row(candidate) for candidate in candidates)
    return f"""#include \"cutlass_conv_profiler_core.cuh\"

#include <dlfcn.h>

namespace dinoml::cutlass_conv_profiler {{

void* resolve_profile_symbol(const std::string& symbol) {{
  void* ptr = dlsym(RTLD_DEFAULT, symbol.c_str());
  if (ptr == nullptr) {{
    throw std::runtime_error(\"Could not resolve CUTLASS Conv profiler symbol: \" + symbol);
  }}
  return ptr;
}}

const std::vector<ConvCandidate>& profiler_candidates() {{
  static const std::vector<ConvCandidate> candidates = {{
{rows}
  }};
  return candidates;
}}

}}  // namespace dinoml::cutlass_conv_profiler
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_dispatch(args.op, args.dtype), encoding="utf-8")


if __name__ == "__main__":
    main()
