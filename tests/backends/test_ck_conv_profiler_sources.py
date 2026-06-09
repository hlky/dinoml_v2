from __future__ import annotations

import re
from pathlib import Path


def test_ck_conv_profiler_core_matches_exported_iterations_abi():
    source = Path("tools/ck_conv_profiler_core.hpp").read_text(encoding="utf-8")
    ck_conv_source = Path("kernels/rocm/src/ck_conv.hip").read_text(encoding="utf-8")

    no_residual = re.compile(
        r"using Fn = float \(\*\)\(\s*"
        r"const void\*,\s*"
        r"const void\*,\s*"
        r"const void\*,\s*"
        r"void\*,\s*"
        r"(?:int,\s*){16}"
        r"hipStream_t\);",
        re.DOTALL,
    )
    residual = re.compile(
        r"using Fn = float \(\*\)\(\s*"
        r"const void\*,\s*"
        r"const void\*,\s*"
        r"const void\*,\s*"
        r"const void\*,\s*"
        r"void\*,\s*"
        r"(?:int,\s*){16}"
        r"hipStream_t\);",
        re.DOTALL,
    )
    transposed_base = re.compile(
        r"using Fn = float \(\*\)\(\s*"
        r"const void\*,\s*"
        r"const void\*,\s*"
        r"void\*,\s*"
        r"(?:int,\s*){18}"
        r"hipStream_t\);",
        re.DOTALL,
    )
    transposed_bias = re.compile(
        r"using Fn = float \(\*\)\(\s*"
        r"const void\*,\s*"
        r"const void\*,\s*"
        r"const void\*,\s*"
        r"void\*,\s*"
        r"(?:int,\s*){18}"
        r"hipStream_t\);",
        re.DOTALL,
    )
    transposed_residual = re.compile(
        r"using Fn = float \(\*\)\(\s*"
        r"const void\*,\s*"
        r"const void\*,\s*"
        r"const void\*,\s*"
        r"const void\*,\s*"
        r"void\*,\s*"
        r"(?:int,\s*){18}"
        r"hipStream_t\);",
        re.DOTALL,
    )
    assert no_residual.search(source)
    assert residual.search(source)
    assert transposed_base.search(source)
    assert transposed_bias.search(source)
    assert transposed_residual.search(source)
    assert "request.output_pad_h" in source
    assert "request.output_pad_w" in source
    assert "request.has_bias" in source
    assert "request.transposed" in source
    assert "__global__ void dinoml_ck_conv_nchw_to_nhwc_kernel(" in ck_conv_source
    assert "launch_ck_conv_nchw_to_nhwc(" in ck_conv_source
    assert "using TransposedResidualLayout = conv_layout::G_NHW_C;" in ck_conv_source
    assert "const std::array<ck::index_t, NDimSpatial + 3> residual_lengths{g, n, c, hi, wi};" in ck_conv_source
    assert "hi * wi * c,\n      ck::index_t{1},\n      wi * c,\n      c};" in ck_conv_source
    assert "static_cast<Storage*>(residual_packed)" in ck_conv_source
