from __future__ import annotations

import re
from pathlib import Path


def test_ck_conv_profiler_core_matches_exported_iterations_abi():
    source = Path("tools/ck_conv_profiler_core.hpp").read_text(encoding="utf-8")

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

    assert no_residual.search(source)
    assert residual.search(source)
