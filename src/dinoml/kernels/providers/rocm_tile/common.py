from __future__ import annotations

from typing import Any, Mapping


def rocm_tile_fp32_fallback_required(dtype: str, target: Mapping[str, Any] | None) -> bool:
    if dtype != "float32" or target is None:
        return False
    arch = str(target.get("arch", "")).strip()
    return arch.startswith("gfx11") or arch.startswith("gfx120")
