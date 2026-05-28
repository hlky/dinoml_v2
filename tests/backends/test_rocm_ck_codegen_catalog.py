from __future__ import annotations

import re
from pathlib import Path

from dinoml.kernels.providers.ck import gemm as ck_gemm_provider


def test_ck_gemm_provider_tracks_upstream_multiple_d_codegen_tiles():
    source = Path("third_party/composable_kernel/codegen/src/device_gemm_multiple_d_operation_xdl_cshuffle.cpp").read_text(
        encoding="utf-8"
    )
    upstream_tiles = _parse_ck_tile_rows(source)
    provider_tiles = [
        (
            int(config["tile"]["block_size"]),
            int(config["tile"]["m_per_block"]),
            int(config["tile"]["n_per_block"]),
            int(config["tile"]["k_per_block"]),
            int(config["tile"]["ak1"]),
            int(config["tile"]["bk1"]),
            int(config["tile"]["m_per_xdl"]),
            int(config["tile"]["n_per_xdl"]),
            int(config["tile"]["m_xdl_per_wave"]),
            int(config["tile"]["n_xdl_per_wave"]),
            int(config["tile"]["num_gemmk_prefetch_stage"]),
        )
        for config in ck_gemm_provider._CK_GEMM_CODEGEN_TILES
    ]

    assert provider_tiles == upstream_tiles
    assert len(ck_gemm_provider._CK_GEMM_SCHEDULER_PIPELINES) == 3
    assert len(ck_gemm_provider.ck_gemm_candidates("gemm_rcr_bias_add_relu", "float16")) == (
        len(upstream_tiles) * len(ck_gemm_provider._CK_GEMM_SCHEDULER_PIPELINES)
    )


def _parse_ck_tile_rows(source: str) -> list[tuple[int, ...]]:
    marker = "std::vector<operation::TileDesc> tile_descriptions = {"
    table = source.split(marker, 1)[1].split("};", 1)[0]
    rows: list[tuple[int, ...]] = []
    for match in re.finditer(r"\{\s*((?:\d+\s*,\s*){10}\d+)\s*\}", table):
        rows.append(tuple(int(part.strip()) for part in match.group(1).split(",")))
    return rows
