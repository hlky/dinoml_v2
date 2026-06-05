from __future__ import annotations

import importlib.util
from pathlib import Path


def _tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "measure_temp_arena_vram.py"
    spec = importlib.util.spec_from_file_location("measure_temp_arena_vram", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_measure_temp_arena_tool_parses_common_key_values():
    module = _tool_module()

    parsed = module._parse_key_value_items(
        [
            "snapshot=G:/checkpoints/zai-org/GLM-OCR",
            "prompt_len=64",
            "enabled=true",
            "ratio=1.5",
        ]
    )

    assert parsed == {
        "snapshot": "G:/checkpoints/zai-org/GLM-OCR",
        "prompt_len": 64,
        "enabled": True,
        "ratio": 1.5,
    }


def test_measure_temp_arena_tool_parses_inline_scenario():
    module = _tool_module()

    scenario = module._parse_inline_scenario("small:grid_thw=1,16,16;prompt_len=1;flag=false")

    assert scenario == {
        "name": "small",
        "kwargs": {
            "grid_thw": "1,16,16",
            "prompt_len": 1,
            "flag": False,
        },
    }
