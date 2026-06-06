from __future__ import annotations

from types import SimpleNamespace

import pytest

import dinoml.models.glm_ocr.glm_ocr_decode as decode_workflow
import dinoml.models.glm_ocr.glm_ocr_image as image_workflow


def test_glm_ocr_equal_interval_buckets_use_even_steps():
    assert image_workflow._equal_interval_buckets(16, 32, count=3, divisible_by=2) == (16, 24, 32)
    assert image_workflow._equal_interval_buckets(1, 64, count=3) == (1, 32, 64)


def test_glm_ocr_build_spec_emits_bucketed_grid_and_prompt_profile_scenarios(monkeypatch):
    fake_config = SimpleNamespace(
        image_token_id=59280,
        vision_config=SimpleNamespace(
            spatial_merge_size=2,
            patch_dim=1176,
            head_dim=64,
            dtype="bfloat16",
        ),
        text_config=SimpleNamespace(
            head_dim=128,
            dtype="bfloat16",
            num_attention_heads=16,
        ),
    )

    monkeypatch.setattr(image_workflow, "build_config", lambda **_kwargs: fake_config)
    monkeypatch.setattr(image_workflow, "build_weights", lambda **_kwargs: {})
    monkeypatch.setattr(image_workflow, "GlmOcrForConditionalGenerationImagePrefill", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        image_workflow.dml,
        "trace",
        lambda model, *, inputs, name: SimpleNamespace(model=model, inputs=inputs, name=name, ir={"metadata": {}}),
    )

    spec = image_workflow.build_spec(
        snapshot="unused",
        min_grid_thw="1,16,16",
        grid_thw="1,24,24",
        max_grid_thw="1,32,32",
        min_prompt_len=1,
        prompt_len=17,
        max_prompt_len=33,
        grid_bucket_count=3,
        prompt_bucket_count=3,
    )

    scenarios = spec.ir["metadata"]["profiling"]["shape_scenarios"]

    assert len(scenarios) == 9
    assert scenarios[0]["source"] == "workflow_equal_interval_buckets"
    assert scenarios[0]["dim_values"]["grid_h"] == 16
    assert scenarios[0]["dim_values"]["grid_w"] == 16
    assert scenarios[0]["dim_values"]["prompt_len"] == 1
    assert scenarios[0]["overrides"]["input_ids"] == [1, 66]
    assert scenarios[0]["overrides"]["pixel_values"] == [256, 1176]
    assert {scenario["dim_values"]["grid_h"] for scenario in scenarios} == {16, 24, 32}
    assert {scenario["dim_values"]["grid_w"] for scenario in scenarios} == {16, 24, 32}
    assert all(scenario["dim_values"]["grid_h"] == scenario["dim_values"]["grid_w"] for scenario in scenarios)

    last = scenarios[-1]
    assert last["dim_values"] == {
        "grid_t": 1,
        "grid_h": 32,
        "grid_w": 32,
        "prompt_len": 33,
        "patch_count": 1024,
        "seq_len": 290,
    }
    assert last["overrides"] == {
        "input_ids": [1, 290],
        "pixel_values": [1024, 1176],
        "vision_cos": [1024, 64],
        "vision_sin": [1024, 64],
        "text_cos": [1, 290, 128],
        "text_sin": [1, 290, 128],
    }


def test_glm_ocr_square_profile_scenarios_require_matching_spatial_bounds(monkeypatch):
    fake_config = SimpleNamespace(
        image_token_id=59280,
        vision_config=SimpleNamespace(
            spatial_merge_size=2,
            patch_dim=1176,
            head_dim=64,
            dtype="bfloat16",
        ),
        text_config=SimpleNamespace(
            head_dim=128,
            dtype="bfloat16",
            num_attention_heads=16,
        ),
    )

    monkeypatch.setattr(image_workflow, "build_config", lambda **_kwargs: fake_config)
    monkeypatch.setattr(image_workflow, "build_weights", lambda **_kwargs: {})
    monkeypatch.setattr(image_workflow, "GlmOcrForConditionalGenerationImagePrefill", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        image_workflow.dml,
        "trace",
        lambda model, *, inputs, name: SimpleNamespace(model=model, inputs=inputs, name=name, ir={"metadata": {}}),
    )

    with pytest.raises(ValueError, match="square profiling buckets require matching min/max grid height and width bounds"):
        image_workflow.build_spec(
            snapshot="unused",
            min_grid_thw="1,16,24",
            max_grid_thw="1,32,48",
            grid_bucket_count=3,
        )


def test_glm_ocr_image_build_spec_can_select_cache_variant(monkeypatch):
    fake_config = SimpleNamespace(
        image_token_id=59280,
        vision_config=SimpleNamespace(
            spatial_merge_size=2,
            patch_dim=1176,
            head_dim=64,
            dtype="bfloat16",
        ),
        text_config=SimpleNamespace(
            head_dim=128,
            dtype="bfloat16",
            num_attention_heads=16,
        ),
    )

    monkeypatch.setattr(image_workflow, "build_config", lambda **_kwargs: fake_config)
    monkeypatch.setattr(image_workflow, "build_weights", lambda **_kwargs: {})
    monkeypatch.setattr(image_workflow, "GlmOcrForConditionalGenerationImagePrefillWithCache", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        image_workflow.dml,
        "trace",
        lambda model, *, inputs, name: SimpleNamespace(model=model, inputs=inputs, name=name, ir={"metadata": {}}),
    )

    spec = image_workflow.build_spec(
        snapshot="unused",
        grid_thw="1,24,24",
        max_grid_thw="1,32,32",
        prompt_len=17,
        max_prompt_len=33,
        compile_cache=True,
    )

    assert "attention_mask" in spec.inputs
    assert "with_cache" in spec.name
    assert "attention_mask" in spec.ir["metadata"]["profiling"]["shape_scenarios"][0]["overrides"]


def test_glm_ocr_decode_build_spec_emits_dynamic_standard_profile_scenarios(monkeypatch):
    fake_config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
            dtype="bfloat16",
        )
    )

    monkeypatch.setattr(decode_workflow, "build_config", lambda **_kwargs: fake_config)
    monkeypatch.setattr(decode_workflow, "build_weights", lambda **_kwargs: {})
    monkeypatch.setattr(decode_workflow, "GlmOcrForConditionalGenerationDecode", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        decode_workflow.dml,
        "trace",
        lambda model, *, inputs, name: SimpleNamespace(model=model, inputs=inputs, name=name, ir={"metadata": {}}),
    )

    spec = decode_workflow.build_spec(
        snapshot="unused",
        min_batch=1,
        batch=2,
        max_batch=3,
        min_past_len=2,
        past_len=4,
        max_past_len=6,
        cache_variant="none",
    )

    scenarios = spec.ir["metadata"]["profiling"]["shape_scenarios"]

    assert "cache_seqlens" not in spec.inputs
    assert "past_key_0" in spec.inputs
    assert len(scenarios) == 9
    assert scenarios[0]["overrides"]["attention_mask"] == [8, 1, 3]
    assert scenarios[-1]["dim_values"] == {"batch": 3, "past_len": 6, "total_len": 7}
    assert scenarios[-1]["overrides"]["past_key_0"] == [3, 4, 6, 128]


def test_glm_ocr_decode_build_spec_can_select_static_cache_variant(monkeypatch):
    fake_config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
            dtype="bfloat16",
        )
    )

    monkeypatch.setattr(decode_workflow, "build_config", lambda **_kwargs: fake_config)
    monkeypatch.setattr(decode_workflow, "build_weights", lambda **_kwargs: {})
    monkeypatch.setattr(decode_workflow, "GlmOcrForConditionalGenerationDecodeStaticCache", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        decode_workflow.dml,
        "trace",
        lambda model, *, inputs, name: SimpleNamespace(model=model, inputs=inputs, name=name, ir={"metadata": {}}),
    )

    spec = decode_workflow.build_spec(
        snapshot="unused",
        batch=1,
        max_batch=2,
        past_len=5,
        max_past_len=7,
        cache_variant="static",
    )

    scenario = spec.ir["metadata"]["profiling"]["shape_scenarios"][-1]

    assert "cache_seqlens" in spec.inputs
    assert scenario["dim_values"] == {"batch": 2, "past_len": 7, "total_len": 8}
    assert scenario["overrides"]["past_key_0"] == [2, 4, 8, 128]


def test_glm_ocr_decode_build_spec_can_select_session_cache_variant(monkeypatch):
    fake_config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
            dtype="bfloat16",
        )
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(decode_workflow, "build_config", lambda **_kwargs: fake_config)
    monkeypatch.setattr(decode_workflow, "build_weights", lambda **_kwargs: {})
    monkeypatch.setattr(
        decode_workflow,
        "GlmOcrForConditionalGenerationDecodeSessionStaticCache",
        lambda *args, **kwargs: captured.setdefault("max_cache_len", kwargs["max_cache_len"]) or object(),
    )
    monkeypatch.setattr(
        decode_workflow.dml,
        "trace",
        lambda model, *, inputs, name: SimpleNamespace(model=model, inputs=inputs, name=name, ir={"metadata": {}}),
    )

    spec = decode_workflow.build_spec(
        snapshot="unused",
        batch=1,
        max_batch=3,
        past_len=5,
        max_past_len=7,
        max_cache_len=12,
        cache_variant="session",
    )

    assert captured["max_cache_len"] == 12
    assert "cache_seqlens" not in spec.inputs
    assert "past_key_0" not in spec.inputs
    assert spec.ir["metadata"]["profiling"]["shape_scenarios"][-1]["overrides"]["attention_mask"] == [24, 1, 12]
