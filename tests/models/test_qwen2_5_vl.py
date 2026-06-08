from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import dinoml.models.qwen2_5_vl.qwen2_5_vl_decode as decode_workflow
import dinoml.models.qwen2_5_vl.qwen2_5_vl_image as image_workflow
from dinoml.models.qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
    qwen2_5_vl_config_from_transformers_dict,
    qwen2_5_vl_patch_embed_linear_weight,
    qwen2_5_vl_prepare_inputs_for_generation,
    qwen2_5_vl_required_text_weight_names,
    qwen2_5_vl_rope_index,
    qwen2_5_vl_stitch_image_features,
    qwen2_5_vl_text_rope_embeddings,
    qwen2_5_vl_vision_cu_seqlens,
    qwen2_5_vl_vision_position_ids,
    qwen2_5_vl_vision_rope_embeddings,
    qwen2_5_vl_vision_window_index,
    qwen2_5_vl_weights_from_safetensors_index,
)
from dinoml.models.qwen2_5_vl.workflow_common import enable_flash_attention_bias_for_target


SOURCES = Path(r"H:\dinoml_v2_agents\agents\plans\transformers\qwen2_5_vl\_sources")


def test_qwen2_5_vl_3b_config_normalizes_rope_and_tied_head():
    payload = json.loads((SOURCES / "Qwen__Qwen2_5-VL-3B-Instruct__config.json").read_text(encoding="utf-8"))

    config = qwen2_5_vl_config_from_transformers_dict(payload)

    assert config.text_config.hidden_size == 2048
    assert config.text_config.num_attention_heads == 16
    assert config.text_config.num_key_value_heads == 2
    assert config.text_config.head_dim == 128
    assert config.text_config.rope_parameters == {
        "rope_type": "default",
        "mrope_section": [16, 24, 24],
        "rope_theta": 1_000_000.0,
    }
    assert config.tie_word_embeddings
    assert "lm_head.weight" not in qwen2_5_vl_required_text_weight_names(config)
    assert config.vision_config.depth == 32
    assert config.vision_config.patch_dim == 1176
    assert config.vision_config.out_hidden_size == 2048


def test_qwen2_5_vl_32b_short_vision_schema_gets_effective_defaults():
    payload = json.loads((SOURCES / "Qwen__Qwen2_5-VL-32B-Instruct__config.json").read_text(encoding="utf-8"))

    config = qwen2_5_vl_config_from_transformers_dict(payload)

    assert config.text_config.hidden_size == 5120
    assert config.text_config.num_key_value_heads == 8
    assert config.vision_config.depth == 32
    assert config.vision_config.num_heads == 16
    assert config.vision_config.patch_size == 14
    assert config.vision_config.spatial_merge_size == 2
    assert config.vision_config.temporal_patch_size == 2
    assert config.vision_config.window_size == 112
    assert config.vision_config.fullatt_block_indexes == (7, 15, 23, 31)
    assert config.vision_config.out_hidden_size == 5120


@pytest.mark.parametrize("sliding_window", [None, 8])
def test_qwen2_5_vl_rejects_sliding_window_configs(sliding_window):
    with pytest.raises(NotImplementedError, match="sliding-window text attention"):
        Qwen2_5_VLTextConfig(
            vocab_size=32,
            hidden_size=12,
            intermediate_size=10,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 1], "rope_theta": 10000.0},
            dtype="float32",
            use_sliding_window=True,
            sliding_window=sliding_window,
        )


def test_qwen2_5_vl_sharded_safetensors_loader_uses_index_weight_map(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    config = _tiny_config()
    shard_a = tmp_path / "a.safetensors"
    shard_b = tmp_path / "b.safetensors"
    embed = np.arange(config.text_config.vocab_size * config.text_config.hidden_size, dtype=np.float32).reshape(
        config.text_config.vocab_size,
        config.text_config.hidden_size,
    )
    norm = np.arange(config.text_config.hidden_size, dtype=np.float32)
    safetensors_torch.save_file({"model.embed_tokens.weight": torch.from_numpy(embed)}, shard_a)
    safetensors_torch.save_file({"model.norm.weight": torch.from_numpy(norm)}, shard_b)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    "model.embed_tokens.weight": shard_a.name,
                    "model.norm.weight": shard_b.name,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = qwen2_5_vl_weights_from_safetensors_index(
        index,
        config,
        dtype="float32",
        required_names=["model.embed_tokens.weight", "model.norm.weight"],
    )

    np.testing.assert_array_equal(loaded["model.embed_tokens.weight"], embed)
    np.testing.assert_array_equal(loaded["model.norm.weight"], norm)


def test_qwen2_5_vl_sharded_loader_reports_missing_index_entries(tmp_path):
    config = _tiny_config()
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {}}), encoding="utf-8")

    with pytest.raises(KeyError, match="Missing Qwen2.5-VL safetensors index entries"):
        qwen2_5_vl_weights_from_safetensors_index(index, config, required_names=["model.norm.weight"])


def test_qwen2_5_vl_text_rope_uses_half_rotation_mrope_sections():
    config = Qwen2_5_VLTextConfig(
        vocab_size=32,
        hidden_size=12,
        intermediate_size=10,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 1], "rope_theta": 10000.0},
        dtype="float32",
    )
    position_ids = np.asarray([[[1, 2]], [[10, 20]], [[100, 200]]], dtype=np.int64)

    cos, sin = qwen2_5_vl_text_rope_embeddings(position_ids, config)

    inv = np.asarray([1.0, 10000.0 ** (-2.0 / 6.0), 10000.0 ** (-4.0 / 6.0)], dtype=np.float32)
    freqs = np.asarray(
        [[[[1 * inv[0], 1 * inv[1], 1 * inv[2]], [2 * inv[0], 2 * inv[1], 2 * inv[2]]]],
         [[[10 * inv[0], 10 * inv[1], 10 * inv[2]], [20 * inv[0], 20 * inv[1], 20 * inv[2]]]],
         [[[100 * inv[0], 100 * inv[1], 100 * inv[2]], [200 * inv[0], 200 * inv[1], 200 * inv[2]]]]],
        dtype=np.float32,
    )
    emb = np.concatenate([freqs, freqs], axis=-1)
    expected = np.concatenate([emb[0, :, :, 0:2], emb[1, :, :, 2:4], emb[2, :, :, 4:6]], axis=-1)
    np.testing.assert_allclose(cos, np.cos(expected), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sin, np.sin(expected), rtol=1e-6, atol=1e-6)


def test_qwen2_5_vl_rope_index_matches_image_grouping():
    input_ids = np.asarray([[11, 12, 151655, 151655, 13]], dtype=np.int64)
    mm_token_type_ids = np.asarray([[0, 0, 1, 1, 0]], dtype=np.int64)

    position_ids, deltas = qwen2_5_vl_rope_index(
        input_ids,
        mm_token_type_ids,
        image_grid_thw=np.asarray([[1, 2, 4]], dtype=np.int64),
        spatial_merge_size=2,
    )

    expected = np.asarray([[[0, 1, 2, 2, 4]], [[0, 1, 2, 2, 4]], [[0, 1, 2, 3, 4]]], dtype=np.int64)
    np.testing.assert_array_equal(position_ids, expected)
    np.testing.assert_array_equal(deltas, np.asarray([[0]], dtype=np.int64))


def test_qwen2_5_vl_rope_index_rejects_video_in_image_first_scope():
    with pytest.raises(NotImplementedError, match="video rope indexing is deferred"):
        qwen2_5_vl_rope_index(
            np.asarray([[151656]], dtype=np.int64),
            np.asarray([[2]], dtype=np.int64),
            image_grid_thw=np.asarray([[1, 2, 2]], dtype=np.int64),
        )


def test_qwen2_5_vl_vision_position_window_and_cu_metadata():
    grid_thw = np.asarray([[1, 4, 4]], dtype=np.int64)

    position_ids = qwen2_5_vl_vision_position_ids(grid_thw, spatial_merge_size=2)
    window_index, cu_window_seqlens = qwen2_5_vl_vision_window_index(
        grid_thw,
        spatial_merge_size=2,
        window_size=4,
        patch_size=2,
    )
    cu_seqlens = qwen2_5_vl_vision_cu_seqlens(grid_thw)

    expected_pos = np.asarray(
        [
            [0, 0], [0, 1], [1, 0], [1, 1],
            [0, 2], [0, 3], [1, 2], [1, 3],
            [2, 0], [2, 1], [3, 0], [3, 1],
            [2, 2], [2, 3], [3, 2], [3, 3],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(position_ids, expected_pos)
    np.testing.assert_array_equal(window_index, np.asarray([0, 1, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(cu_window_seqlens, np.asarray([0, 4, 8, 12, 16], dtype=np.int32))
    np.testing.assert_array_equal(cu_seqlens, np.asarray([0, 16], dtype=np.int32))


def test_qwen2_5_vl_vision_rope_embeddings_have_head_dim_columns():
    position_ids = np.asarray([[0, 0], [0, 1], [1, 0]], dtype=np.int64)

    cos, sin = qwen2_5_vl_vision_rope_embeddings(position_ids, head_dim=4)

    assert cos.shape == (3, 4)
    assert sin.shape == (3, 4)
    np.testing.assert_allclose(cos[0], np.ones(4, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sin[0], np.zeros(4, dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_qwen2_5_vl_patch_embed_linear_weight_flattens_conv3d_weight():
    weight = np.arange(5 * 3 * 2 * 2 * 2, dtype=np.float32).reshape(5, 3, 2, 2, 2)
    linear = qwen2_5_vl_patch_embed_linear_weight(weight)
    patches = np.arange(7 * 3 * 2 * 2 * 2, dtype=np.float32).reshape(7, -1)

    np.testing.assert_array_equal(linear, weight.reshape(5, -1))
    np.testing.assert_allclose(patches @ linear.T, patches @ weight.reshape(5, -1).T)


def test_qwen2_5_vl_stitch_image_features_replaces_placeholders_in_row_major_order():
    input_ids = np.asarray([[1, 30, 2, 30]], dtype=np.int64)
    inputs_embeds = np.zeros((1, 4, 3), dtype=np.float32)
    image_features = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    stitched = qwen2_5_vl_stitch_image_features(input_ids, inputs_embeds, image_features, image_token_id=30)

    np.testing.assert_array_equal(stitched[0, 1], image_features[0])
    np.testing.assert_array_equal(stitched[0, 3], image_features[1])


def test_qwen2_5_vl_stitch_image_features_handles_multiple_spans_across_batches():
    input_ids = np.asarray(
        [
            [30, 1, 30, 2],
            [3, 30, 4, 30],
        ],
        dtype=np.int64,
    )
    inputs_embeds = np.full((2, 4, 2), -1.0, dtype=np.float32)
    image_features = np.asarray(
        [
            [1.0, 1.5],
            [2.0, 2.5],
            [3.0, 3.5],
            [4.0, 4.5],
        ],
        dtype=np.float32,
    )

    stitched = qwen2_5_vl_stitch_image_features(input_ids, inputs_embeds, image_features, image_token_id=30)

    np.testing.assert_array_equal(
        stitched,
        np.asarray(
            [
                [[1.0, 1.5], [-1.0, -1.0], [2.0, 2.5], [-1.0, -1.0]],
                [[-1.0, -1.0], [3.0, 3.5], [-1.0, -1.0], [4.0, 4.5]],
            ],
            dtype=np.float32,
        ),
    )


def test_qwen2_5_vl_generation_preparation_drops_visual_tensors_after_first_cached_step():
    prepared = qwen2_5_vl_prepare_inputs_for_generation(
        {"input_ids": np.asarray([[1]]), "pixel_values": object(), "pixel_values_videos": object()},
        is_first_iteration=False,
        use_cache=True,
    )

    assert prepared["pixel_values"] is None
    assert prepared["pixel_values_videos"] is None


def test_qwen2_5_vl_image_workflow_build_spec_keeps_grid_contract(monkeypatch):
    fake_config = SimpleNamespace(
        image_token_id=30,
        vision_config=SimpleNamespace(
            spatial_merge_size=2,
            spatial_merge_unit=4,
            patch_dim=24,
            patch_size=2,
            head_dim=4,
            dtype="bfloat16",
            window_size=4,
            vit_merger_window_size=1,
        ),
        text_config=SimpleNamespace(
            head_dim=6,
            dtype="bfloat16",
            num_attention_heads=2,
        ),
    )

    monkeypatch.setattr(image_workflow, "build_config", lambda **_kwargs: fake_config)
    monkeypatch.setattr(image_workflow, "build_weights", lambda **_kwargs: {})
    monkeypatch.setattr(image_workflow, "Qwen2_5_VLForConditionalGenerationImagePrefillWithCache", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        image_workflow.dml,
        "trace",
        lambda model, *, inputs, name: SimpleNamespace(model=model, inputs=inputs, name=name, ir={"metadata": {}}),
    )

    spec = image_workflow.build_spec(
        snapshot="unused",
        min_grid_thw="1,4,4",
        grid_thw="1,4,4",
        max_grid_thw="1,8,8",
        prompt_len=3,
        max_prompt_len=5,
    )
    scenarios = spec.ir["metadata"]["profiling"]["shape_scenarios"]

    assert "image_grid_thw" in spec.inputs
    assert "vision_full_cu_seqlens" in spec.inputs
    assert "vision_window_cu_seqlens" in spec.inputs
    assert "vision_reverse_window_index" in spec.inputs
    assert scenarios[0]["overrides"]["image_grid_thw"] == [1, 3]
    assert scenarios[-1]["overrides"]["pixel_values"] == [64, 24]
    assert scenarios[-1]["overrides"]["vision_reverse_window_index"] == [16]
    assert scenarios[-1]["overrides"]["input_ids"] == [1, 22]


def test_qwen2_5_vl_image_workflow_fixed_bounds_keeps_window_metadata_external(monkeypatch):
    fake_config = SimpleNamespace(
        image_token_id=30,
        vision_config=SimpleNamespace(
            spatial_merge_size=2,
            spatial_merge_unit=4,
            patch_dim=24,
            patch_size=2,
            head_dim=4,
            dtype="bfloat16",
            window_size=4,
            vit_merger_window_size=1,
        ),
        text_config=SimpleNamespace(
            head_dim=6,
            dtype="bfloat16",
            num_attention_heads=2,
        ),
    )
    created = {}

    def fake_model(config, weights, *, logits_to_keep, max_full_seqlen, max_window_seqlen):
        created["logits_to_keep"] = logits_to_keep
        created["max_full_seqlen"] = max_full_seqlen
        created["max_window_seqlen"] = max_window_seqlen
        return object()

    monkeypatch.setattr(image_workflow, "build_config", lambda **_kwargs: fake_config)
    monkeypatch.setattr(image_workflow, "build_weights", lambda **_kwargs: {})
    monkeypatch.setattr(image_workflow, "Qwen2_5_VLForConditionalGenerationImagePrefillWithCache", fake_model)
    monkeypatch.setattr(
        image_workflow.dml,
        "trace",
        lambda model, *, inputs, name: SimpleNamespace(model=model, inputs=inputs, name=name, ir={"metadata": {}}),
    )

    spec = image_workflow.build_spec(
        snapshot="unused",
        min_grid_thw="1,4,4",
        grid_thw="1,4,4",
        max_grid_thw="1,4,4",
        min_prompt_len=3,
        prompt_len=3,
        max_prompt_len=3,
    )

    assert created == {"logits_to_keep": 1, "max_full_seqlen": 16, "max_window_seqlen": 4}
    assert spec.inputs["pixel_values"].shape == [16, 24]
    assert spec.inputs["vision_full_cu_seqlens"].shape == [2]
    assert spec.inputs["vision_window_cu_seqlens"].shape == [5]
    assert spec.inputs["vision_reverse_window_index"].shape == [4]
    assert spec.inputs["input_ids"].shape == [1, 8]


def test_qwen2_5_vl_decode_workflow_build_spec_uses_gqa_cache_width(monkeypatch):
    fake_config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=1,
            num_key_value_heads=2,
            num_attention_heads=8,
            head_dim=4,
            dtype="bfloat16",
        ),
    )

    monkeypatch.setattr(decode_workflow, "build_config", lambda **_kwargs: fake_config)
    monkeypatch.setattr(decode_workflow, "build_weights", lambda **_kwargs: {})
    monkeypatch.setattr(decode_workflow, "Qwen2_5_VLForConditionalGenerationDecodeStaticCache", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        decode_workflow.dml,
        "trace",
        lambda model, *, inputs, name: SimpleNamespace(model=model, inputs=inputs, name=name, ir={"metadata": {}}),
    )

    spec = decode_workflow.build_spec(
        snapshot="unused",
        batch=1,
        max_batch=2,
        past_len=3,
        max_past_len=5,
        cache_variant="static",
        use_attention_mask=True,
    )
    scenarios = spec.ir["metadata"]["profiling"]["shape_scenarios"]

    assert spec.inputs["past_key_0"].shape_spec[1] == 2
    assert spec.inputs["attention_mask"].shape_spec[0] == {
        "kind": "int_expr",
        "op": "mul",
        "lhs": {
            "kind": "dim",
            "name": "batch",
            "min": 1,
            "max": 2,
            "divisible_by": 1,
            "typical": 1,
            "buckets": [1, 2],
        },
        "rhs": 8,
    }
    assert scenarios[-1]["overrides"]["past_key_0"] == [2, 2, 6, 4]
    assert scenarios[-1]["overrides"]["attention_mask"] == [16, 1, 6]


def test_qwen2_5_vl_rocm_masked_workflows_enable_flash_attention_bias():
    config = Qwen2_5_VLConfig()

    rocm_config = enable_flash_attention_bias_for_target(config, target="rocm", needs_attention_mask=True)
    cuda_config = enable_flash_attention_bias_for_target(config, target="cuda", needs_attention_mask=True)

    assert rocm_config.text_config.use_flash_attention_bias
    assert not cuda_config.text_config.use_flash_attention_bias


def _tiny_config() -> Qwen2_5_VLConfig:
    text = Qwen2_5_VLTextConfig(
        vocab_size=32,
        hidden_size=12,
        intermediate_size=10,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 1], "rope_theta": 10000.0},
        dtype="float32",
    )
    vision = Qwen2_5_VLVisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=12,
        window_size=4,
        fullatt_block_indexes=(0,),
        dtype="float32",
        use_flash_attention=False,
    )
    return Qwen2_5_VLConfig(text_config=text, vision_config=vision, image_token_id=30, video_token_id=31)
