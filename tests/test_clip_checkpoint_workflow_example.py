import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from dinoml.models.clip import LegacyCLIPTextConfig, LegacyCLIPVisionConfig


EXAMPLE = REPO_ROOT / "examples" / "clip_checkpoint_workflow.py"


def _load_example() -> dict[str, object]:
    return runpy.run_path(str(EXAMPLE))


def test_clip_checkpoint_workflow_inputs_cover_legacy_and_non_legacy_eos():
    example = _load_example()
    build_inputs = example["_build_runtime_inputs"]

    legacy_text = LegacyCLIPTextConfig(
        vocab_size=32,
        max_position_embeddings=6,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        projection_dim=4,
        eos_token_id=2,
    )
    non_legacy_text = LegacyCLIPTextConfig(
        vocab_size=32,
        max_position_embeddings=6,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        projection_dim=4,
        eos_token_id=7,
    )
    vision = LegacyCLIPVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        projection_dim=4,
        image_size=4,
        patch_size=2,
    )

    seq_len, legacy_inputs = build_inputs(text_config=legacy_text, vision_config=vision)
    _, non_legacy_inputs = build_inputs(text_config=non_legacy_text, vision_config=vision)

    assert seq_len == 4
    assert legacy_inputs["input_ids"].tolist() == [[0, 1, 3, 31]]
    assert non_legacy_inputs["input_ids"].tolist() == [[0, 1, 2, 7]]
    assert legacy_inputs["attention_mask"].dtype.name == "bool"
    assert legacy_inputs["pixel_values"].shape == (1, 3, 4, 4)


def test_clip_checkpoint_workflow_limits_match_target_contract():
    example = _load_example()
    limits_for = example["_limits_for"]

    cpu_limits = limits_for(checkpoint_id="openai/clip-vit-base-patch32", target="cpu")
    cuda_base_limits = limits_for(checkpoint_id="openai/clip-vit-base-patch32", target="cuda")
    cuda_large_limits = limits_for(checkpoint_id="openai/clip-vit-large-patch14", target="cuda")

    assert set(cpu_limits) == {"logits_per_image", "logits_per_text", "text_embeds", "image_embeds"}
    assert all(limit == 3.0e-5 for limit in cpu_limits.values())
    assert all(limit == 1.0e-5 for limit in cuda_base_limits.values())
    assert cuda_large_limits["logits_per_image"] == 2.0e-5
    assert cuda_large_limits["logits_per_text"] == 2.0e-5
    assert cuda_large_limits["text_embeds"] == 1.0e-5
    assert cuda_large_limits["image_embeds"] == 1.0e-5


@pytest.mark.filterwarnings("ignore:overflow encountered in exp:RuntimeWarning")
def test_clip_checkpoint_workflow_cli_cpu_smoke_cached_base(tmp_path):
    example = _load_example()
    available, reason = example["_checkpoint_is_available"](
        checkpoint_id="openai/clip-vit-base-patch32",
        transformers_src=Path("/workspace/transformers/src"),
        hf_home=Path("/workspace/.cache/huggingface"),
    )
    if not available:
        pytest.skip(f"cached base checkpoint workflow example unavailable: {reason}")

    artifact_dir = tmp_path / "clip_checkpoint_workflow_cpu.dinoml"
    env = os.environ.copy()
    env["DINOML_CACHE_DIR"] = str(tmp_path / "cache")
    env["HF_HOME"] = "/workspace/.cache/huggingface"
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE),
            "--artifact-dir",
            str(artifact_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    summary = json.loads(result.stdout)
    assert summary["name"] == "clip_checkpoint_workflow"
    assert summary["checkpoint_id"] == "openai/clip-vit-base-patch32"
    assert summary["target"]["name"] == "cpu"
    assert summary["target"]["arch"] == "native"
    assert summary["artifact"]["path"] == str(artifact_dir.resolve())
    assert summary["artifact"]["module_exists"] is True
    assert summary["artifact"]["manifest_exists"] is True
    assert summary["input_shapes"] == {
        "attention_mask": [1, 4],
        "input_ids": [1, 4],
        "pixel_values": [1, 3, summary["vision_config"]["image_size"], summary["vision_config"]["image_size"]],
    }
    assert summary["output_shapes"]["logits_per_image"] == [1, 1]
    assert summary["output_shapes"]["logits_per_text"] == [1, 1]
    assert summary["output_shapes"]["text_embeds"] == [1, summary["text_config"]["projection_dim"]]
    assert summary["output_shapes"]["image_embeds"] == [1, summary["vision_config"]["projection_dim"]]
    assert all(summary["allclose"].values())
    assert all(metric <= summary["limits"][name] for name, metric in summary["max_abs_diff"].items())
