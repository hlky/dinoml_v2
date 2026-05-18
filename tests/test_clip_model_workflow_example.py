import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import numpy as np

from dinoml.reference import reference_numpy


EXAMPLE = REPO_ROOT / "examples" / "clip_model_workflow.py"


def _load_example() -> dict[str, object]:
    return runpy.run_path(str(EXAMPLE))


def test_clip_model_workflow_bridge_kernel_detection_is_exact():
    example = _load_example()
    patterns = example["_GENERATED_HELPER_PATTERNS"]
    generated = "static int gemm_rcr_bias_float32_123456789abc(\n"

    assert not patterns["gemm_rcr"].search(generated)
    assert patterns["gemm_rcr_bias"].search(generated)


def test_clip_model_workflow_example_proves_bounded_two_tower_surface(tmp_path, monkeypatch):
    example = _load_example()
    spec = example["build_spec"]()
    inputs = example["build_validation_inputs"]()

    actual = reference_numpy(spec, inputs)
    expected = example["reference_outputs"]()

    np.testing.assert_allclose(actual["logits_per_image"], expected["logits_per_image"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["logits_per_text"], expected["logits_per_text"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["text_embeds"], expected["text_embeds"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["image_embeds"], expected["image_embeds"], atol=1e-5, rtol=1e-5)

    text_only = reference_numpy(
        example["build_text_features_spec"](),
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        },
    )["text_features"]
    image_only = reference_numpy(
        example["build_image_features_spec"](),
        {"pixel_values": inputs["pixel_values"]},
    )["image_features"]

    np.testing.assert_allclose(text_only, expected["text_features"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(image_only, expected["image_features"], atol=1e-5, rtol=1e-5)

    artifact_dir = tmp_path / "clip_model_workflow_cpu.dinoml"
    summary = example["run_example"](artifact_dir=artifact_dir)
    assert summary["name"] == "clip_model_workflow"
    assert summary["input_names"] == ["input_ids", "pixel_values", "attention_mask"]
    assert summary["output_names"] == ["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"]
    assert summary["output_shapes"] == {
        "logits_per_image": [example["IMAGE_BATCH"], example["TEXT_BATCH"]],
        "logits_per_text": [example["TEXT_BATCH"], example["IMAGE_BATCH"]],
        "text_embeds": [example["TEXT_BATCH"], example["PROJECTION"]],
        "image_embeds": [example["IMAGE_BATCH"], example["PROJECTION"]],
    }
    assert summary["node_op_counts"]["conv2d_bias"] == 1
    assert summary["node_op_counts"]["embedding"] == 3
    assert summary["node_op_counts"]["vector_norm"] == 2
    assert summary["node_op_counts"]["div"] == 2
    assert summary["node_op_counts"]["gemm_rcr"] == 3
    assert summary["node_op_counts"]["exp"] == 1
    assert summary["uses_traced_default_text_positions"] is True
    assert summary["vision_input_shape"] == [
        example["IMAGE_BATCH"],
        example["NUM_CHANNELS"],
        example["IMAGE_SIZE"],
        example["IMAGE_SIZE"],
    ]
    assert summary["patch_grid"] == [example["IMAGE_SIZE"] // example["PATCH_SIZE"]] * 2
    assert summary["cutlass_conv_statuses"] == ["bounded_runtime"]
    assert summary["has_cutlass_conv_runtime"] is True
    assert summary["has_cutlass_conv_scaffold"] is False
    assert summary["uses_similarity_transpose"] is True
    assert summary["provider_kernel_ops"] == [
        "bmm_rcr",
        "bmm_rrr",
        "conv2d_bias",
        "gemm_rcr",
        "gemm_rcr_bias",
        "gemm_rcr_bias_quick_gelu",
    ]
    assert summary["provider_kernel_libraries"] == ["cutlass_bmm", "cutlass_conv", "cutlass_gemm"]
    assert "model" in summary["required_kernel_libraries"]
    assert summary["generated_cuda_kernel_count"] >= 12
    assert summary["artifact"]["path"] == str(artifact_dir.resolve())
    assert summary["artifact"]["retained"] is True
    assert summary["artifact"]["module_exists"] is True
    assert summary["artifact"]["manifest_exists"] is True
    assert summary["artifact"]["generated_module_exists"] is True
    assert summary["artifact"]["target"]["name"] == "cpu"
    assert summary["artifact"]["target"]["arch"] == "native"
    assert summary["artifact"]["bridge_kernels"] == [
        "conv2d_bias",
        "gemm_rcr",
        "gemm_rcr_bias",
        "gemm_rcr_bias_quick_gelu",
        "bmm_rcr",
        "bmm_rrr",
    ]
    assert all(item["allclose"] for item in summary["feature_parity_vs_transformers"].values())
    for parity_group in summary["parity"].values():
        assert all(item["allclose"] for item in parity_group.values())
        assert all(item["max_abs_diff"] <= 1.0e-5 for item in parity_group.values())


def test_clip_model_workflow_example_script_smoke(tmp_path):
    artifact_dir = tmp_path / "clip_model_workflow_cli_cpu.dinoml"
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--artifact-dir", str(artifact_dir)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    summary = json.loads(result.stdout)
    assert summary["name"] == "clip_model_workflow"
    assert summary["uses_traced_default_text_positions"] is True
    assert summary["cutlass_conv_statuses"] == ["bounded_runtime"]
    assert summary["has_cutlass_conv_runtime"] is True
    assert summary["has_cutlass_conv_scaffold"] is False
    assert summary["output_names"] == ["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"]
    assert summary["artifact"]["path"] == str(artifact_dir.resolve())
    assert summary["artifact"]["retained"] is True
    assert summary["artifact"]["module_exists"] is True
    assert summary["artifact"]["target"]["name"] == "cpu"
    assert summary["artifact"]["target"]["arch"] == "native"
    for parity_group in summary["parity"].values():
        assert all(item["allclose"] for item in parity_group.values())
    assert len(summary["text_features"]) == 2
    assert len(summary["image_features"]) == 3
