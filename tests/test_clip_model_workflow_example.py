import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

from dinoml.backends.cpu import execute_cpu


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "clip_model_workflow.py"


def _load_example() -> dict[str, object]:
    return runpy.run_path(str(EXAMPLE))


def test_clip_model_workflow_example_proves_bounded_two_tower_surface():
    example = _load_example()
    spec = example["build_spec"]()
    inputs = example["build_validation_inputs"]()

    actual = execute_cpu(spec, inputs)
    expected = example["reference_outputs"]()

    np.testing.assert_allclose(actual["logits_per_image"], expected["logits_per_image"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["logits_per_text"], expected["logits_per_text"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["text_embeds"], expected["text_embeds"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual["image_embeds"], expected["image_embeds"], atol=1e-5, rtol=1e-5)

    text_only = execute_cpu(
        example["build_text_features_spec"](),
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        },
    )["text_features"]
    image_only = execute_cpu(
        example["build_image_features_spec"](),
        {"pixel_values": inputs["pixel_values"]},
    )["image_features"]

    np.testing.assert_allclose(text_only, expected["text_features"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(image_only, expected["image_features"], atol=1e-5, rtol=1e-5)

    summary = example["inspect_workflow"]()
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
    assert summary["has_cutlass_conv_scaffold"] is True
    assert summary["uses_similarity_transpose"] is True
    assert summary["provider_kernel_ops"] == [
        "bmm_rcr",
        "bmm_rrr",
        "conv2d_bias",
        "gemm_rcr",
        "gemm_rcr_bias",
        "gemm_rcr_bias_fast_gelu",
    ]
    assert summary["provider_kernel_libraries"] == ["cutlass_bmm", "cutlass_conv", "cutlass_gemm"]
    assert "model" in summary["required_kernel_libraries"]
    assert summary["generated_cuda_kernel_count"] >= 12


def test_clip_model_workflow_example_script_smoke():
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
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
    assert summary["has_cutlass_conv_scaffold"] is True
    assert summary["output_names"] == ["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"]
    assert len(summary["text_features"]) == 2
    assert len(summary["image_features"]) == 3
