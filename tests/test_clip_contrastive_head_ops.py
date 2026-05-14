import numpy as np
import pytest

import dinoml as dml
from dinoml.backends.cpu import execute_cpu
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.passes import PassManager, validate_ir


TEXT_BATCH = 2
IMAGE_BATCH = 3
FEATURE_DIM = 4


class _ClipContrastiveHeadBase(dml.Module):
    def _logits_per_text(self, text_features, image_features, logit_scale):
        text_norm = dml.ops.vector_norm(text_features, dim=-1, keepdim=True)
        image_norm = dml.ops.vector_norm(image_features, dim=-1, keepdim=True)
        text_embeds = dml.ops.div(text_features, text_norm)
        image_embeds = dml.ops.div(image_features, image_norm)
        logits = dml.ops.gemm_rcr(text_embeds, image_embeds)
        logits = dml.ops.mul(logits, dml.ops.exp(logit_scale))
        return logits


class ClipContrastiveHeadTextModule(_ClipContrastiveHeadBase):
    def forward(self, text_features, image_features, logit_scale):
        return dml.ops.output(self._logits_per_text(text_features, image_features, logit_scale), "logits_per_text")


class ClipContrastiveHeadImageModule(_ClipContrastiveHeadBase):
    def forward(self, text_features, image_features, logit_scale):
        logits_per_text = self._logits_per_text(text_features, image_features, logit_scale)
        logits_per_image = dml.ops.transpose(logits_per_text, 0, 1)
        return dml.ops.output(logits_per_image, "logits_per_image")


def _trace(output_kind: str):
    module = ClipContrastiveHeadTextModule() if output_kind == "text" else ClipContrastiveHeadImageModule()
    return dml.trace(
        module,
        inputs={
            "text_features": dml.TensorSpec([TEXT_BATCH, FEATURE_DIM], "float32"),
            "image_features": dml.TensorSpec([IMAGE_BATCH, FEATURE_DIM], "float32"),
            "logit_scale": dml.TensorSpec([1], "float32"),
        },
        name=f"clip_contrastive_head_{output_kind}",
    )


def _text_features():
    return np.array(
        [
            [0.25, -0.50, 0.75, 1.00],
            [1.50, 0.25, -0.75, 0.50],
        ],
        dtype=np.float32,
    )


def _image_features():
    return np.array(
        [
            [0.50, -1.25, 1.00, 0.25],
            [-0.75, 0.50, 0.25, 1.50],
            [1.00, 0.75, -0.50, -0.25],
        ],
        dtype=np.float32,
    )


def _logit_scale():
    return np.array([np.log(2.5)], dtype=np.float32)


def _reference_logits_per_text(text_features, image_features, logit_scale):
    text_norm = np.linalg.norm(text_features, axis=-1, keepdims=True).astype(np.float32)
    image_norm = np.linalg.norm(image_features, axis=-1, keepdims=True).astype(np.float32)
    text_embeds = (text_features / text_norm).astype(np.float32)
    image_embeds = (image_features / image_norm).astype(np.float32)
    return (text_embeds @ image_embeds.T * np.exp(logit_scale).astype(np.float32)).astype(np.float32)


@pytest.mark.parametrize(
    ("output_kind", "expected_shape", "expected_name"),
    [
        ("text", [TEXT_BATCH, IMAGE_BATCH], "logits_per_text"),
        ("image", [IMAGE_BATCH, TEXT_BATCH], "logits_per_image"),
    ],
)
def test_clip_contrastive_head_frontend_ir_and_cpu_reference_match_numpy(output_kind, expected_shape, expected_name):
    spec = _trace(output_kind)
    text_features = _text_features()
    image_features = _image_features()
    logit_scale = _logit_scale()

    node_ops = [node["op"] for node in spec.ir["nodes"]]
    assert node_ops.count("vector_norm") == 2
    assert node_ops.count("div") == 2
    assert node_ops.count("gemm_rcr") == 1
    assert node_ops.count("exp") == 1
    assert node_ops.count("mul") == 1
    assert node_ops.count("permute") == (1 if output_kind == "image" else 0)
    assert spec.ir["outputs"][0]["shape"] == expected_shape
    assert spec.ir["outputs"][0]["dtype"] == "float32"
    assert spec.ir["outputs"][0]["name"] == expected_name

    actual = execute_cpu(
        spec,
        {
            "text_features": text_features,
            "image_features": image_features,
            "logit_scale": logit_scale,
        },
    )["logits_per_image" if output_kind == "image" else "logits_per_text"]
    expected = _reference_logits_per_text(text_features, image_features, logit_scale)
    if output_kind == "image":
        expected = expected.T

    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_clip_contrastive_head_manifest_keeps_gemm_and_generated_math_roles_honest():
    spec = _trace("image")
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)

    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    ops = [entry["op"] for entry in manifest["required_kernels"]]

    assert "gemm_rcr" in ops
    assert "vector_norm" in ops
    assert "permute" in ops
    assert "fused_elementwise" in ops

    gemm_entries = [entry for entry in manifest["required_kernels"] if entry["op"] == "gemm_rcr"]
    norm_entries = [entry for entry in manifest["required_kernels"] if entry["op"] == "vector_norm"]
    fused_entries = [entry for entry in manifest["required_kernels"] if entry["op"] == "fused_elementwise"]

    assert gemm_entries and all(entry["kernel_library"] == "cutlass_gemm" for entry in gemm_entries)
    assert gemm_entries[0]["kernel_symbol"].startswith("dinoml_cutlass_gemm_rcr_float32_")
    assert norm_entries and all(entry["kernel_library"] == "model" for entry in norm_entries)
    assert norm_entries[0]["kernel_symbol"] == "generated_reduction"
    assert fused_entries and all(entry["kernel_library"] == "model" for entry in fused_entries)
    assert all(entry["kernel_symbol"] == "generated_fused_elementwise" for entry in fused_entries)
