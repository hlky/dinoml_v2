from __future__ import annotations

from collections import Counter

from dinoml.reference import reference_numpy
from examples import clip_model_workflow, clip_text_workflow


def test_clip_text_model_uses_nn_wrappers_without_changing_trace_contract():
    spec = clip_text_workflow.build_spec()
    op_counts = Counter(node["op"] for node in spec.ir["nodes"])

    assert spec.ir["outputs"][0]["name"] == "text_features"
    assert spec.ir["outputs"][0]["shape"] == [clip_text_workflow.BATCH, clip_text_workflow.PROJECTION]
    assert op_counts["embedding"] == 2
    assert op_counts["gemm_rcr_bias"] == 5
    assert op_counts["gemm_rcr_bias_quick_gelu"] == 1
    assert op_counts["layer_norm"] == 3

    outputs = reference_numpy(spec, clip_text_workflow.build_validation_inputs())
    assert outputs["text_features"].shape == (clip_text_workflow.BATCH, clip_text_workflow.PROJECTION)


def test_clip_model_uses_nn_wrappers_without_changing_trace_contract():
    spec = clip_model_workflow.build_spec()
    op_counts = Counter(node["op"] for node in spec.ir["nodes"])
    output_shapes = {output["name"]: output["shape"] for output in spec.ir["outputs"]}

    assert output_shapes == {
        "logits_per_image": [clip_model_workflow.IMAGE_BATCH, clip_model_workflow.TEXT_BATCH],
        "logits_per_text": [clip_model_workflow.TEXT_BATCH, clip_model_workflow.IMAGE_BATCH],
        "text_embeds": [clip_model_workflow.TEXT_BATCH, clip_model_workflow.PROJECTION],
        "image_embeds": [clip_model_workflow.IMAGE_BATCH, clip_model_workflow.PROJECTION],
    }
    assert op_counts["conv2d_bias"] == 1
    assert op_counts["embedding"] == 3
    assert op_counts["gemm_rcr_bias"] == 10
    assert op_counts["gemm_rcr_bias_quick_gelu"] == 2
    assert op_counts["layer_norm"] == 7

    outputs = reference_numpy(spec, clip_model_workflow.build_validation_inputs())
    assert outputs["logits_per_image"].shape == (
        clip_model_workflow.IMAGE_BATCH,
        clip_model_workflow.TEXT_BATCH,
    )
