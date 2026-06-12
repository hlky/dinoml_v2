from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from dinoml.ir import canonical_json
from dinoml.lowering.ops.add_layer_norm import ADD_LAYER_NORM_LOWERING
from dinoml.lowering.ops.argmax import ARGMAX_LOWERING
from dinoml.lowering.ops.arange import ARANGE_LOWERING
from dinoml.lowering.ops.avg_pool1d import AVG_POOL1D_LOWERING
from dinoml.lowering.ops.avg_pool2d import AVG_POOL2D_LOWERING
from dinoml.lowering.ops.batch_gather import BATCH_GATHER_LOWERING
from dinoml.lowering.ops.batch_layernorm_sigmoid_mul import BATCH_LAYERNORM_SIGMOID_MUL_LOWERING
from dinoml.lowering.ops.bmm import BMM_LOWERINGS
from dinoml.lowering.ops.concatenate import CONCATENATE_LOWERING
from dinoml.lowering.ops.conv import (
    CONV1D_BIAS_ADD_LOWERING,
    CONV1D_BIAS_ADD_RELU_LOWERING,
    CONV1D_BIAS_LOWERING,
    CONV1D_BIAS_RELU_LOWERING,
    CONV2D_BIAS_ADD_LOWERING,
    CONV2D_BIAS_ADD_RELU_LOWERING,
    CONV2D_BIAS_LOWERING,
    CONV2D_BIAS_RELU_LOWERING,
    CONV3D_BIAS_LOWERING,
    DEPTHWISE_CONV3D_LOWERING,
    TRANSPOSED_CONV2D_BIAS_ADD_LOWERING,
    TRANSPOSED_CONV2D_BIAS_ADD_RELU_LOWERING,
    TRANSPOSED_CONV2D_BIAS_LOWERING,
    TRANSPOSED_CONV2D_BIAS_RELU_LOWERING,
    TRANSPOSED_CONV2D_LOWERING,
)
from dinoml.lowering.ops.base import OpLowering
from dinoml.lowering.ops.dynamic_slice import DYNAMIC_SLICE_LOWERING
from dinoml.lowering.ops.embedding import EMBEDDING_LOWERING
from dinoml.lowering.ops.expand import EXPAND_LOWERING
from dinoml.lowering.ops.flash_attention import (
    FLASH_ATTENTION_BIAS_LOWERING,
    FLASH_ATTENTION_LOWERING,
    FLASH_ATTENTION_QKV_LOWERING,
    FLASH_ATTENTION_STATIC_KV_CACHE_LOWERING,
    FLASH_ATTENTION_STATIC_KV_CACHE_BIAS_LOWERING,
    FLASH_ATTENTION_VARLEN_LOWERING,
)
from dinoml.lowering.ops.flip import FLIP_LOWERING
from dinoml.lowering.ops.fused_elementwise import FUSED_ELEMENTWISE_LOWERING
from dinoml.lowering.ops.gather import GATHER_LOWERING
from dinoml.lowering.ops.glm_ocr_stitch_image_features import GLM_OCR_STITCH_IMAGE_FEATURES_LOWERING
from dinoml.lowering.ops.glm_ocr_rope import GLM_OCR_TEXT_ROPE_LOWERING, GLM_OCR_VISION_ROPE_LOWERING
from dinoml.lowering.ops.group_norm import GROUP_NORM_LOWERING, GROUP_NORM_SWISH_LOWERING
from dinoml.lowering.ops.group_layernorm import GROUP_LAYERNORM_LOWERING, GROUP_LAYERNORM_SIGMOID_MUL_LOWERING
from dinoml.lowering.ops.full import FULL_LOWERING
from dinoml.lowering.ops.get_1d_rotary_pos_embed import GET_1D_ROTARY_POS_EMBED_LOWERINGS
from dinoml.lowering.ops.get_timestep_embedding import GET_TIMESTEP_EMBEDDING_LOWERING
from dinoml.lowering.ops.gemm import GEMM_LOWERINGS
from dinoml.lowering.ops.index_select import INDEX_SELECT_LOWERING
from dinoml.lowering.ops.layer_norm import LAYER_NORM_LOWERING
from dinoml.lowering.ops.layernorm_sigmoid_mul import LAYERNORM_SIGMOID_MUL_LOWERING
from dinoml.lowering.ops.masked_select import MASKED_SELECT_LOWERING
from dinoml.lowering.ops.max_pool2d import MAX_POOL2D_LOWERING
from dinoml.lowering.ops.nms_family import BATCHED_NMS_LOWERING, EFFICIENT_NMS_LOWERING, NMS_LOWERING
from dinoml.lowering.ops.pad import PAD_LOWERING
from dinoml.lowering.ops.padding_layout_helpers import PADDING_LAYOUT_HELPER_LOWERINGS
from dinoml.lowering.ops.permute import PERMUTE_LOWERINGS
from dinoml.lowering.ops.positional_helper_fusions import POSITIONAL_HELPER_FUSION_LOWERINGS
from dinoml.lowering.ops.qkv_split import QKV_SPLIT_LOWERING
from dinoml.lowering.ops.qwen2_5_vl_stitch_image_features import QWEN2_5_VL_STITCH_IMAGE_FEATURES_LOWERING
from dinoml.lowering.ops.randn import RANDN_LOWERING
from dinoml.lowering.ops.reduction import REDUCTION_LOWERINGS
from dinoml.lowering.ops.repeat_interleave import REPEAT_INTERLEAVE_LOWERING
from dinoml.lowering.ops.roi_align_family import MULTI_LEVEL_ROI_ALIGN_LOWERING, ROI_ALIGN_LOWERING
from dinoml.lowering.ops.rotary_positional_fusions import ROTARY_POSITIONAL_FUSION_LOWERINGS
from dinoml.lowering.ops.runtime_index_select import RUNTIME_INDEX_SELECT_LOWERING
from dinoml.lowering.ops.shape_buffer_count_true import SHAPE_BUFFER_COUNT_TRUE_LOWERING
from dinoml.lowering.ops.slice_scatter import SLICE_SCATTER_LOWERING
from dinoml.lowering.ops.softmax import SOFTMAX_LOWERING
from dinoml.lowering.ops.stack import STACK_LOWERING
from dinoml.lowering.ops.swiglu import SWIGLU_LOWERING
from dinoml.lowering.ops.t5_layer_norm import T5_LAYER_NORM_LOWERING
from dinoml.lowering.ops.tensor_filters import TENSOR_FILTER_HELPER_LOWERINGS
from dinoml.lowering.ops.topk import TOPK_LOWERINGS
from dinoml.lowering.ops.upsampling import UPSAMPLING_FAMILY_LOWERINGS
from dinoml.lowering.target_specs import generated_source_extension
from dinoml.ops.elementwise import FUSABLE_ELEMENTWISE_OPS


OP_LOWERINGS: dict[str, OpLowering] = {
    ADD_LAYER_NORM_LOWERING.op_name: ADD_LAYER_NORM_LOWERING,
    ARGMAX_LOWERING.op_name: ARGMAX_LOWERING,
    ARANGE_LOWERING.op_name: ARANGE_LOWERING,
    AVG_POOL1D_LOWERING.op_name: AVG_POOL1D_LOWERING,
    AVG_POOL2D_LOWERING.op_name: AVG_POOL2D_LOWERING,
    BATCH_GATHER_LOWERING.op_name: BATCH_GATHER_LOWERING,
    BATCH_LAYERNORM_SIGMOID_MUL_LOWERING.op_name: BATCH_LAYERNORM_SIGMOID_MUL_LOWERING,
    CONCATENATE_LOWERING.op_name: CONCATENATE_LOWERING,
    CONV1D_BIAS_ADD_LOWERING.op_name: CONV1D_BIAS_ADD_LOWERING,
    CONV1D_BIAS_ADD_RELU_LOWERING.op_name: CONV1D_BIAS_ADD_RELU_LOWERING,
    CONV1D_BIAS_LOWERING.op_name: CONV1D_BIAS_LOWERING,
    CONV1D_BIAS_RELU_LOWERING.op_name: CONV1D_BIAS_RELU_LOWERING,
    CONV2D_BIAS_ADD_LOWERING.op_name: CONV2D_BIAS_ADD_LOWERING,
    CONV2D_BIAS_ADD_RELU_LOWERING.op_name: CONV2D_BIAS_ADD_RELU_LOWERING,
    CONV2D_BIAS_LOWERING.op_name: CONV2D_BIAS_LOWERING,
    CONV2D_BIAS_RELU_LOWERING.op_name: CONV2D_BIAS_RELU_LOWERING,
    CONV3D_BIAS_LOWERING.op_name: CONV3D_BIAS_LOWERING,
    DEPTHWISE_CONV3D_LOWERING.op_name: DEPTHWISE_CONV3D_LOWERING,
    TRANSPOSED_CONV2D_LOWERING.op_name: TRANSPOSED_CONV2D_LOWERING,
    TRANSPOSED_CONV2D_BIAS_LOWERING.op_name: TRANSPOSED_CONV2D_BIAS_LOWERING,
    TRANSPOSED_CONV2D_BIAS_RELU_LOWERING.op_name: TRANSPOSED_CONV2D_BIAS_RELU_LOWERING,
    TRANSPOSED_CONV2D_BIAS_ADD_LOWERING.op_name: TRANSPOSED_CONV2D_BIAS_ADD_LOWERING,
    TRANSPOSED_CONV2D_BIAS_ADD_RELU_LOWERING.op_name: TRANSPOSED_CONV2D_BIAS_ADD_RELU_LOWERING,
    DYNAMIC_SLICE_LOWERING.op_name: DYNAMIC_SLICE_LOWERING,
    EMBEDDING_LOWERING.op_name: EMBEDDING_LOWERING,
    EXPAND_LOWERING.op_name: EXPAND_LOWERING,
    FLASH_ATTENTION_BIAS_LOWERING.op_name: FLASH_ATTENTION_BIAS_LOWERING,
    FLASH_ATTENTION_LOWERING.op_name: FLASH_ATTENTION_LOWERING,
    FLASH_ATTENTION_QKV_LOWERING.op_name: FLASH_ATTENTION_QKV_LOWERING,
    FLASH_ATTENTION_STATIC_KV_CACHE_LOWERING.op_name: FLASH_ATTENTION_STATIC_KV_CACHE_LOWERING,
    FLASH_ATTENTION_STATIC_KV_CACHE_BIAS_LOWERING.op_name: FLASH_ATTENTION_STATIC_KV_CACHE_BIAS_LOWERING,
    FLASH_ATTENTION_VARLEN_LOWERING.op_name: FLASH_ATTENTION_VARLEN_LOWERING,
    FLIP_LOWERING.op_name: FLIP_LOWERING,
    FUSED_ELEMENTWISE_LOWERING.op_name: FUSED_ELEMENTWISE_LOWERING,
    GATHER_LOWERING.op_name: GATHER_LOWERING,
    GLM_OCR_STITCH_IMAGE_FEATURES_LOWERING.op_name: GLM_OCR_STITCH_IMAGE_FEATURES_LOWERING,
    GLM_OCR_TEXT_ROPE_LOWERING.op_name: GLM_OCR_TEXT_ROPE_LOWERING,
    GLM_OCR_VISION_ROPE_LOWERING.op_name: GLM_OCR_VISION_ROPE_LOWERING,
    GROUP_NORM_LOWERING.op_name: GROUP_NORM_LOWERING,
    GROUP_NORM_SWISH_LOWERING.op_name: GROUP_NORM_SWISH_LOWERING,
    GROUP_LAYERNORM_LOWERING.op_name: GROUP_LAYERNORM_LOWERING,
    GROUP_LAYERNORM_SIGMOID_MUL_LOWERING.op_name: GROUP_LAYERNORM_SIGMOID_MUL_LOWERING,
    FULL_LOWERING.op_name: FULL_LOWERING,
    GET_TIMESTEP_EMBEDDING_LOWERING.op_name: GET_TIMESTEP_EMBEDDING_LOWERING,
    INDEX_SELECT_LOWERING.op_name: INDEX_SELECT_LOWERING,
    LAYER_NORM_LOWERING.op_name: LAYER_NORM_LOWERING,
    LAYERNORM_SIGMOID_MUL_LOWERING.op_name: LAYERNORM_SIGMOID_MUL_LOWERING,
    MASKED_SELECT_LOWERING.op_name: MASKED_SELECT_LOWERING,
    MAX_POOL2D_LOWERING.op_name: MAX_POOL2D_LOWERING,
    NMS_LOWERING.op_name: NMS_LOWERING,
    BATCHED_NMS_LOWERING.op_name: BATCHED_NMS_LOWERING,
    EFFICIENT_NMS_LOWERING.op_name: EFFICIENT_NMS_LOWERING,
    PAD_LOWERING.op_name: PAD_LOWERING,
    QKV_SPLIT_LOWERING.op_name: QKV_SPLIT_LOWERING,
    QWEN2_5_VL_STITCH_IMAGE_FEATURES_LOWERING.op_name: QWEN2_5_VL_STITCH_IMAGE_FEATURES_LOWERING,
    RANDN_LOWERING.op_name: RANDN_LOWERING,
    REPEAT_INTERLEAVE_LOWERING.op_name: REPEAT_INTERLEAVE_LOWERING,
    ROI_ALIGN_LOWERING.op_name: ROI_ALIGN_LOWERING,
    RUNTIME_INDEX_SELECT_LOWERING.op_name: RUNTIME_INDEX_SELECT_LOWERING,
    SHAPE_BUFFER_COUNT_TRUE_LOWERING.op_name: SHAPE_BUFFER_COUNT_TRUE_LOWERING,
    SLICE_SCATTER_LOWERING.op_name: SLICE_SCATTER_LOWERING,
    SOFTMAX_LOWERING.op_name: SOFTMAX_LOWERING,
    STACK_LOWERING.op_name: STACK_LOWERING,
    SWIGLU_LOWERING.op_name: SWIGLU_LOWERING,
    T5_LAYER_NORM_LOWERING.op_name: T5_LAYER_NORM_LOWERING,
    MULTI_LEVEL_ROI_ALIGN_LOWERING.op_name: MULTI_LEVEL_ROI_ALIGN_LOWERING,
}
OP_LOWERINGS.update(REDUCTION_LOWERINGS)
OP_LOWERINGS.update(GEMM_LOWERINGS)
OP_LOWERINGS.update(BMM_LOWERINGS)
OP_LOWERINGS.update(TOPK_LOWERINGS)
OP_LOWERINGS.update(GET_1D_ROTARY_POS_EMBED_LOWERINGS)
OP_LOWERINGS.update(POSITIONAL_HELPER_FUSION_LOWERINGS)
OP_LOWERINGS.update(ROTARY_POSITIONAL_FUSION_LOWERINGS)
OP_LOWERINGS.update(PERMUTE_LOWERINGS)
OP_LOWERINGS.update(PADDING_LAYOUT_HELPER_LOWERINGS)
OP_LOWERINGS.update(TENSOR_FILTER_HELPER_LOWERINGS)
OP_LOWERINGS.update(UPSAMPLING_FAMILY_LOWERINGS)


def generated_source_provenance(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    lowering = OP_LOWERINGS.get(str(node["op"]))
    if lowering is None or lowering.source_key is None:
        return None
    source_key = lowering.source_key(target, node, tensor_map)
    if source_key is None:
        return None
    provenance = {
        "source_key": source_key,
        "source_hash": _source_hash(source_key),
    }
    if lowering.generated_function_name is not None:
        provenance["generated_function_name"] = lowering.generated_function_name(target, node, tensor_map)
    return provenance


def render_generated_kernels(
    target: str,
    nodes: Sequence[Mapping[str, Any]],
    tensor_map: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    return collect_generated_sources(target, nodes, tensor_map)["kernels"]


def collect_generated_sources(
    target: str,
    nodes: Sequence[Mapping[str, Any]],
    tensor_map: Mapping[str, Mapping[str, Any]],
    *,
    generated_src_dir: Path | None = None,
) -> dict[str, Any]:
    kernels: list[str] = []
    manifest_sources: list[dict[str, Any]] = []
    seen_source_keys: dict[str, dict[str, Any]] = {}
    extension = _source_extension(target)
    for node in nodes:
        provenance = generated_source_provenance(target, node, tensor_map)
        lowering = OP_LOWERINGS.get(node["op"])
        if lowering is None:
            continue
        source_key = None if provenance is None else provenance["source_key"]
        function_name = None if provenance is None else provenance.get("generated_function_name")
        kernel = (
            None
            if source_key is not None and source_key in seen_source_keys
            else lowering.render_generated_kernel(target, node, tensor_map)
        )
        if kernel:
            source_key = source_key or kernel
        if source_key is None:
            continue
        existing = seen_source_keys.get(source_key)
        if existing is None:
            if not kernel:
                continue
            source_hash = str(provenance["source_hash"]) if provenance is not None else _source_hash(source_key)
            source_path = Path("ops") / str(node["op"]) / f"{source_hash}.{extension}"
            kernels.append(kernel)
            if generated_src_dir is not None:
                full_source_path = generated_src_dir / source_path
                full_source_path.parent.mkdir(parents=True, exist_ok=True)
                full_source_path.write_text(kernel, encoding="utf-8")
            existing = {
                "source_hash": source_hash,
                "emitted_source_path": source_path.as_posix(),
            }
            seen_source_keys[source_key] = existing
            emitted_new_source = True
        else:
            emitted_new_source = False
        manifest_sources.append(
            {
                "node_id": node.get("id"),
                "op": node["op"],
                "target": target,
                "generated_function_name": function_name,
                "source_key": source_key,
                "source_hash": existing["source_hash"],
                "emitted_source_path": existing["emitted_source_path"],
                "emitted_new_source": emitted_new_source,
            }
        )
    manifest = {
        "schema_version": 1,
        "target": target,
        "deduplication": "exact_source_key",
        "sources": manifest_sources,
    }
    if generated_src_dir is not None:
        generated_src_dir.mkdir(parents=True, exist_ok=True)
        (generated_src_dir / "source_manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    return {"kernels": kernels, "manifest": manifest}


def _source_extension(target: str) -> str:
    try:
        return generated_source_extension(target)
    except ValueError as exc:
        raise ValueError(f"Unsupported generated source target: {target}") from exc


def _source_hash(source_key: str) -> str:
    return hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]


def render_launch(
    target: str,
    node: Mapping[str, Any],
    tensor_map: Mapping[str, Mapping[str, Any]],
    *,
    kernel_manifest: Mapping[str, Any] | None = None,
) -> str:
    if node["op"] in FUSABLE_ELEMENTWISE_OPS:
        raise ValueError(f"{node['op']} must be lowered through fused_elementwise before {target} codegen")
    try:
        lowering = OP_LOWERINGS[node["op"]]
    except KeyError as exc:
        raise ValueError(f"Unsupported op for {target} lowering: {node['op']}") from exc
    return lowering.render_launch(target, node, tensor_map, kernel_manifest)


__all__ = [
    "OP_LOWERINGS",
    "collect_generated_sources",
    "generated_source_provenance",
    "render_generated_kernels",
    "render_launch",
]
