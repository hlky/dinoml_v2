from __future__ import annotations

from typing import Any, Mapping, Sequence

from dinoml.frontend import Tensor, as_tensor
from dinoml.ops.collections import GATHER_INDEX_DTYPES
from dinoml.ops.registry import AttrDef, FrontendBinding, KernelBinding, OpDef, OpSchema, op_def


QWEN2_5_VL_STITCH_IMAGE_FEATURES_DTYPES = ("float16", "float32", "bfloat16")


def normalize_qwen2_5_vl_stitch_image_features_attrs(*, image_token_id: Any) -> dict[str, int]:
    if not isinstance(image_token_id, int) or isinstance(image_token_id, bool):
        raise ValueError(
            f"qwen2_5_vl_stitch_image_features image_token_id must be an integer, got {image_token_id!r}"
        )
    return {"image_token_id": int(image_token_id)}


def infer_qwen2_5_vl_stitch_image_features_shape(input_shapes: Sequence[Sequence[int]]) -> list[int]:
    return infer_qwen2_5_vl_stitch_image_features_shape_with_attrs(input_shapes, {})


def infer_qwen2_5_vl_stitch_image_features_shape_with_attrs(
    input_shapes: Sequence[Sequence[int]],
    attrs: Mapping[str, Any],
) -> list[int]:
    del attrs
    normalize_qwen2_5_vl_stitch_image_features_shapes(input_shapes)
    return list(input_shapes[1])


def normalize_qwen2_5_vl_stitch_image_features_shapes(input_shapes: Sequence[Sequence[int]]) -> None:
    if len(input_shapes) != 3:
        raise ValueError("qwen2_5_vl_stitch_image_features expects exactly three inputs")
    input_ids_shape, inputs_embeds_shape, image_features_shape = [list(shape) for shape in input_shapes]
    if len(input_ids_shape) != 2:
        raise ValueError("qwen2_5_vl_stitch_image_features expects input_ids with shape [batch, seq]")
    if len(inputs_embeds_shape) != 3:
        raise ValueError("qwen2_5_vl_stitch_image_features expects inputs_embeds with shape [batch, seq, hidden]")
    if len(image_features_shape) != 2:
        raise ValueError("qwen2_5_vl_stitch_image_features expects image_features with shape [image_seq, hidden]")
    if input_ids_shape[0] != inputs_embeds_shape[0]:
        raise ValueError("qwen2_5_vl_stitch_image_features input_ids and inputs_embeds batch sizes must match")
    if input_ids_shape[1] != inputs_embeds_shape[1]:
        raise ValueError(
            "qwen2_5_vl_stitch_image_features input_ids and inputs_embeds sequence lengths must match"
        )
    if inputs_embeds_shape[2] != image_features_shape[1]:
        raise ValueError(
            "qwen2_5_vl_stitch_image_features image_features hidden size must match inputs_embeds"
        )


@op_def
class Qwen2_5_VLStitchImageFeatures(OpDef):
    name = "qwen2_5_vl_stitch_image_features"
    schema = OpSchema(
        inputs=("input_ids", "inputs_embeds", "image_features"),
        attrs=(AttrDef("image_token_id", "int", required=True),),
    )
    infer_shape = infer_qwen2_5_vl_stitch_image_features_shape
    infer_shape_with_attrs = infer_qwen2_5_vl_stitch_image_features_shape_with_attrs
    allowed_dtypes = QWEN2_5_VL_STITCH_IMAGE_FEATURES_DTYPES
    backend_kernels = {
        "cpu": KernelBinding(
            "generated_qwen2_5_vl_stitch_image_features",
            "model",
            source_template="qwen2_5_vl_stitch_image_features_cpu.cpp.j2",
        ),
        "cuda": KernelBinding(
            "generated_qwen2_5_vl_stitch_image_features",
            "model",
            source_template="qwen2_5_vl_stitch_image_features_gpu.j2",
        ),
        "rocm": KernelBinding(
            "generated_qwen2_5_vl_stitch_image_features",
            "model",
            source_template="qwen2_5_vl_stitch_image_features_gpu.j2",
        ),
    }
    frontend = FrontendBinding("qwen2_5_vl_stitch_image_features")
    description = (
        "Apply Qwen2.5-VL masked-scatter image stitching semantics by replacing every image placeholder token "
        "row in row-major batch/sequence order with the matching runtime image feature row."
    )

    @classmethod
    def forward(
        cls,
        input_ids: Any,
        inputs_embeds: Any,
        image_features: Any,
        *,
        image_token_id: int,
    ) -> Tensor:
        input_ids_tensor = as_tensor(input_ids, dtype_hint="int64")
        inputs_embeds_tensor = as_tensor(inputs_embeds, dtype_hint="float32")
        image_features_tensor = as_tensor(image_features, dtype_hint=inputs_embeds_tensor.dtype)
        if any(
            tensor.builder is not input_ids_tensor.builder
            for tensor in (inputs_embeds_tensor, image_features_tensor)
        ):
            raise ValueError("Cannot combine tensors from different DinoML traces")
        if input_ids_tensor.dtype not in GATHER_INDEX_DTYPES:
            raise ValueError(
                "qwen2_5_vl_stitch_image_features input_ids must have dtype int64 or int32, "
                f"got {input_ids_tensor.dtype}"
            )
        if inputs_embeds_tensor.dtype not in QWEN2_5_VL_STITCH_IMAGE_FEATURES_DTYPES:
            raise ValueError(
                "qwen2_5_vl_stitch_image_features does not support dtype "
                f"{inputs_embeds_tensor.dtype}"
            )
        if image_features_tensor.dtype != inputs_embeds_tensor.dtype:
            raise ValueError(
                "qwen2_5_vl_stitch_image_features image_features dtype must match inputs_embeds dtype"
            )
        attrs = normalize_qwen2_5_vl_stitch_image_features_attrs(image_token_id=image_token_id)
        normalize_qwen2_5_vl_stitch_image_features_shapes(
            [input_ids_tensor.shape, inputs_embeds_tensor.shape, image_features_tensor.shape]
        )
        output_shape = infer_qwen2_5_vl_stitch_image_features_shape_with_attrs(
            [input_ids_tensor.shape, inputs_embeds_tensor.shape, image_features_tensor.shape],
            attrs,
        )
        return input_ids_tensor.builder.emit(
            "qwen2_5_vl_stitch_image_features",
            [input_ids_tensor, inputs_embeds_tensor, image_features_tensor],
            output_shape,
            inputs_embeds_tensor.dtype,
            attrs,
            shape_spec=[_copy_shape_dim(dim) for dim in inputs_embeds_tensor.shape_spec],
        )


def qwen2_5_vl_stitch_image_features(
    input_ids: Any,
    inputs_embeds: Any,
    image_features: Any,
    *,
    image_token_id: int,
) -> Tensor:
    return Qwen2_5_VLStitchImageFeatures.forward(
        input_ids,
        inputs_embeds,
        image_features,
        image_token_id=image_token_id,
    )


def _copy_shape_dim(dim: Any) -> Any:
    return dict(dim) if isinstance(dim, Mapping) else dim


__all__ = [
    "QWEN2_5_VL_STITCH_IMAGE_FEATURES_DTYPES",
    "Qwen2_5_VLStitchImageFeatures",
    "infer_qwen2_5_vl_stitch_image_features_shape",
    "infer_qwen2_5_vl_stitch_image_features_shape_with_attrs",
    "normalize_qwen2_5_vl_stitch_image_features_attrs",
    "normalize_qwen2_5_vl_stitch_image_features_shapes",
    "qwen2_5_vl_stitch_image_features",
]
