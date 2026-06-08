from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import dinoml as dml
from dinoml.ir import array_from_storage, array_to_storage, dtype_numpy, normalize_dtype
from dinoml.models.kv_cache import append_static_kv_cache
from dinoml.shapes import symbolic_int_expr


_QWEN_FLOAT_DTYPES = frozenset({"float16", "float32", "bfloat16"})


def _normalize_qwen_dtype(dtype: str) -> str:
    normalized = normalize_dtype(dtype)
    if normalized not in _QWEN_FLOAT_DTYPES:
        raise ValueError(f"Qwen2.5-VL currently supports float16/float32/bfloat16 parameters, got {dtype!r}")
    return normalized


def _numpy_qwen_dtype(dtype: str) -> np.dtype:
    dtype = _normalize_qwen_dtype(dtype)
    return np.dtype(np.uint16) if dtype == "bfloat16" else dtype_numpy(dtype)


def _qwen_float_storage(values: np.ndarray, dtype: str) -> np.ndarray:
    dtype = _normalize_qwen_dtype(dtype)
    if dtype == "bfloat16":
        return array_to_storage(values.astype(np.float32, copy=False), "bfloat16")
    return values.astype(dtype_numpy(dtype), copy=False)


@dataclass(frozen=True)
class Qwen2_5_VLVisionConfig:
    depth: int = 32
    hidden_size: int = 1280
    hidden_act: str = "silu"
    intermediate_size: int = 3420
    num_heads: int = 16
    in_channels: int = 3
    out_hidden_size: int = 2048
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    window_size: int = 112
    fullatt_block_indexes: tuple[int, ...] = (7, 15, 23, 31)
    tokens_per_second: int = 2
    rms_norm_eps: float = 1.0e-6
    dtype: str = "bfloat16"
    use_flash_attention: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _normalize_qwen_dtype(self.dtype))
        object.__setattr__(self, "fullatt_block_indexes", tuple(int(v) for v in self.fullatt_block_indexes))
        _require_positive(self.depth, "vision depth", allow_zero=True)
        _require_positive(self.hidden_size, "vision hidden_size")
        _require_positive(self.intermediate_size, "vision intermediate_size")
        _require_positive(self.out_hidden_size, "vision out_hidden_size")
        _require_positive(self.num_heads, "vision num_heads")
        _require_positive(self.in_channels, "vision in_channels")
        _require_positive(self.patch_size, "vision patch_size")
        _require_positive(self.temporal_patch_size, "vision temporal_patch_size")
        _require_positive(self.spatial_merge_size, "vision spatial_merge_size")
        _require_positive(self.window_size, "vision window_size")
        if self.hidden_act != "silu":
            raise ValueError(f"Qwen2.5-VL vision only supports hidden_act='silu', got {self.hidden_act!r}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("vision hidden_size must be divisible by num_heads")
        if self.window_size % (self.spatial_merge_size * self.patch_size) != 0:
            raise ValueError("vision window_size must be divisible by spatial_merge_size * patch_size")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size

    @property
    def spatial_merge_unit(self) -> int:
        return self.spatial_merge_size * self.spatial_merge_size

    @property
    def vit_merger_window_size(self) -> int:
        return self.window_size // self.spatial_merge_size // self.patch_size


@dataclass(frozen=True)
class Qwen2_5_VLTextConfig:
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 128000
    initializer_range: float = 0.02
    rms_norm_eps: float = 1.0e-6
    use_cache: bool = True
    attention_dropout: float = 0.0
    rope_parameters: Mapping[str, Any] = field(
        default_factory=lambda: {
            "rope_type": "default",
            "mrope_section": [16, 24, 24],
            "rope_theta": 1_000_000.0,
        }
    )
    bos_token_id: int | None = 151643
    eos_token_id: int | None = 151645
    pad_token_id: int | None = 151643
    tie_word_embeddings: bool = True
    sliding_window: int | None = None
    use_sliding_window: bool = False
    dtype: str = "bfloat16"
    mask_fill_value: float = -1.0e4
    use_flash_attention: bool = True
    use_flash_attention_bias: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _normalize_qwen_dtype(self.dtype))
        _require_positive(self.vocab_size, "text vocab_size")
        _require_positive(self.hidden_size, "text hidden_size")
        _require_positive(self.intermediate_size, "text intermediate_size")
        _require_positive(self.num_hidden_layers, "text num_hidden_layers", allow_zero=True)
        _require_positive(self.num_attention_heads, "text num_attention_heads")
        _require_positive(self.num_key_value_heads, "text num_key_value_heads")
        if self.hidden_act != "silu":
            raise ValueError(f"Qwen2.5-VL text only supports hidden_act='silu', got {self.hidden_act!r}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("text hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.attention_dropout != 0:
            raise ValueError("Qwen2.5-VL inference expects attention_dropout=0")
        rope = _normalize_qwen_rope_parameters(self.rope_parameters, rope_theta=1_000_000.0)
        mrope_section = tuple(int(v) for v in rope.get("mrope_section", (16, 24, 24)))
        if len(mrope_section) != 3 or any(v <= 0 for v in mrope_section):
            raise ValueError("rope_parameters.mrope_section must contain three positive integers")
        if sum(mrope_section) != self.rotary_freq_dim:
            raise ValueError("sum(rope_parameters.mrope_section) must equal head_dim / 2")
        rope["mrope_section"] = list(mrope_section)
        object.__setattr__(self, "rope_parameters", rope)
        if not self.use_sliding_window:
            object.__setattr__(self, "sliding_window", None)
        else:
            raise NotImplementedError("Qwen2.5-VL sliding-window text attention is not currently supported")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_proj_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def rotary_dim(self) -> int:
        return self.head_dim

    @property
    def rotary_freq_dim(self) -> int:
        return self.head_dim // 2

    @property
    def layer_types(self) -> tuple[str, ...]:
        if self.use_sliding_window and self.sliding_window is not None:
            return tuple("sliding_attention" for _ in range(self.num_hidden_layers))
        return tuple("full_attention" for _ in range(self.num_hidden_layers))


@dataclass(frozen=True)
class Qwen2_5_VLConfig:
    text_config: Qwen2_5_VLTextConfig = field(default_factory=Qwen2_5_VLTextConfig)
    vision_config: Qwen2_5_VLVisionConfig = field(default_factory=Qwen2_5_VLVisionConfig)
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    image_token_id: int = 151655
    video_token_id: int = 151656
    tie_word_embeddings: bool = True


def qwen2_5_vl_config_from_transformers_dict(
    payload: Mapping[str, Any],
    *,
    dtype: str | None = None,
) -> Qwen2_5_VLConfig:
    if str(payload.get("model_type")) != "qwen2_5_vl":
        raise ValueError(f"expected model_type='qwen2_5_vl', got {payload.get('model_type')!r}")
    requested_dtype = dtype or payload.get("torch_dtype") or payload.get("dtype") or "bfloat16"
    vision_payload = _normalize_qwen_vision_payload(dict(payload.get("vision_config") or {}), payload, requested_dtype)
    text_payload = _normalize_qwen_text_payload(dict(payload), requested_dtype)
    text_config = Qwen2_5_VLTextConfig(**text_payload)
    vision_config = Qwen2_5_VLVisionConfig(**vision_payload)
    return Qwen2_5_VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        vision_start_token_id=int(payload.get("vision_start_token_id", 151652)),
        vision_end_token_id=int(payload.get("vision_end_token_id", 151653)),
        vision_token_id=int(payload.get("vision_token_id", 151654)),
        image_token_id=int(payload.get("image_token_id", 151655)),
        video_token_id=int(payload.get("video_token_id", 151656)),
        tie_word_embeddings=bool(payload.get("tie_word_embeddings", text_config.tie_word_embeddings)),
    )


def qwen2_5_vl_config_from_transformers_config(config: object, *, dtype: str | None = None) -> Qwen2_5_VLConfig:
    if isinstance(config, Mapping):
        return qwen2_5_vl_config_from_transformers_dict(config, dtype=dtype)
    return qwen2_5_vl_config_from_transformers_dict(_object_public_attrs(config), dtype=dtype)


def _normalize_qwen_text_payload(payload: dict[str, Any], dtype: str) -> dict[str, Any]:
    allowed = {
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_act",
        "max_position_embeddings",
        "initializer_range",
        "rms_norm_eps",
        "use_cache",
        "attention_dropout",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "tie_word_embeddings",
        "sliding_window",
        "use_sliding_window",
    }
    text = {key: payload[key] for key in allowed if key in payload}
    if "pad_token_id" not in text and "bos_token_id" in text:
        text["pad_token_id"] = text["bos_token_id"]
    if "num_key_value_heads" not in text or text["num_key_value_heads"] is None:
        text["num_key_value_heads"] = int(text.get("num_attention_heads", Qwen2_5_VLTextConfig.num_attention_heads))
    text["rope_parameters"] = _normalize_qwen_rope_parameters(
        payload.get("rope_parameters") or payload.get("rope_scaling") or {},
        rope_theta=float(payload.get("rope_theta", 1_000_000.0)),
    )
    text["dtype"] = dtype
    return text


def _normalize_qwen_vision_payload(vision_payload: dict[str, Any], root_payload: Mapping[str, Any], dtype: str) -> dict[str, Any]:
    vision = dict(vision_payload)
    if "in_channels" not in vision and "in_chans" in vision:
        vision["in_channels"] = vision.pop("in_chans")
    else:
        vision.pop("in_chans", None)
    if "patch_size" not in vision and "spatial_patch_size" in vision:
        vision["patch_size"] = vision["spatial_patch_size"]
    vision.pop("spatial_patch_size", None)
    vision.pop("model_type", None)
    vision.pop("torch_dtype", None)
    vision["out_hidden_size"] = int(vision.get("out_hidden_size", root_payload.get("hidden_size", 2048)))
    vision["dtype"] = dtype
    allowed = set(Qwen2_5_VLVisionConfig.__dataclass_fields__)
    return {key: value for key, value in vision.items() if key in allowed}


def _normalize_qwen_rope_parameters(raw: Mapping[str, Any], *, rope_theta: float) -> dict[str, Any]:
    rope = dict(raw or {})
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type == "mrope":
        rope_type = "default"
    if rope_type != "default":
        raise ValueError(f"Qwen2.5-VL currently supports default/mrope RoPE only, got {rope_type!r}")
    return {
        "rope_type": "default",
        "mrope_section": list(rope.get("mrope_section", [16, 24, 24])),
        "rope_theta": float(rope.get("rope_theta", rope_theta)),
    }


def qwen2_5_vl_patch_embed_linear_weight(conv3d_weight: object, *, dtype: str = "float32") -> np.ndarray:
    weight = _state_value_to_numpy(conv3d_weight, dtype=dtype)
    if weight.ndim != 5:
        raise ValueError(f"patch embedding Conv3d weight must be rank 5, got shape {weight.shape}")
    return np.reshape(weight, (weight.shape[0], int(np.prod(weight.shape[1:]))))


def qwen2_5_vl_text_rope_embeddings(
    position_ids: object,
    config: Qwen2_5_VLTextConfig,
    *,
    dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(position_ids, dtype=np.float32)
    if pos.ndim != 3 or pos.shape[0] != 3:
        raise ValueError(f"position_ids must have shape [3, batch, seq], got {pos.shape}")
    inv_freq = qwen2_5_vl_text_inv_freq(config, dtype="float32")
    freqs = np.einsum("f,tbs->tbsf", inv_freq, pos, dtype=np.float32)
    emb = np.concatenate([freqs, freqs], axis=-1)
    section = [int(v) * 2 for v in config.rope_parameters["mrope_section"]]
    chunks = np.split(emb, np.cumsum(section)[:-1], axis=-1)
    selected = np.concatenate([chunk[idx % 3] for idx, chunk in enumerate(chunks)], axis=-1)
    return _qwen_float_storage(np.cos(selected), dtype), _qwen_float_storage(np.sin(selected), dtype)


def qwen2_5_vl_text_inv_freq(config: Qwen2_5_VLTextConfig, *, dtype: str = "float32") -> np.ndarray:
    base = float(config.rope_parameters["rope_theta"])
    dim = config.head_dim
    values = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / float(dim)))
    return _qwen_float_storage(values, dtype)


def qwen2_5_vl_vision_position_ids(grid_thw: object, spatial_merge_size: int) -> np.ndarray:
    grids = np.asarray(grid_thw, dtype=np.int64)
    if grids.ndim != 2 or grids.shape[1] != 3:
        raise ValueError(f"grid_thw must have shape [num_items, 3], got {grids.shape}")
    rows: list[np.ndarray] = []
    merge = int(spatial_merge_size)
    for grid_t, grid_h, grid_w in grids:
        grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
        if grid_h % merge != 0 or grid_w % merge != 0:
            raise ValueError("grid_thw height/width must be divisible by spatial_merge_size")
        hpos = np.arange(grid_h, dtype=np.int64).reshape(grid_h, 1)
        hpos = np.repeat(hpos, grid_w, axis=1)
        hpos = hpos.reshape(grid_h // merge, merge, grid_w // merge, merge)
        hpos = np.transpose(hpos, (0, 2, 1, 3)).reshape(-1)
        wpos = np.arange(grid_w, dtype=np.int64).reshape(1, grid_w)
        wpos = np.repeat(wpos, grid_h, axis=0)
        wpos = wpos.reshape(grid_h // merge, merge, grid_w // merge, merge)
        wpos = np.transpose(wpos, (0, 2, 1, 3)).reshape(-1)
        rows.append(np.stack([hpos, wpos], axis=-1).repeat(grid_t, axis=0))
    return np.concatenate(rows, axis=0) if rows else np.empty((0, 2), dtype=np.int64)


def qwen2_5_vl_vision_rope_embeddings(
    position_ids: object,
    *,
    head_dim: int,
    theta: float = 10000.0,
    dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(position_ids, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"vision position_ids must have shape [seq, 2], got {pos.shape}")
    rotary_dim = int(head_dim) // 2
    inv_freq = 1.0 / (float(theta) ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / float(rotary_dim)))
    freqs = (pos[..., None] * inv_freq).reshape(pos.shape[0], -1)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return _qwen_float_storage(np.cos(emb), dtype), _qwen_float_storage(np.sin(emb), dtype)


def qwen2_5_vl_vision_cu_seqlens(grid_thw: object) -> np.ndarray:
    grids = np.asarray(grid_thw, dtype=np.int64)
    if grids.ndim != 2 or grids.shape[1] != 3:
        raise ValueError(f"grid_thw must have shape [num_items, 3], got {grids.shape}")
    lengths = np.repeat(grids[:, 1] * grids[:, 2], grids[:, 0]).astype(np.int64, copy=False)
    return np.concatenate([np.zeros((1,), dtype=np.int32), np.cumsum(lengths, dtype=np.int64).astype(np.int32)])


def qwen2_5_vl_vision_window_index(
    grid_thw: object,
    *,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    grids = np.asarray(grid_thw, dtype=np.int64)
    if grids.ndim != 2 or grids.shape[1] != 3:
        raise ValueError(f"grid_thw must have shape [num_items, 3], got {grids.shape}")
    merge = int(spatial_merge_size)
    vit_window = int(window_size) // merge // int(patch_size)
    spatial_merge_unit = merge * merge
    window_index: list[np.ndarray] = []
    cu_window_seqlens: list[int] = [0]
    window_index_id = 0
    for grid_t, grid_h, grid_w in grids:
        grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
        llm_grid_h = grid_h // merge
        llm_grid_w = grid_w // merge
        index = np.arange(grid_t * llm_grid_h * llm_grid_w, dtype=np.int64).reshape(grid_t, llm_grid_h, llm_grid_w)
        pad_h = vit_window - llm_grid_h % vit_window
        pad_w = vit_window - llm_grid_w % vit_window
        num_windows_h = (llm_grid_h + pad_h) // vit_window
        num_windows_w = (llm_grid_w + pad_w) // vit_window
        padded = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
        padded = padded.reshape(grid_t, num_windows_h, vit_window, num_windows_w, vit_window)
        padded = np.transpose(padded, (0, 1, 3, 2, 4)).reshape(grid_t, num_windows_h * num_windows_w, vit_window, vit_window)
        seqlens = (padded != -100).sum(axis=(2, 3)).reshape(-1)
        index_new = padded.reshape(-1)
        index_new = index_new[index_new != -100]
        window_index.append(index_new + window_index_id)
        cumulative = np.cumsum(seqlens, dtype=np.int64) * spatial_merge_unit + cu_window_seqlens[-1]
        cu_window_seqlens.extend(int(v) for v in cumulative.tolist())
        window_index_id += grid_t * llm_grid_h * llm_grid_w
    merged = np.concatenate(window_index, axis=0) if window_index else np.empty((0,), dtype=np.int64)
    cu = np.asarray(cu_window_seqlens, dtype=np.int32)
    cu = cu[np.concatenate([[True], cu[1:] != cu[:-1]])]
    return merged, cu


def qwen2_5_vl_rope_index(
    input_ids: object,
    mm_token_type_ids: object,
    *,
    image_grid_thw: object | None = None,
    attention_mask: object | None = None,
    spatial_merge_size: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(input_ids)
    token_types = np.asarray(mm_token_type_ids)
    if ids.ndim != 2 or token_types.shape != ids.shape:
        raise ValueError("input_ids and mm_token_type_ids must have matching [batch, seq] shapes")
    image_iter = iter(np.asarray(image_grid_thw, dtype=np.int64)) if image_grid_thw is not None else None
    masks = None if attention_mask is None else np.asarray(attention_mask).astype(bool)
    position_ids = np.zeros((3, ids.shape[0], ids.shape[1]), dtype=ids.dtype)
    deltas: list[int] = []
    for batch_idx in range(ids.shape[0]):
        current_types = token_types[batch_idx]
        current_mask = None if masks is None else masks[batch_idx]
        if current_mask is not None:
            current_types = current_types[current_mask]
        current_pos = 0
        pieces: list[np.ndarray] = []
        for modality_type, group in groupby(enumerate(current_types.tolist()), lambda item: item[1]):
            group_items = list(group)
            start_idx = group_items[0][0]
            end_idx = group_items[-1][0] + 1
            if modality_type == 0:
                text_len = end_idx - start_idx
                text_pos = np.arange(text_len, dtype=ids.dtype).reshape(1, -1) + current_pos
                pieces.append(np.repeat(text_pos, 3, axis=0))
                current_pos += text_len
            elif modality_type == 1:
                if image_iter is None:
                    raise ValueError("missing image_grid_thw for image token type")
                grid = next(image_iter)
                vision_pos = _single_qwen_vision_rope_index(
                    current_pos,
                    grid,
                    spatial_merge_size=spatial_merge_size,
                    dtype=ids.dtype,
                )
                pieces.append(vision_pos)
                current_pos += int(max(grid[1], grid[2]) // spatial_merge_size)
            elif modality_type == 2:
                raise NotImplementedError("Qwen2.5-VL video rope indexing is deferred for the image-first milestone")
            else:
                raise ValueError(f"unsupported mm_token_type_id {modality_type}")
        llm_positions = np.concatenate(pieces, axis=1) if pieces else np.empty((3, 0), dtype=ids.dtype)
        if current_mask is not None:
            position_ids[:, batch_idx, current_mask] = llm_positions
            unpadded_len = int(current_mask.sum())
        else:
            position_ids[:, batch_idx, :] = llm_positions
            unpadded_len = ids.shape[1]
        deltas.append(int(llm_positions.max() + 1 - unpadded_len) if llm_positions.size else 0)
    return position_ids, np.asarray(deltas, dtype=ids.dtype).reshape(-1, 1)


def qwen2_5_vl_stitch_image_features(
    input_ids: object,
    inputs_embeds: object,
    image_features: object,
    *,
    image_token_id: int,
) -> np.ndarray:
    ids = np.asarray(input_ids)
    embeds = np.asarray(inputs_embeds).copy()
    features = np.asarray(image_features, dtype=embeds.dtype)
    if ids.shape != embeds.shape[:2]:
        raise ValueError("input_ids shape must match inputs_embeds leading [batch, seq] dimensions")
    if features.ndim != 2 or features.shape[-1] != embeds.shape[-1]:
        raise ValueError("image_features must have shape [num_image_tokens, hidden_size]")
    positions = np.argwhere(ids == int(image_token_id))
    if positions.shape[0] != features.shape[0]:
        raise ValueError(
            f"Image features and image tokens do not match, tokens: {positions.shape[0]}, features: {features.shape[0]}"
        )
    for feature_idx, (batch_idx, seq_idx) in enumerate(positions):
        embeds[batch_idx, seq_idx, :] = features[feature_idx]
    return embeds


def qwen2_5_vl_prepare_inputs_for_generation(
    model_inputs: Mapping[str, Any],
    *,
    is_first_iteration: bool,
    use_cache: bool = True,
) -> dict[str, Any]:
    prepared = dict(model_inputs)
    if not is_first_iteration and use_cache:
        prepared["pixel_values"] = None
        prepared["pixel_values_videos"] = None
    return prepared


def _optional_constant_parameter(name: str, value: object | None, dtype: str) -> dml.Parameter | None:
    if value is None:
        return None
    array = np.asarray(value)
    return dml.Parameter(list(array.shape), dtype=_normalize_qwen_dtype(dtype), name=name, value=value)


def _prefill_rope_input(name: str, provided: object | None, parameter: dml.Parameter | None):
    if provided is not None:
        return provided
    if parameter is None:
        raise ValueError(f"{name} must be provided as a trace input or baked prefill constant")
    return parameter


def _decode_rope_input(name: str, provided: object | None, parameter: dml.Parameter | None, cache_seqlens: object | None):
    if provided is not None:
        return provided
    if parameter is None:
        raise ValueError(f"{name} must be provided as a trace input or baked decode constant")
    if cache_seqlens is None:
        raise ValueError(f"{name} baked decode constant requires cache_seqlens")
    batch = int(cache_seqlens.shape[0])
    indices = dml.ops.reshape(cache_seqlens, [batch, 1])
    return dml.ops.batch_gather(parameter, indices)


class Qwen2_5_VLTextAttention(dml.nn.Module):
    def __init__(self, config: Qwen2_5_VLTextConfig, weights: Mapping[str, np.ndarray], prefix: str):
        self.config = config
        self.q_proj = _loaded_linear(
            weights,
            parameter_prefix="q_proj",
            weight_key=f"{prefix}.q_proj.weight",
            bias_key=f"{prefix}.q_proj.bias",
            in_features=config.hidden_size,
            out_features=config.q_proj_size,
            dtype=config.dtype,
        )
        self.k_proj = _loaded_linear(
            weights,
            parameter_prefix="k_proj",
            weight_key=f"{prefix}.k_proj.weight",
            bias_key=f"{prefix}.k_proj.bias",
            in_features=config.hidden_size,
            out_features=config.kv_proj_size,
            dtype=config.dtype,
        )
        self.v_proj = _loaded_linear(
            weights,
            parameter_prefix="v_proj",
            weight_key=f"{prefix}.v_proj.weight",
            bias_key=f"{prefix}.v_proj.bias",
            in_features=config.hidden_size,
            out_features=config.kv_proj_size,
            dtype=config.dtype,
        )
        self.o_proj = _loaded_linear(
            weights,
            parameter_prefix="o_proj",
            weight_key=f"{prefix}.o_proj.weight",
            in_features=config.q_proj_size,
            out_features=config.hidden_size,
            dtype=config.dtype,
        )

    def _project_qkv(self, hidden_states, cos, sin):
        batch, seq_len, _ = _rank3_shape(hidden_states.shape_spec, "Qwen2.5-VL hidden_states")
        q = dml.ops.reshape(self.q_proj(hidden_states), [batch, seq_len, self.config.num_attention_heads, self.config.head_dim])
        k = dml.ops.reshape(self.k_proj(hidden_states), [batch, seq_len, self.config.num_key_value_heads, self.config.head_dim])
        v = dml.ops.reshape(self.v_proj(hidden_states), [batch, seq_len, self.config.num_key_value_heads, self.config.head_dim])
        q = dml.ops.permute(q, (0, 2, 1, 3))
        k = dml.ops.permute(k, (0, 2, 1, 3))
        v = dml.ops.permute(v, (0, 2, 1, 3))
        q, k = _qwen_text_rope(q, k, cos, sin)
        return q, k, v, batch, seq_len

    def _project_attention_output(self, context, batch: Any, seq_len: Any):
        context = dml.ops.permute(context, (0, 2, 1, 3))
        context = dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size])
        return self.o_proj(context)

    def _dense_attention_output(self, q, k, v, attention_mask, batch: Any, q_len: Any):
        repeated_k = _repeat_kv_heads(k, self.config.num_key_value_groups)
        repeated_v = _repeat_kv_heads(v, self.config.num_key_value_groups)
        _, heads, _, head_dim = _rank4_shape(q.shape_spec, "Qwen2.5-VL q")
        _, _, kv_len, _ = _rank4_shape(repeated_k.shape_spec, "Qwen2.5-VL k")
        q_flat = dml.ops.reshape(q, [_shape_mul(batch, heads), q_len, head_dim])
        k_flat = dml.ops.reshape(repeated_k, [_shape_mul(batch, heads), kv_len, head_dim])
        v_flat = dml.ops.reshape(repeated_v, [_shape_mul(batch, heads), kv_len, head_dim])
        scores = dml.ops.mul(dml.ops.bmm_rcr(q_flat, k_flat), float(self.config.head_dim) ** -0.5)
        if attention_mask is not None:
            scores = dml.ops.add(scores, attention_mask)
        probs = dml.ops.softmax(scores, dim=-1)
        context = dml.ops.bmm_rrr(probs, v_flat)
        context = dml.ops.reshape(context, [batch, heads, q_len, head_dim])
        return self._project_attention_output(context, batch, q_len)

    def _require_dense_prefill_mask(self, attention_mask) -> None:
        if attention_mask is None:
            raise ValueError("Qwen2.5-VL dense text prefill requires an explicit causal attention_mask")

    def _require_dense_cached_mask(self, attention_mask, seq_len: Any) -> None:
        if attention_mask is not None or seq_len == 1:
            return
        raise ValueError(
            "Qwen2.5-VL dense cached text attention requires an explicit attention_mask when query length != 1"
        )

    def forward(self, hidden_states, cos, sin, attention_mask=None):
        q, k, v, batch, seq_len = self._project_qkv(hidden_states, cos, sin)
        self._require_dense_prefill_mask(attention_mask)
        return self._dense_attention_output(q, k, v, attention_mask, batch, seq_len)

    def prefill_with_cache(self, hidden_states, cos, sin, attention_mask=None):
        q, k, v, batch, seq_len = self._project_qkv(hidden_states, cos, sin)
        self._require_dense_prefill_mask(attention_mask)
        return self._dense_attention_output(q, k, v, attention_mask, batch, seq_len), k, v

    def forward_with_cache(self, hidden_states, cos, sin, past_key, past_value, attention_mask=None):
        q, new_key, new_value, batch, seq_len = self._project_qkv(hidden_states, cos, sin)
        present_key = dml.ops.concatenate([past_key, new_key], dim=2)
        present_value = dml.ops.concatenate([past_value, new_value], dim=2)
        self._require_dense_cached_mask(attention_mask, seq_len)
        return (
            self._dense_attention_output(q, present_key, present_value, attention_mask, batch, seq_len),
            present_key,
            present_value,
        )

    def forward_with_static_cache(
        self,
        hidden_states,
        cos,
        sin,
        past_key,
        past_value,
        attention_mask=None,
        cache_seqlens=None,
    ):
        del cache_seqlens
        q, new_key, new_value, batch, seq_len = self._project_qkv(hidden_states, cos, sin)
        attn_key, attn_value = append_static_kv_cache(past_key, past_value, new_key, new_value)
        self._require_dense_cached_mask(attention_mask, seq_len)
        return (
            self._dense_attention_output(q, attn_key, attn_value, attention_mask, batch, seq_len),
            new_key,
            new_value,
        )


class Qwen2_5_VLTextFlashAttention(Qwen2_5_VLTextAttention):
    def _flash_attention_output(self, q, k, v, attention_mask, *, causal: bool, batch: Any, seq_len: Any):
        q_flash = dml.ops.permute(q, (0, 2, 1, 3))
        k_flash = dml.ops.permute(k, (0, 2, 1, 3))
        v_flash = dml.ops.permute(v, (0, 2, 1, 3))
        if attention_mask is None:
            context = dml.ops.flash_attention(q_flash, k_flash, v_flash, causal=causal)
            return self.o_proj(dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size]))
        if self.config.use_flash_attention_bias:
            context = dml.ops.flash_attention_bias(q_flash, k_flash, v_flash, attention_mask, causal=causal)
            return self.o_proj(dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size]))
        return self._dense_attention_output(q, k, v, attention_mask, batch, seq_len)

    def forward(self, hidden_states, cos, sin, attention_mask=None):
        q, k, v, batch, seq_len = self._project_qkv(hidden_states, cos, sin)
        return self._flash_attention_output(q, k, v, attention_mask, causal=True, batch=batch, seq_len=seq_len)

    def prefill_with_cache(self, hidden_states, cos, sin, attention_mask=None):
        q, k, v, batch, seq_len = self._project_qkv(hidden_states, cos, sin)
        return self._flash_attention_output(q, k, v, attention_mask, causal=True, batch=batch, seq_len=seq_len), k, v

    def forward_with_cache(self, hidden_states, cos, sin, past_key, past_value, attention_mask=None):
        q, new_key, new_value, batch, seq_len = self._project_qkv(hidden_states, cos, sin)
        present_key = dml.ops.concatenate([past_key, new_key], dim=2)
        present_value = dml.ops.concatenate([past_value, new_value], dim=2)
        if attention_mask is None and seq_len == 1:
            return (
                self._flash_attention_output(q, present_key, present_value, None, causal=False, batch=batch, seq_len=seq_len),
                present_key,
                present_value,
            )
        if self.config.use_flash_attention_bias and attention_mask is not None and seq_len == 1:
            return (
                self._flash_attention_output(
                    q,
                    present_key,
                    present_value,
                    attention_mask,
                    causal=False,
                    batch=batch,
                    seq_len=seq_len,
                ),
                present_key,
                present_value,
            )
        return super().forward_with_cache(hidden_states, cos, sin, past_key, past_value, attention_mask)

    def forward_with_static_cache(
        self,
        hidden_states,
        cos,
        sin,
        past_key,
        past_value,
        attention_mask=None,
        cache_seqlens=None,
    ):
        q, new_key, new_value, batch, seq_len = self._project_qkv(hidden_states, cos, sin)
        if cache_seqlens is not None:
            q_flash = dml.ops.permute(q, (0, 2, 1, 3))
            if attention_mask is None:
                context = dml.ops.flash_attention_static_kv_cache(
                    q_flash,
                    past_key,
                    past_value,
                    new_key,
                    new_value,
                    cache_seqlens,
                )
                return self.o_proj(dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size])), new_key, new_value
            if self.config.use_flash_attention_bias:
                context = dml.ops.flash_attention_static_kv_cache_bias(
                    q_flash,
                    past_key,
                    past_value,
                    new_key,
                    new_value,
                    cache_seqlens,
                    attention_mask,
                )
                return self.o_proj(dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size])), new_key, new_value
        return super().forward_with_static_cache(
            hidden_states,
            cos,
            sin,
            past_key,
            past_value,
            attention_mask=attention_mask,
            cache_seqlens=cache_seqlens,
        )


class Qwen2_5_VLTextMLP(dml.nn.Module):
    def __init__(self, config: Qwen2_5_VLTextConfig, weights: Mapping[str, np.ndarray], prefix: str):
        self.gate_proj = _loaded_linear(
            weights,
            parameter_prefix="gate_proj",
            weight_key=f"{prefix}.gate_proj.weight",
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            dtype=config.dtype,
        )
        self.up_proj = _loaded_linear(
            weights,
            parameter_prefix="up_proj",
            weight_key=f"{prefix}.up_proj.weight",
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            dtype=config.dtype,
        )
        self.down_proj = _loaded_linear(
            weights,
            parameter_prefix="down_proj",
            weight_key=f"{prefix}.down_proj.weight",
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            dtype=config.dtype,
        )

    def forward(self, hidden_states):
        return self.down_proj(dml.ops.mul(dml.ops.silu(self.gate_proj(hidden_states)), self.up_proj(hidden_states)))


class Qwen2_5_VLTextDecoderLayer(dml.nn.Module):
    def __init__(self, config: Qwen2_5_VLTextConfig, weights: Mapping[str, np.ndarray], layer_idx: int):
        prefix = f"model.layers.{layer_idx}"
        self.input_layernorm = _loaded_rms_norm(
            weights,
            parameter_prefix="input_layernorm",
            weight_key=f"{prefix}.input_layernorm.weight",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )
        attn_cls = (
            Qwen2_5_VLTextFlashAttention
            if config.use_flash_attention and config.dtype in {"float16", "bfloat16"}
            else Qwen2_5_VLTextAttention
        )
        self.self_attn = attn_cls(config, weights, f"{prefix}.self_attn")
        self.post_attention_layernorm = _loaded_rms_norm(
            weights,
            parameter_prefix="post_attention_layernorm",
            weight_key=f"{prefix}.post_attention_layernorm.weight",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )
        self.mlp = Qwen2_5_VLTextMLP(config, weights, f"{prefix}.mlp")

    def prefill_with_cache(self, hidden_states, cos, sin, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key, present_value = self.self_attn.prefill_with_cache(hidden_states, cos, sin, attention_mask)
        hidden_states = dml.ops.add(residual, attn_output)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = dml.ops.add(residual, self.mlp(hidden_states))
        return hidden_states, present_key, present_value

    def forward_with_cache(self, hidden_states, cos, sin, past_key, past_value, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key, present_value = self.self_attn.forward_with_cache(
            hidden_states, cos, sin, past_key, past_value, attention_mask
        )
        hidden_states = dml.ops.add(residual, attn_output)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = dml.ops.add(residual, self.mlp(hidden_states))
        return hidden_states, present_key, present_value

    def forward_with_static_cache(self, hidden_states, cos, sin, past_key, past_value, attention_mask=None, cache_seqlens=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, new_key, new_value = self.self_attn.forward_with_static_cache(
            hidden_states, cos, sin, past_key, past_value, attention_mask, cache_seqlens
        )
        hidden_states = dml.ops.add(residual, attn_output)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = dml.ops.add(residual, self.mlp(hidden_states))
        return hidden_states, new_key, new_value


class Qwen2_5_VLTextModel(dml.nn.Module):
    def __init__(self, config: Qwen2_5_VLTextConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.embed_tokens = _loaded_embedding(
            weights,
            parameter_prefix="embed_tokens",
            weight_key="model.embed_tokens.weight",
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
        )
        self.layers = dml.nn.ModuleList(Qwen2_5_VLTextDecoderLayer(config, weights, idx) for idx in range(config.num_hidden_layers))
        self.norm = _loaded_rms_norm(
            weights,
            parameter_prefix="norm",
            weight_key="model.norm.weight",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )

    def encode_inputs_embeds_with_cache(self, inputs_embeds, cos, sin, attention_mask=None):
        hidden_states = inputs_embeds
        present = []
        for layer in self.layers:
            hidden_states, present_key, present_value = layer.prefill_with_cache(hidden_states, cos, sin, attention_mask)
            present.append((present_key, present_value))
        return self.norm(hidden_states), present

    def forward(self, input_ids, cos, sin, attention_mask=None):
        hidden_states, _ = self.encode_inputs_embeds_with_cache(self.embed_tokens(input_ids), cos, sin, attention_mask)
        return hidden_states

    def decode(self, input_ids, cos, sin, attention_mask=None, past_key_values=None):
        hidden_states = self.embed_tokens(input_ids)
        present = []
        for layer_idx, layer in enumerate(self.layers):
            past_key, past_value = past_key_values[layer_idx]
            hidden_states, present_key, present_value = layer.forward_with_cache(
                hidden_states, cos, sin, past_key, past_value, attention_mask
            )
            present.append((present_key, present_value))
        return self.norm(hidden_states), present

    def decode_static_cache(self, input_ids, cos, sin, attention_mask=None, past_key_values=None, cache_seqlens=None):
        hidden_states = self.embed_tokens(input_ids)
        updates = []
        for layer_idx, layer in enumerate(self.layers):
            past_key, past_value = past_key_values[layer_idx]
            hidden_states, new_key, new_value = layer.forward_with_static_cache(
                hidden_states, cos, sin, past_key, past_value, attention_mask, cache_seqlens
            )
            updates.append((new_key, new_value))
        return self.norm(hidden_states), updates


class Qwen2_5_VLVisionPatchEmbedLinear(dml.nn.Module):
    def __init__(self, config: Qwen2_5_VLVisionConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.proj = _loaded_linear(
            weights,
            parameter_prefix="visual_patch_embed_proj",
            weight_key="visual.patch_embed.proj.weight",
            in_features=config.patch_dim,
            out_features=config.hidden_size,
            dtype=config.dtype,
            reshape_rank5_weight=True,
        )

    def forward(self, pixel_values):
        return self.proj(pixel_values)


class Qwen2_5_VLVisionMLP(dml.nn.Module):
    def __init__(self, config: Qwen2_5_VLVisionConfig, weights: Mapping[str, np.ndarray], prefix: str):
        self.gate_proj = _loaded_linear(
            weights,
            parameter_prefix="vision_gate_proj",
            weight_key=f"{prefix}.gate_proj.weight",
            bias_key=f"{prefix}.gate_proj.bias",
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            dtype=config.dtype,
        )
        self.up_proj = _loaded_linear(
            weights,
            parameter_prefix="vision_up_proj",
            weight_key=f"{prefix}.up_proj.weight",
            bias_key=f"{prefix}.up_proj.bias",
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            dtype=config.dtype,
        )
        self.down_proj = _loaded_linear(
            weights,
            parameter_prefix="vision_down_proj",
            weight_key=f"{prefix}.down_proj.weight",
            bias_key=f"{prefix}.down_proj.bias",
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            dtype=config.dtype,
        )

    def forward(self, hidden_states):
        return self.down_proj(dml.ops.mul(dml.ops.silu(self.gate_proj(hidden_states)), self.up_proj(hidden_states)))


class Qwen2_5_VLVisionAttention(dml.nn.Module):
    def __init__(
        self,
        config: Qwen2_5_VLVisionConfig,
        weights: Mapping[str, np.ndarray],
        prefix: str,
        *,
        max_seqlen: int | None = None,
    ):
        self.config = config
        self.max_seqlen = int(max_seqlen) if max_seqlen is not None else None
        self.qkv = _loaded_linear(
            weights,
            parameter_prefix="vision_qkv",
            weight_key=f"{prefix}.qkv.weight",
            bias_key=f"{prefix}.qkv.bias",
            in_features=config.hidden_size,
            out_features=config.hidden_size * 3,
            dtype=config.dtype,
        )
        self.proj = _loaded_linear(
            weights,
            parameter_prefix="vision_proj",
            weight_key=f"{prefix}.proj.weight",
            bias_key=f"{prefix}.proj.bias",
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            dtype=config.dtype,
        )

    def _project_qkv(self, hidden_states, cos, sin):
        seq_len, _ = _rank2_shape(hidden_states.shape_spec, "Qwen2.5-VL vision hidden_states")
        qkv = self.qkv(hidden_states)
        q, k, v = dml.ops.qkv_split(qkv)
        q = dml.ops.reshape(q, [seq_len, self.config.num_heads, self.config.head_dim])
        k = dml.ops.reshape(k, [seq_len, self.config.num_heads, self.config.head_dim])
        v = dml.ops.reshape(v, [seq_len, self.config.num_heads, self.config.head_dim])
        q, k = dml.ops.glm_ocr_vision_rope(q, k, cos, sin)
        return q, k, v, seq_len

    def forward(self, hidden_states, cos, sin, cu_seqlens=None):
        q, k, v, seq_len = self._project_qkv(hidden_states, cos, sin)
        if cu_seqlens is not None:
            raise ValueError("Qwen2.5-VL runtime vision cu_seqlens require flash attention")
        context = self._dense_attention(q, k, v)
        context = dml.ops.reshape(context, [seq_len, self.config.hidden_size])
        return self.proj(context)

    def _dense_attention(self, q, k, v):
        q_flat = dml.ops.permute(q, (1, 0, 2))
        k_flat = dml.ops.permute(k, (1, 0, 2))
        v_flat = dml.ops.permute(v, (1, 0, 2))
        _, _, head_dim = _rank3_shape(q.shape_spec, "Qwen2.5-VL vision attention")
        scores = dml.ops.mul(dml.ops.bmm_rcr(q_flat, k_flat), float(head_dim) ** -0.5)
        probs = dml.ops.softmax(scores, dim=-1)
        context = dml.ops.bmm_rrr(probs, v_flat)
        return dml.ops.permute(context, (1, 0, 2))


class Qwen2_5_VLVisionFlashAttention(Qwen2_5_VLVisionAttention):
    def forward(self, hidden_states, cos, sin, cu_seqlens=None):
        q, k, v, seq_len = self._project_qkv(hidden_states, cos, sin)
        length, heads, head_dim = _rank3_shape(q.shape_spec, "Qwen2.5-VL vision attention")
        if cu_seqlens is not None:
            if self.max_seqlen is None:
                if not isinstance(length, int):
                    raise ValueError("Qwen2.5-VL dynamic vision cu_seqlens require max_seqlen")
                max_seqlen = int(length)
            else:
                max_seqlen = self.max_seqlen
            context = dml.ops.flash_attention_varlen(q, k, v, cu_seqlens, max_seqlen=max_seqlen, causal=False)
        else:
            q_flash = dml.ops.reshape(q, [1, length, heads, head_dim])
            k_flash = dml.ops.reshape(k, [1, length, heads, head_dim])
            v_flash = dml.ops.reshape(v, [1, length, heads, head_dim])
            context = dml.ops.reshape(
                dml.ops.flash_attention(q_flash, k_flash, v_flash, causal=False),
                [length, heads, head_dim],
            )
        context = dml.ops.reshape(context, [seq_len, self.config.hidden_size])
        return self.proj(context)


class Qwen2_5_VLVisionBlock(dml.nn.Module):
    def __init__(
        self,
        config: Qwen2_5_VLVisionConfig,
        weights: Mapping[str, np.ndarray],
        layer_idx: int,
        *,
        max_seqlen: int | None = None,
    ):
        prefix = f"visual.blocks.{layer_idx}"
        self.norm1 = _loaded_rms_norm(
            weights,
            parameter_prefix="vision_norm1",
            weight_key=f"{prefix}.norm1.weight",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )
        attn_cls = (
            Qwen2_5_VLVisionFlashAttention
            if config.use_flash_attention and config.dtype in {"float16", "bfloat16"}
            else Qwen2_5_VLVisionAttention
        )
        self.attn = attn_cls(config, weights, f"{prefix}.attn", max_seqlen=max_seqlen)
        self.norm2 = _loaded_rms_norm(
            weights,
            parameter_prefix="vision_norm2",
            weight_key=f"{prefix}.norm2.weight",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )
        self.mlp = Qwen2_5_VLVisionMLP(config, weights, f"{prefix}.mlp")

    def forward(self, hidden_states, cos, sin, cu_seqlens=None):
        hidden_states = dml.ops.add(hidden_states, self.attn(self.norm1(hidden_states), cos, sin, cu_seqlens))
        hidden_states = dml.ops.add(hidden_states, self.mlp(self.norm2(hidden_states)))
        return hidden_states


class Qwen2_5_VLPatchMerger(dml.nn.Module):
    def __init__(self, config: Qwen2_5_VLVisionConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.ln_q = _loaded_rms_norm(
            weights,
            parameter_prefix="visual_merger_ln_q",
            weight_key="visual.merger.ln_q.weight",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )
        merged = config.hidden_size * config.spatial_merge_unit
        self.fc1 = _loaded_linear(
            weights,
            parameter_prefix="visual_merger_mlp_0",
            weight_key="visual.merger.mlp.0.weight",
            bias_key="visual.merger.mlp.0.bias",
            in_features=merged,
            out_features=merged,
            dtype=config.dtype,
        )
        self.fc2 = _loaded_linear(
            weights,
            parameter_prefix="visual_merger_mlp_2",
            weight_key="visual.merger.mlp.2.weight",
            bias_key="visual.merger.mlp.2.bias",
            in_features=merged,
            out_features=config.out_hidden_size,
            dtype=config.dtype,
        )

    def forward(self, hidden_states):
        seq_len, _ = _rank2_shape(hidden_states.shape_spec, "Qwen2.5-VL patch merger input")
        hidden_states = self.ln_q(hidden_states)
        hidden_states = dml.ops.reshape(hidden_states, [_shape_div(seq_len, self.config.spatial_merge_unit), self.config.hidden_size * self.config.spatial_merge_unit])
        return self.fc2(dml.ops.gelu(self.fc1(hidden_states)))


class Qwen2_5_VLVisionModel(dml.nn.Module):
    def __init__(
        self,
        config: Qwen2_5_VLVisionConfig,
        weights: Mapping[str, np.ndarray],
        *,
        grid_thw: object | None = None,
        max_full_seqlen: int | None = None,
        max_window_seqlen: int | None = None,
    ):
        self.config = config
        self.patch_embed = Qwen2_5_VLVisionPatchEmbedLinear(config, weights)
        if grid_thw is not None:
            raise ValueError(
                "Qwen2.5-VL vision grid_thw must remain external; pass grid_thw=None and provide "
                "window-ordered pixel_values/cos/sin plus runtime cu_seqlens/reverse_window_index"
            )
        blocks = []
        for idx in range(config.depth):
            max_seqlen = max_full_seqlen if idx in config.fullatt_block_indexes else max_window_seqlen
            blocks.append(Qwen2_5_VLVisionBlock(config, weights, idx, max_seqlen=max_seqlen))
        self.blocks = dml.nn.ModuleList(blocks)
        self.merger = Qwen2_5_VLPatchMerger(config, weights)

    def forward(self, pixel_values, cos, sin, full_cu_seqlens=None, window_cu_seqlens=None, reverse_window_index=None):
        hidden_states = self.patch_embed(pixel_values)
        for idx, block in enumerate(self.blocks):
            runtime_cu = full_cu_seqlens if idx in self.config.fullatt_block_indexes else window_cu_seqlens
            hidden_states = block(hidden_states, cos, sin, runtime_cu)
        hidden_states = self.merger(hidden_states)
        if reverse_window_index is not None:
            hidden_states = dml.ops.runtime_index_select(hidden_states, 0, reverse_window_index)
        return dml.ops.output(hidden_states, "pooler_output")


class Qwen2_5_VLForConditionalGeneration(dml.nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, weights: Mapping[str, np.ndarray], *, logits_to_keep: int = 0):
        self.config = config
        self.logits_to_keep = int(logits_to_keep)
        if self.logits_to_keep not in {0, 1}:
            raise ValueError("Qwen2_5_VLForConditionalGeneration currently supports logits_to_keep=0 or 1")
        self.language_model = Qwen2_5_VLTextModel(config.text_config, weights)
        self.lm_head = _loaded_lm_head(weights, config)

    def forward(self, input_ids, cos, sin, attention_mask=None):
        hidden_states = self.language_model(input_ids, cos, sin, attention_mask)
        if self.logits_to_keep == 1:
            batch, seq_len, hidden = _rank3_shape(hidden_states.shape_spec, "Qwen2.5-VL hidden_states")
            hidden_states = dml.ops.dynamic_slice(
                hidden_states,
                (0, dml.ops.int_sub(seq_len, 1), 0),
                (batch, 1, hidden),
            )
        logits = self.lm_head(hidden_states)
        return {"logits": dml.ops.output(logits, "logits")}


class Qwen2_5_VLForConditionalGenerationImagePrefillWithCache(dml.nn.Module):
    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        weights: Mapping[str, np.ndarray],
        *,
        grid_thw: object | None = None,
        logits_to_keep: int = 1,
        vision_cos: object | None = None,
        vision_sin: object | None = None,
        text_cos: object | None = None,
        text_sin: object | None = None,
        vision_rope_dtype: str = "float32",
        max_full_seqlen: int | None = None,
        max_window_seqlen: int | None = None,
    ):
        self.config = config
        self.logits_to_keep = int(logits_to_keep)
        if self.logits_to_keep not in {0, 1}:
            raise ValueError("Qwen2_5_VLForConditionalGenerationImagePrefillWithCache currently supports logits_to_keep=0 or 1")
        self._vision_cos = _optional_constant_parameter("vision_cos", vision_cos, vision_rope_dtype)
        self._vision_sin = _optional_constant_parameter("vision_sin", vision_sin, vision_rope_dtype)
        self._text_cos = _optional_constant_parameter("text_cos", text_cos, config.text_config.dtype)
        self._text_sin = _optional_constant_parameter("text_sin", text_sin, config.text_config.dtype)
        self.visual = Qwen2_5_VLVisionModel(
            config.vision_config,
            weights,
            grid_thw=grid_thw,
            max_full_seqlen=max_full_seqlen,
            max_window_seqlen=max_window_seqlen,
        )
        self.language_model = Qwen2_5_VLTextModel(config.text_config, weights)
        self.lm_head = _loaded_lm_head(weights, config)

    def forward(
        self,
        input_ids,
        pixel_values,
        vision_cos=None,
        vision_sin=None,
        text_cos=None,
        text_sin=None,
        attention_mask=None,
        image_grid_thw=None,
        vision_full_cu_seqlens=None,
        vision_window_cu_seqlens=None,
        vision_reverse_window_index=None,
    ):
        del image_grid_thw
        vision_cos = _prefill_rope_input("vision_cos", vision_cos, self._vision_cos)
        vision_sin = _prefill_rope_input("vision_sin", vision_sin, self._vision_sin)
        text_cos = _prefill_rope_input("text_cos", text_cos, self._text_cos)
        text_sin = _prefill_rope_input("text_sin", text_sin, self._text_sin)
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        image_features = self.visual(
            pixel_values,
            vision_cos,
            vision_sin,
            vision_full_cu_seqlens,
            vision_window_cu_seqlens,
            vision_reverse_window_index,
        )
        inputs_embeds = dml.ops.qwen2_5_vl_stitch_image_features(
            input_ids,
            inputs_embeds,
            image_features,
            image_token_id=self.config.image_token_id,
        )
        hidden_states, present = self.language_model.encode_inputs_embeds_with_cache(inputs_embeds, text_cos, text_sin, attention_mask)
        if self.logits_to_keep == 1:
            batch, seq_len, hidden = _rank3_shape(hidden_states.shape_spec, "Qwen2.5-VL hidden_states")
            hidden_states = dml.ops.dynamic_slice(
                hidden_states,
                (0, dml.ops.int_sub(seq_len, 1), 0),
                (batch, 1, hidden),
            )
        logits = self.lm_head(hidden_states)
        outputs: dict[str, Any] = {"logits": dml.ops.output(logits, "logits")}
        for layer_idx, (present_key, present_value) in enumerate(present):
            outputs[f"present_key_{layer_idx}"] = dml.ops.output(present_key, f"present_key_{layer_idx}")
            outputs[f"present_value_{layer_idx}"] = dml.ops.output(present_value, f"present_value_{layer_idx}")
        return outputs


class Qwen2_5_VLForConditionalGenerationDecode(dml.nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, weights: Mapping[str, np.ndarray], *, text_cos: object | None = None, text_sin: object | None = None):
        self.config = config
        self._text_cos = _optional_constant_parameter("text_cos", text_cos, config.text_config.dtype)
        self._text_sin = _optional_constant_parameter("text_sin", text_sin, config.text_config.dtype)
        self.language_model = Qwen2_5_VLTextModel(config.text_config, weights)
        self.lm_head = _loaded_lm_head(weights, config)

    def forward(self, input_ids, cos=None, sin=None, attention_mask=None, cache_seqlens=None, **past_key_values):
        cos = _decode_rope_input("text_cos", cos, self._text_cos, cache_seqlens)
        sin = _decode_rope_input("text_sin", sin, self._text_sin, cache_seqlens)
        past = {}
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            past[layer_idx] = (past_key_values[f"past_key_{layer_idx}"], past_key_values[f"past_value_{layer_idx}"])
        hidden_states, present = self.language_model.decode(input_ids, cos, sin, attention_mask, past)
        logits = self.lm_head(hidden_states)
        outputs: dict[str, Any] = {"logits": dml.ops.output(logits, "logits")}
        for layer_idx, (present_key, present_value) in enumerate(present):
            outputs[f"present_key_{layer_idx}"] = dml.ops.output(present_key, f"present_key_{layer_idx}")
            outputs[f"present_value_{layer_idx}"] = dml.ops.output(present_value, f"present_value_{layer_idx}")
        return outputs


class Qwen2_5_VLForConditionalGenerationDecodeStaticCache(dml.nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, weights: Mapping[str, np.ndarray], *, text_cos: object | None = None, text_sin: object | None = None):
        self.config = config
        self._text_cos = _optional_constant_parameter("text_cos", text_cos, config.text_config.dtype)
        self._text_sin = _optional_constant_parameter("text_sin", text_sin, config.text_config.dtype)
        self.language_model = Qwen2_5_VLTextModel(config.text_config, weights)
        self.lm_head = _loaded_lm_head(weights, config)

    def forward(self, input_ids, cos=None, sin=None, attention_mask=None, cache_seqlens=None, **past_key_values):
        cos = _decode_rope_input("text_cos", cos, self._text_cos, cache_seqlens)
        sin = _decode_rope_input("text_sin", sin, self._text_sin, cache_seqlens)
        past = {}
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            past[layer_idx] = (past_key_values[f"past_key_{layer_idx}"], past_key_values[f"past_value_{layer_idx}"])
        hidden_states, updates = self.language_model.decode_static_cache(input_ids, cos, sin, attention_mask, past, cache_seqlens)
        logits = self.lm_head(hidden_states)
        outputs: dict[str, Any] = {"logits": dml.ops.output(logits, "logits")}
        for layer_idx, (new_key, new_value) in enumerate(updates):
            outputs[f"new_key_{layer_idx}"] = dml.ops.output(new_key, f"new_key_{layer_idx}")
            outputs[f"new_value_{layer_idx}"] = dml.ops.output(new_value, f"new_value_{layer_idx}")
        return outputs


def qwen2_5_vl_required_text_weight_names(config: Qwen2_5_VLConfig) -> list[str]:
    text = config.text_config
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    if not config.tie_word_embeddings and not text.tie_word_embeddings:
        names.append("lm_head.weight")
    for idx in range(text.num_hidden_layers):
        prefix = f"model.layers.{idx}"
        names.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
    return names


def qwen2_5_vl_required_weight_names(config: Qwen2_5_VLConfig) -> list[str]:
    names = [*qwen2_5_vl_required_text_weight_names(config)]
    vision = config.vision_config
    names.extend(
        [
            "visual.patch_embed.proj.weight",
            "visual.merger.ln_q.weight",
            "visual.merger.mlp.0.weight",
            "visual.merger.mlp.0.bias",
            "visual.merger.mlp.2.weight",
            "visual.merger.mlp.2.bias",
        ]
    )
    for idx in range(vision.depth):
        prefix = f"visual.blocks.{idx}"
        names.extend(
            [
                f"{prefix}.norm1.weight",
                f"{prefix}.attn.qkv.weight",
                f"{prefix}.attn.qkv.bias",
                f"{prefix}.attn.proj.weight",
                f"{prefix}.attn.proj.bias",
                f"{prefix}.norm2.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.gate_proj.bias",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.up_proj.bias",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.mlp.down_proj.bias",
            ]
        )
    return names


def qwen2_5_vl_weights_from_transformers_state_dict(
    state_dict: Mapping[str, object],
    config: Qwen2_5_VLConfig,
    *,
    dtype: str | None = None,
    required_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    dtype = config.text_config.dtype if dtype is None else _normalize_qwen_dtype(dtype)
    required = list(required_names) if required_names is not None else qwen2_5_vl_required_weight_names(config)
    missing = [name for name in required if name not in state_dict]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} missing total)"
        raise KeyError(f"Missing Transformers Qwen2.5-VL state_dict weights: {preview}{suffix}")
    return {name: _state_value_to_numpy(state_dict[name], dtype=dtype) for name in required}


def qwen2_5_vl_weights_from_safetensors_file(
    path: str | Path,
    config: Qwen2_5_VLConfig,
    *,
    dtype: str | None = None,
    required_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("qwen2_5_vl_weights_from_safetensors_file requires the optional safetensors package") from exc
    dtype = config.text_config.dtype if dtype is None else _normalize_qwen_dtype(dtype)
    required = list(required_names) if required_names is not None else qwen2_5_vl_required_weight_names(config)
    weights: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        missing = [name for name in required if name not in available]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} missing total)"
            raise KeyError(f"Missing Qwen2.5-VL safetensors weights: {preview}{suffix}")
        for name in required:
            weights[name] = _state_value_to_numpy(handle.get_tensor(name), dtype=dtype)
    return weights


def qwen2_5_vl_weights_from_safetensors_index(
    path: str | Path,
    config: Qwen2_5_VLConfig,
    *,
    dtype: str | None = None,
    required_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("qwen2_5_vl_weights_from_safetensors_index requires the optional safetensors package") from exc
    dtype = config.text_config.dtype if dtype is None else _normalize_qwen_dtype(dtype)
    index_path = _resolve_safetensors_index_path(path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = dict(payload.get("weight_map") or {})
    required = list(required_names) if required_names is not None else qwen2_5_vl_required_weight_names(config)
    missing = [name for name in required if name not in weight_map]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} missing total)"
        raise KeyError(f"Missing Qwen2.5-VL safetensors index entries: {preview}{suffix}")
    grouped: dict[str, list[str]] = {}
    for name in required:
        grouped.setdefault(str(weight_map[name]), []).append(name)
    weights: dict[str, np.ndarray] = {}
    for shard_name, names in grouped.items():
        shard_path = index_path.parent / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            shard_missing = [name for name in names if name not in available]
            if shard_missing:
                preview = ", ".join(shard_missing[:5])
                suffix = "" if len(shard_missing) <= 5 else f", ... ({len(shard_missing)} missing total)"
                raise KeyError(f"Missing Qwen2.5-VL shard weights in {shard_path}: {preview}{suffix}")
            for name in names:
                weights[name] = _state_value_to_numpy(handle.get_tensor(name), dtype=dtype)
    return weights


def _resolve_safetensors_index_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "model.safetensors.index.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"Qwen2.5-VL safetensors index not found: {candidate}")
    return candidate


def _loaded_lm_head(weights: Mapping[str, np.ndarray], config: Qwen2_5_VLConfig) -> dml.nn.Linear:
    weight_key = "lm_head.weight"
    if weight_key not in weights:
        if not (config.tie_word_embeddings or config.text_config.tie_word_embeddings):
            raise KeyError("Missing Qwen2.5-VL weight: lm_head.weight")
        weight_key = "model.embed_tokens.weight"
    return _loaded_linear(
        weights,
        parameter_prefix="lm_head",
        weight_key=weight_key,
        in_features=config.text_config.hidden_size,
        out_features=config.text_config.vocab_size,
        dtype=config.text_config.dtype,
    )


def _loaded_linear(
    weights: Mapping[str, np.ndarray],
    *,
    parameter_prefix: str,
    weight_key: str,
    in_features: int,
    out_features: int,
    bias_key: str | None = None,
    dtype: str = "float32",
    reshape_rank5_weight: bool = False,
) -> dml.nn.Linear:
    dtype = _normalize_qwen_dtype(dtype)
    layer = dml.nn.Linear(in_features, out_features, bias=bias_key is not None, dtype=dtype)
    weight = _weight_value(weights, weight_key, None, dtype=dtype)
    if reshape_rank5_weight:
        weight = np.reshape(weight, (out_features, in_features))
    _check_shape(weight, (out_features, in_features), weight_key)
    layer.weight = dml.Parameter([out_features, in_features], dtype=dtype, name=f"{parameter_prefix}_weight", value=weight)
    if bias_key is not None:
        layer.bias = dml.Parameter(
            [out_features],
            dtype=dtype,
            name=f"{parameter_prefix}_bias",
            value=_weight_value(weights, bias_key, (out_features,), dtype=dtype),
        )
    return layer


def _loaded_embedding(
    weights: Mapping[str, np.ndarray],
    *,
    parameter_prefix: str,
    weight_key: str,
    num_embeddings: int,
    embedding_dim: int,
    dtype: str,
) -> dml.nn.Embedding:
    dtype = _normalize_qwen_dtype(dtype)
    layer = dml.nn.Embedding(num_embeddings, embedding_dim, dtype=dtype)
    layer.weight = dml.Parameter(
        [num_embeddings, embedding_dim],
        dtype=dtype,
        name=f"{parameter_prefix}_weight",
        value=_weight_value(weights, weight_key, (num_embeddings, embedding_dim), dtype=dtype),
    )
    return layer


def _loaded_rms_norm(
    weights: Mapping[str, np.ndarray],
    *,
    parameter_prefix: str,
    weight_key: str,
    hidden_size: int,
    eps: float,
    dtype: str,
) -> dml.nn.RMSNorm:
    dtype = _normalize_qwen_dtype(dtype)
    layer = dml.nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)
    layer.weight = dml.Parameter(
        [hidden_size],
        dtype=dtype,
        name=f"{parameter_prefix}_weight",
        value=_weight_value(weights, weight_key, (hidden_size,), dtype=dtype),
    )
    return layer


def _qwen_text_rope(q, k, cos, sin):
    cos = dml.ops.unsqueeze(cos, 1)
    sin = dml.ops.unsqueeze(sin, 1)
    return (
        dml.ops.add(dml.ops.mul(q, cos), dml.ops.mul(_rotate_half(q), sin)),
        dml.ops.add(dml.ops.mul(k, cos), dml.ops.mul(_rotate_half(k), sin)),
    )


def _rotate_half(x):
    last_dim = int(x.shape[-1])
    half = last_dim // 2
    x1 = dml.ops.index_select(x, -1, range(0, half))
    x2 = dml.ops.index_select(x, -1, range(half, last_dim))
    return dml.ops.concatenate([dml.ops.mul(x2, -1.0), x1], dim=-1)


def _repeat_kv_heads(values, n_rep: int):
    if n_rep == 1:
        return values
    _, kv_heads, _, _ = values.shape
    indices = [head for head in range(int(kv_heads)) for _ in range(int(n_rep))]
    return dml.ops.index_select(values, 1, indices)


def _weight_value(weights: Mapping[str, np.ndarray], name: str, shape: tuple[int, ...] | None, *, dtype: str) -> np.ndarray:
    if name not in weights:
        raise KeyError(f"Missing Qwen2.5-VL weight: {name}")
    value = _state_value_to_numpy(weights[name], dtype=dtype)
    if shape is not None:
        _check_shape(value, shape, name)
    return value


def _state_value_to_numpy(value: object, *, dtype: str) -> np.ndarray:
    dtype = _normalize_qwen_dtype(dtype)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float") and str(getattr(value, "dtype", "")).endswith("bfloat16"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if dtype == "bfloat16":
        return array_to_storage(array, dtype)
    if array.dtype == np.uint16:
        array = array_from_storage(array, "bfloat16")
    return np.asarray(array, dtype=_numpy_qwen_dtype(dtype))


def _check_shape(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if value.shape != shape:
        raise ValueError(f"Weight {name} has shape {value.shape}, expected {shape}")


def _single_qwen_vision_rope_index(start_position: int, grid_thw: Sequence[int], *, spatial_merge_size: int, dtype: np.dtype) -> np.ndarray:
    grid_t, grid_h, grid_w = (int(v) for v in grid_thw)
    llm_grid_t = grid_t
    llm_grid_h = grid_h // spatial_merge_size
    llm_grid_w = grid_w // spatial_merge_size
    temporal = np.arange(llm_grid_t, dtype=dtype).repeat(llm_grid_h * llm_grid_w) + start_position
    height = np.arange(llm_grid_h, dtype=dtype).repeat(llm_grid_w).repeat(llm_grid_t) + start_position
    width = np.tile(np.arange(llm_grid_w, dtype=dtype), llm_grid_h * llm_grid_t) + start_position
    return np.stack([temporal, height, width], axis=0)


def _rank2_shape(shape: Sequence[Any], name: str) -> tuple[Any, Any]:
    if len(shape) != 2:
        raise ValueError(f"{name} must be rank 2")
    return shape[0], shape[1]


def _rank3_shape(shape: Sequence[Any], name: str) -> tuple[Any, Any, Any]:
    if len(shape) != 3:
        raise ValueError(f"{name} must be rank 3")
    return shape[0], shape[1], shape[2]


def _rank4_shape(shape: Sequence[Any], name: str) -> tuple[Any, Any, Any, Any]:
    if len(shape) != 4:
        raise ValueError(f"{name} must be rank 4")
    return shape[0], shape[1], shape[2], shape[3]


def _shape_mul(value: Any, factor: Any) -> Any:
    if isinstance(value, int) and isinstance(factor, int):
        return value * factor
    return symbolic_int_expr("mul", value, factor)


def _shape_div(value: Any, divisor: int) -> Any:
    if isinstance(value, int):
        return value // int(divisor)
    return symbolic_int_expr("div", value, int(divisor))


def _require_positive(value: int, name: str, *, allow_zero: bool = False) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    elif value <= 0:
        raise ValueError(f"{name} must be positive")


def _object_public_attrs(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return {name: getattr(value, name) for name in dir(value) if not name.startswith("_") and not callable(getattr(value, name))}


__all__ = [
    "Qwen2_5_VLConfig",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGenerationDecode",
    "Qwen2_5_VLForConditionalGenerationDecodeStaticCache",
    "Qwen2_5_VLForConditionalGenerationImagePrefillWithCache",
    "Qwen2_5_VLTextConfig",
    "Qwen2_5_VLTextModel",
    "Qwen2_5_VLVisionConfig",
    "Qwen2_5_VLVisionModel",
    "Qwen2_5_VLVisionPatchEmbedLinear",
    "qwen2_5_vl_config_from_transformers_config",
    "qwen2_5_vl_config_from_transformers_dict",
    "qwen2_5_vl_patch_embed_linear_weight",
    "qwen2_5_vl_prepare_inputs_for_generation",
    "qwen2_5_vl_required_text_weight_names",
    "qwen2_5_vl_required_weight_names",
    "qwen2_5_vl_rope_index",
    "qwen2_5_vl_stitch_image_features",
    "qwen2_5_vl_text_inv_freq",
    "qwen2_5_vl_text_rope_embeddings",
    "qwen2_5_vl_vision_cu_seqlens",
    "qwen2_5_vl_vision_position_ids",
    "qwen2_5_vl_vision_rope_embeddings",
    "qwen2_5_vl_vision_window_index",
    "qwen2_5_vl_weights_from_safetensors_file",
    "qwen2_5_vl_weights_from_safetensors_index",
    "qwen2_5_vl_weights_from_transformers_state_dict",
]
