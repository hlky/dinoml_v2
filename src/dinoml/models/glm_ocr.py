from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from itertools import groupby
from typing import Any, Mapping, Sequence

import numpy as np

import dinoml as dml
from dinoml.ir import array_to_storage, dtype_numpy, normalize_dtype
from dinoml.models.kv_cache import append_static_kv_cache


_GLM_OCR_FLOAT_DTYPES = frozenset({"float16", "float32", "bfloat16"})


def _normalize_glm_ocr_dtype(dtype: str) -> str:
    normalized = normalize_dtype(dtype)
    if normalized not in _GLM_OCR_FLOAT_DTYPES:
        raise ValueError(f"GLM-OCR currently supports float16/float32/bfloat16 parameters, got {dtype!r}")
    return normalized


def _numpy_glm_ocr_dtype(dtype: str) -> np.dtype:
    dtype = _normalize_glm_ocr_dtype(dtype)
    return np.dtype(np.uint16) if dtype == "bfloat16" else dtype_numpy(dtype)


def _glm_ocr_float_storage(values: np.ndarray, dtype: str) -> np.ndarray:
    dtype = _normalize_glm_ocr_dtype(dtype)
    if dtype == "bfloat16":
        return array_to_storage(values.astype(np.float32, copy=False), "bfloat16")
    return values.astype(dtype_numpy(dtype), copy=False)


@dataclass(frozen=True)
class GlmOcrVisionConfig:
    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "silu"
    attention_bias: bool = True
    attention_dropout: float = 0.0
    num_heads: int = 16
    in_channels: int = 3
    image_size: int | tuple[int, int] = 336
    patch_size: int = 14
    rms_norm_eps: float = 1.0e-5
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 1536
    intermediate_size: int = 4096
    initializer_range: float = 0.02
    dtype: str = "bfloat16"
    use_flash_attention: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _normalize_glm_ocr_dtype(self.dtype))
        _require_positive(self.depth, "vision depth", allow_zero=True)
        _require_positive(self.hidden_size, "vision hidden_size")
        _require_positive(self.intermediate_size, "vision intermediate_size")
        _require_positive(self.out_hidden_size, "vision out_hidden_size")
        _require_positive(self.num_heads, "vision num_heads")
        _require_positive(self.in_channels, "vision in_channels")
        _require_positive(self.patch_size, "vision patch_size")
        _require_positive(self.temporal_patch_size, "vision temporal_patch_size")
        _require_positive(self.spatial_merge_size, "vision spatial_merge_size")
        if self.hidden_act != "silu":
            raise ValueError(f"GLM-OCR vision only supports hidden_act='silu', got {self.hidden_act!r}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("vision hidden_size must be divisible by num_heads")
        if self.attention_dropout != 0:
            raise ValueError("GLM-OCR inference expects attention_dropout=0")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size


@dataclass(frozen=True)
class GlmOcrTextConfig:
    vocab_size: int = 59392
    hidden_size: int = 1536
    intermediate_size: int = 4608
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1.0e-5
    use_cache: bool = True
    attention_dropout: float = 0.0
    rope_parameters: Mapping[str, Any] = field(
        default_factory=lambda: {
            "rope_type": "default",
            "mrope_section": [16, 24, 24],
            "partial_rotary_factor": 1.0,
            "rope_theta": 10000.0,
        }
    )
    pad_token_id: int | None = None
    tie_word_embeddings: bool = False
    dtype: str = "bfloat16"
    mask_fill_value: float = -1.0e4
    use_flash_attention: bool = True
    use_flash_attention_bias: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _normalize_glm_ocr_dtype(self.dtype))
        _require_positive(self.vocab_size, "text vocab_size")
        _require_positive(self.hidden_size, "text hidden_size")
        _require_positive(self.intermediate_size, "text intermediate_size")
        _require_positive(self.num_hidden_layers, "text num_hidden_layers", allow_zero=True)
        _require_positive(self.num_attention_heads, "text num_attention_heads")
        _require_positive(self.num_key_value_heads, "text num_key_value_heads")
        _require_positive(self.head_dim, "text head_dim")
        if self.hidden_act != "silu":
            raise ValueError(f"GLM-OCR text only supports hidden_act='silu', got {self.hidden_act!r}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.attention_dropout != 0:
            raise ValueError("GLM-OCR inference expects attention_dropout=0")
        rope = dict(self.rope_parameters)
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"GLM-OCR currently supports default RoPE only, got {rope.get('rope_type')!r}")
        mrope_section = tuple(int(v) for v in rope.get("mrope_section", (16, 24, 24)))
        if len(mrope_section) != 3 or any(v <= 0 for v in mrope_section):
            raise ValueError("rope_parameters.mrope_section must contain three positive integers")
        if sum(mrope_section) != self.rotary_freq_dim:
            raise ValueError(
                "sum(rope_parameters.mrope_section) must equal head_dim * partial_rotary_factor / 2"
            )
        rope["mrope_section"] = list(mrope_section)
        rope.setdefault("partial_rotary_factor", 1.0)
        rope.setdefault("rope_theta", 10000.0)
        object.__setattr__(self, "rope_parameters", rope)

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
        return int(self.head_dim * float(self.rope_parameters.get("partial_rotary_factor", 1.0)))

    @property
    def rotary_freq_dim(self) -> int:
        return self.rotary_dim // 2


@dataclass(frozen=True)
class GlmOcrConfig:
    text_config: GlmOcrTextConfig = field(default_factory=GlmOcrTextConfig)
    vision_config: GlmOcrVisionConfig = field(default_factory=GlmOcrVisionConfig)
    image_token_id: int = 59280
    video_token_id: int = 59281
    image_start_token_id: int = 59256
    image_end_token_id: int = 59257
    video_start_token_id: int = 59258
    video_end_token_id: int = 59259
    tie_word_embeddings: bool = False


def glm_ocr_config_from_transformers_dict(payload: Mapping[str, Any], *, dtype: str | None = None) -> GlmOcrConfig:
    if str(payload.get("model_type")) != "glm_ocr":
        raise ValueError(f"expected model_type='glm_ocr', got {payload.get('model_type')!r}")
    text_payload = dict(payload.get("text_config") or {})
    vision_payload = dict(payload.get("vision_config") or {})
    requested_dtype = dtype or text_payload.get("dtype") or payload.get("dtype") or "bfloat16"
    text_payload["dtype"] = requested_dtype
    vision_payload["dtype"] = requested_dtype
    text_payload.pop("model_type", None)
    vision_payload.pop("model_type", None)
    vision_payload.pop("hidden_dropout_prob", None)
    text_payload.pop("eos_token_id", None)
    text_payload.pop("attention_bias", None)
    text_payload.pop("num_nextn_predict_layers", None)
    text_config = GlmOcrTextConfig(**text_payload)
    vision_config = GlmOcrVisionConfig(**vision_payload)
    return GlmOcrConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=int(payload.get("image_token_id", 59280)),
        video_token_id=int(payload.get("video_token_id", 59281)),
        image_start_token_id=int(payload.get("image_start_token_id", 59256)),
        image_end_token_id=int(payload.get("image_end_token_id", 59257)),
        video_start_token_id=int(payload.get("video_start_token_id", 59258)),
        video_end_token_id=int(payload.get("video_end_token_id", 59259)),
        tie_word_embeddings=bool(payload.get("tie_word_embeddings", False)),
    )


def glm_ocr_config_from_transformers_config(config: object, *, dtype: str | None = None) -> GlmOcrConfig:
    if isinstance(config, Mapping):
        return glm_ocr_config_from_transformers_dict(config, dtype=dtype)
    payload = {
        "model_type": getattr(config, "model_type", "glm_ocr"),
        "text_config": _object_public_attrs(getattr(config, "text_config", None)),
        "vision_config": _object_public_attrs(getattr(config, "vision_config", None)),
        "image_token_id": getattr(config, "image_token_id", 59280),
        "video_token_id": getattr(config, "video_token_id", 59281),
        "image_start_token_id": getattr(config, "image_start_token_id", 59256),
        "image_end_token_id": getattr(config, "image_end_token_id", 59257),
        "video_start_token_id": getattr(config, "video_start_token_id", 59258),
        "video_end_token_id": getattr(config, "video_end_token_id", 59259),
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", False),
    }
    return glm_ocr_config_from_transformers_dict(payload, dtype=dtype)


def glm_ocr_patch_embed_linear_weight(conv3d_weight: object, *, dtype: str = "float32") -> np.ndarray:
    weight = _state_value_to_numpy(conv3d_weight, dtype=dtype)
    if weight.ndim != 5:
        raise ValueError(f"patch embedding Conv3d weight must be rank 5, got shape {weight.shape}")
    return np.reshape(weight, (weight.shape[0], int(np.prod(weight.shape[1:]))))


def glm_ocr_text_inv_freq(config: GlmOcrTextConfig, *, dtype: str = "float32") -> np.ndarray:
    base = float(config.rope_parameters["rope_theta"])
    dim = config.rotary_dim
    values = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / float(dim)))
    return _glm_ocr_float_storage(values, dtype)


def glm_ocr_text_rope_embeddings(
    position_ids: object,
    config: GlmOcrTextConfig,
    *,
    dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(position_ids, dtype=np.float32)
    if pos.ndim != 3 or pos.shape[0] != 3:
        raise ValueError(f"position_ids must have shape [3, batch, seq], got {pos.shape}")
    inv_freq = glm_ocr_text_inv_freq(config, dtype="float32")
    freqs = np.einsum("f,tbs->tbsf", inv_freq, pos, dtype=np.float32)
    chunks = np.split(freqs, np.cumsum(config.rope_parameters["mrope_section"])[:-1], axis=-1)
    freqs = np.concatenate([chunk[idx % 3] for idx, chunk in enumerate(chunks)], axis=-1)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return _glm_ocr_float_storage(np.cos(emb), dtype), _glm_ocr_float_storage(np.sin(emb), dtype)


def glm_ocr_vision_inv_freq(head_dim: int, *, theta: float = 10000.0, dtype: str = "float32") -> np.ndarray:
    dim = head_dim // 2
    values = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / float(dim)))
    return _glm_ocr_float_storage(values, dtype)


def glm_ocr_vision_rope_embeddings(
    position_ids: object,
    *,
    head_dim: int,
    theta: float = 10000.0,
    dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(position_ids, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"vision position_ids must have shape [seq, 2], got {pos.shape}")
    inv_freq = glm_ocr_vision_inv_freq(head_dim, theta=theta, dtype="float32")
    freqs = (pos[..., None] * inv_freq).reshape(pos.shape[0], -1)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return _glm_ocr_float_storage(np.cos(emb), dtype), _glm_ocr_float_storage(np.sin(emb), dtype)


def glm_ocr_vision_position_ids(grid_thw: object, spatial_merge_size: int) -> np.ndarray:
    grids = np.asarray(grid_thw, dtype=np.int64)
    if grids.ndim != 2 or grids.shape[1] != 3:
        raise ValueError(f"grid_thw must have shape [num_items, 3], got {grids.shape}")
    rows: list[np.ndarray] = []
    for grid_t, grid_h, grid_w in grids:
        if grid_h % spatial_merge_size != 0 or grid_w % spatial_merge_size != 0:
            raise ValueError("grid_thw height/width must be divisible by spatial_merge_size")
        hpos = np.arange(grid_h, dtype=np.int64).reshape(grid_h, 1)
        hpos = np.repeat(hpos, grid_w, axis=1)
        hpos = hpos.reshape(grid_h // spatial_merge_size, spatial_merge_size, grid_w // spatial_merge_size, spatial_merge_size)
        hpos = np.transpose(hpos, (0, 2, 1, 3)).reshape(-1)
        wpos = np.arange(grid_w, dtype=np.int64).reshape(1, grid_w)
        wpos = np.repeat(wpos, grid_h, axis=0)
        wpos = wpos.reshape(grid_h // spatial_merge_size, spatial_merge_size, grid_w // spatial_merge_size, spatial_merge_size)
        wpos = np.transpose(wpos, (0, 2, 1, 3)).reshape(-1)
        merged = np.stack([hpos, wpos], axis=-1)
        rows.extend([merged] * int(grid_t))
    return np.concatenate(rows, axis=0) if rows else np.empty((0, 2), dtype=np.int64)


def glm_ocr_rope_index(
    input_ids: object,
    mm_token_type_ids: object,
    *,
    image_grid_thw: object | None = None,
    video_grid_thw: object | None = None,
    attention_mask: object | None = None,
    spatial_merge_size: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(input_ids)
    token_types = np.asarray(mm_token_type_ids)
    if ids.ndim != 2 or token_types.shape != ids.shape:
        raise ValueError("input_ids and mm_token_type_ids must have matching [batch, seq] shapes")
    image_iter = iter(np.asarray(image_grid_thw, dtype=np.int64)) if image_grid_thw is not None else None
    video_grids = None
    if video_grid_thw is not None:
        video_base = np.asarray(video_grid_thw, dtype=np.int64)
        video_grids = np.repeat(video_base, video_base[:, 0], axis=0)
        video_grids[:, 0] = 1
    video_iter = iter(video_grids) if video_grids is not None else None
    masks = None if attention_mask is None else np.asarray(attention_mask).astype(bool)
    position_ids = np.zeros((3, ids.shape[0], ids.shape[1]), dtype=ids.dtype)
    deltas: list[int] = []
    for batch_idx in range(ids.shape[0]):
        current_ids = ids[batch_idx]
        current_types = token_types[batch_idx]
        current_mask = None if masks is None else masks[batch_idx]
        if current_mask is not None:
            current_ids = current_ids[current_mask]
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
            else:
                source_iter = image_iter if modality_type == 1 else video_iter
                if source_iter is None:
                    raise ValueError(f"missing grid_thw for modality token type {modality_type}")
                grid = next(source_iter)
                vision_pos = _single_vision_rope_index(
                    current_pos,
                    grid,
                    spatial_merge_size=spatial_merge_size,
                    dtype=ids.dtype,
                )
                pieces.append(vision_pos)
                current_pos += int(max(grid[1], grid[2]) // spatial_merge_size)
        llm_positions = np.concatenate(pieces, axis=1) if pieces else np.empty((3, 0), dtype=ids.dtype)
        if current_mask is not None:
            position_ids[:, batch_idx, current_mask] = llm_positions
        else:
            position_ids[:, batch_idx, :] = llm_positions
        deltas.append(int(llm_positions.max() + 1 - len(current_ids)) if llm_positions.size else 0)
    return position_ids, np.asarray(deltas, dtype=ids.dtype).reshape(-1, 1)


def glm_ocr_stitch_image_features(
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


def glm_ocr_prepare_inputs_for_generation(
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


class GlmOcrTextAttention(dml.nn.Module):
    def __init__(self, config: GlmOcrTextConfig, weights: Mapping[str, np.ndarray], prefix: str):
        self.config = config
        self.q_proj = _loaded_linear(
            weights,
            parameter_prefix="q_proj",
            weight_key=f"{prefix}.self_attn.q_proj.weight",
            in_features=config.hidden_size,
            out_features=config.q_proj_size,
            dtype=config.dtype,
        )
        self.k_proj = _loaded_linear(
            weights,
            parameter_prefix="k_proj",
            weight_key=f"{prefix}.self_attn.k_proj.weight",
            in_features=config.hidden_size,
            out_features=config.kv_proj_size,
            dtype=config.dtype,
        )
        self.v_proj = _loaded_linear(
            weights,
            parameter_prefix="v_proj",
            weight_key=f"{prefix}.self_attn.v_proj.weight",
            in_features=config.hidden_size,
            out_features=config.kv_proj_size,
            dtype=config.dtype,
        )
        self.o_proj = _loaded_linear(
            weights,
            parameter_prefix="o_proj",
            weight_key=f"{prefix}.self_attn.o_proj.weight",
            in_features=config.q_proj_size,
            out_features=config.hidden_size,
            dtype=config.dtype,
        )

    def _project_qkv(self, hidden_states, cos, sin):
        batch, seq_len, _ = _rank3_shape(hidden_states.shape, "GLM-OCR text hidden_states")
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = dml.ops.reshape(q, [batch, seq_len, self.config.num_attention_heads, self.config.head_dim])
        k = dml.ops.reshape(k, [batch, seq_len, self.config.num_key_value_heads, self.config.head_dim])
        v = dml.ops.reshape(v, [batch, seq_len, self.config.num_key_value_heads, self.config.head_dim])
        q, k = apply_glm_ocr_text_rope(q, k, cos, sin, self.config.rotary_dim)
        return q, k, v

    def forward(self, hidden_states, cos, sin, attention_mask=None):
        batch, seq_len, _ = _rank3_shape(hidden_states.shape, "GLM-OCR text hidden_states")
        q, k, v = self._project_qkv(hidden_states, cos, sin)
        return self._attention_output(q, k, v, attention_mask, batch, seq_len)

    def prefill_with_cache(self, hidden_states, cos, sin, attention_mask=None):
        batch, seq_len, _ = _rank3_shape(hidden_states.shape, "GLM-OCR text hidden_states")
        q, k, v = self._project_qkv(hidden_states, cos, sin)
        present_key = dml.ops.permute0213(k)
        present_value = dml.ops.permute0213(v)
        return self._attention_output(q, k, v, attention_mask, batch, seq_len), present_key, present_value

    def _attention_output(self, q, k, v, attention_mask, batch: int, seq_len: int):
        if self.config.use_flash_attention and q.dtype in {"float16", "bfloat16"}:
            if attention_mask is None:
                context = dml.ops.flash_attention(q, k, v, causal=True)
                context = dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size])
                return self.o_proj(context)
            if self.config.use_flash_attention_bias:
                context = dml.ops.flash_attention_bias(q, k, v, attention_mask, causal=True)
                context = dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size])
                return self.o_proj(context)
        q = dml.ops.permute(q, (0, 2, 1, 3))
        k = dml.ops.permute(k, (0, 2, 1, 3))
        v = dml.ops.permute(v, (0, 2, 1, 3))
        k = _repeat_kv_heads(k, self.config.num_key_value_groups)
        v = _repeat_kv_heads(v, self.config.num_key_value_groups)
        k = _materialize(k)
        v = _materialize(v)
        q = dml.ops.reshape(q, [batch * self.config.num_attention_heads, seq_len, self.config.head_dim])
        k = dml.ops.reshape(k, [batch * self.config.num_attention_heads, seq_len, self.config.head_dim])
        v = dml.ops.reshape(v, [batch * self.config.num_attention_heads, seq_len, self.config.head_dim])
        scores = dml.ops.mul(dml.ops.bmm_rcr(q, k), 1.0 / math.sqrt(self.config.head_dim))
        if attention_mask is not None:
            scores = dml.ops.add(scores, attention_mask)
        probs = dml.ops.softmax(scores, dim=-1)
        context = dml.ops.bmm_rrr(probs, v)
        context = dml.ops.reshape(context, [batch, self.config.num_attention_heads, seq_len, self.config.head_dim])
        context = dml.ops.permute(context, (0, 2, 1, 3))
        context = dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size])
        return self.o_proj(context)

    def forward_with_cache(self, hidden_states, cos, sin, past_key, past_value, attention_mask=None):
        batch, seq_len, _ = _rank3_shape(hidden_states.shape, "GLM-OCR text hidden_states")
        q, k, v = self._project_qkv(hidden_states, cos, sin)
        q = dml.ops.permute(q, (0, 2, 1, 3))
        k = dml.ops.permute(k, (0, 2, 1, 3))
        v = dml.ops.permute(v, (0, 2, 1, 3))
        present_key = dml.ops.concatenate([past_key, k], dim=2)
        present_value = dml.ops.concatenate([past_value, v], dim=2)
        if (
            self.config.use_flash_attention
            and q.dtype in {"float16", "bfloat16"}
            and attention_mask is None
            and seq_len == 1
        ):
            attn_key = dml.ops.permute(present_key, (0, 2, 1, 3))
            attn_value = dml.ops.permute(present_value, (0, 2, 1, 3))
            q = dml.ops.permute(q, (0, 2, 1, 3))
            context = dml.ops.flash_attention(q, attn_key, attn_value, causal=False)
            context = dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size])
            return self.o_proj(context), present_key, present_value
        if (
            self.config.use_flash_attention
            and self.config.use_flash_attention_bias
            and q.dtype in {"float16", "bfloat16"}
            and attention_mask is not None
            and seq_len == 1
        ):
            attn_key = dml.ops.permute(present_key, (0, 2, 1, 3))
            attn_value = dml.ops.permute(present_value, (0, 2, 1, 3))
            q = dml.ops.permute(q, (0, 2, 1, 3))
            context = dml.ops.flash_attention_bias(q, attn_key, attn_value, attention_mask, causal=False)
            context = dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size])
            return self.o_proj(context), present_key, present_value
        attn_key = _repeat_kv_heads(present_key, self.config.num_key_value_groups)
        attn_value = _repeat_kv_heads(present_value, self.config.num_key_value_groups)
        attn_key = _materialize(attn_key)
        attn_value = _materialize(attn_value)
        total_len = int(attn_key.shape[2])
        q = dml.ops.reshape(q, [batch * self.config.num_attention_heads, seq_len, self.config.head_dim])
        attn_key = dml.ops.reshape(attn_key, [batch * self.config.num_attention_heads, total_len, self.config.head_dim])
        attn_value = dml.ops.reshape(attn_value, [batch * self.config.num_attention_heads, total_len, self.config.head_dim])
        scores = dml.ops.mul(dml.ops.bmm_rcr(q, attn_key), 1.0 / math.sqrt(self.config.head_dim))
        if attention_mask is not None:
            scores = dml.ops.add(scores, attention_mask)
        probs = dml.ops.softmax(scores, dim=-1)
        context = dml.ops.bmm_rrr(probs, attn_value)
        context = dml.ops.reshape(context, [batch, self.config.num_attention_heads, seq_len, self.config.head_dim])
        context = dml.ops.permute(context, (0, 2, 1, 3))
        context = dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size])
        return self.o_proj(context), present_key, present_value

    def forward_with_static_cache(self, hidden_states, cos, sin, past_key, past_value, attention_mask=None, cache_seqlens=None):
        batch, seq_len, _ = _rank3_shape(hidden_states.shape, "GLM-OCR text hidden_states")
        q, k, v = self._project_qkv(hidden_states, cos, sin)
        new_key = dml.ops.permute(k, (0, 2, 1, 3))
        new_value = dml.ops.permute(v, (0, 2, 1, 3))
        if self.config.use_flash_attention and q.dtype in {"float16", "bfloat16"} and cache_seqlens is not None:
            if attention_mask is None:
                context = dml.ops.flash_attention_static_kv_cache(
                    q,
                    past_key,
                    past_value,
                    new_key,
                    new_value,
                    cache_seqlens,
                )
                context = dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size])
                return self.o_proj(context), new_key, new_value
            if self.config.use_flash_attention_bias:
                context = dml.ops.flash_attention_static_kv_cache_bias(
                    q,
                    past_key,
                    past_value,
                    new_key,
                    new_value,
                    cache_seqlens,
                    attention_mask,
                )
                context = dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size])
                return self.o_proj(context), new_key, new_value
        q = dml.ops.permute(q, (0, 2, 1, 3))
        attn_key, attn_value = append_static_kv_cache(past_key, past_value, new_key, new_value)
        attn_key = _repeat_kv_heads(attn_key, self.config.num_key_value_groups)
        attn_value = _repeat_kv_heads(attn_value, self.config.num_key_value_groups)
        attn_key = _materialize(attn_key)
        attn_value = _materialize(attn_value)
        total_len = int(attn_key.shape[2])
        q = dml.ops.reshape(q, [batch * self.config.num_attention_heads, seq_len, self.config.head_dim])
        attn_key = dml.ops.reshape(attn_key, [batch * self.config.num_attention_heads, total_len, self.config.head_dim])
        attn_value = dml.ops.reshape(attn_value, [batch * self.config.num_attention_heads, total_len, self.config.head_dim])
        scores = dml.ops.mul(dml.ops.bmm_rcr(q, attn_key), 1.0 / math.sqrt(self.config.head_dim))
        if attention_mask is not None:
            scores = dml.ops.add(scores, attention_mask)
        probs = dml.ops.softmax(scores, dim=-1)
        context = dml.ops.bmm_rrr(probs, attn_value)
        context = dml.ops.reshape(context, [batch, self.config.num_attention_heads, seq_len, self.config.head_dim])
        context = dml.ops.permute(context, (0, 2, 1, 3))
        context = dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size])
        return self.o_proj(context), new_key, new_value

    def forward_with_session_static_cache(
        self,
        hidden_states,
        cos,
        sin,
        past_key,
        past_value,
        attention_mask=None,
        cache_seqlens=None,
    ):
        batch, seq_len, _ = _rank3_shape(hidden_states.shape, "GLM-OCR text hidden_states")
        q, k, v = self._project_qkv(hidden_states, cos, sin)
        new_key = dml.ops.permute(k, (0, 2, 1, 3))
        new_value = dml.ops.permute(v, (0, 2, 1, 3))
        if not (self.config.use_flash_attention and q.dtype in {"float16", "bfloat16"} and cache_seqlens is not None):
            raise ValueError("Session static KV cache decode requires float16/bfloat16 flash attention and cache_seqlens")
        if attention_mask is None:
            context = dml.ops.flash_attention_static_kv_cache(
                q,
                past_key,
                past_value,
                new_key,
                new_value,
                cache_seqlens,
            )
        elif self.config.use_flash_attention_bias:
            context = dml.ops.flash_attention_static_kv_cache_bias(
                q,
                past_key,
                past_value,
                new_key,
                new_value,
                cache_seqlens,
                attention_mask,
            )
        else:
            raise ValueError("Session static KV cache decode with a mask requires use_flash_attention_bias=True")
        context = dml.ops.reshape(context, [batch, seq_len, self.config.q_proj_size])
        return self.o_proj(context)


class GlmOcrTextMLP(dml.nn.Module):
    def __init__(self, config: GlmOcrTextConfig, weights: Mapping[str, np.ndarray], prefix: str):
        self.config = config
        self.gate_proj = _loaded_linear_row_slice(
            weights,
            parameter_prefix="gate_proj",
            weight_key=f"{prefix}.mlp.gate_up_proj.weight",
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            start_row=0,
            dtype=config.dtype,
        )
        self.up_proj = _loaded_linear_row_slice(
            weights,
            parameter_prefix="up_proj",
            weight_key=f"{prefix}.mlp.gate_up_proj.weight",
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            start_row=config.intermediate_size,
            dtype=config.dtype,
        )
        self.down_proj = _loaded_linear(
            weights,
            parameter_prefix="down_proj",
            weight_key=f"{prefix}.mlp.down_proj.weight",
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            dtype=config.dtype,
        )

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(dml.ops.mul(up, dml.ops.silu(gate)))


class GlmOcrTextDecoderLayer(dml.nn.Module):
    def __init__(self, config: GlmOcrTextConfig, weights: Mapping[str, np.ndarray], layer_idx: int):
        prefix = f"model.language_model.layers.{layer_idx}"
        self.input_layernorm = _loaded_rms_norm(
            weights,
            parameter_prefix="input_layernorm",
            weight_key=f"{prefix}.input_layernorm.weight",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )
        self.self_attn = GlmOcrTextAttention(config, weights, prefix)
        self.post_self_attn_layernorm = _loaded_rms_norm(
            weights,
            parameter_prefix="post_self_attn_layernorm",
            weight_key=f"{prefix}.post_self_attn_layernorm.weight",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )
        self.post_attention_layernorm = _loaded_rms_norm(
            weights,
            parameter_prefix="post_attention_layernorm",
            weight_key=f"{prefix}.post_attention_layernorm.weight",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )
        self.mlp = GlmOcrTextMLP(config, weights, prefix)
        self.post_mlp_layernorm = _loaded_rms_norm(
            weights,
            parameter_prefix="post_mlp_layernorm",
            weight_key=f"{prefix}.post_mlp_layernorm.weight",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )

    def forward(self, hidden_states, cos, sin, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = dml.ops.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        return dml.ops.add(residual, hidden_states)

    def forward_with_cache(self, hidden_states, cos, sin, past_key, past_value, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key, present_value = self.self_attn.forward_with_cache(
            hidden_states,
            cos,
            sin,
            past_key,
            past_value,
            attention_mask,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = dml.ops.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = dml.ops.add(residual, hidden_states)
        return hidden_states, present_key, present_value

    def forward_with_static_cache(self, hidden_states, cos, sin, past_key, past_value, attention_mask=None, cache_seqlens=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_key, new_value = self.self_attn.forward_with_static_cache(
            hidden_states,
            cos,
            sin,
            past_key,
            past_value,
            attention_mask,
            cache_seqlens,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = dml.ops.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = dml.ops.add(residual, hidden_states)
        return hidden_states, new_key, new_value

    def forward_with_session_static_cache(
        self,
        hidden_states,
        cos,
        sin,
        past_key,
        past_value,
        attention_mask=None,
        cache_seqlens=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_with_session_static_cache(
            hidden_states,
            cos,
            sin,
            past_key,
            past_value,
            attention_mask,
            cache_seqlens,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = dml.ops.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        return dml.ops.add(residual, hidden_states)

    def prefill_with_cache(self, hidden_states, cos, sin, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key, present_value = self.self_attn.prefill_with_cache(
            hidden_states,
            cos,
            sin,
            attention_mask,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = dml.ops.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = dml.ops.add(residual, hidden_states)
        return hidden_states, present_key, present_value


class GlmOcrTextModel(dml.nn.Module):
    def __init__(self, config: GlmOcrTextConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.embed_tokens = _loaded_embedding(
            weights,
            parameter_prefix="embed_tokens",
            weight_key="model.language_model.embed_tokens.weight",
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
        )
        self.layers = dml.nn.ModuleList(
            GlmOcrTextDecoderLayer(config, weights, idx) for idx in range(config.num_hidden_layers)
        )
        self.norm = _loaded_rms_norm(
            weights,
            parameter_prefix="norm",
            weight_key="model.language_model.norm.weight",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )

    def encode_inputs_embeds(self, inputs_embeds, cos, sin, attention_mask=None):
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attention_mask)
        return self.norm(hidden_states)

    def encode_inputs_embeds_with_cache(self, inputs_embeds, cos, sin, attention_mask=None):
        hidden_states = inputs_embeds
        present_key_values: list[tuple[Any, Any]] = []
        for layer in self.layers:
            hidden_states, present_key, present_value = layer.prefill_with_cache(hidden_states, cos, sin, attention_mask)
            present_key_values.append((present_key, present_value))
        return self.norm(hidden_states), present_key_values

    def encode(self, input_ids, cos, sin, attention_mask=None):
        return self.encode_inputs_embeds(self.embed_tokens(input_ids), cos, sin, attention_mask)

    def forward(self, input_ids, cos, sin, attention_mask=None):
        return dml.ops.output(self.encode(input_ids, cos, sin, attention_mask), "last_hidden_state")

    def decode(self, input_ids, cos, sin, attention_mask, past_key_values: Mapping[int, tuple[Any, Any]]):
        hidden_states = self.embed_tokens(input_ids)
        present_key_values: list[tuple[Any, Any]] = []
        for layer_idx, layer in enumerate(self.layers):
            past_key, past_value = past_key_values[layer_idx]
            hidden_states, present_key, present_value = layer.forward_with_cache(
                hidden_states,
                cos,
                sin,
                past_key,
                past_value,
                attention_mask,
            )
            present_key_values.append((present_key, present_value))
        return self.norm(hidden_states), present_key_values

    def decode_static_cache(
        self,
        input_ids,
        cos,
        sin,
        attention_mask,
        past_key_values: Mapping[int, tuple[Any, Any]],
        cache_seqlens=None,
    ):
        hidden_states = self.embed_tokens(input_ids)
        new_key_values: list[tuple[Any, Any]] = []
        for layer_idx, layer in enumerate(self.layers):
            past_key, past_value = past_key_values[layer_idx]
            hidden_states, new_key, new_value = layer.forward_with_static_cache(
                hidden_states,
                cos,
                sin,
                past_key,
                past_value,
                attention_mask,
                cache_seqlens,
            )
            new_key_values.append((new_key, new_value))
        return self.norm(hidden_states), new_key_values

    def decode_session_static_cache(
        self,
        input_ids,
        cos,
        sin,
        attention_mask,
        *,
        max_cache_len: int,
        cache_seqlens=None,
    ):
        batch, _ = _rank2_shape(input_ids.shape, "GLM-OCR decode input_ids")
        hidden_states = self.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            past_key = dml.state(
                f"past_key_{layer_idx}",
                dml.TensorSpec(
                    [batch, self.config.num_key_value_heads, int(max_cache_len), self.config.head_dim],
                    self.config.dtype,
                ),
            )
            past_value = dml.state(
                f"past_value_{layer_idx}",
                dml.TensorSpec(
                    [batch, self.config.num_key_value_heads, int(max_cache_len), self.config.head_dim],
                    self.config.dtype,
                ),
            )
            hidden_states = layer.forward_with_session_static_cache(
                hidden_states,
                cos,
                sin,
                past_key,
                past_value,
                attention_mask,
                cache_seqlens,
            )
        return self.norm(hidden_states)


class GlmOcrVisionMlp(dml.nn.Module):
    def __init__(self, config: GlmOcrVisionConfig, weights: Mapping[str, np.ndarray], prefix: str):
        bias_key = f"{prefix}.mlp.gate_proj.bias" if config.attention_bias else None
        self.gate_proj = _loaded_linear(
            weights,
            parameter_prefix="gate_proj",
            weight_key=f"{prefix}.mlp.gate_proj.weight",
            bias_key=bias_key,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            dtype=config.dtype,
        )
        self.up_proj = _loaded_linear(
            weights,
            parameter_prefix="up_proj",
            weight_key=f"{prefix}.mlp.up_proj.weight",
            bias_key=f"{prefix}.mlp.up_proj.bias" if config.attention_bias else None,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            dtype=config.dtype,
        )
        self.down_proj = _loaded_linear(
            weights,
            parameter_prefix="down_proj",
            weight_key=f"{prefix}.mlp.down_proj.weight",
            bias_key=f"{prefix}.mlp.down_proj.bias" if config.attention_bias else None,
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            dtype=config.dtype,
        )

    def forward(self, hidden_state):
        return self.down_proj(dml.ops.mul(dml.ops.silu(self.gate_proj(hidden_state)), self.up_proj(hidden_state)))


class GlmOcrVisionAttention(dml.nn.Module):
    def __init__(self, config: GlmOcrVisionConfig, weights: Mapping[str, np.ndarray], prefix: str):
        self.config = config
        self.qkv = _loaded_linear(
            weights,
            parameter_prefix="qkv",
            weight_key=f"{prefix}.attn.qkv.weight",
            bias_key=f"{prefix}.attn.qkv.bias" if config.attention_bias else None,
            in_features=config.hidden_size,
            out_features=config.hidden_size * 3,
            dtype=config.dtype,
        )
        self.proj = _loaded_linear(
            weights,
            parameter_prefix="proj",
            weight_key=f"{prefix}.attn.proj.weight",
            bias_key=f"{prefix}.attn.proj.bias" if config.attention_bias else None,
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            dtype=config.dtype,
        )
        self.q_norm = _loaded_rms_norm(
            weights,
            parameter_prefix="q_norm",
            weight_key=f"{prefix}.attn.q_norm.weight",
            hidden_size=config.head_dim,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )
        self.k_norm = _loaded_rms_norm(
            weights,
            parameter_prefix="k_norm",
            weight_key=f"{prefix}.attn.k_norm.weight",
            hidden_size=config.head_dim,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )

    def forward(self, hidden_states, cos, sin):
        seq_len, _ = _rank2_shape(hidden_states.shape, "GLM-OCR vision hidden_states")
        qkv = self.qkv(hidden_states)
        q, k, v = dml.ops.qkv_split(qkv)
        q = dml.ops.reshape(q, [seq_len, self.config.num_heads, self.config.head_dim])
        k = dml.ops.reshape(k, [seq_len, self.config.num_heads, self.config.head_dim])
        v = dml.ops.reshape(v, [seq_len, self.config.num_heads, self.config.head_dim])
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_glm_ocr_vision_rope(q, k, cos, sin)
        q = _materialize(q)
        k = _materialize(k)
        v = _materialize(v)
        if self.config.use_flash_attention and self.config.dtype in {"float16", "bfloat16"}:
            q4 = dml.ops.reshape(q, [1, seq_len, self.config.num_heads, self.config.head_dim])
            k4 = dml.ops.reshape(k, [1, seq_len, self.config.num_heads, self.config.head_dim])
            v4 = dml.ops.reshape(v, [1, seq_len, self.config.num_heads, self.config.head_dim])
            out = dml.ops.flash_attention(q4, k4, v4, causal=False)
            out = dml.ops.reshape(out, [seq_len, self.config.hidden_size])
            return self.proj(out)
        q = dml.ops.permute(q, (1, 0, 2))
        k = dml.ops.permute(k, (1, 0, 2))
        v = dml.ops.permute(v, (1, 0, 2))
        scores = dml.ops.mul(dml.ops.bmm_rcr(q, k), 1.0 / math.sqrt(self.config.head_dim))
        probs = dml.ops.softmax(scores, dim=-1)
        out = dml.ops.bmm_rrr(probs, v)
        out = dml.ops.permute(out, (1, 0, 2))
        out = dml.ops.reshape(out, [seq_len, self.config.hidden_size])
        return self.proj(out)


class GlmOcrVisionBlock(dml.nn.Module):
    def __init__(self, config: GlmOcrVisionConfig, weights: Mapping[str, np.ndarray], layer_idx: int):
        prefix = f"model.visual.blocks.{layer_idx}"
        self.norm1 = _loaded_rms_norm(
            weights,
            parameter_prefix="norm1",
            weight_key=f"{prefix}.norm1.weight",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )
        self.attn = GlmOcrVisionAttention(config, weights, prefix)
        self.norm2 = _loaded_rms_norm(
            weights,
            parameter_prefix="norm2",
            weight_key=f"{prefix}.norm2.weight",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )
        self.mlp = GlmOcrVisionMlp(config, weights, prefix)

    def forward(self, hidden_states, cos, sin):
        hidden_states = dml.ops.add(hidden_states, self.attn(self.norm1(hidden_states), cos, sin))
        return dml.ops.add(hidden_states, self.mlp(self.norm2(hidden_states)))


class GlmOcrVisionPatchEmbedLinear(dml.nn.Module):
    def __init__(self, config: GlmOcrVisionConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.proj = _loaded_linear(
            weights,
            parameter_prefix="patch_embed_proj",
            weight_key="model.visual.patch_embed.proj.weight",
            bias_key="model.visual.patch_embed.proj.bias",
            in_features=config.patch_dim,
            out_features=config.hidden_size,
            dtype=config.dtype,
            reshape_rank5_weight=True,
        )

    def forward(self, pixel_values):
        return self.proj(pixel_values)


class GlmOcrVisionPatchMerger(dml.nn.Module):
    def __init__(self, config: GlmOcrVisionConfig, weights: Mapping[str, np.ndarray]):
        dim = config.out_hidden_size
        context_dim = config.out_hidden_size * config.in_channels
        self.proj = _loaded_linear(
            weights,
            parameter_prefix="merger_proj",
            weight_key="model.visual.merger.proj.weight",
            in_features=dim,
            out_features=dim,
            dtype=config.dtype,
        )
        self.post_projection_norm = _loaded_layer_norm(
            weights,
            parameter_prefix="merger_post_projection_norm",
            weight_key="model.visual.merger.post_projection_norm.weight",
            bias_key="model.visual.merger.post_projection_norm.bias",
            hidden_size=dim,
            eps=1.0e-5,
            dtype=config.dtype,
        )
        self.gate_proj = _loaded_linear(
            weights,
            parameter_prefix="merger_gate_proj",
            weight_key="model.visual.merger.gate_proj.weight",
            in_features=dim,
            out_features=context_dim,
            dtype=config.dtype,
        )
        self.up_proj = _loaded_linear(
            weights,
            parameter_prefix="merger_up_proj",
            weight_key="model.visual.merger.up_proj.weight",
            in_features=dim,
            out_features=context_dim,
            dtype=config.dtype,
        )
        self.down_proj = _loaded_linear(
            weights,
            parameter_prefix="merger_down_proj",
            weight_key="model.visual.merger.down_proj.weight",
            in_features=context_dim,
            out_features=dim,
            dtype=config.dtype,
        )

    def forward(self, hidden_state):
        hidden_state = self.proj(hidden_state)
        hidden_state = dml.ops.gelu(self.post_projection_norm(hidden_state), approximation="none")
        return self.down_proj(dml.ops.mul(dml.ops.silu(self.gate_proj(hidden_state)), self.up_proj(hidden_state)))


class GlmOcrVisionModel(dml.nn.Module):
    def __init__(self, config: GlmOcrVisionConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.patch_embed = GlmOcrVisionPatchEmbedLinear(config, weights)
        self.blocks = dml.nn.ModuleList(GlmOcrVisionBlock(config, weights, idx) for idx in range(config.depth))
        self.post_layernorm = _loaded_rms_norm(
            weights,
            parameter_prefix="post_layernorm",
            weight_key="model.visual.post_layernorm.weight",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.dtype,
        )
        self.downsample = _loaded_linear(
            weights,
            parameter_prefix="downsample",
            weight_key="model.visual.downsample.weight",
            bias_key="model.visual.downsample.bias",
            in_features=config.hidden_size * config.spatial_merge_size * config.spatial_merge_size,
            out_features=config.out_hidden_size,
            dtype=config.dtype,
            reshape_rank4_weight=True,
        )
        self.merger = GlmOcrVisionPatchMerger(config, weights)

    def encode(self, pixel_values, cos, sin):
        hidden_states = self.patch_embed(pixel_values)
        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin)
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = dml.ops.reshape(
            hidden_states,
            [-1, self.config.spatial_merge_size, self.config.spatial_merge_size, self.config.hidden_size],
        )
        hidden_states = dml.ops.permute(hidden_states, (0, 3, 1, 2))
        hidden_states = _materialize(hidden_states)
        hidden_states = dml.ops.reshape(
            hidden_states,
            [-1, self.config.hidden_size * self.config.spatial_merge_size * self.config.spatial_merge_size],
        )
        hidden_states = self.downsample(hidden_states)
        return hidden_states, self.merger(hidden_states)

    def forward(self, pixel_values, cos, sin):
        last_hidden_state, pooler_output = self.encode(pixel_values, cos, sin)
        return dml.ops.output(last_hidden_state, "last_hidden_state"), dml.ops.output(pooler_output, "pooler_output")


class GlmOcrForConditionalGeneration(dml.nn.Module):
    def __init__(self, config: GlmOcrConfig, weights: Mapping[str, np.ndarray], *, logits_to_keep: int = 0):
        self.config = config
        self.logits_to_keep = int(logits_to_keep)
        if self.logits_to_keep not in {0, 1}:
            raise ValueError("GlmOcrForConditionalGeneration currently supports logits_to_keep=0 or 1")
        self.language_model = GlmOcrTextModel(config.text_config, weights)
        self.lm_head = _loaded_linear(
            weights,
            parameter_prefix="lm_head",
            weight_key="lm_head.weight",
            in_features=config.text_config.hidden_size,
            out_features=config.text_config.vocab_size,
            dtype=config.text_config.dtype,
        )

    def forward(self, input_ids, cos, sin, attention_mask=None):
        hidden_states = self.language_model.encode(input_ids, cos, sin, attention_mask)
        if self.logits_to_keep == 1:
            batch, seq_len, hidden = _rank3_shape(hidden_states.shape, "GLM-OCR hidden_states")
            hidden_states = dml.ops.dynamic_slice(hidden_states, (0, seq_len - 1, 0), (batch, 1, hidden))
        logits = self.lm_head(hidden_states)
        return dml.ops.output(logits, "logits")


class GlmOcrForConditionalGenerationImagePrefill(dml.nn.Module):
    """Single-image multimodal prefill with a fixed contiguous placeholder span."""

    def __init__(
        self,
        config: GlmOcrConfig,
        weights: Mapping[str, np.ndarray],
        *,
        image_token_start: int,
        logits_to_keep: int = 0,
    ):
        self.config = config
        self.image_token_start = int(image_token_start)
        self.logits_to_keep = int(logits_to_keep)
        if self.image_token_start < 0:
            raise ValueError("image_token_start must be non-negative")
        if self.logits_to_keep not in {0, 1}:
            raise ValueError("GlmOcrForConditionalGenerationImagePrefill currently supports logits_to_keep=0 or 1")
        self.visual = GlmOcrVisionModel(config.vision_config, weights)
        self.language_model = GlmOcrTextModel(config.text_config, weights)
        self.lm_head = _loaded_linear(
            weights,
            parameter_prefix="lm_head",
            weight_key="lm_head.weight",
            in_features=config.text_config.hidden_size,
            out_features=config.text_config.vocab_size,
            dtype=config.text_config.dtype,
        )

    def forward(self, input_ids, pixel_values, vision_cos, vision_sin, text_cos, text_sin, attention_mask=None):
        _, image_features = self.visual.encode(pixel_values, vision_cos, vision_sin)
        image_features = dml.ops.unsqueeze(image_features, 0)
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        inputs_embeds = dml.ops.slice_scatter(inputs_embeds, image_features, [0, self.image_token_start, 0])
        hidden_states = self.language_model.encode_inputs_embeds(inputs_embeds, text_cos, text_sin, attention_mask)
        if self.logits_to_keep == 1:
            batch, seq_len, hidden = _rank3_shape(hidden_states.shape, "GLM-OCR hidden_states")
            hidden_states = dml.ops.dynamic_slice(hidden_states, (0, seq_len - 1, 0), (batch, 1, hidden))
        logits = self.lm_head(hidden_states)
        return dml.ops.output(logits, "logits")


class GlmOcrForConditionalGenerationImagePrefillWithCache(dml.nn.Module):
    """Single-image multimodal prefill that also emits decode-compatible KV cache."""

    def __init__(
        self,
        config: GlmOcrConfig,
        weights: Mapping[str, np.ndarray],
        *,
        image_token_start: int,
        logits_to_keep: int = 1,
    ):
        self.config = config
        self.image_token_start = int(image_token_start)
        self.logits_to_keep = int(logits_to_keep)
        if self.image_token_start < 0:
            raise ValueError("image_token_start must be non-negative")
        if self.logits_to_keep not in {0, 1}:
            raise ValueError("GlmOcrForConditionalGenerationImagePrefillWithCache currently supports logits_to_keep=0 or 1")
        self.visual = GlmOcrVisionModel(config.vision_config, weights)
        self.language_model = GlmOcrTextModel(config.text_config, weights)
        self.lm_head = _loaded_linear(
            weights,
            parameter_prefix="lm_head",
            weight_key="lm_head.weight",
            in_features=config.text_config.hidden_size,
            out_features=config.text_config.vocab_size,
            dtype=config.text_config.dtype,
        )

    def forward(self, input_ids, pixel_values, vision_cos, vision_sin, text_cos, text_sin, attention_mask=None):
        _, image_features = self.visual.encode(pixel_values, vision_cos, vision_sin)
        image_features = dml.ops.unsqueeze(image_features, 0)
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        inputs_embeds = dml.ops.slice_scatter(inputs_embeds, image_features, [0, self.image_token_start, 0])
        hidden_states, present = self.language_model.encode_inputs_embeds_with_cache(
            inputs_embeds,
            text_cos,
            text_sin,
            attention_mask,
        )
        if self.logits_to_keep == 1:
            batch, seq_len, hidden = _rank3_shape(hidden_states.shape, "GLM-OCR hidden_states")
            hidden_states = dml.ops.dynamic_slice(hidden_states, (0, seq_len - 1, 0), (batch, 1, hidden))
        logits = self.lm_head(hidden_states)
        outputs: dict[str, Any] = {"logits": dml.ops.output(logits, "logits")}
        for layer_idx, (present_key, present_value) in enumerate(present):
            outputs[f"present_key_{layer_idx}"] = dml.ops.output(present_key, f"present_key_{layer_idx}")
            outputs[f"present_value_{layer_idx}"] = dml.ops.output(present_value, f"present_value_{layer_idx}")
        return outputs


class GlmOcrForConditionalGenerationDecode(dml.nn.Module):
    def __init__(self, config: GlmOcrConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.language_model = GlmOcrTextModel(config.text_config, weights)
        self.lm_head = _loaded_linear(
            weights,
            parameter_prefix="lm_head",
            weight_key="lm_head.weight",
            in_features=config.text_config.hidden_size,
            out_features=config.text_config.vocab_size,
            dtype=config.text_config.dtype,
        )

    def forward(self, input_ids, cos, sin, attention_mask, **past_key_values):
        past = {}
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            past[layer_idx] = (
                past_key_values[f"past_key_{layer_idx}"],
                past_key_values[f"past_value_{layer_idx}"],
            )
        hidden_states, present = self.language_model.decode(input_ids, cos, sin, attention_mask, past)
        logits = self.lm_head(hidden_states)
        outputs: dict[str, Any] = {"logits": dml.ops.output(logits, "logits")}
        for layer_idx, (present_key, present_value) in enumerate(present):
            outputs[f"present_key_{layer_idx}"] = dml.ops.output(present_key, f"present_key_{layer_idx}")
            outputs[f"present_value_{layer_idx}"] = dml.ops.output(present_value, f"present_value_{layer_idx}")
        return outputs


class GlmOcrForConditionalGenerationDecodeStaticCache(dml.nn.Module):
    def __init__(self, config: GlmOcrConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.language_model = GlmOcrTextModel(config.text_config, weights)
        self.lm_head = _loaded_linear(
            weights,
            parameter_prefix="lm_head",
            weight_key="lm_head.weight",
            in_features=config.text_config.hidden_size,
            out_features=config.text_config.vocab_size,
            dtype=config.text_config.dtype,
        )

    def forward(self, input_ids, cos, sin, attention_mask, cache_seqlens=None, **past_key_values):
        past = {}
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            past[layer_idx] = (
                past_key_values[f"past_key_{layer_idx}"],
                past_key_values[f"past_value_{layer_idx}"],
            )
        hidden_states, updates = self.language_model.decode_static_cache(
            input_ids,
            cos,
            sin,
            attention_mask,
            past,
            cache_seqlens,
        )
        logits = self.lm_head(hidden_states)
        outputs: dict[str, Any] = {"logits": dml.ops.output(logits, "logits")}
        for layer_idx, (new_key, new_value) in enumerate(updates):
            outputs[f"new_key_{layer_idx}"] = dml.ops.output(new_key, f"new_key_{layer_idx}")
            outputs[f"new_value_{layer_idx}"] = dml.ops.output(new_value, f"new_value_{layer_idx}")
        return outputs


class GlmOcrForConditionalGenerationDecodeSessionStaticCache(dml.nn.Module):
    def __init__(self, config: GlmOcrConfig, weights: Mapping[str, np.ndarray], *, max_cache_len: int):
        self.config = config
        self.max_cache_len = int(max_cache_len)
        if self.max_cache_len <= 0:
            raise ValueError("max_cache_len must be positive")
        self.language_model = GlmOcrTextModel(config.text_config, weights)
        self.lm_head = _loaded_linear(
            weights,
            parameter_prefix="lm_head",
            weight_key="lm_head.weight",
            in_features=config.text_config.hidden_size,
            out_features=config.text_config.vocab_size,
            dtype=config.text_config.dtype,
        )

    def forward(self, input_ids, cos, sin, attention_mask, cache_seqlens):
        hidden_states = self.language_model.decode_session_static_cache(
            input_ids,
            cos,
            sin,
            attention_mask,
            max_cache_len=self.max_cache_len,
            cache_seqlens=cache_seqlens,
        )
        logits = self.lm_head(hidden_states)
        return {"logits": dml.ops.output(logits, "logits")}


def apply_glm_ocr_text_rope(q, k, cos, sin, rotary_dim: int):
    cos = dml.ops.repeat_interleave(_prefix_last_dim(cos, rotary_dim // 2), 2, dim=-1)
    sin = dml.ops.repeat_interleave(_prefix_last_dim(sin, rotary_dim // 2), 2, dim=-1)
    cos = dml.ops.unsqueeze(cos, 2)
    sin = dml.ops.unsqueeze(sin, 2)
    if rotary_dim < int(q.shape[-1]):
        q_rot, q_pass = dml.ops.split(q, [rotary_dim, int(q.shape[-1]) - rotary_dim], dim=-1)
        k_rot, k_pass = dml.ops.split(k, [rotary_dim, int(k.shape[-1]) - rotary_dim], dim=-1)
    else:
        q_rot, q_pass = q, None
        k_rot, k_pass = k, None
    q_embed = dml.ops.add(dml.ops.mul(q_rot, cos), dml.ops.mul(_rotate_even_odd(q_rot), sin))
    k_embed = dml.ops.add(dml.ops.mul(k_rot, cos), dml.ops.mul(_rotate_even_odd(k_rot), sin))
    if q_pass is not None:
        q_embed = dml.ops.concatenate([q_embed, q_pass], dim=-1)
        k_embed = dml.ops.concatenate([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def apply_glm_ocr_vision_rope(q, k, cos, sin):
    q_dtype = q.dtype
    k_dtype = k.dtype
    q_rot = dml.ops.cast(q, "float32") if q_dtype != "float32" else q
    k_rot = dml.ops.cast(k, "float32") if k_dtype != "float32" else k
    cos = dml.ops.unsqueeze(cos, 1)
    sin = dml.ops.unsqueeze(sin, 1)
    cos = dml.ops.cast(cos, "float32") if cos.dtype != "float32" else cos
    sin = dml.ops.cast(sin, "float32") if sin.dtype != "float32" else sin
    q_embed = dml.ops.add(dml.ops.mul(q_rot, cos), dml.ops.mul(_rotate_half(q_rot), sin))
    k_embed = dml.ops.add(dml.ops.mul(k_rot, cos), dml.ops.mul(_rotate_half(k_rot), sin))
    if q_dtype != "float32":
        q_embed = dml.ops.cast(q_embed, q_dtype)
    if k_dtype != "float32":
        k_embed = dml.ops.cast(k_embed, k_dtype)
    return q_embed, k_embed


def glm_ocr_weights_from_transformers_state_dict(
    state_dict: Mapping[str, object],
    config: GlmOcrConfig,
    *,
    dtype: str | None = None,
) -> dict[str, np.ndarray]:
    dtype = config.text_config.dtype if dtype is None else _normalize_glm_ocr_dtype(dtype)
    required = glm_ocr_required_weight_names(config)
    missing = [name for name in required if name not in state_dict]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} missing total)"
        raise KeyError(f"Missing Transformers GLM-OCR state_dict weights: {preview}{suffix}")
    return {name: _state_value_to_numpy(state_dict[name], dtype=dtype) for name in required}


def glm_ocr_weights_from_safetensors_file(
    path: str | Path,
    config: GlmOcrConfig,
    *,
    dtype: str | None = None,
    required_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("glm_ocr_weights_from_safetensors_file requires the optional safetensors package") from exc
    dtype = config.text_config.dtype if dtype is None else _normalize_glm_ocr_dtype(dtype)
    required = list(required_names) if required_names is not None else glm_ocr_required_weight_names(config)
    weights: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        missing = [name for name in required if name not in available]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} missing total)"
            raise KeyError(f"Missing GLM-OCR safetensors weights: {preview}{suffix}")
        for name in required:
            weights[name] = _state_value_to_numpy(handle.get_tensor(name), dtype=dtype)
    return weights


def glm_ocr_hf_snapshot_path(model_id: str = "zai-org/GLM-OCR", *, revision: str | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("glm_ocr_hf_snapshot_path requires the optional huggingface_hub package") from exc
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "model.safetensors",
            ],
        )
    )


def glm_ocr_required_weight_names(config: GlmOcrConfig) -> list[str]:
    names = [
        "model.visual.patch_embed.proj.weight",
        "model.visual.patch_embed.proj.bias",
        "model.visual.post_layernorm.weight",
        "model.visual.downsample.weight",
        "model.visual.downsample.bias",
        "model.visual.merger.proj.weight",
        "model.visual.merger.post_projection_norm.weight",
        "model.visual.merger.post_projection_norm.bias",
        "model.visual.merger.gate_proj.weight",
        "model.visual.merger.up_proj.weight",
        "model.visual.merger.down_proj.weight",
    ]
    names = [*glm_ocr_required_text_weight_names(config), *names]
    for idx in range(config.vision_config.depth):
        prefix = f"model.visual.blocks.{idx}"
        names.extend(
            [
                f"{prefix}.norm1.weight",
                f"{prefix}.attn.qkv.weight",
                f"{prefix}.attn.qkv.bias",
                f"{prefix}.attn.proj.weight",
                f"{prefix}.attn.proj.bias",
                f"{prefix}.attn.q_norm.weight",
                f"{prefix}.attn.k_norm.weight",
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


def glm_ocr_required_text_weight_names(config: GlmOcrConfig) -> list[str]:
    names = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    for idx in range(config.text_config.num_hidden_layers):
        prefix = f"model.language_model.layers.{idx}"
        names.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.post_self_attn_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate_up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.post_mlp_layernorm.weight",
            ]
        )
    return names


def _rotate_even_odd(x):
    last_dim = int(x.shape[-1])
    x1 = dml.ops.index_select(x, -1, range(0, last_dim, 2))
    x2 = dml.ops.index_select(x, -1, range(1, last_dim, 2))
    rotated = dml.ops.stack([dml.ops.mul(x2, -1.0), x1], dim=-1)
    return dml.ops.reshape(rotated, list(x.shape))


def _rotate_half(x):
    last_dim = int(x.shape[-1])
    half = last_dim // 2
    x1 = dml.ops.index_select(x, -1, range(0, half))
    x2 = dml.ops.index_select(x, -1, range(half, last_dim))
    return dml.ops.concatenate([dml.ops.mul(x2, -1.0), x1], dim=-1)


def _materialize(x):
    return dml.ops.permute(x, tuple(range(len(x.shape))))


def _repeat_kv_heads(values, n_rep: int):
    if n_rep == 1:
        return values
    _, kv_heads, _, _ = values.shape
    indices = [head for head in range(int(kv_heads)) for _ in range(int(n_rep))]
    return dml.ops.index_select(values, 1, indices)


def _prefix_last_dim(x, prefix: int):
    last_dim = int(x.shape[-1])
    if prefix == last_dim:
        return x
    return dml.ops.index_select(x, -1, range(int(prefix)))


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
    reshape_rank4_weight: bool = False,
) -> dml.nn.Linear:
    dtype = _normalize_glm_ocr_dtype(dtype)
    layer = dml.nn.Linear(in_features, out_features, bias=bias_key is not None, dtype=dtype)
    weight = _weight_value(weights, weight_key, None, dtype=dtype)
    if reshape_rank4_weight and reshape_rank5_weight:
        raise ValueError("Only one GLM-OCR linear weight reshape mode can be enabled")
    if reshape_rank4_weight:
        weight = np.reshape(weight, (out_features, in_features))
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


def _loaded_linear_row_slice(
    weights: Mapping[str, np.ndarray],
    *,
    parameter_prefix: str,
    weight_key: str,
    in_features: int,
    out_features: int,
    start_row: int,
    dtype: str = "float32",
) -> dml.nn.Linear:
    dtype = _normalize_glm_ocr_dtype(dtype)
    layer = dml.nn.Linear(in_features, out_features, bias=False, dtype=dtype)
    full_weight = _weight_value(weights, weight_key, None, dtype=dtype)
    end_row = int(start_row) + int(out_features)
    weight = np.ascontiguousarray(full_weight[int(start_row) : end_row])
    _check_shape(weight, (out_features, in_features), weight_key)
    layer.weight = dml.Parameter([out_features, in_features], dtype=dtype, name=f"{parameter_prefix}_weight", value=weight)
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
    dtype = _normalize_glm_ocr_dtype(dtype)
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
    dtype = _normalize_glm_ocr_dtype(dtype)
    layer = dml.nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)
    layer.weight = dml.Parameter(
        [hidden_size],
        dtype=dtype,
        name=f"{parameter_prefix}_weight",
        value=_weight_value(weights, weight_key, (hidden_size,), dtype=dtype),
    )
    return layer


def _loaded_layer_norm(
    weights: Mapping[str, np.ndarray],
    *,
    parameter_prefix: str,
    weight_key: str,
    bias_key: str,
    hidden_size: int,
    eps: float,
    dtype: str,
) -> dml.nn.LayerNorm:
    dtype = _normalize_glm_ocr_dtype(dtype)
    layer = dml.nn.LayerNorm(hidden_size, eps=eps, dtype=dtype)
    layer.weight = dml.Parameter(
        [hidden_size],
        dtype=dtype,
        name=f"{parameter_prefix}_weight",
        value=_weight_value(weights, weight_key, (hidden_size,), dtype=dtype),
    )
    layer.bias = dml.Parameter(
        [hidden_size],
        dtype=dtype,
        name=f"{parameter_prefix}_bias",
        value=_weight_value(weights, bias_key, (hidden_size,), dtype=dtype),
    )
    return layer


def _loaded_conv2d(
    weights: Mapping[str, np.ndarray],
    *,
    parameter_prefix: str,
    weight_key: str,
    bias_key: str | None,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    dtype: str,
) -> dml.nn.Conv2d:
    dtype = _normalize_glm_ocr_dtype(dtype)
    layer = dml.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        bias=bias_key is not None,
        dtype=dtype,
    )
    layer.weight = dml.Parameter(
        [out_channels, in_channels, kernel_size, kernel_size],
        dtype=dtype,
        name=f"{parameter_prefix}_weight",
        value=_weight_value(weights, weight_key, (out_channels, in_channels, kernel_size, kernel_size), dtype=dtype),
    )
    if bias_key is not None:
        layer.bias = dml.Parameter(
            [out_channels],
            dtype=dtype,
            name=f"{parameter_prefix}_bias",
            value=_weight_value(weights, bias_key, (out_channels,), dtype=dtype),
        )
    return layer


def _weight_value(
    weights: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...] | None,
    *,
    dtype: str,
) -> np.ndarray:
    if name not in weights:
        raise KeyError(f"Missing GLM-OCR weight: {name}")
    value = _state_value_to_numpy(weights[name], dtype=dtype)
    if shape is not None:
        _check_shape(value, shape, name)
    return value


def _state_value_to_numpy(value: object, *, dtype: str) -> np.ndarray:
    dtype = _normalize_glm_ocr_dtype(dtype)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float") and str(getattr(value, "dtype", "")).endswith("bfloat16"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if dtype == "bfloat16":
        return array_to_storage(value, dtype)
    return np.asarray(value, dtype=_numpy_glm_ocr_dtype(dtype))


def _check_shape(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if value.shape != shape:
        raise ValueError(f"Weight {name} has shape {value.shape}, expected {shape}")


def _single_vision_rope_index(
    start_position: int,
    grid_thw: Sequence[int],
    *,
    spatial_merge_size: int,
    dtype: np.dtype,
) -> np.ndarray:
    grid_t, grid_h, grid_w = (int(v) for v in grid_thw)
    llm_grid_t = grid_t
    llm_grid_h = grid_h // spatial_merge_size
    llm_grid_w = grid_w // spatial_merge_size
    temporal = np.arange(llm_grid_t, dtype=dtype).repeat(llm_grid_h * llm_grid_w) + start_position
    height = np.arange(llm_grid_h, dtype=dtype).repeat(llm_grid_w).repeat(llm_grid_t) + start_position
    width = np.tile(np.arange(llm_grid_w, dtype=dtype), llm_grid_h * llm_grid_t) + start_position
    return np.stack([temporal, height, width], axis=0)


def _rank2_shape(shape: Sequence[int], name: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"{name} must be rank 2")
    return int(shape[0]), int(shape[1])


def _rank3_shape(shape: Sequence[int], name: str) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError(f"{name} must be rank 3")
    return int(shape[0]), int(shape[1]), int(shape[2])


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
    "GlmOcrConfig",
    "GlmOcrForConditionalGeneration",
    "GlmOcrForConditionalGenerationDecode",
    "GlmOcrForConditionalGenerationDecodeSessionStaticCache",
    "GlmOcrForConditionalGenerationDecodeStaticCache",
    "GlmOcrForConditionalGenerationImagePrefill",
    "GlmOcrForConditionalGenerationImagePrefillWithCache",
    "GlmOcrTextConfig",
    "GlmOcrTextModel",
    "GlmOcrVisionConfig",
    "GlmOcrVisionModel",
    "GlmOcrVisionPatchEmbedLinear",
    "apply_glm_ocr_text_rope",
    "apply_glm_ocr_vision_rope",
    "glm_ocr_config_from_transformers_config",
    "glm_ocr_config_from_transformers_dict",
    "glm_ocr_patch_embed_linear_weight",
    "glm_ocr_prepare_inputs_for_generation",
    "glm_ocr_required_weight_names",
    "glm_ocr_required_text_weight_names",
    "glm_ocr_rope_index",
    "glm_ocr_stitch_image_features",
    "glm_ocr_text_rope_embeddings",
    "glm_ocr_vision_position_ids",
    "glm_ocr_vision_rope_embeddings",
    "glm_ocr_hf_snapshot_path",
    "glm_ocr_weights_from_safetensors_file",
    "glm_ocr_weights_from_transformers_state_dict",
]
