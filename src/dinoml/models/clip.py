from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

import dinoml as dml
from dinoml.ir import dtype_numpy, normalize_dtype


_CLIP_FLOAT_DTYPES = frozenset({"float16", "float32"})


def _normalize_clip_dtype(dtype: str) -> str:
    normalized = normalize_dtype(dtype)
    if normalized not in _CLIP_FLOAT_DTYPES:
        raise ValueError(f"CLIP currently supports float16/float32 parameters, got {dtype!r}")
    return normalized


def _numpy_clip_dtype(dtype: str) -> np.dtype:
    return dtype_numpy(_normalize_clip_dtype(dtype))


@dataclass(frozen=True)
class LegacyCLIPTextConfig:
    """Bounded legacy OpenAI CLIP text-tower config.

    This helper intentionally models only the bounded text path currently
    landed in DinoML: static traced sequence length, optional default
    `position_ids`, and the two source CLIP EOS pooling branches
    (`eos_token_id == 2` argmax/highest-token-id compatibility, otherwise
    first EOS equality match).
    """

    vocab_size: int
    max_position_embeddings: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    projection_dim: int
    layer_norm_eps: float = 1.0e-5
    eos_token_id: int = 2
    mask_fill_value: float = -1.0e4
    use_flash_attention: bool = False
    assume_unpadded_attention_mask: bool = False
    dtype: str = "float32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _normalize_clip_dtype(self.dtype))
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if self.num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be non-negative")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass(frozen=True)
class LegacyCLIPVisionEmbeddingsConfig:
    """Bounded CLIP vision-embeddings config.

    This helper intentionally models only the admitted CLIP vision embedding
    slice currently landing in DinoML: fixed square NCHW pixel input, patch
    projection, patch flatten/transposition into a sequence, CLS prepend, and
    learned absolute position add. It does not admit interpolation, arbitrary
    image sizes, or the full vision encoder/projection path.
    """

    hidden_size: int
    image_size: int
    patch_size: int
    num_channels: int = 3
    dtype: str = "float32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _normalize_clip_dtype(self.dtype))
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if self.patch_size > self.image_size:
            raise ValueError("patch_size must be less than or equal to image_size")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_positions(self) -> int:
        return self.num_patches + 1


@dataclass(frozen=True)
class LegacyCLIPVisionConfig:
    """Bounded CLIP vision-wrapper config.

    This helper intentionally models only the admitted CLIP vision-wrapper
    slice currently landed in DinoML: fixed square NCHW pixel input,
    source-faithful embeddings, pre-LayerNorm, either a no-op encoder
    (`num_hidden_layers == 0`) or one-or-more real encoder blocks,
    CLS pool, post-LayerNorm, and bias-free visual projection. It does not
    admit interpolation or arbitrary image sizes.
    """

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    projection_dim: int
    image_size: int
    patch_size: int
    num_channels: int = 3
    layer_norm_eps: float = 1.0e-5
    use_flash_attention: bool = False
    dtype: str = "float32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _normalize_clip_dtype(self.dtype))
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be non-negative")
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if self.patch_size > self.image_size:
            raise ValueError("patch_size must be less than or equal to image_size")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_positions(self) -> int:
        return self.num_patches + 1

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def build_clip_causal_mask(seq_len: int, mask_fill_value: float = -1.0e4, *, dtype: str = "float32") -> np.ndarray:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    numpy_dtype = _numpy_clip_dtype(dtype)
    causal = np.zeros((1, seq_len, seq_len), dtype=numpy_dtype)
    rows, cols = np.triu_indices(seq_len, k=1)
    causal[:, rows, cols] = np.asarray(mask_fill_value, dtype=numpy_dtype)
    return causal


def _loaded_linear(
    weights: Mapping[str, np.ndarray],
    *,
    parameter_prefix: str,
    weight_key: str,
    in_features: int,
    out_features: int,
    bias_key: str | None = None,
    specialization: str | None = None,
    dtype: str = "float32",
) -> dml.nn.Linear:
    dtype = _normalize_clip_dtype(dtype)
    layer = dml.nn.Linear(
        in_features,
        out_features,
        bias=bias_key is not None,
        dtype=dtype,
        specialization=specialization,
    )
    layer.weight = dml.Parameter(
        [out_features, in_features],
        dtype=dtype,
        name=f"{parameter_prefix}_weight",
        value=_weight_value(weights, weight_key, (out_features, in_features), dtype=dtype),
    )
    if bias_key is not None:
        layer.bias = dml.Parameter(
            [out_features],
            dtype=dtype,
            name=f"{parameter_prefix}_bias",
            value=_weight_value(weights, bias_key, (out_features,), dtype=dtype),
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
    dtype: str = "float32",
) -> dml.nn.LayerNorm:
    dtype = _normalize_clip_dtype(dtype)
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


def _loaded_embedding(
    weights: Mapping[str, np.ndarray],
    *,
    parameter_prefix: str,
    weight_key: str,
    num_embeddings: int,
    embedding_dim: int,
    dtype: str = "float32",
) -> dml.nn.Embedding:
    dtype = _normalize_clip_dtype(dtype)
    layer = dml.nn.Embedding(num_embeddings, embedding_dim, dtype=dtype)
    layer.weight = dml.Parameter(
        [num_embeddings, embedding_dim],
        dtype=dtype,
        name=f"{parameter_prefix}_weight",
        value=_weight_value(weights, weight_key, (num_embeddings, embedding_dim), dtype=dtype),
    )
    return layer


def _loaded_conv2d(
    weights: Mapping[str, np.ndarray],
    *,
    parameter_prefix: str,
    weight_key: str,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    dtype: str = "float32",
) -> dml.nn.Conv2d:
    dtype = _normalize_clip_dtype(dtype)
    layer = dml.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        bias=False,
        dtype=dtype,
    )
    layer.weight = dml.Parameter(
        [out_channels, in_channels, kernel_size, kernel_size],
        dtype=dtype,
        name=f"{parameter_prefix}_weight",
        value=_weight_value(weights, weight_key, (out_channels, in_channels, kernel_size, kernel_size), dtype=dtype),
    )
    return layer


class _LegacyCLIPSelfAttention(dml.nn.Module):
    def __init__(
        self,
        config: LegacyCLIPTextConfig | LegacyCLIPVisionConfig,
        weights: Mapping[str, np.ndarray],
        prefix: str,
    ):
        self.config = config
        hidden = config.hidden_size
        self.qkv_proj = dml.nn.Linear(hidden, hidden * 3, dtype=config.dtype)
        self.qkv_proj.weight = dml.Parameter(
            [hidden * 3, hidden],
            dtype=config.dtype,
            name="qkv_proj_weight",
            value=np.concatenate(
                [
                    _weight_value(weights, f"{prefix}.self_attn.q_proj.weight", (hidden, hidden), dtype=config.dtype),
                    _weight_value(weights, f"{prefix}.self_attn.k_proj.weight", (hidden, hidden), dtype=config.dtype),
                    _weight_value(weights, f"{prefix}.self_attn.v_proj.weight", (hidden, hidden), dtype=config.dtype),
                ],
                axis=0,
            ),
        )
        self.qkv_proj.bias = dml.Parameter(
            [hidden * 3],
            dtype=config.dtype,
            name="qkv_proj_bias",
            value=np.concatenate(
                [
                    _weight_value(weights, f"{prefix}.self_attn.q_proj.bias", (hidden,), dtype=config.dtype),
                    _weight_value(weights, f"{prefix}.self_attn.k_proj.bias", (hidden,), dtype=config.dtype),
                    _weight_value(weights, f"{prefix}.self_attn.v_proj.bias", (hidden,), dtype=config.dtype),
                ],
                axis=0,
            ),
        )
        self.out_proj = _loaded_linear(
            weights,
            parameter_prefix="out_proj",
            weight_key=f"{prefix}.self_attn.out_proj.weight",
            bias_key=f"{prefix}.self_attn.out_proj.bias",
            in_features=hidden,
            out_features=hidden,
            dtype=config.dtype,
        )

    def forward(self, hidden_states, attention_mask=None, causal_mask=None, *, causal: bool = False):
        seq_len = _hidden_sequence_length(hidden_states.shape)
        batch = _first_static_dim(hidden_states.shape)
        hidden = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        head_dim = self.config.head_dim

        if self._use_flash_attention(hidden_states, attention_mask):
            qkv = self.qkv_proj(hidden_states)
            qkv_5d = dml.ops.reshape(qkv, [batch, seq_len, 3, num_heads, head_dim])
            context = dml.ops.flash_attention_qkv(qkv_5d, causal=bool(causal) or causal_mask is not None)
            context = dml.ops.reshape(context, [batch, seq_len, hidden])
            return self.out_proj(context)

        if causal and causal_mask is None:
            raise ValueError("non-flash causal CLIP attention requires a causal_mask tensor")
        q_4d, k_4d, v_4d = self._project_qkv_heads_4d(hidden_states, batch, seq_len, hidden, num_heads, head_dim)
        q = self._flatten_heads(q_4d)
        k = self._flatten_heads(k_4d)
        v = self._flatten_heads(v_4d)
        scores = dml.ops.bmm_rcr(q, k)
        scores = dml.ops.mul(scores, 1.0 / math.sqrt(head_dim))
        if causal_mask is not None:
            scores = dml.ops.add(scores, causal_mask)
        if attention_mask is not None:
            scores = self._apply_attention_mask(scores, attention_mask, batch, seq_len, num_heads)

        probs = dml.ops.softmax(scores, dim=-1)
        context = dml.ops.bmm_rrr(probs, v)
        context = dml.ops.reshape(context, [batch, num_heads, seq_len, head_dim])
        context = dml.ops.permute0213(context)
        context = dml.ops.reshape(context, [batch, seq_len, hidden])
        return self.out_proj(context)

    def _project_qkv_heads_4d(self, hidden_states, batch: int, seq_len: int, hidden: int, num_heads: int, head_dim: int):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = dml.ops.split(qkv, hidden, dim=2)
        return (
            dml.ops.reshape(q, [batch, seq_len, num_heads, head_dim]),
            dml.ops.reshape(k, [batch, seq_len, num_heads, head_dim]),
            dml.ops.reshape(v, [batch, seq_len, num_heads, head_dim]),
        )

    def _flatten_heads(self, values):
        values = dml.ops.permute0213(values)
        return dml.ops.flatten(values, start_dim=0, end_dim=1)

    def _use_flash_attention(self, hidden_states, attention_mask) -> bool:
        if not bool(getattr(self.config, "use_flash_attention", False)):
            return False
        if hidden_states.dtype != "float16":
            return False
        if attention_mask is None:
            return True
        return bool(getattr(self.config, "assume_unpadded_attention_mask", False))

    def _apply_attention_mask(self, scores, attention_mask, batch: int, seq_len: int, num_heads: int):
        keep = dml.ops.reshape(attention_mask, [batch, 1, 1, seq_len])
        keep = dml.ops.expand(keep, [batch, num_heads, seq_len, seq_len])
        keep = dml.ops.reshape(keep, [batch * num_heads, seq_len, seq_len])
        zeros = dml.ops.full([batch * num_heads, seq_len, seq_len], 0.0, dtype=scores.dtype)
        masked = dml.ops.full(
            [batch * num_heads, seq_len, seq_len],
            float(self.config.mask_fill_value),
            dtype=scores.dtype,
        )
        return dml.ops.add(scores, dml.ops.where(keep, zeros, masked))


class _LegacyCLIPMLP(dml.nn.Module):
    def __init__(
        self,
        config: LegacyCLIPTextConfig | LegacyCLIPVisionConfig,
        weights: Mapping[str, np.ndarray],
        prefix: str,
    ):
        self.fc1 = _loaded_linear(
            weights,
            parameter_prefix="fc1",
            weight_key=f"{prefix}.mlp.fc1.weight",
            bias_key=f"{prefix}.mlp.fc1.bias",
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            specialization="quick_gelu",
            dtype=config.dtype,
        )
        self.fc2 = _loaded_linear(
            weights,
            parameter_prefix="fc2",
            weight_key=f"{prefix}.mlp.fc2.weight",
            bias_key=f"{prefix}.mlp.fc2.bias",
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            dtype=config.dtype,
        )

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        return self.fc2(hidden_states)


class _LegacyCLIPEncoderLayer(dml.nn.Module):
    def __init__(
        self,
        config: LegacyCLIPTextConfig | LegacyCLIPVisionConfig,
        weights: Mapping[str, np.ndarray],
        prefix: str,
    ):
        self.config = config
        self.layer_norm1 = _loaded_layer_norm(
            weights,
            parameter_prefix="layer_norm1",
            weight_key=f"{prefix}.layer_norm1.weight",
            bias_key=f"{prefix}.layer_norm1.bias",
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=config.dtype,
        )
        self.self_attn = _LegacyCLIPSelfAttention(config, weights, prefix)
        self.layer_norm2 = _loaded_layer_norm(
            weights,
            parameter_prefix="layer_norm2",
            weight_key=f"{prefix}.layer_norm2.weight",
            bias_key=f"{prefix}.layer_norm2.bias",
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=config.dtype,
        )
        self.mlp = _LegacyCLIPMLP(config, weights, prefix)

    def forward(self, hidden_states, attention_mask=None, causal_mask=None, *, causal: bool = False):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            causal=causal,
        )
        residual, hidden_states = dml.ops.add_layer_norm(
            hidden_states,
            residual,
            self.layer_norm2.weight,
            self.layer_norm2.bias,
            eps=self.config.layer_norm_eps,
        )
        hidden_states = self.mlp(hidden_states)
        return dml.ops.add(residual, hidden_states)


class _LegacyCLIPTextEncoderLayer(_LegacyCLIPEncoderLayer):
    def __init__(self, config: LegacyCLIPTextConfig, weights: Mapping[str, np.ndarray], layer_idx: int):
        super().__init__(config, weights, f"text_model.encoder.layers.{layer_idx}")

    def forward(self, hidden_states, attention_mask, causal_mask, *, causal: bool = False):
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            causal=causal,
        )


class _LegacyCLIPVisionEncoderLayer(_LegacyCLIPEncoderLayer):
    def __init__(self, config: LegacyCLIPVisionConfig, weights: Mapping[str, np.ndarray], layer_idx: int):
        super().__init__(config, weights, f"vision_model.encoder.layers.{layer_idx}")


class LegacyCLIPTextModelWithProjection(dml.nn.Module):
    def __init__(self, config: LegacyCLIPTextConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.token_embedding = _loaded_embedding(
            weights,
            parameter_prefix="token_embedding",
            weight_key="text_model.embeddings.token_embedding.weight",
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
        )
        self.position_embedding = _loaded_embedding(
            weights,
            parameter_prefix="position_embedding",
            weight_key="text_model.embeddings.position_embedding.weight",
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
        )
        self.layers = dml.nn.ModuleList(
            _LegacyCLIPTextEncoderLayer(config, weights, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.final_layer_norm = _loaded_layer_norm(
            weights,
            parameter_prefix="final_layer_norm",
            weight_key="text_model.final_layer_norm.weight",
            bias_key="text_model.final_layer_norm.bias",
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=config.dtype,
        )
        self.text_projection = _loaded_linear(
            weights,
            parameter_prefix="text_projection",
            weight_key="text_projection.weight",
            in_features=config.hidden_size,
            out_features=config.projection_dim,
            dtype=config.dtype,
        )
        self.causal_mask = None
        if not self._can_skip_causal_mask_parameter():
            self.causal_mask = dml.Parameter(
                [1, config.max_position_embeddings, config.max_position_embeddings],
                dtype=config.dtype,
                name="causal_mask",
                value=build_clip_causal_mask(
                    config.max_position_embeddings,
                    config.mask_fill_value,
                    dtype=config.dtype,
                ),
            )

    def _can_skip_causal_mask_parameter(self) -> bool:
        return bool(
            self.config.use_flash_attention
            and self.config.assume_unpadded_attention_mask
            and self.config.dtype == "float16"
        )

    def _causal_mask_for_sequence(self, seq_len: int):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                "traced seq_len must be less than or equal to max_position_embeddings"
            )
        if self.causal_mask is None:
            raise ValueError("causal mask tensor is not available for this flash-attention CLIP text config")
        return dml.ops.dynamic_slice(
            self.causal_mask,
            start_indices=(0, 0, 0),
            slice_sizes=(1, seq_len, seq_len),
        )

    def _pool_hidden_state(self, input_ids, hidden_states):
        if self.config.eos_token_id == 2:
            indices = dml.ops.argmax(input_ids, dim=-1, keepdim=True)
        else:
            eos_mask = dml.ops.eq(input_ids, self.config.eos_token_id)
            indices = dml.ops.argmax(eos_mask, dim=-1, keepdim=True)
        pooled = dml.ops.batch_gather(hidden_states, indices)
        return dml.ops.squeeze(pooled, 1)

    def _default_position_ids(self, seq_len: int) -> dml.Parameter:
        return dml.Parameter(np.arange(seq_len, dtype=np.int64), dtype="int64")

    def encode_text(self, input_ids, attention_mask, position_ids=None):
        seq_len = _sequence_length(input_ids.shape)
        if position_ids is None:
            position_ids = self._default_position_ids(seq_len)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        hidden_states = dml.ops.add(token_embeddings, position_embeddings)
        causal_mask = None if self._can_skip_causal_mask_parameter() else self._causal_mask_for_sequence(seq_len)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, causal_mask, causal=True)

        last_hidden_state = self.final_layer_norm(hidden_states)
        pooled_output = self._pool_hidden_state(input_ids, last_hidden_state)
        return self.text_projection(pooled_output)

    def forward(self, input_ids, attention_mask, position_ids=None):
        text_features = self.encode_text(input_ids, attention_mask, position_ids)
        return dml.ops.output(text_features, "text_features")


class LegacyCLIPVisionEmbeddings(dml.nn.Module):
    def __init__(
        self,
        config: LegacyCLIPVisionEmbeddingsConfig | LegacyCLIPVisionConfig,
        weights: Mapping[str, np.ndarray],
    ):
        self.config = config
        self.class_embedding = dml.Parameter(
            [1, 1, config.hidden_size],
            dtype=config.dtype,
            name="class_embedding",
            value=_weight_value(
                weights,
                "vision_model.embeddings.class_embedding",
                (config.hidden_size,),
                dtype=config.dtype,
            ).reshape(1, 1, config.hidden_size),
        )
        self.patch_embedding = _loaded_conv2d(
            weights,
            parameter_prefix="patch_embedding",
            weight_key="vision_model.embeddings.patch_embedding.weight",
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            dtype=config.dtype,
        )
        self.position_embedding = _loaded_embedding(
            weights,
            parameter_prefix="position_embedding",
            weight_key="vision_model.embeddings.position_embedding.weight",
            num_embeddings=config.num_positions,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
        )
        self.position_ids = dml.Parameter(
            np.arange(config.num_positions, dtype=np.int64).reshape(1, config.num_positions),
            dtype="int64",
            name="position_ids",
        )

    def encode_pixels(self, pixel_values):
        batch, channels, height, width = _nchw_image_shape(pixel_values.shape)
        if channels != self.config.num_channels:
            raise ValueError(
                f"expected pixel_values channel dimension {self.config.num_channels}, got {channels}"
            )
        if height != self.config.image_size or width != self.config.image_size:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model "
                f"({self.config.image_size}*{self.config.image_size})."
            )

        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = dml.ops.flatten(patch_embeds, start_dim=2)
        patch_embeds = dml.ops.permute021(patch_embeds)

        class_embeds = dml.ops.expand(self.class_embedding, [batch, 1, self.config.hidden_size])
        embeddings = dml.ops.concatenate([class_embeds, patch_embeds], dim=1)
        position_embeddings = self.position_embedding(self.position_ids)
        return dml.ops.add(embeddings, position_embeddings)

    def forward(self, pixel_values):
        embeddings = self.encode_pixels(pixel_values)
        return dml.ops.output(embeddings, "embeddings")


class LegacyCLIPVisionModelWithProjection(dml.nn.Module):
    def __init__(self, config: LegacyCLIPVisionConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.embeddings = LegacyCLIPVisionEmbeddings(config, weights)
        self.layers = dml.nn.ModuleList(
            _LegacyCLIPVisionEncoderLayer(config, weights, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.pre_layrnorm = _loaded_layer_norm(
            weights,
            parameter_prefix="pre_layrnorm",
            weight_key="vision_model.pre_layrnorm.weight",
            bias_key="vision_model.pre_layrnorm.bias",
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=config.dtype,
        )
        self.post_layernorm = _loaded_layer_norm(
            weights,
            parameter_prefix="post_layernorm",
            weight_key="vision_model.post_layernorm.weight",
            bias_key="vision_model.post_layernorm.bias",
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=config.dtype,
        )
        self.visual_projection = _loaded_linear(
            weights,
            parameter_prefix="visual_projection",
            weight_key="visual_projection.weight",
            in_features=config.hidden_size,
            out_features=config.projection_dim,
            dtype=config.dtype,
        )

    def encode_vision(self, pixel_values):
        hidden_states = self.embeddings.encode_pixels(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        batch = _first_static_dim(hidden_states.shape)
        pooled_output = dml.ops.dynamic_slice(
            hidden_states,
            start_indices=(0, 0, 0),
            slice_sizes=(batch, 1, self.config.hidden_size),
        )
        pooled_output = dml.ops.squeeze(pooled_output, 1)
        pooled_output = self.post_layernorm(pooled_output)
        image_features = self.visual_projection(pooled_output)
        return hidden_states, pooled_output, image_features

    def forward(self, pixel_values):
        last_hidden_state, pooler_output, image_features = self.encode_vision(pixel_values)
        return (
            dml.ops.output(last_hidden_state, "last_hidden_state"),
            dml.ops.output(pooler_output, "pooler_output"),
            dml.ops.output(image_features, "image_features"),
        )


class LegacyCLIPModel(dml.nn.Module):
    """Bounded CLIP two-tower wrapper composed from the admitted tower slices."""

    def __init__(
        self,
        text_config: LegacyCLIPTextConfig,
        vision_config: LegacyCLIPVisionConfig,
        weights: Mapping[str, np.ndarray],
    ):
        if text_config.projection_dim != vision_config.projection_dim:
            raise ValueError("text and vision projection_dim must match")
        if text_config.dtype != vision_config.dtype:
            raise ValueError("text and vision dtype must match")
        dtype = text_config.dtype
        numpy_dtype = _numpy_clip_dtype(dtype)
        self.text_model = LegacyCLIPTextModelWithProjection(text_config, weights)
        self.vision_model = LegacyCLIPVisionModelWithProjection(vision_config, weights)
        self.logit_scale = dml.Parameter(
            [1],
            dtype=dtype,
            value=np.asarray([_scalar_weight_value(weights, "logit_scale", dtype=dtype)], dtype=numpy_dtype),
        )

    def get_text_features(self, input_ids, attention_mask, position_ids=None):
        return self.text_model.encode_text(input_ids, attention_mask, position_ids)

    def get_image_features(self, pixel_values):
        _, _, image_features = self.vision_model.encode_vision(pixel_values)
        return image_features

    def _normalize_features(self, features):
        return dml.ops.div(features, dml.ops.vector_norm(features, dim=-1, keepdim=True))

    def forward(self, input_ids, pixel_values, attention_mask, position_ids=None):
        text_features = self.get_text_features(input_ids, attention_mask, position_ids)
        image_features = self.get_image_features(pixel_values)
        text_embeds = self._normalize_features(text_features)
        image_embeds = self._normalize_features(image_features)
        logits_per_text = dml.ops.gemm_rcr(text_embeds, image_embeds)
        logits_per_text = dml.ops.mul(logits_per_text, dml.ops.exp(self.logit_scale))
        logits_per_image = dml.ops.transpose(logits_per_text, 0, 1)
        return (
            dml.ops.output(logits_per_image, "logits_per_image"),
            dml.ops.output(logits_per_text, "logits_per_text"),
            dml.ops.output(text_embeds, "text_embeds"),
            dml.ops.output(image_embeds, "image_embeds"),
        )


def legacy_clip_configs_from_transformers_clip_config(
    clip_config: object,
    *,
    use_flash_attention: bool = False,
    assume_unpadded_attention_mask: bool = False,
    dtype: str = "float32",
) -> tuple[LegacyCLIPTextConfig, LegacyCLIPVisionConfig]:
    """Derive bounded LegacyCLIP configs from a Transformers CLIPConfig.

    This adapter intentionally admits only the already-supported inference
    surface: dense CLIP text/vision towers with `hidden_act="quick_gelu"` and
    the existing DinoML `LegacyCLIPModel` weight namespace.
    """

    text_config = getattr(clip_config, "text_config", None)
    vision_config = getattr(clip_config, "vision_config", None)
    projection_dim = getattr(clip_config, "projection_dim", None)
    if text_config is None or vision_config is None or projection_dim is None:
        raise TypeError("expected a Transformers CLIPConfig-like object with text_config, vision_config, and projection_dim")
    _validate_transformers_clip_hidden_act(text_config, tower="text")
    _validate_transformers_clip_hidden_act(vision_config, tower="vision")
    dtype = _normalize_clip_dtype(dtype)
    return (
        LegacyCLIPTextConfig(
            vocab_size=int(text_config.vocab_size),
            max_position_embeddings=int(text_config.max_position_embeddings),
            hidden_size=int(text_config.hidden_size),
            intermediate_size=int(text_config.intermediate_size),
            num_attention_heads=int(text_config.num_attention_heads),
            num_hidden_layers=int(text_config.num_hidden_layers),
            projection_dim=int(projection_dim),
            layer_norm_eps=float(text_config.layer_norm_eps),
            eos_token_id=int(text_config.eos_token_id),
            use_flash_attention=bool(use_flash_attention),
            assume_unpadded_attention_mask=bool(assume_unpadded_attention_mask),
            dtype=dtype,
        ),
        LegacyCLIPVisionConfig(
            hidden_size=int(vision_config.hidden_size),
            intermediate_size=int(vision_config.intermediate_size),
            num_attention_heads=int(vision_config.num_attention_heads),
            num_hidden_layers=int(vision_config.num_hidden_layers),
            projection_dim=int(projection_dim),
            image_size=int(vision_config.image_size),
            patch_size=int(vision_config.patch_size),
            num_channels=int(vision_config.num_channels),
            layer_norm_eps=float(vision_config.layer_norm_eps),
            use_flash_attention=bool(use_flash_attention),
            dtype=dtype,
        ),
    )


def legacy_clip_weights_from_transformers_state_dict(
    state_dict: Mapping[str, object],
    text_config: LegacyCLIPTextConfig,
    vision_config: LegacyCLIPVisionConfig,
    *,
    dtype: str | None = None,
) -> dict[str, np.ndarray]:
    """Convert a Transformers CLIPModel state dict into LegacyCLIP weights."""

    dtype = text_config.dtype if dtype is None else _normalize_clip_dtype(dtype)
    if text_config.dtype != vision_config.dtype:
        raise ValueError("text and vision dtype must match")
    required = _legacy_clip_required_weight_names(text_config, vision_config)
    missing = [name for name in required if name not in state_dict]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} missing total)"
        raise KeyError(f"Missing Transformers CLIP state_dict weights: {preview}{suffix}")
    return {name: _transformers_state_value_to_numpy(state_dict[name], name, dtype=dtype) for name in required}


def legacy_clip_model_from_transformers_clip_model(
    clip_model: object,
    *,
    use_flash_attention: bool = False,
    assume_unpadded_attention_mask: bool = False,
    dtype: str = "float32",
) -> LegacyCLIPModel:
    """Build a bounded DinoML LegacyCLIPModel from a local Transformers CLIPModel.

    The adapter is intentionally inference-only: it transfers config and
    checkpoint weights for the existing LegacyCLIP text/vision towers and
    contrastive head, but it does not add tokenizer/processor plumbing, loss,
    position interpolation, or other broader CLIP
    surfaces beyond the current LegacyCLIPModel contract.
    """

    clip_config = getattr(clip_model, "config", None)
    state_dict_fn = getattr(clip_model, "state_dict", None)
    if clip_config is None or state_dict_fn is None:
        raise TypeError("expected a Transformers CLIPModel-like object with config and state_dict()")
    text_config, vision_config = legacy_clip_configs_from_transformers_clip_config(
        clip_config,
        use_flash_attention=use_flash_attention,
        assume_unpadded_attention_mask=assume_unpadded_attention_mask,
        dtype=dtype,
    )
    weights = legacy_clip_weights_from_transformers_state_dict(
        state_dict_fn(),
        text_config,
        vision_config,
        dtype=dtype,
    )
    return LegacyCLIPModel(text_config, vision_config, weights)


def _weight_value(
    weights: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: str = "float32",
) -> np.ndarray:
    if name not in weights:
        raise KeyError(f"Missing CLIP weight: {name}")
    value = np.asarray(weights[name], dtype=_numpy_clip_dtype(dtype))
    if value.shape != shape:
        raise ValueError(f"Weight {name} has shape {value.shape}, expected {shape}")
    return value


def _scalar_weight_value(weights: Mapping[str, np.ndarray], name: str, *, dtype: str = "float32") -> np.generic:
    if name not in weights:
        raise KeyError(f"Missing CLIP weight: {name}")
    numpy_dtype = _numpy_clip_dtype(dtype)
    value = np.asarray(weights[name], dtype=numpy_dtype)
    if value.shape not in {(), (1,)}:
        raise ValueError(f"Weight {name} has shape {value.shape}, expected scalar or [1]")
    return np.asarray(value.reshape(-1)[0] if value.shape == (1,) else value, dtype=numpy_dtype).reshape(())[()]


def _first_static_dim(shape: list[int]) -> int:
    if not shape:
        raise ValueError("expected rank >= 1 tensor")
    batch = int(shape[0])
    if batch <= 0:
        raise ValueError("expected positive static batch dimension")
    return batch


def _sequence_length(shape: list[int]) -> int:
    if not shape:
        raise ValueError("expected rank >= 1 tensor")
    dim = int(shape[-1])
    if dim <= 0:
        raise ValueError("expected positive static sequence dimension")
    return dim


def _hidden_sequence_length(shape: list[int]) -> int:
    if len(shape) < 2:
        raise ValueError("expected rank >= 2 tensor")
    dim = int(shape[-2])
    if dim <= 0:
        raise ValueError("expected positive static sequence dimension")
    return dim


def _nchw_image_shape(shape: list[int]) -> tuple[int, int, int, int]:
    if len(shape) != 4:
        raise ValueError("expected rank-4 NCHW image tensor")
    batch, channels, height, width = (int(dim) for dim in shape)
    if batch <= 0:
        raise ValueError("expected positive static batch dimension")
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("expected positive static NCHW image dimensions")
    return batch, channels, height, width


def _validate_transformers_clip_hidden_act(config: object, *, tower: str) -> None:
    hidden_act = getattr(config, "hidden_act", None)
    if hidden_act != "quick_gelu":
        raise ValueError(
            f"LegacyCLIP {tower} adapter only supports Transformers hidden_act='quick_gelu', got {hidden_act!r}"
        )


def _legacy_clip_required_weight_names(
    text_config: LegacyCLIPTextConfig,
    vision_config: LegacyCLIPVisionConfig,
) -> list[str]:
    names = [
        "text_model.embeddings.token_embedding.weight",
        "text_model.embeddings.position_embedding.weight",
        "text_model.final_layer_norm.weight",
        "text_model.final_layer_norm.bias",
        "text_projection.weight",
        "vision_model.embeddings.class_embedding",
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.embeddings.position_embedding.weight",
        "vision_model.pre_layrnorm.weight",
        "vision_model.pre_layrnorm.bias",
        "vision_model.post_layernorm.weight",
        "vision_model.post_layernorm.bias",
        "visual_projection.weight",
        "logit_scale",
    ]
    for layer_idx in range(text_config.num_hidden_layers):
        prefix = f"text_model.encoder.layers.{layer_idx}"
        names.extend(
            [
                f"{prefix}.layer_norm1.weight",
                f"{prefix}.layer_norm1.bias",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.out_proj.weight",
                f"{prefix}.self_attn.out_proj.bias",
                f"{prefix}.layer_norm2.weight",
                f"{prefix}.layer_norm2.bias",
                f"{prefix}.mlp.fc1.weight",
                f"{prefix}.mlp.fc1.bias",
                f"{prefix}.mlp.fc2.weight",
                f"{prefix}.mlp.fc2.bias",
            ]
        )
    for layer_idx in range(vision_config.num_hidden_layers):
        prefix = f"vision_model.encoder.layers.{layer_idx}"
        names.extend(
            [
                f"{prefix}.layer_norm1.weight",
                f"{prefix}.layer_norm1.bias",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.out_proj.weight",
                f"{prefix}.self_attn.out_proj.bias",
                f"{prefix}.layer_norm2.weight",
                f"{prefix}.layer_norm2.bias",
                f"{prefix}.mlp.fc1.weight",
                f"{prefix}.mlp.fc1.bias",
                f"{prefix}.mlp.fc2.weight",
                f"{prefix}.mlp.fc2.bias",
            ]
        )
    return names


def _transformers_state_value_to_numpy(value: object, name: str, *, dtype: str = "float32") -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=_numpy_clip_dtype(dtype))
    if array.size == 0:
        raise ValueError(f"Transformers CLIP weight {name} is empty")
    return array
