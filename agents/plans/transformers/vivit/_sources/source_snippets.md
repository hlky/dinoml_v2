# Vivit source snippets

Source basis: `X:/H/transformers`, commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

## Files inspected

- `src/transformers/models/vivit/configuration_vivit.py`
- `src/transformers/models/vivit/modeling_vivit.py`
- `src/transformers/models/vivit/modular_vivit.py`
- `src/transformers/models/vivit/image_processing_vivit.py`
- `src/transformers/models/vivit/convert_vivit_flax_to_pytorch.py`
- `src/transformers/masking_utils.py` for `create_bidirectional_mask`

## Config defaults

```python
model_type = "vivit"
image_size = 224
num_frames = 32
tubelet_size = (2, 16, 16)
num_channels = 3
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
intermediate_size = 3072
hidden_act = "gelu_fast"
hidden_dropout_prob = 0.0
attention_probs_dropout_prob = 0.0
layer_norm_eps = 1e-6
qkv_bias = True
pooler_output_size = None  # post-init -> hidden_size
pooler_act = "tanh"
```

## Tubelet embedding

```python
self.num_patches = (
    (config.num_frames // tubelet_size[0])
    * (image_size[0] // tubelet_size[1])
    * (image_size[1] // tubelet_size[2])
)
self.projection = nn.Conv3d(
    config.num_channels, config.hidden_size, kernel_size=tubelet_size, stride=tubelet_size
)

# forward
pixel_values = pixel_values.transpose(1, 2)  # B,T,C,H,W -> B,C,T,H,W
hidden = self.projection(pixel_values).flatten(2).transpose(1, 2)
```

For defaults, `num_patches = (32 // 2) * (224 // 16) * (224 // 16) = 3136`; with CLS the encoder sequence length is `3137`.

## Embedding and positional interpolation

```python
cls_tokens = self.cls_token.expand(batch_size, -1, -1)
embeddings = torch.cat((cls_tokens, embeddings), dim=1)

if interpolate_pos_encoding:
    embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
else:
    if height != self.image_size[0] or width != self.image_size[1]:
        raise ValueError(...)
    embeddings = embeddings + self.position_embeddings
```

Spatial interpolation reshapes patch position embeddings as a square 2D grid only:

```python
sqrt_num_positions = torch_int(num_positions**0.5)
patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
patch_pos_embed = nn.functional.interpolate(
    patch_pos_embed,
    size=(height // patch_h, width // patch_w),
    mode="bicubic",
    align_corners=False,
)
patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
```

The current implementation does not interpolate the temporal patch axis separately.

## Attention

```python
self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=config.qkv_bias)
self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=config.qkv_bias)
self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=config.qkv_bias)
self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=True)
self.is_causal = False

query_states = self.q_proj(hidden_states).view(*input_shape, -1, head_dim).transpose(1, 2)
key_states = self.k_proj(hidden_states).view(*input_shape, -1, head_dim).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(*input_shape, -1, head_dim).transpose(1, 2)
```

Eager attention uses matmul, optional additive mask, fp32 softmax over `dim=-1`, dropout, value matmul, transpose back, and output projection. `VivitPreTrainedModel` advertises SDPA, FlashAttention, FlexAttention, and generic attention backend support.

## Encoder and heads

```python
for layer in self.layers:
    hidden_states = layer(hidden_states, attention_mask, **kwargs)
sequence_output = self.layernorm(hidden_states)
pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

# classification
logits = self.classifier(sequence_output[:, 0, :])
```

Each layer is pre-norm self-attention plus residual, then pre-norm MLP plus residual. The optional base-model pooler takes `hidden_states[:, 0] -> Linear(hidden_size, pooler_output_size) -> pooler_act`.

## Image/video processor

The processor accepts one video as a list of frames or a batch as list of lists. It applies per-frame resize, center crop, rescale, normalize, and channel formatting. Default current class values:

```python
size = {"shortest_edge": 256}
crop_size = {"height": 224, "width": 224}
rescale_factor = 1 / 127.5
offset = True       # image * scale - 1
do_normalize = True # then normalize by ImageNet standard mean/std unless overridden
data_format = ChannelDimension.FIRST
```

Official processor configs override some of these defaults; see `hf_config_summaries.md`.
