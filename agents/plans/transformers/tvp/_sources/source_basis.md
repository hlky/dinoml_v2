# TVP source/config snapshots

Source basis:

- Transformers checkout: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Source directory: `src/transformers/models/tvp`
- Official HF repositories inspected:
  - `https://huggingface.co/Intel/tvp-base`
  - `https://huggingface.co/Intel/tvp-base-ANet`
  - `https://huggingface.co/Jiqing/tiny-random-tvp`

## Source snippets

### `configuration_tvp.py`

`TvpConfig` defaults include `model_type="tvp"`, `hidden_size=768`,
`intermediate_size=3072`, `num_hidden_layers=12`, `num_attention_heads=12`,
`max_position_embeddings=512`, `max_grid_col_position_embeddings=100`,
`max_grid_row_position_embeddings=100`, `max_img_size=448`, `num_frames=48`,
`visual_prompter_type="framepad"`, and `visual_prompter_apply="replace"`.

The config delegates visual feature extraction through `backbone_config`:

```python
self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
    backbone_config=self.backbone_config,
    default_config_type="resnet",
    default_config_kwargs={"out_features": ["stage4"]},
    **kwargs,
)
```

### `modeling_tvp.py`

The vision path flattens frames into the batch dimension, calls an AutoBackbone,
applies a 3x3 conv, max-pools spatially, then returns NHWC-like grid features:

```python
batch_size, num_frames, num_channels, height, width = pixel_values.shape
pixel_values = pixel_values.view(batch_size * num_frames, num_channels, height, width)
grid_feat_outputs = self.backbone(pixel_values)["feature_maps"][0]
grid = self.grid_encoder_conv(grid_feat_outputs)
grid = nn.functional.max_pool2d(grid, kernel_size=2, stride=2)
grid = nn.functional.relu(grid, inplace=True)
grid = grid.view(batch_size, num_frames, new_channel, new_height, new_width)
grid = grid.permute(0, 1, 3, 4, 2)
```

Visual embeddings mean-pool across frames before adding row/column position
embeddings:

```python
batch_size, num_frames, height, width, num_channels = grid.shape
grid = grid.mean(1)
grid = self.add_2d_positional_embeddings(grid, interpolate_pos_encoding=interpolate_pos_encoding)
visual_tokens = grid.view(batch_size, -1, num_channels)
```

The encoder attention is dense bidirectional MHA with separate Q/K/V projections:

```python
attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
attention_scores = attention_scores / math.sqrt(self.attention_head_size)
if attention_mask is not None:
    attention_scores = attention_scores + attention_mask
attention_probs = nn.functional.softmax(attention_scores, dim=-1)
attn_output = torch.matmul(attention_probs, value_layer)
```

`TvpModel.forward` constructs one concatenated sequence:

```python
visual_attention_mask = attention_mask.new_ones(visual_embedding_output.shape[:2])
pt_mask = torch.ones(attention_mask.shape[0], 10).to(device=attention_mask.device, dtype=attention_mask.dtype)
attention_mask = torch.cat([pt_mask, attention_mask, visual_attention_mask], dim=-1)
text_prompt = self.text_prompt.expand(text_embedding_output.shape[0], -1, -1)
embedding_output = torch.cat([text_prompt, text_embedding_output, visual_embedding_output], dim=1)
```

The video grounding head is a two-layer MLP:

```python
self.layer_0 = nn.Linear(config.hidden_size, config.hidden_size * 2)
self.layer_1 = nn.Linear(config.hidden_size * 2, 2)
logits = self.activation_0(self.layer_0(pooler_output))
logits = self.activation_1(self.layer_1(logits))
```

Potential guarded-source trap: `TvpFrameDownPadPrompter` reads
`config.frame_num`, but current `TvpConfig` and public configs define
`num_frames`.

### `processing_tvp.py`

`TvpProcessor` wraps an image/video processor plus tokenizer. Text defaults:
`truncation=True`, `padding="max_length"`, `pad_to_max_length=True`,
`return_token_type_ids=False`.

Postprocessing converts normalized logits to seconds:

```python
start = round(logits.tolist()[0][0] * video_durations, 1)
end = round(logits.tolist()[0][1] * video_durations, 1)
```

### `image_processing_tvp.py` and `image_processing_pil_tvp.py`

Default fast/PIL processor behavior in source:

- resize longest edge to 448 with aspect ratio kept
- optional center crop to 448x448
- rescale by 1/255 by default
- ImageNet mean/std normalization by default
- pad to 448x448 by default
- flip channel order RGB to BGR by default
- output key: `pixel_values`
- PyTorch tensor shape after batching nested videos: `[batch, frames, 3, height, width]`

Official checkpoint `preprocessor_config.json` differs from current source
defaults in important legacy fields: `do_center_crop=false`,
`do_rescale=false`, `do_padding=true`, `padding_size={"height":448,"width":448}`,
custom mean/std `[8.2381, 7.3115, 6.6981]` / `[9.6335, 9.0659, 8.7213]`,
and tokenizer `"bert-base-uncased"`.

## Representative config facts

| Model id | Hidden | Layers | Heads | FFN | Frames | Max text | Backbone |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Intel/tvp-base` | 768 | 12 | 12 | 3072 | 48 | 100 | ResNet bottleneck, hidden sizes `[256,512,1024,2048]`, `stage4` |
| `Intel/tvp-base-ANet` | 768 | 12 | 12 | 3072 | 64 | 100 | Same ResNet shape as `tvp-base` |
| `Jiqing/tiny-random-tvp` | 128 | 4 | 4 | 384 | 1 | 20 | Tiny ResNet, hidden sizes `[64,128]`, `stage2` |

All three configs advertise `use_cache=true`, but inspected source has no
cache object, cache arguments, cache updates, or generation path.
