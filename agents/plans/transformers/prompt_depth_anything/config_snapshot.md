# prompt_depth_anything config/source snapshot

Source basis: Transformers checkout `transformers` at
`b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

## Representative HF configs

| repo | status | source facts |
|---|---:|---|
| `depth-anything/prompt-depth-anything-vits-hf` | open | `model_type=prompt_depth_anything`, `depth_estimation_type=metric`, `max_depth=1`, `patch_size=14`, Dinov2 backbone `hidden_size=384`, effective `num_hidden_layers=12` from Dinov2 default because omitted, `num_attention_heads=6`, `out_indices=[3,6,9,12]`, `fusion_hidden_size=64`, `neck_hidden_sizes=[48,96,192,384]`, `torch_dtype=float32`. |
| `depth-anything/prompt-depth-anything-vits-transparent-hf` | open | Same operator-significant config as `vits-hf`; model-card metadata marks it as a transparent-image variant. |
| `depth-anything/prompt-depth-anything-vitl-hf` | open | `depth_estimation_type=metric`, `max_depth=1`, `patch_size=14`, Dinov2 backbone `hidden_size=1024`, `num_hidden_layers=24`, `num_attention_heads=16`, `out_indices=[5,12,18,24]`, `fusion_hidden_size=256`, `neck_hidden_sizes=[256,512,1024,1024]`, `torch_dtype=float32`. |
| `depth-anything/prompt-depth-anything-vitb-hf` | unavailable | HF lookup returned not found. The converter has a source preset for `vitb/base`, but no matching HF-native `*-hf` repo was found in this audit. |

All open HF-native processors inspected use:

```json
{
  "image_processor_type": "PromptDepthAnythingImageProcessor",
  "do_resize": true,
  "size": {"height": 756, "width": 756},
  "keep_aspect_ratio": true,
  "ensure_multiple_of": 14,
  "do_rescale": true,
  "rescale_factor": 0.00392156862745098,
  "do_normalize": true,
  "image_mean": [0.485, 0.456, 0.406],
  "image_std": [0.229, 0.224, 0.225],
  "do_pad": false,
  "prompt_scale_to_meter": 0.001
}
```

## Source-derived shape skeleton

For `B` images with preprocessed `pixel_values` `[B,3,H,W]`, `H` and `W` are multiples of `14`.

```text
patch_h = H // 14
patch_w = W // 14
Dinov2 patch tokens: [B, patch_h * patch_w + 1, C]
PromptDepthAnything reassemble removes CLS and reshapes:
  [B, patch_h * patch_w, C] -> [B, patch_h, patch_w, C] -> [B,C,patch_h,patch_w]
```

Reassemble factors are `[4,2,1,0.5]`:

```text
factor 4: Conv1x1 C->neck[0], ConvTranspose2d kernel=stride=4
factor 2: Conv1x1 C->neck[1], ConvTranspose2d kernel=stride=2
factor 1: Conv1x1 C->neck[2], Identity
factor 0.5: Conv1x1 C->neck[3], Conv2d kernel=3 stride=2 padding=1
```

Fusion reverses the four feature maps, repeatedly applies residual conv units,
optional prompt-depth conv injection, bilinear upsampling, and a final `1x1`
projection. The depth head upsamples to `[patch_h * 14, patch_w * 14]`, applies
two convolutions plus activation, then squeezes channel dimension to `[B,H,W]`.
