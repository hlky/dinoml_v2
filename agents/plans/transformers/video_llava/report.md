# Video-LLaVA (`video_llava`) Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: LanguageBind/Video-LLaVA-7B-hf
Config source: HF config/preprocessor/tokenizer snapshots under _sources/
Source files inspected:
- transformers/src/transformers/models/video_llava/modeling_video_llava.py
- transformers/src/transformers/models/video_llava/configuration_video_llava.py
- transformers/src/transformers/models/video_llava/processing_video_llava.py
- transformers/src/transformers/models/video_llava/image_processing_video_llava.py
- transformers/src/transformers/models/video_llava/video_processing_video_llava.py
- transformers/src/transformers/models/video_llava/convert_video_llava_weights_to_hf.py
- transformers/src/transformers/models/llama/modeling_llama.py and configuration_llama.py for delegated decoder behavior
- transformers/src/transformers/models/clip/modeling_clip.py and configuration_clip.py for delegated vision-tower behavior
- transformers/src/transformers/models/llava_next_video/* for comparison only
Any missing files or assumptions: no native `video_llava` modular source file; `modeling_video_llava.py` is the source to audit. No gated/401/403 checkpoints were encountered.
```

HF snapshots saved:

- `agents/plans/transformers/video_llava/_sources/LanguageBind__Video-LLaVA-7B-hf/`
- `agents/plans/transformers/video_llava/_sources/Mantis-VL__videollava-7b-video-eval-20k_2048/`
- `agents/plans/transformers/video_llava/_sources/Mantis-VL__videollava-7b-video-eval-95k_2048/`
- `agents/plans/transformers/video_llava/_sources/LanguageBind__Video-LLaVA-7B/` is a legacy non-native comparison snapshot with `model_type: llava`, not in scope for native `video_llava`.

Primary HF links:

- [LanguageBind/Video-LLaVA-7B-hf](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf)
- [Mantis-VL/videollava-7b-video-eval-20k_2048](https://huggingface.co/Mantis-VL/videollava-7b-video-eval-20k_2048)
- [Mantis-VL/videollava-7b-video-eval-95k_2048](https://huggingface.co/Mantis-VL/videollava-7b-video-eval-95k_2048)

Primary runtime target: multimodal autoregressive generation with image and/or video prompt embeddings stitched into a LLaMA/Vicuna-style decoder prefill, followed by text decode with delegated decoder KV cache.

## 2. High-level architecture

Video-LLaVA is a composite VLM:

```text
CPU/image-video preprocessing
  -> separate CLIP-like image_tower / video_tower
  -> shared 2-layer multimodal projector
  -> masked_scatter replacement of repeated <image>/<video> placeholders in text embeddings
  -> LLaMA/Vicuna decoder prefill
  -> cached autoregressive decode
  -> lm_head logits
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize shortest edge to 224, center crop 224x224, rescale by 1/255, CLIP mean/std normalization, channels-first packing, prompt placeholder expansion.
- Vision/projector stage: image inputs run through `image_tower`; video frames are reshaped from `[B, F, C, H, W]` to `[B*F, C, H, W]` and run through `video_tower`. Both use the same projector module.
- Prefix construction: text token embeddings are produced first, then image/video embeddings replace special-token embedding slots via boolean masks and `masked_scatter`.
- Prefill: full stitched sequence enters the delegated language model.
- Decode: `prepare_inputs_for_generation` forwards pixel tensors only on the first generation iteration, or when `use_cache=False`; after that, visual embeddings are assumed to be in the decoder KV cache.

Independently cacheable pieces:

- Image/video encoder + projector outputs can be precomputed for an unchanged prompt prefix.
- Decoder KV cache owns the stitched multimodal prefix after prefill. There is no separate Video-LLaVA-specific cache object beyond delegated LLaMA `Cache`.

## 3. Important config dimensions

Effective dimensions for the official native checkpoint. Fields marked "default" are omitted in `config.json` and filled by the pinned source config classes.

| Field | LanguageBind/Video-LLaVA-7B-hf |
|---|---:|
| top model_type | `video_llava` |
| architecture | `VideoLlavaForConditionalGeneration` |
| top torch_dtype | `bfloat16` in config metadata |
| text model_type | `llama` |
| text `_name_or_path` | `lmsys/vicuna-7b-v1.5` |
| text hidden_size | 4096 default |
| text layers | 32 default |
| text attention heads | 32 default |
| text KV heads | 32 default, MHA not GQA |
| text head_dim | 128 default |
| text intermediate_size | 11008 default |
| text max_position_embeddings | 4096 from config |
| text vocab_size | 32064 from config |
| text activation | `silu` default gated LLaMA MLP |
| text RMSNorm eps | `1e-5` from config |
| text use_cache | `True` default |
| vision model_type | `clip_vision_model` |
| vision hidden_size | 1024 |
| vision layers | 24 |
| vision attention heads | 16 |
| vision head_dim | 64 |
| vision MLP size | 4096 |
| vision patch_size | 14 |
| effective image size | 224 default/source processor |
| vision patches at 224 | 16 x 16 = 256 |
| image placeholder count | 256 with `default` selection |
| video placeholder count for 8 frames | `(256 + CLS) * 8 = 2056` |
| projector | Linear(1024 -> 4096) + GELU + Linear(4096 -> 4096), bias true |
| image_token_id / video_token_id / pad_token_id | 32000 / 32001 / 32002 |

Representative checkpoint sweep:

| Checkpoint/config | Scope | Structural notes |
|---|---|---|
| source default `VideoLlavaConfig()` | debug/source default | CLIP-ViT-L/14-like 224 vision tower, default LLaMA 7B-like decoder, `image_seq_length=256`, `video_seq_length=2056`. |
| Transformers unit-test micro config | debug/test only | 2-layer LLaMA and 2-layer CLIP configs; validates image/video placeholder mismatch errors and multi-layer vision feature concat. |
| `LanguageBind/Video-LLaVA-7B-hf` | common production native checkpoint | Native `video_llava`; official HF snapshot. Config omits many LLaMA defaults but source fills 4096/32/32/11008. |
| `Mantis-VL/videollava-7b-video-eval-20k_2048` | finetuned native checkpoint | Same native architecture/config dimensions as official snapshot; generated-from-trainer finetune of `LanguageBind/Video-LLaVA-7B-hf`. |
| `Mantis-VL/videollava-7b-video-eval-95k_2048` | finetuned native checkpoint | Same native architecture/config dimensions as official snapshot. |
| `LanguageBind/Video-LLaVA-7B` | out of native scope | `model_type: llava`, architecture `LlavaLlamaForCausalLM`, legacy/original-style keys. Route to a separate LLaVA/remote-code conversion audit. |

## 3a. Family variation traps

- Video tokens keep the CLIP CLS token. Image tokens drop CLS when `vision_feature_select_strategy == "default"`, but video features do not. This is the central Video-LLaVA packing trap.
- `get_video_features` accepts `vision_feature_layer` but not `vision_feature_select_strategy`; it never crops CLS in the pinned source.
- Prompt expansion mirrors that asymmetry: image count is `(H/patch)*(W/patch) + additional - 1` for default; video count is `((H/patch)*(W/patch) + additional) * num_frames`.
- Image and video towers are separate modules with separate weights, even though both use `config.vision_config` and feed the same projector.
- Multi-layer vision feature selection concatenates selected hidden states along the channel dimension, so projector `linear_1.in_features = vision_hidden_size * len(vision_feature_layer)`.
- The official config omits `image_seq_length` and `video_seq_length`; source defaults still make them 256 and 2056, but runtime shape checks use actual placeholder count and feature tensor size, not these fields.
- The native source delegates decoder internals to `AutoModel.from_config(text_config)`. DinoML should audit/load the delegated LLaMA/Vicuna decoder as a LLaMA family implementation, not as Video-LLaVA-specific decoder code.
- The config claims `_supports_flash_attn` and `_supports_sdpa` through the composite model; actual attention math and cache layout are those of the nested CLIP/LLaMA models.
- NCHW is semantic source layout for processor output and CLIP Conv2d patch embedding. NHWC/channel-last is only an optimization region around the vision tower and must rewrite Conv2d/flatten/transpose axes.
- LLaVA-NeXT-Video is not a drop-in variant. It has any-resolution image packing, `image_newline`, `image_grid_pinpoints`, a single `vision_tower`, video spatial pooling, and video token count divided by 4 for default average pooling. None of those apply to native `video_llava`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Reshape video frames `[B, F, C, H, W] -> [B*F, C, H, W]`.
- CLIP patch Conv2d output flatten/transpose `[B, 1024, 16, 16] -> [B, 256, 1024]`.
- Concatenate CLIP class token with patch embeddings.
- Hidden-state slicing for image default selection: `[:, 1:]`.
- Concatenate selected vision layers on `dim=-1`.
- Boolean placeholder masks from `input_ids == image_token_id/video_token_id`.
- `unsqueeze(-1)`, `expand_as(inputs_embeds)`, dtype/device conversion.
- `masked_scatter` for image/video embedding stitch.
- Logits slicing via `logits_to_keep`: `hidden_states[:, slice_indices, :]`.

Neural network primitives:

- CLIP Conv2d patch embedding: `Conv2d(3 -> 1024, kernel=14, stride=14, bias=False)`.
- CLIP learned class embedding and learned absolute position embedding.
- CLIP encoder block repeated 24 times: LayerNorm, MHA, residual, LayerNorm, MLP Linear(1024 -> 4096) + quick_gelu + Linear(4096 -> 1024).
- Multimodal projector: Linear(1024 or 1024*n_layers -> 4096) + GELU + Linear(4096 -> 4096), bias by `multimodal_projector_bias`.
- LLaMA decoder block repeated 32 times: RMSNorm, causal MHA, residual, RMSNorm, gated MLP, residual.
- LM head: Linear(4096 -> 32064), bias false.

Attention primitives:

- CLIP noncausal self-attention: MHA, 16 heads, head_dim 64, no cache.
- LLaMA causal self-attention: MHA, 32 Q heads, 32 KV heads, head_dim 128, RoPE before cache update, delegated `Cache`.
- SDPA/FlashAttention-compatible backend dispatch via Transformers attention interface for the nested models.

Position/rotary ops:

- CLIP learned absolute position embedding; possible interpolation only if `interpolate_pos_encoding=True`, which Video-LLaVA does not pass in normal forward.
- LLaMA RoPE from delegated decoder; default source config has `rope_parameters=None`, standard RoPE, `head_dim=128`.

Generation/cache ops:

- `Cache.update(key_states, value_states, layer_idx)` in LLaMA attention after RoPE.
- First generation iteration includes pixel tensors; later decode omits them when cache is enabled.
- `logits_to_keep` support for last-token-only or indexed logits.

Preprocessing-coupled ops:

- RGB conversion, resize, center crop, rescale, normalize, channels-first output.
- Prompt string replacement expanding `<image>` and `<video>` into repeated special tokens before tokenization.
- `_check_special_mm_tokens` parity check from `ProcessorMixin`.

Scatter/indexed update ops for multimodal embedding stitch:

- Required for first parity: boolean mask/scatter with exact flatten order matching PyTorch `masked_scatter`.
- Failure path must reject mismatched placeholder count vs feature element count.

Quantized/packed weight metadata ops:

- None in native source. BitsAndBytes examples are loading-time options, not architecture requirements.

Parameter sharing / tied weights:

- Top-level `_tied_weights_keys` maps `lm_head.weight` to `model.language_model.embed_tokens.weight`. Official safetensors snapshot contains `language_model.lm_head.weight` and `language_model.model.embed_tokens.weight`; DinoML should preserve configured tying semantics when applying HF load/tie rules.

## 5. Layer/block breakdown

Image path:

```text
pixel_values_images: [B_img, 3, 224, 224]
image_outputs = image_tower(pixel_values_images, output_hidden_states=True)
selected = hidden_states[layer]                         # [B_img, 257, 1024]
if image default: selected = selected[:, 1:]             # [B_img, 256, 1024]
if layer list: concat selected layers on last dim
image_features = projector(selected)                     # [B_img, 256, 4096]
inputs_embeds = masked_scatter(image_mask, image_features)
```

Video path:

```text
pixel_values_videos: [B_vid, F, 3, 224, 224]
frames = reshape(pixel_values_videos, [B_vid * F, 3, 224, 224])
video_outputs = video_tower(frames, output_hidden_states=True)
selected = hidden_states[layer]                         # [B_vid * F, 257, 1024]
if layer list: concat selected layers on last dim
# no CLS crop for video
video_features = projector(selected)                    # [B_vid * F, 257, 4096]
inputs_embeds = masked_scatter(video_mask, video_features)
```

Projector:

```text
x = Linear(vision_hidden * num_feature_layers -> text_hidden, bias=config.multimodal_projector_bias)(features)
x = GELU(x)
x = Linear(text_hidden -> text_hidden, bias=config.multimodal_projector_bias)(x)
```

Delegated LLaMA decoder block:

```text
residual = x
x = RMSNorm(x)
q = Linear(4096 -> 4096, bias=False)(x).view(B, S, 32, 128).transpose(1, 2)
k = Linear(4096 -> 4096, bias=False)(x).view(B, S, 32, 128).transpose(1, 2)
v = Linear(4096 -> 4096, bias=False)(x).view(B, S, 32, 128).transpose(1, 2)
q, k = RoPE(q, k, cos, sin)
k, v = cache.update(k, v, layer_idx) if cache is present
a = causal_attention(q, k, v, mask, scale=128**-0.5)
x = residual + Linear(4096 -> 4096, bias=False)(a)
residual = x
x = RMSNorm(x)
x = residual + down_proj(silu(gate_proj(x)) * up_proj(x))
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention in CLIP image/video towers.
- Source shape after projections: `[B_img or B_vid*F, heads=16, tokens=257, head_dim=64]`.
- Attention mask normally absent for fixed full-image/video-frame encoding.
- No KV cache.
- Backend can use eager/SDPA/FlashAttention through Transformers attention interface if supported.

Decoder attention:

- Causal self-attention only; no cross-attention. Visual context is embedded into the token sequence before decoder entry.
- MHA for official configs: `num_key_value_heads == num_attention_heads == 32`.
- Cache tensor per layer, before repeat expansion: keys and values `[batch, 32, cached_seq, 128]`.
- Cached keys are stored after RoPE because `apply_rotary_pos_emb` is called before `past_key_values.update`.
- Eager math order: `q @ k.transpose * scale`, add causal mask, softmax in fp32, cast to query dtype, dropout, `attn @ v`, output projection.
- FlashAttention/SDPA parity must preserve RoPE-before-cache and mask semantics from delegated LLaMA.

There is no Video-LLaVA-specific cross-modal attention. The multimodal interaction happens only after visual embeddings are stitched into the decoder prefix.

## 7. Position encoding and custom math

CLIP position math:

```python
patch_embeds = conv2d(pixel_values).flatten(2).transpose(1, 2)
embeddings = torch.cat([class_embedding.expand(batch, 1, -1), patch_embeds], dim=1)
embeddings = embeddings + position_embedding(position_ids)
```

At the default 224x224 resolution, CLIP uses learned positions for 257 tokens. Position interpolation exists in CLIP source but Video-LLaVA does not request it in its calls.

LLaMA RoPE:

```python
cos, sin = rotary_emb(hidden_states, position_ids)
q, k = apply_rotary_pos_emb(q, k, cos, sin)
key_states, value_states = cache.update(key_states, value_states, layer_idx)
```

Precompute opportunities:

- CLIP position IDs and 224x224 position embeddings are static.
- LLaMA RoPE cos/sin can be cached by sequence positions and dtype/device, but position IDs shift by past cache length during decode.

## 8. Preprocessing and input packing

Image processor contract:

- Accepts PIL/numpy/torch images.
- Converts to RGB by default.
- Resizes shortest edge to 224 preserving aspect ratio, then center crops 224x224.
- Rescales by `1/255`.
- Normalizes with OpenAI CLIP mean `[0.48145466, 0.4578275, 0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`.
- Emits `pixel_values_images` in channels-first format `[num_images, 3, 224, 224]`.

Video processor contract:

- Uses `BaseVideoProcessor` with the same CLIP resize/crop/rescale/normalize defaults.
- `do_sample_frames=False` in the class for backward compatibility; examples sample frames outside the processor.
- Emits `pixel_values_videos` shaped `[num_videos, num_frames, 3, 224, 224]`.

Prompt expansion:

```python
image_tokens = (height // patch_size) * (width // patch_size) + num_additional_image_tokens
if vision_feature_select_strategy == "default":
    image_tokens -= 1

video_frame_tokens = (height // patch_size) * (width // patch_size) + num_additional_image_tokens
video_tokens = video_frame_tokens * num_frames
```

For official 224x224, patch 14, `num_additional_image_tokens=1`:

- `<image>` expands to 256 token IDs under default feature selection.
- `<video>` expands to `257 * num_frames`; the common 8-frame path expands to 2056 token IDs.

Stitching:

- `input_ids == 32000` marks image slots.
- `input_ids == 32001` marks video slots.
- Masks expand over hidden dimension and `masked_scatter` consumes flattened feature values in source tensor order.
- If caller supplies `inputs_embeds` instead of `input_ids`, masks are found by comparing embeddings to the special token embeddings. This path is fragile for compilers; first DinoML integration can require `input_ids`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fixed CLIP patch Conv2d -> patch GEMM

Source pattern:

```text
Conv2d(C=3 -> D=1024, kernel=14, stride=14, padding=0, bias=False)
-> flatten(2)
-> transpose(1, 2)
```

Replacement:

```text
WindowFlatten [B, 16, 16, 3*14*14] -> GEMM(W_flat.T) -> [B, 256, 1024]
```

Preconditions:

- `kernel_size == stride == patch_size`
- `padding == 0`, `dilation == 1`, `groups == 1`
- Input spatial dims divisible by 14 and equal to model image size unless CLIP interpolation path is explicitly implemented.
- Source semantic layout is NCHW; NHWC optimization must include a guarded layout rewrite.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
```

Failure cases:

- Non-224 images without `interpolate_pos_encoding` support.
- Any processor/config variant changing patch size, channel count, or input layout.

Parity test sketch:

- Compare Conv2d+flatten+transpose against WindowFlatten+GEMM for random fp32/fp16 tensors and real CLIP patch weights.

### Rewrite: placeholder masked_scatter -> indexed copy

Source pattern:

```text
mask = (input_ids == special_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(mask, features)
```

Replacement:

```text
positions = nonzero(input_ids == special_id)
copy rows features.reshape(num_slots, hidden) into inputs_embeds[positions]
```

Preconditions:

- `input_ids` path only for first integration.
- Number of special-token positions equals `features.numel() / hidden_size`.
- Placeholder order must be row-major batch/sequence order, matching PyTorch boolean masked scatter.

Failure cases:

- `inputs_embeds`-only special-token embedding comparison.
- Mixed prompts where placeholder counts do not match provided visual batches.

Parity test sketch:

- Random text embeddings, deterministic placeholder positions, random image/video features; compare exact output to PyTorch `masked_scatter`.

### Rewrite: cache visual prefix independently

Source pattern:

```text
image/video tower + projector + stitched prefill -> decoder KV cache
```

Replacement:

```text
precompute visual projected features, or precompute full multimodal prefix KV cache
```

Preconditions:

- Same visual inputs, same text prefix around placeholders, same position IDs and attention mask.
- Decoder cache ABI matches delegated LLaMA layout.

Failure cases:

- Changing prompt tokens before/around placeholders changes positions and invalidates prefix KV.

Parity test sketch:

- Compare generation logits for full prefill vs reused prefix cache plus continuation tokens.

### Layout optimization region: vision tower NCHW -> channel-last

Source pattern:

- Processor emits NCHW.
- CLIP patch embedding uses NCHW Conv2d.
- CLIP token sequence is layout-free after patch flatten.

Preconditions:

- Layout translation is local to Conv2d/patch extraction.
- Axis-sensitive rewrites: Conv2d channel axis `1`, spatial axes `2/3`, flatten over spatial grid, transpose from `[B, D, grid_h*grid_w]` to `[B, grid, D]`.

Failure cases:

- Do not translate LLaMA token sequence axes or placeholder masks as if they were image tensors.

## 10. Kernel fusion candidates

Highest priority:

- LLaMA decoder RMSNorm + QKV projection + RoPE + attention prefill/decode. This dominates after visual prefix construction.
- LLaMA SwiGLU MLP: `silu(gate_proj) * up_proj -> down_proj`.
- Placeholder indexed-copy stitch. It is small but correctness-critical and awkward as generic `masked_scatter`.
- CLIP LayerNorm + MHA/MLP blocks for batched frames; video inflates batch to `B*F`.

Medium priority:

- CLIP patch Conv2d lowered to GEMM for fixed 224x224 frames.
- Multimodal projector MLP fusion: Linear + GELU + Linear, batched over image/video tokens.
- Last-token-only logits through `logits_to_keep` to avoid full vocab projection over the prefill sequence during generation.

Lower priority:

- Full CPU/data-pipeline preprocessing in DinoML runtime. First integration can keep it in Python/processor.
- CLIP position embedding interpolation. Not used by default Video-LLaVA path.
- Multi-layer vision feature concat optimization. Supported by source but not observed in representative native configs.

## 11. Runtime staging plan

Stage 1: config/load admission

- Parse `VideoLlavaConfig`.
- Require native `model_type: video_llava`.
- Reject or route legacy `LanguageBind/Video-LLaVA-7B` style `model_type: llava`.
- Load separate image/video towers, shared projector, delegated LLaMA decoder, and LM head.

Stage 2: processor-compatible tensor contract

- Accept already-processed `pixel_values_images`, `pixel_values_videos`, `input_ids`, `attention_mask`.
- Stub full image/video decoding and frame sampling outside DinoML.
- Validate placeholder counts exactly.

Stage 3: vision/projector parity

- Run image_tower and video_tower independently.
- Validate image CLS crop and video no-crop behavior.
- Validate projector output for one image and one 8-frame video.

Stage 4: prefill parity

- Implement indexed-copy stitch and run full LLaMA prefill with visual embeddings.
- Compare logits for fixed prompts.

Stage 5: decode with KV cache

- Use delegated LLaMA cache ABI.
- Ensure pixel tensors are consumed only on first iteration when cache is enabled.

Stage 6: optimized kernels

- Enable optimized LLaMA attention/RMSNorm/SwiGLU and CLIP patch GEMM or Conv2d provider.
- Add visual-prefix and/or full-prefix cache reuse after parity is stable.

## 12. Parity and validation plan

Focused unit parity:

- Video frame reshape: `[B, F, C, H, W] -> [B*F, C, H, W]`.
- Image feature selection: `default` crops CLS and `full` keeps it.
- Video feature selection: default still keeps CLS in pinned source.
- Multi-layer vision feature concat changes projector input width.
- Placeholder count mismatch raises before decoder.
- Indexed-copy stitch equals PyTorch `masked_scatter`.

Component parity:

- CLIP image tower output hidden states for one image.
- CLIP video tower output hidden states for one video with 2 and 8 frames.
- Projector output vs PyTorch for image and video features.
- LLaMA one-block and full-decoder parity using stitched embeddings.

End-to-end parity:

- Official integration prompt with one 8-frame video, greedy decode.
- Mixed batch with one image prompt and one video prompt, left padding.
- Decode continuation with cache enabled vs full recompute.

Tolerances:

- fp32 component tests: `rtol=1e-4`, `atol=1e-5`.
- fp16/bf16 end-to-end logits: start with `rtol=5e-2`, `atol=5e-2`; tighten per backend once kernels are stable.
- Greedy token parity should be tested after logits parity for the first 1-5 decode steps.

## 13. Performance probes

- Processor throughput: images/sec and frames/sec for resize/crop/normalize/frame sampling.
- Vision encoder throughput: CLIP image-only, video-only with frame-count sweep `[1, 2, 4, 8, 16]`.
- Projector throughput by visual token count: 256 image tokens and `257*F` video tokens.
- Prefill sequence-length sweep including text-only, image prompt, 8-frame video prompt, mixed image/video.
- Decode tokens/sec with and without visual prefix cache reuse.
- KV cache memory by batch, prefix length, and decode length.
- `logits_to_keep` benchmark: full prefill logits vs last-token-only logits.
- Attention backend comparison: eager, SDPA, FlashAttention-compatible path for delegated LLaMA and CLIP.
- Patch embedding provider comparison: Conv2d vs lowered GEMM for fixed 224x224.

## 14. Skip/defer list

Safe to defer for first integration:

- Training, loss, gradients, gradient checkpointing.
- BitsAndBytes/4-bit loader behavior; treat as a separate quantized-loading contract.
- Full image/video decoding and frame sampling inside DinoML.
- `inputs_embeds`-only placeholder detection by embedding equality.
- CLIP position interpolation and arbitrary image resolutions.
- Multi-layer `vision_feature_layer` beyond config default, after projector-width admission is in place.
- Beam search and advanced generation controllers beyond greedy/sampling over delegated decoder logits.
- LLaVA-NeXT-Video/OneVision packing: anyres images, newline token, spatial video pooling, grid pinpoints.

Do not defer:

- Video CLS retention.
- Exact placeholder expansion and stitch order.
- Delegated LLaMA KV cache.
- Separate image and video tower weights.

## 15. Final implementation checklist

- [ ] Parse native `VideoLlavaConfig` and reject non-native `model_type: llava` checkpoints for this path.
- [ ] Load `image_tower`, `video_tower`, `multi_modal_projector`, delegated LLaMA, and `lm_head` with HF key mapping.
- [ ] Implement or compose CLIP vision tower operators: patch Conv2d/GEMM, class token concat, position embedding, LayerNorm, MHA, MLP.
- [ ] Implement projector MLP with configurable feature-layer concat input width.
- [ ] Implement image feature selection with default CLS crop.
- [ ] Implement video frame flattening and video feature selection with CLS retained.
- [ ] Implement placeholder count validation for image and video.
- [ ] Lower `masked_scatter` stitch as ordered indexed row copy.
- [ ] Compose delegated LLaMA decoder with RoPE-before-cache KV behavior.
- [ ] Support `prepare_inputs_for_generation` semantics: visual tensors first iteration only when cache is enabled.
- [ ] Add component parity tests for image tower, video tower, projector, stitch, and one decoder block.
- [ ] Add prefill logits parity for image-only, video-only, and mixed prompts.
- [ ] Add decode cache parity for greedy continuation.
- [ ] Benchmark processor, vision/projector, prefill, decode, and KV memory separately.
