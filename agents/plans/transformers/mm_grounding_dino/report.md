# Transformers audit: `mm_grounding_dino`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det, plus sweep below
Config source: public Hugging Face config/preprocessor/tokenizer JSON fetched 2026-05-13
Source files inspected:
  transformers/src/transformers/models/mm_grounding_dino/configuration_mm_grounding_dino.py
  transformers/src/transformers/models/mm_grounding_dino/modeling_mm_grounding_dino.py
  transformers/src/transformers/models/mm_grounding_dino/modular_mm_grounding_dino.py
  transformers/src/transformers/models/grounding_dino/processing_grounding_dino.py
  transformers/src/transformers/models/grounding_dino/image_processing_grounding_dino.py
  transformers/src/transformers/models/swin/modeling_swin.py and configuration_swin.py for consumed backbone contract
  transformers/src/transformers/models/bert/modeling_bert.py and configuration_bert.py for text backbone contract
Source URLs:
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mm_grounding_dino/modeling_mm_grounding_dino.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mm_grounding_dino/modular_mm_grounding_dino.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mm_grounding_dino/configuration_mm_grounding_dino.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/grounding_dino/processing_grounding_dino.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/grounding_dino/image_processing_grounding_dino.py
Representative HF config URLs:
  https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det/resolve/main/config.json
  https://huggingface.co/openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det/resolve/main/config.json
  https://huggingface.co/openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg/resolve/main/config.json
Any missing files or assumptions:
  modeling/configuration files are generated from modular_mm_grounding_dino.py; future HF edits should target the modular file.
  The audit owns the MM Grounding DINO detector/fusion/deformable-attention ABI. Swin and BERT internals should compose separate family audits.
  Sampled processor_config.json URLs returned 404, but preprocessor/tokenizer configs identify GroundingDinoProcessor.
```

Representative config sweep is recorded in `agents/plans/transformers/mm_grounding_dino/config_sweep.md`. Public sampled repos: `openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg`, `openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det`, `openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det`, `openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg`, and `iSEE-Laboratory/llmdet_tiny`.

## 2. High-level architecture

MM Grounding DINO is a prompt-conditioned, non-autoregressive zero-shot detector:

```text
CPU image resize/normalize/pad + BERT tokenization
-> Swin backbone feature maps + BERT text encoder
-> 1x1/3x3 vision projections + text projection
-> repeated text enhancement + bidirectional text-image fusion + multiscale deformable vision encoder
-> two-stage top-k proposal selection
-> learned object-query decoder with self-attn, text cross-attn, deformable vision cross-attn
-> contrastive text-conditioned class logits + iterative box refinement
-> threshold/box-scale/text-label postprocess
```

Stage boundaries: preprocessing/tokenization is CPU/data-pipeline work; Swin backbone and BERT text encoder are independently cacheable for fixed image/text prompts; the detector encoder fuses both modalities so it is not separable after projection; decoder is fixed-query detection, not generation and has no KV cache.

## 3. Important config dimensions

| Field | Common sampled value | Runtime effect |
| --- | --- | --- |
| `d_model` | 256 | Detector hidden width and deformable attention embed dim |
| `encoder_layers` / `decoder_layers` | 6 / 6 | Fusion/deformable encoder repeats and decoder repeats |
| `encoder_attention_heads` / `decoder_attention_heads` | 8 / 8 | Dense MHA heads; deformable heads |
| Fusion heads/dim | `encoder_attention_heads // 2 = 4`, `encoder_ffn_dim // 2 = 1024` | Bidirectional image-text attention uses head dim 256 |
| FFN dims | 2048 | ReLU MLPs in detector encoder/decoder |
| `num_queries` | 900 | Fixed decoder slots and top-k proposals |
| `num_feature_levels` | 4 for tiny/base/LLMDet tiny; 5 for large | Number of multiscale maps and deformable-attn levels |
| `encoder_n_points` / `decoder_n_points` | 4 / 4 | Samples per level per head in deformable attention |
| `max_text_len` | 256 | Contrastive logits padded to `[B,Q,256]` |
| Text backbone | BERT base, hidden 768, layers 12, heads 12 | Projected by Linear 768 -> 256 |
| Vision backbone | Swin-T/B/L variants | Produces NCHW feature maps selected by `out_features` |
| Processor image size | shortest edge 800, longest edge 1333, padded batch | Produces `pixel_values [B,3,H,W]`, `pixel_mask [B,H,W]` |

## 3a. Family variation traps

- Large checkpoints use `out_features=["stage1","stage2","stage3","stage4"]` and `num_feature_levels=5`; tiny/base use `stage2..stage4` and synthesize one extra stride-2 feature level.
- Swin window size and channel widths vary: tiny 7/768, base 12/1024, large 12/1536. Detector projection hides channel width after Conv2d -> 256, but backbone operator cost changes heavily.
- `disable_custom_kernels` is stored and passed into deformable attention, but the inspected source always calls the `MultiScaleDeformableAttention` module decorated with `use_kernel_forward_from_hub` and has an eager `grid_sample` implementation. DinoML should treat the custom kernel as an optional provider path, with eager grid-sample parity as the source fallback.
- Config flags for shared decoder heads exist in sampled configs but are ignored by native source; native MM Grounding DINO uses per-layer head modules and ties decoder references to those lists.
- Text masks invert polarity in different call sites. Keep explicit mask ABI names: processor `attention_mask` uses 1=real, model text padding masks often use True=masked after inversion, and deformable attention receives both polarities.
- Layout-sensitive regions: processor/model use NCHW images and Swin/Conv2d NCHW feature maps, then flatten feature maps with `flatten(2).transpose(1,2)`. A channel-last pass must rewrite Conv/GroupNorm/position embedding axes together and preserve downstream `[B, sum(H_l W_l), C]` order.

## 4. Operator coverage checklist

Tensor/layout ops: NCHW resize/rescale/normalize/pad; mask interpolation nearest-ish via float interpolate then bool; Conv2d 1x1 and 3x3 stride-2; GroupNorm(32); flatten/transpose/cat/stack/view/repeat/gather/topk; `torch.isin`, `cummax`, `cummin`, `cumsum`, `meshgrid`, `linspace`, `arange`; `masked_fill_`; `where`; `clamp`; `log`, `sigmoid`, `torch.special.logit(eps=1e-5)`.

Neural primitives: composed Swin backbone; composed BERT encoder; Linear 768 -> 256 text projection; detector Linear 256 -> 2048 -> 256 ReLU FFNs; detector MLP prediction heads 256 -> 256 -> 256 -> 4; LayerNorm eps 1e-5; FrozenBatchNorm2d replacement if backbone has BatchNorm2d; dropout is inference identity.

Attention primitives: dense noncausal MHA for text enhancer and decoder self/text cross-attn; bidirectional image-text attention using paired BMMs, max subtraction, clamp to [-50000, 50000], masks, softmax; multiscale deformable attention with grid-sample fallback.

Pre/postprocessing-coupled ops: BERT tokenization with merged lowercase labels `"label. label."`; special token IDs `[101,102,1012,1029]`; output logits `[B,900,256]`; boxes `[B,900,4]` normalized cxcywh; postprocess sigmoid, max over text positions, threshold, cxcywh -> xyxy, scale to target `(height,width)`, text-threshold phrase extraction. No source NMS.

## 5. Layer/block breakdown

Backbone/projection:

```text
pixel_values [B,3,H,W], pixel_mask [B,H,W]
SwinBackbone -> feature maps [(B,C_l,H_l,W_l)]
for selected levels: Conv2d(C_l -> 256, 1x1) -> GroupNorm(32)
for extra levels: Conv2d(256 or C_last -> 256, 3x3 stride=2 pad=1) -> GroupNorm(32)
position embedding per level -> flatten to [B,H_l*W_l,256] + level_embed
cat levels -> vision_features [B,Sv,256], mask_flatten [B,Sv]
```

Encoder layer, repeated 6:

```text
text_position = sinusoidal ids or supplied ids, width 256
vision,text = LayerNorm each
vision_delta,text_delta = bidirectional cross attention
vision += layer_scale_v * vision_delta
text += layer_scale_t * text_delta
text = dense self-attn over label-block mask + residual + LayerNorm
text = Linear(256->1024) + ReLU + Linear(1024->256) + residual + LayerNorm
vision = multiscale deformable self-attn + residual + LayerNorm
vision = Linear(256->2048) + ReLU + Linear(2048->256) + residual + LayerNorm
```

Decoder layer, repeated 6:

```text
query_pos = MLP(sinusoidal(reference_points), 512->256->256->256)
x = dense self-attn(q=k=x+query_pos, v=x) + residual + LayerNorm
x = dense text cross-attn(q=x+query_pos, k/v=text_encoder) + residual + LayerNorm
x = multiscale deformable cross-attn(q=x+query_pos, kv=vision_encoder, refs) + residual + LayerNorm
x = Linear(256->2048) + ReLU + Linear(2048->256) + residual + LayerNorm
reference_points = sigmoid(bbox_mlp(x) + logit(previous_reference))
```

Detection heads: per decoder layer `class_embed` computes scaled dot product between query hidden states and encoded text, adds learned scalar bias, masks padded text positions to `-inf`, then pads last dim to 256. Box head predicts deltas and combines with initial/intermediate reference points.

## 6. Attention requirements

Dense MHA is noncausal, batch-first, MHA only. Detector width is 256, heads 8 in decoder and deformable modules; text enhancer uses 4 heads with width 256; BERT text encoder uses its own 12-head self-attention and should compose the BERT audit. There is no autoregressive prefill/decode and no KV cache.

Deformable attention ABI:

```text
value: [B, sum_l(H_l*W_l), heads, head_dim]
spatial_shapes: int64 [L,2] as (H,W)
level_start_index: int64 [L]
reference_points encoder: [B,Sv,L,2]
reference_points decoder: [B,900,L,2 or 4] after valid-ratio scaling
sampling_offsets: [B,Q,heads,L,points,2]
attention_weights: softmax over L*points -> [B,Q,heads,L,points]
sampling_locations: normalized [0,1], converted to grid_sample coordinates by `2*x - 1`
grid_sample: bilinear, zeros padding, align_corners=False
```

Mask order matters: values are zeroed before deformable sampling where `attention_mask` is false; dense attention masks are additive `finfo(dtype).min` in some paths and `-inf` masked-fill in others.

## 7. Position encoding and custom math

Image sine position embedding:

```python
y = pixel_mask.cumsum(1) / (pixel_mask.cumsum(1)[:, -1:, :] + 1e-6) * 2*pi
x = pixel_mask.cumsum(2) / (pixel_mask.cumsum(2)[:, :, -1:] + 1e-6) * 2*pi
dim_t = temperature ** (2 * floor(arange(C/2)/2) / (C/2))
pos = concat([sin/cos(y/dim_t), sin/cos(x/dim_t)]).permute(0,3,1,2)
```

Reference-point sinusoidal encoding swaps x/y order for DETR convention and concatenates per-coordinate sin/cos embeddings. Encoder reference points are meshgrid centers normalized by valid ratios, then multiplied by valid ratios. Two-stage proposals use grid centers plus level-dependent width/height `0.05 * 2**level`, inverse-sigmoid via `log(p/(1-p))`, and invalid/padded proposals are filled with `inf`.

## 8. Preprocessing and input packing

`GroundingDinoProcessor` lowercases and merges candidate labels without periods into `"a cat. a dog."`; tokenizer is lower-case BERT with token type IDs returned. The model derives block-local text self-attention masks and position IDs from `[CLS]`, `[SEP]`, period, and question-mark IDs. Sequences longer than `max_text_len=256` are truncated inside model logic.

Image preprocessing returns NCHW `pixel_values` and `pixel_mask` with 1 for valid pixels. Default resize preserves aspect ratio with shortest edge 800 and longest edge 1333, rescales by 1/255, normalizes with ImageNet mean/std, and pads the batch to common H/W. `target_sizes` for postprocess are original `(height,width)`.

No multimodal placeholder embedding scatter is used. Text and image streams are encoded independently, then fused by attention.

Postprocess required for parity: `sigmoid(logits)`, per-query max over 256 text positions, keep `score > threshold`, convert boxes from normalized cxcywh to xyxy, scale by target width/height, then extract phrases from token positions where per-token probability exceeds `text_threshold`. No NMS appears in source postprocess.

## 9. Graph rewrite / lowering opportunities

### Rewrite: projection Conv2d 1x1 -> per-pixel Linear

Preconditions: source feature map is dense NCHW, `kernel_size=1`, stride 1, padding 0, groups 1, followed by GroupNorm over channels. Replacement: flatten spatial to `[B,H*W,Cin]`, GEMM with `weight.reshape(Cout,Cin).T`, bias add, reshape or keep sequence. Layout constraints: only safe if GroupNorm is lowered on the same channel axis or fused after projection. Failure cases: extra 3x3 stride-2 levels and Swin internal convs are not covered.

### Rewrite: extra-level Conv2d 3x3 stride-2 -> guarded im2col/GEMM

Preconditions: padding 1, dilation 1, groups 1, NCHW feature map, known or guarded H/W. Replacement: im2col over 3x3 windows -> GEMM -> GroupNorm. This is not a non-overlap patch rewrite; padding and overlap must be preserved.

### Rewrite: contrastive class head -> batched GEMM plus masked pad

Pattern: `[B,Q,256] @ [B,T,256]^T / sqrt(256) + scalar_bias`, masked fill over text padding, pad to 256. Replacement: BMM/GEMM with explicit `T <= max_text_len` guard and fixed output fill of `-inf`. Failure: arbitrary `max_text_len` changes require output width specialization.

### Rewrite: deformable attention provider

Pattern: offset/weight projections, sampling-location math, per-level `grid_sample`, weighted sum. Replacement: custom multiscale deformable attention provider with source-compatible `align_corners=False`, zero padding, level order, valid-ratio/reference-point equations, and optional eager fallback. Failure: treating it as dense attention or ordinary convolution is incorrect.

### Rewrite: text label mask generation to CPU

`isin/cummax/cummin/cumsum` over token IDs can be computed in preprocessing when `input_ids` are fixed for a request. Keep GPU implementation only for fully dynamic text.

## 10. Kernel fusion candidates

Highest priority: multiscale deformable attention provider; detector Linear+ReLU+Linear FFNs; LayerNorm; contrastive BMM + mask + pad; dense MHA/SDPA for decoder and text enhancer.

Medium priority: Swin backbone window attention and patch merging via separate Swin audit; BERT encoder via separate BERT audit; projection Conv/GroupNorm fusion; two-stage top-k/gather proposal path.

Lower priority: learned position embedding path, training loss/Hungarian matcher, output attentions materialization, dropout/drop-path training behavior.

## 11. Runtime staging plan

Stage 1: parse MM Grounding DINO config, load weights, and admit public configs with `model_type="mm-grounding-dino"`, BERT text config, and Swin backbone config allowlist.

Stage 2: compose existing or separately audited BERT/Swin encoders; validate projected feature map shapes and `spatial_shapes/level_start_index/valid_ratios`.

Stage 3: implement detector encoder one-layer parity: text enhancer, bidirectional fusion, deformable self-attn fallback.

Stage 4: implement two-stage proposal top-k and one decoder layer parity with iterative reference update.

Stage 5: end-to-end `MMGroundingDinoForObjectDetection` logits/boxes parity before postprocess.

Stage 6: add processor/postprocess parity and optimized deformable-attention/custom kernels.

## 12. Parity and validation plan

- Unit parity for `generate_masks_with_special_tokens_and_transfer_map` on prompts like `"a cat. a dog."`, punctuation, padding, and truncation at 256.
- Random tensor parity for `encode_sinusoidal_position_embedding`, image sine positions, proposal generation, contrastive embedding, and top-k/gather proposal selection.
- Deformable attention parity against eager `grid_sample` fallback for L=4/5, heads=8, points=4, rectangular H/W, and padding masks.
- Single encoder layer and decoder layer parity in fp32 with dropout disabled; then all 6 layers.
- End-to-end public tiny checkpoint parity on one image/prompt for logits `[1,900,256]`, boxes `[1,900,4]`, and postprocessed records. Suggested tolerances: fp32 atol/rtol around 1e-4 for module outputs; fp16/bf16 use looser attention/deformable tolerances and validate postprocess stability around thresholds separately.

## 13. Performance probes

Measure image preprocessing throughput, Swin-only throughput by backbone size, BERT-only text throughput by prompt length, detector encoder/decoder throughput as a function of `Sv`, deformable attention provider versus eager grid-sample, top-k/proposal overhead, postprocess threshold/phrase extraction overhead, and batch-size sweeps at typical padded 800x1333 shapes. Track memory for multiscale flattened vision sequence and deformable attention temporaries separately from backbone activations.

## 14. Skip/defer list

Defer training loss, Hungarian matcher, auxiliary-loss outputs for training, gradient checkpointing, dropout/drop-path randomness, panoptic/segmentation annotation preprocessing, output attentions, learned image position embeddings unless a checkpoint uses them, non-Swin backbones/timm backbones, remote-code variants, and any config with non-BERT text backbone until separately audited.

## 15. Final implementation checklist

- [ ] Parse `MMGroundingDinoConfig` and sampled OpenMMLab/LLMDet config variants.
- [ ] Add admission guards for Swin+BERT native-source combinations and ignored sharing flags.
- [ ] Compose or audit Swin backbone feature maps.
- [ ] Compose or audit BERT text encoder output.
- [ ] Implement image/text projection, level embedding, spatial metadata, and valid ratios.
- [ ] Implement text special-token mask and position-id generation.
- [ ] Implement bidirectional fusion attention with clamp/mask semantics.
- [ ] Implement multiscale deformable attention fallback and provider ABI.
- [ ] Implement two-stage proposal generation, top-k, and gather.
- [ ] Implement decoder self/text/deformable attention and iterative box refinement.
- [ ] Implement contrastive text-conditioned class head and bbox heads.
- [ ] Implement grounded detection postprocess with no NMS.
- [ ] Add unit, layer, full-model, and postprocess parity tests.
- [ ] Benchmark backbone, detector encoder/decoder, deformable attention, and postprocess separately.
