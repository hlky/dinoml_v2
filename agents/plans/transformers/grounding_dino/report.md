# Grounding DINO Transformers Family Audit

Primary target: open-vocabulary / text-conditioned object detection with `GroundingDinoForObjectDetection`. This report covers the native Transformers `grounding_dino` family, not `mm_grounding_dino`.

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary: IDEA-Research/grounding-dino-tiny.
  Sweep: IDEA-Research/grounding-dino-base,
  hf-internal-testing/tiny-random-GroundingDinoForObjectDetection.

Config source:
  https://huggingface.co/IDEA-Research/grounding-dino-tiny/raw/main/config.json
  https://huggingface.co/IDEA-Research/grounding-dino-tiny/raw/main/preprocessor_config.json
  https://huggingface.co/IDEA-Research/grounding-dino-tiny/raw/main/tokenizer_config.json
  https://huggingface.co/IDEA-Research/grounding-dino-base/raw/main/config.json
  https://huggingface.co/IDEA-Research/grounding-dino-base/raw/main/preprocessor_config.json
  https://huggingface.co/IDEA-Research/grounding-dino-base/raw/main/tokenizer_config.json
  https://huggingface.co/hf-internal-testing/tiny-random-GroundingDinoForObjectDetection/raw/main/config.json

Source files inspected:
  X:/H/transformers/src/transformers/models/grounding_dino/configuration_grounding_dino.py
  X:/H/transformers/src/transformers/models/grounding_dino/modeling_grounding_dino.py
  X:/H/transformers/src/transformers/models/grounding_dino/processing_grounding_dino.py
  X:/H/transformers/src/transformers/models/grounding_dino/image_processing_grounding_dino.py
  X:/H/transformers/src/transformers/models/grounding_dino/image_processing_pil_grounding_dino.py
  X:/H/transformers/src/transformers/models/grounding_dino/modular_grounding_dino.py
  X:/H/transformers/src/transformers/models/grounding_dino/convert_grounding_dino_to_hf.py
  X:/H/transformers/tests/models/grounding_dino/test_modeling_grounding_dino.py
  X:/H/transformers/tests/models/grounding_dino/test_processing_grounding_dino.py
  X:/H/transformers/tests/models/grounding_dino/test_image_processing_grounding_dino.py

Any missing files or assumptions:
  No remote code is required for the inspected official checkpoints. Native
  source supports Swin or AutoBackbone/timm-style backbones through
  `load_backbone`, but official IDEA configs use in-library Swin configs.
  The modeling file is the runtime authority. `modular_grounding_dino.py` only
  contains the generated image-processor subclass skeleton; it is not the
  complete model authority. This is docs-only; no DinoML tests were run.
```

Pinned source URLs:

- `modeling_grounding_dino.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/grounding_dino/modeling_grounding_dino.py
- `configuration_grounding_dino.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/grounding_dino/configuration_grounding_dino.py
- `processing_grounding_dino.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/grounding_dino/processing_grounding_dino.py
- `image_processing_grounding_dino.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/grounding_dino/image_processing_grounding_dino.py

## 2. High-level architecture

Grounding DINO is a text-conditioned detector built from a Swin vision backbone, a BERT text backbone, a multimodal encoder, a Deformable-DETR-style decoder, and token-similarity detection heads.

```text
CPU image processor + BERT tokenizer
  -> pixel_values[B,3,H,W], pixel_mask[B,H,W], input_ids[B,T], attention_mask[B,T]
  -> Swin multiscale feature maps + 2D sine positions
  -> per-level Conv/GN projection to d_model and feature flatten
  -> BERT text encoder + Linear(text_hidden -> d_model)
  -> multimodal encoder layers:
       bidirectional image/text fusion attention
       text enhancer self-attention with phrase-block masks
       multiscale deformable vision self-attention
  -> two-stage proposal selection/topk object queries
  -> decoder layers:
       query self-attention
       query-to-text cross-attention
       query-to-vision multiscale deformable cross-attention
  -> token-similarity class logits + iterative bbox heads
  -> postprocess sigmoid scores, text phrase extraction, cxcywh->xyxy scaling
```

Stage decomposition:

- CPU/data pipeline: image resize/rescale/normalize/pad, pixel mask creation, candidate-label merging, BERT tokenization.
- Vision backbone stage: Swin feature extraction in NCHW, feature masks, and 2D sine positions. Official tiny/base backbones expose stages 2, 3, and 4; a fourth feature level is produced by stride-2 projection on the last feature.
- Text stage: BERT encoder can be validated independently up to `last_hidden_state [B,T,H_text]`, then `text_projection` maps to `d_model=256`.
- Encoder fusion stage: each layer fuses image/text with cross-modal attention, enhances text with phrase-constrained self-attention, and updates vision with multiscale deformable attention.
- Two-stage proposal stage: encoder vision tokens predict proposal boxes and token-similarity scores; top `num_queries` proposals initialize decoder reference points.
- Decoder/detection stage: object queries attend to themselves, text, and multiscale vision features; per-layer heads refine boxes and produce text-token logits.
- Postprocessing: required for end-to-end parity. The grounded postprocessor extracts text phrases from token-level probabilities and does not run NMS.

Independently stageable units: processor/tokenizer handoff, Swin backbone, projection/flatten/position/mask packing, text backbone/projection, one encoder layer, proposal/topk path, one decoder layer, detection heads, and postprocessing.

## 3. Important config dimensions

Official tiny/base configs set `text_config: {"model_type":"bert"}` only. Effective text dimensions below therefore come from `BertConfig` source defaults unless an inspected config overrides them.

| Field | IDEA tiny | IDEA base | tiny-random | Source / notes |
| --- | ---: | ---: | ---: | --- |
| `model_type` | `grounding-dino` | `grounding-dino` | `grounding-dino` | config.json |
| architecture | `GroundingDinoForObjectDetection` | same | same | config.json |
| `d_model` | 256 | 256 | 256 | config.json |
| encoder / decoder layers | 6 / 6 | 6 / 6 | 1 / 1 | config.json |
| encoder heads / decoder heads | 8 / 8 | 8 / 8 | 2 / 2 | config.json |
| encoder/decoder head dim | 32 | 32 | 128 | inferred `d_model / heads` |
| fusion/text-enhancer heads | 4 | 4 | 1 | source uses `encoder_attention_heads // 2` |
| fusion embed dim / head dim | 1024 / 256 | 1024 / 256 | 1024 / 1024 | source uses `encoder_ffn_dim // 2` |
| encoder / decoder FFN | 2048 / 2048 | 2048 / 2048 | 2048 / 2048 | config.json |
| activation | ReLU | ReLU | ReLU | config.json |
| `num_queries` | 900 | 900 | 900 | config.json |
| `max_text_len` | 256 | 256 | 256 | config.json |
| `num_feature_levels` | 4 | 4 | 4 | config.json |
| encoder / decoder sampling points | 4 / 4 | 4 / 4 | 4 / 4 | config.json |
| `two_stage` | true | true | true | config.json |
| `decoder_bbox_embed_share` | true | true | true | config.json/source tied keys |
| `position_embedding_type` | sine | sine | sine | config.json |
| position temperature | 20 | 20 | 20 | config.json |
| text hidden/layers/heads | 768 / 12 / 12 | 768 / 12 / 12 | 32 / 1 / 2 | BERT defaults or config override |
| text vocab / max positions | 30522 / 512 | 30522 / 512 | BERT default vocab / override not complete | BERT defaults |
| backbone | Swin-T style | Swin-B style | tiny Swin | config.json |
| backbone depths | `[2,2,6,2]` | `[2,2,18,2]` | `[1,1,2,1]` | config.json |
| backbone embed dim | 96 | 128 | 12 | config/conversion source |
| backbone window size | 7 effective from conversion source | 12 | Swin default if omitted | config/conversion source |
| backbone out features | stage2, stage3, stage4 | same | same | config.json |
| dtype | float32 | float32 | float32 | config `torch_dtype` |
| processor resize | shortest 800, longest 1333 | same | not inspected | preprocessor_config |
| processor mean/std | ImageNet | ImageNet | not inspected | preprocessor_config |
| tokenizer | BERT uncased | BERT uncased | not inspected | tokenizer_config |
| cache support | none | none | none | source: detector, no generation cache |

Representative checkpoint sweep:

| Checkpoint | Role | Backbone operator variation | Text variation | Detector variation |
| --- | --- | --- | --- | --- |
| `IDEA-Research/grounding-dino-tiny` | common smaller production checkpoint | Swin tiny, depths `[2,2,6,2]`, stage channels inferred as 192/384/768 for projected stages plus extra stride-2 level | BERT default, projected 768->256 | 6 encoder/decoder layers, 900 queries |
| `IDEA-Research/grounding-dino-base` | larger production checkpoint | Swin base, depths `[2,2,18,2]`, `embed_dim=128`, `window_size=12`, more expensive backbone | BERT default, projected 768->256 | same transformer/detector dimensions as tiny |
| `hf-internal-testing/tiny-random-GroundingDinoForObjectDetection` | small/debug and shape stress | tiny Swin, `embed_dim=12`, shallow depths | BERT hidden 32, 1 layer, 2 heads, projected 32->256 | 1 encoder/decoder layer, heads=2 but still `d_model=256` |

Effective defaults that may be omitted from configs: `encoder_ffn_dim=2048`, `decoder_ffn_dim=2048`, `dropout=0.1`, `attention_dropout=0.0`, `activation_dropout=0.0`, `layer_norm_eps=1e-5`, `num_feature_levels=4`, `two_stage=True`, `disable_custom_kernels=False`, `tie_word_embeddings=True`, and BERT text defaults when `text_config` only names `model_type`.

## 3a. Family variation traps

- `model_type` is `grounding-dino` with a hyphen, while the source directory is `grounding_dino`. Keep config routing exact.
- This is not a CLIP-style final embedding similarity model. Conditioning happens repeatedly through cross-modal attention in the encoder, text cross-attention in the decoder, and token-similarity detection heads.
- The source calls `self.text_backbone(input_ids, text_self_attention_masks, token_type_ids, position_ids, ...)` positionally. For BERT this means the second positional argument is the BERT `attention_mask`, but it is a 4D phrase-block mask generated from punctuation/special tokens, not a normal `[B,T]` padding mask.
- Candidate labels are expected as `"label1. label2."`; the processor lowercases and joins list labels with periods. Special token ids are hard-coded for BERT `[CLS]=101`, `[SEP]=102`, period `1012`, question mark `1029`, and PAD `0` in loss label-map code.
- `max_text_len=256` truncates model-side text masks and input ids even though the tokenizer reports `model_max_length=512`.
- Class logits are `[B,num_queries,max_text_len]`, padded with `-inf` beyond actual token length. They are token-level phrase scores, not fixed COCO class logits.
- `num_labels` is not the detection head output size for inference. Text prompt length and token phrase grouping define labels.
- The official configs use `two_stage=True`; decoder reference points are selected with `topk` from encoder proposal scores. A non-two-stage artifact would use learned query embeddings and learned reference points instead.
- `decoder_bbox_embed_share=True` creates a tied-weight contract: all decoder bbox heads after layer 0 logically alias `bbox_embed.0` through `_tied_weights_keys`. Lowering should preserve one logical parameter if weights are tied.
- `disable_custom_kernels=False` advertises use of the `MultiScaleDeformableAttention` kernel from the Hub when available; the Python fallback uses `grid_sample` loops over feature levels.
- Swin backbones are not NCHW conv-only CNNs. They include patch embedding, window attention, shifted windows, relative position bias, patch merging, LayerNorm, and GELU MLPs.
- Source tensor layout is NCHW for image processor/backbone feature maps and masks. Flattened transformer features are `[B,S,C]`. Treat NHWC/channel-last as a guarded optimization region, not a semantic default.
- Axis-sensitive layout traps:
  - `pixel_values` public ABI is `[B,3,H,W]`.
  - `pixel_mask` is `[B,H,W]`; valid pixels are `1`, padding is `0`.
  - Mask interpolation uses `pixel_mask[None].float()` to feature-map `[-2:]`, then `[0]`.
  - Feature flatten is `source.flatten(2).transpose(1,2)`, preserving NCHW row-major `(h,w)` order inside each level.
  - Multiscale metadata order is `(height,width)`, but reference-point normalizers stack `[width,height]`.
  - Postprocess target sizes are `(height,width)`, scale order `[width,height,width,height]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image input `pixel_values [B,3,Hpad,Wpad]`, contiguous initially.
- `pixel_mask [B,Hpad,Wpad]` int64/bool valid-pixel mask; interpolation to each feature level.
- Swin patch embedding/window partition/window reverse/roll or shifted-window equivalents, relative-position index lookup, patch merging.
- Feature projection Conv2d/GN per level: official stage outputs to `d_model=256`, then one extra stride-2 Conv2d/GN level.
- Flatten spatial axes, transpose to `[B,S_level,256]`, concatenate feature levels on sequence axis.
- `spatial_shapes [L,2]`, `level_start_index [L]`, `valid_ratios [B,L,2]`, reference-point meshgrid construction.
- TopK over encoder proposal scores, gather boxes/features by top proposal indices, repeat query embeddings.
- Token mask generation: `isin`, `where`, `cummax`, `cummin`, `flip`, equality broadcast, identity matrix OR, clamp.
- Postprocess thresholding, boolean indexing, per-image variable-length records, token id collection and batch decode.

Neural network primitives:

- BERT text backbone: embedding lookup, token type/position embeddings, LayerNorm, bidirectional self-attention, GELU MLP.
- `text_projection`: Linear(`text_hidden -> 256`) with bias.
- Swin backbone: Conv2d patch embedding, LayerNorm over channel-last token features, window MHA with relative bias, GELU MLP, patch merging.
- Input projections: `Conv2d(C_i -> 256, kernel=1) + GroupNorm(32,256)` for backbone stages; extra `Conv2d(256 -> 256, kernel=3, stride=2, padding=1) + GroupNorm`.
- LayerNorm over last dim in encoder/decoder/fusion.
- Linear projections with bias for all attention and FFN paths.
- ReLU MLPs for encoder/decoder FFNs and bbox heads.
- Elementwise residual adds, learned layer-scale vectors `vision_param/text_param`, dropout/drop-path no-op in inference.

Attention primitives:

- BERT bidirectional self-attention with a model-supplied 4D phrase-block mask.
- Encoder fusion bidirectional cross-modal attention:
  - image queries attend text keys/values.
  - text queries attend vision keys/values.
  - separate output projections back to 256.
- Text enhancer self-attention with `encoder_attention_heads // 2`, additive mask from phrase blocks.
- Vision multiscale deformable self-attention with `encoder_attention_heads`, `num_feature_levels`, `encoder_n_points`.
- Decoder query self-attention with `decoder_attention_heads`.
- Decoder query-to-text cross-attention with additive text padding mask.
- Decoder query-to-vision multiscale deformable cross-attention with `decoder_n_points`.

Position/custom math ops:

- 2D sine position embedding from feature masks, temperature 20 for official configs.
- Text position ids from delimiter blocks, then 1D sinusoidal encoding through `encode_sinusoidal_position_embedding`.
- Query position embedding from reference boxes: encode reference points with `d_model//2` features, then MLP `512 -> 256 -> 256 -> 256`.
- Reference proposal generation with meshgrid, valid-size scaling, inverse sigmoid/logit, validity masking with `inf`.

Preprocessing-coupled ops:

- BERT tokenizer uncased, special token IDs as above.
- Candidate-label joining: list labels become lowercase `"a cat. a dog."`.
- Image resize shortest edge 800 / longest edge 1333, bilinear, ImageNet rescale/normalize, right/bottom pad to batch max.

Detection/postprocessing ops:

- Token-similarity class embed: `vision_hidden @ text_hidden.T`, mask invalid text tokens to `-inf`, pad logits to `max_text_len=256`.
- Bbox head: 3-layer ReLU MLP `256 -> 256 -> 256 -> 4`.
- Iterative refinement: add bbox deltas to `logit(reference, eps=1e-5)`, then sigmoid.
- Grounded postprocess: sigmoid token logits, max over token dimension for object score, threshold boxes, phrase extraction with `prob > text_threshold`, center-to-corners conversion, target-size scale.
- No NMS in native postprocess.

Training/loss deferred ops:

- Label-map construction, focal classification loss, L1/GIoU bbox loss, Hungarian matching if used by loss function, auxiliary layer losses. These are not required for inference parity.

Parameter aliasing:

- Decoder bbox heads may be tied when `decoder_bbox_embed_share=True`; source declares `_tied_weights_keys` for `bbox_embed.(?![0])\d+ -> bbox_embed.0` and `model.decoder.bbox_embed -> bbox_embed`.
- `GroundingDinoContrastiveEmbedding` has no learned weights, so class head module lists do not add parameter aliases.

## 5. Layer/block breakdown

Processor output:

```text
pixel_values:   [B,3,Hpad,Wpad] float, NCHW, ImageNet-normalized
pixel_mask:     [B,Hpad,Wpad] int64/bool, 1 valid and 0 padding
input_ids:      [B,T] int64 BERT ids
attention_mask: [B,T] int64/bool, 1 token and 0 padding
token_type_ids: [B,T] int64, defaults to zeros
```

Backbone and multiscale projection:

```text
features = Swin(pixel_values).feature_maps          # stage2/stage3/stage4 NCHW maps
for feature_map:
  mask_l = interpolate(pixel_mask[None].float(), feature_map.HW).bool()[0]
  pos_l = sine_position(feature_map, mask_l)         # [B,256,Hl,Wl]
  proj_l = Conv2d(C_l -> 256, 1x1) -> GroupNorm(32)

extra levels until L=4:
  proj_l = Conv2d(prev -> 256, 3x3 stride=2 pad=1) -> GroupNorm(32)
  mask_l = interpolate(pixel_mask[None].float(), proj_l.HW).bool()[0]
  pos_l = sine_position(proj_l, mask_l)

source_l = proj_l.flatten(2).transpose(1,2)         # [B,Hl*Wl,256]
mask_l = mask_l.flatten(1)                          # [B,Hl*Wl], valid True
pos_l = pos_l.flatten(2).transpose(1,2) + level_embed[l]
source = concat_l(source_l)                         # [B,S,256]
mask = concat_l(mask_l)                             # [B,S]
```

Text setup:

```text
phrase_mask, position_ids = generate_masks_with_special_tokens(input_ids)
truncate all text tensors to max_text_len if needed
bert_out = BertModel(input_ids, attention_mask=phrase_mask[:,None,:,:],
                     token_type_ids, position_ids)
text_features = Linear(H_text -> 256)(bert_out.last_hidden_state)
text_token_mask = attention_mask.bool()
```

Encoder layer, repeated `encoder_layers`:

```text
text_pos = sinusoidal(position_ids[...,None], num_pos_feats=256)

v0 = LayerNorm(vision_features)
t0 = LayerNorm(text_features)
delta_v, delta_t = BiAttention(v0, t0, vision_pad_mask, text_pad_mask)
vision_features = v0 + vision_param * delta_v
text_features = t0 + text_param * delta_t

q,k = Linear(text_features + text_pos)
v = Linear(text_features)
text_features = LayerNorm(text_features + MHA(q,k,v, phrase_block_mask))
text_features = LayerNorm(text_features + Linear(ReLU(Linear(text_features))))

vision_q = vision_features + vision_pos
vision_features = DeformableAttention(vision_q, vision_features, reference_points)
vision_features = LayerNorm(residual + output_proj)
vision_features = LayerNorm(vision_features + Linear(ReLU(Linear(vision_features))))
```

Two-stage proposal path:

```text
proposals = meshgrid_per_level(valid_width, valid_height, base_wh=0.05*2**level)
proposal_logits = log(proposals / (1 - proposals))
object_query = enc_output_norm(Linear(masked_encoder_vision))
enc_class = object_query @ text_features.T, masked/padded to max_text_len
enc_bbox_logits = bbox_mlp(object_query) + proposal_logits
topk_idx = topk(max(enc_class, dim=-1), k=num_queries)
reference_points = gather(enc_bbox_logits, topk_idx).sigmoid()
target = query_embedding.weight.repeat(B,1,1)        # official config path
```

Decoder layer, repeated `decoder_layers`:

```text
ref_input = reference_points[:, :, None] * valid_ratios_broadcast
query_pos = reference_points_head(sinusoidal(ref_input[:, :, 0, :], d_model//2))

q,k = Linear(hidden + query_pos)
v = Linear(hidden)
hidden = LayerNorm(hidden + MHA(q,k,v))

q = Linear(hidden + query_pos)
k,v = Linear(text_features)
hidden = LayerNorm(hidden + MHA(q,k,v, text_padding_mask))

hidden = LayerNorm(hidden + DeformableCrossAttention(hidden + query_pos,
                                                     vision_features,
                                                     ref_input))
hidden = LayerNorm(hidden + Linear(ReLU(Linear(hidden))))

delta = bbox_embed[layer](hidden)
reference_points = sigmoid(delta + logit(reference_points))
intermediate_hidden[layer] = final_decoder_layernorm(hidden)
```

Detection head:

```text
for layer in decoder_layers:
  logits_l = intermediate_hidden[l] @ encoder_text.T
  logits_l = mask invalid text tokens to -inf, pad last dim to max_text_len
  box_l = sigmoid(bbox_embed[l](hidden_l) + logit(reference_l))

logits = logits_last                              # [B,900,256]
pred_boxes = boxes_last                           # [B,900,4] normalized cxcywh
```

## 6. Attention requirements

BERT text backbone:

- Bidirectional self-attention, no KV cache.
- Official effective BERT defaults: 12 heads, hidden 768, head dim 64, 12 layers. Tiny-random overrides to hidden 32, 2 heads, 1 layer.
- Source passes a phrase-block attention mask, not just padding. It is generated from `[CLS]`, `[SEP]`, `.` and `?` delimiter tokens.

Fusion attention:

- Bidirectional cross-modal attention inside each encoder layer.
- `num_heads = encoder_attention_heads // 2`, `embed_dim = encoder_ffn_dim // 2`.
- Official shape: input vision/text hidden 256; Q/K/V projected to 1024; 4 heads; head dim 256.
- Image-to-text scores use image queries and text keys. Text-to-image reuses the transposed score matrix. Scores subtract a global max and clamp to `[-50000, 50000]` before softmax.
- Mask semantics: `text_attention_mask` masks language tokens for image queries; `vision_attention_mask` masks vision tokens for text queries.

Text enhancer attention:

- Noncausal self-attention over text features, phrase-block mask, official 4 heads, head dim 64 because `GroundingDinoMultiheadAttention` uses hidden 256 with `encoder_attention_heads // 2`.
- Q/K receive sinusoidal text position embeddings; V does not.

Multiscale deformable attention:

- Encoder vision self-attention and decoder vision cross-attention are sparse deformable attention, not dense `S x S` attention.
- Official heads: 8, head dim 32, 4 feature levels, 4 sample points per level.
- Inputs include `spatial_shapes [L,2]`, `level_start_index [L]`, `valid_ratios [B,L,2]`, and per-query reference points.
- The fallback path splits flattened values per level, reshapes to `[B*heads, head_dim, H_l, W_l]`, samples by `grid_sample`, multiplies by softmaxed attention weights, sums over `levels * points`, then output-projects.
- Custom kernel path is advertised through `@use_kernel_forward_from_hub("MultiScaleDeformableAttention")`. DinoML should treat this as a specialized op or guarded lowering target.

Decoder dense attentions:

- Query self-attention: noncausal MHA over 900 object queries, 8 heads, head dim 32.
- Query-to-text cross-attention: noncausal MHA, Q length 900, K/V text length up to 256, additive padding mask.
- No autoregressive generation, causal masks, KV cache, sliding window, ALiBi, or RoPE.

Fused attention parity notes:

- Preserve Q/K position-add but not V position-add in text enhancer and decoder self-attention.
- Preserve fusion attention’s unusual scaling/clamp/order: image query projection is multiplied by `head_dim**-0.5`, then BMM, global max subtraction, clamp, mask, softmax.
- Preserve deformable attention’s bilinear `grid_sample(..., align_corners=False)` fallback math when not using a native kernel.

## 7. Position encoding and custom math

2D sine position embedding:

```python
def grounding_dino_sine_pos(pixel_mask, d_model=256, temperature=20):
    y = pixel_mask.cumsum(1, dtype=float)
    x = pixel_mask.cumsum(2, dtype=float)
    y = y / (y[:, -1:, :] + 1e-6) * (2 * pi)
    x = x / (x[:, :, -1:] + 1e-6) * (2 * pi)
    dim = temperature ** (2 * floor(arange(d_model // 2) / 2) / (d_model // 2))
    px = stack([sin((x[..., None] / dim)[..., 0::2]),
                cos((x[..., None] / dim)[..., 1::2])], dim=4).flatten(3)
    py = stack([sin((y[..., None] / dim)[..., 0::2]),
                cos((y[..., None] / dim)[..., 1::2])], dim=4).flatten(3)
    return concat([py, px], dim=3).permute(0, 3, 1, 2)
```

Generic sinusoidal reference/text position encoding:

```python
def encode_pos(pos_tensor, num_pos_feats=128, temperature=10000):
    dim = temperature ** (2 * floor(arange(num_pos_feats) / 2) / num_pos_feats)
    parts = []
    for coord in unbind(pos_tensor, -1):
        e = coord[..., None] * (2 * pi) / dim
        parts.append(stack([sin(e[..., 0::2]), cos(e[..., 1::2])], -1).flatten(-2))
    if len(parts) >= 2:
        parts[0], parts[1] = parts[1], parts[0]
    return concat(parts, dim=-1).to(pos_tensor.dtype)
```

Phrase mask and position IDs:

```python
special = isin(input_ids, [101, 102, 1012, 1029])
prev = cummax(where(special, arange(T), -1), dim=1)
next = flip(cummin(flip(where(special, arange(T), T), dim=1), dim=1), dim=1)
valid_block = (next != 0) & (next != T - 1) & (next != T)
phrase_mask = eye(T) | ((next[:, :, None] == next[:, None, :]) & valid_block[:, None, :])
position_ids = clamp(arange(T) - prev - 1, min=0) where valid_block else 0
```

Proposal generation:

```python
grid = (meshgrid_xy(H_l, W_l) + 0.5) / [valid_width, valid_height]
wh = ones_like(grid) * 0.05 * (2.0 ** level)
proposal = concat([grid, wh], -1)
proposal_valid = ((proposal > 0.01) & (proposal < 0.99)).all(-1)
proposal_logits = log(proposal / (1 - proposal))
proposal_logits = masked_fill(padding_or_invalid, inf)
```

Precompute opportunities:

- Feature-level sine positions can be cached per `(feature sizes, pixel-mask pattern, dtype)`. Mixed padded batches need mask-aware keys.
- `spatial_shapes`, `level_start_index`, and full-valid reference grids are bucket-cacheable.
- Text phrase masks and position ids are tokenizer-output dependent; cacheable per prompt.
- Query embedding table is constant, but two-stage topk reference points are image/text dependent and cannot be precomputed.

## 8. Preprocessing and input packing

Image processor contract:

- Input images are resized with aspect ratio: shortest edge 800, longest edge 1333.
- Resample is bilinear (`resample=2` in inspected configs).
- Rescale factor is `1/255`; normalize with ImageNet mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`.
- Batch images are padded on the bottom/right to the maximum resized `H,W` unless `pad_size` is provided.
- Output names are `pixel_values [B,3,Hpad,Wpad]` and `pixel_mask [B,Hpad,Wpad]`.

Text processor/tokenizer contract:

- `GroundingDinoProcessor` accepts either a string prompt such as `"a cat. a dog."` or nested candidate-label lists.
- A list like `["A cat", "a dog"]` becomes lowercase `"a cat. a dog."`.
- Official tokenizer config is BERT uncased with `[PAD]=0`, `[UNK]=100`, `[CLS]=101`, `[SEP]=102`, `[MASK]=103`, `model_max_length=512`.
- Processor text kwargs default to `add_special_tokens=True`, `padding=False`, and `return_token_type_ids=True`. Pipeline callers may pass `padding="longest"` for batched prompts.

Recommended first DinoML runtime boundary:

```text
CPU processor/tokenizer owns image decode/resize/normalize/pad and BERT tokenization.
DinoML runtime accepts:
  pixel_values:   [B,3,Hpad,Wpad] float32/float16 NCHW
  pixel_mask:     [B,Hpad,Wpad] int64/bool valid-pixel mask
  input_ids:      [B,T] int64 BERT ids
  attention_mask: [B,T] int64/bool token mask
  token_type_ids: [B,T] int64 optional, default zeros
```

Raw model outputs:

```text
logits:     [B,900,256] token-level logits, padded to max_text_len
pred_boxes: [B,900,4] normalized center_x, center_y, width, height
input_ids:  [B,T] carried in output for phrase decoding
```

Grounded postprocessing:

- `probs = sigmoid(logits)`.
- `scores = max(probs, dim=-1)`.
- Convert boxes from normalized `cxcywh` to `xyxy`.
- If target sizes are present, scale with `[width,height,width,height]`.
- For each image, keep queries where `score > threshold`.
- For each kept query, take `prob > text_threshold`, zero out first/last positions in `get_phrases_from_posmap`, collect token IDs, and `batch_decode` to text labels.
- Return variable-length per-image records: `scores`, `boxes`, `text_labels`; `labels` currently duplicates text labels with a deprecation warning.
- No NMS, class softmax, or no-object class is used in native grounded postprocess.

The image processor also has a generic `post_process_object_detection` that treats the max token index as an integer label. For open-vocabulary parity, `GroundingDinoProcessor.post_process_grounded_object_detection` is the more relevant product contract.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Swin patch embedding Conv2d -> WindowFlatten + GEMM

Source pattern:

```text
Conv2d(3 -> embed_dim, kernel=patch_size, stride=patch_size)
```

Replacement:

```text
NCHW WindowFlatten[B,Hp*Wp,3*patch*patch] -> GEMM_RCR_Bias -> [B,Hp*Wp,embed_dim]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Runtime `H,W` divisible by patch size, or the Swin source padding behavior has been explicitly represented.
- Flatten order matches PyTorch NCHW convolution.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
b = conv.bias
```

Layout constraints: NHWC internal lowering needs an explicit activation layout transform and weight permutation. Failure case: changing patch order breaks all downstream window positions and boxes.

Parity test sketch: compare Swin patch embedding output after source flatten against lowered GEMM for tiny and base configs.

### Rewrite: input projection 1x1 Conv2d -> GEMM

Source pattern:

```text
Conv2d(C_l -> 256, kernel=1) -> GroupNorm(32,256) -> flatten(2).transpose(1,2)
```

Replacement:

```text
spatial rows [B*H_l*W_l,C_l] -> GEMM_RCR_Bias(C_l -> 256) -> GroupNorm -> reshape [B,S_l,256]
```

Preconditions:

- Kernel 1, stride 1, padding 0, dilation 1, groups 1.
- Flatten order preserves `(h,w)` row-major order.
- GroupNorm is still over source channel axis before flatten.

Failure cases: extra feature levels use 3x3 stride-2 convolution and need normal Conv2d lowering, not this rewrite.

### Rewrite: token-similarity class head -> batched GEMM

Source pattern:

```text
output = vision_hidden @ text_hidden.transpose(-1, -2)
output = output.masked_fill(~text_token_mask[:,None,:], -inf)
pad last dim to max_text_len
```

Replacement:

```text
BatchedGEMM([B,Q,256], [B,T,256]^T) -> [B,Q,T] -> MaskFill -> PadTo256
```

Preconditions:

- Text and vision features are both already projected to `d_model`.
- Output orientation remains query-major `[B,Q,T]`.
- Token mask is original attention mask truncated to `T`.

Failure cases: replacing with class-softmax or prompt-level pooling is not equivalent.

### Rewrite: multiscale deformable attention as specialized op

Source pattern:

```text
Linear value_proj, sampling_offsets, attention_weights
reference_points + normalized offsets
per-level grid_sample + weighted sum + output_proj
```

Replacement:

```text
DeformableAttention(value, spatial_shapes, level_start_index,
                    sampling_locations, attention_weights)
```

Preconditions:

- `num_levels`, `num_heads`, `num_points`, and head dim are known or guarded.
- `grid_sample` mode is bilinear, padding zeros, `align_corners=False`.
- Reference points last dim is 2 or 4 and branch is preserved.

Failure cases: treating this as dense attention changes semantics and cost model.

### Rewrite: fixed prompt phrase mask precompute

Source pattern:

```text
generate_masks_with_special_tokens_and_transfer_map(input_ids)
```

Replacement:

```text
CPU/tokenizer-side mask and position_id precompute for static prompts
```

Preconditions:

- Prompt tokens are fixed or cached by `input_ids`.
- Same hard-coded delimiter token ids and max-text truncation are used.

Failure cases: tokenizer vocabulary or delimiter IDs differ from BERT uncased.

### Layout pass candidate: guarded NCHW/NHWC vision islands

Candidate region:

```text
pixel_values NCHW -> Swin patch/projection/GN/local feature maps -> flattened [B,S,C]
```

Required guards and axis rewrites:

- Public ABI remains NCHW unless a separate processor contract is declared.
- Conv weights transform from PyTorch OIHW to provider layout.
- GroupNorm channel axis `1` becomes last axis only inside a fully controlled NHWC island.
- `pixel_mask` stays `[B,H,W]`; do not apply channel-last rewrites to masks.
- `flatten(2).transpose(1,2)` must still produce row-major spatial order.
- `spatial_shapes` and reference point math remain in `(height,width)` source order.

No-layout-translation guards:

- Token/phrase mask generation.
- Multiscale metadata and `grid_sample` coordinates unless a layout-aware deformable attention op owns them.
- Postprocess box conversion/scaling.

## 10. Kernel fusion candidates

Highest priority:

- Multiscale deformable attention. It is central to encoder and decoder vision conditioning; Python `grid_sample` fallback is unlikely to be acceptable for production.
- Swin backbone window attention and patch merging. Base uses deeper Swin-B stages and will spend substantial time before the detector transformer.
- Token-similarity BatchedGEMM + mask/pad for class logits. It is small but exactly defines text-conditioned detection output.
- Dense decoder query self-attention and query-to-text cross-attention at `Q=900`, `T<=256`.
- TopK/gather proposal selection over all multiscale encoder tokens.

Medium priority:

- LayerNorm + Linear projection fusion in Swin, encoder, decoder, and BERT projection paths.
- GroupNorm after input projection Conv2d, possibly fused with projection for fixed `C=256, groups=32`.
- BERT text branch caching for fixed prompts; text embeddings can be reused across repeated images with the same prompt until fusion.
- Box MLP + logit(reference) + sigmoid fusion for iterative refinement.
- Grounded postprocess vectorization: sigmoid, max, threshold, cxcywh-to-xyxy, scaling.

Lower priority:

- CPU/GPU image preprocessing; CPU handoff is enough for first parity.
- Training losses and Hungarian matching.
- Learned position embedding path; official inspected configs use sine.
- Dynamic arbitrary-layout accessors before source-faithful NCHW works.

## 11. Runtime staging plan

Stage 1: config/processor/tokenizer parsing.

- Parse `GroundingDinoConfig`, nested Swin and BERT configs, processor resize/normalization, tokenizer special ids.
- Accept preprocessed tensors at the runtime boundary.

Stage 2: independent text and vision branches.

- Validate BERT text backbone plus `text_projection`.
- Validate Swin backbone feature maps for tiny and base.
- Validate feature masks and sine positions per feature level.

Stage 3: multiscale packing and projection.

- Lower input projections, extra stride-2 level, flatten/concat, `spatial_shapes`, `level_start_index`, `valid_ratios`.

Stage 4: one encoder layer parity.

- Implement fusion attention, text enhancer attention with phrase masks, and deformable vision self-attention.
- Compare one layer, then all encoder layers.

Stage 5: two-stage proposal path.

- Implement proposal generation, encoder token-similarity logits, bbox MLP, `topk`, `gather`, and initial reference points.

Stage 6: decoder parity.

- Implement query self-attention, text cross-attention, deformable vision cross-attention, query position MLP, and iterative bbox refinement.

Stage 7: detection heads and postprocess.

- Implement per-layer bbox/class heads, raw `logits`/`pred_boxes`, and grounded postprocess as CPU helper first.

Stage 8: optimize.

- Replace deformable attention fallback with a provider op, add Swin/window fusions, guarded layout islands, text prompt caching, and postprocess kernels.

Initial stubs:

- Keep tokenizer and image processor on CPU.
- Keep variable-length text-label decode on CPU.
- Require official `two_stage=True` and sine position embeddings for first artifacts.

## 12. Parity and validation plan

Random/operator tests:

- Phrase mask and position id generation for prompts like `"a cat. a dog."`, including punctuation and padding.
- 2D sine position embedding on full-valid and padded masks.
- `encode_sinusoidal_position_embedding` for 2D and 4D reference points.
- Proposal generation and inverse-sigmoid masking for multiple feature levels.
- Multiscale deformable attention fallback against PyTorch `grid_sample`.
- Token-similarity class head mask/pad behavior.
- Bbox refinement with `logit(reference, eps=1e-5)`.

Model slice tests:

- Processor fixture: save HF `pixel_values`, `pixel_mask`, `input_ids`, `attention_mask`.
- Text backbone + projection parity.
- Swin feature maps and input projections for tiny and base.
- Flatten/concat metadata parity: `spatial_shapes`, `level_start_index`, `valid_ratios`.
- One encoder layer parity, then six-layer encoder parity.
- Two-stage topk proposal indices and reference points parity.
- One decoder layer parity, then six-layer decoder/intermediate states parity.
- Raw `GroundingDinoForObjectDetection` `logits [B,900,256]` and `pred_boxes [B,900,4]` parity.
- End-to-end grounded postprocess parity using Transformers integration-test cat image and prompts.

Suggested tolerances:

- fp32 source-faithful subgraphs: start `rtol=1e-4`, `atol=1e-4`; deformable attention and Swin reductions may need `rtol=1e-3` end-to-end.
- fp16/bf16 optimized: compare against reduced-precision PyTorch or HF accelerator runs with `rtol=1e-2`, `atol=1e-2` near attention/deformable sampling; inspect threshold decisions separately.
- Postprocess keep masks and text labels should be exact for non-near-threshold fixtures.

## 13. Performance probes

- CPU image processor throughput: resize/normalize/pad images/sec at common COCO sizes.
- BERT tokenizer and BERT encoder throughput by prompt length and batch size; cache hit/miss for repeated prompts.
- Swin backbone throughput for tiny vs base, with resized/padded image buckets.
- Projection/flatten/position metadata time by number of feature tokens.
- Encoder layer breakdown: fusion attention, text enhancer, deformable vision attention, FFN.
- Deformable attention backend comparison: Python fallback vs custom/provider kernel.
- Proposal `topk`/gather cost over multiscale token count.
- Decoder cost split: query self-attention `900x900`, text cross-attention `900xT`, deformable cross-attention.
- Detection head/postprocess throughput as threshold changes retained box count.
- Batch-size and image-resolution bucket sweep, including padding waste.
- NCHW baseline vs guarded NHWC/channel-last vision islands.

No benchmark measurements are included; these are source-derived probe recommendations.

## 14. Skip/defer list

- Training, gradients, dropout/drop-path stochastic behavior, gradient checkpointing.
- Hungarian matching, focal/L1/GIoU losses, auxiliary loss outputs, label-map loss helpers.
- Non-official `two_stage=False` query/reference path for first integration.
- Learned 2D position embeddings unless a target checkpoint uses them.
- `mm_grounding_dino`, Grounding DINO 1.5, and remote-code-only variants; audit separately.
- Compiled tokenizer, text label decoding, and variable-length result assembly.
- NMS, because native Grounding DINO postprocess does not use it.
- Autoregressive generation, beam search, KV cache, speculative decoding; not applicable.
- Quantization and multi-GPU tensor parallelism.

## 15. Final implementation checklist

- [ ] Parse `GroundingDinoConfig`, nested Swin `backbone_config`, effective BERT `text_config`, and processor/tokenizer configs.
- [ ] Load Swin, BERT, projection, encoder, decoder, bbox-head, query, level-embed, and tied bbox weights.
- [ ] Accept CPU-preprocessed `pixel_values`, `pixel_mask`, `input_ids`, `attention_mask`, and optional `token_type_ids`.
- [ ] Implement BERT phrase-block mask and position-id generation exactly.
- [ ] Implement BERT text backbone plus `text_projection`.
- [ ] Implement Swin backbone parity for official tiny/base configs.
- [ ] Implement NCHW mask interpolation, 2D sine positions, input Conv/GN projections, extra feature level, flatten/concat metadata.
- [ ] Implement valid-ratio and reference-point grid generation.
- [ ] Implement encoder fusion attention with source clamp/mask order.
- [ ] Implement text enhancer self-attention with phrase masks and text position embeddings.
- [ ] Implement multiscale deformable attention fallback/provider op.
- [ ] Implement two-stage proposal generation, token-similarity encoder logits, topk, and gather.
- [ ] Implement decoder query self-attention, text cross-attention, deformable vision cross-attention, and reference-point MLP.
- [ ] Implement iterative bbox refinement and token-similarity detection logits padded to `max_text_len`.
- [ ] Implement grounded postprocess with no NMS and phrase extraction from token probabilities.
- [ ] Add parity tests for tiny-random, IDEA tiny, and IDEA base shape/config sweeps.
- [ ] Add microbenchmarks for Swin, deformable attention, decoder attention, topk/gather, and postprocess.
- [ ] Add guarded layout/GEMM rewrites only after source-faithful NCHW parity is stable.
