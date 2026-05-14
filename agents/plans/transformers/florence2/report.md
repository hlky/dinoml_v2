# Florence2 audit for DinoML v2

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: florence-community/Florence-2-base as the native-source representative; microsoft/Florence-2-{base,base-ft,large,large-ft} and florence-community/Florence-2-{base,base-ft,large,large-ft} inspected for variation.
Config source: HF raw configs saved under agents/plans/transformers/florence2/_sources/
Source files inspected:
  X:/H/transformers/src/transformers/models/florence2/configuration_florence2.py
  X:/H/transformers/src/transformers/models/florence2/modeling_florence2.py
  X:/H/transformers/src/transformers/models/florence2/modular_florence2.py
  X:/H/transformers/src/transformers/models/florence2/processing_florence2.py
  X:/H/transformers/src/transformers/models/florence2/convert_florence2_original_pytorch_to_hf.py
Any missing files or assumptions:
  modeling/configuration/processing files are generated from modular_florence2.py; modular_florence2.py is authoritative for future Transformers edits.
  Text stack is delegated to AutoModel with BART configs; this report records the Florence2-owned bridge plus BART-shaped requirements, but detailed BART parity should compose the separate BART audit.
  No tests/imports were run.
```

Representative URLs:

- [florence-community/Florence-2-base](https://huggingface.co/florence-community/Florence-2-base)
- [florence-community/Florence-2-base-ft](https://huggingface.co/florence-community/Florence-2-base-ft)
- [florence-community/Florence-2-large](https://huggingface.co/florence-community/Florence-2-large)
- [florence-community/Florence-2-large-ft](https://huggingface.co/florence-community/Florence-2-large-ft)
- [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base)
- [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)

## 2. High-level architecture

Florence2 is prompt-conditioned multimodal seq2seq generation:

```text
CPU image resize/rescale/normalize + task prompt construction + BART tokenization
  -> DaViT-like NCHW vision backbone
  -> multimodal projector produces image-token embeddings
  -> masked image-token embedding stitch into BART encoder inputs
  -> BART encoder
  -> BART decoder prefill/decode with cross-attention to encoder outputs
  -> LM head logits
  -> tokenizer decode + task-specific regex/coordinate postprocess
```

Stage decomposition:

- CPU/data pipeline: CLIPImageProcessor resize to 768x768, rescale by 1/255, ImageNet mean/std normalization, task-token-to-natural-language prompt rewrite, insertion of repeated `<image>` placeholders.
- Vision encoder/projector: cacheable per image and preprocessing shape. Base emits `[B, 577, 768]`; large emits `[B, 577, 1024]`.
- Prefix construction: text embeddings are materialized, then projected image features replace placeholder-token embeddings by `masked_scatter`.
- Encoder prefill: BART encoder over the stitched prompt sequence. `encoder_outputs` are independently cacheable across decoder steps.
- Decode: BART decoder uses autoregressive self-attention cache and cross-attends to cached encoder states.
- Postprocess: task-dependent Python/string parsing maps generated text into text, boxes, quadrilateral OCR boxes, polygons, or mixed detection/segmentation structures.

First useful DinoML target: image-conditioned caption/OCR/detection generation for batch size 1 or static small batches, with processor and postprocessor kept in Python.

## 3. Important config dimensions

Source defaults versus normalized checkpoint configs:

| Field | Source default | Community base/base-ft | Community large | Community large-ft | Notes |
|---|---:|---:|---:|---:|---|
| dtype | source init | float16 | float16 | float16 | From config `dtype`; older Microsoft configs use `torch_dtype`. |
| image_token_id | 51289 | 51289 | 51289 | 51289 | Processor/tokenizer coupling. |
| text model_type | bart default | bart | bart | bart | Native source delegates to BART. |
| text d_model | BART default 1024 | 768 | 1024 | 1024 | Must match projector dim and LM head input. |
| text encoder layers | BART default 12 | 6 | 12 | 12 | BART encoder blocks. |
| text decoder layers | BART default 12 | 6 | 12 | 12 | BART decoder blocks. |
| text heads | BART default 16 | 12 | 16 | 16 | Head dim is 64 in inspected configs. |
| text FFN dim | BART default 4096 | 3072 | 4096 | 4096 | GELU FFN. |
| text vocab_size | BART default 50265 | 51328 | 51328 | 51328 | Older Microsoft configs report 51289. |
| text max positions | BART default 1024 | 1024 | 4096 | 1024 | Large and large-ft differ. |
| use_cache | BART default | true | true | true | In `text_config`. |
| vision embed_dim | `(128,256,512,1024)` | same | `(256,512,1024,2048)` | same as large | Current configs keep both `dim_embed` and `embed_dim`; source reads `embed_dim`. |
| vision depths | `(1,1,9,1)` | same | same | same | 12 Florence2 vision blocks total. |
| vision heads/groups | `(4,8,16,32)` | same | `(8,16,32,64)` | same as large | Channel attention uses `num_groups`; window attention uses `num_heads`. |
| patch size/stride/pad | `(7,3,3,3)` / `(4,2,2,2)` / `(3,1,1,1)` | same | same | same | Four Conv2d patch stages. |
| patch prenorm | `(False,True,True,True)` | same | same | same | Pre/post LayerNorm axis changes. |
| window_size | 12 | 12 | 12 | 12 | Vision local attention pads H/W to multiples of 12. |
| processor image size | n/a | 768x768 | assumed same from family | assumed same | Base preprocessor saved; public family uses fixed 768 crop/resize. |
| image_seq_length | n/a | 577 | likely 577 | likely 577 | From preprocessor; must equal projected feature count. |
| generation | n/a | beams 3, forced BOS 0, forced EOS 2, no-repeat 3 | same pattern | same pattern | From generation config/model config. |

Representative checkpoint sweep:

| Model id | Config access | Text stack | Vision stack | Operator-significant variation |
|---|---:|---|---|---|
| `florence-community/Florence-2-base` | yes, saved | 6x encoder + 6x decoder, 768, 12 heads | DaViT width `[128,256,512,1024]` projected to 768 | Good native-source base target. |
| `florence-community/Florence-2-base-ft` | yes, saved | same as base | same as base | Fine-tune changes weights/task behavior, not topology. |
| `florence-community/Florence-2-large` | yes, saved | 12x encoder + 12x decoder, 1024, 16 heads, max pos 4096 | DaViT width `[256,512,1024,2048]` projected to 1024 | Larger GEMMs and longer encoder/decoder position tables. |
| `florence-community/Florence-2-large-ft` | yes, saved | same as large but max positions 1024 | same as large | Fine-tuned large has shorter text max position than large. |
| `microsoft/Florence-2-*` | yes, inspected live | older remote-code style, vocab 51289 | legacy `dim_embed`, `image_pos_embed.max_pos_embeddings` | Treat as compatibility inputs or route through remote-code audit unless normalized. |

## 3a. Family variation traps

- Native source reads `vision_config.embed_dim`; older Microsoft configs advertise `dim_embed`. DinoML should normalize or reject configs lacking `embed_dim` for the in-library path.
- `image_seq_length=577` is an ABI contract between processor placeholders and projector output. It assumes 768x768 preprocessing and the four patch stages produce a 24x24 final grid plus one pooled token.
- Text vocab differs by source basis: normalized community configs use `51328`; older Microsoft configs use `51289` while still using `image_token_id=51289`.
- Large and large-ft differ in `text_config.max_position_embeddings` (`4096` versus `1024`), so long-context admission must read the checkpoint, not the family name.
- The model is encoder-decoder, not decoder-only. KV cache is decoder-owned; image/encoder features are prefix/encoder caches, not decoder KV.
- Vision layout alternates NCHW conv/depthwise conv with NHWC/token layouts. A global NHWC translation is unsafe without guarding every flatten/transpose/window reconstruction axis.
- Inference has dropout/drop-path modules in source but they are identity in eval mode. DinoML can reject training mode.
- Postprocessing ABI is task-specific and string/tokenizer-driven; end-to-end detection/OCR/segmentation parity is not just neural logits.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW `Conv2d` with stride/padding: stage patch embeds `3->128/256`, then `128->256`, `256->512`, `512->1024` for base; doubled large widths.
- Depthwise `Conv2d` with `groups=C`, kernel 3, padding 1, residual add.
- `permute`, `transpose`, `flatten`, `view`/`reshape`, `contiguous`, `unsqueeze`, `repeat`, slicing/crop after window unpadding.
- `F.pad` for NHWC window padding: pad last channel with zero width and spatial H/W to window multiples.
- Mean reductions over visual temporal/spatial axes, `torch.cat`.
- Boolean compare, `sum`, `expand_as`, and `masked_scatter` for image embedding stitch.
- Embedding lookup, position ID/arange, attention masks from BART.

Neural network primitives:

- LayerNorm over channel-last/token hidden dim in vision and standard BART LayerNorm.
- Linear/GEMM with bias for vision QKV/proj/MLP and BART projections; projector `Linear(vision_embed_dim -> text_d_model, bias=False)`.
- GELU FFN activation.
- Tied LM head weight alias: `lm_head.weight` tied to `model.language_model.shared.weight`.

Attention primitives:

- Vision channel attention: noncausal grouped channel attention with unusual shapes `q,k,v = [B, groups, C/groups, num_tokens]`; score shape attends over channel subdimensions, not sequence positions.
- Vision window attention: noncausal local MHA over non-overlapping `12x12` windows after spatial padding; q/k/v shape `[B*num_windows, heads, 144, head_dim]`.
- BART encoder self-attention: noncausal MHA.
- BART decoder self-attention: causal MHA with KV cache.
- BART decoder cross-attention: rectangular noncausal attention from decoder query length to encoder sequence length.

Position/custom math:

- Learned 2D absolute visual embeddings from separate row/column tables, concatenated to channel dim.
- Fixed 1D sinusoidal temporal embedding buffer, sliced to visual token sequence length.
- BART learned token/position embeddings.

Preprocessing-coupled ops:

- CLIPImageProcessor resize/rescale/normalize to NCHW `pixel_values`.
- Prompt rewrite from task token to natural language.
- `<image>` placeholder repetition count exactly equals `image_seq_length`.
- Postprocessor regex parsing, coordinate dequantization from `<loc_0>`...`<loc_999>` bins.

Generation/cache ops:

- Encoder output cache.
- Decoder self-attention KV cache, plus BART cross-attention handling.
- Generation controller: beams, forced BOS/EOS, no-repeat ngram, decoder start token.
- `logits_to_keep` slicing for last-token-only or selected logits.

## 5. Layer/block breakdown

Vision patch stage `s`:

```text
if patch_prenorm[s]:
  x: [B,C,H,W] -> [B,H,W,C] -> LayerNorm(C) -> [B,C,H,W]
x = Conv2d(C_in -> C_s, kernel=patch_size[s], stride=patch_stride[s], padding=patch_padding[s])
if not patch_prenorm[s]:
  x: [B,C_s,Hs,Ws] -> [B,Hs,Ws,C_s] -> LayerNorm(C_s) -> [B,C_s,Hs,Ws]
repeat depth[s] Florence2VisionBlock
```

Florence2 vision block, repeated `sum(depths)=12` times:

```text
Spatial block:
  x = depthwise_conv3x3(x) + x                         # [B,C,H,W]
  t = flatten_hw(x).transpose -> [B,H*W,C]
  t = LayerNorm(t)
  t = view [B,H,W,C]
  t = window_partition_pad(t, window=12)
  q,k,v = Linear(C -> 3C, bias=qkv_bias), split [q,k,v]
  t = noncausal window MHA(q,k,v)
  t = Linear(C -> C)
  x = residual + t, then back to [B,C,H,W]
  x = depthwise_conv3x3(x) + x
  t = LayerNorm(flatten_hw(x))
  t = Linear(C -> 4C) -> GELU -> Linear(4C -> C)
  x = residual + t, back to [B,C,H,W]

Channel block:
  x = depthwise_conv3x3(x) + x
  t = LayerNorm(flatten_hw(x))
  q,k,v = Linear(C -> 3C), reshape [B,N,3,groups,C/groups]
  t = grouped channel attention over C/groups by N values
  t = Linear(C -> C)
  x = residual + t, back to [B,C,H,W]
  x = depthwise_conv3x3(x) + x
  t = LayerNorm(flatten_hw(x))
  t = Linear(C -> 4C) -> GELU -> Linear(4C -> C)
  x = residual + t, back to [B,C,H,W]
```

Multimodal projector:

```text
f = final vision map [B,C,H,W]
f = f + learned_2d_abs_pos([B,C,H,W])
tokens = flatten_hw(f).transpose -> [B,H*W,C]
tokens = tokens + sinusoidal_temporal_embed(tokens[:, :1, :])
tokens = tokens.unsqueeze(1) -> [B,1,H*W,C]
spatial = mean(tokens, dim=2) -> [B,1,C]
temporal = mean(tokens, dim=1) -> [B,H*W,C]
image_tokens = cat([spatial, temporal], dim=1) -> [B,H*W+1,C]
image_tokens = Linear(C -> d_model, bias=False) -> LayerNorm(d_model)
```

BART text path, composed from `text_config`:

```text
encoder_inputs = token_embedding(input_ids)
encoder_inputs = masked_scatter(image_placeholder_mask, image_tokens)
encoder_hidden = BART encoder self-attention stack
decoder_hidden = BART decoder causal self-attention + cross-attention to encoder_hidden
logits = tied LM head(decoder_hidden[:, slice_indices, :])
```

## 6. Attention requirements

Vision channel attention:

- Noncausal self-attention with no mask.
- Source q/k/v shape after packed projection: `[B, groups, C/groups, N]`.
- Scale is `num_tokens ** -0.5`, not `head_dim ** -0.5`.
- This is not standard sequence MHA; a first port can lower to explicit batched matmul/softmax/matmul with layout guards.

Vision window attention:

- Noncausal local MHA with no mask.
- Input semantic layout inside attention is `[B,H,W,C]`.
- Pad H/W to multiples of `window_size=12`; window token length is 144.
- q/k/v shape `[B * ceil(H/12) * ceil(W/12), heads, 144, C/heads]`.
- Output unpartitions windows and crops to original H/W.
- SDPA/FlashAttention-compatible for the inner fixed-window attention if the window packing is explicit and stable.

BART attention:

- Encoder self-attention is dense noncausal MHA with padding mask.
- Decoder self-attention is dense causal MHA with autoregressive cache.
- Cross-attention is rectangular MHA from decoder query length to encoder key/value length.
- No GQA/MQA in inspected configs; `head_dim = d_model / heads = 64`.
- Cache type is standard encoder-decoder generation cache. Pixel values are sent only on the first generation iteration or when cache is disabled; subsequent decode uses cached encoder and decoder states.

## 7. Position encoding and custom math

Visual learned 2D absolute position:

```python
def visual_2d_pos(row_table, col_table, batch, height, width):
    x = col_table[arange(width)]              # [W, C_col]
    y = row_table[arange(height)]             # [H, C_row]
    pos = cat([repeat_h(x), repeat_w(y)], -1) # [H, W, C]
    return pos.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
```

Visual coordinate dequantization in postprocess:

```python
def dequantize_loc_bins(loc, image_width, image_height):
    per_bin_w = image_width / 1000
    per_bin_h = image_height / 1000
    x = (loc_x + 0.5) * per_bin_w
    y = (loc_y + 0.5) * per_bin_h
    return int(x), int(y)
```

The 1D visual temporal embedding is a registered constant buffer built from sin/cos at init and sliced by sequence length. BART learned positions are delegated to the text model.

## 8. Preprocessing and input packing

Processor ABI:

- `images` -> CLIPImageProcessor output `pixel_values` with source layout `[B, 3, 768, 768]` for the saved base preprocessor.
- Resize is enabled; center crop is disabled; rescale factor is `1/255`; normalize with mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
- `text` must be a string/list of strings. Task tokens are rewritten before tokenization.
- With images, each prompt becomes `<image>` repeated `image_seq_length`, then BOS, prompt, EOS. Tokenization is called with `add_special_tokens=False`.
- Optional `mm_token_type_ids` can be returned, but the Florence2 model forward does not consume it in the inspected source.

Task prompts:

| Task token | Prompt behavior | Postprocess type |
|---|---|---|
| `<OCR>` | no extra input | pure text |
| `<OCR_WITH_REGION>` | no extra input | OCR quadrilateral boxes |
| `<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>` | no extra input | pure text |
| `<OD>`, `<DENSE_REGION_CAPTION>` | no extra input | labels + boxes |
| `<CAPTION_TO_PHRASE_GROUNDING>` | task token plus caption text | phrase grounding boxes |
| `<REFERRING_EXPRESSION_SEGMENTATION>`, `<REGION_TO_SEGMENTATION>` | task token plus region/text | polygons |
| `<OPEN_VOCABULARY_DETECTION>` | task token plus vocabulary text | boxes or polygons |
| `<REGION_TO_CATEGORY>`, `<REGION_TO_DESCRIPTION>`, `<REGION_TO_OCR>` | task token plus region | pure text |
| `<REGION_PROPOSAL>` | no extra input | boxes |

Postprocess ABI:

- Detection-like outputs parse generated `<loc_i>` tokens into `[xmin,ymin,xmax,ymax]` boxes in original image `(width,height)`.
- OCR parses eight location tokens into quadrilateral boxes and optional area filtering.
- Segmentation parses polygon blocks delimited by `<poly>`, `</poly>`, and `<sep>`.
- No NMS is implemented in processor postprocess; generated sequence order is source behavior.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fixed-size patch Conv2d -> GEMM/im2col

Preconditions:

- Source path is a `Florence2VisionConvEmbed` stage.
- `groups == 1`, `dilation == 1`, padding/stride/kernel match config.
- Input layout is NCHW or an explicitly guarded NHWC equivalent.
- Dynamic H/W either fixed to processor size or bucketed with exact output-shape equations.

Replacement:

```text
Pad -> WindowExtract/im2col -> GEMM(weight_flat.T) -> BiasAdd -> Reshape
```

Failure cases: arbitrary user image sizes that break placeholder count, nonzero dilation, or missing conv support fallback.

Parity sketch: compare each patch stage output for base and large widths at 768x768 and one nonstandard guarded image size that still fits max position embeddings.

### Rewrite: depthwise 3x3 residual conv fusion

Preconditions:

- `groups == channels`, kernel 3, padding 1, stride 1.
- Input/output channel count equal.
- Eval mode.

Replacement:

```text
DepthwiseConv3x3 -> AddResidual
```

Candidate kernel: fused depthwise conv plus residual add in NCHW, or NHWC guarded if the surrounding region is translated.

Failure cases: training/drop-path active, non-contiguous layout, or dynamic shape without halo bounds.

### Rewrite: window partition + local attention

Preconditions:

- Window size exactly 12.
- H/W known or bucketed; pad extents computed as `(12 - dim % 12) % 12`.
- No attention mask.
- qkv packed projection layout is `[out=3*C, in=C]` from `nn.Linear`.

Replacement:

```text
PadNHWC -> PartitionWindows -> PackedQKVGEMM -> FixedLenAttention(144) -> OutGEMM -> Unpartition -> Crop
```

Failure cases: global NHWC rewrite changes flatten order; odd image sizes not represented in placeholder/token ABI; attention backend cannot handle the small fixed windows efficiently.

### Rewrite: image placeholder stitch -> indexed copy

Preconditions:

- `input_ids` are available, not only `inputs_embeds`.
- Count of `image_token_id` equals `B * image_seq_length`.
- Image placeholders are contiguous prefix tokens as emitted by Florence2Processor.

Replacement:

```text
Embedding(input_ids) -> Copy image_tokens into prefix slice
```

Weight transform: none.

Failure cases: caller passes custom prompts with non-prefix image tokens, `inputs_embeds` only path, or multiple images per sample if future processor changes ABI.

### Layout guard: vision NCHW/NHWC region

Candidate optimized layout: keep conv/depthwise in NCHW initially. Later, translate only a complete local region spanning depthwise conv, flatten, LayerNorm, attention/MLP, and back if all axis rewrites are explicit.

Required rewrites:

- `permute(0,2,3,1)` LayerNorm axes become channel-last native normalization.
- `flatten(2).transpose(1,2)` maps NCHW `[B,C,H,W]` to token `[B,H*W,C]`; NHWC must preserve row-major H/W token order.
- Window attention `view/permute/view` assumes `[B,H,W,C]`.

Failure cases: treating `dim=1` channel reductions as `dim=-1` globally, or eliding `contiguous` where source view order depends on it.

## 10. Kernel fusion candidates

Highest priority:

- BART GEMM/attention/LayerNorm stack reuse: the text encoder-decoder is most of the recurring generation cost after vision prefill.
- Vision window attention fixed length 144: compact local attention kernel or SDPA wrapper can avoid many tiny matmuls.
- Placeholder prefix copy rewrite: removes general boolean `masked_scatter` from the hot prefill graph.
- DepthwiseConv3x3 + residual add: occurs twice per spatial and channel block.

Medium priority:

- Vision QKV packed projection + attention + output projection for both window and channel attention.
- LayerNorm + Linear for MLP/projection blocks.
- Projector flatten/mean/cat/projection chain for the fixed 24x24 grid.
- Last-token-only logits via `logits_to_keep=1` during decode.

Lower priority:

- Postprocess acceleration. Regex/string parsing can remain Python until neural parity is stable.
- Beam-search controller optimization. Keep generation control in Python initially.
- Full NHWC vision rewrite. Useful later, but axis-sensitive enough to require dedicated layout tests.

## 11. Runtime staging plan

Stage 1: parse normalized Florence2 config and processor metadata. Reject or explicitly normalize legacy Microsoft remote-code fields.

Stage 2: implement/cache the vision encoder/projector for fixed 768x768 images with NCHW conv/depthwise conv, window attention, channel attention, LayerNorm, and GELU.

Stage 3: implement image-feature prefix construction using the contiguous-placeholder fast path, with a fallback or rejection for arbitrary `masked_scatter`.

Stage 4: compose existing BART encoder-decoder lowering for prefill without cache; validate logits for teacher-forced decoder inputs.

Stage 5: enable generation decode with encoder output cache, decoder KV cache, `logits_to_keep=1`, and Python generation controller.

Stage 6: add task postprocess parity for caption, OCR, OD, phrase grounding, and polygons in Python.

Stage 7: optimize fixed-window vision attention, depthwise residual kernels, BART attention, and layout-local fusions.

Initially stub: training loss, gradient checkpointing, `output_attentions`, `output_hidden_states`, beam-search internals inside DinoML, and structured postprocess on GPU.

## 12. Parity and validation plan

- Processor parity: compare `input_ids`, `<image>` token count, `attention_mask`, and `pixel_values` statistics for one image and each task token.
- Vision unit parity: ConvEmbed stages, one spatial block, one channel block, then full vision backbone at fp32/fp16. Recommended tolerances: fp32 `1e-4`; fp16 `5e-2` for full stack until fused attention is mature.
- Projector parity: verify output shape `[B,577,d_model]` and exact placeholder count agreement.
- Stitch parity: compare general source `masked_scatter` with DinoML prefix-copy rewrite for processor-generated prompts and rejection for non-prefix masks.
- BART compose parity: encoder output for text-only and image-text inputs; decoder one-step logits with fixed `decoder_input_ids`.
- Cache parity: generate first step with `pixel_values`, then next step with cache and no vision recomputation; compare logits and cache shapes.
- End-to-end parity: caption text, OCR with regions, OD boxes, phrase grounding, and segmentation polygons against Transformers for small deterministic prompts.

## 13. Performance probes

- CPU processor throughput: image resize/normalize plus tokenizer/prompt expansion.
- Vision encoder latency and memory at batch sizes 1, 2, 4 and 768x768.
- Projector/stitch prefill overhead versus BART encoder cost.
- BART encoder-only throughput by sequence length: 577 image tokens + prompt length.
- Decode tokens/sec with cached encoder outputs and `logits_to_keep=1`.
- Beam width sweep for Python generation controller overhead.
- Window attention backend comparison: eager matmul/softmax, SDPA, custom fixed-144 kernel.
- Depthwise conv residual kernel throughput in NCHW versus guarded NHWC.
- End-to-end requests/hour split into processor, vision, encoder, decode, and postprocess.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- Remote-code-only Microsoft legacy behavior unless configs are normalized.
- GPU regex/string postprocessing.
- Arbitrary image sizes that change final feature grid and placeholder count.
- General non-prefix `masked_scatter`.
- Returning attentions/hidden states.
- Multi-image per sample, video, or temporal image stacks beyond the source's single-image processor ABI.
- Multi-GPU/tensor parallel.
- Quantized or packed weights beyond normal dense fp16/fp32 loading.

## 15. Final implementation checklist

- [ ] Parse native Florence2 config and nested BART config.
- [ ] Normalize/reject legacy `dim_embed`-only configs.
- [ ] Load tied text embedding/LM-head weights without cloning aliases.
- [ ] Implement CLIPImageProcessor-compatible CPU preprocessing metadata.
- [ ] Implement Florence2 prompt expansion and image placeholder count validation.
- [ ] Implement NCHW Conv2d and depthwise Conv2d coverage needed by vision stages.
- [ ] Implement vision LayerNorm axis transitions and layout guards.
- [ ] Implement Florence2 window attention with pad/partition/unpartition/crop.
- [ ] Implement Florence2 channel attention grouped over channel subdimensions.
- [ ] Implement visual learned 2D and sinusoidal 1D position embedding paths.
- [ ] Implement multimodal projector `[B,C,H,W] -> [B,H*W+1,d_model]`.
- [ ] Add prefix-copy rewrite for processor-generated image placeholders.
- [ ] Compose BART encoder-decoder prefill.
- [ ] Compose BART decoder KV-cache generation.
- [ ] Add Python postprocess parity for caption/OCR/OD/phrase/polygon tasks.
- [ ] Add fixed-shape and cache parity tests.
- [ ] Benchmark processor, vision, encoder, decode, and postprocess separately.

## Gated gaps for DinoML

- `Conv2d`/depthwise Conv2d are required; current op checklist marks convolution as unported.
- General NHWC/channel-last translation must be guarded. Florence2 source depends on exact NCHW-to-token and NHWC window view order.
- Florence2 channel attention is not normal sequence MHA and should be admitted as a bounded custom attention pattern or lowered through explicit BMM/softmax/BMM.
- `masked_scatter` should not be admitted generally for first integration; use a prefix-copy rewrite with processor-generated prompts.
- End-to-end structured tasks require processor/postprocessor ABI parity, including task prompt rewrites, 1000-bin coordinate dequantization, and no-NMS behavior.
