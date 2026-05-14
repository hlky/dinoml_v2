# TVP audit report

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `X:/H/transformers`.

Model id: primary target `Intel/tvp-base`; sweep also inspected `Intel/tvp-base-ANet` and `Jiqing/tiny-random-tvp`.

Config source: official HF `config.json` and `preprocessor_config.json` for the three repositories above where available. `Intel/tvp-base` has no `tokenizer_config.json` at the raw path checked; the preprocessor config names `bert-base-uncased`.

Source files inspected:

- `src/transformers/models/tvp/configuration_tvp.py`
- `src/transformers/models/tvp/modeling_tvp.py`
- `src/transformers/models/tvp/processing_tvp.py`
- `src/transformers/models/tvp/image_processing_tvp.py`
- `src/transformers/models/tvp/image_processing_pil_tvp.py`
- `src/transformers/models/tvp/__init__.py`

Snapshots: see `_sources/source_basis.md` for inspected source/config excerpts.

Any missing files or assumptions: no tokenizer implementation is TVP-specific. The text branch uses a normal external tokenizer, effectively BERT-style for official checkpoints. No gated/401 configs were encountered.

## 2. High-level architecture

TVP is an encoder-style video-language temporal grounding model, not an autoregressive video-language generator.

Dataflow:

```text
video frames + text query
  -> CPU/data preprocessing and tokenization
  -> visual prompt added to NCHW video frames
  -> ResNet AutoBackbone per frame
  -> 3x3 conv + maxpool + ReLU
  -> temporal mean pooling over frames
  -> 2D row/column visual position embeddings
  -> text embeddings + learned 10-token prompt + visual tokens
  -> dense bidirectional Transformer encoder
  -> first-token pooler
  -> MLP + sigmoid
  -> normalized [start, end] logits
  -> processor postprocess scales by video duration
```

Stage decomposition:

- CPU/data pipeline: frame extraction is outside Transformers; TVP processor resizes/pads/normalizes/flips channel order and tokenizes text.
- Visual encoder: independently cacheable per video if text prompts vary. Current source mean-pools over frames before text fusion, so no per-frame token sequence remains after visual embedding.
- Fusion encoder: text prompt, text tokens, and visual tokens are concatenated into one bidirectional sequence.
- Head/postprocess: pooled first token -> two normalized times -> duration scaling.

First useful DinoML target: `TvpForVideoGrounding` inference with fixed checkpoint config, externally supplied frames, and CPU preprocessing.

## 3. Important config dimensions

Source/default dimensions:

| Field | Default/source value | Notes |
| --- | ---: | --- |
| `model_type` | `tvp` | In-library TVP source. |
| `hidden_size` | 768 | Encoder width and visual conv output width. |
| `num_hidden_layers` | 12 | Dense encoder layers. |
| `num_attention_heads` | 12 | MHA, no GQA/MQA. |
| `head_dim` | 64 | Inferred as `hidden_size / num_attention_heads`; source requires divisibility. |
| `intermediate_size` | 3072 | BERT-style FFN. |
| `vocab_size` | 30522 | BERT uncased shape in official configs. |
| `max_position_embeddings` | 512 | Text embeddings and visual `position_embeddings` module; visual path uses row/column embeddings. |
| `max_grid_row_position_embeddings` | 100 | Row embedding table. |
| `max_grid_col_position_embeddings` | 100 | Column embedding table. |
| `max_img_size` | 448 | Visual prompt and default processor size. |
| `num_frames` | 48 | 64 in ANet, 1 in tiny random. |
| `visual_prompt_size` | 96 | Prompt border width. |
| `hidden_act` | `gelu` | FFN activation. |
| `layer_norm_eps` | `1e-12` | BERT-like LayerNorm. |
| `cache support` | none in source | Configs advertise `use_cache=true`, but source ignores it. |

Representative checkpoint sweep:

| Model id | Hidden | Layers | Heads | FFN | Frames | Text max | Backbone feature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Intel/tvp-base` | 768 | 12 | 12 | 3072 | 48 | 100 | ResNet stage4, 2048 channels |
| `Intel/tvp-base-ANet` | 768 | 12 | 12 | 3072 | 64 | 100 | ResNet stage4, 2048 channels |
| `Jiqing/tiny-random-tvp` | 128 | 4 | 4 | 384 | 1 | 20 | Tiny ResNet stage2, 128 channels |

## 3a. Family variation traps

- `backbone_config` owns the visual operator body. The report covers TVP wrapper/fusion logic and notes the default ResNet body, but full ResNet op coverage should compose the separately audited ResNet family.
- `num_frames` changes prompt parameter shapes and input ABI. `Intel/tvp-base-ANet` uses 64 frames instead of 48.
- `framedownpad` is unsafe with normal current configs: source reads `config.frame_num`, but `TvpConfig` and public configs define `num_frames`.
- `use_cache=true` in configs is historical/ignored for current source. DinoML should reject cache/generation expectations for TVP.
- Processor configs use legacy field names and values: `do_padding`/`padding_size`, `do_rescale=false`, no center crop, and custom mean/std. Current source class defaults differ, so checkpoint processor config must be honored.
- Source modeling accepts `interpolate_pos_encoding`; this introduces bicubic interpolation for visual prompt/row-column embeddings when spatial shapes exceed training table sizes.
- Visual tensor layout is axis-sensitive: source input is `[B, F, C, H, W]`, backbone is NCHW, TVP grid after conv is `[B, F, H, W, C]`, and visual tokens flatten `[H, W]` in row-major order. Layout passes need guards around `permute`, `mean(1)`, `view`, and row/column embedding axes.
- The model temporally mean-pools visual grid features before fusion. It does not expose frame-level attention tokens or a temporal Transformer despite consuming video frames.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`/reshape `[B,F,C,H,W] -> [B*F,C,H,W]`
- `view`/reshape `[B*F,C,H',W'] -> [B,F,C,H',W']`
- `permute` NCHW grid to `[B,F,H',W',C]`
- mean reduction over frame axis 1
- flatten visual grid `[B,H',W',C] -> [B,H'*W',C]`
- `torch.cat` along sequence axis for 10 prompt tokens, text tokens, visual tokens
- `expand` for learned text prompt
- arange/zeros/ones for position, token type, and masks
- broadcast add for embeddings and masks

Neural network primitives:

- delegated ResNet backbone feature extraction on `[B*F,3,H,W]`
- Conv2d `in_channels=backbone_last_hidden_size -> hidden_size`, `kernel=3`, `stride=1`, `padding=1`, no bias
- MaxPool2d `kernel=2`, `stride=2`
- ReLU
- Embedding lookups for words, text positions, token types, visual row/column positions
- LayerNorm with eps `1e-12`
- Dropout is no-op for inference
- Linear projections for Q/K/V/O, FFN, pooler, and grounding head
- GELU FFN activation, tanh pooler activation, sigmoid final head activation

Attention primitives:

- Dense bidirectional self-attention over the fused sequence.
- MHA only: heads = `num_attention_heads`, key/value heads equal query heads.
- Attention shape `[B, heads, S, S]`, where `S = 10 + text_len + visual_grid_tokens`.
- Additive broadcast attention mask from `get_extended_attention_mask`.

Position/custom math:

- Text absolute position embeddings.
- Visual row + column absolute embeddings, optional bicubic interpolation.
- Learned 10-token text prompt.
- Learned visual prompt border in pixel space before backbone.

Preprocessing-coupled ops:

- Resize longest edge, optional center crop, optional rescale, normalize, pad to 448x448, RGB-to-BGR channel flip.
- Output `pixel_values` shape `[B,F,3,H,W]`.
- Text tokenization from external tokenizer with no TVP-specific tokenizer code.

Postprocessing:

- `post_process_video_grounding(logits, video_duration)` returns rounded seconds: `start=logit[0]*duration`, `end=logit[1]*duration`.
- No NMS, no beam search, no decoding, no box conversion.

## 5. Layer/block breakdown

Visual branch:

```text
pixel_values: [B,F,3,H,W]
prompted = visual_prompter(pixel_values)
frames = reshape(prompted, [B*F,3,H,W])
features = AutoBackbone(frames)["feature_maps"][0]        # NCHW
grid = Conv2d(backbone_C -> hidden_size, 3x3, pad=1, bias=False)
grid = MaxPool2d(2,2)
grid = ReLU(grid)
grid = reshape(grid, [B,F,hidden_size,Hg,Wg])
grid = permute(grid, [B,F,Hg,Wg,hidden_size])
grid = mean(grid, dim=1)                                  # [B,Hg,Wg,H]
grid = grid + row_pos[Hg] + col_pos[Wg]
visual_tokens = reshape(grid, [B,Hg*Wg,H])
visual_tokens = visual_tokens + visual_token_type[0]
visual_tokens = LayerNorm(visual_tokens)
```

Text branch:

```text
text = word_embedding(input_ids) + pos_embedding(arange(T)) + token_type_embedding(zeros)
text = LayerNorm(text)
prompt = learned_text_prompt.expand(B, 10, H)
```

Encoder layer, repeated `num_hidden_layers`:

```text
q = Linear(H -> H)(x)
k = Linear(H -> H)(x)
v = Linear(H -> H)(x)
q,k,v = reshape to [B, heads, S, head_dim]
scores = q @ k.T / sqrt(head_dim)
scores = scores + additive_attention_mask
p = softmax(scores, dim=-1)
attn = p @ v
attn = Linear(H -> H)(merge_heads(attn))
x = LayerNorm(attn + residual)
mlp = GELU(Linear(H -> intermediate_size)(x))
x = LayerNorm(Linear(intermediate_size -> H)(mlp) + x)
```

Pool/head:

```text
pooled = tanh(Linear(H -> H)(last_hidden_state[:,0]))
hidden = ReLU(Linear(H -> 2H)(pooled))
logits = Sigmoid(Linear(2H -> 2)(hidden))
```

All linear layers use PyTorch `nn.Linear` default bias unless stated otherwise. The visual grid conv has `bias=False`.

## 6. Attention requirements

TVP requires only encoder-style dense self-attention for the primary target.

- Causal: no.
- Cross-attention: no separate cross-attention module; fusion is done by sequence concatenation followed by bidirectional self-attention.
- MHA/GQA/MQA: MHA only.
- Head count/head dim: base `12 x 64`; tiny `4 x 32`.
- Query/key/value widths: all equal to `hidden_size`.
- Rectangular attention: no; Q and K/V lengths are the same fused sequence length.
- Masking: optional caller text mask is extended with ones for the 10 prompt tokens and all visual tokens, then converted to additive extended attention mask by `PreTrainedModel`.
- Packed/varlen: none.
- Sliding/local attention: none.
- RoPE/ALiBi/relative bias: none.
- KV cache: none. Config `use_cache` must be ignored or rejected for DinoML TVP.
- FlashAttention/SDPA compatibility: mathematically compatible with noncausal dense attention if additive mask and output-attention behavior are handled. Source uses eager matmul/softmax/dropout/matmul.

Independently cacheable state: visual encoder outputs or visual tokens can be cached per video before text fusion, but this is not a KV cache and changes only if frames, visual prompts, interpolation mode, or backbone weights change.

## 7. Position encoding and custom math

Text uses absolute learned position embeddings.

Visual uses learned row and column embeddings:

```python
row = row_position_embeddings(torch.arange(min(max_rows, H))).view(1, Hc, 1, hidden)
col = col_position_embeddings(torch.arange(min(max_cols, W))).view(B, 1, Wc, hidden)
pos = row + col
grid = grid + maybe_bicubic_interpolate(pos, H, W)
```

Interpolation is conditional on `interpolate_pos_encoding=True` and `H` or `W` exceeding configured table sizes. It permutes `[B,H,W,C] -> [B,C,H,W]`, calls bicubic `interpolate(..., align_corners=False)`, then permutes back.

Visual prompt math is pixel-space parameter injection before the backbone. For `framepad`, four learned border tensors are concatenated around a zero center and added to `pixel_values` when apply mode is `replace` or `add`. Source `replace` does not actually zero a border region in `TvpFramePadPrompter`; it multiplies by an all-ones mask, then adds prompt. DinoML should preserve source behavior for parity.

## 8. Preprocessing and input packing

Model-coupled processor output:

- `pixel_values`: `[batch, frames, 3, height, width]`, channels-first.
- Official preprocessor config for inspected TVP repos: resize on, center crop off, rescale off, normalize on, pad on, custom mean/std, pad to 448x448, tokenizer `bert-base-uncased`.
- Current source class defaults differ: center crop on, rescale on, ImageNet mean/std, `do_pad` field. Checkpoint configs should override these.
- Channel order is flipped RGB to BGR by default in the source processors.
- Frame extraction/sampling policy is not implemented in the inspected source. Callers must supply the selected frames.

Text packing:

- The processor's text defaults request truncation and max-length padding and suppress token type IDs.
- Model forward does not accept `token_type_ids`; text token type IDs are internally all zeros.
- Attention mask is expected as `[B,T]` for text only; TVP prepends 10 ones and appends visual-token ones.

No modality placeholder tokens, masked scatter, grid metadata tensor, cu-seqlens, generation controller, or packed sequence descriptors are used.

## 9. Graph rewrite / lowering opportunities

### Rewrite: visual grid conv/pool region to guarded NCHW provider region

Source pattern:

```text
ResNet feature map NCHW -> Conv2d 3x3 pad1 -> MaxPool2d 2x2 stride2 -> ReLU -> permute to NHWC-like grid
```

Replacement: keep the region in NCHW through conv/pool/ReLU and materialize NHWC only at the visual-token flatten boundary.

Preconditions:

- Backbone returns image-like NCHW feature map.
- Conv is `groups=1`, `bias=False`, stride 1, padding 1.
- MaxPool is `kernel=2`, `stride=2`.
- Consumer is exactly TVP visual embedding flatten/row-column position path.

Failure cases: non-ResNet or non-image-like backbone feature outputs, changed pooling, or consumers that expect NHWC before this boundary.

Parity sketch: compare visual tokens after `TvpVisualInputEmbedding` for random frame tensors and fixed weights.

### Rewrite: dense encoder attention to fused noncausal attention

Source pattern:

```text
Q = Linear(x); K = Linear(x); V = Linear(x)
scores = QK^T / sqrt(head_dim) + mask
probs = softmax(scores)
context = probs V
```

Replacement: fused dense noncausal attention, optionally after separate or packed QKV projection.

Preconditions:

- `hidden_size % num_attention_heads == 0`.
- No output attentions requested for optimized path, or backend can return dense attentions.
- Dropout disabled in inference.
- Additive mask semantics match Transformers extended mask.

Weight transform: optional pack Q/K/V weights in source split order `[q, k, v]`; each source weight is standard PyTorch linear `[out, in]`.

### Rewrite: TVP frame temporal mean precompute

Source pattern:

```text
per-frame feature grid -> mean(dim=frames) -> visual tokens
```

Replacement: cache mean-pooled visual grid or visual tokens for repeated text queries over the same video.

Preconditions:

- Same frames and same `interpolate_pos_encoding` flag.
- Same visual prompt parameters and backbone weights.
- No training dropout.

Failure cases: interactive prompt changes, different frame sampling, or any requested hidden state before mean pooling.

### Rewrite: postprocess outside graph

Source pattern: `sigmoid logits` then Python processor scales by duration.

Replacement: keep model graph output as normalized `[start,end]`; do duration scaling on CPU/control plane.

Preconditions: end-to-end API owns `video_durations`; model parity tests compare normalized logits first.

## 10. Kernel fusion candidates

Highest priority:

- ResNet visual branch provider coverage by composition with the ResNet audit, because TVP depends on a CNN backbone before any Transformer work.
- Encoder QKV projection + dense noncausal attention + output projection, because fused sequence length includes text and visual grid tokens.
- LayerNorm + residual patterns in attention/FFN blocks.

Medium priority:

- Conv2d 3x3 + MaxPool + ReLU visual grid region, keeping NCHW until flatten.
- FFN Linear + GELU + Linear with residual LayerNorm.
- Embedding sum + LayerNorm for text and visual tokens.

Lower priority:

- Visual row/column position interpolation. Important for high-resolution parity but not hot for fixed 448x448 base inference.
- Grounding head MLP fusion.
- Processor acceleration; first integration can keep preprocessing on CPU.

## 11. Runtime staging plan

Stage 1: parse config and load weights for `TvpForVideoGrounding`; reject `framedownpad` unless a checkpoint supplies `frame_num` or DinoML adds a compatibility shim.

Stage 2: compose the ResNet backbone audit/runtime and verify TVP visual branch through grid output for fixed `[B,F,3,448,448]` inputs.

Stage 3: implement text embeddings, visual row/column embeddings, prompt concatenation, and attention-mask packing; validate fused input sequence before encoder.

Stage 4: run one encoder layer parity, then full encoder parity with eager dense attention.

Stage 5: add pooler/head parity and processor postprocess parity for normalized and duration-scaled times.

Stage 6: enable optimized noncausal attention and guarded NCHW visual layout region.

Stage 7: add visual-token caching for multi-query-per-video workloads.

Can stub initially: training losses, output attentions, output hidden states, visual position interpolation for first fixed-size checkpoint, and CPU preprocessing inside DinoML.

## 12. Parity and validation plan

- Config tests: parse all three inspected configs and confirm effective defaults for omitted fields.
- Processor ABI tests: given synthetic frames, confirm output key/shape and channel flip/pad/normalize behavior against Transformers processor for each checkpoint preprocessor config.
- Visual prompter tests: compare `framepad` pixel output for `replace`, `add`, and `remove`; add an explicit negative test for unshimmed `framedownpad`.
- Visual branch parity: random `[B,F,3,448,448]`, compare TVP grid after conv/pool/ReLU/permute.
- Embedding parity: compare text embeddings, visual embeddings, attention-mask concatenation, and final fused sequence.
- Single-layer encoder parity: fp32 tolerance around `1e-5` absolute/relative; fp16 around `1e-2`.
- Full `TvpModel` parity: compare `last_hidden_state` and `pooler_output`.
- Head parity: compare normalized logits from `TvpForVideoGrounding`.
- End-to-end postprocess parity: compare rounded seconds for representative `video_durations`.
- Cache negative test: passing or expecting cache state should be rejected because source does not implement it.

## 13. Performance probes

- CPU preprocessing throughput: frame count sweep 1, 48, 64 and resolution sweep.
- Backbone-only throughput: `[B*F,3,448,448]` ResNet stage output.
- Visual branch throughput: backbone + conv/pool/ReLU + temporal mean.
- Encoder-only throughput: sequence length sweep from tiny grid/text to base grid/text.
- Attention backend comparison: eager dense vs fused noncausal attention.
- Batch-size sweep: `B=1,2,4` with `F=48/64`.
- Visual-token cache probe: one video with many text queries, measuring saved backbone/visual embedding time.
- Memory probe: activation memory for fused sequence attention at expected visual grid sizes.

## 14. Skip/defer list

- Training losses (`TvpLoss`) and label supervision.
- Output attentions and hidden states on optimized fast paths.
- Autoregressive generation, beam search, KV cache, and decoding.
- `framedownpad` until the `frame_num`/`num_frames` source mismatch is handled.
- High-resolution interpolation path until fixed 448x448 parity is complete.
- Owning frame extraction/video decoding in DinoML; accept preselected frames first.
- Non-ResNet or custom backbone configs unless separately audited.

## 15. Final implementation checklist

- [ ] Parse `TvpConfig` and checkpoint processor config, including legacy processor fields.
- [ ] Compose/allowlist supported ResNet `backbone_config` variants.
- [ ] Load TVP weights with prompt, row/column embedding, text prompt, encoder, pooler, and head parameters.
- [ ] Implement visual prompter `framepad` source behavior.
- [ ] Implement `[B,F,C,H,W]` visual branch and guarded layout transitions.
- [ ] Implement text and visual embeddings plus sequence concatenation.
- [ ] Implement additive attention-mask packing for prompt/text/visual tokens.
- [ ] Implement dense bidirectional MHA encoder blocks.
- [ ] Implement pooler and two-layer sigmoid grounding head.
- [ ] Implement duration-scaling postprocess outside the graph.
- [ ] Add negative/admission checks for cache expectations and unsafe `framedownpad`.
- [ ] Add parity tests for processor ABI, visual branch, one encoder layer, full model logits, and postprocess.
- [ ] Benchmark preprocessing, backbone, encoder, attention backend, and visual-token caching.
