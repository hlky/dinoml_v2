# PPFormulaNet Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: PaddlePaddle/PP-FormulaNet-L_safetensors, PaddlePaddle/PP-FormulaNet_plus-L_safetensors
Config source: HF config.json / processor_config.json / tokenizer_config.json snapshots in _sources/
Source files inspected: configuration_pp_formulanet.py, modeling_pp_formulanet.py, image_processing_pp_formulanet.py, processing_pp_formulanet.py, modular_pp_formulanet.py
Any missing files or assumptions: no code tests/imports run; PaddleOCR deployment configs for PP-FormulaNet-S/plus-S/plus-M are not native Transformers configs for this source basis
```

Authoritative implementation note: the generated files state they are produced from `modular_pp_formulanet.py`; future Transformers source edits should target the modular file. DinoML should audit generated `modeling_pp_formulanet.py` for runtime parity because that is the code users import at this commit.

Primary runtime target for DinoML: image-to-LaTeX/formula OCR generation using `PPFormulaNetForConditionalGeneration.generate(pixel_values=...)` followed by processor decode/postprocess.

## 2. High-level architecture

PPFormulaNet is a vision encoder plus autoregressive text decoder for formula recognition. It is not a text-only LM and it does not stitch image embeddings into text token positions. The processor prepares a normalized image tensor; the vision stack produces a compact encoder sequence; an MBart-like decoder performs causal generation with cross-attention to that sequence; an untied LM head produces tokenizer logits.

```text
image preprocessing -> NCHW pixel_values
  -> Conv2d patch embed -> channels-last vision map
  -> window/global vision attention blocks -> vision neck NCHW map
  -> stride-2 conv projector -> encoder sequence [B, Senc, 512]
  -> decoder prefill/decode with self KV cache + cross-attention cache
  -> logits -> tokenizer decode -> formula text normalization
```

Stage decomposition:

- CPU/data pipeline: margin crop, optional long-axis rotation, resize/thumbnail, center pad, rescale, normalize, tokenizer decode, optional `ftfy`, regex-based formula cleanup.
- Independently cacheable encoder/projector: `pixel_values -> encoder_outputs.pooler_output`, shape usually `[B, 144, 512]` for 768 input with 16 patch size and two stride-2 projector convolutions.
- Decoder prefill/decode: token embeddings, learned positions, causal self-attention cache, cross-attention over cached encoder sequence, LM head.
- Postprocess ABI: batch decode generated token ids, remove Chinese `\text{...}` wrapping, optional unicode fix, spacing normalization around LaTeX macros/operators.

## 3. Important config dimensions

Native in-library configs:

| Field | PP-FormulaNet-L_safetensors | PP-FormulaNet_plus-L_safetensors | Source/default note |
|---|---:|---:|---|
| `model_type` | `pp_formulanet` | `pp_formulanet` | HF `config.json` |
| image size | 768 | 768 | HF `vision_config` |
| input channels | 3 | 3 | HF `vision_config` |
| patch size | 16 | 16 | HF `vision_config` |
| patch grid | 48 x 48 | 48 x 48 | inferred from config |
| vision hidden | 768 | 768 | HF `vision_config` |
| vision layers / heads | 12 / 12 | 12 / 12 | HF `vision_config` |
| vision head dim | 64 | 64 | inferred, `768 / 12` |
| window size | 14 | 14 | HF `vision_config` |
| global attention indexes | `[2,5,8,11]` | `[2,5,8,11]` | HF `vision_config` |
| vision MLP dim | 3072 | 3072 | HF `vision_config` |
| neck output channels | 256 | 256 | HF `vision_config` |
| projector conv mid/out | 512 / 1024 | 512 / 1024 | HF `vision_config` |
| projector decoder width | 512 | 512 effective | L config explicit; plus-L omits field and source default supplies 512 |
| decoder hidden `d_model` | 512 | 512 | HF `text_config` |
| decoder layers / heads | 8 / 16 | 8 / 16 | HF `text_config` |
| decoder head dim | 32 | 32 | inferred, `512 / 16` |
| decoder FFN | 2048 | 2048 | HF `text_config` |
| vocab size | 50000 | 50000 | HF `text_config` |
| max positions | 1024 | 2560 | HF `text_config`; operator-significant |
| token ids | bos 0, pad 1, eos/start 2 | same | HF `text_config` / generation config |
| tie embeddings | false | false | HF `text_config`; LM head is separate |
| generation `max_length` | 1537 | 1537 | HF `generation_config.json`; note plus-L model max position is 2560 |

Representative checkpoint sweep:

| Model id | Native Transformers basis? | Operator-significant facts |
|---|---|---|
| `PaddlePaddle/PP-FormulaNet-L_safetensors` | yes | Same topology as plus-L, shorter decoder position table: `max_position_embeddings=1024`. |
| `PaddlePaddle/PP-FormulaNet_plus-L_safetensors` | yes | Same topology, longer decoder position table: `max_position_embeddings=2560`; `decoder_hidden_size` omitted from config but source default is 512. |
| `PaddlePaddle/PP-FormulaNet-S` | no for this report | Fetched `config.json` is PaddleOCR deployment config, with grayscale `x` dynamic shapes spanning batch 1 to 8 at `[B,1,384,384]` and UniMERNet preprocess/postprocess; do not run through native `PPFormulaNetConfig` without a separate audit. |
| `PaddlePaddle/PP-FormulaNet_plus-S` | no for this report | Same deployment-config caveat; max sequence metadata appears in PaddleOCR preprocess, not the inspected Transformers source. |
| `PaddlePaddle/PP-FormulaNet_plus-M` | no for this report | Same deployment-config caveat; route to PaddleOCR/UniMERNet audit or fallback. |

## 3a. Family variation traps

- The native Transformers surface currently has one main topology; the most important in-library variation is decoder maximum position length.
- Do not infer support for the PaddleOCR S/M configs from this source. Their `config.json` is not a native `PPFormulaNetConfig` and uses a different preprocessing/deployment ABI.
- Processor output is NCHW RGB, but most vision blocks operate on `[B,H,W,C]`. Layout passes need explicit region boundaries.
- Vision attention alternates local window attention and full global attention by layer index. Window size 14 does not divide the 48x48 patch grid, so padding/cropping in `window_partition/window_unpartition` is semantically required.
- Relative position parameters differ between window and global layers: window layers allocate `2*14-1`, global layers allocate `2*48-1` for default 768/16.
- Vision `qkv` is a single Linear with output split order `q,k,v` after reshape to `[B, HW, 3, heads, head_dim]`.
- Decoder projections are separate `q_proj`, `k_proj`, `v_proj`; there is no GQA/MQA.
- The model uses learned absolute decoder positions with offset 2, not RoPE/ALiBi.
- `tie_word_embeddings=false`; token embeddings and LM head must stay distinct logical weights.
- Source gap: generated `PPFormulaNetForConditionalGeneration.get_encoder()` calls `self.model.get_encoder()`, but `PPFormulaNetModel` does not define it in the inspected file. Generic encoder-decoder generation helper parity should be verified or guarded.
- `PPFormulaNetTextConfig` still contains encoder-like attribute names inherited from MBart conventions, but the implemented text module is decoder-only.

## 4. Operator coverage checklist

Tensor/layout ops:

- Static shape guards for image `[B,3,768,768]` and decoder token ranks.
- `permute`, `reshape`, `flatten`, `transpose`, `contiguous`, `slice`, `gather`, `masked_fill`, `arange`, `expand`, `pad`.
- Window partition/unpartition over channels-last maps, including non-divisible spatial padding.
- NCHW/NHWC conversions at patch embedding, neck, and projector boundaries.

Neural network primitives:

- Conv2d patch embedding: `3 -> 768`, `kernel=stride=16`, bias present by PyTorch default.
- Vision LayerNorm over channels-last `[B,H,W,C]` and channels-first neck via permute-wrapped LayerNorm.
- Vision MLP: `Linear(768 -> 3072)`, GELU, `Linear(3072 -> 768)`.
- Neck: `Conv2d(768 -> 256, 1x1, bias=False)`, LayerNorm over channel, `Conv2d(256 -> 256, 3x3 pad=1, bias=False)`, LayerNorm.
- Projector: `Conv2d(256 -> 512, 3x3 stride=2 pad=1, bias=False)`, `Conv2d(512 -> 1024, 3x3 stride=2 pad=1, bias=False)`, flatten to sequence, `Linear(1024 -> 1024)`, `Linear(1024 -> 512)`.
- Decoder token embedding, learned position embedding, LayerNorm, dropout as inference no-op.
- Decoder FFN: `Linear(512 -> 2048)`, GELU, `Linear(2048 -> 512)`.
- LM head: untied `Linear(512 -> 50000, bias=False)`, optionally last-token-only through `logits_to_keep`.

Attention primitives:

- Vision local/global dense MHA with fp32 softmax and decomposed relative position add.
- Decoder causal self-attention, MHA with 16 heads, head dim 32, KV cache.
- Decoder cross-attention, MHA with query from decoder hidden states and K/V from encoder sequence.
- HF attention backend dispatch for decoder via `_attn_implementation`; encoder attention is always eager in this source.

Position/custom math:

- Vision absolute positional table `[1,48,48,768]` for default native configs.
- Vision decomposed relative position interpolation/indexing and two einsums.
- Decoder learned positions with offset 2 and cache-length-aware position ids.

Preprocessing-coupled ops:

- RGB to grayscale for margin crop; `nonzero`, min/max bounding rectangle, crop.
- Resize shortest edge, thumbnail preserving aspect ratio, center pad to target size, rescale/normalize.
- No OCR word boxes, no image placeholder ids, no masked image scatter in native source.

Generation/cache ops:

- Encoder output cache independent of decoder KV cache.
- Encoder-decoder cache with self-attention cache and cross-attention cache per decoder layer.
- `shift_tokens_right` for training/teacher-forced labels; generation uses decoder start id through generation config.

Tokenizer/postprocess ABI:

- `NougatTokenizer`/ByteLevel tokenizer files for native safetensors checkpoints.
- Special ids: `<s>=0`, `<pad>=1`, `</s>=2`, `<unk>=3`; additional formula/document markers begin at 4.
- Decode then normalize formula string; optional `ftfy` is a CPU-side postprocess dependency, not a graph op.

## 5. Layer/block breakdown

Vision encoder:

```text
pixel_values [B,3,768,768]
  -> Conv2d patch embed [B,768,48,48]
  -> permute [B,48,48,768]
  -> add abs pos [1,48,48,768]
  -> 12 x vision block
  -> neck permute to [B,768,48,48]
  -> Conv1x1/LayerNorm/Conv3x3/LayerNorm [B,256,48,48]
  -> projector conv stride2 [B,512,24,24]
  -> projector conv stride2 [B,1024,12,12]
  -> flatten+transpose [B,144,1024]
  -> Linear(1024->1024) -> Linear(1024->512)
```

Vision block, repeated 12 times:

```text
residual = x                          # x [B,H,W,768]
x = LayerNorm(x)
if local layer: x = pad + window_partition(x, 14)
q,k,v = Linear(768 -> 2304, bias=qkv_bias).reshape/split(q,k,v)
scores = (q * 1/sqrt(64)) @ k.T
scores += decomposed_relative_position(q, rel_pos_h, rel_pos_w)
attn = softmax(scores, dtype=float32).to(q.dtype)
x = attn @ v -> projection Linear(768 -> 768)
if local layer: x = window_unpartition/crop(x)
x = residual + x
x = x + Linear(3072 -> 768)(GELU(Linear(768 -> 3072)(LayerNorm(x))))
```

Decoder block, repeated 8 times:

```text
residual = x                          # x [B,T,512]
x = LayerNorm(x)
x = causal_self_attention(q/k/v Linear(512->512), cache)
x = residual + out_proj(x)
residual = x
x = LayerNorm(x)
x = cross_attention(q from x, k/v from encoder [B,Senc,512], cross cache)
x = residual + out_proj(x)
residual = x
x = LayerNorm(x)
x = Linear(2048 -> 512)(GELU(Linear(512 -> 2048)(x)))
x = residual + x
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention over image patch maps.
- MHA, not GQA/MQA.
- Heads: 12; head dim: 64.
- Local layers use windows of 14x14 with padding when the patch grid is not divisible by 14.
- Global layers at indexes 2, 5, 8, 11 attend over the full 48x48 grid for native 768/16 configs.
- Masking: no attention mask in source vision path; padding introduced by local windows is not masked because padded tokens are cropped after window unpartition. This is source behavior and should be preserved.
- Softmax explicitly computes in fp32 and casts back to query dtype.
- Relative position bias is generated from query and learned axis tables, not a simple static additive table unless dimensions are fixed and precomputed carefully.

Decoder self-attention:

- Causal autoregressive self-attention.
- MHA: 16 heads, head dim 32.
- Causal mask generated by `create_causal_mask`, supports cache length.
- KV cache shape per layer is logically `[B, 16, T_cache, 32]` for keys and values before attention backend-specific packing.
- Cached self keys/values are stored after linear projection and head transpose; no RoPE is applied.
- SDPA/Flash-compatible in principle through HF decoder attention interface, but eager fallback math is matmul -> mask add -> softmax -> dropout -> matmul.

Decoder cross-attention:

- Query length is decoder step length; key/value length is encoder sequence length, usually 144.
- Q/K/V widths all 512 with 16 heads, head dim 32.
- Cross-attention K/V may be cached after the first decode step using `EncoderDecoderCache.cross_attention_cache`.
- `encoder_attention_mask` is optional; the primary image encoder path does not produce padding masks.

## 7. Position encoding and custom math

Vision decomposed relative position is the custom math DinoML should preserve:

```python
def decomposed_rel_pos(query, rel_pos_h, rel_pos_w, q_hw, k_hw):
    rh = interpolate_rel_pos(rel_pos_h, 2 * max(q_hw[0], k_hw[0]) - 1)
    rw = interpolate_rel_pos(rel_pos_w, 2 * max(q_hw[1], k_hw[1]) - 1)
    idx_h = scaled_relative_coords(q_hw[0], k_hw[0])
    idx_w = scaled_relative_coords(q_hw[1], k_hw[1])
    Rh = rh[idx_h]                       # [Qh, Kh, head_dim]
    Rw = rw[idx_w]                       # [Qw, Kw, head_dim]
    q = query.reshape(BH, Qh, Qw, head_dim)
    rel_h = einsum("bhwc,hkc->bhwk", q, Rh)
    rel_w = einsum("bhwc,wkc->bhwk", q, Rw)
    return rel_h[..., None] + rel_w[..., None, :]
```

For native static image size, the relative tables match either 14x14 windows or 48x48 global attention. A DinoML first pass can avoid interpolation by admitting only configured static sizes matching table initialization; a more general pass needs linear interpolation and index generation.

Decoder positions are learned embeddings with offset 2. Position ids start at the self-cache length during decode. There is no RoPE or ALiBi.

## 8. Preprocessing and input packing

Processor/runtime input:

- Input images may be PIL, NumPy, or torch tensors; processor supports channels-first and channels-last input formats.
- The image processor returns `pixel_values`; tokenizer is used only for decode/postprocess in the primary image-only inference call.
- Native safetensors processor config emits torch tensors and targets `768x768`.

Preprocessing order from source:

```text
optional crop_margin:
  RGB -> grayscale -> normalize to 0..255 -> threshold < 200
  -> nonzero -> bounding rect -> crop
group by shape
optional long-axis align by rot90
resize shortest edge
thumbnail so neither dimension exceeds target
center pad to target size
rescale and normalize
return pixel_values
```

Layout ABI:

- Processor output and model input: NCHW `[B,3,768,768]`.
- Patch embedding output becomes channels-last `[B,48,48,768]`.
- Vision neck is an explicit no-layout-translation boundary: it permutes to NCHW for Conv2d and uses `PPFormulaNetLayerNorm(..., data_format="channels_first")`.
- Projector remains NCHW through its two Conv2d layers, then flattens spatial axes to sequence `[B, H*W, C]`.

Tokenizer/postprocess ABI:

- Native safetensors checkpoints use `NougatTokenizer` with ByteLevel tokenizer JSON.
- Generation config uses `decoder_start_token_id=2`, `eos_token_id=2`, `forced_eos_token_id=2`, `pad_token_id=1`, `use_cache=true`, `max_length=1537`.
- `processor.post_process` expects generated ids shaped `[B,T]` or `[T]`, calls `batch_decode(skip_special_tokens=True)`, then normalizes formula text.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d -> Linear

Source pattern:

```text
Conv2d(3 -> 768, kernel=16, stride=16, padding=0) -> permute NCHW to NHWC
```

Replacement:

```text
WindowFlatten([B,3,768,768], patch=16, raster order)
  -> GEMM([B,2304,768], weight_flat.T)
  -> bias add
  -> reshape [B,48,48,768]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Static image height/width equal config and divisible by patch size.
- Weight flatten order matches PyTorch Conv2d NCHW storage `[out,in,kh,kw]`.
- Fallback to Conv2d if dynamic image sizes or grouped/dilated variants appear.

Parity sketch: compare patch embedding output after Conv2d+permute against rewrite for random fp32/fp16 inputs.

### Rewrite: projector Conv2d stride-2 sequence projection

Source pattern:

```text
NCHW [B,256,48,48]
  -> Conv3x3 stride2 pad1 bias=False
  -> Conv3x3 stride2 pad1 bias=False
  -> flatten(2).transpose(1,2)
  -> Linear(1024->1024) -> Linear(1024->512)
```

Replacement: keep Conv2d first; optionally fuse `flatten+transpose+Linear` into a batched GEMM over raster spatial positions.

Preconditions:

- Projector convs keep NCHW semantics.
- Flatten order is row-major over `H,W`.
- Linear weights are dense row-major logical weights.
- Do not translate projector to NHWC unless both Conv2d lowering and flatten order are rewritten together.

### Rewrite: channels-first LayerNorm via layout-aware kernel

Source pattern:

```text
permute NCHW -> NHWC -> LayerNorm(C) -> permute NHWC -> NCHW
```

Replacement: native channels-first per-pixel LayerNorm over channel dimension.

Preconditions:

- Normalized shape equals channel count.
- Axis is exactly channel dimension of NCHW tensor.
- Consumer remains NCHW.
- Validate epsilon and affine weight/bias behavior.

### Rewrite: fixed-size vision relative bias precompute

Source pattern: interpolate/index relative position tables per attention call.

Replacement: for fixed native image sizes, precompute relative index tensors or pre-expanded relative tables per layer/window shape; keep query-dependent einsums in graph.

Preconditions:

- Static `q_size == k_size` for either 14x14 local windows or 48x48 global maps.
- Relative parameter table length matches source initialization.
- No dynamic input image size or patch size.

Failure case: if q/k sizes differ, source scales coordinates and interpolates; require fallback or full implementation.

### Rewrite: decoder last-token logits

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :])
```

Replacement: during decode, compute LM head only for current token or requested `logits_to_keep`.

Preconditions:

- Caller/generation needs only last-token logits.
- Training/loss and full-sequence logits are out of scope for first inference path.

### Layout guard boundaries

Safe NHWC/channel-last candidate regions:

- Patch embedding output through all vision blocks is naturally channels-last.
- Vision LayerNorm and attention/MLP can operate on channels-last logical maps.

No-layout-translation guards:

- Raw `pixel_values` input and image processor output are NCHW.
- Patch Conv2d source expects NCHW input unless rewritten as patch flatten.
- Vision neck and projector Conv2d blocks are explicitly NCHW.
- `flatten(2).transpose(1,2)` after projector fixes raster order for cross-attention K/V.

Axis-sensitive attrs to rewrite if a layout pass crosses boundaries:

- Conv2d channel axis, LayerNorm normalized channel axis, `flatten(2)`, `transpose(1,2)`, attention reshape `[B,H,W,C] -> [B,HW,heads,head_dim]`, window partition reshape/permute axes.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d lowering/fusion: first image-stage cost and clean static rewrite to GEMM/im2col-free patch flatten.
- Vision attention with relative position: local/global attention dominates encoder; needs custom support for decomposed relative bias and fp32 softmax.
- Decoder cross-attention with cached encoder K/V: encoder sequence is short and static per image; caching K/V projections avoids repeated work during decode.
- LayerNorm + Linear/GELU FFN blocks in decoder and vision: repeated many times, conventional fusion target.
- Last-token-only LM head: vocab 50000 makes full-sequence logits expensive during decode.

Medium priority:

- Channels-first LayerNorm kernel for neck to remove permute pairs.
- Projector flatten+Linear fusion after Conv2d.
- Window partition/unpartition fused with local attention input/output staging.
- Encoder output precompute/cache API: useful for batching multiple decode strategies per image.

Lower priority:

- General dynamic relative-position interpolation; native first pass can guard to fixed config sizes.
- Training-only dropout/layerdrop/loss.
- Full PaddleOCR S/M deployment graph; requires separate source audit.

## 11. Runtime staging plan

Stage 1: config and weights

- Parse native `PPFormulaNetConfig`.
- Reject non-native PaddleOCR deployment configs unless routed to a separate importer.
- Load untied embeddings/LM head and vision relative/absolute position weights.

Stage 2: vision encoder static parity

- Admit only `[B,3,768,768]`, patch size 16, static 48x48 patch grid.
- Implement patch Conv2d, channels-last vision blocks, relative position math, neck/projector.
- Validate `encoder_outputs.pooler_output` independently.

Stage 3: decoder prefill without cache

- Implement token/position embeddings, causal mask, self-attn, cross-attn, FFN, LM head.
- Run full decoder sequence from known `decoder_input_ids`.

Stage 4: generation decode with cache

- Add self KV cache and cross-attention K/V cache.
- Ensure `pixel_values` or cached `encoder_outputs` are used only on first iteration.
- Add last-token logits path.

Stage 5: processor/postprocess parity

- Match processor image pipeline or define CPU-owned preprocessing boundary.
- Match tokenizer decode and formula normalization ABI.

Stage 6: optimize

- Add guarded patch Conv2d rewrite, channels-first LayerNorm, relative-position precompute, attention fusions, and cross-attention K/V preprojection.

Initial stubs allowed: dropout/layerdrop as inference no-ops; training loss; output attentions/hidden-state capture; `ftfy` absence can match source warning behavior.

## 12. Parity and validation plan

- Processor parity: fixed image fixtures through crop/resize/thumbnail/pad/normalize; compare `pixel_values` to Transformers processor.
- Patch embedding parity: random `[B,3,768,768]` through Conv2d+permute and any patch-linear rewrite.
- Relative position unit tests: local 14x14 and global 48x48, compare decomposed bias tensors for random query.
- Vision block parity: one local layer and one global layer fp32, then fp16 tolerance after fp32 baseline.
- Encoder parity: full vision encoder/projector, compare `last_hidden_state` `[B,256,48,48]` and `pooler_output` `[B,144,512]`.
- Decoder block parity: self-attn only, cross-attn only, and combined block with synthetic encoder sequence.
- Cache parity: prefill + one-token decode should match full-sequence decode logits within tolerance.
- End-to-end formula OCR: image -> generate ids -> processor postprocess text against Transformers.

Recommended tolerances: fp32 `rtol=1e-4, atol=1e-4` for block outputs; fp16/bf16 `rtol=5e-3, atol=5e-3` for logits/hidden states, with stricter fp32 softmax reference for attention score debugging.

## 13. Performance probes

- CPU preprocessing throughput by image size and batch size, split by crop-margin on/off.
- Vision encoder throughput for batch sizes 1, 4, 8, with local/global attention timing separated.
- Projector throughput and resulting encoder sequence cache size.
- Decoder prefill latency for target lengths 64, 256, 1024, 1537.
- Decode tokens/sec with cached encoder and self KV cache.
- Cross-attention K/V cache memory and preprojection benefit for `Senc=144`.
- LM head full-sequence versus last-token-only timing.
- Layout strategy comparison: faithful NCHW/NHWC transitions versus guarded patch-linear/channel-first-LN optimized path.
- Relative position implementation comparison: dynamic interpolation/indexing versus fixed-shape precomputed index/table path.

## 14. Skip/defer list

- Training, labels/loss, gradient checkpointing, LayerDrop behavior.
- Output attentions/hidden states beyond parity debugging.
- Beam-search-specific scheduling; greedy/sampling decode is enough for first runtime.
- PaddleOCR deployment configs for S/M/plus-S unless separately audited.
- General dynamic image sizes, dynamic patch sizes, or non-768 native shapes.
- General layout translation across Conv2d boundaries.
- Quantization and packed weights; inspected native source has dense weights.
- Multi-GPU/tensor parallel.

## 15. Final implementation checklist

- [ ] Parse native `PPFormulaNetConfig` and reject/reroute PaddleOCR deployment configs.
- [ ] Load dense vision, decoder, and untied LM head weights.
- [ ] Implement/validate processor boundary for `pixel_values` or document CPU-owned preprocessing.
- [ ] Implement patch Conv2d and static image shape guards.
- [ ] Implement channels-last vision LayerNorm, MLP, residuals, and window partition/unpartition.
- [ ] Implement vision decomposed relative position attention.
- [ ] Implement NCHW neck Conv2d + channels-first LayerNorm.
- [ ] Implement projector Conv2d -> sequence -> Linear stack.
- [ ] Implement decoder token/position embedding and masks.
- [ ] Implement decoder causal self-attention and cross-attention.
- [ ] Implement encoder-decoder cache ABI and last-token logits.
- [ ] Implement tokenizer decode/postprocess ABI outside the graph.
- [ ] Add patch Conv2d -> Linear rewrite with strict guards.
- [ ] Add channels-first LayerNorm rewrite/kernel for neck.
- [ ] Add relative-position fixed-shape precompute path.
- [ ] Add one-layer, encoder, decoder-cache, and end-to-end formula OCR parity tests.
- [ ] Benchmark preprocessing, encoder, prefill, decode, LM head, and layout variants.
