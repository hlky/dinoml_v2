# Nougat Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/nougat-small, facebook/nougat-base
Config source: public Hugging Face config/preprocessor/generation/tokenizer JSON fetched from main
Source files inspected:
  src/transformers/models/nougat/configuration_nougat.py
  src/transformers/models/nougat/processing_nougat.py
  src/transformers/models/nougat/image_processing_nougat.py
  src/transformers/models/nougat/image_processing_pil_nougat.py
  src/transformers/models/nougat/tokenization_nougat.py
  src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py
  src/transformers/models/donut/configuration_donut_swin.py
  src/transformers/models/donut/modeling_donut_swin.py
  src/transformers/models/mbart/configuration_mbart.py
  src/transformers/models/mbart/modeling_mbart.py
Any missing files or assumptions: no modeling_nougat.py exists; official Nougat neural execution is VisionEncoderDecoderModel over donut-swin encoder + MBart causal decoder. Official public configs found for small/base only; no gated gaps encountered.
```

Snapshots are under `agents/plans/transformers/nougat/_sources/`, including fetched `facebook_nougat_{small,base}_{config,preprocessor_config,generation_config,tokenizer_config}.json`.

## 2. High-level architecture

Nougat is document image OCR generation:

```text
document image preprocessing -> DonutSwin encoder -> MBart causal decoder prefill/decode -> logits/sampling -> tokenizer Markdown postprocess
```

Stage decomposition:

- CPU/data pipeline: crop white margins, optional rotate, shortest-edge resize, thumbnail, centered pad to 896x672, rescale, ImageNet normalize, output `pixel_values` in NCHW by default.
- Encoder: `DonutSwinModel` consumes `[B,3,896,672]`, emits final image token sequence `[B,588,1024]` for official configs.
- Decoder: `MBartForCausalLM` with self-attention KV cache and encoder-decoder cross-attention over the full encoded image sequence.
- Generation controller: uses BOS/EOS/PAD IDs from generation/tokenizer config; source generation config only records BOS=0, EOS=2, forced EOS=2, PAD=1.
- Postprocess: Nougat tokenizer regex/NLTK/Levenshtein cleanup converts raw decoded text into Markdown-like output.

The encoder output is independently cacheable per page image; decoder prefill/decode can be validated separately once encoder hidden states are available.

## 3. Important config dimensions

| Field | facebook/nougat-small | facebook/nougat-base | Source |
| --- | ---: | ---: | --- |
| top architecture | `VisionEncoderDecoderModel` | `VisionEncoderDecoderModel` | config.json |
| encoder model type | `donut-swin` | `donut-swin` | config.json |
| image size | `[896,672]` | `[896,672]` | config.json / preprocessor |
| patch size | 4 | 4 | config.json |
| patch grid | `224 x 168 = 37632` | same | inferred from config |
| encoder embed dim | 128 | 128 | config.json |
| encoder depths | `[2,2,14,2]` | `[2,2,14,2]` | config.json |
| encoder heads | `[4,8,16,32]` | `[4,8,16,32]` | config.json |
| encoder dims by stage | `[128,256,512,1024]` | same | inferred from config |
| final encoder grid | `28 x 21 = 588` | same | inferred from patch merging |
| window size | 7 | 7 | config.json |
| qkv bias | true | true | config.json |
| relative position bias | per Swin window/head | same | source |
| decoder model type | `mbart` | `mbart` | config.json |
| decoder layers used by MBartDecoder | 4 | 10 | config.json |
| decoder d_model | 1024 | 1024 | config.json |
| decoder heads / head dim | `16 / 64` | `16 / 64` | config + source |
| decoder FFN | 4096 | 4096 | config.json |
| max decoder positions | 3584 | 4096 | config.json |
| vocab size | 50000 | 50000 | config.json |
| activation | GELU | GELU | config.json |
| scale embedding | true | true | config.json |
| use_cache | true | true | config.json |
| tied word embeddings | false | false | config.json |
| tokenizer max length | 4096 | 4096 | tokenizer_config.json |

Processor config is identical for small/base: `do_crop_margin=true`, `do_resize=true`, `do_thumbnail=true`, `do_pad=true`, `do_rescale=true`, `do_normalize=true`, `do_align_long_axis=false`, mean/std are ImageNet defaults, target size is 896x672.

## 3a. Family variation traps

- The encoder is `donut-swin`, not plain `swin`. It is copied from Swin but owns the exact class names and checkpoint keys.
- `config.decoder.num_hidden_layers` may show MBart defaults/history, but MBartDecoder source uses `decoder_layers`; use 4 for small and 10 for base.
- Top-level `decoder_start_token_id`, `pad_token_id`, and `eos_token_id` are absent in fetched `config.json`; generation/tokenizer files provide BOS/PAD/EOS. DinoML config loading should normalize generation metadata before decode.
- The fixed official image size makes every stage divisible by window size: `224x168`, `112x84`, `56x42`, `28x21`. Dynamic image sizes need padding guards for patch size, 2x patch merging, and window size.
- DonutSwin source enters local NHWC shape for window partition/roll/reverse, while processor and patch conv are NCHW. Treat NHWC as a guarded local layout opportunity, not a global semantic default.
- Decoder self-attention and cross-attention share MBartAttention ABI but use different cache ownership under `EncoderDecoderCache`.
- Cross-attention has no encoder padding mask in `VisionEncoderDecoderModel.forward`; `encoder_attention_mask = None`.
- Nougat tokenizer postprocessing requires optional `nltk` and `Levenshtein`; this is end-to-end parity work outside the neural graph.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor input, dtype float32 after preprocessing.
- `Conv2d(3 -> 128, kernel=4, stride=4, padding=0)` patch embedding.
- Flatten spatial to sequence: `[B,128,224,168] -> [B,37632,128]`.
- NHWC view/reshape/transpose for window partition and reverse.
- `torch.roll` cyclic shift on H/W axes for shifted windows.
- Padding for dynamic patch/window/merge divisibility.
- Patch merging: NHWC gather/slice interleave four 2x2 neighborhoods, concat channels, LayerNorm over `4C`, Linear `4C -> 2C`.
- Final optional pooling is present in DonutSwinModel but not needed by VisionEncoderDecoder hidden-state path.

Neural primitives:

- LayerNorm epsilon `1e-5` in encoder/decoder.
- Linear projections with bias in encoder attention Q/K/V/out and decoder attention/FFN.
- Encoder MLP: Linear `C -> 4C`, GELU, Linear `4C -> C`.
- Decoder MLP: Linear `1024 -> 4096`, GELU, Linear `4096 -> 1024`.
- Dropout/DropPath are training-time or disabled at inference except deterministic identity handling.
- Token embedding scaled by `sqrt(1024)` because `scale_embedding=true`.
- Learned positional embedding with MBart offset `+2`.
- LM head Linear `1024 -> 50000`, bias false.

Attention primitives:

- DonutSwin local noncausal MHA per 7x7 window, relative position bias table of `(13*13, heads)`.
- Shifted-window attention mask with 0 and `-100.0` additive values.
- MBart causal decoder self-attention, MHA `16 x 64`, KV cache.
- MBart dense cross-attention from decoder queries to encoder sequence `[B,588,1024]`; cross K/V can be cached after first decode step.
- Attention backend uses `ALL_ATTENTION_FUNCTIONS` with eager fallback; SDPA/Flash-like backends must preserve mask and cache ABI.

Position/relative-bias ops:

- DonutSwin relative position index construction for 7x7 windows.
- MBart learned absolute positions for decoder token positions, offset by 2 and shifted by self-cache length.

Generation/cache ops:

- `EncoderDecoderCache(DynamicCache self, DynamicCache cross)`.
- Self cache grows with generated sequence length per decoder layer: keys/values `[B,16,T,64]`.
- Cross cache stores projected encoder K/V per layer: keys/values `[B,16,588,64]` for official image size.
- Logits can be sliced through `logits_to_keep` in MBartForCausalLM.

Preprocessing-coupled ops:

- Grayscale conversion, min/max normalization to `[0,255]`, threshold `<200`, nonzero bounding rect crop.
- Shortest-edge resize to 672, then thumbnail preserving aspect ratio under 896x672, then centered pad.
- Rescale by `1/255`, normalize with ImageNet mean/std.

Discrete/tokenizer ops:

- BPE tokenizer with NFKC normalizer, digit splitting, punctuation splitting, newline splitting, byte-level pretokenizer/decoder.
- Template adds BOS and EOS around a single sequence.
- Postprocess regex cleanup for Markdown, LaTeX/table formatting, repeated tails, hallucinated references.

## 5. Layer/block breakdown

Processor:

```text
image -> crop_margin -> optional rotate -> resize(shortest_edge=672)
      -> thumbnail(max 896x672) -> center pad to 896x672
      -> rescale -> normalize -> pixel_values [B,3,896,672]
```

DonutSwin patch stem:

```text
x = Conv2d(3, 128, kernel=4, stride=4)(pixel_values)
x = flatten_hw_transpose(x)                  # [B,37632,128]
x = LayerNorm(x)
```

DonutSwin layer, repeated by stage depths `[2,2,14,2]` with alternating shift:

```text
shortcut = x
x = LayerNorm(x)
x = view [B,H,W,C]
x = pad_to_window_multiple(x)
x = roll(x, -shift, -shift) if shifted
w = window_partition(x, 7)                   # [B*num_windows,49,C]
w = MHA(q,k,v) + relative_position_bias + optional shifted mask
w = window_reverse(w)
w = reverse_roll_and_unpad(w)
x = shortcut + attention_output
y = LayerNorm(x)
y = Linear(C -> 4C) -> GELU -> Linear(4C -> C)
x = x + y
```

After each stage except the last:

```text
x = view [B,H,W,C]
x = concat([x[:,0::2,0::2], x[:,1::2,0::2], x[:,0::2,1::2], x[:,1::2,1::2]], dim=-1)
x = LayerNorm(4C)
x = Linear(4C -> 2C, bias=False)
```

Official final encoder output:

```text
[B,37632,128] -> [B,9408,256] -> [B,2352,512] -> [B,588,1024]
```

VisionEncoderDecoder bridge:

```text
encoder_hidden_states = encoder(pixel_values)[0]
if encoder.hidden_size != decoder.hidden_size: Linear bridge
decoder(..., encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=None)
```

Official Nougat does not need `enc_to_dec_proj` because both widths are 1024.

MBart decoder layer, repeated 4 or 10 times:

```text
residual = x
x = LayerNorm(x)
x = causal_self_attention(x, self_cache, causal_mask)
x = residual + x

residual = x
x = LayerNorm(x)
x = cross_attention(query=x, key_value=encoder_hidden_states, cross_cache)
x = residual + x

residual = x
x = LayerNorm(x)
x = Linear(1024 -> 4096) -> GELU -> Linear(4096 -> 1024)
x = residual + x
```

Decoder output:

```text
x = final LayerNorm(x)
logits = Linear(1024 -> 50000, bias=False)(x[:, logits_to_keep])
```

## 6. Attention requirements

Encoder attention:

- Noncausal self-attention inside fixed local windows.
- Standard MHA, no GQA/MQA.
- Per stage: heads `[4,8,16,32]`, head dim always 32 for stage dims `[128,256,512,1024]`.
- Window query/key length is 49.
- Relative position bias is added before shifted-window mask and softmax.
- Shifted layers use cyclic roll, partition, additive mask with `-100.0`, reverse partition, reverse roll.
- No KV cache.

Decoder attention:

- Causal self-attention, MHA `16 x 64`, cacheable.
- Dense encoder-decoder cross-attention, MHA `16 x 64`, rectangular attention with query length `T_dec` and key/value length 588 for official configs.
- `EncoderDecoderCache` separates self cache and cross cache. Cross K/V are projected once per layer and marked updated, then reused.
- Cached keys are stored after linear projection and reshape/transpose; no RoPE is applied.
- Causal mask is created from current inputs, attention mask, and self-cache length; cross mask is bidirectional and absent when `encoder_attention_mask=None`.
- Flash/SDPA compatibility is plausible for decoder dense attention if the implementation preserves MBart mask creation, cross-cache reuse, and dropout=0 in inference.

## 7. Position encoding and custom math

DonutSwin relative position index:

```python
coords = meshgrid(arange(Wh), arange(Ww))
relative = flatten(coords)[:, :, None] - flatten(coords)[:, None, :]
relative = permute(relative, (1, 2, 0))
relative[..., 0] += Wh - 1
relative[..., 1] += Ww - 1
relative[..., 0] *= 2 * Ww - 1
index = relative.sum(-1)
bias = table[index.reshape(-1)].reshape(Wh*Ww, Wh*Ww, heads).permute(2,0,1)
```

This is static for a fixed window size and can be precomputed per stage/head count. The bias table is learned.

MBart positions:

```python
positions = arange(past_len, past_len + seq_len).expand(batch, -1)
position_embedding = learned_positions(positions + 2)
hidden = scaled_token_embedding + position_embedding
```

Position IDs depend on decode step/self-cache length.

## 8. Preprocessing and input packing

Nougat preprocessing is model-coupled and affects encoder shapes. The processor emits `pixel_values`; there are no OCR words/boxes supplied to the neural graph.

Default CPU/data-pipeline work:

- Input image converted to tensor/array in channel-first format.
- `crop_margin`: RGB to grayscale, min/max normalize grayscale to 0..255, threshold `<200`, find nonzero coords, crop bounding rectangle.
- `align_long_axis`: disabled by official preprocessor config; if enabled, rotates 90 degrees when source orientation conflicts with target orientation.
- `resize`: shortest edge is `min(896,672)=672`, preserving aspect ratio.
- `thumbnail`: ensure both dimensions fit within 896x672 using bicubic.
- `pad_images`: centered pad to exact target.
- `rescale_and_normalize`: `x / 255`, then `(x - mean) / std`.

Runtime input contract for first DinoML graph should accept already preprocessed `[B,3,896,672]` tensors. Full end-to-end parity needs a separate processor implementation or a trusted Python preprocessing boundary.

Generation/controller work outside the core module:

- Decode starts from BOS or caller-provided decoder IDs; fetched generation config does not include beam settings beyond token IDs.
- Enforce forced EOS token ID 2 when using HF generation parity.
- After token decode, call `NougatTokenizer.post_process_generation` for Markdown cleanup. This requires `nltk` and `Levenshtein`; it should be optional in a first neural parity target.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d -> GEMM

Source pattern:

```text
Conv2d(3 -> 128, kernel=4, stride=4, padding=0) on NCHW
```

Replacement:

```text
WindowFlatten over non-overlapping 4x4 patches -> GEMM([48] x [48,128]) -> reshape [B,224,168,128] -> sequence
```

Preconditions:

- `kernel_size == stride == patch_size`
- `padding == 0`, `dilation == 1`, `groups == 1`
- input H/W divisible by patch size or explicit source-equivalent padding is applied first
- flatten order matches PyTorch Conv2d NCHW memory semantics

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw).T
b = conv.bias
```

Failure cases: dynamic image sizes without padding equivalence, non-contiguous input layout, changed patch size/channel count.

Parity test: compare patch embedding output before LayerNorm for random `[1,3,896,672]` and edge sizes requiring patch padding.

### Rewrite: fixed-size Swin windows to batched GEMM attention

Source pattern:

```text
view/roll/window_partition -> Q,K,V linears -> matmul 49x49 -> bias/mask/softmax -> matmul V -> reverse
```

Replacement:

```text
batched local-window attention over [B*num_windows, heads, 49, head_dim]
```

Preconditions:

- window size is 7 and static
- mask is either absent or the standard shifted-window mask for the stage resolution
- relative bias table/index are static for the window
- source NHWC partition order is preserved

Failure cases: arbitrary dynamic window size, changed attention mask semantics, layout pass crossing the NCHW patch conv boundary without rewriting axes.

Parity test: per layer with and without shift at each stage resolution.

### Rewrite: patch merging as layout-aware gather + GEMM

Source pattern:

```text
concat four 2x2 NHWC strided slices -> LayerNorm(4C) -> Linear(4C -> 2C)
```

Replacement:

```text
specialized 2x2 pack kernel or view-aware gather -> LayerNorm -> GEMM
```

Preconditions:

- exact slice order `[row0 col0, row1 col0, row0 col1, row1 col1]`
- even H/W or source-equivalent bottom/right zero padding before pack
- NHWC local layout preserved

Failure cases: using common Swin order assumptions without checking DonutSwin concat order.

### Rewrite: decoder cross-attention K/V precompute

Source pattern:

```text
for each decoder layer: k_proj/v_proj(encoder_hidden_states) on first decode step, then cache
```

Replacement:

```text
encoder output -> per-layer cross K/V projection cache -> decode consumes static cross cache
```

Preconditions:

- encoder output unchanged across generated tokens
- decoder layer weights fixed
- batch/beam reorder updates cache consistently

Failure cases: generation paths with encoder output mutation, beam reorder bugs, missing cache update flag parity.

## 10. Kernel fusion candidates

Highest priority:

- DonutSwin window attention with relative bias and shifted mask. It dominates encoder work and has tiny fixed `49x49` attention where generic attention overhead can matter.
- LayerNorm + Linear/GELU/Linear blocks for encoder and decoder.
- Decoder self-attention and cross-attention with cache ABI. This is required for generation throughput.
- Cross-attention K/V projection precompute per layer.

Medium priority:

- Patch Conv2d -> GEMM or specialized patch embedding kernel for fixed 896x672 pages.
- Patch merging pack + LayerNorm + Linear.
- Decoder last-token-only logits using `logits_to_keep=1`.
- Processor GPU fusion for rescale+normalize if preprocessing moves into runtime.

Lower priority:

- Full margin crop/thumbnail/pad on GPU. It is data-pipeline heavy and not needed for neural graph parity.
- Markdown postprocess acceleration. Regex/NLTK/Levenshtein work is CPU-side end-to-end polish.
- Pooler path in DonutSwinModel; not required by VisionEncoderDecoder generation.

## 11. Runtime staging plan

Stage 1: Parse Nougat config, normalize generation/tokenizer IDs, load already preprocessed `pixel_values`, and instantiate weights for DonutSwin + MBart decoder.

Stage 2: Encoder-only parity for `DonutSwinModel.last_hidden_state` on `[B,3,896,672]`, including relative bias, shifted masks, and patch merging.

Stage 3: Decoder cross-attention parity with fixed encoder hidden states and short decoder sequences, no generation loop.

Stage 4: Prefill and greedy decode with `EncoderDecoderCache`, including self-cache growth and cross-cache reuse.

Stage 5: End-to-end page generation with Python preprocessing and tokenizer decode/postprocess kept outside DinoML.

Stage 6: Add guarded rewrites/fusions: patch conv GEMM, local window attention, cross K/V precompute, last-token logits.

Stage 7: Optional data-pipeline ownership for preprocessing and Markdown postprocess parity.

## 12. Parity and validation plan

- Processor parity: compare Python processor outputs for representative RGB pages, blank pages, large margins, portrait inputs with `do_align_long_axis` true/false.
- Patch embedding parity: random tensors and real processed images, fp32 tolerance `1e-5`.
- DonutSwin single-layer parity: no-shift and shifted layer at each stage resolution.
- DonutSwin full encoder parity: compare final `[B,588,1024]`; fp32 `atol=1e-4`, fp16/bf16 looser after LayerNorm/softmax.
- Decoder single-layer parity: self-attention only, cross-attention only, full layer.
- Cache parity: prefill then one-token decode equals full-sequence decode logits for small/base.
- Cross-cache parity: verify cross K/V are reused after first decode step and shapes are `[B,16,588,64]`.
- Generation parity: greedy decode for one page with HF processor/model/tokenizer; compare token IDs first, Markdown postprocess second.
- Config variation parity: run both small and base because decoder layer count/context changes.

## 13. Performance probes

- Preprocessing throughput: pages/sec for crop/resize/thumbnail/pad/normalize.
- Encoder throughput by batch size for fixed 896x672.
- Encoder stage breakdown: patch embedding, each Swin stage, patch merging.
- Window attention backend comparison for `49 x 49` across stage head dims.
- Decoder prefill time by target length and batch size.
- Decode tokens/sec with self-cache and cross-cache enabled.
- Cross K/V precompute cost and memory: base has 10 decoder layers, each cross K/V cache is `[B,16,588,64]`.
- Last-token logits vs full logits projection.
- End-to-end page throughput with Python preprocessing vs preprocessed tensors.
- Memory probe for encoder activations and decoder cache across small/base max positions.

## 14. Skip/defer list

- Training, labels loss, DropPath randomness, LayerDrop, gradient checkpointing.
- DonutSwin classification/pooler heads.
- Output hidden states and attentions unless needed for debugging.
- Dynamic or arbitrary image sizes beyond source-equivalent padding guards.
- Beam search and beam cache reorder in the first greedy decode target.
- Full Markdown postprocess acceleration; keep it as CPU postprocess initially.
- Gated/remote-code checkpoint behavior; official public small/base require no remote code.
- Quantization and ONNX/community fork variants.

## 15. Final implementation checklist

- [ ] Parse `NougatConfig` as VisionEncoderDecoder with `donut-swin` encoder and `mbart` decoder.
- [ ] Normalize generation/token IDs from generation/tokenizer configs when top-level config omits them.
- [ ] Load DonutSwin patch, relative-bias, LayerNorm, MLP, and patch-merging weights.
- [ ] Load MBart causal decoder and untied LM head weights.
- [ ] Implement/pre-admit preprocessed input ABI `[B,3,896,672]`.
- [ ] Implement DonutSwin patch embedding and stage shape bookkeeping.
- [ ] Implement fixed 7x7 window attention with relative position bias.
- [ ] Implement shifted-window mask and `torch.roll`/window reverse parity.
- [ ] Implement patch merging with exact source slice order.
- [ ] Implement MBart learned positions with offset 2 and scaled token embeddings.
- [ ] Implement decoder self-attention cache and encoder-decoder cross-attention cache.
- [ ] Add encoder-only parity tests for small/base.
- [ ] Add decoder prefill/decode parity tests with cached self and cross attention.
- [ ] Add end-to-end greedy token parity with Python processor/tokenizer boundary.
- [ ] Add guarded patch Conv2d -> GEMM rewrite.
- [ ] Add local Swin window attention performance probe.
- [ ] Add cross K/V precompute and last-token logits probes.
