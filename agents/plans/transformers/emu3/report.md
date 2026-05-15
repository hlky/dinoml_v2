# Transformers family audit: emu3

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `transformers`.

Model id: native-source target is `BAAI/Emu3-Chat-hf` / `BAAI/Emu3-Gen-hf`, both `model_type="emu3"` and `architectures=["Emu3ForConditionalGeneration"]`. Remote-code repositories with `model_type="Emu3"` are listed as variation traps rather than folded into this native audit.

Config source: Hugging Face raw files fetched through the HF connector/API. Compact snapshots are in this folder:

- `BAAI__Emu3-Chat-hf__config.snapshot.json`
- `BAAI__Emu3-Gen-hf__config.snapshot.json`
- `BAAI__Emu3-Chat__config.snapshot.json`
- `BAAI__Emu3-Gen__config.snapshot.json`
- `BAAI__Emu3-Stage1__config.snapshot.json`
- `BAAI__Emu3-VisionTokenizer__config.snapshot.json`
- `BAAI__Emu3_5__config.snapshot.json`
- `BAAI__Emu3_5-Image__config.snapshot.json`
- `hf_config_sweep_summary.json`

Source files inspected:

- `src/transformers/models/emu3/configuration_emu3.py`
- `src/transformers/models/emu3/modeling_emu3.py`
- `src/transformers/models/emu3/modular_emu3.py`
- `src/transformers/models/emu3/processing_emu3.py`
- `src/transformers/models/emu3/image_processing_emu3.py`
- `src/transformers/models/emu3/convert_emu3_weights_to_hf.py`

`modeling_emu3.py` is generated from `modular_emu3.py`; future source edits should inspect/patch the modular file, but this report audits the generated file because that is the import-time implementation at the pinned commit.

Any missing files or assumptions: native checkpoints are open. The older `BAAI/Emu3-*` and newer `BAAI/Emu3.5*` repos use remote-code causal-LM configs and should receive a separate audit if DinoML wants those exact repos. No gated links were encountered.

Primary runtime target for this report: native text decoder parity plus the image-conditioning path implemented by `Emu3Model`. Important source caveat: at this pinned commit, `Emu3ForConditionalGeneration.forward` accepts `pixel_values` and `image_sizes` but does not pass either into `self.model(...)`; `prepare_inputs_for_generation` passes `pixel_values` metadata to the generation helper but still no `image_sizes`. Treat top-level image-conditioned logits as requiring an upstream source fix/guard, or target `Emu3Model` stitch parity plus a separate LM head wrapper in DinoML.

## 2. High-level architecture

Emu3 is a tokenized-image multimodal autoregressive decoder. Images are converted by a frozen VQ-VAE into discrete visual code ids, visual code ids are mapped into text vocabulary ids, and a Llama-like causal decoder consumes a single mixed token stream.

Dataflow:

```text
CPU image preprocessing -> VQ-VAE image tokenizer -> visual-token BPE ids/embeds
text tokenization + placeholder expansion -> embedding stitch -> causal decoder prefill/decode -> logits
optional generated image tokens -> BPE-to-code mapping -> VQ-VAE decoder -> image postprocess
```

Stage decomposition:

- CPU/data pipeline: resize, RGB conversion, rescale, normalize, pad, tokenizer placeholder expansion.
- VQ-VAE encoder: cacheable image-token feature stage for image-conditioned text generation.
- Prefix construction: replace `<image>` placeholders with VQ-derived embedding rows.
- Decoder prefill/decode: Llama-style RMSNorm, GQA attention with RoPE, SwiGLU MLP, KV cache.
- Optional image generation decode: generated visual-token rows are mapped back to VQ ids and decoded by VQ-VAE.
- Native top-level caveat: `Emu3Model.forward` owns the VQ/stitch path; `Emu3ForConditionalGeneration.forward` at this commit drops image inputs before calling `Emu3Model`.

## 3. Important config dimensions

Native effective dimensions combine checkpoint config fields with defaults from `Emu3TextConfig` and `Emu3VQVAEConfig` when omitted.

| Field | `Emu3-Chat-hf` effective | `Emu3-Gen-hf` config/source | Source/default notes |
|---|---:|---:|---|
| text hidden size | 4096 | 4096 | Chat-hf omits it; default supplies 4096 |
| text layers | 32 | 32 | Chat-hf omitted |
| attention heads | 32 | 32 | head_dim inferred as 128 |
| KV heads | 8 | 8 | GQA, 4 query groups per KV head |
| intermediate size | 14336 | 14336 | gated MLP |
| vocab size | 184622 | 184622 | includes 32768 visual tokens |
| max positions | 131072 | 9216 | checkpoint config |
| RoPE theta | 1000000.0 | 1000000.0 | config/default standardized into `rope_parameters` |
| activation | silu | silu | SwiGLU |
| attention/MLP bias | false/false | false/false | source config defaults |
| RMSNorm eps | 1e-5 | 1e-5 | text decoder |
| VQ codebook size | 32768 | 32768 | visual token count |
| VQ embed/latent channels | 4 / 4 | 4 / 4 | codebook embedding and latent channels |
| VQ channels | 3 in, 3 out | 3 in, 3 out | image RGB |
| VQ base/ch multipliers | 256 / [1,2,2,4] | same | spatial factor 8 |
| VQ temporal factor | 4 | 4 | images are repeated across temporal dim for encode |
| dtype metadata | float32 | float32 | HF repo metadata/config, not a runtime mandate |

Representative checkpoint sweep:

| Repo | Scope | Architecture | Key variation |
|---|---|---|---|
| `BAAI/Emu3-Chat-hf` | native composite | `Emu3ForConditionalGeneration` | long 131072 context; text/vq dims mostly omitted and source-defaulted |
| `BAAI/Emu3-Gen-hf` | native composite | `Emu3ForConditionalGeneration` | 9216 context; explicit native text/vq fields |
| `BAAI/Emu3-Chat` | remote code | `Emu3ForCausalLM` | `model_type="Emu3"`, 131072 context, no native VQ composite |
| `BAAI/Emu3-Gen` | remote code | `Emu3ForCausalLM` | 9216 context, custom code |
| `BAAI/Emu3-Stage1` | remote code | `Emu3ForCausalLM` | 5120 context |
| `BAAI/Emu3-VisionTokenizer` | remote code/tokenizer | `Emu3VisionVQModel` | separate VQ model with different field names (`ch`, `ch_mult`, `z_channels`) |
| `BAAI/Emu3.5*` | remote code, separate follow-up | `Emu3ForCausalLM` | 5120 hidden, 64 layers, 64 heads, 8 KV heads, vocab 282926, bf16 metadata |

## 3a. Family variation traps

- Native `model_type="emu3"` is not the same config class as remote-code `model_type="Emu3"`; DinoML should reject or separately route the remote-code repos until audited.
- `BAAI/Emu3-Chat-hf` omits most subconfig fields; loaders must apply source defaults before deriving dimensions.
- The raw configs use legacy `rope_scaling={"rope_type":"default"}` while the pinned source reads `config.rope_parameters`; `PreTrainedConfig` standardizes this.
- `num_key_value_heads=8` with `num_attention_heads=32`, so attention is GQA, not MHA.
- Visual-token insertion is via boolean `masked_scatter`, but the processor creates a strict placeholder run per image. DinoML can lower to checked indexed/prefix row copies with guards.
- `Emu3ForConditionalGeneration.forward` currently drops `pixel_values`/`image_sizes`; do not validate image-conditioned logits through that exact top-level call without first confirming/fixing the source behavior.
- VQ-VAE modeling code is NCHW/NCTHW. NHWC is only an optimization candidate inside guarded local conv/norm regions.
- Image generation appends textual size markers (`height*width`) and visual-token rows; this is generation-controller ABI, not neural operator work.
- `prepare_inputs_for_generation` removes `pixel_values` after the first cached decode step.
- `Emu3.5*` configs contain `sliding_window` and `use_sliding_window=false`, but that is remote-code scope and not native `emu3` source behavior.

## 4. Operator coverage checklist

Tensor/layout ops:

- reshape/view, transpose/permute, contiguous, unsqueeze/squeeze, repeat, chunk, split, cat
- bool equality mask, masked select/count, `masked_scatter` replacement for image embeddings
- token id lookup/gather for vocab mappings

Neural primitives:

- Embedding and tied logical LM-head aliasing contract (`lm_head.weight` tied key to text embeddings, although config says `tie_word_embeddings=false`)
- Linear GEMMs: Q 4096->4096, K/V 4096->1024, O 4096->4096, MLP gate/up 4096->14336, down 14336->4096, LM head 4096->184622
- RMSNorm over hidden dim with fp32 variance, eps 1e-5
- SiLU and gated multiply for SwiGLU
- dropout is present but disabled for inference

Attention primitives:

- causal self-attention, GQA, RoPE before cache update
- KV cache update per layer; cache stores rotated K and raw V after projection/position encoding
- eager path repeats KV heads, matmul scores, additive mask, fp32 softmax, cast back, matmul V
- source can dispatch to SDPA/Flash/Flex through Transformers attention interfaces

VQ-VAE/tokenizer ops:

- Conv2d, Conv3d, BatchNorm3d, GroupNorm, nearest interpolate, constant padding
- swish-like `x * sigmoid(x)` activations
- noncausal spatial attention over flattened H*W tokens in VQ blocks
- vector quantization distances: `sum(z^2) + sum(e^2) - 2 z @ e.T`, argmin over 32768 codes
- codebook embedding gather for decode
- VQ code <-> BPE id mapping and EOL row insertion/removal

Preprocessing-coupled ops:

- `smart_resize` to multiples of spatial factor 8, min 512^2, max 1024^2 for fetched configs
- RGB conversion, rescale 1/255, normalize with mean/std 0.5 for fetched native repos
- pad bottom/right to max batch H/W

## 5. Layer/block breakdown

Text decoder block, repeated 32 times:

```text
residual = x
x = RMSNorm(x)
q = Linear(4096 -> 32 * 128)(x)
k = Linear(4096 -> 8 * 128)(x)
v = Linear(4096 -> 8 * 128)(x)
q,k = RoPE(q,k, position_ids)
k,v = cache.update(k,v) when cache is present
attn = causal GQA(q,k,v, mask), KV heads repeated 4x in eager fallback
x = residual + Linear(4096 -> 4096)(attn)
residual = x
x = RMSNorm(x)
x = Linear(14336 -> 4096)(SiLU(Linear(4096 -> 14336)(x)) * Linear(4096 -> 14336)(x))
x = residual + x
```

Text model:

```text
input_ids or inputs_embeds -> token embedding -> 32 decoder blocks -> final RMSNorm -> hidden states
hidden states[:, selected positions, :] -> LM head -> logits
```

VQ-VAE image encode:

```text
image [B,3,H,W] -> repeat temporal 4 -> [B,4,3,H,W]
flatten temporal into Conv2d path -> down blocks + middle attention -> Conv2d latent
reshape to [B,C,T,H/8,W/8] -> temporal Conv3d down stack -> quant Conv3d
vector quantize -> image code grid -> crop to original resized H/8,W/8
```

VQ-VAE image decode:

```text
image code grid -> codebook embedding -> post-quant Conv3d
temporal res/upsample -> Conv2d middle/up blocks with SpatialNorm conditioning
Conv2d output -> reshape to RGB image/video -> image postprocess
```

## 6. Attention requirements

Text attention is causal autoregressive self-attention with GQA. Native configs use 32 query heads, 8 key/value heads, head dim 128, so Q width is 4096 and K/V width is 1024. Attention masks are generated by `create_causal_mask` from the attention mask, cache, and position ids.

RoPE is applied to Q/K before `past_key_values.update`, so cached keys are already position-encoded. Eager fallback repeats K/V from `[B,8,S,128]` to `[B,32,S,128]`, computes scores in query dtype scaled by `head_dim**-0.5`, adds the causal mask, applies fp32 softmax, casts to query dtype, and multiplies values.

VQ-VAE attention is noncausal spatial self-attention over flattened image maps `[B,H*W,C]`, with 1 head and hidden size 1024 by default. It has no KV cache and is independently stageable from text decode.

Packed/varlen support is not source-owned beyond Transformers attention interface compatibility. There is no sliding-window/local attention in native `emu3`.

## 7. Position encoding and custom math

Native text uses standard RoPE with theta 1,000,000 and full head dim unless a future config supplies an explicit `head_dim`.

Short source-equivalent snippet:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = cat([freqs, freqs], dim=-1)
cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
```

Cos/sin depend on `position_ids`, dtype/device, and current RoPE parameters; inverse frequencies can be precomputed per config/device unless dynamic RoPE variants are admitted later.

Custom VQ mapping:

```text
img code i -> vocab id vocabulary_map["<|visual token %06d|>" % i]
append EOL id <|extra_200|> to every visual row
decode removes final EOL column before BPE-to-code mapping
```

## 8. Preprocessing and input packing

Image processor output contract:

- `pixel_values`: channel-first `[B,3,H_pad,W_pad]`, float after RGB/rescale/normalize.
- `image_sizes`: per-image resized unpadded `[height,width]`.
- resize uses `smart_resize(height,width,factor=8,min_pixels,max_pixels)`, then bottom/right padding to batch max.

Processor placeholder contract for image-conditioned text:

- Tokenizer placeholder is `<image>` with id 151646 in fetched native snapshots.
- For each input image, the processor computes `height = image_size[0] // 8`, `width = image_size[1] // 8`.
- Placeholder count is `height * (width + 1)`: one extra EOL token per visual row after image-code-to-BPE conversion.
- Text is rewritten as `<bos>...<|image start|>{height}*{width}<|image token|><image repeated N times><|extra_201|><|image end|>...`.
- The model verifies placeholder count equals feature count before `masked_scatter`.

Image generation packing:

- With `return_for_image_generation=True`, caller must not provide images.
- Processor computes token `height,width` from requested ratio and image area, then appends the image-start prompt.
- Generated image tokens are decoded by dropping the last three tokens, reshaping to `[B,height,width+1]`, removing EOL column, mapping BPE ids to VQ code ids, then VQ-VAE decoding.

CPU/data pipeline should own image decode, resize, normalization, tokenizer string construction, and most BPE mapping initially. GPU/runtime should own VQ-VAE and decoder once staged.

## 9. Graph rewrite / lowering opportunities

### Rewrite: placeholder masked scatter -> checked row copy

Source pattern:

```text
inputs_embeds.masked_scatter(input_ids == image_token_id expanded over hidden, image_features)
```

Replacement:

```text
find placeholder positions -> verify count/order -> copy image feature rows into embedding rows
```

Preconditions:

- `input_ids` is available or an equivalent exact placeholder-position side input is supplied.
- Placeholder count equals `sum_i (H_i/8) * (W_i/8 + 1)`.
- Flatten order matches processor/VQ output: image order, row-major visual grid with EOL column.

Failure cases: arbitrary boolean scatter, missing image sizes, mismatched placeholder count, or `inputs_embeds` path without recoverable placeholder positions.

Parity test: compare native `Emu3Model.forward` hidden input embeddings after stitch for one and two images with different padded sizes.

### Rewrite: Conv patch/tokenizer regions to channel-last kernels

Source pattern: VQ-VAE Conv2d/GroupNorm/interpolate blocks operate NCHW; temporal blocks operate NCTHW.

Replacement: guarded NHWC/NTHWC internal kernels with entry/exit layout transforms.

Preconditions:

- Entire local region from conv/norm/activation to next explicit `view(... H*W).transpose` is controlled.
- Axis-sensitive ops rewrite channel axis from `1` to `-1`; GroupNorm/BatchNorm channel semantics preserved.
- Padding order and nearest interpolation semantics match source.

Failure cases: attention flatten boundaries, temporal permute/reshape boundaries, or external layout consumers.

### Rewrite: VQ distance GEMM

Source pattern:

```text
dist = sum(z*z, dim=1, keepdim=True) + sum(E*E, dim=1) - 2 * z @ E.T
argmin(dist, dim=1)
```

Replacement: GEMM plus fused row norm/embedding norm add and argmin.

Preconditions: codebook is dense `[32768,4]`; flattened latent rows have width 4; output indices int64/int32 are acceptable for downstream mapping.

### Rewrite: GQA attention to native fused attention

Source pattern: Q/K/V separate projections, RoPE, cache update, attention backend.

Replacement: packed/fused QKV projection if weight packing is explicit, RoPE+cache+FlashAttention/GQA kernel.

Preconditions: split order is Q, K, V from separate weight tensors; no attention bias for native checkpoints; cached K stored after RoPE.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm fp32 variance + scale for decoder.
- GQA FlashAttention prefill/decode with RoPE and KV cache.
- SwiGLU MLP fusion: gate/up GEMM outputs, SiLU, multiply, down GEMM.
- Last-token-only logits using `logits_to_keep` to avoid full-vocab GEMM for all positions.
- Embedding stitch row-copy kernel replacing general `masked_scatter`.

Medium priority:

- VQ-VAE Conv2d/GroupNorm/swish residual block fusion.
- VQ vector-quantization GEMM + argmin specialized to embed dim 4 and codebook 32768.
- VQ spatial attention over `[B,H*W,1024]`.
- Image-code BPE mapping gather plus EOL insertion/removal.

Lower priority:

- Temporal Conv3d/BatchNorm3d/swish fusion for video/general VQ path.
- Processor resize/normalize on GPU; CPU pipeline is acceptable initially.
- Remote-code Emu3.5 sliding-window fields, because native source does not implement them.

## 11. Runtime staging plan

Stage 1: parse native configs, apply defaults, load text-only weights, run `Emu3ForCausalLM` one-block parity.

Stage 2: implement full text prefill logits with no cache, causal mask, RoPE, RMSNorm, GQA, and SwiGLU.

Stage 3: implement decode with DynamicCache-compatible per-layer K/V tensors and `logits_to_keep`.

Stage 4: add processor-compatible image placeholder expansion as metadata plus checked embedding row-copy against `Emu3Model`. Stub VQ-VAE by accepting precomputed visual BPE ids or image embeddings.

Stage 5: implement VQ-VAE encode for image-conditioned text generation; validate image token ids and image-feature embedding stitch.

Stage 6: add a DinoML top-level logits wrapper, or wait for/confirm upstream source behavior, so image-conditioned hidden states feed the LM head with `pixel_values` and `image_sizes` preserved.

Stage 7: implement optional generated-image decode path: generated BPE visual tokens -> VQ ids -> VQ-VAE decoder -> postprocess.

Stage 8: optimize fused attention, MLP, VQ conv blocks, and vector quantization.

## 12. Parity and validation plan

- Config/default tests: `Emu3-Chat-hf` omitted text/vq fields resolve to source defaults; `rope_scaling` resolves to `rope_parameters`.
- Custom math tests: RoPE cos/sin and `apply_rotary_pos_emb`, RMSNorm fp32 variance, VQ image/BPE mapping, VQ distance argmin.
- Single block parity: random hidden states, masks, position ids, no cache then with cache.
- Full text parity: prompt prefill logits and single-token decode logits against Transformers fp32/bf16.
- Embedding stitch parity: one image and two images with different sizes; compare stitched `inputs_embeds`.
- VQ encode parity: small resized image tensor through VQ-VAE returns identical code ids.
- End-to-end image-to-text parity: processor + VQ + decoder logits for `BAAI/Emu3-Gen-hf` and `BAAI/Emu3-Chat-hf`, but only after the top-level image-input forwarding issue is handled; before that, validate `Emu3Model` stitched hidden states and a separately applied LM head.
- Optional image generation parity: generated visual-token fixture decodes to same pixel tensor before PIL conversion.

Suggested tolerances: fp32 custom ops `1e-5` absolute/relative for logits before softmax; fp16/bf16 decoder logits `1e-2` to `3e-2` depending on attention backend; token ids and VQ argmin must be exact.

## 13. Performance probes

- Processor throughput: image resize/normalize/pad and tokenizer placeholder expansion separately.
- VQ encoder throughput vs image area and batch size.
- Text prefill throughput sweep: sequence lengths 1k, 9k, 32k, 131k where memory permits.
- Decode tokens/sec with cache for batch sizes 1, 4, 8, and mixed image-prefix lengths.
- KV cache memory: 32 layers * 2 tensors * 8 KV heads * head_dim 128 * sequence length.
- LM head cost with full logits vs `logits_to_keep=1`.
- Embedding stitch cost: native masked scatter vs checked row copy.
- VQ quantizer: GEMM+argmin time for latent row counts from common image sizes.
- Optional VQ decoder throughput for image-generation token grids.

## 14. Skip/defer list

- Training, losses, dropout behavior, gradient checkpointing.
- Remote-code `BAAI/Emu3-*` and `BAAI/Emu3.5*` causal-LM implementations until separately audited.
- Video input/output path, even though VQ-VAE has temporal convs; first native target can be image.
- General boolean `masked_scatter`; admit only the processor-guaranteed image-placeholder row-copy pattern.
- GPU resize/tokenizer string logic; keep in CPU/data pipeline initially.
- Beam search, sampling policies, and top-k controller details beyond accepting generation metadata.
- Multi-GPU tensor parallel plans in `_tp_plan` / `_pp_plan`.

## 15. Final implementation checklist

- [ ] Parse native `Emu3Config`, `Emu3TextConfig`, and `Emu3VQVAEConfig`, including omitted-field defaults.
- [ ] Reject or route remote-code `model_type="Emu3"` and `Emu3VisionVQ` repos separately.
- [ ] Load text decoder weights and preserve logical LM-head/input-embedding alias metadata.
- [ ] Implement RMSNorm, RoPE, causal mask, GQA attention, KV cache, and SwiGLU decoder block.
- [ ] Add `logits_to_keep` lowering for last-token-only logits.
- [ ] Implement processor ABI metadata: image sizes, placeholder counts, visual token ids, EOL rows.
- [ ] Replace image `masked_scatter` with guarded embedding row copy.
- [ ] Decide admission for the pinned top-level image-input forwarding caveat: reject, patch upstream behavior in a wrapper, or validate through `Emu3Model` plus LM head.
- [ ] Add VQ image-code/BPE mapping kernels or CPU helpers.
- [ ] Implement VQ-VAE encoder for image-conditioned text parity.
- [ ] Add optional VQ-VAE decoder for image-token generation parity.
- [ ] Add parity tests for config defaults, one decoder block, full text logits, cached decode, embedding stitch, VQ encode, and end-to-end image-to-text.
- [ ] Benchmark processor, VQ encoder, prefill, decode, LM head, cache memory, and embedding stitch.
