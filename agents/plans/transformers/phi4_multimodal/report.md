# Phi4 Multimodal DinoML Audit

## 1. Source basis

Transformers commit/version: local checkout `X:/H/transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: native source targets `microsoft/Phi-4-multimodal-instruct`; public checkpoint configs are mostly historical remote-code schema (`phi4mm`), not a clean native `phi4_multimodal` config.

Config source: local `configuration_phi4_multimodal.py`; HF configs from `microsoft/Phi-4-multimodal-instruct`, `tiny-random/phi-4-multimodal`, `yujiepan/phi-4-multimodal-tiny-random`, `junnei/Phi-4-multimodal-instruct-ko-asr`, and `huihui-ai/Phi-4-multimodal-instruct-abliterated`.

Source files inspected: `configuration_phi4_multimodal.py`, `modeling_phi4_multimodal.py`, `modular_phi4_multimodal.py`, `processing_phi4_multimodal.py`, `image_processing_phi4_multimodal.py`, `feature_extraction_phi4_multimodal.py`. `modeling_phi4_multimodal.py` and `configuration_phi4_multimodal.py` are generated from `modular_phi4_multimodal.py`; future source edits should start from the modular file.

Any missing files or assumptions: no imports/tests were run. HF official JSON was accessible, but the ONNX config raw URL returned an LFS pointer. Current native source does not directly consume many legacy checkpoint fields such as `embd_layer`, `audio_processor`, `attention_bias`, `mlp_bias`, `lm_head_bias`, `full_attn_mod`, LoRA blocks, or `auto_map` remote-code names.

## 2. High-level architecture

Primary runtime target: multimodal causal language generation with optional image and audio prefix embeddings.

Dataflow:

```text
text/images/audio preprocessing
  -> token ids with expanded modality placeholders
  -> image ViT encoder + image projector, audio fbank + Conformer encoder + audio projector
  -> index-copy modality embeddings into token embedding rows
  -> Phi-style causal decoder prefill/decode with RoPE/GQA/KV cache
  -> logits/sampling
```

Stage decomposition:

| Stage | Owner | Cacheability / validation |
| --- | --- | --- |
| Image resize/crop/normalize/HD tiling | CPU/data pipeline first | Validate independently; emits NCHW crops and patch masks. |
| Audio waveform to log-mel fbank | CPU/data pipeline first, optional Torch device path | Validate independently; emits `[B,T,80]` and embed sizes. |
| Image encoder/projector | GPU runtime | Cache per image prompt; output is token-row embeddings for image placeholders. |
| Audio encoder/projector | GPU runtime | Cache per audio prompt; output is token-row embeddings for audio placeholders. |
| Embedding stitch | GPU runtime or guarded CPU-side prepack | Requires strict placeholder count/order guards. |
| Text decoder prefill | GPU runtime | Consumes stitched sequence and creates KV cache. |
| Decode | GPU runtime | Text-only step after first prefill unless caller injects new modalities. |

## 3. Important config dimensions

Native source defaults:

| Field | Text | Vision | Audio |
| --- | ---: | ---: | ---: |
| hidden size | 3072 | 1152 | 1024 |
| layers | 32 | 27 | 24 |
| attention heads | 32 | 16 | 16 |
| KV heads | 8 | n/a | n/a |
| head dim | 96 inferred | 72 inferred | 64 inferred |
| MLP/intermediate | 8192 | 4304 | 1536 |
| vocab size | 200064 | n/a | n/a |
| max positions | 131072 | patch grid 32x32 at 448/14 | relative bias max distance 1000 |
| activation | SiLU gated MLP | GELU tanh approx | swish gated MLP/convs |
| cache | DynamicCache self-attn KV | none | none |
| image/audio tokens | image 200010 | patch 14, image 448, crop 448 | audio 200011, input size 80 |

Representative checkpoint sweep:

| Checkpoint | Scope | Text dims | Modality dims | Operator-significant notes |
| --- | --- | --- | --- | --- |
| `microsoft/Phi-4-multimodal-instruct` | Official remote-code config | 3072, 32 layers, 24 Q heads, 8 KV heads, head dim inferred 128, rotary dim 96 from 0.75 partial factor | Audio config 1024/24/16, image settings embedded in legacy `embd_layer` | `model_type="phi4mm"`, longrope, sliding window 262144, bf16, `tie_word_embeddings=true`; needs schema mapper before native source parity. |
| `tiny-random/phi-4-multimodal` | Debug | 16, 2 layers, 2 Q heads, 1 KV head | Tiny audio dims; no native `vision_config` object | Useful for parser smoke only; remote-code names. |
| `yujiepan/phi-4-multimodal-tiny-random` | Debug mirror | same as tiny-random | same | Same structure. |
| `junnei/Phi-4-multimodal-instruct-ko-asr` | Finetune | official full dims | official audio dims | `torch_dtype=float32`; no structural change in config. |
| `huihui-ai/Phi-4-multimodal-instruct-abliterated` | Finetune | official full dims | official dims | Same structural config as official; bf16. |

## 3a. Family variation traps

- Native source defaults use 32 attention heads, but official public configs use 24 heads plus `partial_rotary_factor=0.75`. Do not infer head count or rotary dimension from native defaults when loading legacy weights.
- Native source expects `rope_parameters`; legacy configs use `rope_scaling` plus top-level `rope_theta` and `partial_rotary_factor`.
- Public configs advertise `sliding_window=262144`; native source passes this to attention backends when not `None`, but the default native config is `None`.
- Official processor names and model class names are `Phi4MM*`; native classes are `Phi4Multimodal*`.
- Image and audio placeholders are expanded dynamically by the processor. The model uses `index_put`, but the processor can provide stricter count/order guarantees.
- Image branch is axis-sensitive: processor and model use NCHW crops, NCHW Conv2d, NHWC patch reshapes, AvgPool2d over NCHW, then NHWC/token flattening. Layout translation needs guarded local regions.
- Audio branch is axis-sensitive: `[B,T,F]` fbank becomes `[B,1,T,F]` Conv2d, then `[B,T,C*F]`; Conformer conv modules permute `[B,T,C] <-> [B,C,T]`.
- Long audio has a chunking path at post-subsample `seq_len > 500` using `F.pad` and `F.unfold`, then reassembles chunks. This is a runtime graph fork, not preprocessing-only.
- Legacy configs contain LoRA metadata and remote-code flags not read by native source. Treat them as ignored until a remote-code audit says otherwise.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup, dynamic `arange`, `view`/`reshape`, `flatten`, `transpose`, `permute`, `contiguous`, `cat`, `chunk`, `repeat`, `expand`, `squeeze`/`unsqueeze`.
- `nonzero` on placeholder equality and `index_put(accumulate=False)` for embedding stitch.
- Boolean masks, bitwise not, comparisons, masked fill, bucketize for image position IDs.
- Dynamic slicing and per-example loops for variable image/audio token counts.

Neural primitives:

- Dense Linear/GEMM including packed QKV and packed gate/up MLP.
- RMSNorm for decoder, LayerNorm for vision/audio.
- Conv2d patch embedding: NCHW `[B,3,H,W] -> [B,1152,H/14,W/14]`, kernel=stride=14.
- Image AvgPool2d kernel=2 stride=2 over NCHW patch feature maps.
- Audio Conv2d subsampling: first conv `1 -> 1024`, kernel 3, stride 2, padding 1; repeated depthwise Conv2d plus pointwise Conv2d for time reduction 8.
- Audio Conv1d pointwise GLU, depthwise Conv1d with causal crop, pointwise Conv1d.
- GELU, GELU tanh approximation, SiLU/swish, dropout as identity in inference.

Attention primitives:

- Decoder causal GQA attention with packed QKV projection, RoPE before cache update, optional sliding-window backend.
- Vision bidirectional self-attention over patch tokens.
- Vision pooling cross-attention: learned single query attends over patch sequence using `nn.MultiheadAttention`.
- Audio self-attention with relative attention bias and streaming/pad mask.

Position/custom math:

- Decoder RoPE/longrope with partial rotary dimension.
- Vision bucketized learned absolute patch IDs based on valid patch extents.
- Audio learned relative attention bias with clipped relative positions.

Generation/cache ops:

- Transformers `DynamicCache` per decoder layer with K/V stored before GQA repeat and after RoPE.
- `prepare_inputs_for_generation` resets cache when crossing `original_max_position_embeddings + 1` for longrope.
- `logits_to_keep` last-token or indexed logits projection.

Preprocessing-coupled ops:

- Image bicubic resize, pad, RGB conversion, rescale/normalize, dynamic HD tiling.
- Audio mono conversion, padding/truncation, Hamming-window STFT/rFFT, magnitude power, mel matmul, clamp min 1, log.

## 5. Layer/block breakdown

Image encoder:

```text
pixel_values [B_or_BC,3,448,448]
  -> Conv2d(3 -> 1152, k=14, s=14)
  -> flatten patches [B,1024,1152]
  -> add bucketized learned position embeddings
  -> repeat 27:
       LayerNorm
       Q/K/V Linear(1152 -> 1152), MHA 16x72, output Linear
       residual
       LayerNorm
       Linear(1152 -> 4304) -> GELU tanh -> Linear(4304 -> 1152)
       residual
  -> post LayerNorm
  -> optional pooling head with learned probe cross-attention
```

Image projector/stitch path:

```text
selected hidden layer (-2)
  -> reshape 32x32 patch grid
  -> permute NHWC -> NCHW
  -> optional ReflectionPad2d if odd patch grid
  -> AvgPool2d(2,2) to 16x16
  -> global/subimage HD reshape with learned separator rows
  -> Linear(1152 -> 3072) -> GELU -> Linear(3072 -> 3072)
  -> index_put into image placeholder rows
```

Audio encoder:

```text
audio_input_features [B,T,80]
  -> global mean/variance affine
  -> Conv2d subsampling over [B,1,T,80], time reduction 8
  -> Linear(1024*nemo_final_size -> 1024)
  -> relative attention bias [1,16,T',T']
  -> repeat 24 Conformer blocks:
       0.5 * LayerNorm -> Linear(1024 -> 3072) -> chunk -> swish gate multiply -> Linear(1536 -> 1024)
       LayerNorm -> MHA 16x64 with relative/streaming mask -> residual
       LayerNorm -> Conv1d GLU -> depthwise Conv1d(k=3,pad=2) -> causal crop -> swish -> Conv1d pointwise
       0.5 * feed-forward out
       final LayerNorm
  -> long-audio unfold/reassemble if T' > 500
  -> Linear(1024 -> 3072) -> GELU -> Linear(3072 -> 3072)
  -> index_put into audio placeholder rows
```

Decoder block, repeated 32:

```text
x = RMSNorm(x)
qkv = Linear(3072 -> q_heads*head_dim + 2*kv_heads*head_dim, bias=False)
q,k,v = split [Q, K, V]
q,k = partial RoPE(q,k)
k,v = cache.update(k,v, layer)
attn = causal/sliding-window GQA attention(q,k,v)
x = residual + Linear(q_heads*head_dim -> 3072, bias=False)(attn)
x = RMSNorm(x)
gate, up = Linear(3072 -> 2*8192, bias=False)(x).chunk(2)
x = residual + Linear(8192 -> 3072, bias=False)(up * silu(gate))
```

## 6. Attention requirements

Decoder attention is causal self-attention with GQA. Native defaults are Q heads 32, KV heads 8, head dim 96; official public configs are Q heads 24, KV heads 8, head dim 128, rotary dim 96 through partial factor 0.75. K/V cache shape per layer is `[B, num_key_value_heads, cached_seq, head_dim]`; GQA repeat to query head count happens inside eager attention or backend equivalent. RoPE is applied before cache update, so cached K is already position encoded.

Vision attention is bidirectional MHA over flattened patch tokens. The source sets `is_causal=True` in the module but builds a bidirectional mask and calls the generic attention interface; DinoML should treat this branch as noncausal encoder attention and test backend mask behavior explicitly.

Vision pooling attention is rectangular cross-attention from one learned query `[B,1,1152]` to patch keys/values `[B,N,1152]`, with key padding mask from the patch attention mask. This is not generation cache attention.

Audio attention is bidirectional/streaming Conformer attention over subsampled frames with a boolean streaming/pad mask plus learned relative bias. Source builds `attention_mask = hs_mask.unsqueeze(1) + relative_attention_bias`; because this mixes bool and float tensors in PyTorch, DinoML should canonicalize to additive float bias with explicit mask convention during parity work.

FlashAttention/SDPA compatibility: all three branches use `ALL_ATTENTION_FUNCTIONS`; decoder also passes `sliding_window`. First integration can use explicit matmul-softmax-matmul references, then admit FlashAttention only after mask, GQA, partial RoPE, relative bias, and sliding-window semantics are covered.

## 7. Position encoding and custom math

Decoder RoPE:

```python
rotary_dim = int(head_dim * partial_rotary_factor)
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
cos = cat(freqs, freqs).cos() * attention_scaling
sin = cat(freqs, freqs).sin() * attention_scaling
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
q = cat(q_rot * cos + rotate_half(q_rot) * sin, q_pass)
k = cat(k_rot * cos + rotate_half(k_rot) * sin, k_pass)
```

Longrope factors can be precomputed for a fixed config, but the generation controller must handle cache invalidation when crossing `original_max_position_embeddings`.

Vision patch position IDs are runtime-derived from valid patch counts:

```text
step_h = 1 / valid_patch_rows
step_w = 1 / valid_patch_cols
bucket_h = bucketize(arange(max_rows) * step_h, learned_grid_boundaries)
bucket_w = bucketize(arange(max_cols) * step_w, learned_grid_boundaries)
pos_id = bucket_h * num_patches_per_side + bucket_w
```

Audio relative bias:

```text
relative = memory_position - context_position
relative = clamp(relative, -max_distance, max_distance - 1)
bias_idx = abs(relative) if symmetric else relative + num_buckets // 2
bias = Embedding(num_buckets, num_heads)(bias_idx).permute(2,0,1)[None]
```

## 8. Preprocessing and input packing

Image ABI:

- Processor emits `image_pixel_values` with shape `[B, CROP_COUNT, 3, 448, 448]`, `image_sizes` `[B,2]`, and `image_attention_mask` `[B,CROP_COUNT,32,32]`.
- Dynamic HD chooses a tiled aspect ratio capped by `dynamic_hd=36`. It pads/resizes to multiples of 448, creates one global crop plus sub-crops, and pads batch members to common crop count.
- `num_img_tokens = 256 + 1 + valid_downsampled_tokens + valid_downsampled_rows + 16`; processor expands each image placeholder into that many image tokens.
- Source layout is NCHW through preprocessing and patch Conv2d. The projector temporarily moves patch features through NHWC-like layouts for HD stitching.

Audio ABI:

- Raw audio must be 16 kHz; stereo/multichannel inputs are averaged to mono by the feature extractor.
- Feature extraction uses 400-sample Hamming windows, hop 160, FFT 512, preemphasis 0.97, mel bins 80, Kaldi mel scale, max frequency 7690.
- Output `audio_input_features` has shape `[B,T_frames,80]`. `audio_attention_mask` is `[B,T_frames]` only for multi-sample batches when requested. `audio_embed_sizes = ceil(ceil(T_frames / audio_compression_rate) / audio_downsample_rate)`.
- Processor expands each audio placeholder into `audio_embed_sizes[i]` audio tokens.

Embedding stitch ABI:

- Image token id is 200010 in native config; official tokenizer maps this id to `<|endoftext10|>`.
- Audio token id is 200011; official tokenizer maps this id to `<|endoftext11|>`.
- Processor validates count of image/audio placeholders equals number of images/audios, then replaces each placeholder with repeated identical special-token text. The source model uses `torch.nonzero(input_ids == token_id)` and `index_put(accumulate=False)`.
- DinoML can lower this to ordered indexed row copy if it guards that placeholder count equals projected feature row count and preserves row-major order from `nonzero` over `[batch, seq]`. It should reject arbitrary boolean scatter in first integration.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap image patch Conv2d -> Linear

Source pattern: `Conv2d(3,1152,kernel=14,stride=14,padding=valid)` followed by flatten patches.

Replacement: `WindowFlatten[N, H/14*W/14, 3*14*14] -> GEMM(weight_flat.T) -> bias -> token sequence`.

Preconditions: static crop size multiple of 14, dilation 1, groups 1, NCHW source layout or explicitly transformed NHWC equivalent, no padding, exact PyTorch flatten order.

Failure cases: non-square or dynamic patch sizes, layout pass that changes channel/spatial order without weight transform, future configs with different patch sizes.

Parity test sketch: compare patch embeddings on random NCHW crops for fp32 and bf16 tolerance, including batch/crop flattening.

### Rewrite: packed decoder QKV split

Source pattern: one Linear from hidden to `[Q, K, V]` concatenated in all-Q/all-K/all-V order, with sizes `q_heads*head_dim`, `kv_heads*head_dim`, `kv_heads*head_dim`.

Replacement: single provider GEMM producing packed output, then metadata split views into Q/K/V.

Preconditions: native packed weight layout preserved, no projection bias, split sizes from config not inferred from hidden size alone.

Failure cases: legacy remote-code configs where head dim is not native default; tensor-parallel sharding must preserve split boundaries.

### Rewrite: modality `index_put` -> ordered row copy

Source pattern: `positions = nonzero(input_ids == token_id); inputs_embeds.index_put(positions, features)`.

Replacement: validate placeholder positions and feature row count, then copy projected feature rows into flattened embedding matrix at computed row offsets.

Preconditions: `accumulate=False`, processor-expanded placeholders, row-major `nonzero` ordering, feature rows concatenated in the same image/audio order as text placeholders.

Failure cases: caller bypasses processor, multiple modalities overlap, feature row count mismatch, arbitrary token positions without count validation.

### Rewrite: audio Conv2d subsampling to fixed conv/GEMM kernels

Source pattern: `[B,T,80] -> unsqueeze -> Conv2d stride-2 stack -> flatten frequency/channel -> Linear`.

Replacement: either native Conv2d provider path or im2col/GEMM for fixed `kernel=3,stride=2,pad=1`, plus depthwise/pointwise conv specialization.

Preconditions: source `[B,1,T,F]` layout, time_reduction power-of-two and multiple of 2, `input_size=80`, static frequency dimension.

Failure cases: variable audio config, short audio after reduction, long-audio unfold branch.

### Layout guard: image HD stitch region

The region after selected image hidden states reshapes tokens to `[crop,base,base,C]`, transposes/reshapes sub-images, concatenates learned extensor rows, and flattens to token rows. Treat this as a no-layout-translation region until an explicit axis-rewrite proof exists.

## 10. Kernel fusion candidates

Highest priority:

- Decoder RMSNorm, packed QKV GEMM, partial RoPE, GQA attention with KV cache, and packed SwiGLU MLP. These dominate prefill/decode.
- Ordered modality embedding row-copy kernel. This avoids admitting general scatter while keeping multimodal prefill practical.
- Audio fbank boundary decision. Keeping STFT/mel in CPU pipeline first reduces GPU op surface; a later GPU path needs FFT/mel kernels.

Medium priority:

- Image patch Conv2d-to-GEMM and image encoder MHA/MLP fusions.
- Audio Conformer block fusions: LayerNorm + gated FFN, Conv1d GLU, depthwise causal Conv1d crop, relative-bias attention.
- Last-token-only logits using `logits_to_keep=1`.

Lower priority:

- Vision pooling learned-query attention optimization.
- Long-audio unfold/reassemble kernel; first integration can reject or route long audio above post-subsample length 500.
- Sliding-window decoder attention because official window exceeds max position and may behave effectively dense for many prompts.

## 11. Runtime staging plan

Stage 1: implement a config compatibility mapper for native `phi4_multimodal`; explicitly reject legacy `phi4mm` remote-code checkpoints unless mapped.

Stage 2: text-only decoder parity with packed QKV, partial RoPE, DynamicCache, last-token logits, and cache reset across longrope boundary.

Stage 3: processor ABI validation without model execution: image/audio placeholder expansion, token ids, feature size equations, and row-copy guards.

Stage 4: image encoder/projector parity for one image and bounded crop counts; stitch projected embeddings into a decoder prefill.

Stage 5: audio feature ABI and audio encoder/projector parity for short audio where post-subsample length <= 500.

Stage 6: full multimodal prefill plus text decode; cache image/audio projected embeddings as prefix inputs.

Stage 7: optimize attention, GEMM epilogues, layout rewrites, and optional GPU preprocessing.

Stub initially: training losses, dropout, LoRA metadata, remote-code-only flags, ONNX artifact configs, long audio unfold branch, arbitrary scatter, and unbounded dynamic-HD crop counts.

## 12. Parity and validation plan

- Config parser tests for native defaults and legacy HF JSONs, checking head dim, rotary dim, token ids, and rejected ignored fields.
- Unit parity for RMSNorm, partial RoPE, packed QKV split, repeat-KV, and longrope cache reset.
- Single decoder layer parity with random hidden states, masks, and cache; then 2-layer and full prefill logits.
- Image processor ABI tests for dynamic-HD shapes and `num_img_tokens`; image patch embedding and image projector parity on one crop and multi-crop cases.
- Audio feature extraction parity for 1-second and variable-length batch audio; audio embed size equation tests.
- Audio encoder parity for short audio; separate test for branch rejection or parity at `seq_len > 500`.
- Embedding stitch parity comparing ordered row copy to PyTorch `index_put`.
- End-to-end text-only, image+text, audio+text, and image+audio+text generation smoke with fixed greedy decode.
- Suggested tolerances: fp32 atol/rtol 1e-4 for isolated ops, bf16/fp16 5e-2 for full branches, stricter logits checks before enabling fused attention.

## 13. Performance probes

- Processor throughput split: image preprocessing images/sec, audio fbank seconds-audio/sec.
- Image encoder/projector latency versus crop count and dynamic-HD area.
- Audio encoder/projector latency versus waveform seconds and post-subsample length; separate `<=500` and `>500` branches.
- Text decoder prefill tokens/sec for text-only and multimodal-prefixed sequences.
- Decode tokens/sec with KV cache and `logits_to_keep=1`.
- KV cache memory by batch, prompt length, and native versus official head dimensions.
- Attention backend comparison: eager, SDPA/FlashAttention-compatible dense, sliding-window.
- GEMM provider probe for packed QKV and gate_up projections.
- Embedding stitch row-copy bandwidth versus PyTorch scatter/reference.
- Quantized weight load/dequant probe if mapping official bf16 weights to GGUF/offload experiments later.

## 14. Skip/defer list

- Training, labels/loss, dropout behavior, gradient checkpointing.
- Remote-code `Phi4MM*` execution unless a separate compatibility audit is done.
- Legacy LoRA metadata application.
- ONNX deployment artifact parity.
- GPU STFT/mel feature extraction; keep audio preprocessing in data pipeline first.
- Long-audio unfold branch for first audio integration, unless ASR is the first product target.
- Arbitrary `index_put`/masked scatter; use guarded ordered row copy.
- General NHWC/channel-last translation across image/audio branches.
- Beam search and sampling policy beyond greedy/logits parity.
- Multi-GPU tensor parallel plans.

## 15. Final implementation checklist

- [ ] Add native `phi4_multimodal` config parser and legacy `phi4mm` rejection or mapper.
- [ ] Load text decoder weights with packed QKV and gate/up layouts.
- [ ] Implement decoder RMSNorm, partial RoPE/longrope, GQA attention, DynamicCache, and cache reset guard.
- [ ] Implement `logits_to_keep` last-token logits path.
- [ ] Implement processor ABI validators for image/audio placeholder expansion.
- [ ] Implement ordered modality row-copy replacement for `index_put`.
- [ ] Implement image patch Conv2d or guarded Conv2d-to-GEMM rewrite.
- [ ] Implement image encoder LayerNorm/MHA/MLP and HD projector stitch region with layout guards.
- [ ] Implement audio fbank CPU/data-pipeline contract.
- [ ] Implement audio Conv2d subsampling, Conformer attention with relative bias, Conv1d GLU/depthwise modules, and audio projector.
- [ ] Add parity tests for text-only prefill/decode, image prefix, audio prefix, and combined multimodal prefill.
- [ ] Benchmark processor, image encoder, audio encoder, prefill, decode, and stitch kernels separately.

## Gated gaps for DinoML

- Legacy public checkpoints are not directly native-source compatible: `model_type="phi4mm"`, remote-code class names, 24 heads, and `rope_scaling`/`partial_rotary_factor` need an explicit mapper or rejection path.
- General boolean scatter/index_put should remain gated behind processor-derived ordered row-copy guards.
- Audio branch requires Conv2d, depthwise/pointwise Conv1d, relative attention bias, streaming masks, and optional long-audio `F.unfold`; this is beyond current core decoder coverage.
- Image branch requires Conv2d patch embed, bucketize-derived patch positions, bidirectional vision attention, AvgPool2d, HD tile stitching, and strict NCHW/NHWC layout guards.
- Decoder needs partial-rotary longrope, GQA cache, sliding-window admission, and cache invalidation when crossing `original_max_position_embeddings`.
