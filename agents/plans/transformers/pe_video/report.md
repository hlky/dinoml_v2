# Transformers Family Audit: `pe_video`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/pe-av-large, facebook/pe-av-large-16-frame
Config source: Hub config.json snapshots plus source defaults
Source files inspected:
  transformers/src/transformers/models/pe_video/configuration_pe_video.py
  transformers/src/transformers/models/pe_video/modeling_pe_video.py
  transformers/src/transformers/models/pe_video/modular_pe_video.py
  transformers/src/transformers/models/pe_video/video_processing_pe_video.py
  transformers/src/transformers/models/pe_video/processing_pe_video.py
  transformers/src/transformers/models/pe_audio_video/modeling_pe_audio_video.py
  transformers/src/transformers/models/timm_wrapper/{configuration,modeling}_timm_wrapper.py
  transformers/src/transformers/models/modernbert/{configuration,modeling}_modernbert.py
Any missing files or assumptions:
  modeling_pe_video.py is generated from modular_pe_video.py. Use modular_pe_video.py for future source edits, but use generated modeling_pe_video.py for exact runtime behavior.
  The Hub checkpoints are pe_audio_video checkpoints, not pure pe_video checkpoints, but their nested video_config is the representative in-library pe_video branch.
  The vision body is delegated to timm architecture vit_pe_core_large_patch14_336; this report treats that body as external/allowlisted, not as source-owned Transformers ops.
```

Hub snapshots stored beside this report:

- `facebook_pe-av-large_config.json`
- `facebook_pe-av-large-16-frame_config.json`
- `facebook_pe-av-large_processor_config.json`
- `facebook_pe-av-large-16-frame_processor_config.json`
- `facebook_pe-av-large_summary.json`
- `facebook_pe-av-large-16-frame_summary.json`

No gated Hub links were encountered. Model pages: [facebook/pe-av-large](https://huggingface.co/facebook/pe-av-large), [facebook/pe-av-large-16-frame](https://huggingface.co/facebook/pe-av-large-16-frame).

## 2. High-level architecture

Primary DinoML target: text-video contrastive embedding and similarity for `PeVideoModel`, with independently stageable `PeVideoEncoder` video embeddings. This is not autoregressive generation and has no decode KV cache.

```text
video decode/frame sampling + resize/normalize -> per-frame timm image classifier logits
  -> frame logits L2 normalize -> Linear(1024 -> 1792) -> Linear(1792 -> 1792)
  -> prepend learned class token -> temporal Conv1d ResNet block
  -> bidirectional RoPE encoder -> final RMSNorm/Linear -> video embedding

text tokens -> ModernBERT encoder -> CLS hidden state -> contrastive head
video/text embeddings -> matrix multiply -> scalar scale/bias -> logits_video_text
```

Stage decomposition:

- CPU/data pipeline: video decode, optional frame sampling, RGB conversion, resize to 336x336, rescale/normalize, padding variable frame counts, tokenization.
- Delegated image stage: `AutoModelForImageClassification` through `TimmWrapperForImageClassification`, one image model call per collapsed `[B*T,C,H,W]` frame batch.
- PE video temporal stage: small sequence model over per-frame logits plus class token.
- Text stage: `AutoModel` ModernBERT encoder. This should compose a separate ModernBERT audit for full operator coverage.
- Similarity head: LayerNorm+Linear projections, `video_embeds @ text_embeds.T`, scale/bias.

## 3. Important config dimensions

Source default `PeVideoEncoderConfig`:

| Field | Value |
|---|---:|
| `hidden_size` | 1792 |
| `intermediate_size` | 4800 |
| `num_hidden_layers` | 6 |
| `num_attention_heads` | 14 |
| `num_key_value_heads` | defaults to 14 |
| `head_dim` | 128 |
| attention Q width | 1792 |
| attention K/V width | 1792 when KV heads = 14 |
| activation | `silu` |
| `max_position_embeddings` | 10000 |
| RoPE | default, `rope_theta=20000` |
| attention bias/dropout | `False`, `0.0` |
| vision config default | `timm_wrapper`, `vit_pe_core_large_patch14_336`, `num_classes=1024`, `global_pool=map` |

Representative checkpoint sweep:

| Config source | Top model | Video layers | Video H | Heads/KV/head_dim | Vision body | Text body | Processor frames |
|---|---|---:|---:|---|---|---|---|
| source default `PeVideoConfig` | `pe_video` | 6 | 1792 | 14/14/128 | timm `vit_pe_core_large_patch14_336`, 1024 classes | ModernBERT 22L, H=1024, vocab=50368 | unspecified |
| `facebook/pe-av-large` config | `pe_audio_video` nested video branch | 4 | 1792 | 14/14/128 | same | ModernBERT 22L, H=1024, local attention 128 | all/variable frames, padded |
| `facebook/pe-av-large-16-frame` config | `pe_audio_video` nested video branch | 4 | 1792 | 14/14/128 | same | same | fixed 16 frames |

Checkpoint facts above come from `config.json` and processor config. The parameter size/license/downloads are Hub metadata, not source behavior; the safetensors file is about 8.94 GB for both official checkpoints.

## 3a. Family variation traps

- Official Hub checkpoints expose `PeAudioVideoModel`; the pure `PeVideoModel` can be derived from `PeAudioVideoConfig.video_config`, but full AV checkpoint loading may have tied-key expectations outside this report.
- Source default video layer count is 6, while official nested video configs use 4.
- `hidden_size == num_attention_heads * head_dim` for observed configs, but source derives projection widths from explicit `head_dim`, so do not infer it from hidden size alone.
- `num_key_value_heads` can be less than `num_attention_heads`; source uses `repeat_kv`, so GQA admission needs a guard even though observed configs are MHA.
- Timm body is selected by string `vision_config.architecture` and `model_args`. DinoML should allowlist exact delegated bodies or route to a separate timm audit.
- Processor frame ABI differs between the two official checkpoints: variable/all-frame versus fixed 16 uniform frames.
- Video processor outputs channels-first `[B,T,C,H,W]`; any NTHWC/channel-last optimization must rewrite the whole processor-to-timm region and respect timm input layout.
- Text encoder is ModernBERT with mixed full/sliding bidirectional attention in the official configs. It is a composed family, not owned by `pe_video`.
- `PeVideoModel.get_text_features` assigns `pooler_output` from the whole `last_hidden_state` through the contrastive head, while `forward` uses `hidden_states[-1][:,0]`. Treat `forward` and embedding helper parity separately.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view` collapse `[B,T,C,H,W] -> [B*T,C,H,W]`
- `view` restore logits `[B*T,1024] -> [B,T,1024]`
- `transpose(1,2)` around temporal Conv1d
- `cat` for class-token prepend and optional padding-mask prepend
- `expand` learned class token to `[B,1,1792]`
- `arange`, `unsqueeze`, mask expansion, contiguous after attention transpose
- Optional variable-length video padding with `pad_sequence`

Neural primitives:

- Delegated timm image classifier: `vit_pe_core_large_patch14_336`, output logits `[B*T,1024]`
- `F.normalize(logits, dim=-1)` L2 normalization
- Linear `1024 -> 1792`, no bias
- Linear `1792 -> 1792`, with bias
- Temporal `GroupNorm(num_groups=1, C=1792)`, optional masked mean/var over hidden and sequence axes
- SiLU
- Conv1d `1792 -> 1792`, kernel 3, `padding="same"`, two applications in a residual block
- RMSNorm over hidden or per-head dim, fp32 variance
- SwiGLU-style MLP: Linear `1792 -> 4800`, Linear `1792 -> 4800`, SiLU, multiply, Linear `4800 -> 1792`
- Contrastive heads: LayerNorm eps `1e-6`, Linear no-bias video `1792 -> 1024`, text `1024 -> 1024`
- Matrix multiply `[Bv,1024] @ [Bt,1024].T`, scalar multiply/add

Attention primitives:

- Bidirectional self-attention, noncausal
- Q Linear `1792 -> num_heads*head_dim = 1792`
- K/V Linear `1792 -> num_kv_heads*head_dim`, observed 1792
- Per-head RMSNorm on Q and K with normalized shape 128
- Custom RoPE on Q/K using `stack_freqs` 2x2 rotation form
- Attention mask addition, fp32 softmax, dropout only training, output projection `1792 -> 1792`
- GQA repeat-KV path if `num_key_value_heads < num_attention_heads`

Preprocessing-coupled ops:

- Video decode is outside model graph. Processor may sample all frames or fixed uniform frames.
- Resize/rescale/normalize emits channels-first frames with mean/std `[0.5,0.5,0.5]`.
- Padding mask `[B,T]` is true for valid frames; model converts it into a bidirectional attention mask after class-token prepend.

Distributed/quantized/packed ops:

- No source-owned tensor parallelism, quantized weight format, packed projection layout, or remote-code-only kernel is required in this family.

## 5. Layer/block breakdown

Video embedding path:

```text
pixel_values_videos: [B,T,C,H,W]
frames = view(pixel_values_videos, [B*T,C,H,W])
vision_logits = timm_for_image_classification(frames).logits       # [B*T,1024]
logits = view(vision_logits, [B,T,1024])
logits = normalize(logits, dim=-1)
x = Linear(1024 -> 1792, bias=False)(logits)
x = Linear(1792 -> 1792, bias=True)(x)
```

Temporal patch embedder:

```text
x = cat([class_embedding.expand(B,1,1792), x], dim=1)              # [B,T+1,1792]
mask = cat([mask[:,[0]], mask], dim=1) if mask else None
y = transpose(x, 1, 2)                                             # [B,1792,T+1]
y = y + Conv1d(SiLU(GroupNorm(y))) -> Conv1d(SiLU(GroupNorm(...)))
x = transpose(y, 1, 2)
```

Encoder block, repeated `num_hidden_layers` times:

```text
residual = x
x_norm = RMSNorm(x)
q = RMSNorm_per_head(Linear(1792 -> Hq)(x_norm).view(B,S,heads,128)).transpose(1,2)
k = RMSNorm_per_head(Linear(1792 -> Hkv)(x_norm).view(B,S,kv_heads,128)).transpose(1,2)
v = Linear(1792 -> Hkv)(x_norm).view(B,S,kv_heads,128).transpose(1,2)
q,k = pe_video_rope(q,k,cos,sin)
attn = bidirectional_attention(q,k,v,mask)
x = residual + Linear(1792 -> 1792)(attn)
residual = x
x = residual + Linear(4800 -> 1792)(SiLU(Linear(1792 -> 4800)(RMSNorm(x))) * Linear(1792 -> 4800)(RMSNorm(x)))
```

Encoder output:

```text
x = RMSNorm(x)
x = Linear(1792 -> 1792, bias=False)(x)
last_hidden_state = x[:,1:]      # per-frame temporal tokens
pooler_output = x[:,0]           # class token
```

Text-video head:

```text
video_embeds = LayerNorm(1792) -> Linear(1792 -> 1024, bias=False)
text_embeds = ModernBERT hidden_states[-1][:,0] -> LayerNorm(1024) -> Linear(1024 -> 1024, bias=False)
logits_video_text = video_embeds @ text_embeds.T
logits_video_text = logits_video_text * text_video_logit_scale + text_video_logit_bias
```

## 6. Attention requirements

The owned PE-video encoder attention is noncausal bidirectional self-attention over `[class_token, frame_tokens]`.

| Field | Requirement |
|---|---|
| Causal | No |
| Cross-attention | No |
| MHA/GQA | MHA in observed configs; GQA supported by source via KV repetition |
| Heads/KV/head dim | observed 14/14/128 |
| Query length | `T+1`, after class-token prepend |
| Mask | bidirectional mask from valid-frame padding mask; no mask for equal-length or unpadded videos |
| RoPE | applied to Q/K before attention |
| Cache | no autoregressive KV cache |
| Backend compatibility | Source advertises SDPA/Flash/Flex attention through Transformers attention interface, but eager fallback is standard matmul/softmax |

ModernBERT text attention is a composed branch: bidirectional, mixed full/sliding attention with local attention 128 in official configs. Its operator and mask coverage should come from a ModernBERT audit.

## 7. Position encoding and custom math

PE-video RoPE is not the common `rotate_half` formulation in the generated source. It builds a 2x2 matrix from cos/sin halves:

```python
def pe_video_rope(q, k, cos, sin):
    dim = cos.size(-1)
    cos = cos[..., : dim // 2]
    sin = sin[..., : dim // 2]
    freqs = stack([cos, -sin, sin, cos], dim=-1).view(*cos.shape, 2, 2)
    freqs = freqs.unsqueeze(1)
    q2 = q.reshape(*q.shape[:-1], -1, 1, 2)
    k2 = k.reshape(*k.shape[:-1], -1, 1, 2)
    return (q2 * freqs).sum(5).flatten(3), (k2 * freqs).sum(5).flatten(3)
```

Cos/sin are computed from `position_ids = arange(S).unsqueeze(0)`, default `rope_theta=20000`, `head_dim=128`, in fp32 under disabled autocast, then cast to the input dtype. For fixed `T`, cos/sin can be precomputed by sequence length; for variable padded batches, only `S=max_T+1` matters because the same positions are used for all examples.

## 8. Preprocessing and input packing

Processor contract:

- Video decode and frame sampling are owned by the processor/data pipeline, not the neural graph.
- Raw decoded frames are RGB `[T,H,W,3]` from video readers, then converted to channels-first by the base video processor.
- With `return_tensors="pt"`, output `pixel_values_videos` is padded to `[B,Tmax,3,336,336]`.
- If video lengths differ, `padding_mask_videos` is emitted as `[B,Tmax]` with true valid entries.
- `facebook/pe-av-large`: `do_sample_frames=false`, `num_frames=null`; variable/all-frame behavior depends on the caller/video reader.
- `facebook/pe-av-large-16-frame`: `do_sample_frames=true`, `num_frames=16`; subclass samples indices `int(i*(total_frames-1)/(num_frames-1))`.
- Mean/std are `[0.5,0.5,0.5]`; rescale factor is `1/255`; resize is bilinear to 336x336.

No placeholder tokens, modality token IDs, cu-seqlens, or scatter-based embedding stitch exist in `PeVideoModel`. Text-video interaction is only a final similarity matrix.

## 9. Graph rewrite / lowering opportunities

### Rewrite: frame batch collapse

Source pattern:

```text
[B,T,C,H,W] -> view [B*T,C,H,W] -> timm image classifier -> view [B,T,1024]
```

Replacement pattern: treat frames as a larger image batch for the delegated vision tower.

Preconditions:

- `pixel_values_videos` contiguous or view-compatible.
- Model uses independent per-frame vision logits; no temporal mixing inside timm.
- Restore order is row-major `[batch, time]`.

Failure cases: non-contiguous caller buffers without an explicit reshape/copy plan; future vision configs that consume video clips instead of images.

Parity test: compare logits restoration against a Python loop over frames for mixed `B,T`.

### Rewrite: temporal Conv1d same-padding block

Source pattern:

```text
transpose B,S,H -> B,H,S
GroupNorm(groups=1) -> SiLU -> Conv1d(k=3,padding=same)
GroupNorm(groups=1) -> SiLU -> Conv1d(k=3,padding=same)
residual add -> transpose B,S,H
```

Replacement pattern: sequence-channel Conv1d kernel or guarded im2col+GEMM.

Preconditions:

- Kernel size 3, stride 1, dilation 1, groups 1.
- Padding exactly PyTorch `"same"` for odd kernel.
- Masked GroupNorm path is either absent or implemented with valid-mask mean/variance.

Failure cases: variable-length padded batches without masked normalization support; even-kernel same padding in future configs.

Parity test: random `[B,S,1792]`, with and without masks, fp32 and reduced precision.

### Rewrite: split Q/K/V projections

Source pattern: separate Q, K, V linears followed by per-head Q/K RMSNorm and RoPE.

Replacement pattern: optional fused QKV GEMM only when weights can be packed as `[Q,K,V]` and Q/K post-projection norms remain separate.

Preconditions:

- Same input tensor.
- Bias setting identical across Q/K/V.
- Preserve split order Q, K, V.
- Apply Q/K RMSNorm before RoPE, no norm on V.

Failure cases: GQA K/V output width differs from Q; packing must handle different output spans.

Parity test: compare packed projection splits, per-head RMSNorm, RoPE, and attention input tensors before softmax.

### Rewrite: layout translation around video frames

Candidate: NTHWC/channel-last decode-to-vision path.

Preconditions:

- DinoML owns processor output and the delegated vision body lowering.
- Axis rewrites include `[B,T,H,W,C] -> [B*T,H,W,C]`, timm patch/conv axes, and restoration to `[B,T,1024]`.
- The PE temporal encoder remains sequence-major `[B,T,H]`.

Failure cases: using opaque timm runtime that requires NCHW; caller-supplied preprocessed `[B,T,C,H,W]`; partial layout rewrite across the timm boundary.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm and per-head RMSNorm: used twice per encoder layer plus Q/K head norms; fp32 variance is required.
- Bidirectional attention with RoPE: sequence length is frame count plus one, but batch throughput matters; support MHA first, GQA with repeat-KV guard second.
- SwiGLU MLP: `gate_proj`, `up_proj`, SiLU, multiply, `down_proj` is a clear GEMM/elementwise/GEMM pattern.
- Frame-batch timm boundary: collapse `[B,T]` efficiently and avoid extra copies before the vision tower.

Medium priority:

- Masked GroupNorm + SiLU + Conv1d for variable-length video batches.
- Contrastive heads and similarity matrix, especially batched retrieval orientation `[video_batch,text_batch]`.
- Fixed-16-frame specialization for the 16-frame checkpoint, allowing static temporal sequence `S=17`.

Lower priority:

- Training contrastive loss.
- Output attention tensors.
- General timm model support beyond the allowlisted PE core architecture.

## 11. Runtime staging plan

Stage 1: parse `PeVideoEncoderConfig` and processor config; admit only `vit_pe_core_large_patch14_336`, `hidden_size=1792`, `head_dim=128`, and MHA observed configs.

Stage 2: run the delegated timm vision body as an external/composed subgraph and validate frame-batch collapse plus `1024 -> 1792 -> 1792` video feature projection.

Stage 3: implement PE temporal patch embedder with unmasked equal-frame batches first, then add padded `padding_mask_videos`.

Stage 4: implement one PE encoder block and full video encoder parity, including custom RoPE and per-head RMSNorm.

Stage 5: compose ModernBERT text branch from its own audit; for first text-video parity, consume precomputed text CLS embeddings if needed.

Stage 6: enable final contrastive heads and similarity matrix.

Stage 7: add optimized attention/MLP/Conv1d fusions and fixed-16-frame specialization.

Initially stub: training loss, output attentions/hidden-state capture, broad timm architectures, and AV joint/audio paths.

## 12. Parity and validation plan

- Processor parity: decode/list-of-frames and tensor input cases; verify `[B,T,C,336,336]`, normalization, fixed-16 frame indices, and variable padding mask.
- Frame collapse parity: compare collapsed timm call with looped per-frame calls.
- PE video projection parity: logits normalization and two linear projections.
- Masked GroupNorm parity: random masks, all-valid masks, and variable frame lengths.
- RoPE parity: compare custom 2x2 rotation against Transformers tensors for `S=17` and variable `S`.
- Single block parity: one encoder layer with eager attention and deterministic weights.
- Full video encoder parity: `last_hidden_state` `[B,T,1792]` and `pooler_output` `[B,1792]`.
- Text-video head parity: precomputed text hidden states through contrastive head and similarity orientation.
- End-to-end parity: official checkpoint video/text embedding similarity for a small batch.

Recommended tolerances: fp32 `rtol=1e-4, atol=1e-4`; fp16/bf16 after fused attention/MLP `rtol=5e-3, atol=5e-3`, with tighter checks on pre-attention projection tensors when debugging.

## 13. Performance probes

- Video decode and frame sampling throughput, split from GPU model time.
- Processor resize/normalize throughput for variable frames and fixed 16 frames.
- Timm image tower throughput as `[B*T,C,H,W]` batch-size sweep.
- PE temporal encoder throughput by `T` sweep: 16, 32, 64, 128, all-frame long clips.
- Masked versus unmasked temporal block overhead.
- Attention backend comparison: eager, SDPA, Flash/Flex-compatible path for noncausal sequence attention.
- Similarity matrix scaling with video batch and text batch orientation.
- Memory probe for storing per-frame vision logits/features before temporal encoder.

## 14. Skip/defer list

- Training and `return_loss`.
- Audio and audio-video joint `PeAudioVideoModel` branches.
- Broad ModernBERT implementation, except as a separately audited composed text encoder.
- Broad timm architecture support beyond `vit_pe_core_large_patch14_336`.
- Output attentions and hidden-state capture.
- GQA optimization for `num_key_value_heads < num_attention_heads` until a representative config requires it.
- Channel-last video layout optimization until the timm body is owned or explicitly lowered.
- Gated/private checkpoints; none were required for the current official configs.

## 15. Final implementation checklist

- [ ] Parse `PeVideoConfig` / nested `PeAudioVideoConfig.video_config`.
- [ ] Parse `PeVideoVideoProcessor` config and emit frame ABI guards.
- [ ] Admit/allowlist timm `vit_pe_core_large_patch14_336` as external delegated body.
- [ ] Implement frame-batch collapse/restore around vision logits.
- [ ] Implement `F.normalize(dim=-1)` for frame logits.
- [ ] Implement video projection linears `1024 -> 1792 -> 1792`.
- [ ] Implement class-token prepend and padding-mask prepend.
- [ ] Implement unmasked and masked `GroupNorm(num_groups=1)` for `[B,H,S]`.
- [ ] Implement Conv1d kernel-3 same-padding temporal residual block.
- [ ] Implement PE-video RMSNorm and per-head Q/K RMSNorm.
- [ ] Implement PE-video custom RoPE 2x2 stack form.
- [ ] Implement bidirectional MHA attention; add GQA repeat-KV guard.
- [ ] Implement SwiGLU MLP.
- [ ] Implement final video norm/output projection and pooler/last-hidden slicing.
- [ ] Compose or stub ModernBERT text CLS embeddings.
- [ ] Implement contrastive heads and `[video,text]` similarity logits.
- [ ] Add processor/frame ABI parity tests.
- [ ] Add single-block and full-video-encoder parity tests.
- [ ] Add end-to-end text-video similarity parity with an official checkpoint.
- [ ] Benchmark processor, timm frame tower, temporal encoder, and similarity matrix separately.
