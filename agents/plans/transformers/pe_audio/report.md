# Transformers Audit: `pe_audio`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/pe-a-frame-small, facebook/pe-a-frame-base, facebook/pe-a-frame-large
Config source: Hub config.json/preprocessor_config.json fetched 2026-05-13; source defaults from local checkout.
Source files inspected:
- transformers/src/transformers/models/pe_audio/configuration_pe_audio.py
- transformers/src/transformers/models/pe_audio/modeling_pe_audio.py
- transformers/src/transformers/models/pe_audio/modular_pe_audio.py
- transformers/src/transformers/models/pe_audio/feature_extraction_pe_audio.py
- transformers/src/transformers/models/pe_audio/processing_pe_audio.py
- transformers/tests/models/pe_audio/test_modeling_pe_audio.py
- transformers/docs/source/en/model_doc/pe_audio.md
Any missing files or assumptions: modeling_pe_audio.py is generated from modular_pe_audio.py; future source edits should target modular_pe_audio.py. Hub ids facebook/pe-a-base and facebook/pe-a-large returned HTTP 401. facebook/pe-av-* are pe_audio_video composite checkpoints, not native pe_audio.
```

Official source URLs at the pinned commit:
- [configuration_pe_audio.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pe_audio/configuration_pe_audio.py)
- [modeling_pe_audio.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pe_audio/modeling_pe_audio.py)
- [modular_pe_audio.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pe_audio/modular_pe_audio.py)
- [feature_extraction_pe_audio.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pe_audio/feature_extraction_pe_audio.py)

Small config snapshot: [config_sweep_snapshot.md](config_sweep_snapshot.md).

## 2. High-level architecture

`pe_audio` is an audio-text contrastive embedding model. It is not an autoregressive generator. The native source implements:

```text
raw waveform CPU preprocessing -> DAC-style Conv1d audio encoder -> Linear codec projection
-> prepend class token -> masked GroupNorm/SILU/Conv1d ResNet patch block
-> noncausal RoPE Transformer audio encoder -> audio projection head
text tokens -> delegated ModernBERT AutoModel -> CLS hidden state -> text projection head
audio/text dot-product similarity -> scale + bias -> logits
```

Primary DinoML target: `PeAudioFrameLevelModel` audio-text retrieval/localization, because the open native `pe_audio` checkpoints advertise `PeAudioFrameLevelModel`. A useful first subtarget is `PeAudioEncoder` plus `audio_head`, with text embeddings supplied by a separately audited ModernBERT path or by cached host tensors.

Stage decomposition:
- CPU/data pipeline: audio load/resample ownership, mono waveform validation, reflect padding to hop multiple, right padding, padding mask.
- Audio codec frontend: frozen/no-grad Conv1d DAC encoder plus 1x1 bottleneck and Linear projection.
- Audio sequence encoder: class token, masked 1D ResNet block, bidirectional self-attention encoder, final RMSNorm + Linear.
- Text branch: delegated `AutoModel.from_config(config.text_config)`; open configs use ModernBERT with local/sliding attention. Treat this as composed `modernbert` coverage, not owned by this report.
- Similarity heads: LayerNorm + bias-free projection for both branches, dot product orientation `[n_audio, n_text]`; frame-level logits `[n_audio, n_text, n_frames]`.

## 3. Important config dimensions

Source defaults for `PeAudioEncoderConfig` are large-like but with 6 layers, while open checkpoints override layer count.

| Field | Source default | Open checkpoint range | Runtime impact |
|---|---:|---:|---|
| audio `hidden_size` | 1792 | 768 / 1024 / 1792 | Transformer width, Conv1d patch block channels |
| audio `num_hidden_layers` | 6 | 12 / 16 / 28 | Repeated encoder layers |
| audio `num_attention_heads` | 14 | 6 / 8 / 14 | MHA heads |
| audio `num_key_value_heads` | defaults to heads | same as heads | MHA, no GQA in observed configs |
| `head_dim` | 128 | 128 | Q/K/V per-head width |
| audio `intermediate_size` | 4800 | 2048 / 2752 / 4800 | SwiGLU MLP width |
| audio max positions | 10000 | 10000 | RoPE cache/position guard |
| audio RoPE theta | 20000 | 20000 | Default RoPE |
| audio attention bias | false | false | Q/K/V/O are bias-free in observed configs |
| DAC downsampling ratios | `[2,8,10,12]` | same | Product 1920 samples, 40 ms at 48 kHz |
| DAC encoder hidden | 64 | same | First Conv1d channels |
| DAC output hidden | 1024 | same | DAC final Conv1d channels |
| DAC codebook dim | 128 | same | Bottleneck/data projection input |
| text model | ModernBERT | ModernBERT | Delegated branch |
| text hidden/layers/heads | 1024/22/16 | same | Text projection head input |
| text vocab/max positions | 50368/8192 | same | Token branch ABI |

Representative checkpoint sweep:

| Model id | Architecture | audio H | layers | heads | MLP | head dim | output |
|---|---|---:|---:|---:|---:|---:|---|
| `facebook/pe-a-frame-small` | `PeAudioFrameLevelModel` | 768 | 12 | 6 | 2048 | 128 | frame embeddings/logits |
| `facebook/pe-a-frame-base` | `PeAudioFrameLevelModel` | 1024 | 16 | 8 | 2752 | 128 | frame embeddings/logits |
| `facebook/pe-a-frame-large` | `PeAudioFrameLevelModel` | 1792 | 28 | 14 | 4800 | 128 | frame embeddings/logits |
| `facebook/pe-a-base` | inaccessible | unknown | unknown | unknown | unknown | unknown | HTTP 401 |
| `facebook/pe-a-large` | inaccessible | unknown | unknown | unknown | unknown | unknown | HTTP 401 |

## 3a. Family variation traps

- `head_dim` is explicit. Do not infer it solely from `hidden_size / num_attention_heads`, even though the open configs happen to match.
- Open configs use MHA (`num_key_value_heads == num_attention_heads`), but source supports GQA/MQA via `repeat_kv`; admission should guard KV heads dividing Q heads.
- The audio encoder is noncausal and has no autoregressive KV-cache requirement, despite accepting ignored `past_key_values/use_cache` kwargs through layer signatures.
- `PeAudioModel` versus `PeAudioFrameLevelModel` changes audio pooling: class-token pooled audio embedding versus all frame hidden states.
- Text is a delegated ModernBERT encoder with sliding/full attention pattern. Do not fold ModernBERT operator assumptions into `pe_audio` admission without a separate audit.
- Audio preprocessing changes user sample length: reflect-pad and then right-pad to multiples of `hop_length=1920`.
- The model uses NCL Conv1d regions with explicit `transpose(1, 2)` boundaries. Treat channel-last layout as a guarded optimization only.
- Masked GroupNorm uses `torch.masked.mean/var` over grouped channel/time axes when padding is present; it is not ordinary GroupNorm in masked batches.
- The DAC encoder runs under `torch.no_grad()` and `torch.backends.cudnn.flags(enabled=False)` in source; for inference this mainly documents a deterministic/compatibility expectation, not a graph op.

## 4. Operator coverage checklist

Tensor/layout ops:
- `transpose(1,2)`, `view/reshape`, `flatten`, `contiguous`, `expand`, `cat`, slicing `hidden_states[:, 1:]`, `hidden_states[:, 0]`, mask downsampling `padding_mask[:, ::hop_length]`.
- `arange`, `unsqueeze`, dtype casts to fp32 and back.

Neural primitives:
- Conv1d NCL: DAC initial `1 -> 64, kernel=7, pad=3`; DAC residual Conv1d `C -> C, kernel=7, dilation=1/3/9`; residual `C -> C, kernel=1`; downsampling Conv1d `C/2 -> C, kernel=2*stride, stride=stride, pad=ceil(stride/2)` for strides 2,8,10,12; final DAC Conv1d `d_model -> 1024, kernel=3,pad=1`; bottleneck `1024 -> 128, kernel=1`; patch Conv1d `H -> H, kernel=3,padding=same`.
- Linear: codec `128 -> H`; attention Q `H -> heads*128`; K/V `H -> kv_heads*128`; O `heads*128 -> H`; MLP gate/up `H -> intermediate`; MLP down `intermediate -> H`; final audio output `H -> H`; contrastive heads `audio H -> text H` and `text H -> text H`.
- Norms: RMSNorm over hidden size and head dim, LayerNorm in contrastive heads, masked GroupNorm with `num_groups=1`.
- Activations: `Snake1d`, SiLU, SwiGLU (`silu(gate) * up`).

Attention primitives:
- Noncausal self-attention, MHA/GQA-capable, additive bidirectional padding mask, RoPE on Q/K, fp32 softmax in eager path, SDPA/Flash/Flex dispatch supported by Transformers.

Position/rotary/custom math:
- Default RoPE with theta 20000 over `head_dim`, dynamic rope helper available for non-default rope types.

Preprocessing-coupled ops:
- Audio load/resample external to model graph; mono waveform expected.
- Reflect padding to hop multiple; batch right padding; padding mask renamed to `padding_mask`.

Delegated/composed ops:
- ModernBERT text branch operators are required for full end-to-end `PeAudioModel`, but should compose the separate `modernbert` audit.

## 5. Layer/block breakdown

DAC audio embedder:

```text
input_values: [B, 1, T]
x = Conv1d(1 -> 64, k=7, pad=3)
for stride in [2,8,10,12]:
  x = residual Snake/Conv1d stack with dilations 1,3,9
  x = Snake1d(x)
  x = Conv1d(C/2 -> C, k=2*stride, stride=stride, pad=ceil(stride/2))
x = Snake1d(x)
x = Conv1d(final_channels -> 1024, k=3, pad=1)
x = Conv1d(1024 -> 128, k=1)
x = transpose to [B, T_codec, 128]
x = Linear(128 -> H)
padding_mask = padding_mask[:, ::1920]
```

Patch embedder:

```text
x = cat([class_embedding.expand(B,1,H), x], dim=1)
padding_mask = cat([padding_mask[:, [0]], padding_mask], dim=1)
x_ncl = transpose(x, 1, 2)
mask = padding_mask.unsqueeze(1).expand_as(x_ncl)
x_ncl = x_ncl + Conv1d(SiLU(MaskedGroupNorm(x_ncl, mask)), k=3, same)
x_ncl = x_ncl + Conv1d(SiLU(MaskedGroupNorm(x_ncl, mask)), k=3, same)
x = transpose(x_ncl, 1, 2)
```

Audio encoder layer, repeated `N`:

```text
residual = x
x = RMSNorm(x)
q = Linear(H -> n_heads*head_dim, bias=attention_bias).view(B,S,n_heads,head_dim).transpose(1,2)
k = Linear(H -> n_kv_heads*head_dim, bias=attention_bias).view(B,S,n_kv_heads,head_dim).transpose(1,2)
v = Linear(H -> n_kv_heads*head_dim, bias=attention_bias).view(B,S,n_kv_heads,head_dim).transpose(1,2)
q = RMSNorm_head_dim(q); k = RMSNorm_head_dim(k)
q,k = RoPE(q,k,cos,sin)
attn = noncausal attention(q,k,v, bidirectional padding mask)
x = residual + Linear(n_heads*head_dim -> H, bias=attention_bias)(attn)
residual = x
x = RMSNorm(x)
x = residual + Linear(intermediate -> H)(silu(Linear(H -> intermediate)(x)) * Linear(H -> intermediate)(x))
```

Encoder output:

```text
x = RMSNorm(x)
x = Linear(H -> H, bias=False)
last_hidden_state = x[:, 1:]
pooler_output = x[:, 0]
```

Contrastive heads:

```text
audio_embeds = Linear(audio_H -> text_H, bias=False)(LayerNorm(audio_repr))
text_audio_embeds = Linear(text_H -> text_H, bias=False)(LayerNorm(text_cls))
PeAudioModel logits_audio_text = audio_embeds @ text_audio_embeds.T
PeAudioFrameLevelModel logits_audio_text = (audio_frame_embeds @ text_audio_embeds.T).transpose(1,2)
logits = logits * learned_scalar_scale + learned_scalar_bias
```

## 6. Attention requirements

Audio attention is encoder-style self-attention:
- Noncausal bidirectional self-attention.
- Q heads: `num_attention_heads`; KV heads: `num_key_value_heads`; `head_dim=128`.
- Open configs are MHA; source supports GQA by repeating KV heads before matmul.
- Masking: `create_bidirectional_mask` converts `[B,S]` padding mask into an additive attention mask compatible with backend attention.
- RoPE applies to Q/K after per-head RMSNorm and before attention.
- Eager math: `matmul(q, k^T) * head_dim**-0.5`, add mask, softmax in fp32, cast to query dtype, dropout only when training, matmul with V.
- Backend dispatch: `ALL_ATTENTION_FUNCTIONS` with source support flags for FlashAttention, SDPA, and FlexAttention.
- No autoregressive decode or KV cache is required for the primary target. Any `Cache` kwargs in layer signatures should be ignored/rejected for first integration.

Text attention is owned by ModernBERT. Full end-to-end PE Audio needs ModernBERT full/sliding attention parity, but `pe_audio` lowering should consume that as a composed branch.

## 7. Position encoding and custom math

Default audio RoPE:

```python
def pe_audio_default_rope(config):
    dim = config.head_dim
    base = config.rope_parameters["rope_theta"]  # observed: 20000
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    return inv_freq

def pe_audio_apply_rope(q, k, cos, sin):
    # q/k: [B, heads, S, head_dim], cos/sin: [1, S, head_dim]
    cos = cos[..., : cos.shape[-1] // 2]
    sin = sin[..., : sin.shape[-1] // 2]
    freqs = torch.stack((cos, -sin, sin, cos), dim=-1).view(*cos.shape, 2, 2)
    freqs = freqs.unsqueeze(1)
    q2 = q.reshape(*q.shape[:-1], -1, 1, 2)
    k2 = k.reshape(*k.shape[:-1], -1, 1, 2)
    return (q2 * freqs).sum(5).flatten(3), (k2 * freqs).sum(5).flatten(3)
```

`cos/sin` are computed in fp32 under autocast disabled, then cast to hidden dtype. For default RoPE, inv_freq can be precomputed; position ids depend on runtime sequence length.

Snake activation:

```python
def snake1d(x, alpha):
    return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
```

## 8. Preprocessing and input packing

Feature extractor contract:
- Input may be paths or waveform arrays. File loading uses `load_audio(audio_file, sampling_rate=48000)`.
- Required sampling rate: 48 kHz. Passing another rate raises.
- `feature_size=1`; stereo is explicitly unsupported.
- Raw arrays are coerced to float32 and transposed to channel/time conventions before validation.
- `_reflect_pad` pads each waveform to a multiple of `hop_length=1920`.
- Batch padding uses right padding, `padding_value=0.0`, `pad_to_multiple_of=1920`, and emits `padding_mask`.
- Model input tensor after tensor conversion is expected as `input_values` shaped `[B, 1, T]`.

There is no STFT/mel frontend. The model consumes waveform samples directly through Conv1d. The feature extractor should remain CPU/data-pipeline work initially; the GPU graph should start at `[B,1,T]` plus optional `[B,T]` padding mask.

Text processor contract:
- `PeAudioProcessor` combines `PeAudioFeatureExtractor` and `AutoTokenizer`.
- Open tokenizer configs use `TokenizersBackend`, model max length 8192, `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]`.
- Text branch uses the final hidden state at position 0, so tokenizer must provide a CLS-like first token.

No multimodal placeholder/scatter token stitching is present in `pe_audio`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: 1x1 Conv1d to Linear/GEMM

Source pattern: `Conv1d(Cin -> Cout, kernel_size=1, stride=1, padding=0)` for DAC residual `conv2` and bottleneck.

Replacement:

```text
NCL -> transpose/reshape [B*T, Cin] -> GEMM(weight.T) + bias -> reshape -> NCL
```

Preconditions:
- Static `kernel_size=1`, dilation 1, groups 1.
- Preserve NCL semantic order or prove surrounding transposes are fused.
- Bias handling matches Conv1d.

Failure cases: non-contiguous NCL layout, grouped convolution, dynamic stride/padding.

Parity test sketch: random NCL tensors for each channel width; compare Conv1d and GEMM path in fp32/fp16.

### Rewrite: non-overlap DAC downsampling Conv1d to im2col/GEMM

Source pattern: `Conv1d(Cin -> Cout, kernel=2*stride, stride=stride, padding=ceil(stride/2))`.

Replacement: guarded `Pad -> WindowFlatten -> GEMM -> reshape`.

Preconditions:
- Exact source padding, dilation 1, groups 1.
- Window extraction must preserve PyTorch Conv1d order.
- Dynamic output length formula must match Conv1d floor behavior.

This is a lowering strategy, not a semantic simplification, because windows overlap when `kernel=2*stride`.

### Rewrite: audio Transformer linear projections to GEMM families

Source pattern: separate Q/K/V Linear, then reshape/transpose.

Replacement: keep separate GEMMs first; optionally fuse QKV only when `attention_bias`, `head_dim`, KV grouping, and weight packing are normalized.

Weight transform: source stores standard PyTorch Linear weights `[out_features, in_features]`; packed QKV would concatenate rows in Q, K, V order.

Failure cases: GQA changes K/V row counts; `attention_bias=True` in source defaults is possible even though open configs use false.

### Rewrite: NCL patch ResNet region layout fusion

Source pattern: `[B,S,H] -> transpose -> masked GroupNorm/SiLU/Conv1d -> transpose`.

Replacement: keep NCL for the Conv1d/norm island, or convert the whole local island under a guarded layout pass.

Layout constraints:
- Mask expansion currently assumes NCL `[B,H,S]`.
- GroupNorm reductions are over channel group and time axes.
- Do not rewrite `dim=1`/`dim=2` silently; protect this region with a no-layout-translation guard unless all consumers are rewritten.

### Rewrite: similarity head as GEMM

Source pattern: `audio_embeds @ text_audio_embeds.T`.

Replacement: `gemm_rcr` with A `[n_audio, D]`, B `[n_text, D]`, output `[n_audio, n_text]`; frame-level flattens `[B_audio, S_audio, D]` to `[B_audio*S_audio, D]` before GEMM and reshapes/transposes to `[B_audio, n_text, S_audio]`.

Preconditions: both embeddings projected to same `text_hidden_size`, no implicit normalization beyond LayerNorm/proj.

## 10. Kernel fusion candidates

Highest priority:
- Conv1d NCL frontend, including stride/padding/dilation coverage for DAC.
- RMSNorm and head-dim RMSNorm.
- GEMM-backed Linear and SwiGLU MLP (`gate`, `up`, SiLU multiply, `down`).
- RoPE + noncausal attention prefill-style encoder attention.
- Masked GroupNorm + SiLU + Conv1d patch block for padded batches.

Medium priority:
- Snake1d fused elementwise inside DAC blocks.
- QKV projection packing for fixed MHA configs.
- Similarity GEMM for global and frame-level logits.
- LayerNorm + projection contrastive heads.

Lower priority:
- Training loss `logsigmoid` path.
- Dynamic/non-default RoPE variants; none observed in open configs.
- GQA optimization path; source supports it but open configs are MHA.

## 11. Runtime staging plan

Stage 1: parse `PeAudioConfig`/`PeAudioEncoderConfig`, load open `pe-a-frame-small` weights, and run config-only shape planning.

Stage 2: implement/compose feature extractor ABI outside compiled graph and accept `[B,1,T]` plus `padding_mask`.

Stage 3: audio DAC Conv1d frontend parity for one short padded waveform bucket.

Stage 4: patch embedder parity, including class token and masked GroupNorm behavior.

Stage 5: one audio encoder layer parity: RMSNorm, Q/K per-head RMSNorm, RoPE, noncausal attention, SwiGLU.

Stage 6: full `PeAudioEncoder` parity, returning `last_hidden_state`, `pooler_output`, and output mask.

Stage 7: audio contrastive head and frame-level audio embeddings. Stub text embeddings with cached tensors.

Stage 8: compose ModernBERT branch from its own audit and validate full `PeAudioFrameLevelModel` similarity logits.

Stage 9: optimize Conv1d/GEMM/attention kernels and frame-level similarity throughput.

## 12. Parity and validation plan

- Feature extractor tests: sampling-rate rejection, mono/stereo rejection, reflect padding to multiples of 1920, right padding mask shape.
- Conv1d DAC unit tests: each Conv1d pattern against PyTorch fp32, then fp16/bf16 tolerances.
- Snake1d elementwise parity over representative channel/time shapes.
- Masked GroupNorm parity with all-valid masks, partial masks, and padding-heavy masks.
- RoPE parity for sequence lengths 1, 2, 128, and padded audio-derived lengths.
- Single attention layer parity with and without padding masks.
- Full audio encoder parity for `small` first, then `base/large` shape-only smoke.
- Similarity parity: global `[n_audio,n_text]` and frame-level `[n_audio,n_text,n_frames]` orientation.
- End-to-end retrieval/localization parity with open Hub checkpoint and a fixed small audio fixture.

Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at `rtol=5e-2, atol=5e-2` for full model, with tighter per-op thresholds where accumulation is fp32.

## 13. Performance probes

- CPU preprocessing throughput: file load/resample, reflect pad, batch pad.
- Conv1d DAC frontend time versus sequence length and batch size.
- Patch masked GroupNorm/Conv1d time with and without masks.
- Encoder-only throughput over audio lengths mapping to 1, 10, 100, and 1000 codec frames.
- Attention backend comparison: eager composition, SDPA, Flash/Flex where supported.
- Frame-level similarity GEMM sweep over `n_audio`, `n_text`, `n_frames`.
- Memory probes for large checkpoint at long audio lengths: attention activations dominate because encoder attention is noncausal dense.
- Branch split benchmark: cached text embeddings versus full ModernBERT text branch.

## 14. Skip/defer list

- Training loss and contrastive training objective.
- Gradient checkpointing.
- `return_loss=True` path except as a small post-hoc validation.
- Text ModernBERT implementation inside this report; compose a separate audit.
- `pe_audio_video` and video paths.
- Gated `facebook/pe-a-base` / `facebook/pe-a-large` until access resolves configs.
- Non-default/dynamic RoPE types unless a checkpoint requires them.
- General GQA optimization unless a native checkpoint uses `num_key_value_heads < num_attention_heads`.
- GPU implementation of feature extraction; CPU/data pipeline is acceptable initially.

## 15. Final implementation checklist

- [ ] Parse `PeAudioConfig`, nested `PeAudioEncoderConfig`, nested DAC config, and ModernBERT text config.
- [ ] Implement/compose `PeAudioFeatureExtractor` CPU ABI: 48 kHz mono, reflect pad, right pad, padding mask.
- [ ] Load audio encoder weights and preserve PyTorch Linear row-major weight semantics.
- [ ] Implement Conv1d NCL coverage for DAC and patch block shapes.
- [ ] Implement `Snake1d`.
- [ ] Implement masked GroupNorm over NCL channel/time groups.
- [ ] Implement class-token prepend and padding-mask prepend/downsample rules.
- [ ] Implement audio RMSNorm and per-head Q/K RMSNorm.
- [ ] Implement default RoPE theta 20000 over `head_dim`.
- [ ] Implement noncausal MHA attention with bidirectional additive mask.
- [ ] Implement SwiGLU MLP and final audio projection.
- [ ] Implement contrastive heads and similarity GEMM orientation.
- [ ] Add `PeAudioEncoder` single-layer and full-stack parity tests.
- [ ] Add frame-level logits orientation parity test.
- [ ] Compose ModernBERT text branch from separate audit for end-to-end parity.
- [ ] Benchmark preprocessing, DAC frontend, encoder attention, and frame-level similarity separately.
