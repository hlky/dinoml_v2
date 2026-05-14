# CSM Transformers family audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: sesame/csm-1b is the official model id but returned 401 for inspected files; open mirrors/finetunes listed below were used for representative configs
Config source: Transformers source defaults plus HF config/preprocessor/generation/tokenizer JSON snapshots under _sources/
Source files inspected: configuration_csm.py, modular_csm.py, modeling_csm.py, generation_csm.py, processing_csm.py, convert_csm.py; codec composition checked against configuration_mimi.py and modeling_mimi.py
Any missing files or assumptions: official sesame/csm-1b config/generation/preprocessor/processor/tokenizer files were gated or unauthorized; checkpoint weights were not downloaded; no DinoML tests were run
```

Primary links:

- Transformers CSM source at commit: [configuration](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/csm/configuration_csm.py), [modular source](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/csm/modular_csm.py), [generated modeling](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/csm/modeling_csm.py), [generation](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/csm/generation_csm.py), [processor](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/csm/processing_csm.py), [conversion](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/csm/convert_csm.py).
- Codec source composed by CSM: [Mimi configuration](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mimi/configuration_mimi.py), [Mimi modeling](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mimi/modeling_mimi.py).
- Official gated/unauthorized repo: [sesame/csm-1b](https://huggingface.co/sesame/csm-1b). Access would resolve official `config.json`, generation config, preprocessor config, tokenizer config, and potentially processor metadata.
- Open native CSM configs inspected: [unsloth/csm-1b](https://huggingface.co/unsloth/csm-1b/blob/main/config.json), [eustlb/csm-1b](https://huggingface.co/eustlb/csm-1b/blob/main/config.json), [NMikka/CSM-1B-Georgian](https://huggingface.co/NMikka/CSM-1B-Georgian/blob/main/config.json), [keanteng/sesame-csm-elise](https://huggingface.co/keanteng/sesame-csm-elise/blob/main/config.json), [ArttuPakarinen/sesame-csm-FIN-parlament-full-finetune](https://huggingface.co/ArttuPakarinen/sesame-csm-FIN-parlament-full-finetune/blob/main/config.json).
- Historical or non-native CSM-like configs inspected and marked out of scope: [thomasgauthier/csm-1b-hf](https://huggingface.co/thomasgauthier/csm-1b-hf/blob/main/config.json), [mlx-community/csm-1b](https://huggingface.co/mlx-community/csm-1b/blob/main/config.json), [mlx-community/csm-1b-fp16](https://huggingface.co/mlx-community/csm-1b-fp16/blob/main/config.json), [mesolitica/Malaysian-sesame-csm-1b](https://huggingface.co/mesolitica/Malaysian-sesame-csm-1b/blob/main/config.json), [senstella/csm-expressiva-1b](https://huggingface.co/senstella/csm-expressiva-1b/blob/main/config.json).
- Local snapshots: `agents/plans/transformers/csm/_sources/`.

`modeling_csm.py` is generated from `modular_csm.py`. Future source edits should target `modular_csm.py`; runtime parity should still inspect generated `modeling_csm.py` because that is the import target.

## 2. High-level architecture

CSM is a speech/audio generation model built from two Llama-like causal decoders plus a Mimi neural audio codec:

```text
text/audio prompt preprocessing -> optional Mimi audio encode -> text/audio embedding stitch -> backbone causal decoder -> first RVQ codebook logits/sampling -> depth decoder generation for remaining codebooks -> RVQ frame sequence -> optional Mimi decode -> waveform
```

Stage decomposition:

- CPU/data pipeline: `CsmProcessor` applies the chat template, expands each `<|AUDIO|>` placeholder to the number of Mimi frames expected from waveform length, tokenizes with left padding, concatenates raw audio per sample, and emits `input_values_cutoffs`.
- Optional prompt-audio tokenizer stage: during prefill, `CsmForConditionalGeneration._merge_input_ids_with_input_values` loops over audio segments and calls `codec_model.encode`; this returns Mimi codes `[B_audio, 32, T]`, transposed to frame-major `[T, 32]` before CSM embedding.
- Prefix construction: text embeddings come from `embed_text_tokens`; audio frame embeddings come from `CsmBackboneModelEmbeddings`, which offsets each of 32 codebook channels by `channel * codebook_size`, embeds, and sums over the channel axis.
- Backbone generation stage: a 16-layer Llama-like causal decoder consumes `[B, S, 2048]` and predicts only the first codebook token through `lm_head: 2048 -> 2051`.
- Depth decoder stage: a separate 4-layer Llama-like decoder receives the backbone last hidden state in position 0, the sampled first codebook token in position 1, then autoregressively emits codebooks 1..31. It has its own KV cache and generation config.
- Optional waveform stage: `generate(output_audio=True)` decodes each sample independently by transposing generated codes from `[T, 32]` to Mimi `[1, 32, T]` and calling `codec_model.decode`.

First useful DinoML target: CSM audio-code generation parity through generated RVQ code tensors `[B, T, 32]`. Mimi waveform decode and Mimi prompt-audio encode are required for end-to-end TTS but should compose the separately audited `mimi` family.

## 3. Important config dimensions

Native 1B-style dimensions from open native CSM configs:

| Field | Backbone | Depth decoder | Mimi codec |
|---|---:|---:|---:|
| hidden_size | 2048 | 1024 | 512 |
| num_hidden_layers | 16 | 4 | encoder/decoder transformer 8 each |
| num_attention_heads | 32 | 8 | 8 |
| num_key_value_heads | 8 | 2 | 8 |
| head_dim | 64 | 128 | 64 |
| projection widths | Q 2048, K/V 512, O 2048 | Q 1024, K/V 256, O 1024 | Q/K/V 512 |
| intermediate_size | 8192 | 8192 | 2048 |
| code/text vocab | codebook vocab 2051, text vocab 128256 | codebook vocab 2051 | codebook size 2048 |
| codebook/channel count | 32 | 32 positions incl. first-codebook prefix | 32 quantizers |
| max_position_embeddings | 2048 | 33 | 8000 transformer positions |
| RoPE | Llama3 scaling, theta 500000 | Llama3 scaling, theta 500000 | default theta 10000 |
| activation | SiLU gated MLP | SiLU gated MLP | GELU MLP |
| cache support | DynamicCache self-attn | DynamicCache self-attn | `use_cache=False` by default; streaming optional |
| dtype metadata | fp16/fp32/bf16 varies by checkpoint config | follows config | often fp32/bf16 metadata |

Representative checkpoint/config sweep:

| Repo | Scope | Status | Operator-significant notes |
|---|---|---|---|
| [sesame/csm-1b](https://huggingface.co/sesame/csm-1b) | official | gated/401 | Would resolve official JSON metadata; source docstrings and conversion script name it as the canonical checkpoint. |
| [unsloth/csm-1b](https://huggingface.co/unsloth/csm-1b/blob/main/config.json) | native mirror | in scope | `model_type=csm`, `CsmForConditionalGeneration`, 16+4 layers, 32 codebooks, `torch_dtype=float16`, Mimi nested config with `frame_rate=12.5`. |
| [eustlb/csm-1b](https://huggingface.co/eustlb/csm-1b/blob/main/config.json) | native mirror/variant | in scope | Same dimensions as Unsloth; `torch_dtype=float32`. |
| [NMikka/CSM-1B-Georgian](https://huggingface.co/NMikka/CSM-1B-Georgian/blob/main/config.json) | native finetune | in scope | Same structure; carries historical `audio_num_codebooks`/`audio_vocab_size`; `pad_token_id=128002`; `torch_dtype=bfloat16`; Mimi uses `_frame_rate=12.5` and `use_streaming=false`. |
| [keanteng/sesame-csm-elise](https://huggingface.co/keanteng/sesame-csm-elise/blob/main/config.json) | native finetune | in scope | Same structure; float32 metadata; native CSM with nested Mimi. |
| [ArttuPakarinen/sesame-csm-FIN-parlament-full-finetune](https://huggingface.co/ArttuPakarinen/sesame-csm-FIN-parlament-full-finetune/blob/main/config.json) | native finetune | in scope | Same dimensions; omits some dtype/frame-rate fields that config defaults fill. |
| [thomasgauthier/csm-1b-hf](https://huggingface.co/thomasgauthier/csm-1b-hf/blob/main/config.json) | historical remote/custom style | out of scope | `architectures=["CSMModel"]`, nested `backbone_config`/`decoder_config`, `decoder_config.vocab_size=128256`; current native source does not instantiate this schema. |
| [mlx-community/csm-1b](https://huggingface.co/mlx-community/csm-1b/blob/main/config.json) | MLX schema | out of scope | `model_type="sesame"` with flavor strings only; current native `CsmConfig` does not implement this schema. |

## 3a. Family variation traps

- `modeling_csm.py` is generated. Use `modular_csm.py` for source changes, but audit generated code for runtime behavior.
- The backbone consumes two incompatible `input_ids` ranks: text prompt IDs `[B, S]` are converted to `inputs_embeds`, while generated audio-code frames are direct `[B, S, 32]` IDs.
- Text and audio code vocabularies are separate: `text_vocab_size=128256`; codebook vocab is `vocab_size=2051`; codebook EOS is `0`; codebook pad is `2050`.
- `CsmConfig.attribute_map` maps `codebook_size` to `vocab_size`. The backbone embedding code reads `config.codebook_size`, so loaders must preserve that alias.
- Audio embeddings use channel offsets and sum over the last axis. Do not flatten `[B, T, 32]` audio codes into sequence length or vocab IDs.
- Backbone and depth decoder are both GQA: 32 Q heads/8 KV heads and 8 Q heads/2 KV heads.
- Depth decoder position 0 is not a normal token: it is replaced by `backbone_last_hidden_state`, then projected from 2048 to 1024.
- Depth decoder generation must emit exactly `num_codebooks - 1 = 31` new tokens. Generation config validation rejects any other depth decoder min/max new token count.
- `logits_to_keep` is effectively constrained by docstring to `0` for training or `1` for generation in the top-level CSM forward path.
- Prompt audio encode and output audio decode are unbatched loops in source; batching them changes visible ordering and shape contracts unless guarded carefully.
- Several configs contain historical fields (`audio_num_codebooks`, `audio_vocab_size`, flavor strings, `backbone_config`, `decoder_config`, MLX `model_type=sesame`). Treat them as unsupported by this native source unless a conversion path is explicitly used.
- No vision layout exists. Layout passes must guard audio code axes, waveform axes, Mimi conv NCL axes, attention head/time axes, and codebook logits axes from generic channel-last rewrites.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`, `reshape`, `transpose`, `contiguous` for attention `[B,T,H,D] <-> [B,H,T,D]`.
- `unsqueeze`, `squeeze`, `pad`, `cat`, `stack`, `repeat`, `sum(dim=2)`, `where`, boolean masks, `nonzero`, indexed assignment, `arange`, `clamp`, slicing.
- Axis guards: audio code frame tensor `[B, T, 32]`; Mimi codes `[B, 32, T]`; waveform `[B, channels, samples]`; attention `[B, H, T, D]`; codebook head logits `[B, selected_codebooks, 2051]`.

Neural primitives:

- Text embedding: `Embedding(128256 -> 2048)`.
- Backbone audio embedding: `Embedding(32 * 2051 -> 2048)`, offset by `[0, 2051, ..., 31*2051]`, then `sum(dim=2)`.
- Backbone block x16: biasless Q `Linear(2048 -> 2048)`, K/V `Linear(2048 -> 512)`, O `Linear(2048 -> 2048)`, gated MLP `2048 -> 8192 -> 2048`.
- Backbone LM head: biasless `Linear(2048 -> 2051)`.
- Depth decoder embedding: `Embedding(32 * 2051 -> 2048)`, plus biasless projector `Linear(2048 -> 1024)`.
- Depth decoder block x4: biasless Q `Linear(1024 -> 1024)`, K/V `Linear(1024 -> 256)`, O `Linear(1024 -> 1024)`, gated MLP `1024 -> 8192 -> 1024`.
- Depth decoder codebook head: parameter `weight [31, 1024, 2051]`; per-position linear heads selected by codebook index and stacked.
- RMSNorm over last dim with fp32 variance; SiLU; fp32 softmax in eager attention.

Attention/cache ops:

- Causal dense self-attention with GQA repeat and optional SDPA/FlashAttention dispatch.
- RoPE on Q/K before cache update; cached K is post-RoPE and pre-repeat.
- DynamicCache allocation/update/get length/reorder for both backbone and depth decoder loops.
- Generation sampling: logits processors, softmax, multinomial or argmax, EOS-all-codebooks stop, pad finished frames.

Preprocessing and codec-coupled ops:

- Processor-side placeholder expansion using encoded audio length from Mimi conv parameters.
- Encodec-style feature extractor config for raw audio: `feature_size=1`, `sampling_rate=24000`, right padding, `return_attention_mask=true`; `CsmProcessor` removes `padding_mask` before model call.
- Mimi encode/decode composition: causal Conv1d/ConvTranspose1d, transformer layers, RVQ codebook lookup/nearest centroid, code tensor transpose.

## 5. Layer/block breakdown

Backbone embedding and block:

```text
prompt path:
  text_ids [B,S] -> Embedding(128256,2048) -> inputs_embeds [B,S,2048]
  audio placeholders are overwritten by summed audio-code embeddings when input_values is present

decode path:
  audio_code_ids [B,S,32]
  offsets [32] = arange(32) * 2051
  embeds = Embedding(65632,2048)(audio_code_ids + offsets)
  x = embeds.sum(dim=2)  # [B,S,2048]

repeated 16 times:
  r = x
  x = RMSNorm(x)
  q = Linear(2048 -> 2048)(x).view(B,T,32,64).transpose(1,2)
  k = Linear(2048 -> 512)(x).view(B,T,8,64).transpose(1,2)
  v = Linear(2048 -> 512)(x).view(B,T,8,64).transpose(1,2)
  q,k = RoPE(q,k)
  k,v = cache.update(k,v)
  x = r + Linear(2048 -> 2048)(causal_gqa_attention(q,k,v))
  r = x
  x = RMSNorm(x)
  x = r + down_proj(silu(gate_proj(x)) * up_proj(x))

x = RMSNorm(x)
logits = Linear(2048 -> 2051)(x[:, last, :])
```

Depth decoder:

```text
input_ids [B,L] are positions along the 32-codebook depth axis
position_ids = arange(past_seen, past_seen + L)
codebook_idx = clamp(position_ids - 1, min=0)
offset = codebook_idx * 2051
x = Embedding(65632,2048)(input_ids + offset)
if backbone_last_hidden_state is provided:
  x[:,0] = backbone_last_hidden_state  # [B,2048]
x = Linear(2048 -> 1024, bias=False)(x)

repeated 4 times:
  r = x
  x = RMSNorm(x)
  q = Linear(1024 -> 1024)(x).view(B,L,8,128).transpose(1,2)
  k/v = Linear(1024 -> 256)(x).view(B,L,2,128).transpose(1,2)
  q,k = RoPE(q,k)
  k,v = cache.update(k,v)
  x = r + Linear(1024 -> 1024)(causal_gqa_attention(q,k,v))
  r = x
  x = RMSNorm(x)
  x = r + down_proj(silu(gate_proj(x)) * up_proj(x))

x = RMSNorm(x)
for each selected codebook position i>=1:
  logits[:, i-1, :] = linear(x[:, i, :], weight[i-1].T)
```

All CSM attention and MLP projections are biasless for inspected native configs.

## 6. Attention requirements

Backbone attention:

- Causal self-attention only.
- GQA with Q heads 32, KV heads 8, repeat factor 4, head dim 64.
- Cache shape per layer before repeat: K/V `[B, 8, T_cache, 64]`; after repeat for eager attention: `[B, 32, T_cache, 64]`.
- Cached keys are stored after RoPE because cache update follows `apply_rotary_pos_emb`.
- Source mask comes from `create_causal_mask`; eager path adds mask before fp32 softmax.

Depth decoder attention:

- Causal self-attention only.
- GQA with Q heads 8, KV heads 2, repeat factor 4, head dim 128.
- Cache shape per layer before repeat: K/V `[B, 2, T_depth_cache, 128]`; generation length is bounded to 33 positions for 32 codebooks.
- It ignores caller-provided `position_ids` and builds identical depth-axis position IDs across the batch.

Backend compatibility:

- `CsmPreTrainedModel` advertises FlashAttention and SDPA support; eager fallback is explicit.
- Fused attention parity must preserve: Q/K/V reshape order, RoPE before cache update, GQA repeat semantics, mask addition before softmax, fp32 softmax accumulation in eager mode, dropout disabled in inference.

There is no cross-attention in native CSM. The backbone/depth split communicates through `backbone_last_hidden_state`, not through attention.

## 7. Position encoding and custom math

CSM uses Llama-style RoPE with config-driven `rope_parameters`. Open native config JSON files still use the older `rope_scaling` key, and Transformers `PreTrainedConfig` exposes that as `rope_parameters` before the modeling code reads it. The conversion script says the original model used Llama3 scaling with `rope_theta=500000`; native configs store equivalent Llama3 scaling:

- Backbone: `factor=32`, `low_freq_factor=0.125`, `high_freq_factor=0.5`, `original_max_position_embeddings=1024`.
- Depth decoder: `factor=32`, `low_freq_factor=0.001953125`, `high_freq_factor=0.0078125`, `original_max_position_embeddings=16`.

Short source-equivalent snippets:

```python
def csm_audio_frame_embed(code_ids, table, vocab_size=2051):
    # code_ids: [B, T, 32]
    offsets = torch.arange(32, device=code_ids.device) * vocab_size
    return torch.nn.functional.embedding(code_ids + offsets, table).sum(dim=2)
```

```python
def csm_rms_norm(x, weight, eps):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * y.to(x.dtype)
```

```python
def csm_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-b, a), dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

RoPE cos/sin can be precomputed by max position per submodel, but decode still needs explicit cache-length offsets for `position_ids`.

## 8. Preprocessing and input packing

Processor/text contract:

- Text input must be a string or list of strings; `return_tensors` must be `"pt"`.
- Defaults: tokenizer padding enabled, left padding, `add_special_tokens=False`.
- Chat template requires stringified integer roles and emits `bos_token`, `[speaker_id]`, text, `eos_token`, then `<|AUDIO|><|audio_eos|>` when audio content exists.
- `audio_token_id=128002`, `audio_eos_token_id=128003` in native configs; tokenizer snapshots define `<|AUDIO|>` and `<|audio_eos|>`.

Audio prompt preprocessing:

- Expected waveform sampling rate is 24000 Hz.
- Feature extractor snapshot is `EncodecFeatureExtractor` with `feature_size=1`, right padding, zero padding value, and `return_attention_mask=true`.
- `CsmProcessor` concatenates all audio segments for each sample into one waveform and records cumulative end positions in `input_values_cutoffs [B, max_num_audio]`, padded with `-1`.
- Processor computes placeholder expansion length using a source-coded conv length recurrence over kernel sizes `[7,3,1,8,3,1,10,3,1,12,3,1,16,3,4]`, strides `[1,1,1,4,1,1,5,1,1,6,1,1,8,1,2]`, causal padding, and 24000 Hz input.

Model prompt-audio stitch:

- `_merge_input_ids_with_input_values` embeds all text first: `inputs_embeds = embed_text_tokens(input_ids)`.
- It loops through each segment, slices `audio_batch = batch_input_values[..., start:end]`, calls `codec_model.encode(audio_batch.unsqueeze(0))`, transposes `audio_codes` from `[1, 32, T]` to `[1, T, 32]`, pads segment token arrays to a common frame length, and masks valid frames through `codec_model.get_audio_codes_mask`.
- Positions where `input_ids == audio_token_id` are replaced by summed codebook embeddings.
- Positions where `input_ids == audio_eos_token_id` are replaced by the all-codebook-EOS embedding.

Generation controller:

- Top-level generation supports greedy and sampling only; beam, contrastive, assisted, and other HF generation modes are rejected.
- `MaxLengthCriteria` is retained; other stopping criteria are ignored with warnings.
- Each loop samples one first-codebook token from backbone logits, then calls depth decoder `.generate` to obtain the remaining 31 codebook tokens.
- Finished sequences are padded with `codebook_pad_token_id=2050`; EOS stop requires all codebooks except the last axis slice used by source `input_ids[:, -1, :-1]` to equal `codebook_eos_token_id=0`.
- `output_audio=True` decodes per sample by trimming at the first all-codebook EOS frame and calling `codec_model.decode(audio_codes_batch.transpose(0,1).unsqueeze(0))`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: multi-codebook embedding offset-sum

Source pattern:

```text
offsets = arange(num_codebooks) * vocab_size
emb = embedding(input_ids + offsets)
out = emb.sum(dim=2)
```

Replacement:

```text
CodebookEmbeddingSum(input_ids[B,T,C], table[C*V,H], offsets[C]) -> [B,T,H]
```

Preconditions: `input_ids` rank 3; last dimension equals `num_codebooks`; offsets are exactly `arange(C) * vocab_size`; embedding table first dim equals `C * vocab_size`; reduction axis is the codebook axis. Failure cases: text `[B,S]` path, nonstandard codebook order, or a later model with weighted channel fusion.

Parity test sketch: random int code IDs in `[0,V)`, compare fused op with PyTorch embedding-plus-sum for multiple batch/time sizes and EOS/PAD IDs.

### Rewrite: depth decoder fixed-length position-specific heads

Source pattern:

```text
weight [31,H,V]
for codebook_idx in range(num_selected):
    logits_i = linear(hidden[:, codebook_idx, :], weight[codebook_idx].T)
stack(logits_i, dim=1)
```

Replacement: grouped batched GEMM or static unrolled GEMMs over the small depth axis.

Preconditions: selected codebook indices are contiguous depth positions after the placeholder; hidden states are `[B, L, H]`; each position selects a distinct weight slice. Failure cases: tensor-valued `logits_to_keep` with non-contiguous indices, training paths selecting arbitrary codebook positions.

Parity test sketch: compare one-frame generation path and full 31-head training path.

### Rewrite: Llama GQA block canonicalization

Source pattern: RMSNorm -> biasless Q/K/V -> RoPE -> cache update -> GQA attention -> O projection -> residual -> RMSNorm -> SwiGLU -> residual.

Replacement: existing decoder block primitive with explicit GQA and RoPE metadata.

Preconditions: no attention/MLP bias; activation `silu`; RoPE source matches Llama3/default config; cached K is post-RoPE; mask semantics preserved. Failure cases: configs with `attention_bias` or `mlp_bias`, non-default attention backend numerics, training dropout.

### Layout guard: audio token/code axes

Do not apply channel-last or generic layout translation to:

- CSM code frames `[B, T, 32]`.
- Mimi codec codes `[B, 32, T]`.
- Waveforms `[B, channels, samples]`.
- `sum(dim=2)` in codebook embedding.
- `transpose(0,1)`/`transpose(1,-1)` around Mimi code tensors.

These axes are semantic codebook/time/channel contracts, not image layouts.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm, biasless QKV projections, RoPE, GQA attention with KV cache for 16-layer backbone.
- Last-token-only backbone logits `Linear(2048 -> 2051)`.
- Depth decoder micro-decode path: 4-layer GQA decoder over at most 33 depth positions, repeatedly called once per generated frame.
- Multi-codebook embedding offset-sum, because it is on every generated frame and every audio prompt frame.

Medium priority:

- SwiGLU MLP fusion for `silu(gate) * up` plus down projection.
- Grouped/static codebook heads for the 31 depth decoder vocab projections.
- Causal mask and RoPE precompute for the fixed depth-axis length.
- Mimi decode batching if waveform output is in scope.

Lower priority:

- Training-only label expansion, random depth label skipping, and depth loss paths.
- General Mimi streaming cache and padding cache, unless prompt-audio streaming becomes a product target.
- Full audio preprocessing on GPU; initial pipeline can keep feature extraction and codec composition outside DinoML compiled graph.

## 11. Runtime staging plan

Stage 1: parse native `CsmConfig` and reject historical/MLX schemas that current source does not implement. Load backbone, depth decoder, embedding aliases, and codebook-head weights.

Stage 2: implement one CSM decoder block parity for both backbone and depth dimensions with GQA, RoPE, RMSNorm, and SwiGLU.

Stage 3: implement backbone prefill from `inputs_embeds` and generated-code `[B,T,32]` paths, including KV cache and last-token logits.

Stage 4: implement depth decoder one-frame schedule: inject `backbone_last_hidden_state`, generate exactly 31 remaining codebooks, and return `[B,32]` frame IDs.

Stage 5: implement top-level CSM generation loop for code tensors `[B,T,32]`, greedy first, sampling second. Stub waveform decode initially.

Stage 6: compose Mimi encode/decode as separate runtime stages for prompt audio and `output_audio=True`.

Stage 7: optimize GQA attention, codebook embedding sum, codebook heads, continuous batching, and cache allocation.

## 12. Parity and validation plan

- Config admission tests: native CSM configs load; `model_type=sesame`, custom `CSMModel`, and flavor-only configs reject or route to a separate converter.
- Random tensor tests: RMSNorm, RoPE, GQA repeat, audio embedding offset-sum, depth codebook head.
- Single-layer parity: backbone block `[B, T, 2048]`, depth block `[B, L<=33, 1024]`, fp32 then reduced precision.
- Cache parity: prefill plus one decode step against full-sequence forward for backbone and depth decoder; verify cached K shapes are pre-repeat and post-RoPE.
- Prompt stitch parity: synthetic audio codes through embedding overwrite positions for `<|AUDIO|>` and `<|audio_eos|>`.
- Generation parity: greedy generation with a small `max_new_tokens`; compare generated `[B,T,32]` frames and EOS/pad behavior.
- Codec composition parity: when Mimi is staged, compare `codec_model.decode(codes.transpose(0,1).unsqueeze(0))` waveform tensors.
- Suggested tolerances: fp32 `1e-5` block/logit tolerance; fp16/bf16 `1e-2` for logits and hidden states, with exact equality for sampled IDs under greedy mode.

## 13. Performance probes

- Processor throughput: chat template/tokenizer/audio placeholder expansion per request.
- Mimi prompt encode throughput by audio seconds and number of segments.
- Backbone prefill latency versus prompt length and count of embedded audio frames.
- Backbone decode tokens/sec for generated frames with cache.
- Depth decoder per-frame latency for greedy and sampled codebook generation.
- End-to-end code-frame generation requests/hour, separating backbone and depth decoder time.
- KV cache memory: backbone `[layers=16, 2, B, 8, T, 64]`; depth `[layers=4, 2, B, 2, <=33, 128]`.
- Codebook head projection cost for 31 small vocab GEMMs.
- Optional Mimi decode throughput by generated duration; source currently loops per sample, so batch-size sweep should expose lost batching.

## 14. Skip/defer list

- Training losses, random depth label skipping, and `labels=-101` behavior.
- Beam search and non-greedy/non-sampling HF generation modes; source rejects them.
- Prompt-audio streaming/padding cache in Mimi.
- Batched Mimi encode/decode improvements; source TODOs mark this as not currently batched.
- Quantization and packed weight formats; native source does not define CSM-specific quantized weights.
- Multi-GPU tensor parallel plans; source `_tp_plan`/`_pp_plan` are disabled for depth decoder.
- Historical remote-code/custom CSM schemas and MLX `model_type=sesame` configs.

## 15. Final implementation checklist

- [ ] Parse native `CsmConfig` and nested `CsmDepthDecoderConfig`/`MimiConfig`
- [ ] Reject or route historical `CSMModel`, `model_type=sesame`, and flavor-only configs
- [ ] Load/tie `backbone_model.embed_tokens.embed_audio_tokens.weight` and `depth_decoder.model.embed_tokens.weight`
- [ ] Implement codebook offset embedding sum for `[B,T,32]`
- [ ] Implement CSM RMSNorm, SiLU gated MLP, RoPE, causal GQA attention, and DynamicCache
- [ ] Implement backbone prefill/decode and last-token codebook-0 logits
- [ ] Implement depth decoder hidden-state injection and fixed 31-token generation
- [ ] Implement codebook-position head lowering for `weight [31,H,2051]`
- [ ] Implement top-level greedy code-frame generation and EOS/pad handling
- [ ] Add sampling parity with top-k/temperature logits processors
- [ ] Compose Mimi prompt encode and waveform decode as later stages
- [ ] Add no-layout-translation guards for codebook/time/audio axes
- [ ] Add block, cache, generation, and codec-composition parity tests
- [ ] Benchmark processor, Mimi encode/decode, backbone prefill/decode, and depth decoder per-frame latency
