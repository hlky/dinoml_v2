# Evolla Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: westlake-repl/Evolla-10B-hf primary; westlake-repl/Evolla-10B-DPO-hf config sweep peer
Config source: Hugging Face raw config.json and processor_config.json retrieved 2026-05-13
Source files inspected:
  X:/H/transformers/src/transformers/models/evolla/configuration_evolla.py
  X:/H/transformers/src/transformers/models/evolla/modeling_evolla.py
  X:/H/transformers/src/transformers/models/evolla/processing_evolla.py
  X:/H/transformers/src/transformers/models/evolla/modular_evolla.py
  X:/H/transformers/docs/source/en/model_doc/evolla.md
  X:/H/transformers/tests/models/evolla/test_modeling_evolla.py
  X:/H/transformers/tests/models/evolla/test_processing_evolla.py
Snapshots:
  agents/plans/transformers/evolla/evolla_10b_config_snapshot.json
  agents/plans/transformers/evolla/hub_snapshot.md
Any missing files or assumptions:
  80B HF-converted config was not available from guessed raw URLs; westlake-repl/Evolla-80B is a raw split checkpoint listing, not a Transformers-format source basis.
```

Hub links inspected: [Evolla-10B-hf](https://huggingface.co/westlake-repl/Evolla-10B-hf), [Evolla-10B-DPO-hf](https://huggingface.co/westlake-repl/Evolla-10B-DPO-hf), [Evolla-80B](https://huggingface.co/westlake-repl/Evolla-80B). The guessed [Evolla-80B-hf](https://huggingface.co/westlake-repl/Evolla-80B-hf) and [Evolla-80B-DPO-hf](https://huggingface.co/westlake-repl/Evolla-80B-DPO-hf) `config.json` URLs returned 401.

## 2. High-level architecture

Evolla is a protein-conditioned text decoder:

```text
protein aa_seq + foldseek preprocessing -> SaProt encoder -> latent sequence compressor
text chat preprocessing -> Llama-like causal decoder + periodic protein cross-attention adapters
decoder final RMSNorm -> LM head -> logits/sampling
```

Stage decomposition:

| Stage | Source owner | Output | Cacheability |
| --- | --- | --- | --- |
| Protein preprocessing | `EvollaProcessor` | `protein_input_ids`, `protein_attention_mask` | CPU/data pipeline; cache per protein and max length. |
| Protein encoder | `EvollaSaProtProteinEncoder` | `[B, P, 1280]` hidden states | Independently cacheable for a fixed protein input. |
| Sequence compressor | `EvollaSequenceCompressorResampler` | `[B, 64, 4096]` protein prefix/features | Independently cacheable after encoder; required by decoder adapters. |
| Text prefill/decode | `EvollaModel` decoder | `[B, T, 4096]` hidden states | Causal KV cache code exists, but config defaults `use_cache=false`. |
| Logits | `EvollaForProteinText2Text` | `[B, kept_T, 128256]` | `logits_to_keep` can reduce last-token work. |

## 3. Important config dimensions

Primary checkpoint dimensions, from `westlake-repl/Evolla-10B-hf/config.json` unless marked as source default:

| Field | Value | Source/provenance |
| --- | ---: | --- |
| Text vocab | 128256 | config |
| Text hidden size | 4096 | config |
| Text layers | 32 | config |
| Text attention heads | 32 | config |
| Text KV heads | 8 | config; GQA with 4 query heads per KV head |
| Text head dim | 128 | inferred from source `hidden_size // num_attention_heads` |
| Text MLP intermediate | 14336 | config |
| Text max positions | 8192 | config |
| Text RoPE theta | 500000.0 | config `rope_theta`; source class default theta also 500000 |
| Text activation | SiLU gated MLP | config/source |
| Text attention/MLP bias | false / false | config |
| RMSNorm eps | 1e-5 | config |
| LM head tied to embeddings | false | config and distinct index entries |
| Protein vocab | 446 | config legacy flat fields; source default `SaProtConfig` same |
| Protein hidden size | 1280 | config legacy flat fields; source default same |
| Protein layers | 33 | config legacy flat fields; source default same |
| Protein heads/head dim | 20 / 64 | config plus inferred |
| Protein MLP intermediate | 5120 | config |
| Protein max positions | 1026 | config |
| Protein RoPE theta | 10000.0 | source `SaProtConfig` default |
| Resampler latents | 64 | config |
| Resampler depth/heads/head dim | 6 / 8 / 64 | config |
| Aligner inserted layers | 8 adapters | config; layers 3,7,11,15,19,23,27,31 for 32 layers |
| dtype | float32 weights in safetensors metadata | Hub metadata |
| Quantization | none in config; integration test uses bitsandbytes 4-bit load | source test |

Representative checkpoint sweep:

| Model id | Public config | Params/dtype metadata | Operator-significant variation |
| --- | --- | --- | --- |
| `westlake-repl/Evolla-10B-hf` | Yes | 10,392,101,680 F32 parameters | Primary architecture. |
| `westlake-repl/Evolla-10B-DPO-hf` | Yes | 10,392,101,680 F32 parameters | Same config as base; weight-only DPO variant. |
| `westlake-repl/Evolla-10B` | Raw checkpoint | Not safetensors metadata | Not a direct HF module source basis. |
| `westlake-repl/Evolla-80B` | No HF config found | Raw split files listed | Treat as unsupported until a Transformers config is available. |

## 3a. Family variation traps

- Hub config uses legacy flat `protein_*`, `rope_theta`, and `rope_scaling` fields, while current `EvollaConfig` source declares nested `protein_encoder_config` and `rope_parameters`. DinoML import should normalize the checkpoint config explicitly instead of assuming only the class defaults.
- Hub `model_type` is `"EvollaModel"`, while source `EvollaConfig.model_type` is `"evolla"`. Treat this as checkpoint metadata drift, not an operator.
- `hidden_size == num_attention_heads * head_dim` for 10B, but source allows an explicit `head_dim`; do not infer projection widths from hidden size when a config supplies `head_dim`.
- Text attention is GQA: K/V projection width is `num_key_value_heads * head_dim = 1024`, Q/O width is `4096`.
- Protein encoder attention is noncausal bidirectional MHA with pre-attention LayerNorm and query scaling before RoPE.
- Protein input is structure-aware paired text: `aa_seq[i].upper() + foldseek[i].lower()`. The processor currently accepts an `msa` key but does not consume it.
- Cross-attention adapters are only constructed with `protein_encoder_dim=config.hidden_size`; `structure_feats` and `msa_feats` are dummy/unsafe in this source basis unless adapter construction changes.
- Top-level `_supports_flash_attn` and `_supports_flex_attn` are false because the sequence compressor uses custom attention; do not assume a full FlashAttention-only lowering.
- `use_cache=false` by default. Generation can pass `use_cache`, but first integration can target prefill without persistent KV cache.
- Aligner uses learned scalar `tanh` gates initialized to zero; preserving initial and loaded gate semantics matters.
- LM head and token embedding are untied logical parameters.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for text `[128256,4096]` and protein `[446,1280]`.
- Reshape/view/transpose/permute/contiguous for QKV heads.
- Concatenate along sequence and feature axes, especially compressor K/V concat `[protein_tokens + latents]`.
- Chunk/split last dimension for compressor `to_kv`.
- Mask creation, broadcasting, `masked_fill`, boolean casts, `cumsum` for protein position ids.
- Slice/gather for `logits_to_keep`.

Neural network primitives:

- Linear/GEMM: text Q `4096->4096`, K/V `4096->1024`, O `4096->4096`, MLP gate/up `4096->14336`, down `14336->4096`, LM head `4096->128256`.
- Protein encoder linear: Q/K/V/O `1280->1280`, FFN `1280->5120->1280`.
- Compressor linear: Q `1280->512`, KV `1280->1024`, out `512->1280`, FFN `1280->5120->1280`, projector `1280->4096`.
- Aligner linear: query/key/value/out all `4096->4096`; FFN `4096->16384->4096`.
- RMSNorm for decoder and aligner, LayerNorm for protein encoder/compressor.
- Activations: SiLU gated MLP, exact erf GELU for SaProt FFN, PyTorch GELU for compressor/aligner FFN, tanh gates.

Attention primitives:

- Protein bidirectional self-attention with RoPE and additive bidirectional mask.
- Compressor latent cross-attention: learned latents query K/V from concatenated protein tokens plus latents, masked by protein mask plus latent ones.
- Text causal GQA self-attention with RoPE and optional dynamic KV cache.
- Periodic dense cross-attention from text queries to compressed protein features.

Position/rotary ops:

- Text RoPE with theta 500000.0 and float32 cos/sin construction.
- Protein RoPE with theta 10000.0 and position ids from unpadded token counts.

Generation/cache ops:

- Dynamic causal KV cache for text self-attention when `use_cache=true`.
- Protein encoder/compressor output cache is separate from KV cache and can be persisted per request.
- `logits_to_keep` last-token logits optimization.

Preprocessing-coupled ops:

- Llama-3 chat template.
- Protein tokenizer on interleaved amino-acid/Foldseek string.
- Truncation/padding: protein default 1024, text default 512.

Distributed/tensor-parallel ops:

- None in source forward. `pretraining_tp` appears in Hub config but inspected modeling code does not read it.

## 5. Layer/block breakdown

Protein encoder, repeated 33 times:

```text
x = protein_embedding(input_ids)
x = mask-token dropout/renorm(x, attention_mask)
cos,sin = protein_rope(position_ids)
x_attn = LayerNorm(x)
q,k,v = Linear(1280 -> 1280)(x_attn)
q = q * head_dim**-0.5
q,k = RoPE(q,k)
x = x + Dense(SelfAttention(q,k,v, bidirectional_mask))
x_ff = LayerNorm(x)
x = x + Linear(5120 -> 1280)(erf_gelu(Linear(1280 -> 5120)(x_ff)))
```

Sequence compressor, repeated 6 times:

```text
latents = learned [64,1280], expanded to [B,64,1280]
q = Linear(1280 -> 512)(LayerNorm(latents))
k,v = split(Linear(1280 -> 1024)(cat(protein_hidden, latents)))
latents = latents + Linear(512 -> 1280)(softmax(masked(q @ k.T)) @ v)
latents = latents + Linear(5120 -> 1280)(GELU(Linear(1280 -> 5120)(LayerNorm(latents))))
protein_features = LayerNorm(Linear(1280 -> 4096)(latents))
```

Text decoder block, repeated 32 times:

```text
residual = x
x = RMSNorm(x)
q = Linear(4096 -> 4096, bias=False)(x)
k,v = Linear(4096 -> 1024, bias=False)(x)
q,k = RoPE(q,k)
x = residual + Linear(4096 -> 4096, bias=False)(causal GQA(q,k,v, cache))
residual = x
x = RMSNorm(x)
x = residual + Linear(14336 -> 4096)(SiLU(gate_proj(x)) * up_proj(x))
if layer in [3,7,11,15,19,23,27,31]:
  x = protein_cross_attention_adapter(x, compressed_protein_features)
```

Adapter block:

```text
residual = x
q = Linear(4096 -> 4096)(RMSNorm(x))
k,v = Linear(4096 -> 4096)(compressed_protein_features)
x = residual + tanh(gate_attention) * Linear(4096 -> 4096, bias=True)(dense_cross_attention(q,k,v))
x = x + tanh(gate_ffw) * FFN(LayerNorm/RMSNorm-like feed-forward)
```

## 6. Attention requirements

Protein encoder attention is noncausal self-attention over protein tokens. It is MHA with 20 heads, 64-dim heads, RoPE applied to Q/K, and an attention mask produced from `protein_attention_mask`. Cached decoding is not applicable.

Compressor attention is a Perceiver-like latent cross-attention, not autoregressive generation. Query source is a learned latent table expanded to `[B,64,1280]`; K/V source is `cat(protein_hidden [B,P,1280], latents [B,64,1280])`; Q/K/V head count is 8 with head dim 64. Mask applies to K/V positions and includes valid protein tokens plus always-valid latent tokens.

Text decoder attention is causal self-attention with GQA: Q has 32 heads, K/V have 8 heads, head dim is 128. Cached keys and values are stored after RoPE application because `past_key_values.update` is called after applying RoPE. The cache shape before repeat expansion is `[B, 8, cached_T, 128]`; attention backends may repeat/broadcast K/V to 32 heads internally.

Aligner attention is rectangular cross-attention from text query length `T` to compressed protein length 64. It is MHA, not GQA: Q/K/V each become `[B,32,T_or_64,128]`. It uses multiplicative masks from query attention and protein batch mask, subtracts row max before softmax, then applies a gated residual.

FlashAttention compatibility: the top-level model declares flash/flex unsupported. SDPA-style lowering may be possible for the text decoder and protein encoder, but compressor and aligner require custom rectangular/masked attention parity first.

## 7. Position encoding and custom math

Text and protein RoPE both compute inverse frequencies in float32 and cast final cos/sin to the hidden dtype. Text uses theta 500000.0; protein uses theta 10000.0.

```python
def evolla_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Protein embedding mask-token dropout is model-specific:

```python
def saprot_mask_dropout(emb, input_ids, attention_mask, mask_id=4):
    emb = emb.masked_fill((input_ids == mask_id).unsqueeze(-1), 0.0)
    train_ratio = 0.15 * 0.8
    observed = (input_ids == mask_id).sum(-1).float() / attention_mask.sum(-1)
    return emb * (1 - train_ratio) / (1 - observed)[:, None, None]
```

The SaProt FFN GELU uses the exact erf formula; source comments warn that substituting `F.gelu` changes results.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- Input protein dicts require `aa_seq` and `foldseek` for useful execution. Processor-valid keys are `aa_seq`, `foldseek`, `msa`, but `msa` is not consumed.
- `aa_seq` and `foldseek` are zipped positionwise; mismatched lengths silently truncate to the shorter sequence in Python `zip`. DinoML should add a parity guard or reproduce this exactly.
- Structure-aware protein token string alternates uppercase amino acid and lowercase Foldseek character.
- Text messages are formatted through the tokenizer chat template with `add_generation_prompt=True`.

GPU/runtime graph inputs:

- `input_ids [B,T]`, `attention_mask [B,T]`.
- `protein_input_ids [B,P]`, `protein_attention_mask [B,P]`.
- Optional `structure_feats` and `msa_feats` are present in the forward signature, but current adapters are not built with structure/MSA projection modules; defer them.

There is no multimodal placeholder scatter into text embeddings. Protein information enters through adapter cross-attention, not as inserted text tokens.

## 9. Graph rewrite / lowering opportunities

### Rewrite: independent protein prefix cache

Source pattern: `protein_encoder(input_ids, mask) -> sequence_compressor_resampler`.

Replacement: precompute compressed protein features `[B,64,4096]` and feed decoder adapters.

Preconditions: same protein ids, mask, model weights, dtype, and resampler config; no training dropout.

Failure cases: training mode, changed protein truncation, future structure/MSA branches.

Parity test sketch: compare decoder logits with inline protein encoder versus supplied cached features through an internal test harness.

### Rewrite: latent compressor attention as rectangular attention kernel

Source pattern: normalize latents/media, project Q and packed KV, concat media+latents, masked softmax over KV.

Replacement: custom rectangular attention kernel with Q length 64, KV length `P+64`, 8 heads, head dim 64.

Preconditions: mask is binary and broadcast as source; preserve `sim - amax(sim).detach()` stabilization.

Failure cases: needing attention outputs, training dropout, nonstandard latent count/head dim.

### Rewrite: adapter cross-attention as protein-only cross-attention

Source pattern: adapter constructed only with `protein_encoder_dim=config.hidden_size`.

Replacement: rectangular text-to-protein attention over `[B,T,4096] x [B,64,4096]`.

Preconditions: no structure/MSA key/value states; protein batch mask all valid or known.

Failure cases: caller supplies structure/MSA features; adapter config changes to include those projections.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])`.

Replacement: for decode, project only the final token hidden state.

Preconditions: no loss requiring all shifted logits; `logits_to_keep` is 1 or equivalent final index.

Failure cases: teacher-forced labels, sequence logit inspection.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm for text decoder and aligner: appears twice per decoder block plus final norm.
- GQA RoPE + causal attention: dominant prefill/decode path.
- SwiGLU MLP: `SiLU(gate) * up -> down` dominates text block FLOPs.
- Protein prefix cache: removes 33-layer protein encoder from every decode step.

Medium priority:

- Compressor rectangular attention and FFN: fixed latent length 64 makes it a good specialized kernel target.
- Adapter cross-attention with tanh-gated residual: 8 insertions per decoder pass.
- Last-token-only LM head: very large vocab projection.

Lower priority:

- SaProt bidirectional attention optimization: important for long proteins but independently cacheable.
- Exact erf GELU fusion in SaProt FFN.
- Mask and position-id construction kernels; keep on CPU initially if graph boundary permits.

## 11. Runtime staging plan

Stage 1: parse/normalize config and load weights. Explicitly map legacy Hub protein fields into `SaProtConfig` and RoPE fields into DinoML text RoPE config.

Stage 2: implement SaProt protein encoder parity on small random inputs, including mask-token dropout behavior in eval.

Stage 3: implement sequence compressor and validate compressed protein features `[B,64,4096]`.

Stage 4: implement text decoder prefill without adapters, then enable protein-only adapters at layers 3,7,11,15,19,23,27,31.

Stage 5: add `EvollaForProteinText2Text` logits and `logits_to_keep`; validate prefill logits against Transformers.

Stage 6: enable optional text KV cache for decode, even though config default is `use_cache=false`.

Stage 7: add optimized attention/fusions and protein feature caching.

Initial stubs: structure/MSA branches, training, dropout, output attentions, and 80B raw checkpoint import.

## 12. Parity and validation plan

- Processor parity: fixed `aa_seq="AAAA"`, `foldseek="dddd"` should match the token ids in Transformers `test_processing_evolla.py`.
- Custom math tests: RoPE, exact SaProt GELU, RMSNorm, mask-token dropout renormalization, compressor mask fill.
- Protein single-layer parity: one SaProt layer with random weights and masks, fp32 tolerance `1e-5` absolute/relative.
- Compressor parity: random protein embeddings and masks, compare `[B,64,4096]` output.
- Adapter parity: protein-only cross-attention adapter with gate parameters set to nonzero as well as loaded zero initialization.
- Decoder block parity: one block with and without adapter.
- Prefill logits parity: full small random config from Transformers tests; then 10B shape smoke with sliced weights or meta initialization.
- Decode parity: one-step cache update should match prefill last-token logits.
- End-to-end integration: public 10B checkpoint with bitsandbytes or smaller extracted test fixture; assert generated text contains expected substrings only after numeric parity is stable.

Recommended tolerances: fp32 `1e-5` to `1e-4`; fp16/bf16 `1e-2` for full logits, tighter on isolated linear/norm ops. Label any SDPA/eager backend differences separately.

## 13. Performance probes

- Processor throughput: protein interleaving/tokenization and chat template formatting.
- Protein encoder throughput versus protein length sweep `P={128,512,1024}`.
- Compressor throughput for `P+64` KV length and fixed latent length 64.
- Prefill throughput with adapters on/off and text length sweep `T={128,512,2048,8192}`.
- Decode tokens/sec with and without protein feature cache.
- KV cache memory: `[layers=32, K/V, B, 8, T, 128]`.
- Adapter cross-attention cost by text length and batch size.
- LM head last-token versus all-token projection.
- Attention backend comparison: eager, SDPA, and any DinoML fused kernels, split by protein/text/compressor/adapter.
- Weight loading and optional quantization/dequant probe; Hub weights are F32, while source integration test uses bitsandbytes 4-bit only at load time.

## 14. Skip/defer list

- Training, gradients, dropout behavior beyond eval parity.
- Structure and MSA branches in `EvollaModel.forward`; current adapter construction does not support them safely.
- 80B raw checkpoint import and tensor-parallel/distributed execution.
- FlashAttention-only lowering for the whole model.
- RAG, DPO training, beam search, speculative decoding, and external generation controllers.
- Output attentions/hidden-state recording.
- Quantized provider path beyond ordinary dense or explicit load-time quantization support.

## 15. Final implementation checklist

- [ ] Normalize Evolla Hub config fields into DinoML config schemas.
- [ ] Load untied text embedding and LM head weights.
- [ ] Load SaProt protein encoder weights and tokenizer metadata.
- [ ] Implement protein position-id creation and mask-token dropout renormalization.
- [ ] Implement SaProt bidirectional RoPE attention and exact GELU FFN.
- [ ] Implement sequence compressor learned latents and rectangular attention.
- [ ] Implement text decoder RMSNorm, GQA RoPE attention, and SwiGLU MLP.
- [ ] Insert protein-only adapters at source-derived layer indices.
- [ ] Implement adapter tanh-gated attention and FFN residuals.
- [ ] Implement `logits_to_keep` and untied LM head projection.
- [ ] Add processor parity tests for protein/text packing.
- [ ] Add single-block and staged full-model parity tests.
- [ ] Add prefill and one-step decode parity tests.
- [ ] Benchmark processor, protein encoder, compressor, prefill, decode, and LM head separately.
