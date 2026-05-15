# BLT Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: blt family; primary accessible mirrors inspected: itazap/blt-1b-hf, itazap/blt-1b-testing, itazap/blt-7b-hf
Config source: Hugging Face raw config snapshots saved in this folder
Source files inspected:
- transformers/src/transformers/models/blt/configuration_blt.py
- transformers/src/transformers/models/blt/modeling_blt.py
- transformers/src/transformers/models/blt/modular_blt.py
- transformers/src/transformers/models/blt/convert_blt_weights_to_hf.py
Any missing files or assumptions:
- modeling_blt.py is generated from modular_blt.py; modular_blt.py is authoritative for upstream source edits.
- Official facebook/blt-1b, facebook/blt-7b, facebook/blt, and facebook/blt-entropy are gated/manual-access Hub repos. Links: https://huggingface.co/facebook/blt-1b, https://huggingface.co/facebook/blt-7b, https://huggingface.co/facebook/blt-entropy.
- This report targets native in-library Transformers BLT, not the original facebookresearch/blt runtime.
- No processor file exists in the in-library BLT directory. Tokenizer coupling was inspected through accessible tokenizer_config.json snapshots only.
```

Small snapshots:

- `itazap_blt-1b-hf_config.json`
- `itazap_blt-1b-testing_config.json`
- `itazap_blt-7b-hf_config.json`
- `itazap_blt-1b-hf_tokenizer_config.json`

Primary runtime target: `BltForCausalLM` byte-level causal language modeling. `BltModel`, `BltPatcher`, `BltLocalEncoder`, `BltGlobalTransformer`, and `BltLocalDecoder` are required submodules for default `patch_in_forward=True`; standalone submodule parity is useful but secondary.

## 2. High-level architecture

BLT is a byte-level causal LM with dynamic byte-to-patch segmentation:

```text
byte tokenizer/input_ids
  -> optional entropy patcher
  -> byte/hash local encoder
  -> patch reduction + encoder patch cross-attention
  -> global patch transformer
  -> local byte decoder with patch cross-attention
  -> byte logits
```

Stage decomposition:

- CPU/data pipeline: byte tokenizer emits `input_ids` and optional `attention_mask`. The accessible tokenizer config reports `PreTrainedTokenizerFast`, vocab 260, BOS/EOS/PAD/UNK tokens, and model inputs `input_ids`, `attention_mask`.
- Runtime patching: if `patch_lengths` is not supplied and `patching_mode=="entropy"` with `patch_in_forward=True`, the frozen `BltPatcher` runs a causal byte LM, computes categorical entropy, thresholds patch starts, and emits ragged-like padded patch lengths.
- Local encoder: byte embeddings plus hashed n-gram byte embeddings; one or more causal transformer layers; `scatter_reduce(..., reduce="amax")` maps bytes into patches; projected patch queries cross-attend back to local byte states.
- Global transformer: causal transformer over patch embeddings, not bytes.
- Local decoder: byte-level transformer states cross-attend to global patch states, then self-attend causally and produce normalized byte hidden states.
- LM head: `Linear(decoder_hidden_size -> vocab_size)` with `logits_to_keep` slicing.

Independently stageable validation points: hash embedding, patcher entropy-to-lengths, patch id construction, local encoder byte states and patch states, global patch hidden states, local decoder byte states, final logits.

## 3. Important config dimensions

Source defaults come from `configuration_blt.py`; checkpoint values below come from raw `config.json` snapshots.

| Field | 1B mirror | 7B mirror | Source default |
|---|---:|---:|---:|
| vocab_size | 260 | 260 | 260 |
| max_position_embeddings, top-level | 4096 | 4096 | 4096 |
| patch_in_forward | true | true | true |
| patching_mode | entropy | entropy | entropy |
| patch_size | 4 | 4 | 4 |
| patching_threshold | 1.335442066192627 | 1.335442066192627 | 1.335442066192627 |
| max_patch_length | null | null | null |
| cross_attn_k | 2 | 4 | 2 |
| hash group sizes | 3,4,5,6,7,8 | 3,4,5,6,7,8 | 3,4,5,6,7,8 |
| hash vocab per group/function | 500002 | 500002 | 500002 |
| hash functions | 1 | 1 | 1 |
| patcher | 14L, hidden 768, heads 12, FFN 2048 | same | same |
| local encoder | 1L, hidden 1024, heads 16, FFN 2816 | 1L, hidden 1280, heads 20, FFN 3584 | 1L, hidden 1024, heads 16, FFN int(8h/3) if omitted |
| global transformer | 25L, hidden 2048, heads 16, FFN 5632 | 32L, hidden 4096, heads 32, FFN 11008 | 25L, hidden 2048, heads 16, FFN 5632 |
| local decoder | 9L, hidden 1024, heads 16, FFN 2816 | 6L, hidden 1280, heads 20, FFN 3584 | 9L, hidden 1024, heads 16, FFN 2816 |
| num_key_value_heads | null in snapshots -> MHA via config post-init | null -> MHA | defaults to num_attention_heads |
| head_dim | hidden / heads: 64 local, 128 global | 64 local, 128 global | hidden / heads |
| RoPE theta | patcher 10000; local/global 500000 | same | local/global default theta 500000 |
| attention backend | local/global `_attn_implementation: sdpa`; patcher carries historical `attn_impl: xformers` | same | source supports SDPA, not flash/flex |
| tie_word_embeddings | false | false | false forced |

Representative checkpoint sweep:

| Checkpoint/config | Access | Operator-significant notes |
|---|---|---|
| `itazap/blt-1b-hf` | open mirror | Standard 1B HF-format config; `cross_attn_k=2`; encoder patch projection width equals global hidden size, so global `token_embedding_projection` is identity. |
| `itazap/blt-1b-testing` | open mirror | Same dimensions as 1B mirror. Difference observed: config key spelling uses `rms_norm_eps` where the 1B HF mirror uses historical `norm_eps` in subconfigs. Native source reads `rms_norm_eps`; effective value is still 1e-5. |
| `itazap/blt-7b-hf` | open mirror | `cross_attn_k=4`; local hidden 1280; global hidden 4096. Encoder cross output is 5120, so global transformer inserts `Linear(5120 -> 4096)`. |
| `facebook/blt-1b` | gated official | Config/weights require accepting Hub terms; use for official provenance once access is granted. |
| `facebook/blt-7b` | gated official | Same; likely needed to confirm official 7B parity against the open mirror. |

## 3a. Family variation traps

- `cross_attn_k` changes tensor ranks and widths. Local encoder/decoder repeat patch tokens by `cross_attn_k`; masks repeat along query or key dimension accordingly.
- 7B has `encoder_hidden_size * cross_attn_k != global_hidden_size`, so the global transformer may need `Linear(5120 -> 4096)`.
- `num_key_value_heads` can be lower than `num_attention_heads` by config, even though inspected mirrors use MHA. GQA/MQA support should not be hardcoded out.
- RoPE is BLT-specific interleaved RoPE. It uses `repeat_interleave(freqs, 2)` and `rotate_half` over even/odd pairs, not Llama's concat-half rotation.
- Patcher configs in mirrors include historical `attn_impl` and `attn_bias_type`, but native `BltPatcherConfig` does not read these fields. Treat them as ignored for this source basis.
- Some converted configs use `norm_eps`, while native source uses `rms_norm_eps`. If loading through HF config normalizes this implicitly, DinoML should still canonicalize to the source field and reject ambiguous non-default drift.
- Runtime `patch_lengths` bypasses entropy patching. First DinoML integration can require caller-provided `patch_lengths` to avoid dynamic entropy segmentation.
- `scatter_reduce(..., reduce="amax")` is central to patch reduction. This is not a dense reshape or simple mean pooling.
- `torch.distributions.Categorical(logits=...).entropy()` in patcher is source behavior for entropy patching.
- `process_patch_lengths(max_patch_length != None)` uses Python loops over per-example positive lengths; this is a staged runtime/data-pipeline concern.
- Cross-attention masks are dense 4D additive masks shaped `[B,1,Q,K]` with `torch.finfo(dtype).min` for disallowed pairs.
- Causal masks use Transformers `create_causal_mask`; backend parity depends on the selected SDPA/eager behavior.
- Input is byte/token sequence `[B,S]`; no image/video/audio layout pass applies. Protect all sequence axes from NHWC/channel-last translation.
- `_tied_weights_keys` advertises local encoder embedding and LM head tying, but `tie_word_embeddings=false` in configs. Weight loader must preserve any actual alias if a checkpoint provides it, without assuming tied weights.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding([260, H])` for local encoder and patcher.
- Fused hash embedding table `[hash_vocab * hash_functions * len(group_sizes), encoder_hidden]`; 1B/7B mirrors: `[3,000,012, H_local]`.
- `cat`, `cumsum`, `arange`, comparisons, `sum`, `masked_fill`, boolean masks, `repeat_interleave`, `expand`, `reshape`, `view`, `transpose`, `contiguous`, slicing.
- Sliding byte windows via `unfold(dim=1, size=group_size, step=1)` after left zero padding.
- Integer polynomial hash and modulo for byte group ids.
- `scatter_reduce(dim=1, reduce="amax", include_self=False)` from byte states to patch states.

Neural network primitives:

- Biasless linear projections throughout.
- RMSNorm with fp32 variance and output cast back to input dtype.
- SwiGLU MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- Residual adds and dropout disabled in eval but present in source.
- Final `Linear(decoder_hidden -> 260)` logits.

Attention primitives:

- Causal self-attention over byte and patch sequences.
- Noncausal cross-attention between repeated patch queries and byte keys, and byte queries to repeated patch keys.
- MHA by inspected configs; GQA/MQA required by config surface.
- SDPA-compatible path; eager fallback repeats KV heads then matmul/softmax/matmul.

Position/rotary/custom math:

- BLT interleaved RoPE, float32 trig, theta 10000 for patcher and 500000 for local/global in inspected configs.
- Dynamic RoPE update decorator exists but inspected configs use `rope_type: default`.

Generation/cache ops:

- `EncoderDecoderCache(DynamicCache, DynamicCache)` is used even though BLT is decoder-like. Local encoder self-attention receives `self_attention_cache`; local decoder self-attention receives `cross_attention_cache`; cross-attention modules themselves do not cache K/V.
- Cache update stores post-RoPE self-attention K and raw V from `BltSelfAttention`.
- Global transformer is recomputed without cache in native forward.
- `logits_to_keep` can be int or tensor indices.

Preprocessing-coupled ops:

- Byte tokenizer ABI with BOS/EOS/PAD/UNK.
- Optional runtime patcher entropy: categorical entropy from logits, threshold, patch starts, patch lengths, optional max-length splitting.
- Caller-supplied `patch_lengths` should be admitted as a first-stage ABI.

Packed/varlen metadata ops:

- `patch_lengths [B,P]` and derived `patch_ids [B,S]`.
- Cross-attention mask construction from patch ids.

## 5. Layer/block breakdown

Transformer layer, repeated in patcher/local/global/decoder:

```text
residual = x
x = RMSNorm(x)
q = Linear(H -> num_heads * head_dim, bias=False)
k = Linear(H -> num_kv_heads * head_dim, bias=False)
v = Linear(H -> num_kv_heads * head_dim, bias=False)
q,k = interleaved RoPE(q,k)
attn = causal_attention(q,k,v, additive_mask, optional self-cache)
x = residual + Linear(num_heads * head_dim -> H, bias=False)(attn)
residual = x
x = RMSNorm(x)
x = residual + Linear(I -> H, bias=False)(silu(Linear(H -> I)(x)) * Linear(H -> I)(x))
```

Patcher:

```text
input_ids -> Embedding(260 -> 768)
14 causal transformer layers
RMSNorm -> Linear(768 -> 260)
Categorical(logits).entropy()
entropy threshold -> patch_lengths [B,P]
```

Local encoder:

```text
input_ids -> byte embedding + summed hash n-gram embeddings
1 causal transformer layer in 1B/7B mirrors
patch_reduce: scatter_reduce amax over patch_ids -> [B,P,H_local]
Linear(H_local -> H_local * cross_attn_k)
reshape -> [B,P*cross_attn_k,H_local]
cross-attention: patch queries attend to byte states
encoder_cross_states -> [B,P,H_local*cross_attn_k]
```

Global transformer:

```text
encoder_cross_states [B,P,H_local*cross_attn_k]
optional Linear(H_local*cross_attn_k -> H_global)
25 or 32 causal transformer layers over patches
global_hidden_states [B,P,H_global]
```

Local decoder:

```text
encoder_hidden_states [B,S,H_local]
global_hidden_states -> Linear(H_global -> H_local*cross_attn_k)
reshape -> [B,P*cross_attn_k,H_local]
for each decoder layer:
  if layer 0 or cross_attn_all_layers:
    byte states cross-attend to patch states and add residual
  causal self-attention transformer layer
RMSNorm -> [B,S,H_local]
LM head -> [B,logits_to_keep,260]
```

## 6. Attention requirements

Self-attention:

- Causal, self-attention, default MHA in inspected mirrors.
- Q heads and KV heads are config-dependent; `num_key_value_groups = num_heads // num_key_value_heads`.
- Head dim is `hidden_size // num_attention_heads`.
- Source applies RoPE before cache update.
- Eager fallback computes `softmax(q @ k^T * head_dim^-0.5 + mask)` in fp32, casts to query dtype, then multiplies V.
- SDPA is the intended native backend in inspected configs; `_supports_flash_attn=False`, `_supports_flex_attn=False`.

Cross-attention:

- Noncausal dense attention with additive mask.
- Query and key states are RMSNormed before projection (`q_norm`, `k_norm`).
- No RoPE and no cache in cross-attention.
- Shapes:
  - Encoder patch cross-attn: Q length `P * cross_attn_k`, KV length `S`.
  - Decoder patch cross-attn: Q length `S`, KV length `P * cross_attn_k`.
- Cross-attention output already adds `hidden_states` internally, and callers add it again in local encoder/decoder. DinoML parity should follow source order exactly unless an upstream fix changes it.

KV/cache:

- Local encoder and local decoder self-attention can update dynamic caches; global patch transformer is not passed a cache.
- Since patching and masks depend on the full byte sequence and patch segmentation, first decode parity should prefer full-prefix recompute or a narrow validated cache mode with fixed supplied `patch_lengths`.

## 7. Position encoding and custom math

BLT RoPE differs from Llama-style half-split rotation:

```python
def blt_rope_tables(position_ids, inv_freq, attention_scaling, dtype):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)
    return (emb.cos() * attention_scaling).to(dtype), (emb.sin() * attention_scaling).to(dtype)

def blt_rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)

def blt_apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + blt_rotate_half(q) * sin, k * cos + blt_rotate_half(k) * sin
```

Precompute opportunity: inverse frequencies and static position tables can be cached for a max sequence/patch length per subconfig. Dynamic inputs: `position_ids`, cache offset, dtype/device, and any future non-default `rope_type`.

Patch-related custom math:

```text
hash_ids = sum(window_byte[i] * prime**i) % encoder_hash_byte_group_vocab
patch_ids = count(patch_start <= token_position) - 1
patch_reduce = scatter_reduce_amax(hidden_states, dim=1, index=patch_ids)
```

## 8. Preprocessing and input packing

The in-library model consumes byte-level `input_ids [B,S]`, not image/audio features. The tokenizer config exposes `input_ids` and `attention_mask`; BOS/EOS are configured as enabled by tokenizer metadata.

Runtime-relevant packing:

- `patch_lengths [B,P]` is either provided by caller or computed by the frozen patcher.
- `patch_ids [B,S]` is derived from cumulative patch starts.
- Encoder cross-attention mask uses `patches_as_queries=True`: `[B,1,P*cross_attn_k,S]`.
- Decoder cross-attention mask uses `patches_as_queries=False`: `[B,1,S,P*cross_attn_k]`.
- Mask entries are `0` for allowed and minimum finite dtype value for blocked.

First DinoML ABI recommendation: accept `input_ids`, `attention_mask`, and required `patch_lengths`. Add entropy patching only after core local/global/decoder parity is stable.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Hash Embedding Pipeline

Source pattern:

```text
left_pad -> unfold byte windows -> polynomial hash -> modulo -> embedding -> sum with byte embedding
```

Replacement pattern:

```text
HashWindowIds(group_sizes, primes, vocab) -> GatherEmbedding -> SumEmbeddings
```

Preconditions:

- `encoder_hash_byte_group_nb_functions` and `encoder_hash_byte_group_size` are static.
- Prime table matches source order.
- Input ids are integer byte ids in tokenizer vocabulary range.
- Hash arithmetic uses int64 semantics and modulo `encoder_hash_byte_group_vocab`.

Failure cases: changed hash function count beyond prime table, non-byte tokenizer semantics, overflow mismatch across runtimes.

Parity test sketch: random byte ids with all configured group sizes, compare hash ids and summed embeddings against PyTorch.

### Rewrite: Patch Cross-Attention Mask Builder

Source pattern:

```text
patch_ids equality -> repeat_interleave(cross_attn_k) -> invert -> masked_fill(finfo.min)
```

Replacement pattern:

```text
PatchIdMask(patch_ids, num_patches, sequence_length, patches_as_queries, cross_attn_k)
```

Preconditions:

- Dense additive attention mask is accepted by attention backend.
- `patch_ids` are monotonic and within `[0,P-1]`.
- Query/key length equations match encoder or decoder cross-attention mode.

Failure cases: non-monotonic/adversarial patch ids, sparse attention backend without additive mask support.

### Rewrite: Patch Reduce Amax

Source pattern:

```text
zeros([B,P,H]).scatter_reduce(dim=1, index=patch_ids[...,None].expand, src=hidden, reduce="amax", include_self=False)
```

Replacement pattern:

```text
SegmentAmaxByPatch(hidden [B,S,H], patch_ids [B,S], P) -> [B,P,H]
```

Preconditions:

- Every real patch has at least one token.
- Dummy/right padding behavior is guarded if `patch_lengths` sum is less than sequence length.
- Match `include_self=False` behavior for empty patches.

Parity test sketch: batch with unequal patch counts and varied patch lengths; compare exact fp32 and reduced precision tolerances.

### Rewrite: BLT Block Fusion

Source pattern:

```text
RMSNorm -> biasless QKV linears -> RoPE -> attention -> output linear -> residual
RMSNorm -> SwiGLU MLP -> residual
```

Replacement pattern: standard DinoML RMSNorm, GEMM, RoPE, attention, SwiGLU, GEMM fusions with BLT-specific RoPE.

Preconditions:

- Bias is absent.
- Head_dim and KV grouping are static for the compiled config.
- Attention mask/caches follow BLT local/global/decoder stage contracts.

Failure cases: non-default RoPE type, unsupported GQA, dynamic patch length not represented in shape plan.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: everywhere, fp32 variance, simple high-impact kernel.
- Biasless GEMM for projections and MLPs: dominant compute in patcher/global/local/decoder.
- SDPA/GQA causal attention with BLT interleaved RoPE: required for self-attention parity and performance.
- Segment amax patch reduction: source-critical and not covered by ordinary transformer kernels.
- Cross-attention mask builder plus dense cross-attention: patch/byte bridge correctness.

Medium priority:

- Hash embedding fused window/hash/gather/sum: avoids many small integer/gather ops.
- SwiGLU fused activation multiply.
- Last-token/logits-to-keep LM head slice.
- Optional global `Linear(H_local*cross_attn_k -> H_global)` for 7B.

Lower priority:

- Entropy patcher kernelization, including categorical entropy and threshold segmentation.
- `max_patch_length` splitting path.
- Cache-optimized decode, because native graph still recomputes patch/global pieces and needs careful patch-length semantics.

## 11. Runtime staging plan

1. Parse BLT config and canonicalize subconfigs, including `norm_eps`/`rms_norm_eps`, `cross_attn_k`, KV heads, and `encoder_cross_output_size`.
2. Load weights for submodules; preserve embedding/LM-head alias metadata if present, but do not assume tying.
3. Implement `input_ids + supplied patch_lengths` forward without entropy patcher; validate hash embedding, patch ids, masks, local encoder, global transformer, local decoder, and logits.
4. Add one-block parity for `BltTransformerLayer` with BLT RoPE.
5. Add full 1B-shape graph parity in fp32/bf16/fp16 with small random weights.
6. Add 7B-shape config smoke/parity with the global projection path.
7. Add optimized SDPA/Flash-style attention where BLT masks and RoPE are supported.
8. Add entropy patcher as a separate optional stage.
9. Revisit decode cache only after full-prefix parity and supplied-patch-length ABI are stable.

Stub initially: training loss, entropy patching, cache decode, gated official weights, `max_patch_length`, non-default RoPE, and any config-advertised fields not read by native source.

## 12. Parity and validation plan

- Config parser tests: 1B mirror, 1B testing mirror, 7B mirror, source-default config, and historical `norm_eps` spelling.
- Hash embedding tests: byte ids across group sizes 3-8, compare hash ids and summed embedding output.
- BLT RoPE tests: compare cos/sin and rotated Q/K against source for random position ids.
- Attention tests: self-attention MHA and forced GQA config; additive causal mask; cross-attention patch masks.
- Patch id/mask tests: unequal patch counts, `cross_attn_k=2` and `4`, encoder and decoder mask modes.
- Segment amax tests: random `[B,S,H]`, patch ids, empty/right-padded patch edge cases.
- Single block parity: RMSNorm/attention/MLP residual in fp32, fp16, bf16.
- Stage parity: local encoder, global transformer, local decoder independently.
- Full prefill logits parity: tiny random model, supplied `patch_lengths`.
- Patcher parity: entropy and threshold path after core graph parity.
- Decode parity: full-prefix recompute first; cache update only after patch semantics are fixed.

Suggested tolerances: fp32 `1e-5` absolute/relative for most ops, looser around softmax/entropy; fp16/bf16 `1e-2` to `3e-2` depending on attention backend and accumulation.

## 13. Performance probes

- Hash embedding throughput versus sequence length and group-size count.
- Patch reduction throughput versus `B,S,P,H`.
- Cross-attention mask construction cost for `S` and `P*cross_attn_k`.
- Patcher-only throughput and entropy segmentation overhead.
- Local encoder throughput, global patch transformer throughput, local decoder throughput.
- Full prefill tokens/sec for supplied `patch_lengths`.
- Full prefill with entropy patching enabled.
- Sequence-length sweep: byte length `S`, patch count `P`, and average patch length.
- `cross_attn_k` sweep: 2 vs 4.
- Attention backend comparison: eager, SDPA, DinoML fused attention.
- KV/cache memory for local encoder/local decoder caches if decode cache is admitted.
- LM-head `logits_to_keep` savings.

## 14. Skip/defer list

- Training and loss.
- Gradient checkpointing and output recording hooks.
- Beam search and generation-controller policy beyond standard byte sampling.
- Entropy patching in the first compiled graph if caller can provide `patch_lengths`.
- `max_patch_length` splitting path.
- Non-default RoPE types.
- Official gated checkpoint validation until access is granted.
- Cache-optimized decode until supplied-patch-length full-prefix parity is solid.
- FlashAttention-specific lowering; source declares no flash support.
- Multi-GPU/tensor parallel.

## 15. Final implementation checklist

- [ ] Parse `BltConfig` plus all four subconfigs.
- [ ] Canonicalize `norm_eps` to `rms_norm_eps` for converted configs.
- [ ] Load byte embedding, hash embedding, local encoder, global transformer, local decoder, patcher, and LM head weights.
- [ ] Preserve actual embedding/LM-head alias metadata where present.
- [ ] Implement BLT interleaved RoPE.
- [ ] Implement RMSNorm fp32 variance.
- [ ] Implement biasless MHA/GQA self-attention with additive causal mask and optional DynamicCache.
- [ ] Implement noncausal patch cross-attention.
- [ ] Implement hash window id generation and hash embedding sum.
- [ ] Implement `patch_lengths -> patch_ids`.
- [ ] Implement patch cross-attention mask builder.
- [ ] Implement segment/scatter amax patch reduction.
- [ ] Implement local encoder forward with supplied `patch_lengths`.
- [ ] Implement optional global input projection.
- [ ] Implement global patch transformer.
- [ ] Implement local decoder and LM head with `logits_to_keep`.
- [ ] Add single-layer and per-stage parity tests.
- [ ] Add full prefill logits parity with supplied `patch_lengths`.
- [ ] Add 7B-shape global projection coverage.
- [ ] Add entropy patcher parity after core graph parity.
- [ ] Benchmark hash, patch-reduce, cross-attention, prefill, and optional patcher stages.
