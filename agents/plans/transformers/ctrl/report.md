# CTRL Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `X:/H/transformers`.

Model id: primary target [`Salesforce/ctrl`](https://huggingface.co/Salesforce/ctrl), with debug/community configs from [`sshleifer/tiny-ctrl`](https://huggingface.co/sshleifer/tiny-ctrl), [`hf-tiny-model-private/tiny-random-CTRLLMHeadModel`](https://huggingface.co/hf-tiny-model-private/tiny-random-CTRLLMHeadModel), [`hf-tiny-model-private/tiny-random-CTRLModel`](https://huggingface.co/hf-tiny-model-private/tiny-random-CTRLModel), and `prajjwal1/ctrl_discovery_*` repos.

Config source: Hub `config.json` files plus native `CTRLConfig` defaults. `Salesforce/ctrl` omits `use_cache`, `pad_token_id`, `bos_token_id`, `eos_token_id`, and `tie_word_embeddings`; native defaults are `use_cache=True`, special token IDs `None`, and tied embeddings enabled.

Source files inspected:

- [`configuration_ctrl.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/ctrl/configuration_ctrl.py)
- [`modeling_ctrl.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/ctrl/modeling_ctrl.py)
- [`tokenization_ctrl.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/ctrl/tokenization_ctrl.py)
- `tests/models/ctrl/test_modeling_ctrl.py` for tested shapes and generation expectations.

Any missing files or assumptions: no fast tokenizer exists for CTRL in the inspected source. No remote-code source is required for the native `model_type="ctrl"` path. This report targets `CTRLLMHeadModel` causal text generation first; `CTRLModel` feature extraction and `CTRLForSequenceClassification` are optional/deferred heads.

## 2. High-level architecture

CTRL is a text-only causal decoder stack with learned token embeddings, fixed sinusoidal absolute position encodings, repeated pre-LayerNorm decoder blocks, and a tied LM head.

```text
text/BPE + required first control-code convention -> token ids
  -> token embedding * sqrt(hidden)
  -> fixed sinusoidal position table lookup
  -> optional token_type embedding from the same token embedding table
  -> N causal decoder blocks
  -> final LayerNorm
  -> LM head logits / generation sampling
```

Primary runtime target: causal LM prefill and decode. The control-code behavior is tokenizer/prompt ABI, not a neural branch: source examples require the first token to be one of the tokenizer control-code IDs, but the model forward path does not validate or special-case those IDs.

## 3. Important config dimensions

| Field | Salesforce/ctrl config | Native default | Runtime significance |
|---|---:|---:|---|
| `vocab_size` | 246534 | 246534 | token embedding and LM head width |
| `n_positions` | 50000 | 256 | fixed position table length; hard context admission limit |
| `n_embd` | 1280 | 1280 | hidden size |
| `n_head` | 16 | 16 | MHA heads |
| `head_dim` | 80 | inferred `n_embd // n_head` | reshape requires `n_embd == n_head * head_dim` |
| `n_layer` | 48 | 48 | decoder blocks |
| `dff` | 8192 | 8192 | FFN intermediate |
| `layer_norm_epsilon` | 1e-6 | 1e-6 | all LayerNorms |
| `embd_pdrop` | 0.1 | 0.1 | disabled in eval |
| `resid_pdrop` | 0.1 | 0.1 | residual dropout, disabled in eval |
| `use_cache` | omitted | True | native default enables DynamicCache |
| `tie_word_embeddings` | omitted | True | LM head weight tied to token embedding |

Representative checkpoint sweep:

| Model id | Architecture | Layers | Hidden | Heads | FFN | Positions | Vocab | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `Salesforce/ctrl` | absent/native CTRL | 48 | 1280 | 16 | 8192 | 50000 | 246534 | production checkpoint; `n_ctx=512` present but native source ignores it |
| `prajjwal1/ctrl_discovery_1` | `CTRLLMHeadModel` | 48 | 1280 | 16 | 8192 | 50000 | 246535 | community fine-tune; one larger vocab slot |
| `prajjwal1/ctrl_discovery_flipped_4` | `CTRLLMHeadModel` | 48 | 1280 | 16 | 8192 | 50000 | 246535 | same topology as discovery fine-tunes |
| `sshleifer/tiny-ctrl` | `CTRLLMHeadModel` | 2 | 16 | 2 | 2 | 50000 | 246534 | tiny/debug; very small FFN but long position table |
| `hf-tiny-model-private/tiny-random-CTRLLMHeadModel` | `CTRLLMHeadModel` | 5 | 32 | 4 | 8192 | 512 | 246534 | random tiny; includes `is_decoder=true`, which native source does not read |

## 3a. Family variation traps

- No GQA/MQA: source has independent dense `Wq`, `Wk`, `Wv` projections all shaped `hidden -> hidden`, with `num_key_value_heads` absent.
- `attn_pdrop` appears in historical configs but native `modeling_ctrl.py` does not apply attention dropout.
- `n_ctx`, `summary_*`, and `is_decoder` appear in some configs but are not read by the inspected native forward path.
- `n_embd` must be divisible by `n_head`; source computes `depth = int(n_embd / n_head)` and later reshapes to `[B, T, n_head, depth]`.
- Position encoding is a fixed table of length `n_positions`; no RoPE, ALiBi, learned position embedding, dynamic extrapolation, or sliding-window behavior.
- Token type embeddings reuse the token embedding table and are added only when `token_type_ids` are explicitly passed. Generation preparation removes `token_type_ids`.
- `pad_token_id=None` is common. Classification with batch size > 1 rejects if no pad token is configured; causal LM can still batch with attention masks.
- LM head has a bias module, but tied weight aliasing with `transformer.w.weight` must be preserved.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(input_ids) -> [B,T,H]`.
- Optional second embedding lookup for `token_type_ids`, using the same token table.
- Fixed position table gather by `position_ids`: `[T]` or `[1,T] -> [1,T,H]`.
- Reshape/permute for heads: `[B,T,H] -> [B,n_head,T,head_dim]`; attention output `[B,n_head,T,D] -> [B,T,H]`.
- Causal mask construction or equivalent backend causal flag for rectangular decode.
- Attention-mask broadcast from `[B,S] -> [B,1,1,S]` and additive mask conversion to dtype minimum.
- Last-token/sliced logits via `logits_to_keep` for generation efficiency.

Neural network primitives:

- Biased dense projections: per block `Wq`, `Wk`, `Wv`, output dense, FFN `Linear(H -> dff)`, ReLU, `Linear(dff -> H)`.
- LayerNorm with epsilon 1e-6: two per block plus final norm.
- Residual adds around attention and FFN.
- LM head `Linear(H -> vocab_size)` with bias and tied weight alias.

Attention primitives:

- Dense causal self-attention MHA only.
- Matmul scores `q @ k^T / sqrt(head_dim)`.
- Add causal mask as `-1e4` in source; add padding mask as `torch.finfo(dtype).min`.
- Softmax over key dimension and attention-value matmul.
- KV cache update per layer through `Cache.update(k, v, layer_idx)`.

Position/custom math:

- Sinusoidal table generation with `angle = pos / 10000 ** (2 * floor(i/2) / H)`, then concatenate all sine even channels followed by all cosine odd channels. This is not the interleaved Transformer layout.

Generation/cache ops:

- DynamicCache creation when `use_cache=True` and no cache is supplied.
- `past_length = cache.get_seq_length()`.
- Position IDs start at `past_length`.
- Beam/cache reorder should use generic Transformers cache semantics; no model-specific reorder override exists.

Preprocessing-coupled ops:

- Slow BPE tokenizer with regex split on non-whitespace spans and fixed `CONTROL_CODES` map.
- First-token control code is an end-to-end prompt admission requirement for parity, not a graph op.

Optional/deferred heads:

- `CTRLForSequenceClassification`: classifier `Linear(H -> num_labels, bias=False)` over every token, then gathers last non-pad token or final token.

## 5. Layer/block breakdown

Decoder block, repeated `n_layer` times:

```text
x0: [B,T,H]
normed = LayerNorm(x0, eps=1e-6)
q = Linear(H -> H, bias=True)(normed) -> [B,heads,T,D]
k = Linear(H -> H, bias=True)(normed) -> [B,heads,T,D]
v = Linear(H -> H, bias=True)(normed) -> [B,heads,T,D]
k,v = cache.update(k,v,layer_idx) when a Cache object is present
attn = softmax((q @ k^T) / sqrt(D) + causal_mask + attention_mask)
attn_out = Linear(H -> H, bias=True)(attn @ v)
x1 = x0 + attn_out
ffn_in = LayerNorm(x1, eps=1e-6)
ffn = Linear(H -> dff, bias=True) -> ReLU -> Linear(dff -> H, bias=True)
x2 = x1 + ffn
```

Embedding and output:

```text
input_embeds = token_embedding(input_ids) * sqrt(H)
token_type_embeds = token_embedding(token_type_ids) * sqrt(H), only if provided
pos_embeds = fixed_pos_encoding[position_ids]
hidden = Dropout(input_embeds + pos_embeds + token_type_embeds)
hidden = decoder_blocks(hidden)
hidden = final LayerNorm(hidden)
logits = lm_head(hidden[:, slice_indices, :])
```

## 6. Attention requirements

CTRL requires causal self-attention with dense MHA:

- Causal/noncausal: causal only for LM path.
- Self/cross: self-attention only; no cross-attention.
- Heads: MHA, `n_head=16`, `head_dim=80` for `Salesforce/ctrl`.
- Query/key/value widths: all `hidden_size`; no separate value width.
- Prefill: `q,k,v` all length `T`, attention scores `[B,heads,T,T]`.
- Decode: query length is current token count, usually 1; key/value length is `past_length + T`.
- Masking: source combines a triangular mask with optional additive padding mask. The triangular mask is sliced as `mask[ns - nd : ns, :ns]`, which makes decode rows align with absolute key length.
- Packed/varlen: no varlen or packed sequence metadata in source.
- Local/sliding: none.
- Positional interaction: fixed absolute sinusoidal embeddings are added before projection; cached K/V are stored after position information is already mixed into hidden states and after K/V projection.
- Optimized attention: no SDPA/FlashAttention dispatch in source. DinoML can use a fused causal attention backend only if it preserves source mask values/order closely enough for parity.

Cache ABI:

```text
per layer key:   [B, n_head, S_cached, head_dim]
per layer value: [B, n_head, S_cached, head_dim]
new k/v before update: [B, n_head, T_new, head_dim]
after update: [B, n_head, S_cached + T_new, head_dim]
```

The source updates cache whenever a `Cache` object is passed to the attention layer. `use_cache=False` prevents automatic DynamicCache creation, but a non-null `past_key_values` still participates in updates.

## 7. Position encoding and custom math

CTRL's position table is fixed and deterministic:

```python
def ctrl_positional_encoding(n_positions, hidden):
    pos = arange(n_positions)[:, None]
    i = arange(hidden)[None, :]
    angle = pos / pow(10000, (2 * (i // 2)) / hidden)
    return concat([sin(angle[:, 0::2]), cos(angle[:, 1::2])], dim=-1)
```

Notable parity detail: the sine half and cosine half are concatenated, not interleaved. The table can be precomputed at compile/load time for a fixed `n_positions` and `hidden`. Runtime only needs gather by `position_ids`, whose default is `past_length..past_length+T-1`. DinoML should reject or route to fallback when `past_length + T > n_positions`.

## 8. Preprocessing and input packing

Tokenizer coupling:

- Source tokenizer is slow BPE with `vocab.json` and `merges.txt`.
- Regex tokenization splits on `\S+\n?`.
- BPE adds `</w>` to the last character and emits `@@` continuation markers, then detokenization removes `@@ `.
- `CONTROL_CODES` maps human prompt categories such as `Legal`, `Wikipedia`, `Opinion`, and `News` to token IDs. Source examples assert the first token belongs to this map.
- Tokenizer defaults include all-zero token type IDs if token type IDs are requested, but `CTRLLMHeadModel.prepare_inputs_for_generation` removes token type IDs before forward.

GPU graph inputs for first LM target:

- Required: `input_ids [B,T]`.
- Optional: `attention_mask [B,S]`, where `S` must cover cached plus current sequence for cached decode.
- Optional: `position_ids [B or 1,T]`; if omitted, generated from cache length.
- Defer `inputs_embeds` for first integration unless an embedding-stitch use case appears; it bypasses input ID validation and tokenizer coupling.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V projections to packed QKV GEMM

Source pattern: three biased linear projections from the same normalized hidden state.

Replacement: one packed `Linear(H -> 3H)` followed by split `[q,k,v]`.

Preconditions:

- Same input tensor for all three projections.
- All projections have bias.
- Weight pack order must be `[Wq, Wk, Wv]`, bias pack order `[bq, bk, bv]`.
- Preserve output split before head reshape.

Parity test sketch: compare q/k/v tensors after projection and reshape for random hidden input against native separate projections.

### Rewrite: source attention to fused causal attention

Source pattern: score matmul, divide by `sqrt(D)`, add sliced triangular mask and optional additive padding mask, softmax, value matmul.

Replacement: fused causal MHA prefill/decode backend.

Preconditions:

- Dense MHA, no output attentions requested.
- No custom `position_ids` beyond fixed absolute embedding already applied.
- Padding mask either absent or expressible as additive key mask.
- Backend supports rectangular decode causal alignment equivalent to `mask[ns-nd:ns, :ns]`.

Failure cases: `output_attentions=True`, unusual attention mask rank/value, or parity-sensitive tests requiring exact `-1e4` causal mask behavior.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])` controlled by `logits_to_keep`.

Replacement: in decode, run LM head only for the last token or requested index tensor.

Preconditions:

- No loss computation.
- Generation caller requests only final logits or passes a static/supported logits index tensor.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + QKV projection packing: appears twice per block path and feeds attention.
- Fused causal attention with KV cache: main prefill/decode bottleneck for 48-layer, 50k-position CTRL.
- FFN `Linear -> ReLU -> Linear`: large `H=1280, dff=8192`; GEMM throughput dominates.
- Last-token-only LM head: vocab is 246k, so avoiding full-sequence logits is important.

Medium priority:

- Embedding scale + position add + optional token-type add fusion.
- Residual add around attention/FFN with adjacent dropout elided in eval.
- Cache append/update kernels or static KV buffer writes for decode.

Lower priority:

- Control-code validation in runtime input ABI.
- Classification pooling/gather path.
- Output attentions materialization.

## 11. Runtime staging plan

Stage 1: parse CTRL config and load weights for `CTRLModel`/`CTRLLMHeadModel`, including tied `lm_head.weight -> transformer.w.weight` identity and LM head bias.

Stage 2: implement one-block and full-prefill parity with explicit dense attention and fixed position table. Stub generation controller beyond greedy/single-step.

Stage 3: add `logits_to_keep` and last-token LM head path for prefill/decode.

Stage 4: add DynamicCache-compatible per-layer KV cache ABI and single-token decode parity.

Stage 5: replace explicit attention with fused causal attention under guards; keep dense fallback for `output_attentions=True` and odd masks.

Stage 6: add tokenizer/control-code admission checks outside the compiled graph for end-to-end generation parity.

Stage 7: optionally add sequence classification head with pad-aware last-token gather.

## 12. Parity and validation plan

- Position table parity: compare generated table for small `n_positions,H` and production `H=1280`; include the non-interleaved sine/cosine layout.
- Projection packing parity: random hidden input through separate native Q/K/V vs packed rewrite.
- Single-block fp32 parity: no cache, no padding mask, then with padding mask.
- Full prefill parity: `Salesforce/ctrl` config dimensions with synthetic or loaded weights, verify logits for `logits_to_keep=0` and `1`.
- Decode parity: prefill `N` tokens, decode one token with cache, compare logits and cache length/shapes.
- Control-code E2E smoke: tokenize prompt beginning with `Legal` or `Wikipedia`; verify first token ID is in the known control-code map before model call.
- Classification optional: batch size > 1 with `pad_token_id=None` must reject; with pad token set, gather rightmost non-pad.

Recommended tolerances: fp32 dense path `rtol=1e-4, atol=1e-4` for logits after full model; fp16/bf16 optimized attention should start with looser logits tolerances and layerwise diagnostics because softmax mask constants differ from many fused kernels.

## 13. Performance probes

- Prefill tokens/sec sweep over `T = 128, 512, 2048, 8192` and batch sizes `1, 2, 4`.
- Decode tokens/sec with KV cache over increasing cached lengths, especially around 512, 2048, and long-context positions.
- KV memory usage: `2 * n_layer * B * n_head * S * head_dim * dtype_size`.
- LM head cost with full logits vs `logits_to_keep=1`.
- Dense attention fallback vs fused causal attention, with and without padding mask.
- FFN GEMM profile for `1280 -> 8192 -> 1280`.
- Tokenizer throughput separately from GPU graph, because BPE and control-code prompt handling are CPU/data-pipeline work.

## 14. Skip/defer list

- Training losses and dropout behavior.
- Beam search, sampling policies, repetition penalties, and other generation-controller features beyond the model ABI.
- `output_attentions=True` optimized path; use dense fallback first.
- Sequence classification head unless a user target requires it.
- `inputs_embeds` entrypoint and custom token type IDs in generation.
- Attention dropout from `attn_pdrop`; native source ignores it for this commit.
- Quantized/packed weight formats; no CTRL-specific packed weights are implemented in native source.

## 15. Final implementation checklist

- [ ] Parse `CTRLConfig`, including effective defaults for omitted `use_cache` and special token IDs.
- [ ] Reject invalid `n_embd % n_head != 0`.
- [ ] Load token embedding, tied LM head weight alias, LM head bias, per-layer Q/K/V/output/FFN weights, and LayerNorm parameters.
- [ ] Precompute CTRL fixed sinusoidal position table with concatenated sine/cosine layout.
- [ ] Implement embedding scale, position gather, optional token-type embedding add.
- [ ] Implement pre-LN decoder block with dense causal MHA and FFN ReLU.
- [ ] Implement additive padding mask and causal mask parity.
- [ ] Add `logits_to_keep` support for last-token logits.
- [ ] Implement per-layer KV cache update and decode position IDs from cache length.
- [ ] Add control-code prompt admission in tokenizer/runtime wrapper.
- [ ] Add parity tests for position table, one block, prefill logits, decode logits, cache shape, and last-token logits.
- [ ] Benchmark prefill, decode, LM head, attention backend, and KV memory.
