# Longformer Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: allenai/longformer-base-4096, allenai/longformer-large-4096, allenai/longformer-large-4096-finetuned-triviaqa, hf-internal-testing/tiny-random-longformer, patrickvonplaten/longformer-random-tiny
Config source: Hugging Face config.json snapshots under agents/plans/transformers/longformer/_sources/
Source files inspected:
  transformers/src/transformers/models/longformer/configuration_longformer.py
  transformers/src/transformers/models/longformer/modeling_longformer.py
  transformers/src/transformers/models/longformer/__init__.py
  transformers/src/transformers/modeling_utils.py, for get_extended_attention_mask semantics
Any missing files or assumptions:
  No remote-code files are required for the in-library Longformer family.
  allenai/* tokenizer sidecar JSON files were not present at the raw URLs checked; the package aliases LongformerTokenizer to RobertaTokenizer.
  Some fine-tuned AllenAI configs are gated/unauthorized from this environment and are not used as evidence.
```

Primary source URLs at the pinned commit:

- `configuration_longformer.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/longformer/configuration_longformer.py
- `modeling_longformer.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/longformer/modeling_longformer.py
- `__init__.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/longformer/__init__.py

## 2. High-level architecture

Longformer is a text-only encoder derived from RoBERTa/BERT blocks, with absolute token/position/type embeddings, repeated encoder blocks, and task heads. Its distinctive runtime requirement is noncausal self-attention with local sliding-window attention plus optional global attention tokens. The audited source explicitly says this implementation lacks autoregressive and dilated attention support, so prefill/decode/KV-cache language is not the primary contract.

```text
tokenization/data pipeline -> input_ids/attention_mask/global_attention_mask/token_type_ids
-> embeddings + automatic pad-to-window
-> N encoder layers with local+global Longformer self-attention and FFN
-> unpad sequence output
-> optional pooler or task head
```

Independently stageable pieces:

- CPU/data pipeline: RoBERTa-style tokenization, special-token layout, attention masks, optional task-specific global mask construction.
- GPU/runtime encoder: embeddings, pad-to-window behavior, local/global attention, FFN, unpadding.
- Heads: masked LM, sequence classification, QA span logits, token classification, multiple choice.
- Optional diagnostics: attentions/global attentions outputs are useful for parity but not needed for first throughput target.

## 3. Important config dimensions

Source defaults from `LongformerConfig` differ from common AllenAI checkpoints. In particular, source default `max_position_embeddings=512`, `type_vocab_size=2`, `layer_norm_eps=1e-12`, and `attention_window=512`; AllenAI long-context configs use 4098 positions, `type_vocab_size=1`, and `layer_norm_eps=1e-5`.

| Field | Source default | Runtime significance |
|---|---:|---|
| `vocab_size` | 30522 | Word embedding and LM decoder width. |
| `hidden_size` | 768 | Encoder width. Must divide by `num_attention_heads`. |
| `num_hidden_layers` | 12 | Encoder block count. |
| `num_attention_heads` | 12 | MHA head count. No GQA/MQA. |
| `head_dim` | 64 inferred | `hidden_size / num_attention_heads`. |
| `intermediate_size` | 3072 | FFN expansion. |
| `hidden_act` | `gelu` | FFN activation and LM head activation. |
| `attention_window` | 512 | Full local window size; one-sided size is `attention_window // 2`. Must be even and positive. |
| `max_position_embeddings` | 512 | Absolute position embedding table. |
| `type_vocab_size` | 2 | Token type embedding table. |
| `layer_norm_eps` | `1e-12` | LayerNorm epsilon. |
| `tie_word_embeddings` | `true` | MLM decoder weight ties to input embedding when enabled. |
| `onnx_export` | `false` | Switches chunk construction away from `as_strided`; runtime should reject or separately stage this path. |
| cache support | absent in source | No `past_key_values` or generation cache path. |

Representative checkpoint sweep, from config snapshots:

| Model/config | Architecture | Hidden | Layers | Heads | Head dim | FFN | Vocab | Max pos | Attention window | Type vocab | LN eps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `patrickvonplaten/longformer-random-tiny` | `LongformerModel` | 8 | 2 | 2 | 4 | 16 | 100 | 64 | `[4,4]` | 2 | `1e-12` |
| `hf-internal-testing/tiny-random-longformer` | `LongformerModel` | 32 | 5 | 4 | 8 | 37 | 1000 | 512 | `[4] * 5` | 16 | `1e-12` |
| `allenai/longformer-base-4096` | not specified | 768 | 12 | 12 | 64 | 3072 | 50265 | 4098 | `[512] * 12` | 1 | `1e-5` |
| `allenai/longformer-large-4096` | not specified | 1024 | 24 | 16 | 64 | 4096 | 50265 | 4098 | `[512] * 24` | 1 | `1e-5` |
| `allenai/longformer-large-4096-finetuned-triviaqa` | `LongformerForQuestionAnswering` | 1024 | 24 | 16 | 64 | 4096 | 50265 | 4098 | `[512] * 24` | 1 | `1e-5` |

Historical/config-only fields observed in configs but not read by the current native source include `attention_mode`, `ignore_attention_mask`, `position_embedding_type`, and tiny config `use_cache`. DinoML should not infer runtime behavior from those fields for the in-library implementation.

## 3a. Family variation traps

- `attention_window` may be an int or per-layer list. The model constructor mutates an int into a list of length `num_hidden_layers`.
- Each layer requires an even positive attention window. The local attention kernels require padded sequence length to be a multiple of that layer's full `attention_window`.
- `_pad_to_window_size` pads to `max(config.attention_window)`, so heterogeneous attention-window lists need layout guards that ensure every layer's window divides the padded sequence length.
- `max_position_embeddings=4098` in AllenAI checkpoints is not a typo: position IDs start at `pad_token_id + 1`, and padding uses `pad_token_id`.
- `type_vocab_size` is 1 for AllenAI RoBERTa-derived checkpoints. Supplying nonzero token type IDs would be out-of-range for those weights.
- Global attention uses separate `query_global`, `key_global`, and `value_global` parameters. These are not aliases of local Q/K/V.
- Sequence classification silently creates `global_attention_mask[:, 0] = 1` if no global mask is passed.
- QA and multiple choice can auto-create global masks from separator-token positions and assert a three-SEP QA-style layout. That behavior depends on `input_ids`, not only masks.
- The local attention code uses `as_strided`, advanced indexing/scatter-like updates, padding, diagonalization, and `einsum`; a naive dense attention lowering will be correct but has the wrong asymptotic memory.
- `output_attentions=True` changes output tensor structures and carries dynamic `max_num_global_attn_indices`.
- `onnx_export=True` changes chunk construction to a slow loop with `torch.empty` writes; first DinoML support should reject or separately test this path.
- No causal mask, no RoPE/ALiBi, no KV cache, no GQA/MQA, no generation controller.
- Layout translation should be conservative. Source tensors are `[batch, seq, hidden]`, attention internals use `[batch, seq, heads, head_dim]`, and local-attention window dimensions are semantic, not channel dimensions.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for `input_ids`, `position_ids`, `token_type_ids`.
- `zeros_like`, `ones`, `arange`, `cumsum`, `ne`, `long/int/bool` casts for default position IDs and automatic global masks.
- `pad` along sequence dimension for ids, masks, token types, positions, and optional `inputs_embeds`.
- `cat`, `view/reshape`, `transpose`, `contiguous`, `narrow`, `squeeze`, `split`, `stack`, `expand_as`.
- `as_strided` or an equivalent overlapping-window view/materialization for sliding chunks.
- Advanced indexing and indexed writes for global attention selection/scatter.
- Unpadding slice `hidden_states[:, :seq_len - padding_len]`.

Neural network primitives:

- Dense Linear with bias for all attention projections, attention output, FFN, pooler, LM head, and task heads.
- LayerNorm over hidden dimension with per-config epsilon.
- Dropout can be disabled for inference but must remain graph-visible for training parity if ever needed.
- GELU for FFN and LM head; tanh for pooler and classification head.
- Residual add + LayerNorm after attention output and FFN output.
- Softmax with fp32 accumulation and cast back to attention dtype.
- Masked fill to dtype min and zeroing masked probability rows.

Attention primitives:

- Noncausal MHA with local sliding-window scores of shape `[B, S, H, 2w+1]`.
- Optional global-key scores appended to local score width, shape `[B, S, H, Gmax]`.
- Optional global-query dense attention from global tokens to every token, shape `[B, H, Gmax, S]`.
- Per-batch variable number of global tokens with zero-padded global slots and masking.
- Local output from sliding `attn_probs @ V`; global contributions add a dense `attn_probs_global @ V_global_only`.
- Global-token output overwrite from separate global Q/K/V dense attention.

Position/custom math:

- Absolute learned positions only.
- Position IDs for `input_ids`: non-pad tokens get cumulative positions beginning at `pad_token_id + 1`, pad tokens remain at `pad_token_id`.
- No RoPE, relative bias, or ALiBi.

Generation/cache ops:

- None required. Source forwards do not accept `past_key_values`, return cache objects, or implement decode.

Preprocessing-coupled ops:

- RoBERTa tokenizer contract, `<s>`/`</s>`-style special tokens, pad token id 1, sep/eos token id 2 in inspected configs.
- QA/multiple-choice automatic global-mask generation uses separator token positions and `torch.arange` comparisons.

Parameter aliasing:

- `LongformerForMaskedLM` declares tied weights: `lm_head.decoder.weight -> longformer.embeddings.word_embeddings.weight` and `lm_head.decoder.bias -> lm_head.bias` when `tie_word_embeddings=True`.

## 5. Layer/block breakdown

Embeddings:

```text
position_ids = cumsum(input_ids != pad_id) * mask + pad_id, unless supplied
token_type_ids = zeros([B,S]) if omitted
x = word_embedding(input_ids) + position_embedding(position_ids) + token_type_embedding(token_type_ids)
x = LayerNorm(x)
x = Dropout(x)
```

Encoder block, repeated `num_hidden_layers` times:

```text
q = Linear(hidden -> hidden, bias)(x) / sqrt(head_dim)
k = Linear(hidden -> hidden, bias)(x)
v = Linear(hidden -> hidden, bias)(x)
local_scores = sliding_chunks_qk(q, k, one_sided_window)
local_scores += diagonal padding/local mask
if any global token:
    global_key_scores = q @ selected_global_k
    scores = concat(global_key_scores, local_scores, dim=-1)
else:
    scores = local_scores
probs = softmax(scores, dim=-1, dtype=float32).cast(scores.dtype)
probs[masked_tokens] = 0
attn = sliding_chunks_pv(probs_without_global, v) + optional global-value contribution
if any global token:
    global_output = dense_global_attention(query_global(selected tokens), key_global(all), value_global(all))
    overwrite output at global token positions
x = LayerNorm(Linear(attn) + residual)
ff = GELU(Linear(hidden -> intermediate, bias)(x))
x = LayerNorm(Linear(intermediate -> hidden, bias)(ff) + residual)
```

Base/heads:

- `LongformerModel`: optional pooler takes `hidden_states[:, 0] -> Linear(hidden, hidden) -> tanh`.
- `LongformerForMaskedLM`: encoder without pooler, then `Linear(hidden, hidden) -> GELU -> LayerNorm -> Linear(hidden, vocab)`.
- `LongformerForSequenceClassification`: encoder without pooler, first token, dropout, `Linear(hidden, hidden)`, tanh, dropout, `Linear(hidden, num_labels)`.
- `LongformerForQuestionAnswering`: encoder without pooler, `Linear(hidden, num_labels)` then split into start/end logits; normal configs use `num_labels=2`.
- `LongformerForTokenClassification`: dropout then `Linear(hidden, num_labels)` per token.
- `LongformerForMultipleChoice`: flatten `[B,C,S] -> [B*C,S]`, base model with pooler, dropout, `Linear(hidden,1)`, reshape to `[B,C]`.

## 6. Attention requirements

Longformer attention is encoder-style, bidirectional, noncausal self-attention with local sliding windows plus optional global tokens.

| Requirement | Longformer behavior |
|---|---|
| Causal | No. |
| Cross-attention | No. |
| MHA/MQA/GQA | Standard MHA only. `num_kv_heads == num_attention_heads`. |
| Head count | Base 12, large 16; tiny configs vary. |
| Head dim | 64 for AllenAI base/large; inferred as `hidden_size / num_attention_heads`. |
| Masking | Encodes masked/local/global in one extended mask: negative means masked, zero means local, positive means global. |
| Sliding window | Required. One-sided window is `attention_window // 2`; output local score width is `attention_window + 1`. |
| Global attention | Optional but task heads may create it. Tokens with global attention attend densely to all unmasked tokens; all tokens attend to global tokens. |
| Packed/varlen | No FlashAttention-style `cu_seqlens`; batch variation is via padding and masks. |
| Cache | None. Config `use_cache` in a tiny checkpoint is ignored by native Longformer source. |
| SDPA/FlashAttention compatibility | Source does not dispatch SDPA/FlashAttention. Dense SDPA can only be a fallback for very short sequences or parity debugging, not the production target. |

Attention-mask semantics are easy to get wrong. `LongformerModel.forward` first merges masks in raw integer space:

```text
attention_mask: 0 masked, 1 local
global_attention_mask: 0 local, 1 global
merged = attention_mask * (global_attention_mask + 1)
```

Then `get_extended_attention_mask` maps values using `(1 - mask) * torch.finfo(dtype).min`. Therefore:

```text
raw 0 -> dtype min -> masked
raw 1 -> 0 -> local
raw 2 -> -dtype min -> positive -> global
```

The local attention code then uses `attention_mask < 0` for masked positions and `attention_mask > 0` for global positions. The comments mention `-10000/0/+10000`, but at this commit the effective values are dtype extrema.

KV cache: not applicable. Cached encoder outputs for downstream reuse are a product-level optimization only, not a source-defined cache ABI.

## 7. Position encoding and custom math

Longformer uses learned absolute position embeddings. There is no RoPE, ALiBi, relative-position bias, or convolutional position embedding.

Custom position-id behavior:

```python
def longformer_position_ids(input_ids, padding_idx):
    mask = (input_ids != padding_idx).int()
    incremental = cumsum(mask, dim=1).type_as(mask) * mask
    return incremental.long() + padding_idx
```

Inputs passed as `inputs_embeds` cannot infer padding, so source generates sequential positions:

```python
position_ids = arange(pad_id + 1, seq_len + pad_id + 1)
position_ids = position_ids[None, :].expand(batch, seq_len)
```

Task-specific global mask helper:

```python
question_end = first_sep_index(input_ids, sep_token_id)
if before_sep_token:
    global_mask = arange(seq_len)[None, :] < question_end[:, None]
else:
    global_mask = (arange(seq_len)[None, :] > question_end[:, None] + 1) & (arange(seq_len)[None, :] < seq_len)
```

The helper asserts three separator tokens per QA sample. DinoML can implement it in preprocessing first and keep the GPU graph accepting an explicit `global_attention_mask`.

## 8. Preprocessing and input packing

Runtime tensors:

- `input_ids`: `[B, S]` int token IDs, or for multiple choice `[B, C, S]` before flattening.
- `attention_mask`: `[B, S]`, optional. Source defaults to ones.
- `global_attention_mask`: `[B, S]`, optional. Required for exact control of global tokens.
- `token_type_ids`: `[B, S]`, optional. Source defaults to zeros.
- `position_ids`: `[B, S]`, optional. Source creates Longformer-specific positions when omitted.
- `inputs_embeds`: `[B, S, hidden]`, mutually exclusive with `input_ids`.

CPU/data-pipeline candidates:

- Tokenization and special-token insertion should stay outside DinoML runtime initially.
- QA and multiple-choice automatic global-mask generation should also start outside the runtime. It includes `nonzero`, separator count assertions, `arange`, comparisons, and per-choice stacking.

GPU/runtime-coupled packing:

- `_pad_to_window_size` is part of model parity because it changes sequence length before embeddings/attention and then unpads outputs.
- Padding values are token IDs `pad_token_id`, position IDs `pad_token_id`, attention mask 0, token type 0, and embedded padding produced by running embeddings on pad token IDs when `inputs_embeds` is supplied.
- For first integration, prefer static/bucketed sequence lengths already divisible by the attention window. Keep automatic padding as a correctness fallback or a graph rewrite pass.

Special-token/task behavior:

- Sequence classification: if no global mask, source sets token 0 global.
- QA: if no global mask and `input_ids` exists, source sets question tokens before first separator global.
- Multiple choice: if no global mask, source sets tokens after the first separator plus one global for each choice, then flattens choices into batch.
- Masked LM/token classification/base model do not auto-enable global tokens.

## 9. Graph rewrite / lowering opportunities

### Rewrite: padded static bucket for window divisibility

Source pattern:

```text
runtime seq_len -> pad ids/masks/embeds to multiple of max(attention_window) -> encoder -> unpad
```

Replacement:

```text
compile/profile buckets where S_bucket % all_required_windows == 0
runtime validates S <= S_bucket and explicit padding is folded into input staging
```

Preconditions:

- All per-layer attention windows are known.
- Chosen bucket length is divisible by each layer's full `attention_window`.
- Caller-visible outputs are sliced back to original `S`.

Failure cases:

- Heterogeneous attention windows whose divisibility is not satisfied by `max(attention_window)` alone.
- `inputs_embeds` path, unless pad-token embeddings are precomputed identically.

Parity test sketch:

- Compare source and DinoML for `S` values just below, equal to, and just above a window multiple; include `input_ids` and `inputs_embeds` paths.

### Rewrite: sliding chunks attention -> banded attention kernel

Source pattern:

```text
as_strided overlapping chunks -> einsum QK -> pad/diagonalize/copy -> mask invalid edge locations
softmax -> pad/diagonalize probs -> as_strided V chunks -> einsum PV
```

Replacement:

```text
custom local-attention kernel over [B,H,S,D] producing [B,S,H,2w+1] scores and [B,S,H,D] context
```

Preconditions:

- Noncausal symmetric window with one-sided size `w`.
- `S % (2w) == 0` for the source-equivalent chunking path.
- Dense contiguous `[B,S,H,D]` or a validated equivalent layout.
- Masked edge positions match `_mask_invalid_locations`.

Shape equations:

```text
local_score_width = 2w + 1 = attention_window + 1
chunks_count = S / w - 1
```

Failure cases:

- `onnx_export=True` loop path.
- Dilated/autoregressive attention, which source explicitly does not implement.
- Requesting attention output tensors before the optimized kernel can materialize source-shaped `attentions`.

Parity test sketch:

- Random Q/K/V for tiny `S=8,w=2` and production-like `S=512,w=256`; compare scores after invalid-location mask and final context.

### Rewrite: global attention split into dense subproblems

Source pattern:

```text
select global token K/V into [B,Gmax,H,D]
all tokens attend to global keys
global tokens attend densely to all tokens using separate global Q/K/V projections
scatter global outputs back into sequence
```

Replacement:

```text
GatherGlobal -> BMM/GEMM for query-to-global keys
GlobalDenseAttention -> ScatterGlobalOutput
```

Preconditions:

- `Gmax` is small or bucketed.
- Global index gather/scatter order matches source nonzero order.
- Padded global slots are masked to dtype min before softmax.

Failure cases:

- Dynamic `Gmax` with no shape-buffer/output-shape support.
- All-global sequences may prefer dense attention, but still must use separate global projections for global-token output.

Parity test sketch:

- Batch with different global counts per row, including zero global tokens and masked padding.

### Rewrite: inference dropout elimination

Preconditions:

- `model.eval()` / inference-only compile.

Replacement:

```text
Dropout(x, p) -> Identity(x)
```

Failure cases:

- Training or stochastic parity tests.

### Rewrite: QKV projection grouping

Source pattern:

```text
q = Linear(x); k = Linear(x); v = Linear(x)
q_global = Linear(x); k_global = Linear(x); v_global = Linear(x)
```

Replacement:

```text
Grouped/fused GEMM for local QKV, separate grouped/fused GEMM for global QKV
```

Preconditions:

- Preserve separate parameter tensors or apply an explicit weight packing transform.
- Global Q/K/V outputs are only needed when any global token exists, but source computes global projections only inside the global branch.

Failure cases:

- Weight alias assumptions; local and global projections are distinct.

### Layout/axis guard: preserve sequence-major semantics

No NHWC-style translation is useful for this text encoder. Candidate internal layouts are `[B,H,S,D]` for attention kernels and `[B,S,hidden]` for GEMM-friendly projections. Axis-sensitive ops that need guards:

- Position/mask `cumsum(dim=1)`, `arange(seq)`, and padding along sequence.
- Softmax over last attention-score dimension, not sequence hidden dim.
- Classification `hidden_states[:, 0, :]`.
- QA split along hidden/logit last dim.
- Multiple-choice flatten from `[B,C,S]` to `[B*C,S]`.

## 10. Kernel fusion candidates

Highest priority:

- Local sliding-window attention kernel, including invalid edge masking and local `attn_probs @ V`. This is the defining performance path and avoids dense `S x S` memory.
- Global attention gather/BMM/scatter helpers. Without these, global-token support falls back to slow advanced indexing and dynamic scatter.
- LayerNorm + residual for attention and FFN outputs. Every encoder layer has two of these and eps varies by checkpoint.
- GEMM-backed Linear projections and FFN, including optional QKV grouped projection. Base/large models are GEMM-heavy even with sparse attention.

Medium priority:

- Pad-to-window and unpad lowering as a shape/bucket transform.
- Softmax over banded/global-concatenated score rows with fp32 accumulation and masked-row zeroing.
- GELU FFN fusion: Linear -> GELU -> Linear remains a common throughput hotspot.
- First-token classification head and QA split/squeeze as simple graph patterns.

Lower priority:

- `output_attentions=True` materialization for diagnostics.
- ONNX-export chunk path.
- Training losses and dropout.
- Dense full-attention fallback for very short sequences; useful for parity only.

## 11. Runtime staging plan

Stage 1: config/weights/embedding parity.

- Parse LongformerConfig, reject ignored historical fields as behavior flags.
- Load base/tiny weights, preserve MLM tied-weight aliases.
- Run embeddings plus position-id generation parity on tiny random configs.

Stage 2: one block with local attention only.

- Require explicit sequence length divisible by `attention_window`.
- Disable global attention, dropout, and attention outputs.
- Implement or emulate sliding-window QK/softmax/PV and compare a single layer.

Stage 3: full encoder local-only.

- Run all layers for tiny then base config with no global tokens.
- Add pad-to-window and unpad parity.

Stage 4: global attention support.

- Add explicit `global_attention_mask` input.
- Implement global gather, query-to-global scores, global-token dense attention, and output overwrite.
- Validate mixed global counts per batch.

Stage 5: task heads.

- Add masked LM and QA first, then sequence/token classification and multiple choice.
- Move automatic task global-mask generation into preprocessing wrappers before compiling it into GPU graphs.

Stage 6: optimized kernels/fusions.

- Replace reference local attention with provider-backed banded attention.
- Add profiler probes for window sizes, global counts, batch sizes, and sequence buckets.

Stage 7: production integration.

- Bucket long-context inputs, cache encoder outputs only at application level when useful, and add continuous batching around encoder workloads rather than decode caches.

## 12. Parity and validation plan

- Config parsing tests: source defaults vs AllenAI config overrides, including `type_vocab_size=1`, `layer_norm_eps=1e-5`, and `max_position_embeddings=4098`.
- Position ID tests: padded and unpadded `input_ids`, all-pad rows, and `inputs_embeds` sequential positions.
- Pad-to-window tests: `S=window-1`, `S=window`, `S=window+1`; verify output unpadding.
- Sliding attention unit tests: compare local QK scores, invalid-location mask, softmax rows, and PV output against Transformers for tiny windows.
- Global attention tests: batches with global counts `[0,1,3]`, masked padding, global token at sequence edges, and all-global stress case.
- Single-layer parity: random tiny checkpoint in fp32 with local-only and global paths.
- Full-encoder parity: `patrickvonplaten/longformer-random-tiny` and `hf-internal-testing/tiny-random-longformer`.
- Head parity: MLM logits, QA start/end logits with auto and explicit global masks, sequence classification first-token logits, token classification, multiple choice flatten/reshape.
- Dtype tolerances: fp32 `rtol=1e-4, atol=1e-4`; fp16/bf16 initially `rtol=5e-3, atol=5e-3`, tighten per kernel after softmax/mask parity is proven.
- Regression tests for rejected/unsupported paths: `onnx_export=True`, `attention_window` odd/nonpositive, hidden not divisible by heads, `use_cache` requested as behavior.

## 13. Performance probes

- Encoder-only throughput by sequence length: 512, 1024, 2048, 4096.
- Batch-size sweep for base and large: 1, 2, 4, 8 if memory allows.
- Window-size sweep on tiny/synthetic configs: 4, 32, 128, 512.
- Global-token count sweep: 0, 1, 16, 64, 256 global tokens per sample.
- Separate local attention timing from FFN/GEMM timing per layer.
- Pad overhead probe: raw `S` just below bucket multiple vs already divisible `S`.
- Task-head overhead: MLM vocab projection vs QA/classification small heads.
- Memory probe: local attention score/prob buffers `[B,S,H,attention_window+1]` plus global score buffers `[B,S,H,Gmax]` and `[B,H,Gmax,S]`.
- Dense fallback comparison for small `S` only, labeled as a parity/debug baseline.
- Attention-output materialization overhead when `output_attentions=True`.

No benchmark observations are included here; these are source-derived probe recommendations.

## 14. Skip/defer list

- Training losses, gradient checkpointing, dropout randomness.
- Autoregressive attention, dilated attention, KV cache, generation, beam search.
- ONNX export path unless an export-specific target requires it.
- `output_attentions` and `global_attentions` as production outputs; keep for parity/debug later.
- Dense full attention as the main long-context implementation.
- Compiling tokenizer and special-token insertion into the GPU runtime.
- Automatic QA/multiple-choice global-mask generation in the first GPU graph; accept explicit masks first.
- Quantization and multi-GPU/tensor-parallel sharding.

## 15. Final implementation checklist

- [ ] Parse LongformerConfig and checkpoint overrides.
- [ ] Preserve MLM embedding/decoder tied-weight aliases.
- [ ] Implement embedding lookup plus Longformer position IDs.
- [ ] Add pad-to-window and output unpad behavior.
- [ ] Implement local sliding-window attention parity kernel or reference lowering.
- [ ] Add attention-mask mapping for masked/local/global states.
- [ ] Implement global-token gather, dense global attention, and scatter/overwrite.
- [ ] Add LayerNorm, GELU FFN, pooler, and task heads.
- [ ] Add explicit `global_attention_mask` runtime input.
- [ ] Add preprocessing wrappers for sequence classification, QA, and multiple-choice global masks.
- [ ] Add layout/axis guards for `[B,S,H]`, `[B,H,S,D]`, and multiple-choice flattening.
- [ ] Reject unsupported cache/generation/ONNX/dilated/autoregressive paths.
- [ ] Add tiny single-layer and full-encoder parity tests.
- [ ] Add AllenAI base/large config smoke tests.
- [ ] Benchmark local attention, global-count sweep, FFN/GEMM, pad overhead, and head overhead.
