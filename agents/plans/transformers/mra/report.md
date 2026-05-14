# MRA Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` in `X:/H/transformers`.

Model id: primary public checkpoints are `uw-madison/mra-base-512-4` and `uw-madison/mra-base-4096-8-d3`.

Config source: pinned source defaults in `src/transformers/models/mra/configuration_mra.py`; live HF `config.json` for the two public checkpoints; test debug config in `tests/models/mra/test_modeling_mra.py`.

Source files inspected:

- `src/transformers/models/mra/configuration_mra.py`
- `src/transformers/models/mra/modeling_mra.py`
- `src/transformers/models/mra/convert_mra_pytorch_to_pytorch.py`
- `tests/models/mra/test_modeling_mra.py`
- `docs/source/en/model_doc/mra.md`
- External kernel metadata/source from HF repo `kernels-community/mra`, because the pinned modeling source loads it dynamically with `get_kernel("kernels-community/mra")`.

Any missing files or assumptions:

- The pinned Transformers checkout no longer carries local `src/transformers/kernels/mra/*` files; the runtime source uses the `kernels` integration instead.
- The kernel repo is external and dynamically resolved; unless DinoML pins a kernel repo revision, this is not fully reproducible from the Transformers commit alone.
- The two public model repos did not include tokenizer files; raw requests for `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, and `merges.txt` returned 404. The docs/examples use `AutoTokenizer.from_pretrained(...)`, but the model repos themselves do not provide the tokenizer ABI.
- No gated/401 MRA checkpoint was encountered. The report uses the two public official checkpoints plus source/test defaults rather than a 3-5 checkpoint sweep, because only two official MRA checkpoint repos were found.

Primary runtime target for this report: encoder-only masked language modeling and encoder feature/classification tasks. MRA has no autoregressive generation or KV-cache decode path in this source.

## 2. High-level architecture

MRA is a BERT/RoBERTa-like text encoder with absolute position embeddings and a custom multi-resolution approximate self-attention block. It supports masked LM, sequence classification, multiple choice, token classification, and extractive QA heads.

Dataflow:

```text
tokenized text ids + attention mask + optional token_type_ids
-> word/token_type/absolute position embeddings + LayerNorm
-> N encoder layers with MRA approximate noncausal self-attention + FFN
-> task head: MLM logits, CLS classification, per-token labels, multiple-choice scores, or QA spans
```

Stage decomposition:

- CPU/data pipeline: text tokenization, padding, mask creation, optional pair/segment packing. Tokenizer files are not present in the MRA repos, so first integration should require caller-supplied tokenizer assets or a documented RoBERTa-compatible tokenizer.
- GPU/runtime encoder: embeddings, repeated MRA encoder blocks, and task head.
- Independently stageable units: embeddings; one MRA attention block; full encoder; each head.
- Cacheable outputs: encoder hidden states can be cached for downstream tasks, but there is no per-layer KV cache or decode loop.

## 3. Important config dimensions

| Dimension | Source default | `mra-base-512-4` | `mra-base-4096-8-d3` | Runtime significance |
|---|---:|---:|---:|---|
| `vocab_size` | 50265 | 50265 | 50265 | Embedding and MLM decoder width |
| `hidden_size` | 768 | 768 | 768 | Model width |
| `num_hidden_layers` | 12 | 12 | 12 | Encoder block count |
| `num_attention_heads` | 12 | 12 | 12 | MHA head count |
| `head_dim` | inferred 64 | nested config says 64 | legacy config says 64 | Must satisfy `hidden_size / heads`; source has no standalone `head_dim` field |
| `intermediate_size` | 3072 | 3072 | 3072 | FFN expansion |
| `max_position_embeddings` | 512 | 512 | 4096 | Position table length and attention block-budget computation |
| `type_vocab_size` | 1 | 1 | 1 | Public tests also use 16 for generic coverage |
| `hidden_act` | `gelu` | `gelu` | `gelu` | FFN and MLM transform activation |
| `layer_norm_eps` | `1e-5` | `1e-5` | `1e-5` | LayerNorm epsilon |
| `block_per_row` | 4 | 4 | 8 | High-resolution selected block budget |
| `approx_mode` | `full` | `full` | `full` | `full` combines low+high resolution; `sparse` uses only high-resolution blocks |
| `initial_prior_first_n_blocks` | 0 | 0 | 3 | Biases top-k block selection toward first rows/cols |
| `initial_prior_diagonal_n_blocks` | 0 | 0 | 1 | Biases top-k block selection toward diagonal band |
| `torch_dtype` | source unset | `float32` | `float32` | Source casts Q/K/V to float for MRA attention |
| cache support | none | none | none | Encoder-only; no generation cache |

Representative checkpoint/config sweep:

| Case | Provenance | Shape/operator variation |
|---|---|---|
| Source default | `MraConfig` defaults | Base 512-token shape, `block_per_row=4`, `approx_mode=full` |
| `uw-madison/mra-base-512-4` | HF `config.json` | Same operator shape as source defaults; includes legacy nested `model` metadata ignored by current source |
| `uw-madison/mra-base-4096-8-d3` | HF `config.json` | Long context 4096, `block_per_row=8`, first-block prior 3, diagonal prior 1 |
| Tiny test config | `tests/models/mra/test_modeling_mra.py` | `hidden_size=16`, heads=2, seq=64; source pads attention head dim to 32 before MRA kernels |

## 3a. Family variation traps

- Sequence length must be divisible by the hard-coded MRA `block_size=32`; the modeling source raises otherwise.
- The custom CUDA kernels expect fixed 32x32 sparse blocks and float tensors. The source casts Q/K/V and attention mask to float before attention.
- Head dimension below 32 is padded with zeros to a 32-wide kernel shape, then sliced back. This matters for small/debug configs.
- `num_block = min((max_position_embeddings // 32) * block_per_row, (max_position_embeddings // 32) ** 2)` is derived from max position length, not from runtime sequence length. Long and short checkpoints therefore use different top-k block budgets.
- `approx_mode` changes required math. `full` combines selected high-resolution blocks with a low-resolution dense block summary; `sparse` skips the low-resolution output path.
- Public configs include legacy fields such as `dim`, `head_dim`, `num_head`, `num_layers`, `mixed_precision`, `shared_weight`, and nested `model`. The inspected source ignores these fields for forward execution.
- `config.add_cross_attention` is stored on `MraLayer` but no cross-attention module is constructed or used in `forward`; reject decoder/cross-attention interpretations for this source basis.
- The source does not return attention tensors; tests explicitly mark attention-output tests skipped and set `has_attentions=False`.
- The no-kernel fallback in `mra2_attention` returns `torch.zeros_like(query).requires_grad_()`, which is not useful inference parity. DinoML should treat missing MRA kernel support as unsupported unless deliberately using a dense-reference rewrite for validation.
- Tokenizer ABI is under-specified by the official model repos. Vocab size and special ids match a RoBERTa-style setup (`bos=0`, `pad=1`, `eos=2`, `<mask>` implied by MLM examples), but the actual vocab/merges are absent from the MRA repos.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token lookup for `input_ids`, `token_type_ids`, `position_ids`.
- Add three embedding tensors: `[B,S,H] + [B,S,H] + [1 or B,S,H]`.
- `view`, `reshape`, `transpose`, `permute`, `contiguous`, `squeeze`, `repeat`, `cat`, `split`, `clamp`, `where`.
- `topk` over flattened block logits `[B*heads, num_blocks^2]`.
- `index_add` for sparse normalizer accumulation.
- Advanced gather for sparse mask: `mask[batch_idx[:, None], (indices % num_block), :]`.

Neural network primitives:

- Embedding tables: token `[50265, 768]`, position `[max_position_embeddings + 2, 768]`, token type `[type_vocab_size, 768]`.
- LayerNorm over hidden width with epsilon `1e-5`.
- Linear Q/K/V: `Linear(768 -> 768)` with bias.
- Attention output projection: `Linear(768 -> 768)` with bias.
- FFN: `Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768)`.
- Residual add + dropout + LayerNorm after attention and FFN.
- MLM transform: `Linear(768 -> 768) -> GELU -> LayerNorm -> Linear(768 -> 50265)`.
- Classification heads: CLS pooling `hidden[:,0,:]`, dropout, dense, activation, output projection.
- QA head: `Linear(768 -> 2)` then split into start/end logits.

Attention primitives:

- Noncausal encoder self-attention only.
- Custom MRA attention: low-resolution block averaging, block-logit top-k selection, sampled dense block matmul, block sparse-dense matmul, sparse max, sparse reduce-sum, low/high-resolution softmax normalizer correction.
- No RoPE, ALiBi, causal mask, cross-attention, packed varlen attention, or KV cache.

Custom sparse/kernel ops:

- `index_max(index_vals, indices, A_num_block, B_num_block) -> [max_vals, max_vals_scatter]`.
- `mm_to_sparse(dense_A, dense_B, indices) -> [batch_heads, selected_blocks, 32, 32]`.
- `sparse_dense_mm(sparse_A, indices, dense_B, A_num_block) -> dense output blocks`.
- `reduce_sum` exists in the external kernel repo but the inspected modeling source uses a Python `index_add` implementation in `MraReduceSum`.
- External kernel repo also exposes `scatter`, unused by the pinned modeling source.

Position encoding:

- Learned absolute position embeddings with `position_ids = arange(max_position_embeddings) + 2`.
- No rotary or relative-position bias.

Preprocessing-coupled ops:

- Attention mask input `[B,S]`; source converts through `get_extended_attention_mask`, then reverts it inside attention with `1 + mask / 10000`.
- Segment/token type ids default to all zeros and are usually width 1 in public configs.

Tied weights:

- `MraForMaskedLM` ties `cls.predictions.decoder.weight` to `mra.embeddings.word_embeddings.weight`.
- Decoder bias is aliased through `cls.predictions.decoder.bias` and `cls.predictions.bias`.

## 5. Layer/block breakdown

Embeddings:

```text
input_ids/token_type_ids/position_ids -> embeddings
x = word_embeddings + token_type_embeddings + position_embeddings
x = LayerNorm(x)
x = Dropout(x)
```

Encoder block, repeated `num_hidden_layers` times:

```text
q = Linear(H -> H)(x).view(B,S,heads,D).transpose(1,2)
k = Linear(H -> H)(x).view(B,S,heads,D).transpose(1,2)
v = Linear(H -> H)(x).view(B,S,heads,D).transpose(1,2)
if D < 32: pad q/k/v last dim to 32
attn = mra2_attention(q.float(), k.float(), v.float(), mask.float(), num_blocks)
if D < 32: slice attn last dim back to D
attn = attn.permute(B,S,heads,D).contiguous().view(B,S,H)
x = LayerNorm(x + Dropout(Linear(H -> H)(attn)))
ff = GELU(Linear(H -> I)(x))
x = LayerNorm(x + Dropout(Linear(I -> H)(ff)))
```

For the public checkpoints: `H=768`, `heads=12`, `D=64`, `I=3072`, so no head-dim padding is applied.

Task heads:

- Masked LM: per-token hidden `[B,S,768]` to logits `[B,S,50265]`.
- Sequence classification: CLS hidden `[B,768]` to `[B,num_labels]`.
- Multiple choice: flatten `[B,C,S] -> [B*C,S]`, CLS hidden to scalar per choice, reshape `[B,C]`.
- Token classification: per-token hidden to `[B,S,num_labels]`.
- Question answering: per-token hidden to `[B,S,2]`, split/squeeze to start/end `[B,S]`.

## 6. Attention requirements

Attention type:

- Noncausal self-attention in an encoder block.
- Dense Q/K/V projections are standard MHA with `num_attention_heads`; there is no MQA/GQA.
- Query, key, and value widths are all `hidden_size`; per-head width is `hidden_size / num_attention_heads`.
- Source requires `hidden_size % num_attention_heads == 0`.

Masking style:

- Public input mask is `[B,S]` with 1 for valid tokens and 0 for padding.
- `get_extended_attention_mask` creates a large-negative additive mask; `MraSelfAttention` converts it back to a binary-ish mask by `1.0 + attention_mask / 10000.0`, then squeezes/repeats to `[B*heads,S]`.
- Padding mask is used in low-resolution block averages, high-resolution sparse logits, and final context masking.

MRA block pattern:

- Fixed block size is 32 tokens.
- Runtime `S` must be divisible by 32.
- `num_block_per_row = S // 32`.
- `num_blocks` selected per meta-batch row is computed at layer construction from `max_position_embeddings` and `block_per_row`, then capped by full block grid size for max length.
- Low-resolution logits have shape `[B*heads, S/32, S/32]`.
- High-resolution sparse block logits have shape `[B*heads, num_blocks, 32, 32]`.

Packed/varlen support: none in source. Variable sequence lengths are represented by padding and an attention mask; actual runtime `S` still has the 32-divisibility guard.

Sliding-window/local attention: not a fixed window. Priors can bias selection toward diagonal/first blocks, but actual selected blocks are top-k over low-resolution scores.

KV cache: not applicable. This is not an autoregressive decoder.

FlashAttention/SDPA compatibility: not directly compatible. MRA selects a data-dependent sparse block set and combines high-resolution sampled attention with optional low-resolution summary attention. A dense SDPA fallback can validate semantics only if it implements the approximation explicitly, not by calling ordinary full attention.

## 7. Position encoding and custom math

Position encoding is learned absolute embeddings:

```python
position_ids = arange(max_position_embeddings)[None, :] + 2
position_embeddings = Embedding(max_position_embeddings + 2, hidden_size)(position_ids[:, :S])
```

The core model-specific math is MRA block attention:

```python
def mra_block_attention(q, k, v, mask, block_size=32):
    # q/k/v: [B, H, S, D], S % 32 == 0
    meta = B * H
    q, k, v = reshape_to_meta_batch(q, k, v)  # [meta, S, D]
    q_hat = masked_block_mean(q, mask, block_size)  # [meta, S/32, D]
    k_hat = masked_block_mean(k, mask, block_size)
    low_logits = q_hat @ k_hat.transpose(-1, -2) / sqrt(D)
    indices = topk(flatten(low_logits - row_max(low_logits)), num_blocks)
    high_logits = sampled_block_matmul(q, k, indices) / sqrt(D)
    high_attn = exp(high_logits - sparse_block_max(high_logits, indices))
    high_out = sparse_dense_mm(high_attn, indices, v)
    high_norm = sparse_reduce_sum(high_attn, indices)
    if approx_mode == "full":
        low_out, low_norm = low_resolution_value_path(...)
        return corrected_merge(high_out, high_norm, low_out, low_norm)
    return high_out / (high_norm[..., None] + 1e-6)
```

Precomputable:

- Position id buffer.
- Static block-grid metadata for a fixed `max_position_embeddings`, `block_per_row`, and `block_size`.

Dynamic:

- Low-resolution logits, top-k indices, high-resolution sparse blocks, mask-derived token counts, max corrections, and normalizers all depend on input hidden states and masks.

## 8. Preprocessing and input packing

Text inputs:

- `input_ids`: `[B,S]` integer ids, or `inputs_embeds`: `[B,S,H]`.
- `attention_mask`: optional `[B,S]`; defaults to all ones.
- `token_type_ids`: optional `[B,S]`; defaults to all zeros from a non-persistent buffer.
- `position_ids`: optional `[1,S]`/broadcastable ids; defaults to source buffer offset by 2.

Tokenizer ABI:

- The public repos do not include tokenizer artifacts. DinoML should not assume the model repo alone is sufficient for end-to-end text parity.
- Special ids in config are `bos=0`, `pad=1`, `eos=2`; examples use `<mask>`, and vocab size is RoBERTa-like 50265. Treat the actual tokenizer vocab/merges and mask id as external assets to be supplied or pinned separately.

GPU/runtime:

- Tokenization, pair packing, padding, and special-token insertion are CPU/data-pipeline work.
- The runtime graph starts at integer ids/masks or at precomputed `inputs_embeds`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: MRA QKV linear packing

Source pattern:

```text
q = Linear(H -> H)(x)
k = Linear(H -> H)(x)
v = Linear(H -> H)(x)
reshape/transposes into [B, heads, S, D]
```

Replacement pattern:

```text
single GEMM H -> 3H -> split [q,k,v] -> reshape/transposes
```

Preconditions:

- Same input tensor `x`.
- All three projections have bias and the same output width.
- Weight packing order must be `[Q, K, V]`, with PyTorch `Linear` weight layout `[out_features, in_features]`.

Shape equations:

- Input `[B,S,H]`, packed output `[B,S,3H]`, split each `[B,S,H]`.

Failure cases:

- Any checkpoint with nonstandard split layout or missing projection bias must be rejected or routed separately. No such public MRA config was found.

Parity test sketch:

- Random one-layer MRA attention; compare separate Linear path to packed GEMM split before MRA attention.

### Rewrite: low-resolution block mean as reshape-reduce

Source pattern:

```text
x.reshape(meta, S/32, 32, D).sum(dim=-2) / token_count
```

Replacement pattern:

```text
block_reduce_sum over fixed 32-token tile, divide by mask count
```

Preconditions:

- `S % 32 == 0`.
- Dense contiguous `[meta,S,D]` layout or a generated accessor that preserves the same block order.
- Mask is binary `[meta,S]`.

Shape equations:

- `[meta,S,D] -> [meta,S/32,D]`.

Failure cases:

- Non-divisible sequence length, non-binary masks, or layout translation that changes token order.

Parity test sketch:

- Random `q/k/v` and masks with full, partial, and empty blocks; compare against PyTorch source math with `1e-6` denominator.

### Rewrite: MRA sparse block ops as provider-backed custom attention

Source pattern:

```text
topk block indices -> mm_to_sparse -> sparse_max -> exp -> sparse_dense_mm -> reduce_sum/index_add
```

Replacement pattern:

```text
MRAAttentionProvider(q,k,v,mask, block_size=32, selected_blocks, mode)
```

Preconditions:

- CUDA target.
- Float32 attention math.
- `S % 32 == 0`.
- `num_blocks` divisible by launch grouping requirements where using the external kernel behavior (`mm_to_sparse` launch uses `num_block / 4`; `sparse_dense_mm` uses `num_block / 2`).
- `D` is either >=32 and compatible with kernel dim loops, or padded to 32 exactly as source.

Weight transform: none.

Layout constraints:

- Preserve `[B,heads,S,D]` to meta-batch `[B*heads,S,D]` ordering.
- No NHWC-style layout rewrite is relevant; this is sequence data.

Failure cases:

- CPU-only execution, missing kernel, sequence not divisible by 32, unsupported dtype, unsupported `approx_mode`.

Parity test sketch:

- Compare provider output to a Python reference for small block grids. Include `approx_mode=full` and `sparse`, first/diagonal priors, all-valid mask, partial padding mask, and debug `D<32` padding.

### Rewrite: MLM tied embedding output GEMM

Source pattern:

```text
hidden -> dense -> activation -> LayerNorm -> Linear(H -> vocab)
```

Replacement pattern:

```text
fused transform + GEMM using shared embedding weight
```

Preconditions:

- Preserve tied logical identity between input embeddings and decoder weight.
- Decoder bias is separate and must be applied.

Failure cases:

- Weight untied by user mutation or resized embeddings.

Parity test sketch:

- Load MLM checkpoint; compare logits slices and full masked-token logits.

## 10. Kernel fusion candidates

Highest priority:

- MRA custom attention provider. This is the defining operator; without it, the source fallback returns zeros and cannot provide inference parity.
- Block reduce + top-k + sampled block matmul pipeline. The low-resolution block score and high-resolution selected block path determine both accuracy and performance.
- Dense QKV packed GEMM feeding MRA attention. It removes three GEMM launches per layer and gives a clean handoff to the custom attention provider.

Medium priority:

- LayerNorm + residual/dropout-free inference fusion. Dropout is inactive in eval; residual plus LayerNorm appears twice per block.
- FFN GEMM + GELU fusion. Standard encoder bottleneck, `768 -> 3072 -> 768` for public checkpoints.
- MLM transform + tied-vocab GEMM. Important for fill-mask throughput, especially full-sequence logits.

Lower priority:

- Classification/QA head fusions. Heads are small relative to encoder cost.
- Mask preprocessing fusion. Useful for avoiding mask reshape/repeat overhead but not the main bottleneck.
- Low-resolution dense block GEMM optimization for long context. Worth probing after parity, because 4096-token configs produce a 128x128 block-logit matrix per head/meta-batch.

## 11. Runtime staging plan

Stage 1: Parse config and load weights.

- Admit `MraConfig` fields used by source.
- Preserve ignored legacy fields only as metadata.
- Validate `hidden_size % num_attention_heads == 0`, `approx_mode in {"full","sparse"}`, and `max_position_embeddings % 32 == 0`.
- Preserve MLM tied-weight aliases.

Stage 2: Embeddings and FFN block parity.

- Implement embeddings, position offset `+2`, token type defaults, LayerNorm, Linear, GELU, residual adds.
- Stub attention with a controlled reference only for unit tests, not production.

Stage 3: Python/reference MRA attention parity.

- Implement a clear reference for `mra2_attention` on CPU/GPU tensors to validate shapes and masks.
- Include top-k index generation, priors, low/high correction math, and D<32 padding behavior.

Stage 4: CUDA MRA attention provider.

- Add provider manifest for fixed 32-token blocks, float32 math, selected block count, and launch ABI.
- Either vendor/pin the external `kernels-community/mra` source or reimplement equivalent kernels with explicit provenance.

Stage 5: Full encoder and MLM parity.

- Run one-layer, N-layer, and public checkpoint logits comparisons.
- Add long-context `4096-8-d3` shape coverage.

Stage 6: Add task heads.

- Sequence classification, token classification, multiple choice, and QA are straightforward once encoder output parity exists.

Stage 7: Optimize rewrites/fusions.

- Packed QKV, residual LayerNorm, FFN activation, and provider-specific MRA attention specializations.

## 12. Parity and validation plan

Custom op tests:

- `sparse_mask`: random masks and indices, compare gather semantics.
- `get_low_resolution_logit`: all-valid, partial-mask, and empty-block cases.
- `get_block_idxes`: full/sparse modes, first-block and diagonal priors, threshold tie behavior.
- `mra2_attention`: small grids with a Python reference, `S=32/64/128`, `D=16/32/64`, all-valid and padded masks.

Layer tests:

- Single MRA attention layer with random weights, compare hidden output to PyTorch source.
- One full encoder block with eval dropout disabled.
- After-N-layer drift tests for 2-layer tiny config and 12-layer public configs.

Checkpoint tests:

- `uw-madison/mra-base-512-4` base model expected shape `[1,256,768]` and a small output slice, matching the source integration test.
- `uw-madison/mra-base-512-4` MLM expected logits shape `[1,256,50265]`.
- `uw-madison/mra-base-4096-8-d3` MLM expected logits shape `[1,4096,50265]`.

Head tests:

- Sequence classification: CLS-only output shape `[B,num_labels]`.
- Multiple choice: flatten/reshape shape `[B,C]`.
- Token classification: `[B,S,num_labels]`.
- QA: start/end logits `[B,S]`.

Recommended tolerances:

- fp32 attention/provider parity: start at `rtol=1e-4`, `atol=1e-4`, matching source integration tests.
- For optimized fused kernels that reorder reductions/top-k ties, add stricter index/tie tests and allow small numeric drift only after proving selected block indices match.

## 13. Performance probes

- Kernel availability and load time for the MRA provider, including cold Hub/kernel load if DinoML composes external kernels.
- Encoder throughput for `S=512` and `S=4096`, batch sweep `B=1,2,4,8`.
- Sequence length sweep with divisibility guard: `S=32,64,128,256,512,1024,2048,4096`.
- `block_per_row` sweep and selected-block budget: `num_blocks = min((max_pos/32)*block_per_row, (max_pos/32)^2)`.
- Attention provider breakdown: low-resolution block reduce/GEMM, top-k, sampled dense block matmul, sparse max, exp, sparse-dense matmul, reduce/index-add, correction merge.
- Compare `approx_mode=full` versus `sparse` with identical inputs where configs permit.
- Head-dim padding overhead for debug/small models with `D<32`.
- Full MLM logits cost versus encoder-only hidden-state cost.
- Memory probes for sparse temporary tensors: `[B*heads,num_blocks,32,32]` and low-resolution `[B*heads,S/32,S/32]`.

## 14. Skip/defer list

- Training and gradients. The tests already xfail several gradient checkpointing cases.
- Dropout behavior beyond eval-mode identity.
- Attention weight outputs; source tests mark them unsupported.
- Autoregressive generation, KV cache, beam search, and causal masks.
- Cross-attention despite the config field; source does not implement it.
- General block-sparse attention beyond MRA's exact 32x32 data-dependent pattern.
- CPU production kernel for MRA attention. A CPU reference is useful for validation, but first useful runtime target should be CUDA.
- Tokenizer implementation unless a caller supplies/pins RoBERTa-compatible assets.
- External dynamic kernel downloading in production. Prefer vendored/pinned provider source or explicit rejection.

## 15. Final implementation checklist

- [ ] Parse `MraConfig` fields used by source and preserve ignored legacy fields as metadata.
- [ ] Add config admission guards for `approx_mode`, `hidden_size % num_attention_heads`, `S % 32`, and supported dtype/device.
- [ ] Load embeddings, encoder, heads, and MLM tied-weight aliases.
- [ ] Implement embeddings with `position_ids + 2` and token-type default buffer behavior.
- [ ] Implement/evaluate reference `mra2_attention` including low-resolution, top-k, priors, sparse high-resolution path, and correction merge.
- [ ] Add CUDA provider contract for MRA sparse block ops or vendor/pin `kernels-community/mra`.
- [ ] Implement custom ops: sampled dense block matmul, sparse-dense block matmul, sparse max/scatter max, sparse reduce-sum/index-add.
- [ ] Add packed QKV rewrite with `[Q,K,V]` split order and bias packing.
- [ ] Add one-layer and N-layer parity tests.
- [ ] Add public checkpoint parity for `mra-base-512-4` and `mra-base-4096-8-d3`.
- [ ] Add MLM, sequence classification, multiple choice, token classification, and QA head tests.
- [ ] Benchmark encoder-only, MRA attention-only, and MLM logits separately for 512 and 4096 contexts.
