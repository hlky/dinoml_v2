# Transformers ESM Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Source family: src/transformers/models/esm.

Model id:
  Primary encoder examples: facebook/esm2_t6_8M_UR50D,
  facebook/esm2_t33_650M_UR50D, facebook/esm2_t48_15B_UR50D.
  Folding example: facebook/esmfold_v1.
  Legacy absolute-position example: facebook/esm1b_t33_650M_UR50S.

Config source:
  Raw Hugging Face config snapshots saved beside this report:
  facebook__esm2_t6_8M_UR50D.config.json
  facebook__esm2_t12_35M_UR50D.config.json
  facebook__esm2_t33_650M_UR50D.config.json
  facebook__esm2_t48_15B_UR50D.config.json
  facebook__esmfold_v1.config.json
  facebook__esm1b_t33_650M_UR50S.config.json

Source files inspected:
  X:/H/transformers/src/transformers/models/esm/configuration_esm.py
  X:/H/transformers/src/transformers/models/esm/modeling_esm.py
  X:/H/transformers/src/transformers/models/esm/modeling_esmfold.py
  X:/H/transformers/src/transformers/models/esm/tokenization_esm.py
  X:/H/transformers/src/transformers/models/esm/openfold_utils/*.py

Any missing files or assumptions:
  No remote-code files are required for these checkpoints. The sampled Hub
  config URLs were public, not gated. This report treats encoder masked-LM and
  ESMFold inference as the relevant runtime targets; training losses and PDB
  string formatting are deferred.
```

Relevant Hub links:
[esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D),
[esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D),
[esm2_t48_15B_UR50D](https://huggingface.co/facebook/esm2_t48_15B_UR50D),
[esmfold_v1](https://huggingface.co/facebook/esmfold_v1),
[esm1b_t33_650M_UR50S](https://huggingface.co/facebook/esm1b_t33_650M_UR50S).

## 2. High-level architecture

ESM has two different product shapes under one Transformers family.

```text
protein tokens -> token/position embeddings -> bidirectional encoder
  -> masked-LM head / classifier / token head / contact head
```

```text
protein sequence -> ESM-2 encoder hidden-state stack
  -> layer-combination + sequence projection
  -> OpenFold-style pair trunk with recycling
  -> structure module with invariant point attention
  -> atom coordinates, distogram, pLDDT, pTM, aligned-error heads
```

The plain ESM encoder is text-only in the sense that its inputs are amino-acid token IDs. It is noncausal by default and has no useful autoregressive KV-cache path for the public checkpoints. ESMFold is a staged protein-structure model: token preprocessing and amino-acid vocabulary mapping are CPU/data-pipeline work; the frozen language model stem, pair trunk, and structure module can be validated separately. The folding trunk has independently measurable sequence features `[B,L,1024]`, pair features `[B,L,L,128]`, and structure outputs, but it is not a generation decoder.

## 3. Important config dimensions

Representative checkpoint sweep:

| Checkpoint | Architecture | H | Layers | Heads | Head dim | FFN | Position | Emb LN before | Token dropout | Folding trunk |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| facebook/esm2_t6_8M_UR50D | EsmForMaskedLM | 320 | 6 | 20 | 16 | 1280 | rotary | false | true | none |
| facebook/esm2_t12_35M_UR50D | EsmForMaskedLM | 480 | 12 | 20 | 24 | 1920 | rotary | false | true | none |
| facebook/esm2_t33_650M_UR50D | EsmForMaskedLM | 1280 | 33 | 20 | 64 | 5120 | rotary | false | true | none |
| facebook/esm2_t48_15B_UR50D | EsmForMaskedLM | 5120 | 48 | 40 | 128 | 20480 | rotary | false | true | none |
| facebook/esmfold_v1 | EsmForProteinFolding | 2560 | 36 | 40 | 64 | 10240 | rotary | false | true | 48 blocks, `c_s=1024`, `c_z=128` |
| facebook/esm1b_t33_650M_UR50S | EsmForMaskedLM | 1280 | 33 | 20 | 64 | 5120 | absolute | true | true | none |

Common config facts from sampled configs:

| Field | Value / variation | Source |
|---|---|---|
| `vocab_size` | 33 for sampled public configs | config.json |
| `max_position_embeddings` | 1026 | config.json |
| `rope_theta` | source default 10000.0 when omitted | `EsmConfig` default |
| `layer_norm_eps` | 1e-5 in sampled configs | config.json |
| attention and hidden dropout | 0.0 in sampled configs | config.json |
| activation | custom exact GELU using `erf`, not PyTorch tanh approximation | source |
| `use_cache` | true in configs, but public encoder path does not expose persistent KV cache | config plus source inference |
| ESMFold `max_recycles` | 4, implemented as `num_recycles + 1` when provided | config plus source |
| ESMFold structure module | sequence dim 384, pair dim 128, IPA dim 16, 12 IPA heads, 4 q/k points, 8 value points, 8 blocks | config.json |

## 3a. Family variation traps

- ESM-2 uses rotary position embeddings; ESM-1b uses learned absolute positions and an embedding LayerNorm before the encoder.
- `hidden_size / num_attention_heads` is the actual head dim. It varies from 16 to 128 in the sampled configs.
- Attention Q/K/V/O and FFN linears all include bias in `EsmSelfAttention` and `EsmOutput`; ESMFold has a mix of biased and bias-free projections.
- The source scales `query_layer` by `head_dim ** -0.5` before RoPE and then passes `scaling=1.0` into attention. Fusing RoPE plus attention must preserve this order.
- `token_dropout=true` zeroes `<mask>` token embeddings and rescales by the observed mask ratio. This is graph-visible when masked tokens are present.
- Plain `EsmPreTrainedModel.get_output_embeddings()` intentionally returns `None`, while `EsmForMaskedLM` ties `lm_head.decoder.weight` to token embeddings via `_tied_weights_keys`.
- Contact prediction requires all layer attention maps materialized as `[B,layers,heads,S,S]`, then removes EOS and CLS positions, symmetrizes, applies average-product correction, and runs a logistic regression.
- `EsmConfig` exposes decoder and cross-attention flags inherited from BERT-like code, but the sampled ESM/ESMFold checkpoints are encoder-only.
- ESMFold rejects `use_esm_attn_map=True` in `EsmConfig.__post_init__`; public HF ESMFold does not consume ESM attention maps as pair features.
- ESMFold is not a small head. It has a 48-block pair trunk and an 8-block structure module, with O(L^2) pair state and many layout-sensitive pair operations.

## 4. Operator coverage checklist

### Tensor/layout ops

- Token embedding gather `[B,S] -> [B,S,H]`.
- Optional learned absolute position embedding gather for ESM-1b style configs.
- Rotary cos/sin creation from `position_ids`, broadcast to `[B,heads,S,D]`.
- Position IDs from input IDs: pads keep `padding_idx`, non-pads count upward from `padding_idx + 1`.
- Attention reshape and transpose: Q/K/V `[B,S,H] -> [B,A,S,D]`.
- Bidirectional mask broadcast into `[B,A,S,S]`; causal/cross masks only if using nonstandard decoder configs.
- Masked fill for token dropout and attention masks.
- Layer stack capture for hidden states and attentions when heads need them.
- Contact-head stack, slice, view, permute, transpose, sum, in-place divide, sigmoid.
- ESMFold pair tensor creation and updates: `[B,L,L,C_z]`, transpose row/column axes, chunked slices, in-place chunk writes.
- Rigid/rotation tensor wrappers map to tensor ops over rotations `[*,3,3]`, quaternions `[*,4]`, translations `[*,3]`, and points `[*,N_points,3]`.

### Neural network primitives

- LayerNorm with affine weight and bias.
- Linear with bias for ESM encoder:
  - Q/K/V/O: `Linear(H -> H)`.
  - FFN: `Linear(H -> 4H)`, exact GELU, `Linear(4H -> H)`.
  - MLM head: `Linear(H -> H)`, exact GELU, LayerNorm, tied decoder `Linear(H -> V, bias=False)` plus separate bias.
  - Pooler/classifier heads when needed.
- ESMFold trunk:
  - `esm_s_mlp`: LayerNorm, `Linear(2560 -> 1024)`, ReLU, `Linear(1024 -> 1024)` for esmfold_v1.
  - Sequence attention: packed bias-free `Linear(1024 -> 3072)`, gated output projection.
  - Sequence-to-pair: LayerNorm, `Linear(1024 -> 128)`, outer product/difference, `Linear(128 -> 128)`.
  - Pair-to-sequence bias: LayerNorm, `Linear(128 -> 32, bias=False)`.
  - Pair MLP: LayerNorm, `Linear(128 -> 512)`, ReLU, `Linear(512 -> 128)`.
  - Structure module: IPA projections, transition MLPs, backbone update `Linear(384 -> 6)`, angle resnet, lDDT/distogram/pTM heads.

### Attention primitives

- Bidirectional dense MHA for encoder checkpoints.
- Optional SDPA/Flash/Flex backend compatibility for plain ESM, but eager math must remain the parity reference.
- No GQA/MQA in sampled configs.
- No sliding-window/local attention in the encoder.
- ESMFold sequence attention with pairwise additive bias `[B,L,L,H_seq]`.
- ESMFold triangular row/column attention over pair state with triangle bias and mask bias.
- ESMFold invariant point attention, a custom geometric attention family, not ordinary MHA.

### Position/rotary/relative-bias ops

- RoPE with `base=rope_theta`, dim `H/heads`, float32 cos/sin generation, cast back to model dtype.
- Learned absolute positions for legacy configs.
- ESMFold relative position one-hot over clipped residue index differences into `2 * position_bins + 1` bins plus one masked bin, then linear-free embedding by one-hot matmul semantics.

### Generation/cache ops

- No autoregressive decode cache is required for first public ESM/ESMFold integration.
- `use_cache` should be ignored or rejected for encoder-only admission unless DinoML intentionally admits decoder-mode ESM configs.
- ESMFold recycling state is model-internal iterative state, not KV cache: `recycle_s`, `recycle_z`, and `recycle_bins` are recomputed per forward.

### Preprocessing-coupled ops

- `EsmTokenizer` is whitespace tokenization over amino-acid tokens plus `<cls>` and `<eos>` when special tokens are added.
- ESMFold `infer()` bypasses tokenizer and maps raw protein strings to residue constants, dense pads a batch, and builds `position_ids`.
- ESMFold `af2_idx_to_esm_idx` maps AlphaFold residue IDs to ESM vocabulary IDs through `af2_to_esm` gather.
- Atom14/atom37 masks and index maps are residue-type dependent table gathers.

## 5. Layer/block breakdown

Plain ESM encoder block, repeated `N` times:

```text
x = hidden_states
y = LayerNorm(x)
q = Linear(H -> H)(y).view(B,S,A,D).transpose(1,2)
k = Linear(H -> H)(y).view(B,S,A,D).transpose(1,2)
v = Linear(H -> H)(y).view(B,S,A,D).transpose(1,2)
q = q * D**-0.5
if rotary: q,k = RoPE(q,k,cos,sin)
y = Attention(q,k,v, bidirectional_mask, scaling=1.0)
x = x + Dropout(Linear(H -> H)(y))
y = LayerNorm(x)
y = exact_gelu(Linear(H -> 4H)(y))
x = x + Dropout(Linear(4H -> H)(y))
```

ESM embeddings:

```text
emb = word_embedding(input_ids)
if token_dropout: zero mask tokens and rescale by observed mask ratio
if absolute positions: emb += position_embedding(position_ids)
if emb_layer_norm_before: emb = LayerNorm(emb)
emb *= attention_mask[...,None] when mask is present
```

ESMFold high-level forward for esmfold_v1:

```text
aa [B,L] -> af2_to_esm gather -> add BOS/EOS for ESM stem
ESM stem returns hidden states [B,L+2,37,2560]
slice specials -> [B,L,37,2560]
softmax layer combine -> [B,L,2560]
esm_s_mlp -> s_s_0 [B,L,1024]
s_z_0 = zeros [B,L,L,128]
optional amino-acid embedding add
repeat recycle loop:
  add recycled sequence/pair state and relative position embedding
  run 48 triangular self-attention blocks
  project to structure module dims
  run 8 structure blocks with IPA/backbone/angle updates
  derive recycle distogram bins from backbone coordinates
heads: distogram, LM logits, lDDT, pTM, aligned error, atom masks/maps
```

ESMFold triangular block:

```text
bias = PairToSequence(pair)
seq = seq + gated SelfAttention(LayerNorm(seq), mask, bias)
seq = ResidueMLP(seq)
pair = pair + SequenceToPair(seq)
pair = pair + row_drop(TriangleMulOutgoing(pair))
pair = pair + col_drop(TriangleMulIncoming(pair))
pair = pair + row_drop(TriangleAttentionStarting(pair))
pair = pair + col_drop(TriangleAttentionEnding(pair))
pair = PairResidueMLP(pair)
```

## 6. Attention requirements

Plain ESM:

- Noncausal bidirectional self-attention for public checkpoints.
- MHA with `A = num_attention_heads`, `D = H / A`, Q/K/V width `H`.
- Additive attention mask, then softmax, dropout in training only, value matmul.
- Rotary configs apply RoPE after query pre-scaling. Absolute configs add learned embeddings before the encoder.
- Packed/varlen support is not present in source. Padding is handled by attention masks.
- FlashAttention/SDPA/Flex can be optimization backends for plain ESM if they preserve query pre-scaling, mask semantics, and optional attention output capture.
- No KV cache for first integration.

ESMFold:

- Sequence attention is noncausal self-attention with pairwise additive bias.
- Triangular attention treats pair rows or columns as the attention sequence. The `starting=False` path transposes pair axes before and after attention.
- IPA combines scalar attention logits, pair bias, and point-distance penalties. Its output concatenates scalar values, transformed point values, point norms, and pair-value aggregation.
- Chunking via `chunk_layer` and `_inference_forward` is a memory contract. A first dense implementation may ignore chunking only for small `L`, but production admission needs chunk-size and memory guards.

## 7. Position encoding and custom math

RoPE parity snippet:

```python
def esm_rope(q, k, inv_freq, position_ids):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(q.dtype).unsqueeze(1)
    sin = emb.sin().to(q.dtype).unsqueeze(1)
    return q.float().mul(cos) + rotate_half(q.float()).mul(sin), \
           k.float().mul(cos) + rotate_half(k.float()).mul(sin)
```

ESM exact GELU:

```python
def esm_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
```

Contact average-product correction:

```python
def apc(x):
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    return x - (a1 * a2 / a12)
```

ESMFold relative position:

```text
d = residue_index[:, :, None] - residue_index[:, None, :]
d = clip(d + position_bins, 0, 2 * position_bins)
if pair mask invalid: use final "masked" bin
one_hot(d, 2 * position_bins + 2) -> pair bias channels
```

Rigid-body math that needs explicit parity tests includes quaternion update composition, rotation-matrix multiplication written elementwise to avoid AMP downcasts, rotation application/inverse application to point tensors, torsion-angle frame construction, and atom14-to-atom37 gathers.

## 8. Preprocessing and input packing

Plain ESM tokenizer:

- `EsmTokenizer._tokenize` splits on whitespace. Callers usually pass amino acids as separated tokens or use helper processors.
- Special-token layout is `<cls> tokens <eos>` for one sequence, and `<cls> a <eos> b <eos>` for pairs.
- Model inputs are `input_ids` and `attention_mask`; there are no token type IDs.
- Position IDs are either supplied or derived from non-pad input IDs.

ESMFold:

- `infer()` maps raw protein strings through residue constants to AlphaFold-style residue IDs `[B,L]`, pads dense batches, and creates `position_ids`.
- Forward input IDs are AlphaFold residue IDs, not directly ESM vocab IDs. `af2_to_esm` converts them to ESM vocab IDs after adding 1 and masking pads to 0.
- The language-model stem adds BOS and EOS internally and masks padding using ESM pad ID 1.
- `masking_pattern` is training-oriented regularization and can be deferred for inference.
- PDB output formatting is CPU postprocessing and should not be part of the first compiled graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: ESM QKV projection packing

Source pattern:

```text
q = Linear(H -> H)(x)
k = Linear(H -> H)(x)
v = Linear(H -> H)(x)
reshape each to [B,A,S,D]
```

Replacement:

```text
packed_qkv = Linear(H -> 3H)(x) with row-concatenated [q; k; v] weights and biases
split final dim into q,k,v
```

Preconditions: same input tensor, same dtype/device, all three projections have bias, no hooks or weight aliases requiring separate modules. Weight transform is concatenate weights along output rows and concatenate biases. Failure cases: source-level attention output capture that expects module boundaries, quantized per-module storage, or checkpoint loading that must preserve separate names.

### Rewrite: exact GELU fusion

Source pattern: `Linear -> esm_gelu -> Linear`.

Replacement: fused GEMM epilogue only if the epilogue implements the `erf` exact formula. Do not substitute tanh GELU unless parity tolerance explicitly allows it.

### Rewrite: contact head APC kernel

Source pattern: symmetrize attentions, sum over rows/cols/matrix, divide, subtract, logistic regression.

Replacement: specialized pairwise contact feature kernel plus final channel projection.

Preconditions: materialized attentions `[B,layers*heads,L,L]`, CLS/EOS removal already applied, finite `a12`. Failure cases: padded all-empty sequences and attention backends that do not return weights.

### Rewrite: ESMFold chunked triangular update

Source pattern: pair projections, pair-axis permutations, batched matmul over triangle dimension, optional in-place chunk writes.

Replacement: provider-backed triangular multiplication with explicit row/column mode and chunk size.

Preconditions: pair state `[B,L,L,C_z]`, square `L`, fixed outgoing/incoming mode, same mask semantics. Failure cases: dynamic chunk sizes without workspace planning, in-place aliasing not represented in artifact-visible state.

### Rewrite: atom table gathers as static constants

Source pattern: residue-dependent gathers from `residue_constants` for atom masks, group indices, default frames, and literature positions.

Replacement: constant tables with explicit residency and index-gather ops.

Preconditions: residue vocabulary matches ESMFold defaults. Failure cases: alternate residue sets or caller-supplied nonstandard vocab lists.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + dense projections for encoder blocks. ESM uses many BERT-like linears and LayerNorms.
- RoPE + attention prefill for ESM-2, preserving query pre-scaling before RoPE.
- Exact GELU FFN fusion for `H -> 4H -> H`.
- ESMFold triangular multiplication and triangular attention. These dominate folding memory and runtime for large `L`.
- IPA custom kernel family for scalar plus point attention, including rigid transforms and pair bias.

Medium priority:

- Token-dropout embedding kernel for masked-LM parity.
- Contact head APC plus logistic regression for contact prediction.
- ESMFold sequence-to-pair outer product/difference kernel.
- Relative-position one-hot or embedding kernel for pair biases.
- Atom14/atom37 gather and coordinate conversion kernels.

Lower priority:

- Pooler/classification heads.
- Training losses and masking-pattern path.
- PDB string formatting.
- Decoder/cross-attention ESM variants, unless a real checkpoint requires them.

## 11. Runtime staging plan

Stage 1: plain ESM config and weights.

- Parse `EsmConfig`, load embeddings, encoder, MLM head, and tied decoder alias.
- Run one-block and full-encoder parity for small ESM-2 with fixed `[B,S]`.

Stage 2: plain ESM masked-LM inference.

- Add RoPE, exact GELU, token dropout, bidirectional masks, and MLM logits.
- Validate `esm2_t6_8M_UR50D` first, then larger checkpoint dimensions.

Stage 3: legacy absolute-position and contact head.

- Admit `esm1b_t33_650M_UR50S` absolute positions and `emb_layer_norm_before`.
- Add attention-weight capture and contact prediction as an optional path.

Stage 4: ESMFold stem and layer combine.

- Load `esmfold_v1`, run frozen ESM stem, stack hidden states, softmax-combine `esm_s_combine`, and `esm_s_mlp`.
- Stub trunk initially with zero or captured tensors only for ABI tests.

Stage 5: ESMFold trunk small-L parity.

- Implement pair state, relative positions, sequence attention with pair bias, triangular block primitives, and recycling for tiny `L`.
- Treat chunked inference as deferred until dense parity is established.

Stage 6: structure module and heads.

- Implement IPA, rigid updates, angle resnet, torsion frames, atom positions, distogram, pLDDT, pTM, and aligned-error heads.

Stage 7: production folding performance.

- Add chunked triangular kernels, workspace planning, O(L^2) memory admission, and per-stage benchmarks.

## 12. Parity and validation plan

- Unit-test `create_position_ids_from_input_ids` against pad patterns.
- Unit-test RoPE against source for multiple head dims and position IDs, fp32 and fp16.
- Unit-test exact GELU versus source, with a guard against tanh-GELU substitution.
- Unit-test token dropout scaling with no mask tokens, some mask tokens, and padding.
- Single ESM block parity for `facebook/esm2_t6_8M_UR50D`, fp32 tolerance around `1e-5` absolute and relative.
- Full encoder masked-LM logits parity on short protein strings, fp32 tolerance around `1e-4`; fp16 tolerance should be set after source comparison.
- Contact head parity using captured attention tensors and short sequences.
- ESMFold stage parity:
  - `af2_idx_to_esm_idx` and BOS/EOS construction.
  - ESM hidden-state stack and layer combine.
  - one triangular block on random tensors `[B=1,L<=16]`.
  - IPA on random rigid transforms and masks.
  - structure module one recycle on short sequence.
  - full `esmfold_v1` on a tiny peptide with CPU/PyTorch reference outputs for shapes and selected numeric tensors.

## 13. Performance probes

- Tokenization/raw-sequence preprocessing throughput for batch size and sequence length sweeps.
- Plain ESM encoder latency by `(B,S,H,layers)` and attention backend.
- MLM head cost, especially for `H=5120,V=33` where vocab is tiny and encoder dominates.
- Contact prediction overhead from attention materialization `[layers,heads,S,S]`.
- ESMFold ESM stem time versus trunk time versus structure module time.
- Pair memory footprint sweep for `L`: `[B,L,L,128]` plus temporaries.
- Recycling count sweep, including `num_recycles=None` default.
- Chunk size sweep for triangular attention/multiplication once chunked lowering exists.
- IPA scalar attention versus point attention cost.
- Atom-coordinate postprocessing cost separately from trunk.

## 14. Skip/defer list

- Training losses, gradient checkpointing, dropout stochasticity, and masking-pattern regularization.
- Decoder/cross-attention/cache mode unless a real public ESM checkpoint requires it.
- `use_esm_attn_map=True`; the HF config rejects it for this source basis.
- PDB string serialization and file output.
- Chunked ESMFold kernels for first dense small-L parity, but not for production folding.
- CPU offload paths and DeepSpeed conditionals in ESMFold.
- Quantized/packed weight formats; sampled configs use ordinary dense weights.
- Multi-GPU/tensor parallel execution.

## 15. Final implementation checklist

- [ ] Parse `EsmConfig`, including `is_folding_model` and nested `esmfold_config`.
- [ ] Load ESM vocab/token IDs and enforce 33-token protein vocabulary assumptions.
- [ ] Preserve tied embedding/MLM decoder alias for `EsmForMaskedLM`.
- [ ] Implement ESM embeddings, position IDs, token dropout, and optional embedding LayerNorm.
- [ ] Implement exact ESM GELU.
- [ ] Implement RoPE with query pre-scaling before rotation.
- [ ] Implement bidirectional encoder MHA and FFN parity.
- [ ] Add masked-LM logits parity for `facebook/esm2_t6_8M_UR50D`.
- [ ] Add legacy absolute-position parity for `facebook/esm1b_t33_650M_UR50S`.
- [ ] Implement optional contact head with attention capture and APC.
- [ ] Implement ESMFold AlphaFold-to-ESM token mapping and hidden-state layer combine.
- [ ] Implement ESMFold pair state, relative position bins, and triangular block primitives.
- [ ] Implement ESMFold IPA and rigid-body math helpers.
- [ ] Implement atom14/atom37 masks, torsion frames, atom positions, distogram, pLDDT, pTM, and aligned-error heads.
- [ ] Add small-L ESMFold parity before optimizing chunked folding kernels.
- [ ] Benchmark encoder, contact head, ESMFold stem, pair trunk, structure module, and recycling separately.
