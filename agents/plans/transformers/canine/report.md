# CANINE Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary base checkpoints: google/canine-s and google/canine-c.
  Representative variants: hf-internal-testing/tiny-random-CanineForMultipleChoice,
  Splend1dchan/canine-s-squad, celine98/canine-s-finetuned-sst2.

Config source:
  Hugging Face config snapshots saved under
  H:/dinoml_v2/agents/plans/transformers/canine/_sources/.

Source files inspected:
  transformers/src/transformers/models/canine/modeling_canine.py
  transformers/src/transformers/models/canine/configuration_canine.py
  transformers/src/transformers/models/canine/tokenization_canine.py
  transformers/src/transformers/models/canine/__init__.py

Any missing files or assumptions:
  No remote-code files are required for the inspected checkpoints. This report
  targets encoder inference first: base CanineModel, sequence classification,
  token classification, multiple choice, and span QA. Masked LM is not a first
  target because this source has helper MLM heads but no exported
  CanineForMaskedLM class, and ConvProjection final_seq_char_positions raises
  NotImplementedError.
```

Primary source URLs:

- `https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/canine`
- `https://huggingface.co/google/canine-s/blob/main/config.json`
- `https://huggingface.co/google/canine-c/blob/main/config.json`
- `https://huggingface.co/hf-internal-testing/tiny-random-CanineForMultipleChoice/blob/main/config.json`
- `https://huggingface.co/Splend1dchan/canine-s-squad/blob/main/config.json`
- `https://huggingface.co/celine98/canine-s-finetuned-sst2/blob/main/config.json`

## 2. High-level architecture

CANINE is a tokenization-free text encoder. The tokenizer emits Unicode codepoint ids, not wordpiece ids. The model hashes those integer ids into multiple embedding tables, runs a shallow character-level local-attention encoder, downsamples character states into molecule states with strided Conv1d, runs a full-attention deep encoder at the shorter molecule length, repeats molecule states back to character length, projects them with a padded Conv1d, then runs a final shallow character encoder.

```text
Unicode text -> codepoint ids + [CLS]/[SEP]/token_type_ids
  -> multi-hash character embeddings + position/type embeddings
  -> 1-layer local character encoder
  -> strided Conv1d downsample to molecule sequence
  -> N-layer full self-attention molecule encoder
  -> repeat molecules to character length + concat with initial char states
  -> same-padded Conv1d projection
  -> 1-layer full character encoder
  -> optional pooler or task head
```

Stage decomposition:

- CPU/data pipeline: Unicode string splitting, special private-use codepoint insertion, padding, `attention_mask`, `token_type_ids`.
- GPU/runtime: hashed embedding arithmetic and gathers, local shallow encoder, Conv1d downsampling, molecule encoder, repeat/concat/projection, final encoder, task head.
- Independently cacheable: base encoder outputs may be cached by downstream applications; this is not an autoregressive KV cache.
- Independently testable: tokenizer/codepoint ABI, hash embedding, one local attention layer, downsample/upsample path, one full encoder layer, each task head.

## 3. Important config dimensions

Worked example: `google/canine-s` and `google/canine-c`. Config facts below come from `config.json` unless marked as source-derived or inferred.

| Field | Default / base value | Runtime effect |
|---|---:|---|
| `model_type` | `canine` | selects in-library CANINE source |
| tokenizer vocab | 1114112 | tokenizer property: all Unicode codepoints; source-derived |
| learned codepoint tables | 8 x `[16384, 96]` | default hash shards concatenate to hidden 768 |
| `hidden_size` / H | 768 | encoder width |
| `num_hidden_layers` | 12 | deep molecule encoder layers only; two shallow encoders add 1 layer each |
| effective total transformer layers | 14 | inferred from source construction |
| `num_attention_heads` / A | 12 | MHA heads |
| `head_dim` / D | 64 | inferred as `H / A` |
| `intermediate_size` / I | 3072 | FFN expansion |
| `hidden_act` | `gelu` | FFN and Conv1d projection/downsample activation |
| `max_position_embeddings` | 16384 | absolute character position table length |
| `type_vocab_size` | 16 | token type embedding table |
| `num_hash_functions` | 8 | number of shard embedding tables |
| `num_hash_buckets` | 16384 | buckets per hash table and position table size in source |
| `downsampling_rate` | 4 | Conv1d kernel/stride from character to molecule states |
| `upsampling_kernel_size` | 4 | Conv1d projection kernel after molecule repeat |
| `local_transformer_stride` | 128 | initial shallow local-attention window width/stride |
| `layer_norm_eps` | 1e-12 | BERT-style LayerNorm epsilon |
| `use_cache` | true in configs | ignored for this encoder source; no decode cache |

Representative checkpoint sweep:

| Checkpoint | Architecture | H | I | deep layers | total transformer layers | heads | D | max pos | down/up | local stride | task |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| `google/canine-s` | `CanineModel` | 768 | 3072 | 12 | 14 | 12 | 64 | 16384 | 4 / 4 | 128 | feature extraction |
| `google/canine-c` | `CanineModel` | 768 | 3072 | 12 | 14 | 12 | 64 | 16384 | 4 / 4 | 128 | feature extraction |
| `Splend1dchan/canine-s-squad` | `CanineForQuestionAnswering` | 768 | 3072 | 12 | 14 | 12 | 64 | 16384 | 4 / 4 | 128 | span QA |
| `celine98/canine-s-finetuned-sst2` | `CanineForSequenceClassification` | 768 | 3072 | 12 | 14 | 12 | 64 | 16384 | 4 / 4 | 128 | sequence classification |
| `hf-internal-testing/tiny-random-CanineForMultipleChoice` | `CanineForMultipleChoice` | 32 | 37 | 5 | 7 | 4 | 8 | 512 | 4 / 4 | 128 | tiny multiple choice |

## 3a. Family variation traps

- CANINE is character/codepoint-level. Do not allocate a single `[1114112, H]` token embedding table; the source uses multiple hash-bucket embedding tables and concatenates shard embeddings.
- `hidden_size` must be divisible by `num_hash_functions` for hash embedding, and by `num_attention_heads` for attention.
- Base configs expose `use_cache: true`, but `CanineModel.forward` does not consume cache inputs or outputs. DinoML should ignore or reject generation cache mode for this family.
- The source has `CanineLMPredictionHead` helpers but no public `CanineForMaskedLM` class in `__all__`; `final_seq_char_positions` in `ConvProjection` raises. Treat MLM as a gated follow-up, not a base requirement.
- Initial shallow attention is local by construction: non-overlapping `[stride]` chunks, default 128, no CLS-global behavior for the actual `CanineModel` instantiation. The `CanineAttention` class has global-CLS options, but those are disabled in the instantiated local encoder.
- The deep encoder operates on molecule length produced by Conv1d floor output and `[CLS]` reinsertion. For sequence length `S` and rate `R`, Conv1d gives `floor((S - R) / R) + 1` when `S >= R`; then source drops the last molecule and prepends the original CLS state, so molecule length stays `floor(S / R)` for normal `S >= R`.
- The upsample path intentionally repeats `molecules[:, 1:, :]`, then repeats the last molecule `S % R + R` times and concatenates. This can overproduce relative to `S`; the subsequent same-padded Conv1d projection is expected to yield `S` positions. A lowering must preserve the exact slice/repeat/convolution length math.
- Source tensors are sequence-major rank-3 hidden states `[B, S, H]`; Conv1d regions transpose to `[B, H, S]`. Layout translation must be guarded around Conv1d, attention softmax `dim=-1`, LayerNorm last dim, concatenation last dim, and token/molecule sequence axes.
- Checkpoints may set task heads only through `architectures`; the base encoder geometry is otherwise nearly invariant across public configs inspected.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer codepoint ids, modulo/hash arithmetic: `((input_ids + 1) * prime) % num_hash_buckets`.
- Embedding gather for hash shards, character positions, and token types.
- `cat` on last dim for hash shard concat and char/molecule concat; `cat` on sequence dim for CLS/molecule insertion and local attention chunk outputs.
- `transpose(1, 2)` around Conv1d, attention `transpose(1, 2)`, `permute(0, 2, 1, 3)`, `contiguous`, `view`/reshape.
- Slicing: first token, chunk windows, `0:-1`, `1:`, last molecule, QA split.
- `repeat_interleave` on sequence dimension for molecule repeat.
- `ConstantPad1d`, `MaxPool1d`, `squeeze`/`unsqueeze`, `ones`, `zeros`.
- Multiple-choice flatten/unflatten: `[B, C, S] -> [B*C, S]`, logits `[B*C, 1] -> [B, C]`.

Neural network primitives:

- Multi-hash embedding shards: default 8 x `Embedding(16384, 96)` for H=768.
- Position embedding: source uses `Embedding(num_hash_buckets, H)` with runtime `position_ids` sliced from `max_position_embeddings`; admission should guard positions `< num_hash_buckets` for parity with this source.
- Token type embedding: `Embedding(type_vocab_size, H)`.
- LayerNorm over last dim with eps `1e-12`.
- Dense MHA projections with bias: Q/K/V `Linear(H -> H)`, output `Linear(H -> H)`.
- FFN with bias: `Linear(H -> I) -> gelu -> Linear(I -> H)`.
- Conv downsample: `Conv1d(H -> H, kernel_size=R, stride=R, bias=True) -> gelu -> LayerNorm`.
- Conv projection: `ConstantPad1d((floor((K-1)/2), ceil((K-1)/2))) -> Conv1d(2H -> H, kernel_size=K, stride=1, bias=True) -> gelu -> LayerNorm`.
- Pooler: take token 0 from molecule encoder, `Linear(H -> H) -> tanh`.
- Task heads:
  - Sequence classification: dropout elided for inference, `Linear(H -> num_labels)` on pooler output.
  - Multiple choice: same classifier `Linear(H -> 1)`, reshape to `[B, num_choices]`.
  - Token classification: `Linear(H -> num_labels)` over final character states.
  - QA: `Linear(H -> num_labels)`, usually 2, then split/squeeze to start/end `[B, S]`.

Attention primitives:

- Full noncausal self-attention for deep molecule encoder and final character encoder.
- Local noncausal self-attention for initial character encoder, implemented as Python chunk loops over dense attention windows.
- Additive masks use either the standard expanded attention mask or a 3D local mask converted inside `CanineSelfAttention` when rank 3.

Position/custom math:

- Learned absolute character position embeddings; no RoPE, ALiBi, relative bias, rotary, GQA/MQA, MoE, or autoregressive cache.

Preprocessing-coupled ops:

- Unicode codepoint tokenizer and private-use special token ids.
- Token type ids default to all zeros and include special tokens.

## 5. Layer/block breakdown

Embeddings:

```text
input_ids [B,S] int64
hash_i = ((input_ids + 1) * prime_i) % num_hash_buckets
char = cat(Embedding_i(hash_i), dim=-1)  # [B,S,H]
x = char + token_type_embedding + char_position_embedding
x = LayerNorm(x)
```

Initial shallow character encoder, repeated once:

```text
for each non-overlapping local chunk of length up to local_transformer_stride:
  q,k,v = Linear(H -> H)(chunk)
  attn = softmax((q @ k.T) / sqrt(D) + local_mask)
  chunk_out = attn @ v
x_local = cat(chunk_outs, dim=sequence)
x = LayerNorm(Linear(x_local) + input)
x = LayerNorm(Linear(gelu(Linear(x))) + x)
```

Characters to molecules:

```text
cls = char_encoding[:, 0:1, :]
z = transpose(char_encoding, [B,H,S])
z = Conv1d(H -> H, kernel=R, stride=R)(z)
z = gelu(transpose(z, [B,M,H]))
z = cat([cls, z[:, 0:-1, :]], dim=1)
z = LayerNorm(z)
```

Deep molecule encoder, repeated `num_hidden_layers` times:

```text
q,k,v = Linear(H -> H)(z)
attn = softmax((q @ k.T) / sqrt(D) + molecule_mask)
z = LayerNorm(Linear(attn @ v) + z)
z = LayerNorm(Linear(gelu(Linear(z))) + z)
```

Molecules back to characters:

```text
repeated = repeat_interleave(z[:, 1:, :], repeats=R, dim=1)
tail = repeat_interleave(z[:, -1:, :], repeats=(S % R) + R, dim=1)
mol_chars = cat([repeated, tail], dim=1)
concat = cat([initial_char_encoding, mol_chars], dim=-1)
out = ConvProjection(concat)  # padded Conv1d(2H -> H), gelu, LayerNorm
out = final one-layer full character encoder(out, full character mask)
```

## 6. Attention requirements

Primary target uses encoder self-attention only.

| Region | Pattern | Length | Mask | Cache |
|---|---|---|---|---|
| Initial char encoder | local dense MHA per chunk | `S`, chunk width/stride `W` | 3D `[B,S,S]` local slices | none |
| Deep molecule encoder | full noncausal MHA | `M = floor(S/R)` for normal inputs | additive broadcast mask from downsampled attention mask | none |
| Final char encoder | full noncausal MHA | `S` | standard extended attention mask | none |

MHA is standard: Q/K/V widths are all H, `num_attention_heads=A`, `head_dim=H/A`, scores are scaled by `1/sqrt(head_dim)`, mask is added before softmax, and dropout is inference-elided. No packed/varlen attention metadata exists in source. No sliding-window config exists beyond the initial local chunk loop. No KV cache should be admitted for the first DinoML target.

Local attention admission policy:

- Admit only the source-instantiated case first: `local=True`, `always_attend_to_first_position=False`, `first_position_attends_to_all=False`, width equals stride.
- Lower as a sequence of dense chunk attentions, or as a block-local attention kernel with strict guards that output order equals `cat(chunks, dim=1)`.
- Reject or separately audit the generic `CanineAttention` global-CLS options before allowing them.

## 7. Position encoding and custom math

CANINE uses learned absolute position embeddings and hash-bucket codepoint embeddings.

```python
PRIMES = [31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223]

def canine_hash_ids(input_ids, num_hashes=8, num_buckets=16384):
    return [((input_ids + 1) * p) % num_buckets for p in PRIMES[:num_hashes]]
```

Position ids default to `arange(max_position_embeddings)[None, :S]`. Source creates `char_position_embeddings = nn.Embedding(num_hash_buckets, hidden_size)`, while the position id buffer is sized by `max_position_embeddings`. With base configs both are 16384. DinoML should guard `position_ids < num_hash_buckets` rather than assuming arbitrary `max_position_embeddings` is safe if future configs diverge.

Downsample attention mask source pattern:

```text
attention_mask [B,S] -> reshape [B,1,S]
-> MaxPool1d(kernel=R, stride=R) over sequence
-> molecule mask used for deep encoder attention
```

## 8. Preprocessing and input packing

Tokenizer contract:

- Tokenizer class: `CanineTokenizer`.
- It splits Python strings into Unicode characters and converts each character with `ord(token)`.
- Vocab size property is `1114112`, the Unicode codepoint count.
- Special codepoints:
  - `[PAD] = 0`
  - `[CLS] = 0xE000 = 57344`
  - `[SEP] = 0xE001 = 57345`
  - `[BOS] = 0xE002 = 57346`
  - `[MASK] = 0xE003 = 57347`
  - `[RESERVED] = 0xE004 = 57348`
- The tokenizer deliberately has no unknown token. Invalid token/id conversion should fail instead of mapping to unknown.
- `model_input_names = ["input_ids", "attention_mask", "token_type_ids"]`.
- `token_type_ids` default to all zeros and include special tokens.
- `model_max_length` tokenizer default is 2048, but model config position limit is 16384.

GPU graph inputs for first integration:

- `input_ids`: int64 `[B,S]`, values in Unicode/special codepoint range.
- `attention_mask`: float or integer/bool-like `[B,S]`, defaults to ones.
- `token_type_ids`: int64 `[B,S]`, defaults to zeros, must be `< type_vocab_size`.
- Optional `position_ids`: int64 `[1,S]` or broadcast-compatible, must be in position table range.
- Optional `inputs_embeds`: `[B,S,H]` bypasses hash embedding. First DinoML integration can reject it to keep tokenizer/hash ABI explicit.

## 9. Graph rewrite / lowering opportunities

### Rewrite: multi-hash character embedding as fused hash-gather-concat

Source pattern:

```text
for prime in PRIMES[:num_hash_functions]:
  ids_i = ((input_ids + 1) * prime) % num_hash_buckets
  shard_i = Embedding_i(ids_i)
cat(shards, dim=-1)
```

Replacement:

```text
FusedCanineHashEmbedding(input_ids, shard_tables, primes, num_buckets)
```

Preconditions:

- `num_hash_functions <= 16`.
- `hidden_size % num_hash_functions == 0`.
- All shard tables have equal shard width.
- Input ids are integer codepoints; no `inputs_embeds` bypass.

Failure cases: unsupported `num_hash_functions`, divergent shard widths, or external callers supplying `inputs_embeds`.

Parity test sketch: compare fused output to HF `_embed_hash_buckets` on fixed Unicode examples including private-use special ids, zero, BMP, and non-BMP codepoints.

### Rewrite: local chunk attention to block-local dense attention

Source pattern: Python loop slices `[from_start:from_end]` and `[to_start:to_end]`, runs normal MHA per chunk, then concatenates outputs.

Replacement: one block-local attention op over windows of width `W` with output order matching source chunk concatenation.

Preconditions:

- Initial encoder only.
- `attend_from_chunk_width == attend_from_chunk_stride == attend_to_chunk_width == attend_to_chunk_stride`.
- `always_attend_to_first_position == False`.
- `first_position_attends_to_all == False`.
- Noncausal self-attention, no cross-attention.

Shape equations:

- For `S`, chunks are `[i:min(i+W,S)]` for `i=0,W,2W,...`.
- Attention softmax axis is local key length, not global sequence length.

Failure cases: CLS-global options, overlapping windows, custom to/from widths, output attentions requiring a tuple of per-chunk probabilities.

### Rewrite: non-overlap Conv1d downsample to GEMM windows

Source pattern:

```text
transpose [B,S,H] -> [B,H,S]
Conv1d(H -> H, kernel=R, stride=R, padding=0)
transpose -> [B,M,H]
gelu -> drop last -> prepend cls -> LayerNorm
```

Replacement:

```text
WindowFlatten([B,S,H], window=R, stride=R) -> Linear(H*R -> H) -> gelu
```

Preconditions:

- `kernel_size == stride == downsampling_rate`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Source NCL Conv1d flatten order is preserved: weights `[out, in, k]` become linear weight `[out, in*k]`.
- Dynamic sequence lengths must use the exact Conv1d floor output length.

Failure cases: changed Conv1d attrs, non-contiguous hidden layout, or sequence lengths shorter than kernel.

### Rewrite: same-padded Conv1d projection to padded window GEMM

Source pattern:

```text
concat [B,S,2H] -> transpose [B,2H,S]
ConstantPad1d((pad_beg, pad_end), 0)
Conv1d(2H -> H, kernel=K, stride=1)
transpose -> gelu -> LayerNorm
```

Replacement:

```text
PadSequence -> SlidingWindowFlatten(width=K,stride=1) -> Linear(2H*K -> H)
```

Preconditions:

- `groups == 1`, `dilation == 1`, `stride == 1`.
- Padding exactly `pad_beg=(K-1)//2`, `pad_end=(K-1)-pad_beg`.
- Preserve NCL Conv1d flatten order.

Failure cases: layout pass changes channel/sequence axes without rewriting Conv1d weights and pad axes.

### Rewrite: inference dropout elimination

Dropout after embeddings, attention probs, projections, classifier inputs, and projection path can be erased only under inference/eval mode. Training parity is out of scope.

## 10. Kernel fusion candidates

Highest priority:

- Fused hash embedding: integer hash arithmetic plus 8 gathers plus concat is model-specific and appears on every request.
- LayerNorm plus residual for BERT-style blocks: repeated across all encoder regions.
- Dense MHA for molecule and final character encoders: standard full-attention path over `[B,A,S,D]` or `[B,A,M,D]`.
- Local chunk attention: source loop is expensive overhead; block-local kernel avoids per-chunk graph fragmentation.

Medium priority:

- Conv1d downsample/projection lowered to GEMM windows: uses existing GEMM/provider work when guarded.
- FFN `Linear -> gelu -> Linear` with residual+LayerNorm.
- Repeat-interleave plus concat before projection: reduce temporary traffic in the upsample path.
- Task-head fusions: QA `Linear -> split/squeeze`, token classification `Linear` over `[B,S,H]`.

Lower priority:

- Pooler `take token 0 -> Linear -> tanh`.
- Output attentions reconstruction for local attention; should remain optional/debug.
- Training losses and label handling.

## 11. Runtime staging plan

Stage 1: Parse config and tokenizer ABI.

- Load `CanineConfig`.
- Admit base geometry and reject generation/KV cache mode.
- Implement or stub CPU-side Unicode codepoint tokenization contract.

Stage 2: Hash embedding and base tensor ops.

- Implement hash arithmetic, shard embedding gathers, position/type embedding adds, LayerNorm.
- Parity against HF embeddings for short strings.

Stage 3: One local shallow character block.

- Lower initial local attention conservatively as chunked dense attention.
- Guard source-instantiated no-CLS-global options.

Stage 4: Downsample, deep encoder, and projection path.

- Add Conv1d or guarded Conv1d-to-GEMM lowering.
- Implement molecule mask downsampling via MaxPool1d semantics.
- Validate base `CanineModel.last_hidden_state` and pooler.

Stage 5: Task heads.

- Add sequence classification, token classification, multiple choice, and QA heads.
- Keep MLM gated until a public/source-supported class is audited.

Stage 6: Optimize.

- Fuse hash embedding, block-local attention, Conv1d GEMM rewrites, residual/LayerNorm, and FFN pieces.

## 12. Parity and validation plan

- Tokenizer/codepoint tests: ASCII, multilingual BMP, non-BMP characters, `[CLS]`, `[SEP]`, `[MASK]`, no-unknown failure behavior.
- Hash embedding tests: compare `CanineEmbeddings._hash_bucket_tensors` and full embedding output with fixed ids and token type ids.
- Local attention tests: one shallow layer, `S < W`, `S == W`, `S > W`, and tail chunk shorter than W.
- Downsample mask tests: attention masks with padded tails, all-ones masks, and partial windows.
- Downsample/upsample tests: sequence lengths with `S % R` in `{0,1,2,3}` for default `R=4`.
- One deep layer parity: random hidden states and molecule mask.
- Full base model parity: `last_hidden_state` `[B,S,H]` and `pooler_output` `[B,H]`.
- Head parity:
  - sequence classification logits `[B,num_labels]`
  - multiple choice logits `[B,num_choices]`
  - token classification logits `[B,S,num_labels]`
  - QA start/end logits `[B,S]`
- Suggested tolerances: fp32 absolute/relative around `1e-4` for composed full-model parity; fp16/bf16 require looser per-stage tolerances and should first validate fused kernels against unfused DinoML references.

## 13. Performance probes

- Tokenization throughput: Unicode char split/codepoint conversion and special-token packing.
- Hash embedding throughput: batch/sequence sweep over `S={128,512,2048,4096,16384}`.
- Initial local attention throughput: sweep `S` and `local_transformer_stride`, report chunk count and tail chunk cost.
- Molecule encoder throughput: compare cost at `M=floor(S/4)` versus full character attention.
- Downsample/projection Conv1d versus GEMM-window rewrite.
- Repeat-interleave/concat temporary memory volume in the upsample path.
- End-to-end encoder throughput by batch and sequence length.
- Head-only overhead for token classification and QA on long character sequences.
- Memory probes: activation footprint for `S=16384`, especially final full character encoder attention, which is quadratic in `S`.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- `CanineForMaskedLM` / MLM parity: no exported class in inspected source, and `final_seq_char_positions` raises.
- Autoregressive generation, prefill/decode, and KV cache: not source behavior.
- Generic `CanineAttention` global-CLS local-attention modes.
- `inputs_embeds` path unless needed by a downstream integration.
- `output_attentions=True` fast-path parity for local attention; source returns a tuple of per-chunk probabilities, not a single dense attention tensor.
- Layout translation across Conv1d or attention axes until a guarded pass proves axis rewrites.

## 15. Final implementation checklist

- [ ] Parse `CanineConfig` and admit supported geometry.
- [ ] Load tokenizer/codepoint special-token ABI.
- [ ] Implement multi-hash embedding gather/concat.
- [ ] Implement BERT-style embedding add + LayerNorm.
- [ ] Implement source-instantiated local chunk self-attention.
- [ ] Implement full noncausal MHA for molecule and final character encoders.
- [ ] Implement Conv1d downsample and same-padded Conv1d projection.
- [ ] Implement molecule attention-mask downsampling.
- [ ] Implement repeat-interleave upsample path with exact tail behavior.
- [ ] Implement pooler.
- [ ] Implement sequence classification, multiple choice, token classification, and QA heads.
- [ ] Add parity tests for `S % downsampling_rate` cases.
- [ ] Add block-local attention parity tests.
- [ ] Add graph rewrite guards for Conv1d-to-GEMM and local attention.
- [ ] Benchmark hash embedding, local attention, Conv1d rewrites, and end-to-end encoder throughput.

Gated gaps for DinoML admission:

- General local-attention options with CLS-global behavior are not admitted.
- MLM is not admitted from this source basis.
- KV cache/generation is rejected despite `use_cache` appearing in configs.
- Full character encoder attention at long `S` may be impractical without an optimized attention backend or staged sequence limits.
- Position table safety requires guarding `position_ids < num_hash_buckets` because the source position embedding table is sized by `num_hash_buckets`.
