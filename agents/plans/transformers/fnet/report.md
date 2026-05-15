# Transformers FNet Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary target: google/fnet-base.
  Additional configs: google/fnet-large,
  gchhablani/fnet-base-finetuned-sst2,
  gchhablani/fnet-large-finetuned-mnli,
  hf-internal-testing/tiny-random-FNetModel.

Config source:
  https://huggingface.co/google/fnet-base/raw/main/config.json
  https://huggingface.co/google/fnet-large/raw/main/config.json
  https://huggingface.co/gchhablani/fnet-base-finetuned-sst2/raw/main/config.json
  https://huggingface.co/gchhablani/fnet-large-finetuned-mnli/raw/main/config.json
  https://huggingface.co/hf-internal-testing/tiny-random-FNetModel/raw/main/config.json
  Snapshots saved under agents/plans/transformers/fnet/_sources/.

Source files inspected:
  transformers/src/transformers/models/fnet/modeling_fnet.py
  transformers/src/transformers/models/fnet/configuration_fnet.py
  transformers/src/transformers/models/fnet/tokenization_fnet.py
  transformers/src/transformers/activations.py
  transformers/tests/models/fnet/test_modeling_fnet.py

Any missing files or assumptions:
  No remote-code files are required for the in-library FNet family. This report
  targets encoder and masked-LM/fill-mask parity first. Sequence classification,
  token classification, QA, multiple choice, NSP, and pretraining heads are
  documented as staged heads. Training losses, dropout behavior in train mode,
  and gradient checkpointing are deferred.
```

## 2. High-level architecture

FNet is a text-only bidirectional encoder that replaces BERT self-attention with
a Fourier token mixer. There is no autoregressive decoder, no causal mask, and
no KV cache. The default CPU/GPU source path applies an FFT across the sequence
and hidden axes, takes the real part, then follows a normal residual LayerNorm
and feed-forward block.

```text
SentencePiece/Unigram tokenization + [CLS]/[SEP]
  -> word + token_type + absolute position embeddings
  -> LayerNorm + Linear(H -> H) embedding projection
  -> repeated FNet encoder layers:
       real(FFT over seq and hidden axes) -> residual LayerNorm
       -> Linear(H -> I) -> GELU/NewGELU -> Linear(I -> H)
       -> dropout -> residual LayerNorm
  -> optional pooler on token 0
  -> masked-LM / pretraining / classifier / token / QA head
```

Primary runtime path:

```text
input_ids, optional token_type_ids/position_ids
  -> FNetModel encoder
  -> MLM transform + tied vocab projection
  -> logits [B, S, vocab_size]
```

Independently stageable units are tokenizer/input packing, embedding path, one
Fourier mixer, one FFN block, the repeated encoder loop, pooler, and task heads.

## 3. Important config dimensions

Worked example: `google/fnet-base`.

| Field | Value | Source |
|---|---:|---|
| model_type | fnet | config/source |
| vocab_size | 32000 | config |
| hidden_size | 768 | config |
| num_hidden_layers | 12 | config |
| intermediate_size | 3072 | config |
| hidden_act | `gelu_new` | config |
| hidden_dropout_prob | 0.1 | config |
| max_position_embeddings | 512 | config |
| type_vocab_size | 4 | config |
| layer_norm_eps | 1e-12 | config |
| use_tpu_fourier_optimizations | false | config/source |
| tpu_short_seq_length | 512 | config |
| torch_dtype | float32 | config metadata |
| pad/bos/eos token ids | 3 / 1 / 2 | config |
| tie_word_embeddings | true | source default |
| attention/cache support | not applicable | source |

Representative checkpoint sweep:

| Model id | Arch | H | I | Layers | Vocab | Type vocab | Max pos | Act | Task/head |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `google/fnet-base` | `FNetForPreTraining` | 768 | 3072 | 12 | 32000 | 4 | 512 | `gelu_new` | MLM + NSP |
| `google/fnet-large` | `FNetForPreTraining` | 1024 | 4096 | 24 | 32000 | 4 | 512 | `gelu_new` | MLM + NSP |
| `gchhablani/fnet-base-finetuned-sst2` | `FNetForSequenceClassification` | 768 | 3072 | 12 | 32000 | 4 | 512 | `gelu_new` | 2-label classifier |
| `gchhablani/fnet-large-finetuned-mnli` | `FNetForSequenceClassification` | 1024 | 4096 | 24 | 32000 | 4 | 512 | `gelu_new` | 3-label classifier |
| `hf-internal-testing/tiny-random-FNetModel` | `FNetModel` | 32 | 37 | 5 | 32000 | 16 | 512 | `gelu` | debug encoder |

Config fields observed in older official configs but not read by the inspected
source include `actual_seq_length`, `use_fft`, and `use_latest`; DinoML should
record them as ignored metadata for this source basis.

## 3a. Family variation traps

- FNet has no attention mask input in `FNetModel.forward`. Padding tokens are
  mixed by the Fourier transform like any other token; do not silently add a
  BERT-style attention-mask path.
- The Fourier transform is axis-sensitive: source shape is `[B, S, H]`, and the
  default path computes `torch.fft.fftn(x, dim=(1, 2)).real`. Layout translation
  must preserve the semantic sequence and hidden axes.
- TPU optimization changes operator structure. With
  `use_tpu_fourier_optimizations=True`, short sequences may use precomputed
  complex DFT matrices and `einsum`; otherwise source uses axis-wise FFT loops.
  For first CUDA integration, reject or route this flag separately.
- `tpu_short_seq_length` must equal runtime `seq_length` when TPU optimization
  is enabled and `seq_length <= 4096`.
- Base and large checkpoints share topology but differ in `hidden_size`,
  `intermediate_size`, and layer count. There are no head counts because there
  is no attention.
- Debug configs may use `hidden_act="gelu"` rather than official `gelu_new`.
- MLM/pretraining heads tie decoder weights to input word embeddings and alias
  decoder bias to the explicit prediction bias.
- Multiple-choice flattens `[B, C, S]` to `[B*C, S]`, runs the encoder, then
  reshapes logits to `[B, C]`.

## 4. Operator coverage checklist

Required for encoder + masked LM:

- Tensor/layout ops: rank/shape validation, embedding/index lookup, addition,
  broadcasted position ids `[1, S]`, token-type id expansion `[1, S] -> [B, S]`,
  first-token slice `hidden[:, 0]`, optional hidden-state tuple output,
  `split/squeeze/contiguous` only for QA head.
- Neural primitives: `Embedding(vocab_size -> H)`,
  `Embedding(max_position_embeddings -> H)`,
  `Embedding(type_vocab_size -> H)`, `LayerNorm(H, eps=1e-12)`,
  `Linear(H -> H)` embedding projection, `Linear(H -> I)`,
  `gelu_new` or `gelu`, `Linear(I -> H)`, dropout as identity in inference.
- Fourier/token-mixing primitives: complex FFT over axes `(S, H)`, real-part
  extraction, or a guarded dense DFT-matrix fallback for the TPU path.
- Masked-LM head: `Linear(H -> H)`, activation, `LayerNorm(H)`,
  tied `Linear(H -> vocab_size)` with bias.
- Pooler/classification heads: token-0 slice, `Linear(H -> H)`, `tanh`,
  dropout identity, classifier `Linear(H -> num_labels)` or `Linear(H -> 1)`.
- QA head: `Linear(H -> num_labels)`, split along last dim, squeeze last dim.
- Attention primitives: none.
- Position encoding: learned absolute position embedding table only; no RoPE,
  ALiBi, relative bias, or sinusoidal math.
- Generation/cache ops: not applicable.
- Preprocessing-coupled ops: SentencePiece/Albert tokenizer style special-token
  construction and token-type ids, outside the GPU graph.

Current DinoML gated gaps from the local checklist:

- No public FFT/complex tensor primitive is listed; FNet encoder parity is
  gated on a bounded real-output 2D FFT or a source-equivalent DFT rewrite.
- General embedding/model helpers are still checklist gaps; FNet needs three
  embedding lookups plus tied embedding output projection.
- `LayerNorm` family is still unported in the checklist; FNet has multiple
  LayerNorms per layer and in the heads.
- `gelu_new` is listed as an embedding/model-helper gap even though `gelu`
  and `fast_gelu` exist; official FNet uses the tanh cubic NewGELU formula.

## 5. Layer/block breakdown

Embedding block:

```text
input_ids [B, S] or inputs_embeds [B, S, H]
token_type_ids [B, S], default zeros
position_ids [1, S], default arange

x = word_embedding(input_ids) or inputs_embeds
x = x + token_type_embedding(token_type_ids)
x = x + position_embedding(position_ids)
x = LayerNorm(x)
x = Linear(H -> H, bias=True)(x)
x = Dropout(x)  # inference identity
```

Encoder layer, repeated `num_hidden_layers` times:

```text
mix = real(fftn(x, dim=(1, 2)))       # [B, S, H]
x1 = LayerNorm(x + mix)
h = Linear(H -> I, bias=True)(x1)
h = activation(h)
h = Linear(I -> H, bias=True)(h)
h = Dropout(h)                        # inference identity
x = LayerNorm(x1 + h)
```

Official base uses `H=768`, `I=3072`, `N=12`; official large uses `H=1024`,
`I=4096`, `N=24`.

Masked-LM head:

```text
h = Linear(H -> H, bias=True)(sequence_output)
h = activation(h)
h = LayerNorm(h)
logits = Linear(H -> vocab_size, tied word embedding weight, bias=True)(h)
```

Pooler and classifier heads:

```text
pooled = tanh(Linear(H -> H)(sequence_output[:, 0]))
logits = Linear(H -> num_labels)(dropout(pooled))
```

## 6. Attention requirements

No attention is required for the primary target. FNet does not implement
self-attention, cross-attention, causal masks, packed attention metadata,
sliding windows, FlashAttention/SDPA, or KV caches. The only sequence mixing in
the encoder is the Fourier transform over the full sequence and hidden axes.

Important admission rule: `attention_mask` is not a source argument for
`FNetModel.forward`, so FNet parity should reject graph assumptions that depend
on masked attention or padding-aware token exclusion.

## 7. Position encoding and custom math

Position encoding is a learned absolute embedding lookup:

```python
position_ids = arange(max_position_embeddings)[None, :seq_length]
x = x + position_embeddings(position_ids)
```

Default Fourier mixer:

```python
def fnet_fourier_mix(x):
    # x: [batch, seq, hidden], real floating dtype
    return torch.fft.fftn(x, dim=(1, 2)).real
```

Source fallback axis-wise FFT path:

```python
def fnet_axis_fft(x):
    out = x
    for axis in reversed(range(x.ndim)[1:]):
        out = torch.fft.fft(out, axis=axis)
    return out.real
```

TPU short-sequence DFT path:

```python
def fnet_dft_matmul(x, dft_seq, dft_hidden):
    seq = x.shape[1]
    x = x.to(torch.complex64)
    return torch.einsum("bij,jk,ni->bnk", x, dft_hidden, dft_seq[:seq, :seq]).real
```

NewGELU used by official configs:

```python
def gelu_new(x):
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x**3)))
```

The DFT matrices are constant for a fixed hidden size and short sequence length,
but the source slices the sequence matrix to the runtime `seq_length`.

## 8. Preprocessing and input packing

FNet uses an Albert-derived SentencePiece/Unigram tokenizer. Tokenizer metadata
for `google/fnet-base` records `tokenizer_class="FNetTokenizer"`,
`model_max_length=512`, `do_lower_case=false`, `keep_accents=true`, and special
tokens `<unk>`, `[SEP]`, `<pad>`, `[CLS]`, `[MASK]`.

The model graph consumes:

- `input_ids [B, S]` int token ids, or mutually exclusive `inputs_embeds [B,S,H]`.
- Optional `token_type_ids [B, S]`; if absent, source expands a zero buffer.
- Optional `position_ids [1 or B, S]`; if absent, source slices a registered
  arange buffer.

There is no model-side attention mask. Padding and segment ids are tokenizer/data
pipeline responsibilities, but padding positions still enter the Fourier mix.

## 9. Graph rewrite / lowering opportunities

### Rewrite: FNet default Fourier mixer as bounded provider op

Source pattern:

```text
real(torch.fft.fftn(x, dim=(1, 2)))
```

Replacement:

```text
fnet_fft2_real(x, seq_axis=1, hidden_axis=2)
```

Preconditions:

- Input rank exactly 3, layout `[B, S, H]`.
- Floating real input dtype; output dtype matches source real component policy.
- Axes are exactly sequence and hidden axes. No layout pass may move them
  without rewriting the op attrs.
- First integration can require static `S` and `H`; dynamic `B` is safe.

Shape equations:

```text
input [B, S, H] -> output [B, S, H]
```

Failure cases:

- TPU DFT path enabled.
- Non-contiguous or layout-translated tensors without an explicit FFT layout
  contract.
- Missing complex scratch/FFT provider.

Parity test sketch:

- Random `[B,S,H]` tensors for small odd/even `S,H`, compare to
  `torch.fft.fftn(x, dim=(1,2)).real` in fp32.

### Rewrite: TPU DFT path to two constant GEMMs

Source pattern:

```text
einsum("bij,jk,ni->bnk", x_complex, dft_hidden, dft_seq[:S,:S]).real
```

Replacement:

```text
complex cast -> matmul hidden DFT -> matmul seq DFT -> real
```

Preconditions:

- `use_tpu_fourier_optimizations=True`.
- DFT matrices are present and match `hidden_size` and `tpu_short_seq_length`.
- Runtime `S <= 4096` and equals `tpu_short_seq_length` per source validation.

Weight transform:

```text
dft_hidden [H,H] complex64, dft_seq [S,S] complex64
```

Failure cases:

- SciPy-unavailable checkpoint construction can choose the axis-wise FFT path
  instead; DinoML should not assume DFT matrices exist unless weights contain
  them or config admission builds them deterministically.

### Rewrite: inference dropout removal

Preconditions:

- Module in inference/eval mode.

Replacement:

```text
Dropout(x) -> identity(x)
```

### Rewrite: multiple-choice batch flatten

Source pattern:

```text
input_ids [B,C,S] -> view [B*C,S] -> encoder -> classifier -> view [B,C]
```

Preconditions:

- `num_choices` is known from input dim 1.
- Flatten/view preserves row-major choice order.

## 10. Kernel fusion candidates

Highest priority:

- Real-output 2D FFT provider for `[B,S,H]`. This is the defining operator and
  the largest nonstandard gap versus BERT-like encoders.
- LayerNorm + residual add. Every layer has two residual LayerNorms; the
  embedding/head LayerNorms make this unavoidable for parity.
- FFN GEMM + NewGELU + GEMM. Official configs are dense `H -> 4H -> H` blocks
  and should map well to existing CUTLASS GEMM epilogues once `gelu_new` is
  admitted.

Medium priority:

- Embedding sum + LayerNorm + projection fusion for fixed `[B,S]` token inputs.
- MLM head transform + tied vocab projection, with last-mile support for tied
  weight aliases and bias aliasing.
- Token-0 pooler slice + `Linear + tanh` for sequence classifiers.

Lower priority:

- TPU DFT-matrix path using complex GEMMs. It is source behavior but not used by
  the official GPU/CPU configs inspected.
- Hidden-state tuple materialization for `output_hidden_states=True`.

## 11. Runtime staging plan

Stage 1: parse FNet configs and load weights for `FNetModel`, preserving tied
word embedding/LM decoder aliases in metadata.

Stage 2: implement or stub embedding lookup, LayerNorm, `gelu_new`, and one
FFN block; validate all non-Fourier pieces against a tiny config by replacing
Fourier mix with an identity test hook.

Stage 3: add bounded `fnet_fft2_real` provider or CPU reference op for static
`S,H`; run one-layer and full-encoder parity for `FNetModel`.

Stage 4: add `FNetForMaskedLM` and pretraining MLM head, including tied decoder
weight and decoder bias alias behavior.

Stage 5: add pooler and sequence classification heads, then token
classification, QA, multiple choice, and NSP as needed.

Stage 6: evaluate FFT provider performance, residual LayerNorm fusion, and FFN
GEMM epilogue fusion.

Initial stubs can omit training losses, dropout randomness, TPU Fourier
optimizations, output hidden states, and all non-primary heads.

## 12. Parity and validation plan

- Custom op parity: compare `fnet_fft2_real` to PyTorch for fp32 random tensors
  with `S,H` values such as `(6,32)`, `(7,37)`, `(32,768)`, and `(512,768)` if
  feasible.
- Embedding parity: fixed `input_ids`, omitted and explicit `token_type_ids`,
  omitted and explicit `position_ids`.
- One-layer parity: run a tiny FNet layer with dropout disabled and compare
  hidden states after Fourier residual and after FFN residual.
- Full encoder parity: `hf-internal-testing/tiny-random-FNetModel` first, then
  `google/fnet-base` for short sequences.
- MLM parity: compare logits `[B,S,32000]`, especially with tied decoder
  weights.
- Classification parity: `gchhablani/fnet-base-finetuned-sst2` pooled logits.
- QA parity: verify split/squeeze returns start/end logits `[B,S]`.
- Recommended tolerances: fp32 FFT path should target about `rtol=1e-4,
  atol=1e-4` initially; reduced precision should be deferred until provider
  error is characterized.

## 13. Performance probes

- FFT-only throughput across `S` and `H`, separately from FFN.
- Encoder layer throughput for base and large shapes.
- End-to-end masked-LM throughput for sequence lengths 32, 128, 512.
- Batch-size sweep for FFT provider scratch allocation and launch overhead.
- Compare default FFT provider versus DFT-matrix GEMM fallback for short
  sequences only if TPU optimization support is admitted.
- LayerNorm/residual fusion benchmark.
- FFN GEMM candidate profiling with `gelu_new` epilogue versus separate
  activation kernel.
- Weight-load and tied-weight alias memory accounting for MLM head.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Dropout randomness; inference treats dropout as identity.
- TPU Fourier optimizations and SciPy DFT matrix creation.
- `output_hidden_states=True` materialization unless a caller requires it.
- NSP, QA, token classification, multiple choice, and sequence classification
  until encoder + MLM parity is stable.
- Quantization and GGUF/offload policy; no source-specific packed weight format
  was observed.
- Any attention-mask, causal generation, or KV-cache work; not part of FNet.

## 15. Final implementation checklist

- [ ] Parse `FNetConfig`, including ignored historical fields as metadata.
- [ ] Load word, position, token-type, LayerNorm, projection, FFN, and head weights.
- [ ] Preserve MLM/pretraining tied word embedding decoder weight alias.
- [ ] Implement embedding lookup and embedding sum.
- [ ] Implement/admit `LayerNorm(H, eps=1e-12)`.
- [ ] Implement/admit `gelu_new`.
- [ ] Implement bounded `fnet_fft2_real([B,S,H], axes=(1,2))`.
- [ ] Add layout guard that protects `[B,S,H]` Fourier axes from channel-last rewrites.
- [ ] Add one-layer FNet parity test.
- [ ] Add full `FNetModel` tiny-config parity test.
- [ ] Add `google/fnet-base` short-sequence encoder parity test.
- [ ] Add `FNetForMaskedLM` tied-head parity test.
- [ ] Add staged classifier/QA/token/multiple-choice head parity tests.
- [ ] Benchmark FFT-only and encoder-layer throughput.
- [ ] Keep TPU DFT path rejected unless a separate complex-matmul admission lands.

