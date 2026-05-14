# RWKV Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Model id:
- `RWKV/rwkv-4-169m-pile`
- `RWKV/rwkv-4-430m-pile`
- `RWKV/rwkv-4-1b5-pile`
- `RWKV/rwkv-4-3b-pile`
- `RWKV/rwkv-4-7b-pile`

Config source:
- Official Hugging Face `config.json` snapshots saved under `agents/plans/transformers/rwkv/_sources/`.
- Raw URLs used:
  - `https://huggingface.co/RWKV/rwkv-4-169m-pile/raw/main/config.json`
  - `https://huggingface.co/RWKV/rwkv-4-430m-pile/raw/main/config.json`
  - `https://huggingface.co/RWKV/rwkv-4-1b5-pile/raw/main/config.json`
  - `https://huggingface.co/RWKV/rwkv-4-3b-pile/raw/main/config.json`
  - `https://huggingface.co/RWKV/rwkv-4-7b-pile/raw/main/config.json`

Source files inspected:
- `X:/H/transformers/src/transformers/models/rwkv/configuration_rwkv.py`
- `X:/H/transformers/src/transformers/models/rwkv/modeling_rwkv.py`
- `X:/H/transformers/src/transformers/models/rwkv/convert_rwkv_checkpoint_to_hf.py`
- `X:/H/transformers/src/transformers/generation/utils.py` for generation cache/state handling.

Any missing files or assumptions:
- The optimized WKV implementation is loaded from the external Hub kernel `kernels-community/rwkv` through `get_kernel(...)`; its CUDA source was not vendored in the Transformers checkout.
- This report targets inference for `RwkvForCausalLM`. Training, backward kernels, and checkpoint conversion are out of runtime scope except where they reveal weight naming and layout.
- DinoML assumptions: inference-only first, CUDA GPU target, faithful PyTorch axis semantics, and explicit recurrent state artifacts rather than hidden mutable model state.

## 2. High-level architecture

RWKV is a text-only causal language model with transformer-like residual blocks but no self-attention matrix, KV cache, RoPE, ALiBi, or attention mask in the runtime graph. Each block combines:

```text
token ids -> embedding -> repeated RWKV blocks -> final LayerNorm -> LM head -> logits/sampling
```

Each RWKV block is:

```text
optional block-0 pre-LayerNorm
LayerNorm -> time-mix + WKV recurrent channel attention -> output projection -> residual
LayerNorm -> time-mix + squared-ReLU channel mix -> value projection -> residual
```

Generation decomposes into:

```text
prefill prompt tokens -> recurrent state list[5] -> one-token decode loop -> logits/sampling
```

The independently stageable unit is the recurrent state. Unlike KV-cache models, decode does not append keys/values over sequence length. It mutates five state tensors of shape `[batch, hidden_size, num_layers]`, so memory is `O(batch * hidden * layers)` instead of `O(batch * seq * layers * heads * head_dim)`.

## 3. Important config dimensions

Config defaults from `RwkvConfig`:

| Field | Default | Runtime meaning |
|---|---:|---|
| `vocab_size` | 50277 | token embedding rows and LM head rows |
| `context_length` | 1024 | max multi-token CUDA WKV kernel length; RNN mode can continue beyond this by carrying state |
| `hidden_size` | 4096 | residual width |
| `num_hidden_layers` | 32 | recurrent block count |
| `attention_hidden_size` | `hidden_size` if omitted | WKV key/value/receptance width |
| `intermediate_size` | `4 * hidden_size` if omitted | channel-mix MLP width |
| `layer_norm_epsilon` | `1e-5` | all LayerNorms |
| `rescale_every` | 6 | inference-time weight/activation rescale cadence |
| `tie_word_embeddings` | `False` | config default and representative checkpoints |
| `use_cache` | `True` | returns recurrent state, not KV tensors |

Representative checkpoint sweep from official `config.json`:

| Model id | Layers | Hidden | Attention hidden | Intermediate | Context | Vocab | Dtype | Rescale |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `RWKV/rwkv-4-169m-pile` | 12 | 768 | 768 | 3072 | 1024 | 50277 | `float32` | 6 |
| `RWKV/rwkv-4-430m-pile` | 24 | 1024 | 1024 | 4096 | 1024 | 50277 | `float32` | 6 |
| `RWKV/rwkv-4-1b5-pile` | 24 | 2048 | 2048 | 8192 | 1024 | 50277 | `float32` | 6 |
| `RWKV/rwkv-4-3b-pile` | 32 | 2560 | 2560 | 10240 | 1024 | 50277 | `float32` | 6 |
| `RWKV/rwkv-4-7b-pile` | 32 | 4096 | 4096 | 16384 | 1024 | 50277 | `float32` | 6 |

All inspected configs use `attention_hidden_size == hidden_size`, `intermediate_size == 4 * hidden_size`, `use_cache=true`, and `tie_word_embeddings=false`.

## 3a. Family variation traps

- RWKV has no attention heads. Do not infer `num_attention_heads`, `head_dim`, KV-cache tensors, causal masks, RoPE, or FlashAttention requirements.
- `attention_mask` is accepted by the model API but logged as unused. Padding parity must be handled outside the core RWKV graph or by prompt construction.
- The recurrent cache is named `state` in generation utilities. DinoML should treat it as a model-specific recurrent state, not as `past_key_values`.
- `context_length` limits the optimized CUDA WKV multi-token kernel; one-token decode deliberately falls back to the scalar CPU-style recurrence even on CUDA because the source says the kernel is slower for length 1.
- `time_decay` and `time_first` are kept in fp32 by `_keep_in_fp32_modules`, and WKV math uses fp32 exponentials/states.
- Inference toggles `_rescale_layers()`, mutating selected projection weights and dividing hidden states after every `rescale_every` blocks. Artifact loading needs a deterministic policy: either materialize already-rescaled inference weights or reproduce the exact runtime scaling.
- The config class supports `attention_hidden_size != hidden_size` and `intermediate_size != 4 * hidden_size` even though inspected checkpoints do not vary them.
- `head.weight` is logically tied to embeddings by `_tied_weights_keys`, but `tie_word_embeddings` is false in configs. Treat actual checkpoint aliases carefully rather than assuming a clone or tie.

## 4. Operator coverage checklist

Tensor/layout ops:
- Token embedding lookup `[batch, seq] -> [batch, seq, hidden]`.
- `ZeroPad2d((0, 0, 1, -1))` used as a one-token temporal shift over `[batch, seq, hidden]`.
- State slice/read/write at `state[i][:, :, layer_id]`.
- Concatenation/chunk/unsqueeze/squeeze inside optimized WKV state packing when calling the external kernel.
- Last-token / arbitrary-index logits slicing through `logits_to_keep`.

Neural network primitives:
- LayerNorm over hidden dimension, including block-0 `pre_ln`.
- Bias-free linear projections:
  - attention key/value/receptance: `hidden_size -> attention_hidden_size`
  - attention output: `attention_hidden_size -> hidden_size`
  - channel-mix key: `hidden_size -> intermediate_size`
  - channel-mix receptance: `hidden_size -> hidden_size`
  - channel-mix value: `intermediate_size -> hidden_size`
  - LM head: `hidden_size -> vocab_size`
- Elementwise mix: `hidden * mix + shifted * (1 - mix)`.
- Sigmoid gates.
- Squared ReLU: `square(relu(linear(x)))`.
- Residual adds.

Attention primitives:
- No dot-product attention. Required custom primitive is WKV recurrent linear attention over `[batch, seq, attention_hidden_size]`.

Generation/cache ops:
- Allocate five state tensors `[batch, hidden_size, num_layers]`.
- Dtypes: state 0 and 1 use embedding dtype; state 2, 3, and 4 use fp32.
- Initialize state 4 by subtracting `1e30`; per-layer WKV fallback initializes max state to `-1e38` when no state is supplied.
- Generation update must carry `outputs.state` back as `model_kwargs["state"]`.

Preprocessing-coupled ops:
- Tokenization only; representative checkpoints use GPT-NeoX style vocabulary size 50277 from the converter's default tokenizer path. No image/audio/processor tensors.

## 5. Layer/block breakdown

`RwkvModel`:

```text
input_ids or inputs_embeds
embeddings: [B, S] -> [B, S, H]
for layer i in 0..L-1:
  hidden, state = RwkvBlock_i(hidden, state)
ln_out(hidden)
```

`RwkvBlock_i`:

```text
if i == 0:
  hidden = pre_ln(hidden)

a_in = ln1(hidden)
attention_out, state = RwkvSelfAttention_i(a_in, state)
hidden = hidden + attention_out

ffn_in = ln2(hidden)
ffn_out, state = RwkvFeedForward_i(ffn_in, state)
hidden = hidden + ffn_out

if inference rescale is active and (i + 1) % rescale_every == 0:
  hidden = hidden / 2
```

`RwkvSelfAttention_i`:

```text
shifted = previous token hidden, from time shift or state[1][:, :, i]
k_in = hidden * time_mix_key + shifted * (1 - time_mix_key)
v_in = hidden * time_mix_value + shifted * (1 - time_mix_value)
r_in = hidden * time_mix_receptance + shifted * (1 - time_mix_receptance)
key = Linear(H -> AH, bias=False)(k_in)
value = Linear(H -> AH, bias=False)(v_in)
receptance = sigmoid(Linear(H -> AH, bias=False)(r_in))
rwkv = WKV(time_decay[AH], time_first[AH], key[B,S,AH], value[B,S,AH], per-layer state)
out = Linear(AH -> H, bias=False)(receptance * rwkv)
state[1][:, :, i] = hidden[:, -1]
state[2:5][:, :, i] = WKV num/den/max state
```

`RwkvFeedForward_i`:

```text
shifted = previous token hidden, from time shift or state[0][:, :, i]
k_in = hidden * time_mix_key + shifted * (1 - time_mix_key)
r_in = hidden * time_mix_receptance + shifted * (1 - time_mix_receptance)
key = square(relu(Linear(H -> I, bias=False)(k_in)))
value = Linear(I -> H, bias=False)(key)
receptance = sigmoid(Linear(H -> H, bias=False)(r_in))
out = receptance * value
state[0][:, :, i] = hidden[:, -1]
```

For inspected checkpoints, `AH == H` and `I == 4H`.

## 6. Attention requirements

No standard attention is required. The RWKV "attention" module is a causal recurrent WKV operator:

- causal by recurrence order;
- self-only, no cross-attention;
- no MHA/MQA/GQA heads;
- no attention mask in source behavior;
- no packed/varlen metadata in the model source;
- no sliding-window/local attention;
- no RoPE/ALiBi/relative bias;
- no KV cache and no FlashAttention/SDPA compatibility target.

Cache shape for generation:

```text
state[0]: [B, H, L] previous FFN hidden mix input, dtype hidden dtype
state[1]: [B, H, L] previous attention hidden mix input, dtype hidden dtype
state[2]: [B, H, L] WKV numerator state, fp32
state[3]: [B, H, L] WKV denominator state, fp32
state[4]: [B, H, L] WKV max/log state, fp32
```

For a single layer call, WKV consumes `state[2:5][:, :, layer_id]` as three `[B, AH]` tensors. The optimized kernel packs them as `[B, AH, 3]`, mutates them, then chunks them back. If future configs use `attention_hidden_size != hidden_size`, DinoML must reconcile that with the source's global state allocation shape `[B, hidden_size, L]`; current official configs avoid the mismatch.

Decode behavior:
- The generic generation utility recognizes `"state"` as a cache name and copies `outputs.state` into model kwargs.
- Assisted generation is rejected for stateful models such as RWKV because the state cannot be rewound to arbitrary accepted-token prefixes.
- Beam search expands tensor kwargs by repeating along batch. A DinoML runtime that supports beams must implement equivalent state batch gather/reorder semantics or reject beams initially.

## 7. Position encoding and custom math

RWKV has no explicit position encoding. Temporal order enters through the time shift and WKV recurrence.

Core WKV fallback math, simplified from source:

```python
def rwkv_wkv(time_decay, time_first, key, value, state):
    num, den, max_state = state
    decay = -exp(time_decay)
    for t in range(seq_len):
        k = key[:, t].float()
        v = value[:, t]

        out_max = maximum(max_state, k + time_first)
        e_prev = exp(max_state - out_max)
        e_cur = exp(k + time_first - out_max)
        out = (e_prev * num + e_cur * v) / (e_prev * den + e_cur)

        state_max = maximum(max_state + decay, k)
        e_prev = exp(max_state + decay - state_max)
        e_cur = exp(k - state_max)
        num = e_prev * num + e_cur * v
        den = e_prev * den + e_cur
        max_state = state_max
    return outputs, (num, den, max_state)
```

Precomputable:
- `-exp(time_decay)` can be precomputed per layer for inference if weights are fixed and dtype policy is explicit.
- `time_first`, `time_mix_*`, and projection weights are constants.

Dynamic:
- `key`, `value`, and shifted hidden inputs depend on sequence and recurrent state.
- WKV max-trick states are data-dependent and must remain fp32 for parity.

## 8. Preprocessing and input packing

Runtime inputs:
- `input_ids: [B, S]` or `inputs_embeds: [B, S, H]`.
- `attention_mask` may be passed by generation/tokenizer code but RWKV ignores it in `RwkvModel.forward`.
- No `position_ids`, token type IDs, image/audio/video tensors, or packed sequence descriptors are used by RWKV source.

CPU/data-pipeline work:
- Tokenization, BOS/EOS handling, truncation/padding policy, and sampling controller.
- Because the core model ignores `attention_mask`, padded prompt batches must be made parity-safe before reaching the graph, for example by prompt-length grouping or explicit prefill scheduling.

GPU/runtime work:
- Embedding lookup, recurrent block execution, state initialization/update, final logits.
- `logits_to_keep` allows last-token-only or selected-token LM head evaluation; first integration should support at least `0` and `1`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: temporal shift to state read/write in decode

Source pattern:

```text
if S == 1 and state is not None:
  shifted = state[k][:, :, layer]
else:
  shifted = ZeroPad2d(...)(hidden); shifted[:, 0] = state[k][:, :, layer]
```

Replacement:

```text
DecodeStateRead -> elementwise time_mix -> DecodeStateWrite(hidden[:, -1])
```

Preconditions:
- `seq_len == 1`.
- `state` is supplied and has shape `[B, H, L]`.
- State write occurs after all reads that need previous hidden for that layer.

Failure cases:
- Prefill with `seq_len > 1` needs full sequence shift semantics.
- State omitted means no decode cache is active.

Parity test sketch:
- Compare one block run on `S=1` with supplied state against Transformers for random tensors and fixed weights.

### Rewrite: WKV prefill kernel

Source pattern:

```text
rwkv_linear_attention(time_decay, time_first, key[B,S,H], value[B,S,H], state)
```

Replacement:

```text
custom_wkv_scan(time_decay_fp32, time_first_fp32, key, value, initial_state) -> output, final_state
```

Preconditions:
- Preserve recurrence order over `S`.
- Accumulate `num`, `den`, and `max_state` in fp32.
- Match source initialization for absent state.
- For optimized kernel parity, enforce source constraints: CUDA tensors, `S <= context_length`, and `B * H` divisible by `min(H, 32)`.

Failure cases:
- Falling back to an unrolled Python-style loop for long prompts will be too slow.
- Changing exp/max order can produce visible drift, especially in fp16/bf16.

Parity test sketch:
- Random `time_decay`, `time_first`, key/value, and initial state; compare custom scan to source CPU fallback in fp32 and mixed precision.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = head(hidden_states[:, -1:, :])  # when logits_to_keep == 1
```

Replacement:

```text
SliceLastToken -> GEMM(H -> vocab)
```

Preconditions:
- Caller does not request full-sequence logits or loss.
- Sampling only needs final token logits.

Failure cases:
- Training/loss path and prompt-logprob APIs need more logits.

### Rewrite: inference rescale materialization

Source pattern:

```text
attention.output.weight /= 2 ** floor(layer_id / rescale_every)
feed_forward.value.weight /= same
hidden /= 2 after every rescale_every blocks
```

Replacement:

```text
materialize rescaled weights at load time + explicit hidden divide checkpoints
```

Preconditions:
- Inference mode fixed.
- Quantized weights either unsupported initially or dequantized/rescaled with the same policy as source.

Failure cases:
- Training/eval toggles mutate weights in source; DinoML artifacts should reject training-mode toggles or bake one mode.

## 10. Kernel fusion candidates

Highest priority:
- WKV recurrent scan kernel for prefill and state-returning prefill. This is the family-defining bottleneck and cannot be expressed as normal attention.
- One-token WKV decode fused update. The source falls back to scalar recurrence for one token; DinoML should provide a small decode kernel that updates `num/den/max` and emits WKV without Python overhead.
- Time-mix + linear projections. Three attention projections share the same shifted/hidden mix pattern, and two FFN projections share another.

Medium priority:
- LayerNorm + time-mix input preparation per subblock.
- Squared-ReLU channel mix: GEMM -> ReLU -> square -> GEMM with receptance sigmoid multiply.
- Last-token-only LM head GEMM for generation.
- State read/write fusion around decode blocks to avoid repeated slicing and stores.

Lower priority:
- Block-level fusion across residual adds and rescale divides.
- External `kernels-community/rwkv` compatibility shim. Useful for parity comparison, but DinoML should own an artifact-visible provider contract rather than call an opaque Hub kernel directly.
- Quantized bitsandbytes rescale behavior; not required for first dense checkpoints.

## 11. Runtime staging plan

Stage 1: config and weight loading.
- Parse `RwkvConfig`, enforce `model_type == "rwkv"`, load embeddings, blocks, final norm, and head.
- Support inspected dense fp32 checkpoints first.

Stage 2: single-block parity.
- Implement LayerNorm, time shift/mix, linear projections, sigmoid, squared ReLU, residual adds, and CPU/reference WKV scan.
- Validate one block with and without state.

Stage 3: full prefill parity.
- Run full prompt sequences with WKV scan and final state output.
- Initially allow fixed `S <= context_length`; document ignored attention mask.

Stage 4: decode with recurrent state.
- Add explicit state input/output ABI for five tensors.
- Optimize `S == 1` path and support last-token logits.

Stage 5: optimized CUDA WKV provider.
- Add a DinoML WKV provider manifest with dtype, max sequence, state layout, fp32 accumulation, launch ABI, and fallback status.
- Include compile-visible fallback/reject behavior when shape constraints are not satisfied.

Stage 6: graph rewrites/fusions.
- Canonicalize time shift to state operations in decode.
- Fuse time-mix/projection and channel-mix patterns where tests show parity.

Stage 7: generation integration.
- Implement sampling loop state carry, batch expansion/gather for simple beams if needed, and explicit rejection of assisted/speculative generation.

## 12. Parity and validation plan

- WKV unit tests:
  - fp32 random scan with no initial state.
  - fp32 random scan with initial `num/den/max` state.
  - fp16/bf16 inputs with fp32 state, comparing source fallback tolerances.
  - one-token decode equals length-one prefill with the same incoming state.
- Time-mix tests:
  - sequence shift path with `S > 1`.
  - state read path with `S == 1`.
  - first-token replacement from supplied state.
- Single-layer parity:
  - `RwkvBlock` for 169M dimensions at small synthetic shapes.
  - state update for all five state tensors.
- Full-model parity:
  - prefill logits for short prompts against Transformers.
  - decode next-token logits after prefill state.
  - full generation greedy token parity for a small prompt.
- Rescale parity:
  - compare inference outputs with `rescale_every=6` after materialized rescale policy.
  - reject or separately test `rescale_every <= 0`.
- Recommended tolerances:
  - fp32: `rtol=1e-4`, `atol=1e-4` for logits after short prompts; tighter for isolated ops.
  - fp16/bf16: start with `rtol=5e-3`, `atol=5e-3`, then tighten after WKV kernel is stable.

## 13. Performance probes

- WKV prefill throughput by `B`, `S`, and `H`: isolate recurrence kernel time from projections.
- Decode tokens/sec with recurrent state for `B=1`, small batch, and batched decode.
- Compare source-style one-token fallback recurrence versus DinoML decode kernel.
- Projection GEMM throughput per block: attention three projections, output projection, FFN two projections.
- State memory bandwidth: read/write five `[B,H,L]` tensors per token.
- Last-token-only versus full-sequence LM head.
- End-to-end prompt length sweep: `S=1`, `16`, `128`, `1024`.
- Batch-size sweep, especially source kernel constraint `B * H % min(H, 32) == 0`.
- Rescale materialization overhead at load time and no per-token overhead during decode.

No benchmark measurements were taken for this docs-only audit.

## 14. Skip/defer list

- Training and WKV backward kernels.
- Gradient checkpointing.
- Bitsandbytes 4-bit/8-bit rescale compatibility.
- Assisted/speculative generation, explicitly unsupported by source generation utilities for stateful models.
- Beam search until state repeat/gather/reorder semantics are implemented and validated.
- Attention mask semantics; source ignores it, so first integration should document or reject padded batch cases that would depend on masks.
- Remote/custom RWKV variants outside the in-library `rwkv` source.
- Non-dense quantized checkpoint loading.
- Configs with `attention_hidden_size != hidden_size` until state shape semantics are audited.

## 15. Final implementation checklist

- [ ] Parse `RwkvConfig` and representative dense checkpoint configs.
- [ ] Load embeddings, block weights, final LayerNorm, and LM head.
- [ ] Preserve or materialize inference `rescale_every` behavior.
- [ ] Implement explicit recurrent `state[5]` ABI with `[B, H, L]` layout and mixed dtype policy.
- [ ] Implement time-shift/time-mix for prefill and decode.
- [ ] Implement WKV reference scan with fp32 max-trick state.
- [ ] Add CUDA WKV provider manifest and launch ABI.
- [ ] Add optimized one-token decode WKV update.
- [ ] Lower bias-free linear projections to GEMM.
- [ ] Implement squared-ReLU channel mix and sigmoid gates.
- [ ] Support `logits_to_keep=1` last-token LM head.
- [ ] Warn/reject attention-mask-dependent padded batches.
- [ ] Reject assisted/speculative generation.
- [ ] Add WKV unit parity tests.
- [ ] Add single-block parity tests.
- [ ] Add prefill logits parity tests.
- [ ] Add decode token/logit parity tests with carried state.
- [ ] Benchmark WKV prefill, WKV decode, projection GEMMs, state bandwidth, and LM head slicing.
