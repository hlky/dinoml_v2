# Transformers Audit Report: xlstm

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: primarily NX-AI/xLSTM-7b; representative configs listed below
Config source: local configuration_xlstm.py plus HF config.json files
Source files inspected:
- X:/H/transformers/src/transformers/models/xlstm/configuration_xlstm.py
- X:/H/transformers/src/transformers/models/xlstm/modeling_xlstm.py
Any missing files or assumptions:
- No local checkpoint weights were inspected.
- The source conditionally delegates block implementation to the external `xlstm` package when `is_xlstm_available()` is true. This report scopes DinoML operator planning to the pinned in-library fallback implementation, and flags external/Triton kernel configs as an admission trap.
- No tokenizer source is model-specific, but NX-AI/xLSTM-7b tokenizer/generation configs were inspected for special-token ABI.
```

Representative config evidence is summarized in [config_sweep.md](./config_sweep.md). Config URLs inspected include [NX-AI/xLSTM-7b](https://huggingface.co/NX-AI/xLSTM-7b/raw/main/config.json), [ethicalabs/xLSTM-7b-Instruct](https://huggingface.co/ethicalabs/xLSTM-7b-Instruct/raw/main/config.json), [J4bb4wukis/xlstm_247m_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/xlstm_247m_wikipedia_en_shuffeld/raw/main/config.json), [J4bb4wukis/xlstm_406m_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/xlstm_406m_wikipedia_en_shuffeld/raw/main/config.json), [stefan-it/xlstm-transformers-bug-native](https://huggingface.co/stefan-it/xlstm-transformers-bug-native/raw/main/config.json), and [anrilombard/sallm-xlstm-125m](https://huggingface.co/anrilombard/sallm-xlstm-125m/raw/main/config.json).

Primary runtime target for this report: `xLSTMForCausalLM` text generation, including prefill-like recurrent sequence execution and decode step execution. `xLSTMModel` hidden-state output is required as the backbone. Training loss and hidden-state capture are deferred.

## 2. High-level architecture

xLSTM is a text-only recurrent causal language model. It is not a Transformer attention decoder: there is no self-attention, cross-attention, RoPE, ALiBi, or KV cache in the inspected in-library implementation. Each block applies RMSNorm, an mLSTM layer with fixed-size recurrent state, residual add, RMSNorm, gated SiLU feed-forward, and residual add.

```text
token ids -> embedding -> repeated xLSTM blocks with recurrent state -> output RMSNorm -> LM head -> soft-capped logits
```

Stage decomposition:

```text
CPU/tokenizer: GPTNeoX-style tokenization and special-token policy
GPU/runtime prefill: embedding + N recurrent blocks over [B, S] input
State update: per-layer fixed-size C/N/M tensors mutated after each call
Decode: one-token recurrent step with same C/N/M state ABI
Logits: dense LM projection + output soft cap
Generation controller: sampling/stop rules outside model graph
```

Independent validation stages: token embedding and LM-head orientation, one mLSTM backend step, one full xLSTM block, recurrent cache update across chunks, full logits for short prompts, then token-by-token decode equivalence against full-prefix execution.

## 3. Important config dimensions

Pinned source defaults from `xLSTMConfig`:

| Field | Default / behavior | Operator significance |
|---|---:|---|
| `vocab_size` | 50304 | Embedding rows and LM-head output width |
| `hidden_size` | 4096 | Residual width `H`; defaults from `embedding_dim` if omitted |
| `embedding_dim` | `hidden_size` | Input embedding width |
| `num_hidden_layers` / `num_blocks` | 32 | Number of recurrent blocks/states |
| `num_heads` | 8 | mLSTM head count |
| `qk_dim_factor` | 0.5 | Q/K width `round_up(H * factor, 64)` in config property |
| `v_dim_factor` | 1.0 | V/output gate width `round_up(H * factor, 64)` in config property |
| `qk_head_dim` | `qk_dim / num_heads` | C-state row dimension |
| `v_head_dim` | `v_dim / num_heads` | C-state column and per-head output dimension |
| `chunk_size` | 64 | Chunkwise recurrent kernel tile; source also handles arbitrary lengths |
| `ffn_proj_factor` | 2.667 | Gated FFN up width |
| `ffn_round_up_to_multiple_of` | 64 | FFN up-width rounding |
| `use_bias` | false | Dense projections mostly biasless; i/f gates always have bias |
| `weight_mode` | `single` | Separate Q/K/V/O/gates/FFN projections; `fused` changes weight packing |
| `gate_soft_cap` | 15.0 | `cap * tanh(gate / cap)` before recurrent gates |
| `output_logit_soft_cap` | 30.0 | Same soft cap applied after LM head |
| `norm_reduction_force_float32` | true | RMSNorm and multi-head LayerNorm reduce in fp32 |
| `inference_state_dtype` | `float32` | Sequence/step state math dtype |
| `autocast_kernel_dtype` | `bfloat16` | Kernel dtype hint; fallback mostly uses tensor dtype plus fp32 states |
| `use_cache` | true | Enables `xLSTMCache`, not KV cache |

Representative checkpoint sweep:

| Model id | H | Layers | Heads | qk/v head dims | FFN up | Vocab | Kernels | Tied embeddings |
|---|---:|---:|---:|---|---:|---:|---|---|
| `NX-AI/xLSTM-7b` | 4096 inferred from `embedding_dim` | 32 | 8 | 256 / 512 | 10944 | 50304 | Triton strings in config | false |
| `ethicalabs/xLSTM-7b-Instruct` | 4096 | 32 | 8 | 256 / 512 | 10944 | 50560 | native strings | false |
| `stefan-it/xlstm-transformers-bug-native` | 512 | 16 | 4 | 64 / 128 | 1408 | 50304 | native strings, `mode=train` | false |
| `J4bb4wukis/xlstm_247m_wikipedia_en_shuffeld` | 768 inferred | 24 | 4 | 96 / 192 | 2112 | 50257 | native strings | false |
| `J4bb4wukis/xlstm_406m_wikipedia_en_shuffeld` | 1024 inferred | 24 | 4 | 128 / 256 | 2752 | 50257 | native strings | false |
| `anrilombard/sallm-xlstm-125m` | 768 | 12 | 4 | 96 / 192 | 2112 | 65536 | native strings | true |

## 3a. Family variation traps

- External package dispatch: if `xlstm` is installed, Transformers imports `xlstm.xlstm_large.model.mLSTMBlock`, `RMSNorm`, and `soft_cap`; otherwise it uses the fallback implementation in `modeling_xlstm.py`. DinoML should choose one source basis per artifact and reject un-audited external-only behavior.
- Kernel config strings vary between `chunkwise--native_autograd`, `chunkwise--triton_xl_chunk`, `native_sequence__native`, `native_sequence__triton`, and `step_kernel=triton`. The fallback source hard-wires native Python/Torch kernels, so Triton strings in configs are not enough to define DinoML runtime behavior.
- Configs contain historical fields not read by the pinned fallback modeling path, including `add_embedding_dropout`, `add_forward_backend_padding`, `add_post_blocks_norm`, `add_post_norm`, `add_qk_norm`, `cell_norm_eps`, `force_bos_token_insert`, `head_dim`, `igate_bias_init_range`, and `mlstm_round_up_to_multiple_of`.
- `hidden_size` and `num_hidden_layers` may be omitted in older configs; the config class fills them from `embedding_dim` and `num_blocks`.
- `head_dim` in some configs appears to match `v_head_dim`, not Q/K head dim. DinoML should derive Q/K and V widths from `qk_dim_factor`, `v_dim_factor`, `num_heads`, and source-compatible rounding.
- `weight_mode=fused` is implemented in source but not seen in the representative configs. It changes Q/K/V/O and FFN/gate weight layouts.
- `use_bias=false` does not remove i/f gate biases. The input and forget gate projections are always created with `bias=True`.
- Tokenizer/vocab varies: official 7B uses vocab 50304 with GPTNeoX tokenizer config; fine-tunes may use 50560 or 65536; Wikipedia variants use GPT-2-like ids/vocab 50257.
- `tie_word_embeddings=true` appears in `anrilombard/sallm-xlstm-125m`; official 7B configs use untied embeddings. Weight aliasing must be preserved when tied.
- There is no positional embedding or attention mask in the model forward. Sequence order is carried by recurrent state update, not by position ids.
- Layout-sensitive ops are mostly `[B,S,H] <-> [B,heads,S,D]` reshapes/transposes and per-head normalization. A layout pass must not reorder the temporal axis across recurrent updates.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup: `input_ids [B,S] int64 -> inputs_embeds [B,S,H]`.
- Dense reshape/view: `[B,S,qk_dim] -> [B,S,heads,qk_head_dim]`, then transpose to `[B,heads,S,qk_head_dim]`.
- Dense transpose: `[B,heads,S,D] <-> [B,S,heads,D]`.
- Slicing/chunking along sequence axis for `max_inference_chunksize`, chunkwise recurrence, and tail handling.
- `contiguous`, tensor copy into cache state, copy into preallocated final-state output for long inference.
- `tensor_split` for fused projection modes.
- `tril`, boolean mask, `where(..., -inf)` for chunkwise fallback.

Neural primitives:

- `nn.Linear(H -> qk_dim)`, `nn.Linear(H -> v_dim)`, `nn.Linear(H -> num_heads)` gate projections, `nn.Linear(v_dim -> H)`, `nn.Linear(H -> ffn_up_dim)`, `nn.Linear(ffn_up_dim -> H)`, `nn.Linear(H -> vocab)`.
- For 7B: Q/K `4096 -> 2048`, V/O-gate/out `4096 -> 4096`, i/f gates `4096 -> 8`, FFN two up projections `4096 -> 10944`, FFN down `10944 -> 4096`, LM head `4096 -> 50304` or `50560`.
- RMSNorm over last dimension for block pre-norm and output norm.
- Multi-head LayerNorm over per-head value dimension, then flatten to `[B,S,v_dim]`.
- SiLU, sigmoid, tanh soft cap, logsigmoid, exp, max/maximum, abs, reciprocal sqrt, mean, variance, sum, cumsum, division, multiplication, residual add.

Attention primitives:

- No ordinary attention required. No QK softmax attention, no attention mask, no KV cache, no RoPE.
- The mLSTM chunkwise fallback includes matmul patterns similar to attention score/value products, but with exponential gates and recurrent C/N/M state rather than a dense attention probability matrix.

Recurrent/state-space cache ops:

- Per layer cache state tuple:
  - `C`: `[max_batch, heads, qk_head_dim, v_head_dim]`
  - `N`: `[max_batch, heads, qk_head_dim]`
  - `M`: `[max_batch, heads, 1]`
- State dtype: cache allocated with model/input dtype by `xLSTMCache`, but sequence/step update casts state math to `config.inference_state_dtype` in fallback inference.
- State update is fixed-size and mutates in place after each block call by copying returned state into `cache_params.rnn_state[layer_idx][state_idx]`.
- `seqlen_offset` increments by current input sequence length, but fallback recurrent math does not consume absolute positions.
- Reset zeroes all C/N/M states.

Generation/cache ops:

- `use_cache=True` creates an `xLSTMCache` if absent.
- Decode can run with `S=1` through `mlstm_recurrent_step_native`, avoiding full-prefix recompute.
- Cache reorder for beam search is not implemented in `xLSTMCache`; DinoML should reject or implement explicit batch-index reorder before beam parity.

Quantized/packed weight metadata ops:

- None in the inspected source. All weights are standard PyTorch dense tensors. Source-coupled Triton kernels are execution providers, not quantized storage.

Preprocessing-coupled ops:

- Text tokenization only. No image/audio/video preprocessing, no packed varlen metadata, no scatter/stitch operation.

## 5. Layer/block breakdown

Backbone:

```text
input_ids [B,S] -> embeddings [B,S,H]
repeat num_blocks:
  y = RMSNorm(x)                                      # last dim H
  y, (C,N,M) = xLSTMLayer(y, previous_state)
  x = x + y
  z = RMSNorm(x)
  z = FFN(z)
  x = x + z
x = output RMSNorm(x)
logits = Linear(H -> vocab)(x).float()
logits = soft_cap(logits, output_logit_soft_cap)
```

xLSTMLayer, `weight_mode=single`:

```text
query = Linear(H -> qk_dim, bias=use_bias)(x)
key   = Linear(H -> qk_dim, bias=use_bias)(x)
value = Linear(H -> v_dim,  bias=use_bias)(x)
o     = Linear(H -> v_dim,  bias=use_bias)(x)
i     = soft_cap(Linear(H -> heads, bias=True)(x), gate_soft_cap)
f     = soft_cap(Linear(H -> heads, bias=True)(x), gate_soft_cap)

query,key: [B,S,qk_dim] -> [B,heads,S,qk_head_dim]
value:     [B,S,v_dim]  -> [B,heads,S,v_head_dim]
i,f:       [B,S,heads]  -> [B,heads,S]

h, state = mLSTM(query, key, value, i, f, state)
h: [B,heads,S,v_head_dim] -> [B,S,heads,v_head_dim]
h_norm = per-head LayerNorm(h) -> [B,S,v_dim]
y = Linear(v_dim -> H, bias=use_bias)(sigmoid(o) * h_norm)
```

FFN:

```text
single mode: y = Linear(ffn_up -> H)(silu(Linear(H -> ffn_up)(x)) * Linear(H -> ffn_up)(x))
fused mode:  gate,z = split(Linear(H -> 2*ffn_up)(x)); y = Linear(ffn_up -> H)(silu(gate) * z)
```

Fused xLSTMLayer packing, if admitted later:

```text
qkv_opreact = Linear(H -> 2*qk_dim + 2*v_dim)
split order: query, key, value, o_preact
ifgate_preact = Linear(H -> 2*heads, bias=True)
split order: input gate, forget gate
```

## 6. Attention requirements

No attention is required for the primary target. The `query`, `key`, and `value` names belong to mLSTM algebra, not Transformer attention. There is no causal mask argument, attention mask processing, packed/varlen attention metadata, GQA/MQA repeat, sliding window, RoPE, ALiBi, or KV cache.

mLSTM requirements instead:

- Sequence input shapes: `Q/K [B,heads,S,qk_head_dim]`, `V [B,heads,S,v_head_dim]`, `i/f [B,heads,S]`.
- State input/output shapes: `C [B,heads,qk_head_dim,v_head_dim]`, `N [B,heads,qk_head_dim]`, `M [B,heads,1]`.
- Chunkwise path requires chunk multiples internally, but inference wrapper handles arbitrary `S` by running full chunks plus recurrent tail or single-step.
- Stateful decode stores fixed-size C/N/M states after all gate and projection math. States do not grow with sequence length.
- The chunkwise fallback constructs an intra-chunk lower-triangular mask and performs stabilized exp/logsum-style math; this is not equivalent to softmax attention.

FlashAttention/SDPA compatibility: not directly applicable. Some prefill math can be custom-kernelized with GEMM and scan-like kernels, but it is a distinct provider family.

## 7. Position encoding and custom math

There is no position encoding. The recurrence order and cache state provide sequence dependence.

Soft cap:

```python
def soft_cap(x, cap):
    return x if cap is None else cap * tanh(x / cap)
```

Single-step mLSTM update, simplified from source:

```python
f_log = logsigmoid(f_gate)
m_new = maximum(f_log + m_old, i_gate)
f_act = exp(f_log + m_old - m_new)
i_act = exp(i_gate - m_new)

q_scaled = q * (qk_head_dim ** -0.5)
C_new = f_act[..., None] * C_old + i_act[..., None] * (k[..., :, None] @ v[..., None, :])
N_new = f_act * N_old + i_act * k
num = (q_scaled[..., None, :] @ C_new).squeeze(-2)
den = maximum(abs((q_scaled[..., None, :] @ N_new[..., :, None]).squeeze(-1)), exp(-m_new)) + eps
h = num / den
```

Precomputable: dense weights and static shape metadata. Dynamic: gate values, C/N/M states, chunk/tail path selection, and all recurrent updates.

## 8. Preprocessing and input packing

The model graph consumes either `input_ids [B,S]` or `inputs_embeds [B,S,H]`; exactly one must be supplied. There are no `position_ids`, `attention_mask`, token type IDs, cu-seqlens, image grids, or modality placeholders consumed by the model forward.

For `NX-AI/xLSTM-7b`, tokenizer evidence shows `GPTNeoXTokenizer`; `generation_config.json` has `bos_token_id=0`, `eos_token_id=2`, `pad_token_id=1`, while tokenizer config maps BOS/EOS/UNK content to `<|endoftext|>` and has no normal pad token string in `special_tokens_map.json`. This should be treated as generation-controller/tokenizer ABI, not neural graph work.

`force_bos_token_insert` appears in some configs but is not read by the pinned model source. If end-to-end text parity depends on it, handle it in tokenization/generation policy and verify against the exact tokenizer implementation.

## 9. Graph rewrite / lowering opportunities

### Rewrite: recurrent decode step specialization

Source pattern: `S=1` inference path through `mlstm_recurrent_step_native`.

Replacement pattern: one fused recurrent-step kernel per layer:

```text
Q/K/V/gates projections -> mLSTMStep(C,N,M) -> per-head norm -> output gate multiply -> out projection
```

Preconditions:

- `sequence_length == 1`
- cache state allocated for current batch and layer count
- no hidden-state capture
- supported `weight_mode`

Shape equations:

- `Q/K [B,Hd,Dq]`, `V [B,Hd,Dv]`
- `C [B,Hd,Dq,Dv]`, `N [B,Hd,Dq]`, `M [B,Hd,1]`
- `h [B,Hd,Dv] -> [B,1,v_dim]`

Failure cases: beam reorder, partial batch reset, unsupported dtype/state dtype, external package divergence.

Parity test sketch: compare one-token continuation logits and updated C/N/M states against Transformers after a short prefill.

### Rewrite: separate projections -> grouped/fused GEMM

Source pattern: in `weight_mode=single`, Q/K/V/O-gate/i-gate/f-gate are six projections from the same normalized input.

Replacement pattern: packed GEMM or grouped GEMM, then split outputs in source order.

Preconditions:

- Same input tensor and compatible bias policy.
- Preserve gate bias even when `use_bias=false`.
- Packed output order: Q, K, V, O, I, F if using one local packed projection; source has separate modules, so weight transform must be explicit.

Weight transform:

```python
W = concat([W_q, W_k, W_v, W_o, W_i, W_f], dim=0)
b = concat([b_q_or_0, b_k_or_0, b_v_or_0, b_o_or_0, b_i, b_f], dim=0)
```

Failure cases: weight aliasing, `weight_mode=fused`, quantized per-weight metadata, gate bias initialization assumptions.

### Rewrite: fused FFN SwiGLU-style block

Source pattern: `silu(up_gate(x)) * up(x) -> down`.

Replacement pattern: fused `gemm_bias_silu_mul` plus GEMM down, or packed two-up projection.

Preconditions:

- `weight_mode=single` or source-supported fused split order.
- Static `ffn_up_dim`.
- Bias policy preserved.

### Rewrite: output last-token logits

Source pattern: LM head produces `[B,S,V]` then generation usually consumes the final step.

Replacement pattern: when generation controller only needs next token, project `hidden_states[:, -1:, :]`.

Preconditions:

- No caller requests full-sequence logits.
- Loss is absent.
- Soft cap applied after projection on the reduced logits.

### Rewrite: chunkwise mLSTM provider

Source pattern: chunkwise recurrence builds C/N/M chunk states, lower-triangular intra-chunk gates, QK matmuls, and value matmuls.

Replacement pattern: dedicated mLSTM provider with prefill kernels, tail recurrence, and state ABI.

Preconditions:

- `mode=inference`
- fixed `chunk_size` or profiled choices
- supported state dtype and qk/v head dims
- sequence axis preserved

Failure cases: training mode, hidden-state capture with chunked long inference path, external Triton kernel exactness not audited.

## 10. Kernel fusion candidates

Highest priority:

- mLSTM recurrent step kernel: decode performance depends on fixed-size state update, not KV-cache attention.
- mLSTM prefill/chunkwise provider: fallback PyTorch loops and chunk materialization are too expensive for long prompts.
- RMSNorm and per-head LayerNorm: both are on every block and currently require fp32 reductions for parity.
- Packed/grouped same-input projections: Q/K/V/O/gates and FFN up projections are GEMM-heavy and share input activations.

Medium priority:

- Gate soft-cap + gate projection epilogue: `15 * tanh(x/15)` before i/f recurrence.
- Output logit soft cap fused with LM head or post-LM elementwise.
- FFN SiLU multiply fusion.
- Last-token-only LM head for generation.

Lower priority:

- Training loss path and label shift.
- Chunked `max_inference_chunksize` host-loop optimization after base state ABI is stable.
- External Triton parity, if DinoML wants to match installed `xlstm` package behavior rather than the fallback.

## 11. Runtime staging plan

Stage 1: parse config and load dense weights for `weight_mode=single`, native fallback semantics only. Reject external-only/Triton behavior and unsupported historical fields that alter topology.

Stage 2: implement embedding, RMSNorm, Linear/GEMM, SiLU/sigmoid/tanh soft-cap, residual adds, and LM head for a no-cache single block with small random tensors.

Stage 3: implement mLSTM single-step state update as a bounded custom op/provider and validate state output.

Stage 4: implement full model decode with explicit `xLSTMCache` C/N/M tensors and no beam reorder.

Stage 5: implement sequence/prefill path using recurrent loop first for parity, then chunkwise provider for throughput.

Stage 6: add packed projection and FFN rewrites behind exact weight-transform guards.

Stage 7: add selected checkpoint support: official 7B native/fallback route, then community vocab/tied-embedding variants, then optional external/Triton parity if still desired.

Initially stub or reject: training loss, hidden-state output capture, beam cache reorder, `weight_mode=fused`, external package block implementation, and Triton kernel config strings.

## 12. Parity and validation plan

- Config parser tests: omitted `hidden_size`/`num_hidden_layers` defaults; derive qk/v/ffn widths for 512, 768, 1024, and 4096 hidden configs.
- Soft cap tests: fp32 and bf16/fp16 storage, cap `None`, cap 15, cap 30.
- RMSNorm and per-head LayerNorm tests: force-fp32 reduction path, eps `1e-6`, with/without bias.
- mLSTM step tests: random `Q/K/V/i/f/C/N/M`, compare h and updated states against fallback source with tolerances around `1e-5` fp32, `5e-3` bf16/fp16 depending on accumulation.
- One xLSTMLayer parity: compare output and C/N/M state for `S=1` and short `S>1`.
- Chunk/tail tests: `S=1`, `S=15`, `S=16`, `S=64`, `S=65`, and `S > max_inference_chunksize` if host chunking is implemented.
- Block parity: after 1, 2, and N blocks on small configs.
- Decode parity: prefill a prompt, run one token at a time, compare logits and cache states to Transformers using identical inputs.
- End-to-end text parity: official tokenizer prompt -> logits for `NX-AI/xLSTM-7b` or an accessible small/debug checkpoint.
- Tied-embedding test for `anrilombard/sallm-xlstm-125m` style config: preserve logical alias between embedding and LM head when source weights are tied.

## 13. Performance probes

- Embedding + LM-head bandwidth for vocab sizes 50257, 50304, 50560, 65536.
- Decode tokens/sec by batch size for fixed C/N/M state sizes.
- State memory per layer and total: for 7B, each layer stores `C [B,8,256,512]`, `N [B,8,256]`, `M [B,8,1]`.
- Prefill sequence-length sweep: `S=1, 16, 64, 256, 1024, 4096, 16384`.
- Chunk-size sensitivity around source default 64.
- Native recurrent loop versus chunkwise provider.
- Packed projections versus separate GEMMs.
- RMSNorm/per-head LayerNorm standalone bandwidth.
- Last-token-only logits versus full `[B,S,V]` logits.
- State dtype comparison: cache dtype bf16/fp16 versus fp32 update policy.

## 14. Skip/defer list

- Training and CrossEntropy loss.
- Gradient checkpointing and output hidden-state capture.
- Beam search cache reorder until `xLSTMCache` batch-index semantics are explicitly designed.
- `weight_mode=fused` until single mode is validated.
- External `xlstm` package and Triton kernels until their source/version is separately audited.
- Quantization and packed weight formats; none are required by current source.
- Multi-GPU/tensor parallelism.
- Layout translation that reorders temporal state updates.
- Tokenizer policy beyond passing token ids and respecting BOS/EOS/PAD in generation.

## 15. Final implementation checklist

- [ ] Parse `xLSTMConfig`, including omitted `hidden_size`/`num_hidden_layers` defaults.
- [ ] Add admission policy for fallback-native source basis versus external `xlstm` package behavior.
- [ ] Load dense embedding, block, norm, gate, FFN, and LM-head weights.
- [ ] Preserve tied embedding/LM-head alias when `tie_word_embeddings=true`.
- [ ] Implement token embedding lookup for int input ids.
- [ ] Implement/fuse RMSNorm with fp32 reduction.
- [ ] Implement/fuse per-head LayerNorm over `[B,S,heads,Dv]`.
- [ ] Implement soft cap `cap * tanh(x / cap)` for gates and logits.
- [ ] Implement mLSTM recurrent step with C/N/M state ABI.
- [ ] Implement cache allocate/reset/update for all layers.
- [ ] Add decode parity tests for state and logits.
- [ ] Implement sequence prefill recurrent loop for parity.
- [ ] Add chunkwise mLSTM provider or optimized scan/chunk kernels.
- [ ] Add projection packing rewrite for Q/K/V/O/i/f with gate-bias guards.
- [ ] Add FFN SiLU-multiply fusion.
- [ ] Add last-token-only LM-head rewrite for generation.
- [ ] Benchmark decode, prefill, state memory, norm kernels, and LM-head projection.

