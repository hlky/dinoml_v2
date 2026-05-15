# Decision Transformer audit for DinoML

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: decision_transformer; representative public checkpoints under edbeeching/*
Config source: local Transformers config class plus Hugging Face config.json snapshots saved beside this report
Source files inspected:
- transformers/src/transformers/models/decision_transformer/modeling_decision_transformer.py
- transformers/src/transformers/models/decision_transformer/configuration_decision_transformer.py
- transformers/src/transformers/models/decision_transformer/__init__.py
- transformers/src/transformers/pytorch_utils.py for Conv1D weight layout
- transformers/src/transformers/masking_utils.py for causal/bidirectional mask construction
Any missing files or assumptions:
- No tokenizer, processor, feature extractor, or remote-code file is model-coupled for this family.
- This report targets inference parity for DecisionTransformerModel action/state/return prediction, not training or language generation.
- Raw Hub fetches for edbeeching/decision-transformer-gym-hopper-medium-expert and hf-internal-testing/tiny-random-DecisionTransformerModel returned HTTP 401. They are not used as source basis.
```

Pinned source URLs:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/decision_transformer/modeling_decision_transformer.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/decision_transformer/configuration_decision_transformer.py

Saved config snapshots:

- `edbeeching__decision-transformer-gym-hopper-medium.config.json`
- `edbeeching__decision-transformer-gym-hopper-medium-replay.config.json`
- `edbeeching__decision-transformer-gym-hopper-expert.config.json`
- `edbeeching__decision-transformer-gym-halfcheetah-medium.config.json`
- `edbeeching__decision-transformer-gym-halfcheetah-medium-replay.config.json`
- `edbeeching__decision-transformer-gym-halfcheetah-expert.config.json`
- `edbeeching__decision-transformer-gym-walker2d-medium.config.json`
- `edbeeching__decision-transformer-gym-walker2d-medium-replay.config.json`
- `edbeeching__decision-transformer-gym-walker2d-expert.config.json`

## 2. High-level architecture

Decision Transformer is a GPT-2-style causal transformer over offline RL trajectory tokens. It does not consume tokenized text in the primary model path. The model linearly embeds `returns_to_go`, `states`, and `actions`, adds a learned timestep embedding to each, interleaves them in `(return_t, state_t, action_t)` order, runs a causal transformer over the packed sequence, then predicts next return/state from action-token hidden states and action from state-token hidden states.

```text
CPU/data pipeline trajectory tensors
  -> modality linear projections + timestep embedding
  -> stack/permute/reshape to [B, 3T, H]
  -> embed LayerNorm
  -> GPT-2 causal blocks
  -> reshape to [B, 3, T, H]
  -> return/state/action heads
```

Independently stageable pieces:

- Trajectory packing: pure tensor/indexing ABI from `[B,T,*]` inputs to `[B,3T,H]`.
- GPT-2 body: causal self-attention stack with learned absolute position embedding forced to position id 0 in the DecisionTransformerModel wrapper.
- Prediction heads: three small linear heads, with optional final `tanh` only on action prediction.

## 3. Important config dimensions

Source defaults from `DecisionTransformerConfig`:

| Field | Default | Runtime significance |
|---|---:|---|
| `state_dim` | 17 | State input width and state prediction head output width |
| `act_dim` | 4 | Action input width and action prediction head output width |
| `hidden_size` | 128 | Transformer width and modality embedding width |
| `max_ep_len` | 4096 | Learned timestep embedding rows |
| `vocab_size` | 1 | GPT-2 token embedding rows; unused by primary DecisionTransformerModel path |
| `n_positions` / `max_position_embeddings` | 1024 | GPT-2 learned position embedding rows; primary wrapper passes all-zero position ids |
| `n_layer` / `num_hidden_layers` | 3 | Transformer block count |
| `n_head` / `num_attention_heads` | 1 | MHA head count |
| effective `head_dim` | `hidden_size / n_head` | Must divide exactly; source raises otherwise |
| `n_inner` | null | MLP width; null means `4 * hidden_size` |
| `activation_function` | `relu` | MLP activation |
| `action_tanh` | true | Adds final tanh to action head |
| `scale_attn_weights` | true | Dot-product scale `head_dim ** -0.5` |
| `scale_attn_by_inverse_layer_idx` | false | Optional extra attention scale by layer index |
| `reorder_and_upcast_attn` | false | Optional eager-only fp32 baddbmm attention path |
| `add_cross_attention` | false | Optional GPT-2 cross-attention blocks; not used by primary wrapper |
| `use_cache` | true | GPT-2 body can allocate/update a cache, but primary wrapper does not return it |

Representative checkpoint sweep from saved `config.json` files:

| Checkpoint | state_dim | act_dim | hidden_size | layers | heads | max_ep_len | activation | action_tanh | dtype | Source note |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `edbeeching/decision-transformer-gym-hopper-medium` | 11 | 3 | 128 | 3 | 1 | 1000 | relu | true | float32 | config.json |
| `edbeeching/decision-transformer-gym-hopper-expert` | 11 | 3 | 128 | 3 | 1 | 1000 | relu | true | float32 | config.json |
| `edbeeching/decision-transformer-gym-halfcheetah-medium` | 17 | 6 | 128 | 3 | 1 | 1000 | relu | true | float32 | config.json |
| `edbeeching/decision-transformer-gym-halfcheetah-expert` | 17 | 6 | 128 | 3 | 1 | 1000 | relu | true | float32 | config.json |
| `edbeeching/decision-transformer-gym-walker2d-medium` | 17 | 6 | 128 | 3 | 1 | 1000 | relu | true | float32 | config.json |
| `edbeeching/decision-transformer-gym-walker2d-expert` | 17 | 6 | 128 | 3 | 1 | 1000 | relu | true | float32 | config.json |

All inspected public Gym configs also contain historical GPT-2 summary fields and `n_embd=768`. The current Decision Transformer source reads `hidden_size=128` for module dimensions, so `n_embd` is ignored for this source basis.

## 3a. Family variation traps

- `state_dim` and `act_dim` vary by environment and change the first/last linear widths.
- The packed sequence length is `3 * T`; masks and output reshapes must use the tripled length.
- `position_ids` are explicitly zeros in `DecisionTransformerModel`, so learned timestep embeddings carry episode time and the GPT-2 `wpe` contributes only row 0 in the primary path.
- `rewards` is accepted by the forward signature but not read by the model body.
- GPT-2 `Conv1D` stores weights as `[in_features, out_features]`, not PyTorch `Linear`'s `[out_features, in_features]`.
- QKV projection is packed as all-Q, all-K, all-V contiguous feature blocks after `c_attn`, then split along the last dimension.
- `n_inner=None` means the MLP expands to `4 * hidden_size`; do not infer from checkpoint keys alone.
- `add_cross_attention=True` adds an extra attention module and LayerNorm per block, but representative configs do not enable it and the primary wrapper does not pass encoder states.
- `reorder_and_upcast_attn=True` changes eager attention math to fp32 `baddbmm`; representative configs leave it false.
- The primary RL target has no LM logits, no tokenizer ABI, and no autoregressive sampling controller.
- Layout translation is mostly rank-3 sequence work. Stack/permute/reshape regions are axis-sensitive and should be guarded until the whole trajectory packing region is owned.

## 4. Operator coverage checklist

Tensor/layout ops:

- Dense `reshape`, `view`, `permute`, `transpose`, `contiguous`.
- `stack` over three tensors, then `permute(0,2,1,3)`, then flatten `[B,T,3,H] -> [B,3T,H]`.
- Slice/index selects `x[:, 2]` and `x[:, 1]` after `[B,3,T,H]` permutation.
- `arange`, `unsqueeze`, zero-filled `position_ids`, ones default `attention_mask`.
- Optional `pad` inside generic mask preparation when cache/mask lengths differ.

Neural network primitives:

- Embedding lookup: `embed_timestep[max_ep_len, H]`; GPT-2 `wpe[n_positions, H]` row 0 in primary path; GPT-2 `wte[vocab_size, H]` only for direct GPT2Model input/token-type paths.
- Linear with bias: `state_dim -> H`, `act_dim -> H`, `1 -> H`.
- LayerNorm over last dim: one `embed_ln`, two per transformer block, final `ln_f`; epsilon from config.
- GPT-2 Conv1D/addmm projections with transposed storage:
  - self-attention `c_attn`: `H -> 3H`
  - attention output `c_proj`: `H -> H`
  - MLP `c_fc`: `H -> inner`
  - MLP `c_proj`: `inner -> H`
- MLP activation from `ACT2FN`; default and checkpoints use ReLU.
- Prediction heads: `H -> state_dim`, `H -> act_dim`, `H -> 1`, with optional `tanh` after action head.
- Dropout is source-visible but inactive in inference.

Attention primitives:

- Causal dense self-attention, MHA only in inspected configs (`n_head=1`, so no GQA/MQA).
- MatMul QK^T, additive mask, softmax over key length, optional dropout, MatMul with V.
- Optional attention backend dispatch through Transformers `ALL_ATTENTION_FUNCTIONS`; eager fallback is source-defined.
- Optional cache update for keys/values if `use_cache=True`.

Position/custom math:

- Learned timestep embedding added before packing.
- Learned absolute position embedding row lookup inside GPT-2; primary wrapper supplies all-zero position ids.
- No RoPE, ALiBi, relative bias, sliding-window attention, MoE, convolution, or quantized packed weight path in source.

Preprocessing-coupled ops:

- Caller owns environment state normalization, return scaling, and action padding/history construction. The model only consumes already numeric trajectory tensors.
- Attention mask `[B,T]` is triplicated to `[B,3T]`.

## 5. Layer/block breakdown

Primary wrapper, for batch `B`, trajectory length `T`, hidden width `H`, state width `S`, action width `A`:

```text
state_embeddings   = Linear(S -> H)(states[B,T,S])
action_embeddings  = Linear(A -> H)(actions[B,T,A])
return_embeddings  = Linear(1 -> H)(returns_to_go[B,T,1])
time_embeddings    = Embedding(max_ep_len,H)(timesteps[B,T])

state_embeddings  += time_embeddings
action_embeddings += time_embeddings
return_embeddings += time_embeddings

stacked_inputs = stack(return, state, action, dim=1)      # [B,3,T,H]
stacked_inputs = permute(0,2,1,3).reshape(B, 3T, H)
stacked_inputs = LayerNorm(stacked_inputs)
stacked_mask   = stack(mask, mask, mask, dim=1).permute(0,2,1).reshape(B, 3T)
```

GPT-2 block, repeated `n_layer` times:

```text
residual = x
x = LayerNorm(x)
qkv = Conv1D_weight_in_out(H -> 3H)(x)       # packed [Q,K,V]
q,k,v = split(qkv, H, dim=-1)
q,k,v = reshape to [B, heads, L, head_dim]
attn = causal_attention(q,k,v, additive_mask, optional_cache)
x = residual + Conv1D_weight_in_out(H -> H)(attn)

residual = x
x = LayerNorm(x)
x = Conv1D_weight_in_out(H -> inner)(x)
x = activation(x)
x = Conv1D_weight_in_out(inner -> H)(x)
x = residual + x
```

Prediction heads:

```text
x = final_layer_norm(x).reshape(B,T,3,H).permute(0,2,1,3)
return_preds = Linear(H -> 1)(x[:, 2])
state_preds  = Linear(H -> S)(x[:, 2])
action_preds = Linear(H -> A)(x[:, 1])
action_preds = tanh(action_preds) if action_tanh
```

For representative public configs, `H=128`, `heads=1`, `head_dim=128`, `inner=512`, and `n_layer=3`.

## 6. Attention requirements

Required for first target:

- Standard causal self-attention over `[B, L=3T, H]`.
- MHA with `num_heads=n_head`, `head_dim=H/n_head`; inspected checkpoints use one head.
- Q/K/V widths are all `H`; attention output width is `H`.
- Masking combines causal lower-triangular behavior with an optional padding mask expanded from `[B,3T]`.
- Packed/varlen sequence metadata is not used in the primary wrapper because an explicit attention mask is supplied.
- No sliding-window/local/block-sparse/hash/sort attention is implemented by this family.
- KV cache is implemented by the GPT-2 body via Transformers `Cache` classes, but primary `DecisionTransformerModel` does not expose generated tokens or return `past_key_values`. Treat cache as optional/deferred for first action-prediction parity, with a guard or config override if exact default internals are not modeled.

Optional source paths:

- Cross-attention is available only if `add_cross_attention=True` and encoder states are passed to the GPT-2 body. It is out of scope for the inspected Gym checkpoints.
- `reorder_and_upcast_attn=True` in eager mode performs fp32 `baddbmm` for QK^T and softmax before downcasting to V dtype. Reject or route this separately until parity-tested.
- SDPA/FlashAttention compatibility comes through the generic Transformers attention interface. DinoML can lower to its own dense attention primitive if the same scale, mask, dtype, and dropout-off inference order are preserved.

Cache shape if admitted:

```text
per layer key/value before repeat: [B, num_heads, cached_L, head_dim]
new token/update input:            [B, num_heads, q_L, head_dim]
no GQA repeat expansion for current configs
```

## 7. Position encoding and custom math

There is no rotary or relative-position math. Position is the sum of:

- Learned timestep embedding from `timesteps[B,T]`, added separately to return/state/action modality embeddings.
- GPT-2 learned absolute position embedding from `position_ids`. In the primary wrapper, every position id is zero, so this is a constant learned vector added to every packed token.

Source-equivalent packing math:

```python
def pack_decision_tokens(returns_h, states_h, actions_h, time_h, attention_mask):
    r = returns_h + time_h
    s = states_h + time_h
    a = actions_h + time_h
    tokens = stack([r, s, a], dim=1).permute(0, 2, 1, 3)
    tokens = tokens.reshape(B, 3 * T, H)
    mask = stack([attention_mask, attention_mask, attention_mask], dim=1)
    mask = mask.permute(0, 2, 1).reshape(B, 3 * T)
    position_ids = zeros_like(mask, dtype=int64)
    return tokens, mask, position_ids
```

Precomputable:

- `wpe[0]` can be folded as a constant bias-like add for the primary wrapper.
- Causal mask for fixed `T` can be precomputed if batch padding is absent.

Dynamic inputs:

- `timesteps`, `attention_mask`, and all trajectory tensors are runtime inputs.

## 8. Preprocessing and input packing

Runtime input contract:

| Input | Shape | Dtype | Notes |
|---|---|---|---|
| `states` | `[B,T,state_dim]` | float | Environment state features, already normalized by caller if desired |
| `actions` | `[B,T,act_dim]` | float | Prior action trajectory; current action can be zero/masked by caller policy |
| `returns_to_go` | `[B,T,1]` | float | Return target/history |
| `timesteps` | `[B,T]` | int64/long | Indexes `embed_timestep`; must be `< max_ep_len` |
| `attention_mask` | `[B,T]` | int/bool/float accepted by PyTorch path | Default is all ones; source converts/prepares for attention |
| `rewards` | `[B,T]` or `[B,T,1]` by doc examples | float | Accepted but unused |

There is no tokenizer, language control, image/audio/video processor, placeholder token scatter, or postprocessing step. End-to-end parity for RL agents still depends on external environment wrappers that maintain history windows, normalize observations, scale returns, and choose the next action from `action_preds[:, -1]`; that is outside the neural graph.

Axis-sensitive packing guards:

- `stack(..., dim=1)` creates modality axis order return/state/action.
- `permute(0,2,1,3)` changes `[B,3,T,H]` to `[B,T,3,H]`.
- Final `reshape(B,T,3,H).permute(0,2,1,3)` must preserve the same modality order.
- Action prediction reads modality index 1; state/return predictions read modality index 2.

## 9. Graph rewrite / lowering opportunities

### Rewrite: GPT-2 Conv1D to GEMM

Source pattern:

```text
addmm(bias, x.reshape(-1, in), weight[in,out]).reshape(..., out)
```

Replacement:

```text
Flatten leading dims -> GEMM_RRR(A=[M,in], B=[in,out]) -> BiasAdd -> Reshape
```

Preconditions:

- Weight is stored `[in_features, out_features]`.
- Input is contiguous or materialized as row-major over the last dimension.
- Bias is shape `[out_features]`.

Failure cases:

- Do not transpose weights as if loading PyTorch `Linear` unless the loader explicitly converts layout.
- Preserve packed QKV split order `[Q, K, V]`.

Parity test sketch:

- Random `[B,L,H]` against source `Conv1D` for `H -> 3H`, `H -> H`, and `H -> 4H`.

### Rewrite: trajectory stack/permute/reshape to indexed row interleave

Source pattern:

```text
stack(return,state,action, dim=1) -> permute(0,2,1,3) -> reshape(B,3T,H)
```

Replacement:

```text
write packed[:, 3*t + 0, :] = return[:, t, :]
write packed[:, 3*t + 1, :] = state[:, t, :]
write packed[:, 3*t + 2, :] = action[:, t, :]
```

Preconditions:

- Exactly three modalities in the source order.
- All embeddings are `[B,T,H]`.
- Consumer is the GPT-2 body using `[B,3T,H]`.

Failure cases:

- Reject if a future config changes modality order or adds another stream.

### Rewrite: constant zero GPT-2 position ids

Source pattern:

```text
position_ids = zeros([B,3T])
position_embeds = wpe(position_ids)
hidden = inputs_embeds + position_embeds
```

Replacement:

```text
hidden = inputs_embeds + broadcast(wpe[0], [B,3T,H])
```

Preconditions:

- Entry point is `DecisionTransformerModel`, not direct `DecisionTransformerGPT2Model`.
- Caller cannot override position ids through the wrapper.

Failure cases:

- Direct GPT-2 submodel use with nonzero `position_ids`.

### Rewrite: last-step action-only output

Source pattern:

```text
action_preds = head(x[:,1,:,:])
caller consumes action_preds[:, -1, :]
```

Replacement:

```text
optional graph output slice for x[:,1,T-1,:] -> action head
```

Preconditions:

- Serving ABI only needs next action, not full `state_preds`, `action_preds`, and `return_preds`.
- Validation has an end-to-end policy-level test.

Failure cases:

- Training/evaluation users may require all timesteps and all heads.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over `[B,3T,H]`, because every block has two norms plus `embed_ln` and `ln_f`.
- Conv1D/GEMM with bias for QKV and MLP projections, respecting `[in,out]` storage.
- Dense causal attention for small head count, including mask add and softmax.
- ReLU MLP epilogue fusion for `H -> 4H -> H`.

Medium priority:

- Trajectory packing fused copy/interleave after modality projections.
- Add timestep embedding plus modality embedding before stack.
- Final head projection plus optional tanh for action output.
- Constant `wpe[0]` broadcast fold.

Lower priority:

- KV cache update kernels for this family; useful only if DinoML exposes incremental trajectory extension.
- Cross-attention path; not present in representative checkpoints.
- Upcast/reordered attention path; rare config flag and eager-specific.

## 11. Runtime staging plan

Stage 1: Config and weights.

- Parse `DecisionTransformerConfig`, including attribute-map aliases.
- Load Conv1D weights without transposing storage accidentally.
- Reject unsupported `add_cross_attention=True`, `reorder_and_upcast_attn=True`, and non-divisible `hidden_size/n_head` initially.

Stage 2: Trajectory embedding and packing parity.

- Implement modality linears, timestep embedding, packing, stacked mask, and `embed_ln`.
- Validate packed embeddings against Transformers for random tensors.

Stage 3: One GPT-2 block parity.

- Lower Conv1D projections, LayerNorm, causal attention, MLP, residuals.
- Use full-sequence inference with dropout disabled.

Stage 4: Full model parity.

- Run all blocks and final heads for Hopper and HalfCheetah configs.
- Return `state_preds`, `action_preds`, `return_preds`, and optionally `last_hidden_state`.

Stage 5: Optional cache/default cleanup.

- Decide whether first public ABI forces `use_cache=False` for the hidden GPT-2 body or models source default cache updates internally.
- Add cache parity only if incremental trajectory serving is needed.

Stage 6: Fusions and serving specialization.

- Add Conv1D-to-CUTLASS GEMM rewrite, LayerNorm kernels, attention backend, and optional last-action-only output.

## 12. Parity and validation plan

- Config parse tests: source defaults plus saved Hopper/HalfCheetah/Walker2d configs; assert ignored historical `n_embd` does not change `hidden_size`.
- Conv1D layout tests: random tensors against Transformers `Conv1D` for all projection shapes.
- Packing tests: compare packed tokens, stacked masks, and final modality slices for random `B,T`.
- Single-block fp32 parity: disable dropout, compare one block with fixed weights and all-ones mask.
- Full-model fp32 parity: compare `state_preds`, `action_preds`, `return_preds`, and `last_hidden_state` for the saved representative configs.
- Mask parity: all-ones mask, left/right padded mask, and short `T` cases.
- Head parity: with `action_tanh=True` and a synthetic config with `action_tanh=False`.
- Optional cache parity: full prefix versus prefix plus appended step if DinoML admits cache.

Suggested tolerances:

- fp32: absolute/relative `1e-5` for block/model outputs.
- fp16/bf16 optimized kernels: start at `1e-2` absolute or relative for full model, with tighter per-op tolerances where accumulation is fp32.

## 13. Performance probes

- Embedding/packing throughput as `B,T,state_dim,act_dim` vary.
- Transformer throughput over `L=3T`, especially `T=20`, `T=100`, and `T=1000` style windows.
- Attention backend comparison: eager matmul/softmax/matmul versus fused dense attention.
- LayerNorm and MLP GEMM time per block.
- Batch-size sweep for small hidden size `H=128`, where launch overhead may dominate.
- Output specialization probe: full three heads/all timesteps versus last-action-only.
- Cache memory and speed probe only if incremental trajectory extension is admitted.

## 14. Skip/defer list

- Training, losses, gradient checkpointing, dropout behavior.
- Tokenized text input, tokenizer ABI, LM logits, sampling/generation controllers.
- Cross-attention from `add_cross_attention=True`.
- `reorder_and_upcast_attn=True` eager-specific attention path.
- Returning or externally managing `past_key_values` for the primary RL wrapper.
- Hidden-state and attention recording outputs unless debugging requires them.
- Environment simulation, observation normalization, return scaling, and action selection policy outside the neural graph.
- Multi-GPU/tensor parallelism and quantized weight formats.

## 15. Final implementation checklist

- [ ] Parse `DecisionTransformerConfig` and alias fields.
- [ ] Load saved Gym checkpoint configs as smoke fixtures.
- [ ] Load Conv1D weights with `[in,out]` storage contract.
- [ ] Implement modality linears and timestep embedding.
- [ ] Implement trajectory interleave packing and stacked attention mask.
- [ ] Implement/fold zero-position GPT-2 embedding behavior for the primary wrapper.
- [ ] Implement LayerNorm, residual add, ReLU MLP, and prediction heads.
- [ ] Implement causal dense self-attention over `[B,n_head,L,head_dim]`.
- [ ] Add guards for unsupported cross-attention and upcast/reordered attention.
- [ ] Add one-block parity tests.
- [ ] Add full-model parity tests for Hopper and HalfCheetah configs.
- [ ] Add mask/padding parity tests.
- [ ] Add Conv1D-to-GEMM rewrite with QKV split-order validation.
- [ ] Benchmark packing, attention, MLP, and last-action-only serving.
