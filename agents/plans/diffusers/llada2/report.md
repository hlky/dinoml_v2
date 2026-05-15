# Diffusers LLaDA2 Operator and Integration Report

Candidate slug: `llada2`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  inclusionAI/LLaDA2.1-mini
  inclusionAI/LLaDA2.1-flash
  inclusionAI/LLaDA2.0-mini
  inclusionAI/LLaDA2.0-flash
  inclusionAI/LLaDA2.0-mini-CAP
  inclusionAI/LLaDA2.0-flash-CAP

Config sources:
  H:/configs had no cached llada2/LLaDA2 model_index or component config files.
  Official Hugging Face raw/API reads succeeded for the inclusionAI repos above:
    config.json
    tokenizer_config.json
    special_tokens_map.json
    tokenizer.json
    model.safetensors.index.json
    configuration_llada2_moe.py
    modeling_llada2_moe.py
  Hub API snapshots checked:
    inclusionAI/LLaDA2.1-mini @ 20e64e2ad21644d0e5248586ed9c942cdd45de0f
    inclusionAI/LLaDA2.0-mini @ dad945cac317da394b390f82c7b40691d8a881ed
    inclusionAI/LLaDA2.0-flash-CAP @ 3e65ff386063c389ca498d5ca862a687252d75c6
  The official repos are public and not gated. No authenticated retry was needed.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/llada2/pipeline_llada2.py
  diffusers/src/diffusers/pipelines/llada2/__init__.py

Model files inspected:
  No Diffusers model file exists for this target. The pipeline expects an
  external Transformers `AutoModelForCausalLM` loaded with `trust_remote_code`.
  Official remote code inspected:
    inclusionAI/*/configuration_llada2_moe.py
    inclusionAI/*/modeling_llada2_moe.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_block_refinement.py
  diffusers/src/diffusers/schedulers/__init__.py
  diffusers/docs/source/en/api/pipelines/llada2.md
  diffusers/docs/source/en/api/schedulers/block_refinement.md
  diffusers/examples/discrete_diffusion/sample_llada2.py
  diffusers/tests/pipelines/llada2/test_llada2.py
  diffusers/tests/schedulers/test_scheduler_block_refinement.py

External component configs inspected:
  `PreTrainedTokenizerFast` tokenizer configs and `LLaDA2MoeConfig` /
  `LLaDA2MoeModelLM` remote-code configs from inclusionAI official repos.

Any missing files or assumptions:
  These repos do not provide `model_index.json`, `scheduler_config.json`, or
  `generation_config.json`; raw URL reads returned 404 for all sampled repos.
  This is expected because Diffusers does not own the model weights/config here:
  users instantiate the tokenizer/model through Transformers and manually pass
  `BlockRefinementScheduler()` to `LLaDA2Pipeline`. There is no VAE,
  autoencoder, image/video/audio processor, safety checker, or image
  postprocess boundary in the selected target.
```

## 2. Pipeline and component graph

`LLaDA2Pipeline` wires three components: an external causal language model, a
`BlockRefinementScheduler`, and an optional tokenizer. It is a discrete text
diffusion pipeline, not an image latent pipeline.

```text
prompt / messages / input_ids
  -> optional tokenizer chat template or direct tokenization
  -> construct full token canvas [B,total_length] filled with mask_token_id
  -> construct all-ones attention mask [B,total_length] and position_ids
  -> block-wise denoising/refinement loop:
       model(block_x, attention_mask, position_ids).logits
       BlockRefinementScheduler.step over last block logits
       commit mask tokens by confidence/top-k schedule
       optional post-mask editing of non-prompt tokens
       optional EOS early-stop
  -> generated token IDs [B,gen_length or EOS-trimmed]
  -> optional tokenizer.batch_decode
```

Required first-slice components:

| Component | Class/file | Required contract |
| --- | --- | --- |
| Pipeline | `LLaDA2Pipeline`, `pipeline_llada2.py` | Prompt/input validation, mask-token canvas, block loop, model call, callback surface ignored for compilation. |
| Scheduler | `BlockRefinementScheduler`, `scheduling_block_refinement.py` | Discrete token sampling, confidence commit schedule, optional editing, EOS/block continuation helpers. |
| Model | external `LLaDA2MoeModelLM` remote code | Embedding -> decoder-only bidirectional transformer -> logits `[B,S,V]`. |
| Tokenizer | external `PreTrainedTokenizerFast` | Chat template, special token IDs, decode. Can be CPU/data pipeline initially. |

Cacheable or reusable stages:

- Tokenized prompts or `input_ids`.
- All-ones attention masks and monotonic `position_ids` for a fixed
  `total_length`.
- RoPE cos/sin tables for a fixed max sequence length.
- Scheduler transfer schedule for `(block_length, num_inference_steps)`.
- Static model weights, including expert weights and router weights.

Separate candidate reports:

| Candidate | Classes/files | Runtime delta |
| --- | --- | --- |
| `llada2_moe_transformer` | official remote `modeling_llada2_moe.py` | Treat the external Transformers MoE decoder as its own model-port report: GQA self-attention, QK RMSNorm, RoPE, top-k expert routing, shared expert, large vocab LM head. |
| `block_refinement_scheduler` | `scheduling_block_refinement.py` | Standalone discrete scheduler/runtime-state report for confidence commit, top-k/top-p sampling, post-mask editing, EOS early-stop, and training `add_noise`. |
| `llada2_flash_attention_variants` | `LLaDA2MoeFlashAttention2`, `LLaDA2MoeSdpaAttention` in flash repo remote code | Attention backend specialization for noncausal bidirectional GQA with RoPE and QK norm; separate from base eager parity. |
| `llada2_any_to_any_uni` | Hub repos `inclusionAI/LLaDA2.0-Uni` and `LLaDA2.0-Uni-FP8` | Related inclusionAI family with `any-to-any` metadata, but not wired by Diffusers `LLaDA2Pipeline`; needs separate source/config audit before treating as this target. |
| `llada2_offload_weight_residency` | `examples/discrete_diffusion/sample_llada2.py`, Diffusers hooks | Example supports group/sequential offload; this is an explicit residency/weight-loading candidate, not a first parity requirement. |
| `llada2_adapter_mutation` | Transformers/PEFT external surface, no Diffusers loader in this folder | LoRA/PEFT may exist through Transformers ecosystem, but the Diffusers LLaDA2 folder has no family-local loader mixin. |

No family-local IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, img2img, inpaint,
depth2img, upscaling, VAE, or image processor surface exists for the
non-deprecated `llada2` pipeline folder.

## 3. Important config dimensions

Representative checkpoint sweep:

| Config | Hidden | Layers | Heads / KV heads | Head dim | Dense FFN | MoE | Context | Dtype metadata | Safetensors total size |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | --- | ---: |
| `inclusionAI/LLaDA2.1-mini` | 2048 | 20 | 16 / 4 | 128 | 5120 | 256 experts, top-8, expert hidden 512, 1 shared expert | 32768 | `dtype=bfloat16` | 32,511,296,512 bytes |
| `inclusionAI/LLaDA2.1-flash` | 4096 | 32 | 32 / 4 | 128 | 9216 | 256 experts, top-8, expert hidden 1024, 1 shared expert | 32768 | `torch_dtype=bfloat16` | 205,779,426,304 bytes |
| `inclusionAI/LLaDA2.0-mini` | 2048 | 20 | 16 / 4 | 128 | 5120 | same as 2.1 mini | 32768 | `dtype=bfloat16` | 32,511,286,784 bytes |
| `inclusionAI/LLaDA2.0-flash` | 4096 | 32 | 32 / 4 | 128 | 9216 | same as 2.1 flash | 32768 | `torch_dtype=bfloat16` | 205,779,410,432 bytes |
| `inclusionAI/LLaDA2.0-mini-CAP` | 2048 | 20 | 16 / 4 | 128 | 5120 | same as mini | 32768 | `torch_dtype=bfloat16` | 32,511,286,784 bytes |
| `inclusionAI/LLaDA2.0-flash-CAP` | 4096 | 32 | 32 / 4 | 128 | 9216 | same as flash | 32768 | `torch_dtype=bfloat16` | 205,779,410,432 bytes |

Common model config fields:

| Field | Value / source |
| --- | --- |
| `model_type` | `llada2_moe`, official model config. |
| `architectures` | `["LLaDA2MoeModelLM"]`, official model config. |
| `auto_map` | `AutoConfig`, `AutoModel`, and `AutoModelForCausalLM` point to remote code files. |
| `vocab_size` | 157184. |
| `pad_token_id` | 156892, same string as EOS/pad `<|endoftext|>`. |
| `mask_token_id` | From tokenizer added token `<|mask|>` id 156895, not stored in model config. |
| `bos/eos/pad/mask` | `<|startoftext|>` id 156891, `<|endoftext|>` id 156892, `<|mask|>` id 156895. |
| RoPE | `rope_theta=600000`, `partial_rotary_factor=0.5`, `rotary_dim=64`, `rope_scaling=null`. |
| Attention | GQA with `num_attention_heads / num_key_value_heads`; `use_qk_norm=true`, no qkv bias, no output bias. |
| MoE routing | Sigmoid router, fp32 router logits, group-limited top-k over 8 groups and top 4 groups, top-8 experts per token, `routed_scaling_factor=2.5`. |
| Dense/MoE layout | `first_k_dense_replace=1`: layer 0 dense MLP, later layers sparse MoE. |
| Cache/sliding window | Sampled configs set `use_cache=false`, `use_sliding_window=false`; `sliding_window=4096` is present but inactive. |

Scheduler/pipeline defaults:

| Field | Pipeline default | Scheduler config default | Note |
| --- | ---: | ---: | --- |
| `gen_length` | 2048 | N/A | Number of generated tokens after prompt. |
| `block_length` | 32 | 32 | Pipeline passes block shape to loop; scheduler's `set_timesteps` uses its own config block length for transfer schedule. First parity should require these to match. |
| `num_inference_steps` | 32 | 32 | Pipeline clamps to `min(num_inference_steps, gen_length // minimal_topk)`. |
| `threshold` | 0.7 | 0.95 | Pipeline call default overrides scheduler step default. Docs recommend 0.7 for LLaDA2.1 quality. |
| `editing_threshold` | 0.5 | `None` | Pipeline enables post-mask editing by default; docs say disable for LLaDA2.0. |
| `max_post_steps` | 16 | N/A | Bounds editing after all masks in a block are filled. |
| `minimal_topk` | 1 | 1 | Minimum commit count per refinement step. |
| Sampling | `sampling_method="multinomial"`, `temperature=0.0` | N/A | With temperature 0, scheduler uses greedy argmax even when method is multinomial. |

Recommended first Dinoml scheduler slice: `BlockRefinementScheduler` with
greedy sampling (`temperature=0`, no top-p/top-k filtering), no editing, fixed
`block_length=32`, and `threshold > 1` or low threshold tests that force the
top-k transfer schedule. Add top-k/top-p stochastic sampling and editing after
deterministic block parity is stable.

## 3a. Family variation traps

- This is a text-token diffusion pipeline, not an image transformer pipeline.
  There are no latent channels, VAE scale factors, image patch sizes, or
  NCHW/NHWC layout contracts.
- Diffusers owns the pipeline and scheduler only; the denoiser model is
  external Transformers remote code. Operator admission should treat remote
  model code as a separate source basis from Diffusers.
- `LLaDA2.1-mini` and `LLaDA2.0-mini` share the same operator shape; the main
  version difference visible in Diffusers docs is recommended editing behavior.
- Mini versus flash configs are materially different: 20-layer 2048-wide versus
  32-layer 4096-wide, much larger safetensors footprint, and flash remote code
  includes explicit FlashAttention2/SDPA classes.
- The pipeline calls the model on the growing prefix window for every
  refinement step. It does not do autoregressive one-token decode and sampled
  configs set `use_cache=false`, so KV-cache optimized decoding is not the
  first parity path.
- The source builds an all-ones attention mask, then remote model code converts
  it with Transformers `create_bidirectional_mask`; attention is bidirectional
  and `is_causal=False`.
- The scheduler's transfer schedule is based on `scheduler.config.block_length`,
  while `pipeline.__call__` accepts a `block_length` argument. A mismatch would
  make commit counts inconsistent with the active block shape. First Dinoml
  admission should guard equality.
- `masks_remaining = (block_tokens == mask_token_id).any()` is a scalar bool
  across the whole batch. The loop advances `step_idx` for all batch rows while
  any row still has masks.
- EOS early-stop only marks a row finished when the committed EOS exists and
  all generated tokens before that EOS are unmasked. For batch size 1, final
  output is trimmed through first EOS; for larger batches the source does not
  per-row trim.
- MoE inference uses token sorting and per-expert loops in remote Python code,
  including a CPU `.numpy()` read of tokens-per-expert. Dinoml needs a
  provider-backed MoE dispatch path rather than lowering this literally.

## 4. Runtime tensor contract

For batch `B`, prompt length `P`, generation length `G`, block length `K=32`:

| Boundary | Tensor | Shape / dtype | Notes |
| --- | --- | --- | --- |
| Tokenized prompt | `prompt_ids` | `[B,P] int64` | Produced by tokenizer or supplied directly as `input_ids`; CPU/data path can own tokenization. |
| Total canvas | `x` | `[B,T] int64`, `T=ceil((P+G)/K)*K` | Initialized to `mask_token_id`, prompt copied into prefix. |
| Attention mask | `attn_mask` | `[B,T] int64` | All ones; sliced to current window. |
| Position ids | `position_ids` | `[B,T] int64` | `arange(T)` expanded across batch; sliced to current window. |
| Active window | `block_x` | `[B,E] int64`, `E=(block_index+1)*K` | Growing prefix through current block. |
| Active block tokens | `block_tokens` | `[B,K] int64` | Last block only; scheduler updates this slice. |
| Prompt mask in block | `prompt_mask_in_block` | `[K] bool` | Marks prompt tokens inside the current block as non-editable. |
| Model logits | `logits` | `[B,E,V] float`, `V=157184` | Remote model returns full window logits; pipeline slices last block. |
| Block logits | `block_logits` | `[B,K,V] float32 after LM head` | Input to scheduler sampling. |
| Sampled tokens/probs | scheduler output | `[B,K] int64`, `[B,K] float` | Candidate token and probability per block position. |
| Transfer masks | scheduler output | `[B,K] bool` | `transfer_index` for mask commits and `editing_transfer_index` for replacements. |
| Updated block | `prev_sample` | `[B,K] int64` | Written into `block_x[:, -K:]` when any transfer occurs. |
| Final output | `sequences` | `[B,G] int64`, or EOS-trimmed `[1,<=G]` | Decoded only if `output_type="text"` and tokenizer exists. |

Remote model internal tensor shapes:

| Model stage | Mini shape | Flash shape | Notes |
| --- | --- | --- | --- |
| Embedding | `[B,S,2048]` | `[B,S,4096]` | `word_embeddings(input_ids)`. |
| QKV projection | `[B,S,(Hq+2Hkv)*D]` | same formula | Mini: `(16+8)*128=3072`; flash: `(32+8)*128=5120`. |
| Q/K/V heads | Q `[B,Hq,S,128]`, K/V `[B,Hkv,S,128]` | same | K/V repeated to Q heads for eager/SDPA. |
| RoPE | first 64 dims of each head | same | `partial_rotary_factor=0.5`. |
| Router logits | `[B*S,256]` then `[B,S,256]` when returned | same | fp32 linear router over hidden size. |
| Expert FFN | top-8 per token, expert hidden 512 | top-8 per token, expert hidden 1024 | Plus one shared expert per MoE layer. |
| LM head | `[B,S,157184]` | same vocab | Output logits cast to float32 in remote code. |

CPU/data-pipeline work: chat template rendering, tokenization, final decoding,
input validation. GPU/runtime work: full transformer forward, scheduler
sampling/filtering/top-k decisions if compiled, token canvas updates, EOS/block
continuation predicates. No autoencoder encode/decode or image postprocessing
exists.

## 5. Operator coverage checklist

Tensor/layout/integer ops:

- `torch.full`, slice assignment, prefix/window slicing, `arange`, `expand`.
- int64 token copy/update, bool masks, `where`, `nonzero`, `any`, `all`,
  equality/inequality, scatter/gather.
- Dynamic window length `E=(block+1)*K`; fixed block shape `K`.
- Output slicing and single-row EOS trim.

GEMM/linear/embedding ops:

- Token embedding lookup `[V,H]`.
- Bias-free packed QKV projection: `H -> (num_heads + 2*num_kv_heads) * head_dim`.
- Attention output projection: `num_heads*head_dim -> H`, sampled configs have `use_bias=false`.
- Dense layer 0 SwiGLU MLP: gate/up/down projections, `H -> intermediate -> H`.
- Router projection: `H -> 256` in fp32.
- Expert projections: top-8 selected experts per token, each `H -> moe_intermediate -> H`.
- Shared expert MLP per MoE layer: `H -> moe_intermediate*num_shared_experts -> H`.
- LM head: `H -> vocab_size=157184`, bias-free and untied.

Attention primitives:

- Bidirectional self-attention, not causal AR attention.
- GQA with K/V head repeat.
- QK RMSNorm before RoPE.
- Partial RoPE over first 64 dims of head dim 128.
- Optional eager, SDPA, or FlashAttention2 depending remote code and load-time
  `_attn_implementation`.

Normalization and activations:

- RMSNorm with fp32 variance and dtype restore.
- SiLU/SwiGLU.
- Softmax over attention scores in fp32 for eager path.
- Sigmoid router scores, group-limited top-k, normalized top-k weights.

Scheduler and sampling arithmetic:

- Softmax over logits for probabilities.
- Greedy argmax, optional temperature scaling, top-k filtering, nucleus
  top-p filtering, multinomial sampling.
- Confidence threshold compare; fallback `torch.topk` over active mask tokens.
- Editing threshold compare, token-changed mask, prompt-mask exclusion.
- Transfer schedule: distribute block length across inference steps.

VAE/postprocessing ops:

- None for this target. Text decode is tokenizer CPU/data-pipeline work.

## 6. Denoiser/model breakdown

Pipeline-level denoising step:

```text
block_x [B,E] int64
  -> model(input_ids=block_x, attention_mask=ones[B,E], position_ids=[0..E-1])
  -> logits [B,E,V]
  -> block_logits = logits[:, -K:, :]
  -> scheduler.step(block_logits, block_tokens)
  -> optional write prev_sample into block_x[:, -K:]
```

Remote `LLaDA2MoeModelLM` forward:

```text
input_ids
  -> word_embeddings
  -> create_bidirectional_mask(attention_mask)
  -> RoPE cos/sin from position_ids
  -> repeated decoder layers
       RMSNorm
       packed QKV Linear
       split Q/K/V, QK RMSNorm, partial RoPE
       bidirectional self-attention + output Linear
       residual
       RMSNorm
       layer 0 dense SwiGLU or later sparse MoE + shared expert
       residual
  -> final RMSNorm
  -> lm_head, logits.float()
```

Decoder layer config behavior:

- `first_k_dense_replace=1`: only layer 0 uses dense `LLaDA2MoeMLP`; layers 1+
  use `LLaDA2MoeSparseMoeBlock`.
- `use_qk_norm=true`: query/key per-head RMSNorm is active in sampled configs.
- `use_qkv_bias=false` and `use_bias=false`: sampled linear projections are
  bias-free.
- `use_cache=false`: pipeline first-slice should compile full-window forward,
  not cached one-token generation.
- `output_router_logits=false`: router logits are not a required runtime output.

Sparse MoE inference path:

```text
hidden [B,S,H]
  -> router logits = linear(hidden fp32, router_weight fp32)
  -> sigmoid scores
  -> add expert_bias
  -> group_limited_topk: group scores, choose topk_group=4 of n_group=8,
     mask other experts, choose top-8 experts
  -> gather original scores for selected experts
  -> normalize selected scores and multiply routed_scaling_factor=2.5
  -> sort tokens by expert id
  -> run each selected expert MLP on its token slice
  -> unsort, weight, sum over top-8
  -> add shared expert output
```

Dinoml should model this as explicit router + grouped expert dispatch. Literal
Python loops and CPU token-count reads in the remote implementation are parity
evidence, not an acceptable compiled runtime shape.

## 7. Attention requirements

Required base attention:

| Field | Mini | Flash |
| --- | ---: | ---: |
| Query heads | 16 | 32 |
| KV heads | 4 | 4 |
| Head dim | 128 | 128 |
| Rotary dims | 64 | 64 |
| Attention kind | bidirectional self-attention | bidirectional self-attention |
| Mask | all-valid source mask in pipeline, converted to model mask | same |
| QK norm | active RMSNorm on Q and K | active RMSNorm on Q and K |

Source backend paths:

- Mini remote code uses `LLaDA2MoeAttention` and can dispatch through
  Transformers `ALL_ATTENTION_FUNCTIONS` when `_attn_implementation != "eager"`.
  The eager fallback defines parity: repeat K/V heads, matmul QK, add mask,
  fp32 softmax, dropout zero in inference, matmul V.
- Flash remote code defines `ATTENTION_CLASSES = {"eager", "flash_attention_2",
  "sdpa"}` and selects by `config._attn_implementation`.
- SDPA path uses `torch.nn.functional.scaled_dot_product_attention` after
  repeat K/V and requires a 4D mask shape when a mask is present.
- FlashAttention2 path keeps Q/K/V in `[B,S,heads,D]`, unpads when a padding
  mask exists, and does not support `output_attentions`.

Flash feasibility for Dinoml:

- Valid under strict preconditions: inference only, no output attentions,
  all-valid or supported padding mask, noncausal bidirectional attention,
  GQA handled either by native grouped-query support or explicit K/V repeat,
  QK RMSNorm and RoPE fused or pre-applied before the attention provider.
- The pipeline's all-ones mask means a first compiled path can avoid varlen
  unpadding for normal prompt generation, but direct `input_ids` with padding
  are not represented by the pipeline's attention mask today because it always
  uses ones over `total_length`.
- Attention layout candidates: keep transformer tokens as `[B,S,H]` at graph
  boundaries, use provider-local head-major or sequence-major layouts inside
  attention. No NHWC/NCHW layout translation is relevant.

## 8. Scheduler and denoising-loop contract

`BlockRefinementScheduler.set_timesteps(num_inference_steps, device)` creates:

- `timesteps = arange(num_inference_steps - 1, -1, -1)`.
- `_transfer_schedule = get_num_transfer_tokens(config.block_length,
  num_inference_steps)`, where the first `block_length % steps` entries get one
  extra token.

`scheduler.step` contract:

```text
model_output [B,K,V]
sample [B,K] int64 current block tokens
sampled_tokens, sampled_probs = greedy or multinomial sampling(model_output)
active_block = sample == mask_token_id
if any active masks:
  confidence = sampled_probs on active positions else -inf
  if confidence > threshold commits at least scheduled count:
    commit all high-confidence active positions
  else:
    commit top scheduled active positions
if editing_threshold > 0:
  editable = non-mask and not prompt_mask
  edit positions where sampled token differs and confidence > editing_threshold
prev_sample = sample with committed/edited positions replaced
```

Loop-side state:

- Outer loop iterates blocks from `prefill_blocks = prompt_length // block_length`
  through `num_blocks`.
- Inner loop continues while masks remain and steps are not exhausted, or while
  editing still changes tokens within `max_post_steps`.
- `finished[B]` is a host-visible bool tensor updated by EOS logic.
- `global_step` counts all refinement iterations including post-mask editing.
- Progress bars/callback mutation are ignored for first compiled parity.

Host-control first:

- Keep block loop, model invocation sequencing, EOS early-stop, and stochastic
  sampling RNG ownership in host-visible runtime first.
- Compile deterministic scheduler kernels only after the model forward contract
  is stable. Top-p sorting, multinomial sampling, and dynamic post-edit loops
  are later runtime kernels.

## 9. Position, timestep, and custom math

Positions:

- Pipeline `position_ids = arange(total_length).unsqueeze(0).expand(B,-1)`.
- Remote model recomputes RoPE cos/sin each forward from `position_ids` and
  `inv_freq`, using fp32 matmul and `cos`/`sin`, then casts to hidden dtype.
- RoPE applies to first `rotary_dim=64` dimensions and concatenates unrotated
  pass-through dims back to head dim 128.

Timestep:

- The scheduler's `timestep` argument is just the block-local integer
  `step_idx` used to index `_transfer_schedule`. There are no diffusion sigmas,
  alpha products, or continuous timesteps.

Custom math worth preserving:

- RMSNorm computes variance in fp32: `x * rsqrt(mean(x*x) + eps)`.
- Router uses sigmoid, not softmax over all experts; selected scores are
  normalized only over top-k selected experts.
- Group-limited routing first chooses groups using sum of top-2 scores per
  group, then masks experts outside selected groups before top-k expert choice.
- Scheduler top-p filtering sorts logits, masks tokens after cumulative
  probability threshold, shifts mask right to keep the first token above the
  threshold, and scatters back to original vocab order.

Precompute candidates:

- RoPE tables for fixed `total_length` and dtype.
- Transfer schedules.
- Static 2D all-ones masks if exact pipeline behavior is retained.
- Expert weight packing by expert id for grouped GEMM dispatch.

## 10. Preprocessing and input packing

Input sources:

- `input_ids`: already tokenized int64 `[P]` or `[B,P]`; no tokenizer required.
- `messages`: passed to tokenizer `apply_chat_template(..., tokenize=True,
  return_tensors="pt", return_dict=True)`.
- `prompt`: if `use_chat_template=True` and tokenizer has a chat template, a
  single string prompt is wrapped as `{"role": "user", "content": prompt}`.
  Batched prompt lists are rejected in this mode.
- Otherwise `tokenizer(prompt, return_tensors="pt", padding=isinstance(prompt, list))`.

Tokenizer facts:

- `tokenizer_class=PreTrainedTokenizerFast`.
- `model_max_length=32768`.
- `add_bos_token=false`, `add_eos_token=false`.
- Special tokens include `<|startoftext|>`, `<|endoftext|>`, `[CLS]`,
  `[gMASK]`, and `<|mask|>`.
- Chat template is model-specific and includes role tags and optional tool-call
  formatting. Treat it as CPU/data-pipeline logic.

Packing:

- There is no image/video latent packing. The only packing is block-wise text
  canvas padding up to a multiple of `block_length`.
- `num_blocks = ceil((prompt_length + gen_length) / block_length)`.
- `total_length = num_blocks * block_length`.
- Prompt tokens are copied into the prefix; all remaining positions start as
  `mask_token_id`.

## 11. Graph rewrite / lowering opportunities

1. **Packed QKV projection**
   - Source pattern: one linear to `(Q_heads + 2*KV_heads) * head_dim`, view,
     split, transpose.
   - Replacement: provider QKV projection with split metadata.
   - Preconditions: `using_split_qkv_in_self_attention=false`,
     `use_qkv_bias=false`, hidden contiguous `[B,S,H]`.
   - Shape equations: output width mini `3072`, flash `5120`.
   - Failure cases: configs with split QKV or qkv bias need separate lowering.
   - Parity test: compare Q/K/V tensors after split and transpose for random
     hidden input.

2. **QK RMSNorm + RoPE + noncausal GQA attention**
   - Source pattern: QK RMSNorm, partial RoPE, K/V repeat, attention.
   - Replacement: fused attention prelude plus Flash/SDPA provider.
   - Preconditions: inference, no output attentions, supported mask, head dim
     128, rotary dim 64, `is_causal=false`.
   - Failure cases: padding masks needing varlen, output attentions, unsupported
     grouped query behavior.
   - Parity test: eager attention versus fused/provider path for all-ones and
     padded masks.

3. **MoE grouped expert dispatch**
   - Source pattern: router linear/sigmoid/group-topk/topk, sort tokens by
     expert, per-expert MLP loop, weighted sum, shared expert.
   - Replacement: router kernel + grouped GEMM or segmented expert GEMM +
     combine kernel.
   - Preconditions: static `num_experts=256`, `top_k=8`, `n_group=8`,
     `topk_group=4`, fixed expert hidden size, inference only.
   - Failure cases: training path repeat_interleave, changed router score
     function, expert bias update, quantized expert weights.
   - Parity test: fixed hidden states with hand-computed router outputs and
     full layer parity against remote code.

4. **LM head last-block only**
   - Source pattern: remote model computes logits for all `E` positions, then
     pipeline slices last `K`.
   - Replacement: if model graph can expose hidden states before `lm_head`,
     apply LM head only to last block positions.
   - Preconditions: no consumer needs logits outside the active block; final
     hidden states for all tokens still computed for attention context.
   - Failure cases: output hidden/logit debugging, model API parity requiring
     full logits.
   - Parity test: compare `lm_head(hidden[:, -K:])` with `logits[:, -K:]`.

5. **Deterministic scheduler commit kernel**
   - Source pattern: softmax/argmax/topk/confidence masks over `[B,K,V]`.
   - Replacement: fused `argmax + selected probability + transfer mask +
     block update` for `temperature=0`, no top-p/top-k, no editing.
   - Preconditions: deterministic sampling, fixed `K`, transfer schedule
     available, `threshold` semantics preserved.
   - Failure cases: multinomial sampling, top-p/top-k filtering, editing.
   - Parity test: scheduler unit tests for commit counts, forced top-k via
     `threshold > 1`, and no masks left after block completion.

6. **RoPE table precompute**
   - Source pattern: compute `inv_freq @ position_ids`, cos/sin every model
     forward.
   - Replacement: cache cos/sin for `(total_length, rotary_dim, dtype)`.
   - Preconditions: static `rope_theta`, `partial_rotary_factor`, no dynamic
     rope scaling.
   - Failure cases: dynamic rope types from future configs.
   - Parity test: exact cos/sin comparison for selected position ids.

## 12. Kernel fusion candidates

Highest priority:

- Transformer block RMSNorm + packed QKV + QK RMSNorm + RoPE + attention
  prelude. Every refinement step runs full-window self-attention.
- MoE router + grouped expert dispatch. Layers 1+ are MoE; literal per-expert
  Python loops are the biggest runtime mismatch.
- LM head over last block. Vocab is large and only last-block logits feed the
  scheduler.
- Deterministic scheduler commit kernel for greedy first parity.

Medium priority:

- SwiGLU dense/shared expert fusion: gate/up projections plus SiLU multiply
  and down projection.
- RoPE cos/sin cache and application fusion.
- K/V repeat elimination inside GQA attention provider.
- Transfer schedule and block canvas update kernels.

Lower priority:

- Top-p filtering and multinomial sampling kernels.
- Post-mask editing loop acceleration.
- EOS early-stop compaction/trimming for batched outputs.
- KV-cache or sliding-window variants, because sampled pipeline configs disable
  cache and sliding window.

## 13. Runtime staging plan

1. Stage 1: parse official model/tokenizer configs and build a pure scheduler
   parity harness with dummy logits. Validate `BlockRefinementScheduler` and
   `LLaDA2Pipeline` tensor shapes without compiling the external model.
2. Stage 2: compile a small synthetic LLaDA2-like dense layer or one decoder
   block with externally supplied weights. Start with dense layer 0 MLP and
   eager attention.
3. Stage 3: port one full mini decoder layer including QK RMSNorm, RoPE,
   bidirectional GQA attention, residuals, and dense/SwiGLU MLP.
4. Stage 4: add MoE layer support: router, group-limited top-k, selected expert
   MLP dispatch, shared expert, combine.
5. Stage 5: run one refinement step parity with externally supplied `block_x`,
   all-ones mask, fixed position ids, and model logits compared against the
   remote PyTorch model.
6. Stage 6: full block loop with scheduler in Python, deterministic greedy
   sampling, no editing.
7. Stage 7: move deterministic scheduler update into compiled/runtime kernels.
8. Stage 8: add Flash/SDPA attention provider variants and benchmark eager
   parity versus optimized attention.
9. Stage 9: add stochastic sampling, top-p/top-k filtering, editing mode, and
   optional offload/encoded-weight staging.

Stub initially: tokenizer/chat template, final decode, callbacks, progress
bars, PEFT/adapters, offload hooks, stochastic sampling, and any-to-any Uni
models.

## 14. Parity and validation plan

- Scheduler unit tests: reproduce Diffusers tests for `set_timesteps`,
  transfer schedule distribution, forced top-k commits, editing, prompt-mask
  edit prevention, EOS finished checks, and `add_noise`.
- Pipeline dummy-model tests: match `tests/pipelines/llada2/test_llada2.py`
  for output shapes, no mask tokens after forced deterministic generation,
  tuple/dict returns, tokenizer/no-tokenizer paths.
- Remote model primitive tests: RMSNorm, RoPE, QKV split, eager attention, dense
  MLP, router top-k, expert combine.
- Single block parity: one mini decoder layer, then full mini model on a small
  prompt/window length with `output_router_logits=false`, `use_cache=false`.
- One denoising step parity: fixed `block_x`, fixed logits or model output,
  `temperature=0`, no editing, compare `prev_sample`, `transfer_index`,
  `sampled_tokens`, and `sampled_probs`.
- Short loop parity: `gen_length=16 or 32`, `block_length=8 or 32`,
  `threshold > 1`, deterministic dummy or tiny model.
- Config parity: reject or clearly fence block-length mismatches between
  pipeline argument and scheduler config.
- Tolerances: model logits fp32 reference for small random blocks; bf16/fp16
  optimized kernels should use normal transformer tolerances, then exact token
  equality after scheduler argmax/commit.

## 15. Performance probes

- Transformer forward time by `(B,E)` window length: 32, 64, 128, 512, 2048,
  4096, and long-context cases.
- Full generation loop by `(gen_length, block_length, num_inference_steps)`:
  count total model forward tokens recomputed.
- Attention backend comparison: eager, SDPA, FlashAttention2/fused provider,
  all-ones mask versus padded mask.
- MoE dispatch probes: router time, token distribution by expert, grouped GEMM
  occupancy, combine kernel time, shared expert overhead.
- LM head cost for full-window logits versus last-block-only logits.
- Scheduler overhead: greedy deterministic, top-k, top-p, multinomial, editing.
- VRAM and weight residency: mini 32.5 GB safetensors and flash 205.8 GB
  safetensors imply offload/quantized-weight staging will matter.
- RoPE precompute/cache savings across repeated block steps.

## 16. Scope boundary and separate candidates

Separate candidate reports related to this family:

- `llada2_moe_transformer`: external remote-code model architecture and weight
  loading, including MoE dispatch and attention provider choices.
- `block_refinement_scheduler`: discrete token scheduler as a reusable
  non-image diffusion runtime loop.
- `llada2_flash_attention_variants`: FlashAttention2/SDPA provider admission
  for bidirectional GQA + RoPE + QK norm.
- `llada2_any_to_any_uni`: `inclusionAI/LLaDA2.0-Uni` and FP8 related repos,
  not covered by the current Diffusers text pipeline.
- `llada2_offload_weight_residency`: group/sequential offload and future
  encoded/quantized weight loading for very large MoE weights.
- `llada2_adapter_mutation`: external Transformers/PEFT adapter and tokenizer
  mutation surfaces if users pair adapters with the remote model.

Genuinely ignored/out of scope for this audit:

- Diffusers callback mutation and progress bar behavior.
- Training/loss/dropout/gradient checkpointing, including scheduler `add_noise`
  except as parity reference.
- Multi-GPU/context parallel, XLA/NPU/MPS/Flax/ONNX.
- Image/video/audio VAEs, processors, safety checkers, ControlNet, IP-Adapter,
  T2I-Adapter, GLIGEN, img2img, inpaint, depth2img, and upscaling; they do not
  exist in the selected non-deprecated `llada2` pipeline folder.

## 17. Final implementation checklist

- [ ] Parse official `llada2_moe` config and tokenizer special-token metadata.
- [ ] Add a scheduler-only parity harness for `BlockRefinementScheduler`.
- [ ] Admit fixed block-generation runtime schema: `input_ids`, `mask_token_id`,
  `block_length`, `gen_length`, `num_inference_steps`.
- [ ] Guard scheduler config `block_length` against pipeline `block_length`.
- [ ] Implement/tokenize CPU data boundary or accept pre-tokenized `input_ids`
  first.
- [ ] Lower embedding, RMSNorm, RoPE, packed QKV, bidirectional GQA attention,
  residual, dense SwiGLU, and LM head.
- [ ] Add MoE router/group-limited top-k/expert-dispatch/combine provider.
- [ ] Add deterministic greedy scheduler commit kernel.
- [ ] Validate one decoder layer and one full mini forward against remote
  PyTorch code.
- [ ] Validate one refinement step and short deterministic loop.
- [ ] Add Flash/SDPA attention provider candidate with eager fallback.
- [ ] Benchmark attention, MoE dispatch, LM-head last-block optimization, and
  full generation loop.
