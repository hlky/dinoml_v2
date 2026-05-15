# Transformers `pi0` Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: source default targets lerobot/pi0_base; accessible Hub repos are LeRobot policy repos, not native Transformers PI0 configs.
Config source: PI0Config source defaults plus LeRobot Hub policy config sweep.
Source files inspected:
- transformers/src/transformers/models/pi0/configuration_pi0.py
- transformers/src/transformers/models/pi0/modeling_pi0.py
- transformers/src/transformers/models/pi0/processing_pi0.py
- transformers/src/transformers/models/pi0/image_processing_pi0.py
- transformers/src/transformers/models/pi0/modular_pi0.py
- Nested families: paligemma, gemma, siglip configuration/modeling files.
Any missing files or assumptions:
- `configuration_pi0.py`, `modeling_pi0.py`, `processing_pi0.py`, and `image_processing_pi0.py` are generated from `modular_pi0.py`; future source edits should target `modular_pi0.py`.
- Public `lerobot/pi0_*` Hub `config.json` files are LeRobot policy configs. They do not contain native Transformers `model_type="pi0"` nested configs.
- `google/paligemma-3b-pt-224` is manually gated; Hub API metadata was visible, but raw config/preprocessor access returned an authentication/license error.
```

Pinned source URLs:

- [`configuration_pi0.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pi0/configuration_pi0.py)
- [`modeling_pi0.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pi0/modeling_pi0.py)
- [`processing_pi0.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pi0/processing_pi0.py)
- [`image_processing_pi0.py`](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pi0/image_processing_pi0.py)

Small snapshots written beside this report:

- `source_defaults_snapshot.md`
- `hub_config_sweep_snapshot.md`

## 2. High-level architecture

PI0 is a vision-language-action flow-matching model, not an autoregressive text-generation model. The first useful DinoML runtime target should be `sample_actions`: given camera images, tokenized language prompt, and robot state, produce a continuous action chunk.

Source dataflow:

```text
CPU/data preprocessing
  -> image tensor packing + text placeholders + state normalization
  -> PaliGemma/SigLIP vision encoder + projection
  -> image feature stitch into Gemma text embeddings
  -> VLM prefix prefill to KV cache
  -> repeated Gemma-DiT action denoising steps
  -> action velocity projection
  -> Euler flow update
  -> continuous action tensor
```

Stage decomposition:

- CPU/data pipeline: resize/rescale/normalize/pad images to NCHW, tokenize prompts with repeated `<image>` placeholders, normalize/pad state/action vectors.
- Vision/projector stage: flatten `[B, Cams, 3, H, W]` to `[B*Cams, 3, H, W]`; SigLIP patch embed + encoder; PaliGemma projector maps vision hidden width to VLM text hidden width.
- Prefix construction: remove padded cameras with `pixel_attention_mask`, embed text ids, replace placeholder rows with image features using `masked_scatter`.
- VLM prefix cache: run PaliGemma/Gemma text body once with `use_cache=True`; cache is independently reusable across denoising steps for the same image/text prompt.
- DiT denoising: create state token plus `chunk_size` action/noise tokens, run Gemma body with a custom block mask against the VLM prefix cache, project final `chunk_size` hidden states to action velocity.
- Flow loop: `num_inference_steps` Euler updates; after each DiT call the source crops cache back to prefix length, so denoising steps do not accumulate KV state.

## 3. Important config dimensions

Source-default PI0 dimensions:

| Component | Field | Value |
| --- | --- | --- |
| PI0 | `chunk_size` | 50 |
| PI0 | `max_state_dim`, `max_action_dim` | 32, 32 |
| PI0 | `num_inference_steps` | 10 |
| PI0 | timestep periods | 0.004 to 4.0 |
| VLM text Gemma | hidden/layers/intermediate | 2048 / 18 / 16384 |
| VLM text Gemma | heads/KV heads/head_dim | 8 / 1 / source default Gemma `head_dim=256` |
| VLM text Gemma | vocab | 257152 |
| VLM vision SigLIP | hidden/layers/intermediate | 1152 / 27 / 4304 |
| VLM vision SigLIP | heads/patch/image | 16 / 14 / 224 |
| VLM projector | Linear | 1152 -> 2048, bias true |
| DiT Gemma | hidden/layers/intermediate | 1024 / 18 / 4096 |
| DiT Gemma | heads/KV heads/head_dim | 8 / 1 / 256 |
| DiT action head | Linear | 1024 -> 32 |
| Attention flags | VLM text and DiT | bidirectional attention enabled by PI0 config |

Representative checkpoint/config sweep:

| Repo/config | Native Transformers config? | Visual inputs | State/action dims | Dtype metadata | Notes |
| --- | --- | --- | --- | --- | --- |
| `PI0Config()` source default | yes | variable cameras, processor pads to 224 | max 32/32 | source dtype defaults inherited | Basis for this runtime audit |
| `lerobot/pi0_base` | no, LeRobot policy | 3 x `[3,224,224]` | 32/32 | float32 | 50 action steps; 3.501B F32 params by Hub metadata |
| `lerobot/pi0_libero_base` | no, LeRobot policy | 2 x `[3,256,256]`, image resolution 224 | 8/7 padded to 32 | float32 | `empty_cameras=1`, `n_action_steps=10` |
| `lerobot/pi0_libero_finetuned_v044` | no, LeRobot policy | 2 x `[3,256,256]` plus empty `[3,224,224]` | 8/7 padded to 32 | bfloat16/F32 mix | compile and gradient checkpointing flags are training/runtime metadata |
| `lerobot/pi0_old` | no, older LeRobot policy | 3 x `[3,480,640]`, resized/padded to 224 | 6/6 padded to 32 | not native | older fields include `proj_width=1024`, `attention_implementation="eager"` |

## 3a. Family variation traps

- Native Transformers PI0 delegates most neural body coverage to PaliGemma, Gemma, and SigLIP. DinoML should compose those audits rather than treating PI0 as a single flat architecture.
- `hidden_size != num_attention_heads * head_dim` for source-default DiT: 1024 hidden, 8 heads, `head_dim=256`, so Q/O width is 2048. Do not infer projection widths from hidden size.
- Both VLM text and DiT use GQA/MQA-style KV sharing: 8 query heads, 1 KV head, `repeat_kv` expansion before eager attention.
- PI0 forces `use_bidirectional_attention=True` for the VLM text and DiT configs, while DiT also sets `is_causal=True`. The effective mask comes from `create_causal_mask` plus `block_sequence_ids`, not from a plain triangular decoder mask.
- Image placeholder replacement uses broad `masked_scatter`, but the processor emits a stricter pattern: a contiguous run of `<image>` tokens before BOS/text, repeated `image_seq_length * num_cameras`.
- Processor and LeRobot configs can declare camera source sizes larger than 224, but PI0 image processor pads/resizes to 224 for the model path inspected here. Source semantics remain NCHW.
- `PI0ImageProcessor` class defaults expose `size` as `max_height/max_width`, while `PI0Processor.__init__` reads `height/width`; saved processor configs may normalize this, but DinoML should validate processor config keys instead of assuming bare class defaults are loadable.
- `pixel_attention_mask` removes padded cameras by boolean indexing before image features are concatenated. A lowerer needs a bounded packed-camera path or a rejection guard for variable camera counts.
- `sample_actions` mutates cache state by cropping back to the VLM prefix length after every denoising iteration.
- LeRobot `config.json` files include training, optimizer, compile, normalization, and robot I/O metadata. Do not treat those as native Transformers graph fields unless integrating the LeRobot policy wrapper.
- Official PaliGemma checkpoint configs are gated; exact nested checkpoint variations such as PaliGemma2 sliding-window fields require authenticated verification.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image preprocessing output `[B, Cams, 3, 224, 224]`.
- `flatten(0, 1)` for cameras, reshape back to `[B, Cams, Seq_img, H_vlm]`.
- Boolean camera selection from `[B, Cams, Seq_img, H_vlm]` using `pixel_attention_mask`.
- `clone`, equality mask, `unsqueeze`, `expand_as`, `masked_scatter` or guarded indexed row copy for image stitch.
- `cat` for time/action embeddings, state/action token merge, mask merge, block ids.
- `cumsum`, `arange`, scalar tensor creation, `repeat`, cache length query, cache crop.
- Last-token/chunk slice `hidden[:, -chunk_size:]`.

Neural network primitives:

- Embedding lookup with Gemma scaled word embeddings: token embedding multiplied by `sqrt(hidden_size)`.
- SigLIP patch embedding: `Conv2d(3 -> 1152, kernel=14, stride=14, padding=valid)` for source default.
- SigLIP noncausal encoder: LayerNorm, dense MHA, GELU/Tanh MLP, residual adds.
- PaliGemma projector: `Linear(1152 -> 2048, bias=True)`.
- Gemma RMSNorm with `1 + weight`, fp32 norm math, output cast to input dtype.
- Gemma gated MLP: `down_proj(act(gate_proj(x)) * up_proj(x))`, bias false.
- PI0 action/time MLP: `Linear(32 -> 1024)`, `Linear(32 -> 1024)`, `Linear(2048 -> 1024)`, SiLU, `Linear(1024 -> 1024)`.
- Output head: `Linear(1024 -> 32)`.

Attention primitives:

- SigLIP dense bidirectional MHA, no KV cache.
- Gemma self-attention with GQA/MQA: Q `hidden -> heads*head_dim`, K/V `hidden -> kv_heads*head_dim`, O `heads*head_dim -> hidden`.
- RoPE applied to Q/K before cache update.
- Cache update and reuse for VLM prefix and DiT action tokens.
- `create_causal_mask` with block sequence ids. This is a required mask family, not a generic lower-triangular mask.
- SDPA/Flash/Flex backends are source-supported, but eager math is the clear parity baseline.

Position/rotary/custom math:

- SigLIP learned absolute patch position embeddings; optional bicubic interpolation path exists in SigLIP but PI0 source calls without interpolation.
- Gemma RoPE with `rope_theta` from config defaults, fp32 frequency math, cos/sin cast to hidden dtype.
- PI0 sinusoidal timestep embedding over DiT hidden size.

Preprocessing-coupled ops:

- Image resize/rescale/normalize/RGB/pad in data pipeline.
- Tokenizer insertion of repeated `<image>` placeholders, BOS token, prompt, newline.
- State/action mean/std normalization and zero padding to max dims.
- Action output unnormalization belongs to policy postprocessing, not the neural graph.

Scatter/indexed update ops:

- Required first path: replace placeholder token embedding rows with image feature rows.
- Safe lowering candidate: verify placeholder count equals `sum(pixel_attention_mask) * image_seq_length`, placeholder positions are contiguous prefix rows per sample, and feature flatten order matches source camera-major then patch-major order.
- Fallback/reject: arbitrary boolean scatter with non-prefix placeholder positions.

Generation/cache/state ops:

- VLM prefix KV cache.
- DiT cache update during each denoising call, then crop back to prefix length.
- No token sampling, logits processors, beam search, or text decode are needed for the primary target.

Quantized/packed/distributed ops:

- No source-coupled quantized weight format in PI0 Transformers source.
- Tensor-parallel plans exist for Gemma/PaliGemma/PI0 modules, but first DinoML integration can defer multi-GPU.

## 5. Layer/block breakdown

Vision branch, source default:

```text
pixel_values: [B*Cams, 3, 224, 224]
patch = Conv2d(3 -> 1152, kernel=14, stride=14)(pixel_values)
tokens = flatten spatial -> transpose to [B*Cams, 256, 1152]
tokens += learned_position_embedding[256, 1152]
repeat 27 times:
  y = LayerNorm(tokens)
  y = MHA(q/k/v/o: 1152 -> 1152, 16 heads, head_dim 72)
  tokens = tokens + y
  y = LayerNorm(tokens)
  y = Linear(1152 -> 4304) -> GELU/Tanh -> Linear(4304 -> 1152)
  tokens = tokens + y
tokens = post LayerNorm(tokens)
image_features = Linear(1152 -> 2048, bias=True)(tokens)
```

VLM text Gemma prefix, source default:

```text
text/image embeddings: [B, S_prefix, 2048]
position_ids = attention_mask.cumsum(-1) - 1
token_type_ids = zeros for prefix
repeat 18 times:
  y = RMSNorm(x)
  q = Linear(2048 -> 2048, bias=False)
  k = Linear(2048 -> 256, bias=False)
  v = Linear(2048 -> 256, bias=False)
  q,k = RoPE(q,k)
  k,v = cache.update(k,v)
  y = GQA attention(q, repeat_kv(k), repeat_kv(v), block mask)
  y = Linear(2048 -> 2048, bias=False)
  x = x + y
  y = RMSNorm(x)
  y = Linear(2048 -> 16384) -> gelu_pytorch_tanh * Linear(2048 -> 16384)
  y = Linear(16384 -> 2048)
  x = x + y
prefix_cache = per-layer K/V
```

Action/time embedding:

```text
state: [B, 32]
noise: [B, 50, 32]
timestep: [B]
state_emb = Linear(32 -> 1024)(state)
action_emb = Linear(32 -> 1024)(noise)
time_emb = sin/cos timestep embedding [B, 1024], expand to [B, 50, 1024]
action_time = cat(action_emb, time_emb) -> Linear(2048 -> 1024) -> SiLU -> Linear(1024 -> 1024)
action_embeds = cat(state_emb[:,None,:], action_time) -> [B, 51, 1024]
```

DiT Gemma denoiser:

```text
action_embeds: [B, 51, 1024]
dit_attention_mask = cat(prefix attention_mask, ones([B, 51]))
dit_position_ids = cumsum(dit_attention_mask)[:, -51:]
block_sequence_ids = zeros(prefix_len + 1) + ones(50)
repeat 18 times:
  y = RMSNorm(x)
  q = Linear(1024 -> 2048, bias=False)
  k = Linear(1024 -> 256, bias=False)
  v = Linear(1024 -> 256, bias=False)
  q,k = RoPE(q,k)
  k,v = cache.update(k,v) with VLM prefix cache already present
  y = GQA attention with custom bidirectional/block mask
  y = Linear(2048 -> 1024, bias=False)
  x = x + y
  y = RMSNorm(x)
  y = gated MLP 1024 -> 4096 -> 1024
  x = x + y
last_hidden = x[:, -50:, :]
velocity = Linear(1024 -> 32)(last_hidden)
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention over 256 patch tokens for default 224/14.
- MHA, 16 heads, hidden 1152, head_dim inferred as 72.
- No KV cache.
- Standard additive mask path exists but PI0 vision call uses no attention mask.

VLM text prefix attention:

- Gemma self-attention, bidirectional prefix behavior controlled by `use_bidirectional_attention=True` and PaliGemma token type/block ids.
- GQA/MQA: 8 Q heads, 1 KV head, head_dim 256, Q width 2048, K/V width 256.
- Cached K/V are stored after RoPE because `apply_rotary_pos_emb` precedes `past_key_values.update`.
- Prefix cache shape per layer, source default: K and V `[B, 1, S_prefix, 256]` before repeat expansion.
- Eager attention repeats K/V to `[B, 8, S, 256]`, computes softmax in fp32, casts probabilities back to query dtype.

DiT action attention:

- Gemma self-attention over action tokens with existing prefix cache as past.
- Query length is 51 (`state + chunk_size`), key/value length is `S_prefix + 51` during each denoising call.
- DiT per-layer new K/V before repeat: `[B, 1, 51, 256]`.
- Mask is built from `block_sequence_ids`: prefix plus state are one block, actions another block. This needs direct parity tests against `create_causal_mask`.
- After each flow step, source calls `past_key_values.crop(prefix_length)`, removing action-step K/V.

Backend compatibility:

- `_supports_flash_attn`, `_supports_sdpa`, and `_supports_flex_attn` are true on PI0, PaliGemma, and Gemma, but first DinoML parity should use explicit dense attention math plus source mask equivalence.
- FlashAttention optimization is attractive only after block-mask semantics and rectangular prefix/action attention are admitted.

## 7. Position encoding and custom math

Gemma RoPE:

```python
def gemma_rope(q, k, position_ids, rope_theta, head_dim):
    inv = 1.0 / (rope_theta ** (arange(0, head_dim, 2).float() / head_dim))
    freqs = inv[None, :, None] @ position_ids[:, None, :].float()
    emb = cat([freqs.transpose(1, 2), freqs.transpose(1, 2)], dim=-1)
    cos, sin = emb.cos(), emb.sin()
    def rotate_half(x):
        return cat([-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]], dim=-1)
    return q * cos[:, None] + rotate_half(q) * sin[:, None], k * cos[:, None] + rotate_half(k) * sin[:, None]
```

PI0 timestep embedding:

```python
def pi0_timestep_embedding(t, hidden_size, min_period=0.004, max_period=4.0):
    fraction = linspace(0.0, 1.0, hidden_size // 2, dtype=float32)
    period = min_period * (max_period / min_period) ** fraction
    freq = 1.0 / period * 2 * pi
    emb = freq[None, :] * t[:, None]
    return cat([sin(emb), cos(emb)], dim=1)
```

Precomputable:

- RoPE inverse frequencies and timestep frequencies.
- SigLIP absolute patch position embedding for static 224 resolution.

Dynamic-input dependent:

- RoPE cos/sin indexed by prefix/action `position_ids`.
- PI0 timestep embedding changes every denoising step.
- Block attention mask depends on prefix length, action length, and cache length.

## 8. Preprocessing and input packing

Processor contract:

- Text defaults: right padded to length 48.
- For each sample, prompt is `<image>` repeated `image_seq_length * num_cameras`, then BOS token, prompt text, newline.
- Images are nested per sample, padded to the batch max camera count.
- `pixel_values` shape emitted by PI0 processor: `[B, max_num_cameras, 3, height, width]`.
- `pixel_attention_mask` shape: `[B, max_num_cameras]`, true for real cameras and false for padded cameras.
- `state` is normalized by processor mean/std, padded to `max_state_dim`, and reshaped to `[B, 32]`.
- `actions` are normalized similarly and reshaped to `[B, chunk_size, 32]`; actions are training labels, not required for inference.

Image placeholder stitch:

- Source computes image features for all padded cameras, reshapes to `[B, max_cameras, image_seq_length, hidden]`, then boolean-selects real cameras and concatenates.
- Text ids equal to `image_token_id` are set to 0 before token embedding if the image token id is outside the text embedding vocab.
- `masked_scatter` writes image features into embedding positions selected by `input_ids == image_token_id`.
- Processor-generated placeholders are contiguous prefix rows and camera-major; DinoML can lower this as indexed row copy under guards.

Layout:

- Source image tensors are NCHW. Treat NHWC as an optimization only inside a controlled image patch embedding/layout-fusion region.
- Safe layout candidate: NCHW Conv2d patch embedding -> flatten/transpose -> token sequence. A channel-last lowering must rewrite Conv input layout, weight access, flatten order, and downstream token order together.
- No-layout guard: text/action Gemma bodies are sequence-major `[B, S, H]`; do not apply image layout translation beyond the vision-token boundary.

## 9. Graph rewrite / lowering opportunities

### Rewrite: SigLIP patch Conv2d -> Linear

Source pattern:

```text
Conv2d(C=3 -> H_v, kernel=patch, stride=patch, padding=valid)
-> flatten(2)
-> transpose(1, 2)
```

Replacement:

```text
WindowFlatten_NCHW_nonoverlap -> MatMul([patch*patch*3, H_v]) -> BiasAdd -> [B, N_patches, H_v]
```

Preconditions:

- `kernel_size == stride == patch_size`
- padding is valid/zero
- dilation 1
- groups 1
- input height/width divisible by patch size
- preserve source NCHW flatten order

Failure cases:

- `interpolate_pos_encoding=True` with non-default image sizes needs bicubic position interpolation.
- LeRobot raw camera shapes such as 256 or 480x640 must be resolved by preprocessing to model size before this rewrite.

Parity test sketch:

- Compare Conv2d output flattened/transposed against WindowFlatten+Linear for random `[B,3,224,224]` fp32/bf16 inputs.

### Rewrite: placeholder `masked_scatter` -> guarded row copy

Source pattern:

```text
special_image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, total_image_features)
```

Replacement:

```text
validate placeholder counts and contiguous prefix layout
copy image_features.reshape(total_placeholders, hidden) into embedding rows
```

Preconditions:

- Placeholder count equals selected camera count times `image_seq_length`.
- Placeholder positions are contiguous prefix runs per sample in processor order.
- Hidden width of image features equals embedding hidden width.

Failure cases:

- User-supplied `input_ids` with arbitrary image token positions.
- Mixed prompt construction outside `PI0Processor`.

### Rewrite: separate Gemma Q/K/V linears -> packed QKV launch

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement:

```text
one grouped/packed GEMM producing [Q, K, V] segments, then split as Q rows, K rows, V rows
```

Preconditions:

- Bias flags match; default Gemma attention bias is false.
- Split order is Q then K then V.
- Output widths are explicit: Q = `num_heads * head_dim`, K/V = `num_kv_heads * head_dim`.
- Weight packing preserves individual parameter identity for loading/debug metadata.

Failure cases:

- Non-default attention bias or changed nested model family.
- Tensor-parallel sharded weights unless the sharding plan owns the packed layout.

### Rewrite: flow loop prefix-cache hoist

Source pattern:

```text
embed_prefix + VLM(use_cache=True) once
for denoising step:
  run DiT with same prefix cache
  crop cache back to prefix_length
```

Replacement:

```text
cacheable prefix artifact + repeated DiT artifact with explicit prefix cache input/output
```

Preconditions:

- Same images, prompt, attention mask, and state normalization across flow loop.
- Cache crop semantics implemented and tested.

Failure cases:

- User passes external `past_key_values` with unknown length/layout.
- Prompt/image changes during action sampling.

## 10. Kernel fusion candidates

Highest priority:

- Gemma RMSNorm with `1 + weight`, fp32 accumulation, cast-back.
- GQA attention with RoPE and KV cache for VLM prefix and DiT action tokens.
- Q/K/V GEMM packing for Gemma attention, especially DiT `1024 -> 2048/256/256` projections.
- SwiGLU/GELU-gated MLP multiply chain for Gemma: gate projection activation times up projection.
- Prefix image stitch row-copy kernel, avoiding general boolean scatter.

Medium priority:

- SigLIP patch embedding Conv2d-to-GEMM for 224x224 static images.
- SigLIP LayerNorm + MHA + MLP encoder fusions once vision cost is measured.
- PI0 action/time embedding MLP, including timestep sin/cos generation and broadcast.
- Last-chunk projection `Linear(1024 -> 32)` over only action tokens.

Lower priority:

- Bicubic position interpolation for non-224 SigLIP paths.
- Multihead attention pooling head in SigLIP, because PI0 default sets `vision_use_head=False`.
- Training loss and Beta/random timestep sampling.
- Tensor-parallel sharding.

## 11. Runtime staging plan

Stage 1: Config and ABI admission.

- Parse PI0 source-style config with nested PaliGemma/Gemma/SigLIP configs.
- Admit fixed first target: `sample_actions` with processor-controlled inputs, static 224 images, max 3 cameras, chunk 50, state/action max dim 32.
- Reject arbitrary remote-code or LeRobot-only configs for the Transformers PI0 path.

Stage 2: Vision/projector parity.

- Implement or compose SigLIP vision encoder and PaliGemma projector.
- Validate `[B*Cams,3,224,224] -> [B*Cams,256,2048]`.

Stage 3: Prefix stitch and VLM cache.

- Implement guarded image row copy into text embeddings.
- Run VLM Gemma prefix to cache with source-equivalent bidirectional prefix mask.

Stage 4: One DiT denoising step.

- Implement action/time embeddings, DiT block mask, DiT Gemma body, action velocity head.
- Validate one step with fixed noise and timestep.

Stage 5: Full flow loop.

- Add 10-step Euler update and cache crop.
- Return normalized action tensor first; add policy unnormalizer later if integrating LeRobot end-to-end.

Stage 6: Optimized kernels.

- Enable RMSNorm, QKV packing, GQA FlashAttention/block-mask path, patch Conv2d GEMM rewrite, and row-copy stitch.

Stubs acceptable initially:

- Training loss, random noise/timestep generation, action labels, gradient checkpointing, compile flags, and policy postprocessing.

## 12. Parity and validation plan

- Timestep embedding parity: compare sinusoid frequencies and outputs for fixed timesteps in fp32 and bf16/fp16 cast paths.
- Placeholder stitch parity: random embeddings/features with processor-shaped placeholders and variable camera masks.
- SigLIP patch embedding parity: Conv2d vs lowered GEMM.
- Single SigLIP encoder layer parity.
- Single Gemma layer parity for both VLM hidden 2048 and DiT hidden 1024, including GQA widths.
- Mask parity: compare DinoML mask tensors against `create_causal_mask` for prefix lengths, action length 51, and padding masks.
- Prefix cache parity: run PaliGemma prefix and compare per-layer K/V shapes and selected values.
- One DiT step parity: fixed state, noise, timestep, prefix cache.
- Full `sample_actions` parity: fixed noise and deterministic inputs, compare final normalized action tensor.
- Tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 `rtol=5e-2, atol=5e-2` for full model initially, tighter per-op tolerances where reductions are controlled.

## 13. Performance probes

- Processor throughput: image resize/pad/tokenization/state normalization separately from GPU graph.
- Vision encoder throughput by batch cameras: `B*Cams = 1, 2, 3, 6, 12`.
- Prefix stitch overhead: boolean scatter vs guarded row-copy.
- VLM prefix prefill latency and cache memory by prompt length and camera count.
- DiT one-step latency for query length 51 and rectangular key length `S_prefix + 51`.
- Full flow loop latency for `num_inference_steps = 5, 10, 20`.
- Cache crop/update overhead.
- Attention backend comparison: eager dense, SDPA, FlashAttention-compatible block mask fallback.
- GEMM profile sweep for Gemma projections with explicit nonstandard widths: `1024x2048`, `1024x256`, `2048x2048`, `2048x256`, MLP projections.
- End-to-end actions/sec with processor included and excluded.

## 14. Skip/defer list

- Training path: action labels, MSE loss, Beta timestep sampling.
- Random noise generation inside compiled graph; accept caller-provided noise for deterministic first target.
- Beam search, token sampling, logits processors, and text generation heads.
- Multi-GPU tensor parallel plans.
- SigLIP `vision_use_head=True` pooling head.
- SigLIP position interpolation for arbitrary image sizes.
- PaliGemma2 sliding-window variants until gated configs are verified.
- General boolean `masked_scatter`; use guarded image row copy first.
- LeRobot policy wrapper normalization/unnormalization as part of core Transformers PI0 graph.
- Quantized/packed weight loading unless a checkpoint-specific audit requires it.

## 15. Final implementation checklist

- [ ] Parse `PI0Config` with nested PaliGemma, Gemma, and SigLIP configs.
- [ ] Add admission guards for source-default/static first target: 224 images, processor-controlled placeholders, bounded camera count, chunk 50.
- [ ] Compose or audit SigLIP vision encoder coverage.
- [ ] Compose or audit PaliGemma projector and Gemma text prefix coverage.
- [ ] Implement Gemma RMSNorm `1 + weight` semantics.
- [ ] Implement Gemma RoPE with explicit `head_dim`.
- [ ] Implement GQA/MQA attention with KV cache before repeat expansion.
- [ ] Implement PI0 timestep embedding.
- [ ] Implement PI0 action/time embedding and output velocity projection.
- [ ] Implement processor-bounded image placeholder row-copy rewrite.
- [ ] Implement PI0 block mask parity for prefix/state/action attention.
- [ ] Implement cache crop back to prefix length after each denoising step.
- [ ] Add SigLIP patch Conv2d-to-GEMM rewrite with NCHW guards.
- [ ] Add one-step DiT parity test.
- [ ] Add full `sample_actions` parity test with fixed noise.
- [ ] Benchmark vision, prefix prefill, one DiT step, and full flow loop separately.
