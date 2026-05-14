# EXAONE MoE source notes

Audit date: 2026-05-13

## Local source basis

- Transformers checkout: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Model directory: `X:/H/transformers/src/transformers/models/exaone_moe`
- Files inspected:
  - `configuration_exaone_moe.py`
  - `modeling_exaone_moe.py`
  - `modular_exaone_moe.py`
  - `__init__.py`

`configuration_exaone_moe.py` and `modeling_exaone_moe.py` are generated from
`modular_exaone_moe.py`. Future upstream edits should be made in the modular
file, but the generated files are the concrete runtime/import source basis for
this audit.

## Representative config fetches

Primary config:

- `https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B/raw/main/config.json`
- Model type: `exaone_moe`
- Architecture string: `ExaoneMoEForCausalLM`
- Shape: 48 layers, hidden 6144, 64 query heads, 8 KV heads, head_dim 128
- MoE: 128 routed experts, top-8 experts per token, one shared expert,
  `moe_intermediate_size=2048`, first layer dense and remaining layers sparse
- Attention: `sliding_window=128`, `sliding_window_pattern="LLLG"`,
  explicit layer_types with three sliding layers then one full layer repeated
- RoPE: `rope_type="default"`, `rope_theta=1000000`,
  `max_position_embeddings=262144`
- Dtype: `bfloat16`
- Tokenizer metadata: config says `GPT2Tokenizer`, tokenizer_config fetched from
  the same repo says `PreTrainedTokenizerFast`; both are source facts from HF
  metadata rather than modeling code behavior.

Historical official config:

- `https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B/raw/2159cb4eae487475ad360f6b028a558d586c4f61/config.json`
- Same operator-significant dimensions as current main for the neural graph.
- Notable historical field: `first_last_k_dense_replace` instead of current
  `first_k_dense_replace`; the inspected config class reads
  `first_k_dense_replace`, not `first_last_k_dense_replace`.

Quantized mirror config:

- `https://huggingface.co/mlx-community/K-EXAONE-236B-A23B-6bit/raw/main/config.json`
- Same neural graph shape as the official config.
- Adds MLX-style `quantization` / `quantization_config` with
  `bits=6`, `group_size=64`, `mode="affine"`. The inspected Transformers
  modeling source does not read those fields, so DinoML should treat this as a
  separate loading/provider contract, not a graph op requirement.

Generation/tokenizer metadata:

- `generation_config.json` from the official repo: `do_sample=true`,
  `top_p=0.95`, `temperature=1.0`, `bos_token_id=1`, `eos_token_id=53`,
  `pad_token_id=0`.
- `tokenizer_config.json` includes chat/tool/vision-looking special tokens, but
  the `exaone_moe` modeling source is text-only and does not consume modality
  tensors or placeholder embedding scatter.

## Source facts to preserve

- RMSNorm computes variance in fp32 over the last dimension, multiplies learned
  weight after casting normalized values back to the input dtype.
- Attention projections are unfused, bias-free: q/o widths are
  `num_attention_heads * head_dim`; k/v widths are
  `num_key_value_heads * head_dim`.
- QK RMSNorm is applied per head before RoPE/cache/attention.
- RoPE is applied only when `sliding_window is None` or the layer is a
  `sliding_attention` layer. Full-attention layers in the official hybrid config
  use NoPE for Q/K.
- Cache update occurs after optional RoPE, so cached keys store the post-RoPE
  keys for sliding layers and NoPE keys for full layers.
- Eager attention repeats KV heads to query-head count, computes scaled QK,
  adds mask, softmaxes in fp32, drops out only in training, then matmuls V.
- MoE routing uses fp32 router linear, sigmoid scores, optional correction bias
  only for expert choice, group top-k selection, expert top-k, gather original
  sigmoid scores, optional top-k normalization, and `routed_scaling_factor`.
- Routed experts store packed 3D weights:
  `gate_up_proj[num_experts, 2 * moe_intermediate_size, hidden_size]` and
  `down_proj[num_experts, hidden_size, moe_intermediate_size]`.
- Eager expert fallback loops only over hit experts, gathers token rows,
  computes packed gate/up linear, chunks gate then up, applies SiLU(gate) * up,
  down-projects, scales by token/expert route weight, and scatter-adds with
  `index_add_`.
- Shared expert is dense SwiGLU with intermediate
  `moe_intermediate_size * num_shared_experts`, added to the routed expert sum.
- `_keys_to_ignore_on_load_unexpected = [r"mtp.*"]`; the config includes
  `num_nextn_predict_layers=1`, but the inspected model does not implement an
  MTP head. Treat MTP weights as ignored for this source basis.

## Source gaps / gated facts

- Only one official EXAONE MoE model shape was found during this audit. No
  small/debug or alternate official in-library checkpoint config was found.
- The official repo is accessible without auth for config/tokenizer metadata.
  Large safetensors were not inspected, because this audit does not load or
  import model code.
- HF mirrors with 6-bit/6.5-bit MLX configs are not native Transformers runtime
  quantization evidence; they are useful loading-policy signals only.
- Hub kernel decorators are present for RMSNorm, RoPE, and expert implementation
  hooks. This report treats the Python generated source as the semantic basis;
  optional hub kernels are optimization/provider substitutes that must prove
  parity against these semantics.
