# NanoChat source anchors

Transformers checkout: `transformers`
Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Authoritative source relationship:

- `src/transformers/models/nanochat/modeling_nanochat.py` is the generated runtime file used by installed Transformers.
- `src/transformers/models/nanochat/modular_nanochat.py` is the edit source; the generated file header says edits should go there.

Key anchors from `modeling_nanochat.py`:

- `NanoChatRMSNorm`: line 45. Parameter-free RMS/L2 norm, fp32 internal math, no learned scale.
- `NanoChatRotaryEmbedding`: line 60. Reads `config.rope_parameters["rope_type"]` and `["rope_theta"]`; default RoPE dim is `head_dim` or `hidden_size // num_attention_heads`.
- `apply_rotary_pos_emb`: line 126. Applies cos/sin with `unsqueeze_dim=1` to `[B,H,T,D]`.
- `repeat_kv`: line 151. Expands KV heads from `[B,Hkv,T,D]` to `[B,Hq,T,D]` for eager GQA/MQA fallback.
- `eager_attention_forward`: line 163. Matmul attention, mask add, fp32 softmax, dropout, value matmul.
- `rotate_half`: line 188. NanoChat-specific half rotation: concat second half then negative first half.
- `NanoChatAttention`: line 196. q/k/v/o linear projections, optional projection bias, RoPE before q/k norm, cache update after RoPE+norm.
- `NanoChatMLP`: line 270. Dense `fc1 -> ACT2FN[hidden_act] -> fc2`, no biases.
- `NanoChatDecoderLayer`: line 285. Pre-norm attention residual, pre-norm MLP residual.
- `NanoChatModel`: line 358. Token embedding, extra RMSNorm before all layers, final RMSNorm after all layers.
- `NanoChatForCausalLM`: line 433. Untied LM head, `logits_to_keep`, optional final tanh softcap.

Key anchors from `configuration_nanochat.py`:

- `NanoChatConfig`: line 25.
- Defaults include `hidden_act="relu2"`, `rope_parameters=None`, `final_logit_softcapping=15.0`, `attention_bias=False`, `tie_word_embeddings=False`.
- `__post_init__` fills `num_key_value_heads = num_attention_heads` when omitted.

Other source anchors:

- `src/transformers/activations.py`: `ReLUSquaredActivation` line 206, mapped as `"relu2"` line 340.
- `src/transformers/models/nanochat/convert_nanochat_checkpoints.py`: infers `num_key_value_heads` from `k_proj.weight` rows and maps original nanochat checkpoint field names into native config fields.
