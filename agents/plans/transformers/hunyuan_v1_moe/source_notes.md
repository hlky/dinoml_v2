# Hunyuan v1 MoE Source Notes Snapshot

Transformers source checkout:

- Path: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Native family path: `src/transformers/models/hunyuan_v1_moe`

Authoritative source layout:

- `modeling_hunyuan_v1_moe.py` is generated from `modular_hunyuan_v1_moe.py`.
- Future Transformers source edits should target `modular_hunyuan_v1_moe.py`.
- DinoML audit used the generated file for runtime details and the modular file to confirm intended inheritance from Llama/Mixtral/HunyuanDense components.

Key source facts:

- RMSNorm upcasts hidden states to fp32, computes mean square on the last axis, multiplies by `rsqrt(var + eps)`, casts back to input dtype, then applies learned weight.
- Attention is causal self-attention with separate `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Source uses GQA when `num_key_value_heads < num_attention_heads`; A13B uses 32 query heads, 8 KV heads, 4 query groups per KV head, `head_dim=128`.
- RoPE is applied before per-head query/key RMSNorm. The per-head RMSNorm shape is `head_dim`.
- KV cache stores post-RoPE, post-QK-norm key states and raw value states.
- MoE gate is an fp32 linear `hidden_size -> num_experts`.
- Routing is `softmax(router_logits, dim=1)`, `topk`, then top-k weights are renormalized by their selected sum.
- Experts are packed tensors: `gate_up_proj[E, 2 * intermediate, hidden]` and `down_proj[E, hidden, intermediate]`.
- Expert eager fallback builds one-hot expert masks, loops over hit experts, gathers token rows with `where`, runs two expert linears, scales by routing weight, and scatters back with `index_add_`.
- Every MoE layer also has a dense shared SwiGLU MLP branch with separate `gate_proj`, `up_proj`, `down_proj`; output is `expert_result + shared_mlp_result`.
- `ForCausalLM` supports `logits_to_keep`, so prefill/decode can avoid full-sequence logits.
- Sequence classification head is implemented through generic causal sequence classification, but is optional/deferred for text-generation parity.

Remote-code divergence hazards:

- Main official repo `tencent/Hunyuan-A13B-Instruct` is tagged `model_type: hunyuan_v1_moe`, but also includes `auto_map` to remote `configuration_hunyuan.HunYuanConfig` and `hunyuan.HunYuanMoEV1ForCausalLM`.
- GPTQ/FP8 official variants and common mirrors use `model_type: hunyuan`, not `hunyuan_v1_moe`.
- Official configs contain many remote-code fields not read by native `modeling_hunyuan_v1_moe.py`, including `use_cla`, `use_mla`, LoRA-rank fields, vision token fields, and classification-pool fields.
- Native `hunyuan_v1_moe` should reject or separately route quantized and remote-code variants until their actual source is audited.

