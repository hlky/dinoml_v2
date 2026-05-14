# MLCD Source Snapshot

Source checkout: `X:/H/transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Authoritative source-edit file: `src/transformers/models/mlcd/modular_mlcd.py`.

Generated runtime file: `src/transformers/models/mlcd/modeling_mlcd.py`; it states that edits should be made in the modular file.

Key source facts:

- `MLCDVisionConfig.model_type = "mlcd_vision_model"` and source defaults are bigG-like: hidden size 1664, 48 layers, 16 heads, patch 14, image 336, GELU.
- Auto mappings also preserve historical Hub configs with `model_type = "mlcd"` by routing them to `MLCDVisionConfig` / `MLCDVisionModel`.
- `MLCDVisionEmbeddings` uses `Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size, bias=False)`, flattens `[B,C,H/P,W/P]` to patch tokens, prepends a learned class embedding, and does not add learned absolute position embeddings.
- `MLCDRotaryEmbedding` builds 2D RoPE positions from patch-grid height/width; `MLCDVisionModel.forward` prepends a learned class-position embedding before computing cos/sin.
- Attention is noncausal self-attention. Source Q/K/V projections are separate biased linears of `hidden_size -> hidden_size`; RoPE is applied to Q/K before permuting to `[B,H,S,D]`.
- `num_key_value_groups` is present in config/source and only affects `repeat_kv` in eager attention. Current official bigG configs omit it, so source default `1` gives ordinary MHA.
- Block order is pre-LN attention residual, pre-LN MLP residual. The model applies an additional `pre_layernorm` after embeddings and a `post_layernorm` only to the CLS pooled output.
- Public output is `BaseModelOutputWithPooling(last_hidden_state=[B,S,C], pooler_output=[B,C])`; there is no text tower, projection head, logits head, generation, or KV cache in this MLCD source.
