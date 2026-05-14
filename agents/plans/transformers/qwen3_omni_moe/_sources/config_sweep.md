# qwen3_omni_moe representative config sweep

Date: 2026-05-13

Representative official checkpoints:

| Model id | Architecture | `enable_audio_output` | Thinker target | Talker/code2wav target |
|---|---|---:|---|---|
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `Qwen3OmniMoeForConditionalGeneration` | true | multimodal input -> text tokens | optional text/audio output |
| `Qwen/Qwen3-Omni-30B-A3B-Thinking` | `Qwen3OmniMoeForConditionalGeneration` | false | multimodal input -> text tokens | not loaded |
| `Qwen/Qwen3-Omni-30B-A3B-Captioner` | `Qwen3OmniMoeForConditionalGeneration` | false | audio-focused input -> text tokens | not loaded |

## Shared thinker dimensions from official configs

| Field | Value |
|---|---:|
| text `hidden_size` | 2048 |
| text `num_hidden_layers` | 48 |
| text `num_attention_heads` | 32 |
| text `num_key_value_heads` | 4 |
| text `head_dim` | 128 |
| text attention width | 4096 (`32 * 128`) |
| text KV width | 512 per K/V projection (`4 * 128`) |
| text `vocab_size` | 152064 |
| text `max_position_embeddings` | 65536 |
| text `rope_theta` | 1000000 |
| text `intermediate_size` | 768 |
| text `moe_intermediate_size` | 768 |
| text `num_experts` | 128 |
| text `num_experts_per_tok` | 8 |
| text `decoder_sparse_step` | 1 |
| text `attention_bias` | false |
| text `sliding_window` | null |
| vision `depth` | 27 |
| vision `hidden_size` | 1152 |
| vision `num_heads` | 16 |
| vision `intermediate_size` | 4304 |
| vision `patch_size` | 16 |
| vision `temporal_patch_size` | 2 |
| vision `spatial_merge_size` | 2 |
| vision `out_hidden_size` | 2048 |
| vision `deepstack_visual_indexes` | `[8, 16, 24]` |
| audio `num_mel_bins` | 128 |
| audio `d_model` | 1280 |
| audio `encoder_layers` | 32 |
| audio `encoder_attention_heads` | 20 |
| audio `encoder_ffn_dim` | 5120 |
| audio `downsample_hidden_size` | 480 |
| audio `n_window` | 50 |
| audio `n_window_infer` | 800 |
| audio `conv_chunksize` | 500 |
| audio `output_dim` | 2048 |
| multimodal token IDs | audio 151675, image 151655, video 151656 |

## Instruct-only output-audio dimensions

| Field | Value |
|---|---:|
| talker text `hidden_size` | 1024 |
| talker text `num_hidden_layers` | 20 |
| talker text `num_attention_heads` | 16 |
| talker text `num_key_value_heads` | 2 |
| talker text `head_dim` | 128 |
| talker text `vocab_size` | 3072 |
| talker text `num_experts` | 128 |
| talker text `num_experts_per_tok` | 6 |
| talker text `moe_intermediate_size` | 384 |
| talker text `shared_expert_intermediate_size` | 768 |
| code predictor `hidden_size` | 1024 |
| code predictor layers | 5 full-attention layers |
| code predictor heads/KV heads/head dim | 16 / 8 / 128 |
| code predictor `num_code_groups` | 16 in checkpoint config |
| code2wav transformer layers | 8 |
| code2wav heads/KV heads | 16 / 16 |
| code2wav `codebook_size` | 2048 |
| code2wav `num_quantizers` | 16 |
| code2wav `sliding_window` | 72 |
| code2wav `upsample_rates` | `[8, 5, 4, 3]` |
| code2wav `upsampling_ratios` | `[2, 2]` |
| code2wav `decoder_dim` | 1536 |

## Source-default differences worth guarding

- Source default text config is much smaller (`hidden_size=2048`, `num_hidden_layers=28`, `num_attention_heads=28`, `num_key_value_heads=4`, `vocab_size=3584`) than official checkpoint configs. Use checkpoint configs for real model admission.
- Source default `audio_token_id` is `151646`, while official configs use `151675`. Placeholder IDs must come from the loaded config/tokenizer.
- Source default talker code predictor has `num_code_groups=32`; official Instruct config has `16`. The talker/code2wav path must not hard-code the default.
- Source default vision/audio `out_hidden_size`/`output_dim` is `3584`; official configs project to `2048` to match the official thinker text hidden size.
