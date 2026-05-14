# Voxtral Config Sweep Snapshot

The table below preserves the operator-significant fields used in `report.md` without copying full large configs.

| Checkpoint | Source role | Audio tower | Text decoder | Processor / loading notes |
| --- | --- | --- | --- | --- |
| `tiny-random/voxtral` | Debug native config | H=64, FFN=256, L=2, heads=2, head_dim=32, max_source_positions=1500 | H=64, FFN=128, L=2, Q heads=2, KV heads=1, head_dim=32, tied embeddings true | Useful shape-smoke target; still uses 128 mel bins and 131072 vocab. |
| `yujiepan/voxtral-tiny-random` | Debug mirror | Same as `tiny-random/voxtral` | Same as `tiny-random/voxtral` | Duplicate debug dimensions observed. |
| `mistralai/Voxtral-Mini-3B-2507` | Official common Mini | H=1280, FFN=5120, L=32, heads=20, head_dim=64, max_source_positions=1500 | H=3072, FFN=8192, L=30, Q heads=32, KV heads=8, head_dim=128, RoPE theta 1e8 | `torch_dtype=bfloat16`; processor is WhisperFeatureExtractor with 16 kHz, 128 mel bins, 30 s chunks. |
| `MohamedRashad/Voxtral-Mini-3B-2507-transformers` | Open native mirror | Same as official Mini; includes `audio_config.torch_dtype=bfloat16` | Same as official Mini; includes `text_config.torch_dtype=bfloat16` | Useful if official artifacts are inconvenient; mirror provenance should be labeled. |
| `mistralai/Voxtral-Small-24B-2507` | Official production Small | Same as official Mini | H=5120, FFN=32768, L=40, Q heads=32, KV heads=8, head_dim=128, RoPE theta 1e8 | `torch_dtype=bfloat16`; bigger text decoder only. |
| `VincentGOURBIN/voxtral-small-8bit` | Quantized Small mirror | Same as official Small audio; config carries `quantization` map excluding convs/norms/positional embeddings | Same as official Small text; quantization map marks many linear/embed/lm modules | Native source does not consume this `quantization` map. Treat as external provider/loading metadata. |
| `mzbac/voxtral-mini-3b-4bit-mixed` | Quantized Mini mirror | Same as official Mini audio; mixed 6-bit entries for many audio linear layers | Same as official Mini text; mostly 4-bit with some 6-bit entries | Native source does not consume this `quantization` map. Treat as external provider/loading metadata. |

Effective source defaults if configs omit fields:

- `VoxtralConfig.text_config` defaults to Llama `vocab_size=131072`, `hidden_size=3072`, `intermediate_size=8192`, `num_hidden_layers=30`, `num_key_value_heads=8`, `max_position_embeddings=131072`, `rms_norm_eps=1e-5`, `use_cache=true`, `rope_theta=100000000.0`, `head_dim=128`.
- `VoxtralEncoderConfig` defaults to `hidden_size=1280`, `intermediate_size=5120`, `num_hidden_layers=32`, `num_attention_heads=20`, `num_mel_bins=128`, `max_source_positions=1500`, `activation_function=gelu`.
- The processor source default `audio_kwargs.max_source_positions=3000` is the mel-frame chunk size; the encoder config `max_source_positions=1500` is the post-conv sequence length.
- The projector source groups four post-conv audio positions: `[chunks, 1500, 1280] -> [chunks * 375, 5120] -> [chunks * 375, text_hidden]`.
