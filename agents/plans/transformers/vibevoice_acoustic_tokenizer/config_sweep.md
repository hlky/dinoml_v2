# VibeVoice Acoustic Tokenizer Config Sweep

Source basis: local Transformers checkout `transformers` at commit
`b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`, plus Hugging Face raw configs fetched
2026-05-13.

| Model id | Source type | Access | Key operator-significant fields |
| --- | --- | --- | --- |
| `microsoft/VibeVoice-AcousticTokenizer` | Official standalone HF checkpoint | Public, not gated | `model_type=vibevoice_acoustic_tokenizer`, `channels=1`, `hidden_size=64`, `num_filters=32`, `kernel_size=7`, `downsampling_ratios=[2,2,4,5,5,8]`, `depths=[3,3,3,3,3,3,8]`, `hidden_act=gelu`, `ffn_expansion=4`, `vae_std=0.625`, `dtype=bfloat16`. |
| `microsoft/VibeVoice-1.5B` | Official composite VibeVoice config with nested `acoustic_tokenizer_config` | Public, not gated | Legacy/original acoustic keys: `encoder_ratios=[8,5,5,4,2,2]`, `encoder_depths="3-3-3-3-3-3-8"`, `encoder_n_filters=32`, `vae_dim=64`, `fix_std=0.5`. The conversion script maps these to normalized HF fields: reversed ratios become `[2,2,4,5,5,8]`; `fix_std/0.8` becomes `vae_std=0.625`; unused historical fields are dropped. |
| `bezzam/VibeVoice-AcousticTokenizer` | Mirror/converted checkpoint referenced by the converter comments | Public, not gated | Same normalized architecture fields as `microsoft/VibeVoice-AcousticTokenizer`; includes `preprocessor_config.json` with `VibeVoiceAcousticTokenizerFeatureExtractor`. |
| `vibevoice/VibeVoice-Audio-Tokenizer` | Low-download open mirror | Public, not gated | Legacy/original acoustic key names only; no `preprocessor_config.json` found. Native `VibeVoiceAcousticTokenizerConfig` strict parsing should not be assumed to accept this config without the converter-style normalization. |
| `mrfakename/VibeVoice-Acoustic-Tokenizer` | Low-download open mirror | Public, not gated | Same legacy/original acoustic key pattern as `vibevoice/VibeVoice-Audio-Tokenizer`; no `preprocessor_config.json` found. |

Official standalone preprocessor config:

```json
{
  "eps": 1e-06,
  "feature_extractor_type": "VibeVoiceAcousticTokenizerFeatureExtractor",
  "feature_size": 1,
  "normalize_audio": true,
  "padding_side": "right",
  "padding_value": 0.0,
  "return_attention_mask": true,
  "sampling_rate": 24000,
  "target_dB_FS": -25
}
```

