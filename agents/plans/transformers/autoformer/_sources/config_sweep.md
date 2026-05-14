# Autoformer config sweep

Fetched accessible `config.json` files from Hugging Face. No 401/403/404 gaps were hit for the listed configs.

| Model id | Source type | URL | Notable config fields |
| --- | --- | --- | --- |
| `huggingface/autoformer-tourism-monthly` | Official HF checkpoint | https://huggingface.co/huggingface/autoformer-tourism-monthly/blob/main/config.json | `context_length=24`, `prediction_length=24`, `d_model=64`, `encoder_layers=4`, `decoder_layers=4`, `heads=4/4`, `feature_size=22`, `lags_sequence` length 16, `num_time_features=2`, one static categorical feature with cardinality 366, `scaling="mean"`, `distribution_output="student_t"` |
| `kashif/autoformer-traffic-hourly` | Public community representative | https://huggingface.co/kashif/autoformer-traffic-hourly/blob/main/config.json | `context_length=48`, `prediction_length=24`, `d_model=16`, `encoder_layers=2`, `decoder_layers=2`, no static categorical feature, 40 lags up to 721, `num_time_features=5`, `feature_size=47` |
| `kashif/autoformer-electricity-hourly` | Public community representative | https://huggingface.co/kashif/autoformer-electricity-hourly/blob/main/config.json | `context_length=96`, `prediction_length=24`, `d_model=16`, `encoder_layers=4`, `decoder_layers=2`, one static categorical feature with cardinality 321, 40 lags up to 721, `feature_size=49` |
| `elisim/autoformer-exchange-rate-50-epochs` | Public community representative | https://huggingface.co/elisim/autoformer-exchange-rate-50-epochs/blob/main/config.json | `context_length=60`, `prediction_length=30`, `d_model=16`, `encoder_layers=2`, `decoder_layers=2`, 29 lags up to 780, `num_time_features=4`, `scaling="std"`, `feature_size=35` |
| `JLB-JLB/EEG_Autoformer_336_history_96_horizon` | Public community representative; multivariate | https://huggingface.co/JLB-JLB/EEG_Autoformer_336_history_96_horizon/blob/main/config.json | `input_size=22`, `context_length=336`, `prediction_length=96`, `d_model=512`, `encoder_heads=16`, `decoder_heads=8`, `encoder_ffn_dim=2048`, `decoder_ffn_dim=2048`, 20 lags, `feature_size=489`, safetensors metadata reports 23,748,157 F32 parameters |
| `thesven/BTC-Autoformer-v1` | Public community representative | https://huggingface.co/thesven/BTC-Autoformer-v1/blob/main/config.json | `context_length=58`, `prediction_length=29`, `d_model=32`, `encoder_layers=6`, `decoder_layers=4`, 30 lags up to 1093, `feature_size=36`, safetensors metadata reports 104,915 F32 parameters |

Historical official config revisions inspected:

- `huggingface/autoformer-tourism-monthly` revision `3d80f4579222b70b230b7dca1e06ef15e8538e75`: same production dimensions as current but includes explicit `is_encoder_decoder=true`.
- `huggingface/autoformer-tourism-monthly` revision `7e1262fe7634691c43396dd5b63f21e3eccd7033`: early config with `context_length=240`, `lags_sequence=[1]`, and `feature_size=7`.

