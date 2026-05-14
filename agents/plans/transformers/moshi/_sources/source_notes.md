# Moshi Source Notes

Audit scope: Transformers `moshi` family only, source checkout `X:/H/transformers` at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

## Local source files inspected

- `src/transformers/models/moshi/configuration_moshi.py`
  - `MoshiDepthConfig`: defaults for the depth decoder: `hidden_size=1024`, `input_size=4096`, `num_hidden_layers=6`, `num_attention_heads=16`, `audio_vocab_size=2048`, `max_position_embeddings=9`, `sliding_window=8`, `num_codebooks=8`.
  - `MoshiConfig`: defaults for the main decoder: `hidden_size=4096`, `num_hidden_layers=32`, `num_attention_heads=32`, `max_position_embeddings=3000`, `sliding_window=3000`, `num_codebooks=8`; creates nested `audio_encoder_config` via `AutoConfig.for_model("mimi")` and nested `MoshiDepthConfig`.
  - Config validation rejects odd `ffn_dim` and rejects `num_codebooks > audio_encoder_config.num_codebooks`.
- `src/transformers/models/moshi/modeling_moshi.py`
  - Core modules: `MoshiRMSNorm`, `MoshiFlexibleLinear`, `MoshiGatingMLP`, `MoshiAttention`, `MoshiDecoderLayer`, `MoshiDepthDecoder`, `MoshiModel`, `MoshiForCausalLM`, `MoshiForConditionalGeneration`.
  - Attention classes: eager, FlashAttention2, SDPA.
  - Generation code owns audio codebook delay masks, generated audio-code session fields, blank-user-audio code generation, and nested depth-decoder generation.
- `src/transformers/models/moshi/convert_moshi_transformers.py`
  - Conversion stacks original per-codebook depth-decoder gates/projections/heads into `MoshiFlexibleLinear` weight tensors.
  - Conversion splits original packed QKV weights and applies RoPE permutation for main decoder q/k weights.
  - Conversion writes generation config with sliding-window cache and depth decoder `min_length=max_length=num_codebooks+1`.
- `src/transformers/models/mimi/configuration_mimi.py`
  - Nested codec config defaults and properties: `sampling_rate=24000`, `audio_channels=1`, `hidden_size=512`, `codebook_size=2048`, `codebook_dim=256`, `num_quantizers=32`, `num_semantic_quantizers=1`, `upsampling_ratios=[8,6,5,4]`, `frame_rate` compatibility field.
- `src/transformers/models/mimi/modeling_mimi.py`
  - Codec-owned conv/transformer/RVQ encode/decode, streaming padding cache, codebook encode/decode, and waveform reconstruction.
- `src/transformers/generation/utils.py`
  - Special case for `MoshiDepthDecoder` generation uses `config.audio_vocab_size` as generation vocab size.
- `src/transformers/convert_slow_tokenizer.py`
  - Contains `MoshiConverter`, relevant only to tokenizer conversion.

## Representative HF configs checked

- `https://huggingface.co/kmhf/hf-moshiko/raw/main/config.json`
- `https://huggingface.co/kmhf/hf-moshiko/raw/main/generation_config.json`
- `https://huggingface.co/kmhf/hf-moshiko/raw/main/preprocessor_config.json`
- `https://huggingface.co/kmhf/hf-moshika/raw/main/config.json`
- `https://huggingface.co/kmhf/hf-moshika/raw/main/generation_config.json`
- `https://huggingface.co/kmhf/hf-moshika/raw/main/preprocessor_config.json`
- `https://huggingface.co/kmhf/hf-moshiko/raw/main/tokenizer_config.json`

Observed `kmhf/hf-moshiko` and `kmhf/hf-moshika` configs were architecture-identical for native Transformers: `MoshiForConditionalGeneration`, main `32 x 4096`, depth `6 x 1024`, Mimi codec config embedded, `torch_dtype=bfloat16`, `num_codebooks=8`, `audio_vocab_size=2048`, text vocab `32000`, and sliding-window generation.

Kyutai native repos such as `kyutai/moshiko-pytorch-bf16`, `kyutai/moshika-pytorch-bf16`, and Candle/MLX/GGUF variants are public metadata repos tagged `library_name=moshi`, but raw `config.json` was not available through the standard Transformers layout during this audit. Treat them as non-native or external-runtime checkpoints unless a converted Transformers config is supplied.

## Useful command snippets

```powershell
git -C X:\H\transformers rev-parse HEAD
Get-ChildItem X:\H\transformers\src\transformers\models\moshi
rg -n "class Moshi|def forward|def generate|build_delay|apply_delay|DynamicCache|StaticCache|MoshiFlexibleLinear|MoshiDepthDecoder" X:\H\transformers\src\transformers\models\moshi\modeling_moshi.py
rg -n "class Mimi|sampling_rate|frame_rate|codebook|encode\(|decode\(|padding_cache|use_streaming" X:\H\transformers\src\transformers\models\mimi\modeling_mimi.py X:\H\transformers\src\transformers\models\mimi\configuration_mimi.py
```
