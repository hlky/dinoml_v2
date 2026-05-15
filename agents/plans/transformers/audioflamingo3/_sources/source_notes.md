# AudioFlamingo3 Source Notes

Audit target: `audioflamingo3` only.

Local Transformers checkout:

- Path: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `src/transformers/models/audioflamingo3`

Files inspected locally:

- `src/transformers/models/audioflamingo3/configuration_audioflamingo3.py`
- `src/transformers/models/audioflamingo3/modular_audioflamingo3.py`
- `src/transformers/models/audioflamingo3/modeling_audioflamingo3.py`
- `src/transformers/models/audioflamingo3/processing_audioflamingo3.py`
- `src/transformers/models/audioflamingo3/convert_audioflamingo3_to_hf.py`
- Neighbor/composed files:
  - `src/transformers/models/qwen2/configuration_qwen2.py`
  - `src/transformers/models/qwen2/modeling_qwen2.py`
  - `src/transformers/models/whisper/feature_extraction_whisper.py`
  - Auto mappings under `src/transformers/models/auto/*`

Primary HF configs and metadata inspected:

- `https://huggingface.co/nvidia/audio-flamingo-3-hf/raw/main/config.json`
- `https://huggingface.co/nvidia/audio-flamingo-3-hf/raw/main/processor_config.json`
- `https://huggingface.co/nvidia/audio-flamingo-3-hf/raw/main/tokenizer_config.json`
- `https://huggingface.co/nvidia/audio-flamingo-3-hf/raw/main/generation_config.json`
- `https://huggingface.co/nvidia/audio-flamingo-3-hf/raw/main/special_tokens_map.json`
- `https://huggingface.co/nvidia/music-flamingo-hf/raw/main/config.json`
- `https://huggingface.co/nvidia/audio-flamingo-3-hf/raw/main/think/config.json`
- `https://huggingface.co/nvidia/audio-flamingo-3/raw/main/config.json` returned a Git LFS pointer, not JSON.

Important local line anchors from the pinned checkout:

- `configuration_audioflamingo3.py:56-68`: audio encoder defaults: 128 mel bins, 32 layers, 20 heads, hidden 1280, FFN 5120, max source positions 1500.
- `configuration_audioflamingo3.py:96-115`: composite config, default `audio_token_id=151669`, projector GELU+bias, default subconfigs `audioflamingo3_encoder` and `qwen2`.
- `modeling_audioflamingo3.py:1-6`: generated from `modular_audioflamingo3.py`; future source edits belong in the modular file.
- `modeling_audioflamingo3.py:47-70`: audio encoder eager attention matmul/softmax path with query pre-scaling compatibility comment.
- `modeling_audioflamingo3.py:111-114`: audio attention projections: `k_proj` bias false, `q_proj`/`v_proj`/`out_proj` bias true.
- `modeling_audioflamingo3.py:280-300`: audio encoder conv front end and avg pool modules: Conv1d 128->1280, Conv1d 1280->1280 stride 2, learned/frozen positional embedding length 1500, AvgPool1d(2,stride=2), LayerNorm.
- `modeling_audioflamingo3.py:336-365`: audio forward layout: `[B,128,T] -> convs -> permute to [B,Tc,1280] -> bidirectional encoder -> permute/avg_pool/permute -> LayerNorm`.
- `modeling_audioflamingo3.py:372-378`: length equations: `conv_len=(L-1)//2+1`, `post_pool_len=(conv_len-2)//2+1`.
- `modeling_audioflamingo3.py:387-400`: projector: Linear 1280->text_hidden, activation, Linear text_hidden->text_hidden.
- `modeling_audioflamingo3.py:418-420`: composite body: `AutoModel.from_config(audio_config)`, `AutoModelForCausalLM.from_config(text_config)`, projector.
- `modeling_audioflamingo3.py:469-473`: valid projected audio rows are flattened by boolean mask into `pooler_output`.
- `modeling_audioflamingo3.py:477-499`: placeholder mask and shape/count check.
- `modeling_audioflamingo3.py:580-600`: forward embeds text tokens, computes audio features only when `input_features` and `input_ids` are present, stitches with `masked_scatter`, delegates to language model.
- `modeling_audioflamingo3.py:604-616`: generation input prep only forwards audio tensors on first iteration or when cache is disabled.
- `processing_audioflamingo3.py:34-48`: processor defaults: text padding true, audio sampling rate 16 kHz, return attention mask, audio padding max_length, tensors pt, padding_side left.
- `processing_audioflamingo3.py:90-100`: processor placeholder expansion uses the same conv/pool length equations and repeats the `<sound>` string.
- `processing_audioflamingo3.py:158-187`: waveform split into 30 s windows, max windows from 600 s / chunk length, flattened chunks passed to Whisper feature extractor, attention mask renamed to `input_features_mask`.
- `qwen2/modeling_qwen2.py:35-48`: SwiGLU MLP: gate/up/down bias false.
- `qwen2/modeling_qwen2.py:51-146`: default RoPE generation and application.
- `qwen2/modeling_qwen2.py:149-180`: GQA repeat-kv in eager attention and softmax upcast to fp32.
- `qwen2/modeling_qwen2.py:190-245`: Qwen2 attention projections: q 3584->3584, k/v 3584->512 for 4 KV heads and 128 head dim, o 3584->3584; sliding window passed to attention backend when layer type requests it.
- `qwen2/modeling_qwen2.py:248-263`: RMSNorm fp32 variance path.
- `qwen2/modeling_qwen2.py:353-413`: Qwen2 causal model forward, DynamicCache allocation, position ids from cache length, full/sliding causal masks, shared RoPE per layer stack.
- `qwen2/modeling_qwen2.py:433-487`: causal LM head with `logits_to_keep`.
- `whisper/feature_extraction_whisper.py:69-103`: feature extractor fields and mel filter construction.
- `whisper/feature_extraction_whisper.py:118-130` and `149-161`: log-mel math, max clamp to `max-8`, `(log_spec+4)/4`.
- `whisper/feature_extraction_whisper.py:315-337`: waveform tensor is converted to `[B, feature_size, frames]`; attention mask is downsampled by hop length and trimmed if needed.

Representative config observations:

- Current `nvidia/audio-flamingo-3-hf` main config reports `dtype=float32`, `text_config.model_max_length=32768`, `bos_token_id=151670`, `pad_token_id=151671`, `use_cache=false`.
- Historical `audio-flamingo-3-hf` revisions visible in HF search show `dtype=bfloat16`, sometimes `model_max_length=8192`, and an older `init_std` field in the audio config. The current source uses `initializer_range`; `init_std` is not a declared field in the strict audio config.
- `nvidia/music-flamingo-hf` shares the architecture and dimensions with `dtype=bfloat16`, `model_max_length=8192`, and includes both `rope_parameters` and legacy `rope_theta`.
- `think/config.json` in the current `audio-flamingo-3-hf` repo is a historical/remote-code style Llava config with vision/sound/speech towers and quantization/training fields. It is not native `AudioFlamingo3Config` and should not be admitted by this native-source report.

No DinoML code, shared plan files, tests, imports, model execution, or commits were performed for this audit.
