# xLSTM config sweep evidence

Source date: 2026-05-13.

Primary source files:

- Local Transformers checkout: `X:/H/transformers`, commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Model source: `src/transformers/models/xlstm/configuration_xlstm.py`
- Model source: `src/transformers/models/xlstm/modeling_xlstm.py`

Representative Hugging Face configs inspected:

| Model id | Config URL | Important observed fields |
|---|---|---|
| `NX-AI/xLSTM-7b` | https://huggingface.co/NX-AI/xLSTM-7b/raw/main/config.json | `hidden_size` omitted, `embedding_dim=4096`, `num_blocks=32`, `num_heads=8`, `vocab_size=50304`, `weight_mode=single`, `chunkwise_kernel=chunkwise--triton_xl_chunk`, `sequence_kernel=native_sequence__triton`, `step_kernel=triton`, `torch_dtype=float32`, `tie_word_embeddings=false` |
| `ethicalabs/xLSTM-7b-Instruct` | https://huggingface.co/ethicalabs/xLSTM-7b-Instruct/raw/main/config.json | `hidden_size=4096`, `num_hidden_layers=32`, `num_blocks=32`, `num_heads=8`, `vocab_size=50560`, native kernels, `dtype=bfloat16`, `tie_word_embeddings=false` |
| `ethicalabs/xLSTM-7b-Polymath` | https://huggingface.co/ethicalabs/xLSTM-7b-Polymath/raw/main/config.json | Same 7B width/layer/head shape as Instruct, `vocab_size=50560`, native kernels, `dtype=bfloat16` |
| `AirlockRIck/xLSTM-7b` | https://huggingface.co/AirlockRIck/xLSTM-7b/raw/main/config.json | Mirror-like 7B config with Triton kernel strings and `vocab_size=50304` |
| `stefan-it/xlstm-transformers-bug-native` | https://huggingface.co/stefan-it/xlstm-transformers-bug-native/raw/main/config.json | Debug/native: `hidden_size=512`, `num_blocks=16`, `num_heads=4`, `max_inference_chunksize=1`, `mode=train`, native kernels |
| `stefan-it/xlstm-transformers-bug-triton` | https://huggingface.co/stefan-it/xlstm-transformers-bug-triton/raw/main/config.json | Debug/Triton: same small dimensions as native debug but Triton kernel strings |
| `J4bb4wukis/xlstm_247m_wikipedia_en_shuffeld` | https://huggingface.co/J4bb4wukis/xlstm_247m_wikipedia_en_shuffeld/raw/main/config.json | `embedding_dim=768`, `num_blocks=24`, `num_heads=4`, GPT-2-like token ids/vocab `50257`, native kernels |
| `J4bb4wukis/xlstm_406m_wikipedia_en_shuffeld` | https://huggingface.co/J4bb4wukis/xlstm_406m_wikipedia_en_shuffeld/raw/main/config.json | `embedding_dim=1024`, `num_blocks=24`, `num_heads=4`, GPT-2-like token ids/vocab `50257`, native kernels |
| `anrilombard/sallm-xlstm-125m` | https://huggingface.co/anrilombard/sallm-xlstm-125m/raw/main/config.json | `hidden_size=768`, `num_blocks=12`, `num_heads=4`, `vocab_size=65536`, native kernels, `tie_word_embeddings=true` |

Derived dimensions using the pinned `xLSTMConfig` properties:

| Shape family | `qk_dim` | `qk_head_dim` | `v_dim` | `v_head_dim` | `ffn_up_dim` | Per-layer recurrent states |
|---|---:|---:|---:|---:|---:|---|
| 4096 hidden, 8 heads | 2048 | 256 | 4096 | 512 | 10944 | `C [B,8,256,512]`, `N [B,8,256]`, `M [B,8,1]` |
| 512 hidden, 4 heads | 256 | 64 | 512 | 128 | 1408 | `C [B,4,64,128]`, `N [B,4,64]`, `M [B,4,1]` |
| 768 hidden, 4 heads | 384 | 96 | 768 | 192 | 2112 | `C [B,4,96,192]`, `N [B,4,96]`, `M [B,4,1]` |
| 1024 hidden, 4 heads | 512 | 128 | 1024 | 256 | 2752 | `C [B,4,128,256]`, `N [B,4,128]`, `M [B,4,1]` |

Tokenizer/config evidence for `NX-AI/xLSTM-7b`:

- `generation_config.json`: `bos_token_id=0`, `eos_token_id=2`, `pad_token_id=1`.
- `tokenizer_config.json`: `tokenizer_class=GPTNeoXTokenizer`, `bos/eos/unk="<|endoftext|>"`, `pad_token=null`, special added id `1` named `<|padding|>`, multiple added whitespace tokens near ids `50254..50276`.

