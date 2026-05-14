# ModernBERT Decoder Config Sweep

Source basis: Hugging Face Hub `config.json` files downloaded on 2026-05-13 with `huggingface_hub.hf_hub_download`. These are config-derived facts unless noted otherwise.

| Model id | Snapshot | Architecture | H | Layers | Heads | Head dim | MLP | Vocab | Max pos | Layer pattern | Local/window | Cache |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `onnx-internal-testing/tiny-random-ModernBertDecoderForCausalLM` | `0668913bbb02bebd030c43d4165ff36e9216f6b0` | `ModernBertDecoderForCausalLM` | 32 | 2 | 4 | 8 | 32 | 50368 | 7999 | 1 full, 1 sliding | `sliding_window=64`, no `local_attention` key | true |
| `jhu-clsp/ettin-decoder-17m` | `728fb5b6b3dc5916aa20829c027143ae8eca4eeb` | `ModernBertDecoderForCausalLM` | 256 | 7 | 4 | 64 | 384 | 50368 | 7999 | 3 full, 4 sliding | `local_attention=128` -> effective `sliding_window=64` | true |
| `jhu-clsp/ettin-decoder-32m` | `a6ba5b89e2d8a5b71c819ac6c47ba81423019d3e` | `ModernBertDecoderForCausalLM` | 384 | 10 | 6 | 64 | 576 | 50368 | 7999 | 4 full, 6 sliding | `local_attention=128` -> effective `sliding_window=64` | true |
| `jhu-clsp/ettin-decoder-68m` | `5d704138dd3e12a32ca7a357d25b09c618a9fb02` | `ModernBertDecoderForCausalLM` | 512 | 19 | 8 | 64 | 768 | 50368 | 7999 | 7 full, 12 sliding | `local_attention=128` -> effective `sliding_window=64` | true |
| `jhu-clsp/ettin-decoder-150m` | `b099375cba8b9045a87a9e6ede0b7ab32fd2f571` | `ModernBertDecoderForCausalLM` | 768 | 22 | 12 | 64 | 1152 | 50368 | 7999 | 8 full, 14 sliding | `local_attention=128` -> effective `sliding_window=64` | true |
| `jhu-clsp/ettin-decoder-400m` | `ac53c21d2e9284ec150975cfc68f4e09a9c6afa6` | `ModernBertDecoderForCausalLM` | 1024 | 28 | 16 | 64 | 2624 | 50368 | 7999 | 10 full, 18 sliding | `local_attention=128` -> effective `sliding_window=64` | true |
| `jhu-clsp/ettin-decoder-1b` | `7a839081bb4c6f7e10cfb6334f6d9e1e1a76a980` | `ModernBertDecoderForCausalLM` | 1792 | 28 | 28 | 64 | 3840 | 50368 | 7999 | 10 full, 18 sliding | `local_attention=128` -> effective `sliding_window=64` | true |
| `lebe1/lettuceprevent-ettin-decoder-68m-en` | `eacc679b68187bffe20210112e913477ae7c463e` | `EttinTokenClassifier` | 512 | 19 | 8 | 64 | 768 | 50368 | 7999 | 7 full, 12 sliding | `sliding_window=64`, no `local_attention` key | false |

Notes:

- The in-library `ModernBertDecoderConfig.__post_init__` computes `sliding_window = local_attention // 2 if local_attention else -1`. Existing Ettin configs mostly store `local_attention=128`; some downstream/tiny configs store `sliding_window=64` but omit `local_attention`, so the source default `local_attention=128` preserves the same effective value.
- The current inspected source reads `layer_types`, `local_attention`, `rope_parameters`/legacy `global_rope_theta` and `local_rope_theta`, `use_cache`, projection/norm/dropout/bias fields, and token ids. Fields observed in configs but not read by this source path include `is_causal`, `masked_prediction`, `causal_mask`, `position_embedding_type`, `deterministic_flash_attn`, and `reference_compile`.
- `yosefw/SPLADE-Ettin-32m-decoder` was searchable on the Hub, but `config.json` returned 404 at `https://huggingface.co/yosefw/SPLADE-Ettin-32m-decoder/resolve/main/config.json`; it is not used as source basis.
