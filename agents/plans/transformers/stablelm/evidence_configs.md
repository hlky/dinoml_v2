# StableLM Config Evidence Snapshot

Source: Hugging Face raw `config.json` files fetched during audit on 2026-05-13. Transformers source basis is local checkout `transformers` at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

| Model id | Config availability | Operator-significant fields |
| --- | --- | --- |
| `stabilityai/stablelm-3b-4e1t` | `https://huggingface.co/stabilityai/stablelm-3b-4e1t/raw/main/config.json` | `hidden_size=2560`, `layers=32`, `heads=32`, `kv_heads=32`, `intermediate_size=6912`, `vocab_size=50304`, `max_position_embeddings=4096`, `partial_rotary_factor=0.25`, `rope_theta=10000`, `use_qkv_bias=false`, `tie_word_embeddings=false`, `torch_dtype=bfloat16` |
| `stabilityai/stablelm-zephyr-3b` | `https://huggingface.co/stabilityai/stablelm-zephyr-3b/raw/main/config.json` | Same structural fields as `stablelm-3b-4e1t`; instruction tuning/tokenizer behavior differs outside the neural graph. |
| `stabilityai/stablelm-2-1_6b` | `https://huggingface.co/stabilityai/stablelm-2-1_6b/raw/main/config.json` | `hidden_size=2048`, `layers=24`, `heads=32`, `kv_heads=32`, `intermediate_size=5632`, `vocab_size=100352`, `partial_rotary_factor=0.25`, `use_qkv_bias=true`, `torch_dtype=float16` |
| `stabilityai/tiny-random-stablelm-2` | `https://huggingface.co/stabilityai/tiny-random-stablelm-2/raw/main/config.json` | `hidden_size=512`, `layers=8`, `heads=16`, `kv_heads=4`, `intermediate_size=1536`, `vocab_size=100352`, `qk_layernorm=true`, `use_parallel_residual=true`, `use_qkv_bias=false`, `torch_dtype=bfloat16` |
| `afrideva/stablelm-3b-4e1t-GGUF` | `https://huggingface.co/afrideva/stablelm-3b-4e1t-GGUF/raw/main/config.json` | Mirror config only declares `model_type=stablelm`; real dense dimensions must come from GGUF metadata or original model config. |
| `afrideva/stablelm-2-1_6b-GGUF` | raw config returned 404 | Transformers GGUF tests reference the repo/file, but no `config.json` was available at the queried raw path. |
