# ColQwen2 Representative Config Snapshot

Source date: 2026-05-13

| Repo | Scope | Library/task metadata | Key dimensions | Notes |
|---|---|---|---|---|
| `vidore/colqwen2-v1.0-hf` | Native `transformers` ColQwen2 target | `transformers`, `visual-document-retrieval` | `embedding_dim=128`; VLM text `hidden_size=1536`, `layers=28`, `heads=12`, `kv_heads=2`, `intermediate=8960`, `vocab_size=151936`; vision `depth=32`, `embed_dim=1280`, `num_heads=16`, `patch_size=14`, `spatial_merge_size=2`, `temporal_patch_size=2`; `torch_dtype=bfloat16` | Official native config inspected from `config.json`. Processor uses `ColQwen2Processor`; preprocessor caps images at `max_pixels=602112` rather than the base Qwen2-VL default. |
| `Qwen/Qwen2-VL-2B-Instruct` | Delegated backbone reference | `transformers`, conditional generation | Text dimensions match ColQwen2 VLM: `hidden_size=1536`, `layers=28`, `heads=12`, `kv_heads=2`, `intermediate=8960`; vision `hidden_size=1536`; `vocab_size=151936` | Open config. ColQwen2 v1.0-hf declares `_name_or_path=Qwen/Qwen2-VL-2B-Instruct` inside `vlm_config`. |
| `Qwen/Qwen2-VL-7B-Instruct` | Larger delegated backbone variant, not native ColQwen2 checkpoint | `transformers`, conditional generation | `hidden_size=3584`, `layers=28`, `heads=28`, `kv_heads=4`, `intermediate=18944`; vision output `hidden_size=3584`; `vocab_size=152064` | Useful to expose Qwen2-VL family scaling and vocab/tie-word variation, but not required for ColQwen2 v1.0-hf parity. |
| `Qwen/Qwen2-VL-72B-Instruct` | Largest delegated backbone variant, not native ColQwen2 checkpoint | `transformers`, conditional generation | `hidden_size=8192`, `layers=80`, `heads=64`, `kv_heads=8`, `intermediate=29568`; vision output `hidden_size=8192`; `vocab_size=152064` | Demonstrates layer-count and hidden-size scaling if future ColQwen2-like checkpoints wrap larger Qwen2-VL bodies. |
| `vidore/colqwen2-base` | Historical/base repo | `colpali` library metadata | Flat `model_type=qwen2_vl`; `architectures=["ColQwen2"]`; `torch_dtype=float32`; minimal `vision_config` with only `hidden_size`, `in_chans`, `spatial_patch_size` | Not a native `colqwen2` config for the inspected Transformers source. Route through legacy ColPali/remote-library handling or convert first. |
| `vidore/colqwen2-v1.0-merged` | Historical merged repo | `colpali` library metadata | Similar flat Qwen2-VL config; `torch_dtype=bfloat16`; `architectures=["ColQwen2"]` | Not the native `ColQwen2ForRetrieval` checkpoint format. |

Representative URLs:

- https://huggingface.co/vidore/colqwen2-v1.0-hf
- https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
- https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
- https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct
- https://huggingface.co/vidore/colqwen2-base
- https://huggingface.co/vidore/colqwen2-v1.0-merged
