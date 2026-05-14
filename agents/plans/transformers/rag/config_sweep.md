# RAG Representative Config Snapshot

Source: raw Hugging Face Hub `config.json` files fetched for the RAG audit on 2026-05-13.

| Model id | URL | model_type | n_docs | max_combined_length | retrieval_vector_size | index_name | dataset | question encoder | generator |
|---|---|---:|---:|---:|---:|---|---|---|---|
| `facebook/rag-token-base` | https://huggingface.co/facebook/rag-token-base/raw/main/config.json | `rag` | 5 | 300 | 768 | `exact` | `wiki_dpr` | DPR base: hidden 768, 12 layers, 12 heads, FFN 3072, vocab 30522 | BART large: d_model 1024, 12 enc/12 dec layers, 16 heads, FFN 4096, vocab 50265 |
| `facebook/rag-token-nq` | https://huggingface.co/facebook/rag-token-nq/raw/main/config.json | `rag` | 5 | 300 | 768 | `legacy` | `wiki_dpr` | DPR base: hidden 768, 12 layers, 12 heads, FFN 3072, vocab 30522 | BART large: d_model 1024, 12 enc/12 dec layers, 16 heads, FFN 4096, vocab 50265 |
| `facebook/rag-sequence-base` | https://huggingface.co/facebook/rag-sequence-base/raw/main/config.json | `rag` | 5 | 300 | 768 | `exact` | `wiki_dpr` | DPR base: hidden 768, 12 layers, 12 heads, FFN 3072, vocab 30522 | BART large: d_model 1024, 12 enc/12 dec layers, 16 heads, FFN 4096, vocab 50265 |
| `facebook/rag-sequence-nq` | https://huggingface.co/facebook/rag-sequence-nq/raw/main/config.json | `rag` | 5 | 300 | 768 | `legacy` | `wiki_dpr` | DPR base: hidden 768, 12 layers, 12 heads, FFN 3072, vocab 30522 | BART large: d_model 1024, 12 enc/12 dec layers, 16 heads, FFN 4096, vocab 50265 |

Notes:

- No `generation_config.json` was present for these four model repos at the checked raw Hub paths; generation defaults therefore come from config/model-generation defaults.
- The `*-nq` configs contain concrete top-level special token ids in at least `rag-token-nq`; some base configs leave them null and rely on the generator config.
- All four representative official configs use `question_encoder.model_type="dpr"` and `generator.model_type="bart"`.
- `legacy` retrieval uses Google-hosted DPR-compatible FAISS/pickle files and requires `TRUST_REMOTE_CODE=True` in the inspected source before unpickling passage/index metadata.
