# BROS Local Source Notes

Transformers checkout: `transformers`

Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Files inspected:

- `src/transformers/models/bros/configuration_bros.py`
- `src/transformers/models/bros/modeling_bros.py`
- `src/transformers/models/bros/processing_bros.py`
- `src/transformers/models/bros/__init__.py`

Important local anchors:

- `configuration_bros.py`: defaults are BERT-base-like: hidden 768, 12 layers, 12 heads, intermediate 3072, vocab 30522, max positions 512, type vocab 2, `bbox_scale=100.0`, `dim_bbox=8`, `n_relations=1`.
- `configuration_bros.py`: `__post_init__` derives `dim_bbox_sinusoid_emb_2d = hidden_size // 4`, `dim_bbox_sinusoid_emb_1d = dim_bbox_sinusoid_emb_2d // dim_bbox`, and `dim_bbox_projection = hidden_size // num_attention_heads`.
- `processing_bros.py`: `BrosProcessor` only wraps a tokenizer and supplies tokenizer kwargs. It does not invoke OCR, normalize boxes, or expand word boxes to subword boxes itself.
- `modeling_bros.py`: `BrosModel.forward` requires `bbox`; if `bbox.shape[-1] == 4`, it converts `[x1,y1,x2,y2]` to 8-point order `[x1,y1,x2,y1,x2,y2,x1,y2]`, then multiplies by `bbox_scale`.
- `modeling_bros.py`: `BrosBboxEmbeddings.forward` transposes `[B,S,8]` to `[S,B,8]`, forms pairwise relative boxes as `bbox_t[None,:,:,:] - bbox_t[:,None,:,:]`, runs sinusoidal embeddings per coordinate, and projects to `head_dim`.
- `modeling_bros.py`: self-attention uses standard Q/K/V linears and dense attention, but adds `einsum("bnid,bijd->bnij", query_layer, bbox_pos_emb)` to attention logits before scaling and mask addition.
- `modeling_bros.py`: relation heads transpose hidden states to `[S,B,H]`, append a learned dummy node to the key sequence, run query/key linears, and compute relation logits as `[n_relations, B, S, S+1]`; with `n_relations=1` the source squeezes relation dimension.
- `modeling_bros.py`: SPADE heads use `masked_fill` with attention-mask-derived invalid tokens and self-token masks built with `torch.eye(S, S+1)`.
- `modeling_bros.py`: source contains optional decoder/cross-attention fields, but inspected BROS configs are encoder-only and the current `BrosLayer` cross-attention branch appears inconsistent. DinoML should reject `is_decoder=True` or `add_cross_attention=True` for this audit target.
