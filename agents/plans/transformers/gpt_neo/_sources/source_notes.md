# GPT-Neo source notes

Audit scope: `gpt_neo` only. No DinoML code edits, imports, model execution, or tests were run.

## Local source basis

- Transformers checkout: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family files:
  - `src/transformers/models/gpt_neo/modeling_gpt_neo.py`
  - `src/transformers/models/gpt_neo/configuration_gpt_neo.py`
  - `src/transformers/models/gpt_neo/convert_gpt_neo_mesh_tf_to_pytorch.py`
  - `src/transformers/models/gpt_neo/__init__.py`
- Shared source referenced:
  - `src/transformers/activations.py`
  - `src/transformers/cache_utils.py`
  - `src/transformers/masking_utils.py`
- Local test evidence inspected, without execution:
  - `tests/models/gpt_neo/test_modeling_gpt_neo.py`
  - `tests/utils/tiny_model_summary.json`

## Web/config sources

- `EleutherAI/gpt-neo-125m` config: https://huggingface.co/EleutherAI/gpt-neo-125m/blob/8741c104a4aae84316aa969cf1818c75ca44473e/config.json
- `EleutherAI/gpt-neo-1.3B` config: https://huggingface.co/EleutherAI/gpt-neo-1.3B/raw/main/config.json
- `EleutherAI/gpt-neo-2.7B` config: https://huggingface.co/EleutherAI/gpt-neo-2.7B/blob/5e1629a69b40344d3ba97e10662ef6593c5829f7/config.json
- `hf-tiny-model-private/tiny-random-GPTNeoForCausalLM` repo page: https://huggingface.co/hf-tiny-model-private/tiny-random-GPTNeoForCausalLM
- GPT-Neo model docs page for public model/task context: https://huggingface.co/docs/transformers/en/model_doc/gpt_neo

## Source-derived facts to preserve

- `GPTNeoConfig` defaults: vocab 50257, max positions 2048, hidden 2048, layers 24, heads 16, `attention_types=[[["global", "local"], 12]]`, `window_size=256`, `activation_function="gelu_new"`, `use_cache=True`, tied embeddings enabled by default. Config validation requires expanded `attention_layers` length to equal `num_layers`.
- Attention projections are independent dense linear layers: `q_proj`, `k_proj`, and `v_proj` are `hidden_size -> hidden_size` with `bias=False`; `out_proj` is `hidden_size -> hidden_size` with `bias=True`.
- Head dimension is `hidden_size // num_heads`; source raises if `hidden_size` is not divisible by `num_heads`.
- Eager attention computes scores in float32, applies the module-owned causal/local bool mask first, adds the external additive attention mask second, softmaxes on the last axis, casts probabilities back to value dtype, then computes `attn_probs @ value`.
- Local attention is implemented by modifying a lower-triangular bool buffer with `torch.bitwise_xor(bias, torch.tril(bias, -config.window_size))`; for token index 5 and window 4, the local test expects only indices `[2,3,4,5]` to remain nonzero.
- `GPTNeoBlock`: pre-LN attention, residual add, pre-LN MLP, residual add.
- MLP is ungated: `Linear(hidden -> intermediate)`, activation from `ACT2FN`, `Linear(intermediate -> hidden)`, dropout. Default `intermediate_size=None` means `4 * hidden_size`.
- `gelu_new` is `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`.
- Token embedding `wte`, learned absolute position embedding `wpe`, optional token type embedding reuse through `wte(token_type_ids)`, final `ln_f`, and tied LM head are source behavior for causal LM.
- `GPTNeoForCausalLM.forward` supports `logits_to_keep`; only the selected hidden positions are passed to `lm_head`.
- Cache input/output uses the shared `Cache` API; if `use_cache` is true and no cache is supplied, `DynamicCache(config=config)` is constructed. GPT-Neo config uses `window_size`, not shared-cache `sliding_window` or `layer_types`, so the shared cache source does not infer a pruned sliding-window cache for local GPT-Neo layers.
- `GPTNeoPreTrainedModel._can_compile_fullgraph = False` with a comment that it needs a hybrid cache.
- FlashAttention2 is advertised through `_supports_flash_attn=True` and attention implementation dispatch. In the inspected GPT-Neo FlashAttention2 forward path, the call passes `is_causal=True` and `softmax_scale=1.0`, but no GPT-Neo `window_size`/local attention argument is visible. Treat local-attention FlashAttention2 parity as a gated gap until verified.

## Representative config notes

- `EleutherAI/gpt-neo-125m`: 12 layers, hidden 768, 12 heads, head_dim 64, intermediate default 3072, alternating global/local repeated 6, max positions 2048, window 256, GPT2Tokenizer, vocab 50257.
- `EleutherAI/gpt-neo-1.3B`: 24 layers, hidden 2048, 16 heads, head_dim 128, intermediate default 8192, alternating global/local repeated 12, max positions 2048, window 256, GPT2Tokenizer, vocab 50257.
- `EleutherAI/gpt-neo-2.7B`: 32 layers, hidden 2560, 20 heads, head_dim 128, intermediate default 10240, alternating global/local repeated 16, max positions 2048, window 256, GPT2Tokenizer, vocab 50257.
- Tiny/private test repos exist for GPT-Neo, but public config raw access was not used as a source. Local tests provide a debug config with 2 layers, hidden 32, 4 heads, explicit `intermediate_size=37`, and `window_size=7`.

