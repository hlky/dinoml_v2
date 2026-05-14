# X-MOD Source Notes

## Local source

- Transformers checkout: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `src/transformers/models/xmod`
- Primary files:
  - `configuration_xmod.py`
  - `modeling_xmod.py`
  - `convert_xmod_original_pytorch_checkpoint_to_pytorch.py`
- Auto mappings:
  - `src/transformers/models/auto/tokenization_auto.py`
  - `src/transformers/models/auto/modeling_auto.py`

## Source line anchors

- `configuration_xmod.py:25`: `XmodConfig`.
- `configuration_xmod.py:58`: `model_type = "xmod"`.
- `configuration_xmod.py:60-86`: source defaults for hidden size, heads, adapter flags, languages, decoder/cross-attention flags.
- `modeling_xmod.py:51`: embeddings.
- `modeling_xmod.py:141`: padding-aware position-id construction.
- `modeling_xmod.py:158`: eager dense attention fallback.
- `modeling_xmod.py:187`: self-attention.
- `modeling_xmod.py:255`: cross-attention.
- `modeling_xmod.py:346`: attention wrapper and pre/post norm handling.
- `modeling_xmod.py:401`: adapter bottleneck module.
- `modeling_xmod.py:419`: FFN output plus language-adapter routing.
- `modeling_xmod.py:442`: `lang_adapter` boolean-mask route over configured adapters.
- `modeling_xmod.py:466`: transformer layer.
- `modeling_xmod.py:541`: encoder stack and optional final prenorm LayerNorm.
- `modeling_xmod.py:598`: pretrained base class, attention backend support flags, language setter.
- `modeling_xmod.py:667`: base model forward and language-id/default-language handling.
- `modeling_xmod.py:778`: bidirectional versus causal/cross attention mask construction.
- `modeling_xmod.py:816`: causal LM head wrapper.
- `modeling_xmod.py:925`: masked LM head wrapper.
- `modeling_xmod.py:1008`: LM head dense/GELU/LayerNorm/decoder.
- `modeling_xmod.py:1036`: sequence classification.
- `modeling_xmod.py:1116`: multiple choice flattening.
- `modeling_xmod.py:1217`: token classification.
- `modeling_xmod.py:1283`: classification head.
- `modeling_xmod.py:1306`: question answering.
- `auto/tokenization_auto.py:341`: X-MOD maps to `XLMRobertaTokenizer`.
- `auto/modeling_auto.py:509,613,773,1372,1457,1580,1621`: AutoModel class mappings.

## Hugging Face config snapshots

Saved local snapshots:

- `_sources/facebook_xmod-base_config.json`
- `_sources/facebook_xmod-base_tokenizer_config.json`
- `_sources/facebook_xmod-base-75-269k_config.json`
- `_sources/facebook_xmod-large-prenorm_config.json`

Public config URLs:

- [facebook/xmod-base config](https://huggingface.co/facebook/xmod-base/raw/main/config.json), repo SHA from API: `1ff23836a9ee8b9656553630c33506a9a8a59c4f`.
- [facebook/xmod-base tokenizer config](https://huggingface.co/facebook/xmod-base/raw/main/tokenizer_config.json).
- [facebook/xmod-base-75-269k config](https://huggingface.co/facebook/xmod-base-75-269k/raw/main/config.json), repo SHA from API: `e6b16a689b4497bc3f9031f7f4e664ecf16afaf1`.
- [facebook/xmod-large-prenorm config](https://huggingface.co/facebook/xmod-large-prenorm/raw/main/config.json), repo SHA from API: `fe2362ccdffebbb976b51e7ccb13f66c8ce123c2`.

Unavailable/gated probes:

- `facebook/xmod-large`
- `facebook/xmod-base-prenorm`
- `facebook/xmod-base-13`
- `facebook/xmod-base-30`
- `facebook/xmod-base-75`

Those IDs returned 401 when fetching `config.json`; access would be needed to verify whether they are real private/gated variants or absent aliases.

