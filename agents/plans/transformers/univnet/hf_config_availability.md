# UnivNet HF config availability

Retrieved: 2026-05-13

- `dg845/univnet-dev`: public, Transformers-native `UnivNetModel` repo. API SHA `fbfdd9ac17e7708deb785156e18131e7aee10ee3`. Files include `config.json`, `preprocessor_config.json`, and `pytorch_model.bin`.
- `my3bikaht/univnet-RU`: public repo returned by HF model search for `univnet`, but has no `config.json`; files are `.gitattributes`, `README.md`, and `univnet_tts_ru_0200.pt`. Treat as an original-checkpoint conversion candidate, not a native Transformers config source.

No gated/401 UnivNet repos were observed in the small HF search. The main gap is scarcity: only one public in-library UnivNet config was accessible, so the sweep uses source defaults, the public checkpoint, and the local Transformers tiny test config rather than 3-5 production HF configs.
