# GraniteMoeShared Source Notes

Audit target: `granitemoeshared` at Transformers commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Local source inspected:

- `X:/H/transformers/src/transformers/models/granitemoeshared/configuration_granitemoeshared.py`
- `X:/H/transformers/src/transformers/models/granitemoeshared/modeling_granitemoeshared.py`
- `X:/H/transformers/src/transformers/models/granitemoeshared/modular_granitemoeshared.py`
- `X:/H/transformers/tests/models/granitemoeshared/test_modeling_granitemoeshared.py`
- Contrast only: existing DinoML audits for `granitemoe`, `granitemoehybrid`, and `granite`.

Important source facts:

- `modeling_granitemoeshared.py` is generated from `modular_granitemoeshared.py`; parity should follow the generated file, while upstream source edits should target the modular file.
- The modular file subclasses the plain GraniteMoe implementation and changes only the decoder layer/model/LM wrapper to add a shared dense SwiGLU MLP branch after sparse MoE.
- The generated file inlines the inherited GraniteMoe code, so it is self-contained for DinoML parity inspection.
- Public HF configs found for `ibm/PowerMoE-3b` and `ibm-research/PowerMoE-3b` are `model_type: granitemoe`, not `granitemoeshared`, even though the `granitemoeshared` docstring/test examples load `PowerMoE-3b` through the shared class.
- Public `ibm-granite/granite-speech-*` configs are composite speech models whose `text_config` is dense `granite`, not `granitemoeshared`. They were saved only to document the mismatch with the config docstring checkpoint hint.
- Attempts to fetch plausible tiny `granitemoeshared` checkpoints such as `hf-internal-testing/tiny-random-GraniteMoeSharedForCausalLM` returned 401, so no public representative `granitemoeshared` config snapshot was available during this audit.

Saved config snapshots:

- `ibm__PowerMoE-3b.config.json`
- `ibm-research__PowerMoE-3b.config.json`
- `ibm-granite__granite-speech-3.2-8b.config.json`
- `ibm-granite__granite-speech-3.3-2b.config.json`
- `ibm-granite__granite-speech-3.3-8b.config.json`
- `ibm-granite__granite-speech-4.1-2b-plus.config.json`

Gated/unavailable source gap:

- A real public checkpoint with `model_type: granitemoeshared`, nonzero `shared_intermediate_size`, and production weights was not found. Access to the internal/gated tiny repos or an IBM shared-expert checkpoint would resolve exact deployed dimensions.
