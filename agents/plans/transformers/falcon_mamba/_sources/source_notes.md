# Falcon Mamba Source Notes

Audit date: 2026-05-13

## Local source basis

- Transformers checkout: `transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Model directory: `transformers/src/transformers/models/falcon_mamba`
- Runtime source: `modeling_falcon_mamba.py`
- Future edit source: `modular_falcon_mamba.py`
- Config source: `configuration_falcon_mamba.py`
- Shared cache source inspected for state ABI: `transformers/src/transformers/cache_utils.py`
- Shared generation source inspected for `logits_to_keep` and generation slicing: `transformers/src/transformers/generation/utils.py`

`modeling_falcon_mamba.py` and `configuration_falcon_mamba.py` are generated from `modular_falcon_mamba.py`; future upstream source edits should be checked against the modular file, but the generated files are the import/runtime source basis for DinoML parity.

## Hugging Face config snapshots

Saved from `https://huggingface.co/{model_id}/resolve/main/config.json`:

- `tiiuae/falcon-mamba-7b` -> `tiiuae__falcon-mamba-7b.config.json`
- `tiiuae/falcon-mamba-7b-instruct` -> `tiiuae__falcon-mamba-7b-instruct.config.json`
- `tiiuae/falcon-mamba-7b-instruct-4bit` -> `tiiuae__falcon-mamba-7b-instruct-4bit.config.json`

Gated or unavailable during this audit:

- `tiiuae/falcon-mamba-7b-base`: HTTP 401 from config fetch.
- `hf-internal-testing/tiny-random-FalconMambaForCausalLM`: HTTP 401 from config fetch.

## Source-derived hotspots

- `FalconMambaConfig`: defaults and `__post_init__` dimension derivation.
- `FalconMambaMixer.__init__`: depthwise causal conv, input projection, selective `x_proj`, `dt_proj`, `A_log`, `D`, output projection, non-persistent RMS buffers.
- `FalconMambaMixer.cuda_kernels_forward`: fast inference path using `causal-conv1d` and `falcon_mamba-ssm` kernels.
- `FalconMambaMixer.slow_forward`: reference recurrence, convolution state update, B/C/dt RMS, discretization, sequential or associative scan.
- `FalconMambaModel.forward`: `DynamicCache(config=...)` creation and per-layer state threading.
- `FalconMambaForCausalLM.prepare_inputs_for_generation`: drops `attention_mask` after first cached generation step.
- `FalconMambaForCausalLM.forward`: `logits_to_keep` last-token logits optimization.
- `cache_utils.LinearAttentionLayer`: fixed-address `conv_states` and `recurrent_states`, beam reorder on batch dimension, reset/offload/prefetch behavior.

## Config trap

The reachable official configs declare `hidden_size=4096`, `intermediate_size=8192`, and `expand=16`. The inspected pinned `FalconMambaConfig.__post_init__` computes `intermediate_size = int(expand * hidden_size)`, which would produce `65536` from those serialized fields. DinoML should treat this as a source/config compatibility gap rather than silently inferring projection sizes from either field. First admission should require an explicit resolved intermediate size that matches loaded weights.
