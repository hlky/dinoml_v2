# CANINE Source Notes

Local Transformers checkout: `X:/H/transformers` at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Inspected files:

- `src/transformers/models/canine/modeling_canine.py`
- `src/transformers/models/canine/configuration_canine.py`
- `src/transformers/models/canine/tokenization_canine.py`
- `src/transformers/models/canine/__init__.py`

Fetched representative config snapshots:

- `google/canine-s`, repo sha `75d6d0b3f4d0bffed0ca7ddd0b73949977d4ca02`
- `google/canine-c`, repo sha `dc0eaffdff3fa9161613311c7096eeb3e133ee19`
- `hf-internal-testing/tiny-random-CanineForMultipleChoice`, repo sha `fa0451453ed202f903ff7dcf6071aab6630fb89f`
- `Splend1dchan/canine-s-squad`, repo sha `f59594f9476ad15513d6a78af4b3153f4613e5df`
- `celine98/canine-s-finetuned-sst2`, repo sha `3e85d1e3ddb84b98b0766fe587763f45dd6fb821`

Key source-derived observations:

- Tokenizer maps each Unicode character to its codepoint integer. Special codepoints use private-use ids: `[CLS]=0xE000`, `[SEP]=0xE001`, `[BOS]=0xE002`, `[MASK]=0xE003`, `[PAD]=0`.
- Character embeddings are not a single Unicode-sized table. They hash input ids through up to 16 fixed primes; default uses 8 shard embedding tables of shape `[16384, hidden_size / 8]`, then concatenates shards.
- The base model is encoder-only: one local shallow character encoder, strided Conv1d downsampling to molecule tokens, a full-attention deep molecule encoder, repeat-interleave upsampling plus Conv1d projection, and one final full-attention shallow character encoder.
- The only source local-attention use is the initial shallow encoder. It chunks non-overlapping windows of width/stride `local_transformer_stride` and runs dense attention per chunk. The constructor supports CLS-global options, but the instantiated initial encoder sets both CLS-global booleans false.
- `use_cache` appears in checkpoint configs but is not read by `CanineModel.forward`; there is no autoregressive decode or KV cache path.
- `CanineLMPredictionHead` and `CanineOnlyMLMHead` exist, but no public `CanineForMaskedLM` class is exported in this source. `ConvProjection.forward(final_seq_char_positions=...)` raises `NotImplementedError`.
