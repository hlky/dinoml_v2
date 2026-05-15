# ColPali Source Notes

Local Transformers checkout:

- Path: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family source: `src/transformers/models/colpali`

Fetched Hugging Face snapshots:

- `vidore/colpali-v1.2-hf`: native `config.json`, `preprocessor_config.json`, tokenizer metadata, safetensors index.
- `vidore/colpali-v1.3-hf`: native `config.json`, `preprocessor_config.json`, tokenizer metadata, safetensors index.
- `michaelfeil/colpali-v12-random-testing`: `config.json` is `model_type="paligemma"` with `architectures=["ColPali"]`; useful only as a nonstandard/debug snapshot, not native `model_type="colpali"`.
- `vidore/colpali-v1.2`, `vidore/colpali-v1.3`, `vidore/colpali2-3b-pt-448`: no native `config.json` at root; fetched `adapter_config.json`, `preprocessor_config.json`, tokenizer metadata, README/training files. These are original/PEFT-style adapter repos and need a separate adapter/merge admission path.

Key source facts:

- `ColPaliForRetrieval` delegates the neural body to `AutoModel.from_config(config.vlm_config)`, normally PaliGemma.
- Wrapper-owned runtime ops are final projection `Linear(text_hidden_size -> embedding_dim)`, L2 normalization over `dim=-1`, attention-mask multiplication, and retrieval scoring.
- `ColPaliProcessor` processes exactly one modality per call: images or text, never both.
- Image processor defaults include `data_format="channels_first"` and fetched v1.2/v1.3 configs emit `pixel_values` as `[B,3,448,448]`.
- Document/page embeddings and query embeddings are both multi-vector tensors `[B,S,128]`; retrieval scores are `[num_queries,num_passages]`.
- Score orientation in source: `einsum("bnd,csd->bcns").max(dim=3).sum(dim=2)`.
