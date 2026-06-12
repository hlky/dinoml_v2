# Source Data

This skill should stay reusable across different audit sources.

Use the source data the user named first.

## Current bundled snapshot

This skill includes a snapshot of the current Transformers and Diffusers torch API audit under:

- `references/audit-snapshot/`

Treat it as the current default baseline when the user wants to prioritize next DinoML op work from a broad Transformers and Diffusers survey.

## How to use the bundled snapshot

Start with the summary reports:

- `references/audit-snapshot/combined_torch_api_aggregate_report.md`
- `references/audit-snapshot/torch_api_aggregate_report.md`
- `references/audit-snapshot/diffusers_torch_api_aggregate_report.md`

Use these categorization reports when triage depends on op kind or likely implementation shape:

- `references/audit-snapshot/torch_tensor_function_categories.md`
- `references/audit-snapshot/torch_api_unused_categories.md`
- `references/audit-snapshot/diffusers_torch_api_used_categories.md`
- `references/audit-snapshot/diffusers_torch_api_unused_categories.md`

Use the JSON and CSV files only when the summaries are not enough:

- `references/audit-snapshot/torch_api_by_model_family.json`
- `references/audit-snapshot/diffusers_torch_api_by_component.json`
- the corresponding CSV files

Do not load the entire snapshot by default. Read the smallest files that answer the current planning question.

## Future source flexibility

Do not assume this bundled snapshot is the only valid source.

This skill should also work with:

- new torch API audits generated later
- audits from another upstream library or repo
- narrower model-family-specific audits
- manually curated candidate lists

If the user provides a newer or more relevant source, prefer that source over the bundled snapshot.
