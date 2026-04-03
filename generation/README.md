# Generation Pipeline

This directory contains the data-preparation and model-querying workflow used to create synthetic survey responses.

## Workflow

1. Create a year-specific GSS extract with `scripts/00a_create_gss_extract_multiyear.R`.
2. Convert respondent demographics into natural-language personas with `scripts/00b_generate_personas.R`.
3. Query one or more LLMs with `scripts/01_generate_synthetic_GSS.py`.
4. Use the resulting CSV files in the analysis pipeline under `../analysis/`.

## Required External Inputs

- `data/gss7224_r1.dta`: the NORC GSS cumulative Stata file. This file is not tracked in Git.
- `OPENROUTER_API_KEY`: required for `01_generate_synthetic_GSS.py`.

## Core Commands

From `generation/scripts/`:

```bash
Rscript 00a_create_gss_extract_multiyear.R --year 2024
Rscript 00b_generate_personas.R
python 01_generate_synthetic_GSS.py --year 2024 --all-models --personas 1000
```

## Directory Map

- `data/`: processed GSS extracts, persona files, and small metadata tables.
- `scripts/`: extraction, persona generation, API-query, and testing scripts.
- `synthetic_data/year_2024/`: model response CSVs used by the analysis pipeline.
