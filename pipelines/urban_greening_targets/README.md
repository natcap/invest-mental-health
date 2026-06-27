# Urban Greening Targets Pipeline (CLI)

This module is a command-line pipeline for collecting city-level urban greening
targets in the U.S. (tree canopy goals, tree planting goals, and green
infrastructure targets).

It is designed for reproducible, stage-based processing with resumable runs and
structured output files for analysis.

## What this pipeline does

For each city, the pipeline:

- discovers candidate sources (city pages, plans, PDFs)
- collects page/PDF text
- extracts target statements with structured LLM output
- scores evidence quality
- consolidates records at city level
- applies QA checks and fallback classification
- exports machine-readable output files

## Stages (A to G)

- `Stage A — Discovery`: Search and assemble official + secondary candidate URLs
- `Stage B — Collection`: Download/parse HTML and PDF text content
- `Stage C — Extraction`: Extract target records into structured schema
- `Stage D — Scoring`: Assign evidence-strength scores
- `Stage E — Consolidation`: Merge/deduplicate city records
- `Stage F — QA`: Verify records and assign final city status/fallback
- `Stage G — Export`: Write JSON, CSV, Markdown, and summary outputs

## Folder layout

```text
pipelines/urban_greening_targets/
  ├── README.md
  ├── .env.example
  ├── requirements.txt
  ├── pyproject.toml
  ├── scripts/
  │   └── run_all.sh
  ├── src/
  │   └── urban_greening_targets/
  │       ├── cli.py
  │       ├── config.py
  │       ├── cities.py
  │       ├── prompts/
  │       └── pipeline/
  │           ├── db.py
  │           ├── models.py
  │           ├── stage_a_discovery.py
  │           ├── stage_b_collection.py
  │           ├── stage_c_extraction.py
  │           ├── stage_d_scoring.py
  │           ├── stage_e_consolidation.py
  │           ├── stage_f_qa.py
  │           └── stage_g_export.py
  ├── data/               # runtime outputs (gitignored)
  └── db/                 # sqlite runtime state (gitignored)
```

## Prerequisites

- Python 3.10+
- Network access for source fetching
- API keys:
  - `OPENAI_API_KEY` for extraction/verification
  - `SERP_API_KEY` for web source discovery

## Setup

From repo root:

```bash
cd pipelines/urban_greening_targets
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and set your keys.

## Environment variables

Main variables in `.env`:

- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_BASE_URL`: API base URL (default set in code)
- `SERP_API_KEY`: SerpAPI key
- `SERP_API_KEY1`, `SERP_API_KEY2`, ...: optional extra keys for rotation
- `LLM_MODEL`: model name used by extraction/discovery
- `LLM_TEMPERATURE`: model temperature (default `0.1`)
- `SEARCH_RATE_LIMIT`: search requests per second control
- `FETCH_RATE_LIMIT`: fetch requests per second control
- `MAX_RETRIES`: retry count for network operations
- `PAGE_CHAR_LIMIT`, `PDF_CHAR_LIMIT`: truncation limits for LLM context
- `BATCH_SIZE`: city batch size for orchestration

## Commands

All commands are run from `pipelines/urban_greening_targets`.

Run full pipeline:

```bash
PYTHONPATH=src python -m urban_greening_targets.cli run --all
```

Resume previous run:

```bash
PYTHONPATH=src python -m urban_greening_targets.cli run --all --resume
```

Run only selected cities:

```bash
PYTHONPATH=src python -m urban_greening_targets.cli run --cities "New York,NY" "Seattle,WA"
```

Debug run (stop after one stage):

```bash
PYTHONPATH=src python -m urban_greening_targets.cli run --cities "Seattle,WA" --stage discovery
```

Disable LLM-assisted discovery (still runs pipeline):

```bash
PYTHONPATH=src python -m urban_greening_targets.cli run --cities "Seattle,WA" --no-llm
```

Check run status:

```bash
PYTHONPATH=src python -m urban_greening_targets.cli status -v
```

Export outputs:

```bash
PYTHONPATH=src python -m urban_greening_targets.cli export
```

Shell helper script:

```bash
./scripts/run_all.sh
```

## Output files

Generated in `data/output/`:

- `urban_greening_targets.json`: full structured records
- `urban_greening_targets.csv`: tabular export for spreadsheet/stat analysis
- `urban_greening_targets.md`: markdown table summary
- `summary_statistics.md`: counts by status/state and basic totals
- `sources_checked.json`: per-city URL audit trail

Canonical runtime location is this module folder only:
`pipelines/urban_greening_targets/data/` and `pipelines/urban_greening_targets/db/`.

Runtime database:

- `db/pipeline.db`: stage-by-stage progress, resumability, and intermediate state

## Typical workflow

1. Run a small city sample first.
2. Check `status -v`.
3. Run `--all`.
4. Resume with `--resume` if interrupted.
5. Run `export`.
6. Use `data/output/urban_greening_targets.csv` as the main sheet file.

## Troubleshooting

- `No module named urban_greening_targets`:
  - run with `PYTHONPATH=src` or install package with `pip install -e .`
- `No OPENAI_API_KEY`:
  - set key in `.env` and restart shell
- discovery returns very few URLs:
  - verify `SERP_API_KEY`; optionally add multiple keys (`SERP_API_KEY1`, etc.)
- interrupted run:
  - re-run with `--resume`


