# scripts/

Utility entrypoints, grouped by category. Run them from the repo root, e.g.
`uv run python scripts/<category>/<name>.py ...`.

```
scripts/
├── data/        fetch the task repo, inspect CSV structure
├── training/    train / evaluate / smoke-test
└── submission/  official scoring, CodaBench packaging
```

## data/

| Script | Purpose |
|---|---|
| `data/fetch_data.py` | Clone/update the official task repo (data, scorer, starter kit) into `data/raw/dico` |
| `data/inspect_data.py` | Label counts, twin/singleton structure, language pairs, lengths per CSV |

## training/

| Script | Purpose |
|---|---|
| `training/train.py` | Thin wrapper over `src.pipelines.train` (the real CLI) |
| `training/evaluate.py` | Score a checkpoint on labeled files under every decoding mode, optional JSON |
| `training/smoke_test.py` | End-to-end synthetic run with a tiny offline BERT: train, reload, predict |

## submission/

| Script | Purpose |
|---|---|
| `submission/official_score.py` | Run the organizers' scorer (`python -m evaluation_functions`) on a prediction CSV |
| `submission/make_submission.py` | Validate `track<N>_predictions.csv` against templates and zip them at the archive root |

Predictions themselves come from `uv run python -m src.pipelines.predict`.
