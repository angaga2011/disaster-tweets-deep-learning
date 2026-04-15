# Experiment 1 (Supervised Learning)

This folder contains the full Experiment 1 pipeline for the COMP 263 project:
- multi-configuration CNN experiments
- multi-configuration BiLSTM experiments
- regularization/normalization comparisons (dropout, L2, batch norm)
- automated reporting and artifact generation

## Files

- `run_experiment1.py`: main CLI entry point
- `config.py`: experiment grid and defaults
- `models.py`: CNN and BiLSTM builders
- `trainer.py`: per-experiment training and per-run artifact writing
- `evaluator.py`: metrics and classification report helpers
- `plotting.py`: training curves, confusion matrices, comparison chart
- `reporting.py`: global CSV leaderboards and markdown report
- `utils.py`: seed, path, output directory, and file helpers

## Run

From repository root:

```bash
python experiment1_supervised/run_experiment1.py --train data/train.csv --epochs 10 --batch_size 32
```

Optional flags:
- `--max_experiments N`: run first N configs from the curated grid
- `--quick_mode`: run a smaller subset (2 CNN + 2 BiLSTM) for quick testing
- `--output_dir PATH`: override output root (default: `experiment1_supervised/outputs`)

## Output Artifacts

Each run writes to:
- `experiment1_supervised/outputs/archive/<timestamp>/...`
- and mirrors latest run to `experiment1_supervised/outputs/latest/...`

Generated artifacts include:
- per-model training curves (`plots/`)
- validation/test confusion matrices (`confusion_matrices/`)
- per-model metrics JSON (`metrics/`)
- per-model human-readable TXT summaries (`summaries/`)
- combined summaries (`summary/`):
  - `all_experiments_summary.csv`
  - `leaderboard_by_test_f1.csv`
  - `leaderboard_by_test_accuracy.csv`
  - `experiment1_report.md`
  - best-model comparison plot in `plots/best_models_comparison.png`

## Final Report-Ready Results Layer

To create a clean, minimal folder for report writing/presentation:

```bash
python experiment1_supervised/create_results_final.py
```

This generates:

- `experiment1_supervised/results_final/`

Contents include:

- `all_experiments_summary.csv`
- `leaderboard_by_test_f1.csv`
- best CNN, best BiLSTM, and overall-best training curves
- best CNN, best BiLSTM, and overall-best test confusion matrices
- `overall_best_model.png` (best-model comparison plot)
- `final_summary.txt` (human-readable final summary)
- `collection_notes.txt` (copied/missing file notes)
