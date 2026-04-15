# disaster-tweets-deep-learning

Deep learning project for disaster tweet classification using supervised, unsupervised, and state-of-the-art NLP models.

## Setup

Install dependencies from repository root:

```bash
pip install -r requirements.txt
```

## Dataset

Expected Kaggle files:

- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv` (optional for Experiment 1)

## Experiment 1 (Supervised Learning)

Experiment 1 is implemented in `experiment1_supervised/` and evaluates multiple CNN and BiLSTM configurations.

Run from repository root:

```bash
python experiment1_supervised/run_experiment1.py --train data/train.csv --epochs 10 --batch_size 32
```

Optional flags:

- `--quick_mode` for a smaller sanity run
- `--max_experiments N` to cap number of configurations
- `--explore_only` to generate only exploration artifacts
- `--no-balance_train` to disable train split balancing

## Experiment 1 Outputs

Main pipeline outputs:

- `experiment1_supervised/outputs/archive/<timestamp>/...`
- `experiment1_supervised/outputs/latest/...`

For report/presentation-ready results, generate a clean final layer:

```bash
python experiment1_supervised/create_results_final.py
```

This creates:

- `experiment1_supervised/results_final/`

With key artifacts only (leaderboards, best-model plots/confusion matrices, and `final_summary.txt`).
