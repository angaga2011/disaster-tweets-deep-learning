# Experiment 1 Parameter Guide

This quick guide explains the parameter codes used in Experiment 1 model names and what each parameter controls.

## Model Name Pattern

Examples:
- `cnn_f64_k5_d32_do05_l2e4_no_bn`
- `bilstm_u64_stack_d32_do03_l2e3_bn`

## Shared Parameters

- `d32` / `d64`  
  Dense layer units (classifier head size). Larger can model more complex patterns but may overfit.

- `do03` / `do05`  
  Dropout rate (`0.3` / `0.5`). Higher dropout usually regularizes more strongly.

- `l2e4` / `l2e3`  
  L2 regularization strength (`1e-4` / `1e-3`). Larger value penalizes big weights more.

- `bn` / `no_bn`  
  Batch normalization enabled/disabled.

## CNN-Specific Parameters

- `f32` / `f64`  
  Number of convolution filters.

- `k3` / `k5`  
  Convolution kernel size (window length for local n-gram features).

## BiLSTM-Specific Parameters

- `u32` / `u64`  
  Number of LSTM units.

- `single` / `stack`  
  One BiLSTM layer vs stacked BiLSTM layers.

## Training/Vectorization Settings (from pipeline)

- `seq_len=60`  
  Token sequence length after vectorization (pad/truncate to 60).

- `max_tokens=20000`  
  Vocabulary size cap for `TextVectorization`.

- `embed_dim=128`  
  Embedding vector size per token.

- `learning_rate=1e-3`  
  Adam optimizer learning rate.

- `batch_size` and `epochs`  
  Set via CLI.

- `class_weight` and optional `--balance_train`  
  Imbalance handling methods in training.
