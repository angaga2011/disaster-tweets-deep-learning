"""Model builders for CNN and BiLSTM experiment families."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from config import ExperimentConfig


def _maybe_batch_norm(x, enabled: bool):
    if enabled:
        return layers.BatchNormalization()(x)
    return x


def build_cnn_model(cfg: ExperimentConfig, vocab_size: int, seq_len: int) -> tf.keras.Model:
    inp = layers.Input(shape=(seq_len,), dtype="int64")
    x = layers.Embedding(vocab_size, cfg.embed_dim, mask_zero=False)(inp)
    x = layers.Conv1D(
        filters=cfg.filters,
        kernel_size=cfg.kernel_size,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.l2_value),
    )(x)
    x = _maybe_batch_norm(x, cfg.batch_norm)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(cfg.dropout)(x)
    x = layers.Dense(
        cfg.dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.l2_value),
    )(x)
    x = _maybe_batch_norm(x, cfg.batch_norm)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out, name=cfg.name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_bilstm_model(cfg: ExperimentConfig, vocab_size: int, seq_len: int) -> tf.keras.Model:
    inp = layers.Input(shape=(seq_len,), dtype="int64")
    x = layers.Embedding(vocab_size, cfg.embed_dim, mask_zero=True)(inp)
    x = layers.SpatialDropout1D(0.2)(x)

    if cfg.stacked:
        x = layers.Bidirectional(
            layers.LSTM(
                cfg.lstm_units,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.0,
            )
        )(x)
        x = layers.Bidirectional(
            layers.LSTM(
                max(cfg.lstm_units // 2, 8),
                return_sequences=False,
                dropout=0.2,
                recurrent_dropout=0.0,
            )
        )(x)
    else:
        x = layers.Bidirectional(
            layers.LSTM(
                cfg.lstm_units,
                return_sequences=False,
                dropout=0.2,
                recurrent_dropout=0.0,
            )
        )(x)

    x = layers.Dense(
        cfg.dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(cfg.l2_value),
    )(x)
    x = _maybe_batch_norm(x, cfg.batch_norm)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out, name=cfg.name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(cfg: ExperimentConfig, vocab_size: int, seq_len: int) -> tf.keras.Model:
    if cfg.family == "cnn":
        return build_cnn_model(cfg, vocab_size, seq_len)
    if cfg.family == "bilstm":
        return build_bilstm_model(cfg, vocab_size, seq_len)
    raise ValueError(f"Unsupported family: {cfg.family}")
