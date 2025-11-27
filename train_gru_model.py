import os
import numpy as np
import tensorflow as tf

# Paths
DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("trained_models")
MODEL_PATH = os.path.join(MODEL_DIR, "blind_sqli_gru_model.h5")

TABLES_PATH = os.path.join(DATA_DIR, "common-tables.txt")
COLUMNS_PATH = os.path.join(DATA_DIR, "common-columns.txt")

# Hyperparameters
SEQ_LENGTH = 20
BATCH_SIZE = 256
EPOCHS = 50
VALIDATION_SPLIT = 0.1


def configure_devices():
    """Prefer GPU training; fall back to CPU if unavailable."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                # If memory growth cannot be set, continue with default behaviour.
                pass
        strategy = tf.distribute.MirroredStrategy()
        print(f"[+] Using GPU strategy with {strategy.num_replicas_in_sync} replica(s)")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
        print("[!] No GPU detected; using CPU strategy")
    return strategy


def load_names(*paths):
    """Load unique table/column names, ignoring comments and blanks."""
    names = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    names.add(line)
    return sorted(names)


def prepare_text():
    names = load_names(TABLES_PATH, COLUMNS_PATH)
    text = "\n".join(names) + "\n"
    vocab = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(vocab)}
    text_as_int = np.array([char2idx[c] for c in text], dtype=np.int32)
    return text_as_int, vocab


def build_datasets(text_as_int):
    inputs, targets = [], []
    for i in range(len(text_as_int) - SEQ_LENGTH):
        inputs.append(text_as_int[i : i + SEQ_LENGTH])
        targets.append(text_as_int[i + SEQ_LENGTH])

    X = np.array(inputs, dtype=np.int32)
    y = np.array(targets, dtype=np.int32)

    split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(4096)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds


def build_model(vocab_size, strategy):
    embedding_dim = 128
    gru_units = 256

    with strategy.scope():
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(vocab_size, embedding_dim),
                tf.keras.layers.GRU(
                    gru_units,
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.1,
                ),
                tf.keras.layers.GRU(
                    gru_units,
                    dropout=0.2,
                    recurrent_dropout=0.1,
                ),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(vocab_size, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model


def main():
    text_as_int, vocab = prepare_text()
    train_ds, val_ds = build_datasets(text_as_int)

    strategy = configure_devices()
    model = build_model(len(vocab), strategy)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[+] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
