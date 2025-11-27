import os
import random
import numpy as np
import tensorflow as tf

# ================== CONFIG & PATHS ==================

DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

TABLES_PATH = os.path.join(DATA_DIR, "common-tables.txt")
COLUMNS_PATH = os.path.join(DATA_DIR, "common-columns.txt")

BEST_MODEL_H5 = os.path.join(MODEL_DIR, "blind_sqli_gru_best.h5")
FINAL_MODEL_KERAS = os.path.join(MODEL_DIR, "blind_sqli_gru_final.keras")
FINAL_WEIGHTS_H5 = os.path.join(MODEL_DIR, "blind_sqli_gru_final.weights.h5")  # phải .weights.h5
LOG_CSV_PATH = os.path.join(MODEL_DIR, "training_log.csv")

# Hyperparameters
SEQ_LENGTH = 20
BATCH_SIZE = 256
EPOCHS = 50
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

EMBEDDING_DIM = 128
GRU_UNITS = 256
LEARNING_RATE = 1e-3

EXPECTED_TF_PREFIX = "2.19"  # Colab hiện giờ đang dùng TF 2.19.x
SEED = 42


# ================== TIỆN ÍCH CHUNG ==================

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[+] Global seed set to {seed}")


def configure_devices():
    if not tf.__version__.startswith(EXPECTED_TF_PREFIX):
        print(
            f"[!] TensorFlow version {tf.__version__} != {EXPECTED_TF_PREFIX}*. "
            "Vẫn train được bình thường, nhưng log chỉ để ông nhớ env hiện tại thôi."
        )

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        strategy = tf.distribute.MirroredStrategy()
        print(f"[+] Using GPU strategy with {strategy.num_replicas_in_sync} replica(s)")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
        print("[!] No GPU detected; using CPU strategy")
    return strategy


def safe_read_lines(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[!] Không tìm thấy file: {path} – nhớ đặt common-tables.txt và common-columns.txt trong thư mục data/"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def load_names(*paths):
    names = set()
    for path in paths:
        lines = safe_read_lines(path)
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                names.add(line)
    names = sorted(names)
    print(f"[+] Loaded {len(names)} unique names")
    return names


# ================== XỬ LÝ DỮ LIỆU ==================

def prepare_text():
    names = load_names(TABLES_PATH, COLUMNS_PATH)
    text = "\n".join(names) + "\n"

    vocab = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(vocab)}
    text_as_int = np.array([char2idx[c] for c in text], dtype=np.int32)

    print(f"[+] Corpus length (chars): {len(text_as_int)}")
    print(f"[+] Vocab size: {len(vocab)}")
    return text_as_int, vocab


def build_datasets(text_as_int: np.ndarray):
    inputs, targets = [], []
    for i in range(len(text_as_int) - SEQ_LENGTH):
        inputs.append(text_as_int[i: i + SEQ_LENGTH])
        targets.append(text_as_int[i + SEQ_LENGTH])

    X = np.array(inputs, dtype=np.int32)
    y = np.array(targets, dtype=np.int32)
    print(f"[+] Total samples: {len(X)}")

    total = len(X)
    val_size = int(total * VALIDATION_SPLIT)
    test_size = int(total * TEST_SPLIT)
    train_size = total - val_size - test_size

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    print(f"[+] Train size: {len(X_train)}")
    print(f"[+] Val size:   {len(X_val)}")
    print(f"[+] Test size:  {len(X_test)}")

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(4096, seed=SEED)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds


# ================== MODEL GRU ==================

def build_model(vocab_size: int, strategy):
    with strategy.scope():
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=EMBEDDING_DIM,
                    name="embedding",
                ),
                tf.keras.layers.GRU(
                    GRU_UNITS,
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.1,
                    name="gru_1",
                ),
                tf.keras.layers.GRU(
                    GRU_UNITS,
                    dropout=0.2,
                    recurrent_dropout=0.1,
                    name="gru_2",
                ),
                tf.keras.layers.Dense(256, activation="relu", name="dense_1"),
                tf.keras.layers.Dropout(0.2, name="dropout_1"),
                tf.keras.layers.Dense(vocab_size, activation="softmax", name="output"),
            ]
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    model.summary()
    return model


# ================== CALLBACKS ==================

def build_callbacks():
    callbacks = []

    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        )
    )

    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        )
    )

    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_H5,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
    )

    callbacks.append(
        tf.keras.callbacks.CSVLogger(LOG_CSV_PATH, append=False)
    )

    return callbacks


# ================== MAIN ==================

def main():
    set_global_seed(SEED)

    print("[*] Chuẩn bị dữ liệu...")
    text_as_int, vocab = prepare_text()
    train_ds, val_ds, test_ds = build_datasets(text_as_int)

    print("[*] Cấu hình thiết bị...")
    strategy = configure_devices()

    print("[*] Xây dựng mô hình...")
    model = build_model(len(vocab), strategy)

    print("[*] Khởi tạo callbacks...")
    callbacks = build_callbacks()

    print("[*] Bắt đầu train...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    print("[*] Đánh giá trên test set (với best weights do EarlyStopping restore)...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print(f"[+] Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    print("[*] Lưu model cuối cùng (sau epoch cuối cùng)...")
    model.save(FINAL_MODEL_KERAS)
    model.save_weights(FINAL_WEIGHTS_H5)

    print(f"[+] Best model (.h5) saved to:        {BEST_MODEL_H5}")
    print(f"[+] Final model (.keras) saved to:    {FINAL_MODEL_KERAS}")
    print(f"[+] Final weights (.weights.h5) saved to: {FINAL_WEIGHTS_H5}")
    print(f"[+] Training log (.csv) saved to:     {LOG_CSV_PATH}")


if __name__ == "__main__":
    main()

#main()