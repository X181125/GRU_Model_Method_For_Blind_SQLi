import os
import numpy as np
import tensorflow as tf

# Paths
DATA_DIR = os.path.join("data")
MODEL_PATH = os.path.join("trained_models", "blind_sqli_gru_model.h5")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)


def load_vocab():
    """Build vocab from cleaned table/column names (comments ignored)."""
    names = set()
    for path in (
        os.path.join(DATA_DIR, "common-tables.txt"),
        os.path.join(DATA_DIR, "common-columns.txt"),
    ):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    names.add(line)
    text = "\n".join(sorted(names)) + "\n"
    vocab = sorted(set(text))
    return vocab, text


vocab, text = load_vocab()
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for i, c in enumerate(vocab)}


def generate_name(max_len=20):
    sequence = [char2idx["\n"]]
    name = ""
    for _ in range(max_len):
        x = np.array([sequence])
        preds = model.predict(x, verbose=0)[0]
        next_idx = np.random.choice(len(preds), p=preds)
        next_char = idx2char[next_idx]
        if next_char == "\n":
            break
        name += next_char
        sequence.append(next_idx)
    return name


if __name__ == "__main__":
    for _ in range(10):
        print(generate_name())
