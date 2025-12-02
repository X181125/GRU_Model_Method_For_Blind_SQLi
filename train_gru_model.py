import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, LearningRateScheduler
import json
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

# ================== CONFIG & PATHS ==================

DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

TABLES_PATH = os.path.join(DATA_DIR, "common-tables.txt")
COLUMNS_PATH = os.path.join(DATA_DIR, "common-columns.txt")

BEST_MODEL_H5 = os.path.join(MODEL_DIR, "blind_sqli_gru_best.h5")
FINAL_MODEL_KERAS = os.path.join(MODEL_DIR, "blind_sqli_gru_final.keras")
FINAL_WEIGHTS_H5 = os.path.join(MODEL_DIR, "blind_sqli_gru_final.weights.h5")
LOG_CSV_PATH = os.path.join(MODEL_DIR, "training_log.csv")
VOCAB_JSON = os.path.join(MODEL_DIR, "vocab.json")
CONFIG_JSON = os.path.join(MODEL_DIR, "config.json")
HISTORY_PLOT = os.path.join(MODEL_DIR, "training_history.png")

# Hyperparameters - IMPROVED dựa trên paper
SEQ_LENGTH = 5  # Time steps optimal cho table/column names
BATCH_SIZE = 64
EPOCHS = 200
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# Model architecture - ENHANCED
EMBEDDING_DIM = 256
GRU_UNITS = 512
DROPOUT_RATE = 0.3
RECURRENT_DROPOUT = 0.2
L2_REG = 1e-4

# Learning rate schedule
INITIAL_LR = 1e-3
MIN_LR = 1e-6

EXPECTED_TF_PREFIX = "2.19"
SEED = 42

# Special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
END_TOKEN = '*'  # End of sequence marker

# ================== TIỆN ÍCH CHUNG ==================

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[+] Global seed set to {seed}")


def configure_devices():
    if not tf.__version__.startswith(EXPECTED_TF_PREFIX):
        print(
            f"[!] TensorFlow version {tf.__version__} != {EXPECTED_TF_PREFIX}*. "
            "Model vẫn train bình thường."
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


def safe_read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[!] Không tìm thấy file: {path} – nhớ đặt common-tables.txt và common-columns.txt trong thư mục data/"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def load_names(*paths) -> List[str]:
    """Load và preprocess table/column names theo paper"""
    names = set()
    for path in paths:
        lines = safe_read_lines(path)
        for line in lines:
            line = line.strip().lower()  # Lowercase như paper
            if line and not line.startswith("#"):
                # Thêm END token để model học khi nào kết thúc
                names.add(line + END_TOKEN)
    
    names = sorted(names)
    print(f"[+] Loaded {len(names)} unique names (with END markers)")
    return names


# ================== XỬ LÝ DỮ LIỆU - FIXED & IMPROVED ==================

def prepare_text() -> Tuple[np.ndarray, List[str], Dict, Dict, int]:
    """Prepare text với vocabulary frequency-based indexing như paper"""
    names = load_names(TABLES_PATH, COLUMNS_PATH)
    text = "\n".join(names)
    
    # Build vocabulary với frequency sorting (paper: smaller index = higher frequency)
    char_freq = {}
    for c in text:
        char_freq[c] = char_freq.get(c, 0) + 1
    
    # Sort by frequency descending
    vocab = sorted(char_freq.keys(), key=lambda x: char_freq[x], reverse=True)
    
    # Thêm special tokens ở đầu
    vocab = [PAD_TOKEN, UNK_TOKEN] + vocab  # 0: padding, 1: unknown
    
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    
    # FIX: Get actual END_TOKEN index
    end_token_idx = char2idx.get(END_TOKEN, -1)
    if end_token_idx == -1:
        raise ValueError(f"END_TOKEN '{END_TOKEN}' không có trong vocabulary!")
    
    text_as_int = np.array([char2idx.get(c, 1) for c in text], dtype=np.int32)  # 1 = <UNK>
    
    print(f"[+] Corpus length (chars): {len(text_as_int)}")
    print(f"[+] Vocab size: {len(vocab)}")
    print(f"[+] END_TOKEN '{END_TOKEN}' index: {end_token_idx}")
    print(f"[+] Top 10 frequent chars: {vocab[2:12]}")  # Skip PAD and UNK
    
    # Save vocabulary
    with open(VOCAB_JSON, 'w', encoding='utf-8') as f:
        json.dump({
            'char2idx': char2idx, 
            'idx2char': {str(k): v for k, v in idx2char.items()},
            'end_token_idx': end_token_idx
        }, f, ensure_ascii=False, indent=2)
    
    return text_as_int, vocab, char2idx, idx2char, end_token_idx


def build_datasets_improved(text_as_int: np.ndarray, end_token_idx: int) -> Tuple:
    """
    FIX: Build datasets với proper end token handling
    """
    inputs, targets = [], []
    
    i = 0
    while i < len(text_as_int) - SEQ_LENGTH:
        seq = text_as_int[i:i + SEQ_LENGTH]
        target = text_as_int[i + SEQ_LENGTH]
        
        # FIX: Sử dụng end_token_idx đúng thay vì hardcode 10
        if end_token_idx in seq:
            # Nếu có END token trong sequence, skip để tránh học cross-word patterns
            end_positions = np.where(seq == end_token_idx)[0]
            if len(end_positions) > 0:
                # Jump to after last END token
                i = i + end_positions[-1] + 1
                continue
        
        inputs.append(seq)
        targets.append(target)
        i += 1  # Sliding window với step=1
    
    X = np.array(inputs, dtype=np.int32)
    y = np.array(targets, dtype=np.int32)
    print(f"[+] Total samples: {len(X)}")
    
    # Split dataset
    total = len(X)
    test_size = int(total * TEST_SPLIT)
    val_size = int(total * VALIDATION_SPLIT)
    train_size = total - val_size - test_size
    
    # Shuffle trước khi split
    indices = np.arange(total)
    np.random.shuffle(indices)
    
    X = X[indices]
    y = y[indices]
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"[+] Train size: {len(X_train)}")
    print(f"[+] Val size:   {len(X_val)}")
    print(f"[+] Test size:  {len(X_test)}")
    
    # Create tf.data.Dataset với augmentation
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(10000, seed=SEED)
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
    
    return train_ds, val_ds, test_ds, X_test, y_test


# ================== MODEL GRU - ENHANCED ==================

def build_advanced_model(vocab_size: int, strategy):
    """
    Build model theo architecture trong paper nhưng improved:
    - 2 stacked GRU layers với 512 units
    - Stronger regularization
    - BatchNormalization
    """
    with strategy.scope():
        model = tf.keras.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=EMBEDDING_DIM,
                mask_zero=True,  # Mask padding tokens
                embeddings_regularizer=regularizers.l2(L2_REG),
                name="embedding"
            ),
            
            # First GRU layer - return sequences for stacking
            layers.GRU(
                GRU_UNITS,
                return_sequences=True,
                dropout=DROPOUT_RATE,
                recurrent_dropout=RECURRENT_DROPOUT,
                kernel_regularizer=regularizers.l2(L2_REG),
                recurrent_regularizer=regularizers.l2(L2_REG),
                name="gru_1"
            ),
            layers.BatchNormalization(name="bn_1"),
            
            # Second GRU layer
            layers.GRU(
                GRU_UNITS,
                dropout=DROPOUT_RATE,
                recurrent_dropout=RECURRENT_DROPOUT,
                kernel_regularizer=regularizers.l2(L2_REG),
                recurrent_regularizer=regularizers.l2(L2_REG),
                name="gru_2"
            ),
            layers.BatchNormalization(name="bn_2"),
            
            # Dense layers
            layers.Dense(
                512, 
                activation="relu",
                kernel_regularizer=regularizers.l2(L2_REG),
                name="dense_1"
            ),
            layers.Dropout(DROPOUT_RATE, name="dropout_1"),
            layers.BatchNormalization(name="bn_3"),
            
            layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=regularizers.l2(L2_REG),
                name="dense_2"
            ),
            layers.Dropout(DROPOUT_RATE, name="dropout_2"),
            
            # Output layer
            layers.Dense(vocab_size, activation="softmax", name="output")
        ])
        
        # Optimizer với gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=INITIAL_LR,
            clipnorm=1.0  # Gradient clipping để tránh exploding gradients
        )
        
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')]
        )
    
    model.summary()
    return model


# ================== CALLBACKS - ENHANCED ==================

def lr_schedule(epoch, lr):
    """Learning rate schedule: cosine annealing"""
    import math
    if epoch < 10:
        return INITIAL_LR
    else:
        decay = 0.5 * (1 + math.cos(math.pi * (epoch - 10) / (EPOCHS - 10)))
        return INITIAL_LR * decay + MIN_LR


def build_callbacks():
    callbacks = []
    
    # Learning rate scheduler
    callbacks.append(
        LearningRateScheduler(lr_schedule, verbose=1)
    )
    
    # Reduce LR on plateau (backup)
    callbacks.append(
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=MIN_LR,
            verbose=1
        )
    )
    
    # Early stopping với patience cao hơn
    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
    )
    
    # Model checkpoint - save best
    callbacks.append(
        ModelCheckpoint(
            filepath=BEST_MODEL_H5,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    )
    
    # CSV Logger
    callbacks.append(
        CSVLogger(LOG_CSV_PATH, append=False)
    )
    
    return callbacks


# ================== EVALUATION - FIXED ==================

def calculate_perplexity(model, X_test, y_test):
    """
    FIX: Tính perplexity đúng theo paper (equation 4)
    Perplexity = exp(average cross-entropy loss)
    """
    try:
        # Predict trực tiếp trên toàn bộ test set
        predictions = model.predict(X_test, verbose=0)
        
        # Calculate cross-entropy loss manually
        # predictions shape: (n_samples, vocab_size)
        # y_test shape: (n_samples,)
        
        # Get probabilities for true classes
        true_probs = predictions[np.arange(len(y_test)), y_test]
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        true_probs = np.clip(true_probs, epsilon, 1.0)
        
        # Calculate average cross-entropy
        avg_cross_entropy = -np.mean(np.log(true_probs))
        
        # Perplexity = exp(cross_entropy)
        perplexity = np.exp(avg_cross_entropy)
        
        return perplexity
    except Exception as e:
        print(f"[!] Error calculating perplexity: {e}")
        return float('inf')


def plot_training_history(history):
    """NEW: Visualize training history"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Train Acc')
        axes[0, 1].plot(history.history['val_accuracy'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-5 Accuracy
        axes[1, 0].plot(history.history['top5_acc'], label='Train Top-5')
        axes[1, 0].plot(history.history['val_top5_acc'], label='Val Top-5')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-5 Accuracy')
        axes[1, 0].set_title('Top-5 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(HISTORY_PLOT, dpi=300, bbox_inches='tight')
        print(f"[+] Training history plot saved to: {HISTORY_PLOT}")
        plt.close()
    except Exception as e:
        print(f"[!] Could not plot training history: {e}")


# ================== TEXT GENERATION - NEW ==================

def generate_text(model, char2idx, idx2char, seed_text="us", length=20, temperature=1.0):
    """
    NEW: Generate text với temperature sampling
    Lower temperature (e.g., 0.5) = more conservative
    Higher temperature (e.g., 1.5) = more creative
    """
    seed_text = seed_text.lower()
    generated = seed_text
    
    for _ in range(length):
        # Prepare input sequence
        x = [char2idx.get(c, 1) for c in generated[-SEQ_LENGTH:]]
        while len(x) < SEQ_LENGTH:
            x.insert(0, 0)  # Pad with 0 (PAD_TOKEN)
        x = np.array([x])
        
        # Predict
        predictions = model.predict(x, verbose=0)[0]
        
        # Apply temperature
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))
        
        # Sample from distribution
        predicted_idx = np.random.choice(len(predictions), p=predictions)
        predicted_char = idx2char[predicted_idx]
        
        # Stop if END_TOKEN
        if predicted_char == END_TOKEN:
            break
        
        generated += predicted_char
    
    return generated


def test_generation(model, char2idx, idx2char):
    """NEW: Test text generation với multiple seeds"""
    print("\n" + "="*60)
    print("TEXT GENERATION TEST")
    print("="*60)
    
    test_seeds = ["us", "ad", "pa", "id", "na"]
    temperatures = [0.5, 1.0, 1.5]
    
    for seed in test_seeds:
        print(f"\n[Seed: '{seed}']")
        for temp in temperatures:
            generated = generate_text(model, char2idx, idx2char, seed, length=30, temperature=temp)
            print(f"  T={temp}: {generated}")


# ================== MAIN - ENHANCED ==================

def main():
    try:
        set_global_seed(SEED)
        
        print("[*] Chuẩn bị dữ liệu...")
        text_as_int, vocab, char2idx, idx2char, end_token_idx = prepare_text()
        train_ds, val_ds, test_ds, X_test, y_test = build_datasets_improved(text_as_int, end_token_idx)
        
        print("[*] Cấu hình thiết bị...")
        strategy = configure_devices()
        
        print("[*] Xây dựng mô hình NÂNG CẤP...")
        model = build_advanced_model(len(vocab), strategy)
        
        print("[*] Khởi tạo callbacks...")
        callbacks = build_callbacks()
        
        # Save config
        config = {
            'seq_length': SEQ_LENGTH,
            'batch_size': BATCH_SIZE,
            'embedding_dim': EMBEDDING_DIM,
            'gru_units': GRU_UNITS,
            'dropout_rate': DROPOUT_RATE,
            'vocab_size': len(vocab),
            'initial_lr': INITIAL_LR,
            'end_token_idx': end_token_idx
        }
        with open(CONFIG_JSON, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("[*] Bắt đầu training với architecture nâng cấp...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=2
        )
        
        print("[*] Visualizing training history...")
        plot_training_history(history)
        
        print("[*] Đánh giá trên test set...")
        test_results = model.evaluate(test_ds, verbose=2)
        test_loss = test_results[0]
        test_acc = test_results[1]
        test_top5_acc = test_results[2]
        
        print(f"[+] Test loss: {test_loss:.4f}")
        print(f"[+] Test accuracy: {test_acc:.4f}")
        print(f"[+] Test top-5 accuracy: {test_top5_acc:.4f}")
        
        # FIX: Calculate perplexity correctly
        print("[*] Calculating perplexity...")
        perplexity = calculate_perplexity(model, X_test, y_test)
        print(f"[+] Test Perplexity: {perplexity:.2f} (Paper target: table=5.8, column=6.7)")
        
        # NEW: Test text generation
        test_generation(model, char2idx, idx2char)
        
        print("[*] Lưu model cuối cùng...")
        model.save(FINAL_MODEL_KERAS)
        model.save_weights(FINAL_WEIGHTS_H5)
        
        print(f"\n{'='*60}")
        print(f"[✓] TRAINING HOÀN TẤT!")
        print(f"{'='*60}")
        print(f"[+] Best model (.h5):        {BEST_MODEL_H5}")
        print(f"[+] Final model (.keras):    {FINAL_MODEL_KERAS}")
        print(f"[+] Final weights (.h5):     {FINAL_WEIGHTS_H5}")
        print(f"[+] Vocabulary (.json):      {VOCAB_JSON}")
        print(f"[+] Config (.json):          {CONFIG_JSON}")
        print(f"[+] Training log (.csv):     {LOG_CSV_PATH}")
        print(f"[+] History plot (.png):     {HISTORY_PLOT}")
        print(f"{'='*60}\n")
        
        # Save final results
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_top5_accuracy': float(test_top5_acc),
            'perplexity': float(perplexity)
        }
        results_path = os.path.join(MODEL_DIR, "final_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[+] Results saved to: {results_path}")
        
    except Exception as e:
        print(f"\n[✗] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

#main()