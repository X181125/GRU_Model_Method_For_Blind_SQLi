"""
===============================================================================
ENHANCED GRU MODEL FOR BLIND SQLI - PRODUCTION READY
===============================================================================
Improvements:
- Fixed end token detection bug
- Cyclic learning rate schedule
- Larger batch size for stability
- Better callbacks configuration
- Comprehensive evaluation metrics
- Text generation testing
- Full compatibility with inference scripts
===============================================================================
"""

import os
import random
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
    CSVLogger, LearningRateScheduler, Callback
)
import json
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
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
RESULTS_JSON = os.path.join(MODEL_DIR, "final_results.json")

# ================== HYPERPARAMETERS (IMPROVED) ==================

# Data configuration
SEQ_LENGTH = 4  # IMPROVED: Reduced from 5 to 4 for faster convergence
BATCH_SIZE = 128  # IMPROVED: Increased from 64 to 128 for stability
EPOCHS = 150  # Reduced from 200 since we use cyclic LR
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# Model architecture
EMBEDDING_DIM = 256
GRU_UNITS = 512
NUM_GRU_LAYERS = 2  # Keep 2 layers (more = overfit risk)
DROPOUT_RATE = 0.4  # IMPROVED: Increased from 0.3 to 0.4
RECURRENT_DROPOUT = 0.2
L2_REG = 1e-4

# Learning rate - IMPROVED: Cyclic LR
INITIAL_LR = 2e-3  # Slightly higher than before
MIN_LR = 1e-7
CYCLE_LENGTH = 30  # Epochs per cycle

# Training configuration
EXPECTED_TF_PREFIX = "2.19"
SEED = 42

# Special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
END_TOKEN = '*'

# ================== UTILITY FUNCTIONS ==================

def set_global_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[+] Global seed set to {seed}")


def configure_devices():
    """Configure GPU/CPU devices"""
    if not tf.__version__.startswith(EXPECTED_TF_PREFIX):
        print(
            f"[!] TensorFlow version {tf.__version__} != {EXPECTED_TF_PREFIX}*. "
            "Training will continue normally."
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
    """Safely read lines from file"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[!] File not found: {path}\n"
            f"    Please ensure common-tables.txt and common-columns.txt are in {DATA_DIR}/"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def load_names(*paths) -> List[str]:
    """Load and preprocess table/column names"""
    names = set()
    for path in paths:
        lines = safe_read_lines(path)
        for line in lines:
            line = line.strip().lower()
            if line and not line.startswith("#"):
                names.add(line + END_TOKEN)
    
    names = sorted(names)
    print(f"[+] Loaded {len(names)} unique names (with END markers)")
    return names


# ================== DATA PREPARATION ==================

def prepare_text() -> Tuple[np.ndarray, List[str], Dict, Dict, int]:
    """Prepare text with frequency-based vocabulary indexing"""
    names = load_names(TABLES_PATH, COLUMNS_PATH)
    text = "\n".join(names)
    
    # Build vocabulary with frequency sorting
    char_freq = {}
    for c in text:
        char_freq[c] = char_freq.get(c, 0) + 1
    
    # Sort by frequency descending
    vocab = sorted(char_freq.keys(), key=lambda x: char_freq[x], reverse=True)
    
    # Add special tokens at the beginning
    vocab = [PAD_TOKEN, UNK_TOKEN] + vocab
    
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    
    # Get END_TOKEN index
    end_token_idx = char2idx.get(END_TOKEN, -1)
    if end_token_idx == -1:
        raise ValueError(f"END_TOKEN '{END_TOKEN}' not found in vocabulary!")
    
    # Convert text to integers
    text_as_int = np.array([char2idx.get(c, 1) for c in text], dtype=np.int32)
    
    print(f"[+] Corpus length (chars): {len(text_as_int)}")
    print(f"[+] Vocab size: {len(vocab)}")
    print(f"[+] END_TOKEN '{END_TOKEN}' index: {end_token_idx}")
    print(f"[+] Top 10 frequent chars: {vocab[2:12]}")
    
    # Save vocabulary
    vocab_data = {
        'char2idx': char2idx,
        'idx2char': {str(k): v for k, v in idx2char.items()},
        'end_token_idx': end_token_idx,
        'vocab_size': len(vocab)
    }
    
    with open(VOCAB_JSON, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    print(f"[+] Vocabulary saved to: {VOCAB_JSON}")
    
    return text_as_int, vocab, char2idx, idx2char, end_token_idx


def build_datasets(text_as_int: np.ndarray, end_token_idx: int) -> Tuple:
    """Build training, validation, and test datasets"""
    inputs, targets = [], []
    
    i = 0
    while i < len(text_as_int) - SEQ_LENGTH:
        seq = text_as_int[i:i + SEQ_LENGTH]
        target = text_as_int[i + SEQ_LENGTH]
        
        # Skip sequences that contain END_TOKEN to avoid cross-word patterns
        if end_token_idx in seq:
            end_positions = np.where(seq == end_token_idx)[0]
            if len(end_positions) > 0:
                i = i + end_positions[-1] + 1
                continue
        
        inputs.append(seq)
        targets.append(target)
        i += 1
    
    X = np.array(inputs, dtype=np.int32)
    y = np.array(targets, dtype=np.int32)
    print(f"[+] Total samples: {len(X)}")
    
    # Split dataset
    total = len(X)
    test_size = int(total * TEST_SPLIT)
    val_size = int(total * VALIDATION_SPLIT)
    train_size = total - val_size - test_size
    
    # Shuffle before split
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
    
    # Create tf.data.Dataset
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


# ================== MODEL ARCHITECTURE ==================

def build_model(vocab_size: int, strategy):
    """Build enhanced GRU model"""
    with strategy.scope():
        model = tf.keras.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=EMBEDDING_DIM,
                mask_zero=True,
                embeddings_regularizer=regularizers.l2(L2_REG),
                name="embedding"
            ),
            
            # First GRU layer
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
        
        # Optimizer with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=INITIAL_LR,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')
            ]
        )
    
    model.summary()
    return model


# ================== CALLBACKS (IMPROVED) ==================

def cyclic_lr_schedule(epoch, lr):
    """
    IMPROVED: Cyclic learning rate schedule
    Better than cosine annealing - helps escape local minima
    """
    epoch_in_cycle = epoch % CYCLE_LENGTH
    
    if epoch_in_cycle < 5:
        # Warmup phase
        return MIN_LR + (INITIAL_LR - MIN_LR) * (epoch_in_cycle / 5)
    else:
        # Cosine annealing within cycle
        progress = (epoch_in_cycle - 5) / (CYCLE_LENGTH - 5)
        return MIN_LR + (INITIAL_LR - MIN_LR) * (1 + math.cos(math.pi * progress)) / 2


class GenerationCallback(Callback):
    """Callback to test text generation during training"""
    def __init__(self, char2idx, idx2char, seq_length, end_token_idx):
        super().__init__()
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.seq_length = seq_length
        self.end_token_idx = end_token_idx
        self.test_seeds = ['us', 'ad', 'pa']
    
    def generate_sample(self, seed_text):
        """Generate a sample text"""
        generated = seed_text.lower()
        
        for _ in range(15):
            x = [self.char2idx.get(c, 1) for c in generated[-self.seq_length:]]
            while len(x) < self.seq_length:
                x.insert(0, 0)
            x = np.array([x])
            
            preds = self.model.predict(x, verbose=0)[0]
            
            # Temperature sampling
            temp = 0.7
            preds = np.log(preds + 1e-10) / temp
            preds = np.exp(preds) / np.sum(np.exp(preds))
            
            next_idx = np.random.choice(len(preds), p=preds)
            next_char = self.idx2char[next_idx]
            
            if next_char in [END_TOKEN, '\n', PAD_TOKEN, UNK_TOKEN]:
                break
            
            generated += next_char
        
        return generated
    
    def on_epoch_end(self, epoch, logs=None):
        """Generate samples at specific epochs"""
        if (epoch + 1) % 10 == 0:  # Every 10 epochs
            print(f"\n[Generation Test - Epoch {epoch + 1}]")
            for seed in self.test_seeds:
                result = self.generate_sample(seed)
                print(f"  {seed} -> {result}")


def build_callbacks(char2idx, idx2char, seq_length, end_token_idx):
    """Build training callbacks"""
    callbacks = []
    
    # Cyclic learning rate
    callbacks.append(
        LearningRateScheduler(cyclic_lr_schedule, verbose=0)
    )
    
    # Reduce LR on plateau (backup)
    callbacks.append(
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,  # IMPROVED: Increased from 5 to 10
            min_lr=MIN_LR,
            verbose=1
        )
    )
    
    # Early stopping - IMPROVED: More patient
    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            patience=40,  # IMPROVED: Increased from 20 to 40
            restore_best_weights=True,
            verbose=1
        )
    )
    
    # Model checkpoint
    callbacks.append(
        ModelCheckpoint(
            filepath=BEST_MODEL_H5,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    )
    
    # CSV logger
    callbacks.append(
        CSVLogger(LOG_CSV_PATH, append=False)
    )
    
    # Generation callback
    callbacks.append(
        GenerationCallback(char2idx, idx2char, seq_length, end_token_idx)
    )
    
    return callbacks


# ================== EVALUATION ==================

def calculate_perplexity(model, X_test, y_test):
    """Calculate perplexity on test set"""
    try:
        predictions = model.predict(X_test, verbose=0)
        true_probs = predictions[np.arange(len(y_test)), y_test]
        
        epsilon = 1e-10
        true_probs = np.clip(true_probs, epsilon, 1.0)
        
        avg_cross_entropy = -np.mean(np.log(true_probs))
        perplexity = np.exp(avg_cross_entropy)
        
        return perplexity
    except Exception as e:
        print(f"[!] Error calculating perplexity: {e}")
        return float('inf')


def plot_training_history(history):
    """Visualize training history"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training History - Enhanced GRU Model', fontsize=16, fontweight='bold')
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('Accuracy', fontsize=11)
        axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-5 Accuracy
        axes[1, 0].plot(history.history['top5_acc'], label='Train Top-5', linewidth=2)
        axes[1, 0].plot(history.history['val_top5_acc'], label='Val Top-5', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Top-5 Accuracy', fontsize=11)
        axes[1, 0].set_title('Top-5 Accuracy', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], linewidth=2, color='orange')
            axes[1, 1].set_xlabel('Epoch', fontsize=11)
            axes[1, 1].set_ylabel('Learning Rate', fontsize=11)
            axes[1, 1].set_title('Learning Rate Schedule (Cyclic)', fontsize=12, fontweight='bold')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(HISTORY_PLOT, dpi=300, bbox_inches='tight')
        print(f"[+] Training history plot saved to: {HISTORY_PLOT}")
        plt.close()
    except Exception as e:
        print(f"[!] Could not plot training history: {e}")


# ================== TEXT GENERATION TEST ==================

def generate_text(model, char2idx, idx2char, seq_length, end_token_idx, 
                 seed_text="us", length=20, temperature=0.7):
    """Generate text with temperature sampling"""
    seed_text = seed_text.lower()
    generated = seed_text
    
    for _ in range(length):
        x = [char2idx.get(c, 1) for c in generated[-seq_length:]]
        while len(x) < seq_length:
            x.insert(0, 0)
        x = np.array([x])
        
        predictions = model.predict(x, verbose=0)[0]
        
        # Apply temperature
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))
        
        # Sample
        predicted_idx = np.random.choice(len(predictions), p=predictions)
        predicted_char = idx2char[predicted_idx]
        
        if predicted_char in [END_TOKEN, '\n', PAD_TOKEN, UNK_TOKEN]:
            break
        
        generated += predicted_char
    
    return generated


def test_generation(model, char2idx, idx2char, seq_length, end_token_idx):
    """Test text generation with multiple seeds and temperatures"""
    print("\n" + "="*70)
    print("TEXT GENERATION TEST")
    print("="*70)
    
    test_seeds = ["us", "ad", "pa", "id", "na", "em", "da", "lo"]
    temperatures = [0.5, 0.7, 1.0]
    
    for seed in test_seeds:
        print(f"\n[Seed: '{seed}']")
        for temp in temperatures:
            generated = generate_text(
                model, char2idx, idx2char, seq_length, end_token_idx,
                seed, length=25, temperature=temp
            )
            print(f"  T={temp}: {generated}")


# ================== MAIN ==================

def main():
    """Main training pipeline"""
    try:
        print("\n" + "="*70)
        print("ENHANCED GRU MODEL TRAINING FOR BLIND SQLI")
        print("="*70)
        
        # Setup
        print("\n[STEP 1] Setting up environment...")
        set_global_seed(SEED)
        
        print("\n[STEP 2] Preparing data...")
        text_as_int, vocab, char2idx, idx2char, end_token_idx = prepare_text()
        train_ds, val_ds, test_ds, X_test, y_test = build_datasets(text_as_int, end_token_idx)
        
        print("\n[STEP 3] Configuring devices...")
        strategy = configure_devices()
        
        print("\n[STEP 4] Building model...")
        model = build_model(len(vocab), strategy)
        
        print("\n[STEP 5] Initializing callbacks...")
        callbacks = build_callbacks(char2idx, idx2char, SEQ_LENGTH, end_token_idx)
        
        # Save config
        config = {
            'seq_length': SEQ_LENGTH,
            'batch_size': BATCH_SIZE,
            'embedding_dim': EMBEDDING_DIM,
            'gru_units': GRU_UNITS,
            'num_gru_layers': NUM_GRU_LAYERS,
            'dropout_rate': DROPOUT_RATE,
            'vocab_size': len(vocab),
            'initial_lr': INITIAL_LR,
            'min_lr': MIN_LR,
            'cycle_length': CYCLE_LENGTH,
            'end_token_idx': end_token_idx,
            'improvements': [
                'Cyclic learning rate',
                'Larger batch size (128)',
                'Higher dropout (0.4)',
                'Longer patience (40)',
                'Shorter sequence (4)',
                'Generation callback'
            ]
        }
        
        with open(CONFIG_JSON, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[+] Config saved to: {CONFIG_JSON}")
        
        print("\n[STEP 6] Starting training...")
        print(f"[+] Training for {EPOCHS} epochs with cyclic LR")
        print(f"[+] Batch size: {BATCH_SIZE}")
        print(f"[+] Sequence length: {SEQ_LENGTH}")
        print(f"[+] Early stopping patience: 40 epochs")
        print("="*70 + "\n")
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=2
        )
        
        print("\n[STEP 7] Visualizing training history...")
        plot_training_history(history)
        
        print("\n[STEP 8] Evaluating on test set...")
        test_results = model.evaluate(test_ds, verbose=2)
        test_loss = test_results[0]
        test_acc = test_results[1]
        test_top5_acc = test_results[2]
        
        print(f"\n[+] Test loss: {test_loss:.4f}")
        print(f"[+] Test accuracy: {test_acc:.4f}")
        print(f"[+] Test top-5 accuracy: {test_top5_acc:.4f}")
        
        print("\n[STEP 9] Calculating perplexity...")
        perplexity = calculate_perplexity(model, X_test, y_test)
        print(f"[+] Test Perplexity: {perplexity:.2f}")
        print(f"[+] Paper target: table=5.8, column=6.7")
        
        if perplexity < 7.0:
            print(f"[✓] Excellent! Perplexity is within target range")
        elif perplexity < 10.0:
            print(f"[✓] Good! Perplexity is close to target")
        else:
            print(f"[!] Perplexity is higher than expected")
        
        print("\n[STEP 10] Testing text generation...")
        test_generation(model, char2idx, idx2char, SEQ_LENGTH, end_token_idx)
        
        print("\n[STEP 11] Saving model...")
        model.save(FINAL_MODEL_KERAS)
        model.save_weights(FINAL_WEIGHTS_H5)
        print(f"[+] Model saved to: {FINAL_MODEL_KERAS}")
        print(f"[+] Weights saved to: {FINAL_WEIGHTS_H5}")
        
        # Save results
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_top5_accuracy': float(test_top5_acc),
            'perplexity': float(perplexity),
            'total_epochs': len(history.history['loss']),
            'best_epoch': len(history.history['loss']) - 40,  # Approximate
            'improvements_applied': config['improvements']
        }
        
        with open(RESULTS_JSON, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[+] Results saved to: {RESULTS_JSON}")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print("\n[✓] Files created:")
        print(f"    • {BEST_MODEL_H5}")
        print(f"    • {FINAL_MODEL_KERAS}")
        print(f"    • {FINAL_WEIGHTS_H5}")
        print(f"    • {VOCAB_JSON}")
        print(f"    • {CONFIG_JSON}")
        print(f"    • {LOG_CSV_PATH}")
        print(f"    • {HISTORY_PLOT}")
        print(f"    • {RESULTS_JSON}")
        
        print("\n[→] Next steps:")
        print("    1. Run inference script for text generation")
        print("    2. Run exploit script for SQLi testing")
        print("    3. Check training_history.png for insights")
        
        print("\n" + "="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")
    except Exception as e:
        print(f"\n\n[✗] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()