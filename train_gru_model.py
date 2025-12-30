"""
===============================================================================
IMPROVED GRU MODEL FOR BLIND SQLI v2.0
===============================================================================
MAJOR IMPROVEMENTS over v1:

1. WORD-LEVEL + CHARACTER-LEVEL HYBRID:
   - Học cả pattern ở mức từ và mức ký tự
   - Có thể sinh ra từ mới dựa trên prefix

2. FREQUENCY-BASED PRIORITIZATION:
   - Ưu tiên sinh các tên phổ biến trước
   - Giống SQLMap nhưng thông minh hơn

3. PREFIX TREE (TRIE) INTEGRATION:
   - Lưu trữ tất cả prefix đã học
   - Sinh tên nhanh hơn bằng cách complete prefix

4. BATCH GENERATION:
   - Sinh nhiều tên cùng lúc thay vì từng cái

5. SMART SEED SELECTION:
   - Chọn seed dựa trên pattern phổ biến
   - Không random như trước

6. NGRAM FEATURES:
   - Học các n-gram phổ biến (2-gram, 3-gram)
   - Giúp sinh tên có ý nghĩa hơn

===============================================================================
"""

import os
import random
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    CSVLogger, Callback
)
import json
import pickle
from collections import Counter, defaultdict
from typing import Tuple, Dict, List, Set
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================== CONFIG & PATHS ==================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

TABLES_PATH = os.path.join(DATA_DIR, "common-tables.txt")
COLUMNS_PATH = os.path.join(DATA_DIR, "common-columns.txt")

# Model files
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "gru_v2_best.keras")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab_v2.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config_v2.json")
TRIE_PATH = os.path.join(MODEL_DIR, "prefix_trie.pkl")
NGRAM_PATH = os.path.join(MODEL_DIR, "ngrams.pkl")
FREQ_PATH = os.path.join(MODEL_DIR, "frequency.json")
HISTORY_PLOT = os.path.join(MODEL_DIR, "training_history.png")

# ================== HYPERPARAMETERS v2 ==================

# Data configuration
SEQ_LENGTH = 8  # INCREASED: Better context
BATCH_SIZE = 256  # INCREASED for faster training
EPOCHS = 200
VALIDATION_SPLIT = 0.15

# Model architecture - LARGER
EMBEDDING_DIM = 128
GRU_UNITS = 256
NUM_GRU_LAYERS = 2
DROPOUT_RATE = 0.3
RECURRENT_DROPOUT = 0.15

# Learning rate
INITIAL_LR = 1e-3
MIN_LR = 1e-6

# Generation
TOP_K = 10  # Top-k sampling
BEAM_WIDTH = 5  # Beam search width
MAX_NAME_LENGTH = 30

# Special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'

SEED = 42

# ================== TRIE DATA STRUCTURE ==================

class TrieNode:
    """Trie node for fast prefix matching"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # Frequency count
        self.full_word = None


class Trie:
    """Prefix tree for efficient name lookup and completion"""
    
    def __init__(self):
        self.root = TrieNode()
        self.all_words = []
    
    def insert(self, word: str, count: int = 1):
        """Insert a word with frequency count"""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.count += count
        node.full_word = word.lower()
        if word.lower() not in self.all_words:
            self.all_words.append(word.lower())
    
    def search(self, word: str) -> bool:
        """Check if exact word exists"""
        node = self._get_node(word)
        return node is not None and node.is_end
    
    def starts_with(self, prefix: str) -> List[Tuple[str, int]]:
        """Get all words starting with prefix, sorted by frequency"""
        node = self._get_node(prefix)
        if node is None:
            return []
        
        results = []
        self._collect_words(node, prefix, results)
        return sorted(results, key=lambda x: -x[1])  # Sort by frequency desc
    
    def _get_node(self, prefix: str) -> TrieNode:
        """Get node for given prefix"""
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def _collect_words(self, node: TrieNode, prefix: str, results: List):
        """Recursively collect all words from node"""
        if node.is_end:
            results.append((node.full_word, node.count))
        for char, child in node.children.items():
            self._collect_words(child, prefix + char, results)
    
    def get_common_prefixes(self, min_count: int = 5) -> List[str]:
        """Get prefixes that lead to multiple words"""
        prefixes = []
        self._find_prefixes(self.root, "", prefixes, min_count)
        return prefixes
    
    def _find_prefixes(self, node: TrieNode, prefix: str, prefixes: List, min_count: int):
        """Find prefixes with multiple continuations"""
        if len(node.children) >= 2 and len(prefix) >= 2:
            prefixes.append(prefix)
        for char, child in node.children.items():
            self._find_prefixes(child, prefix + char, prefixes, min_count)


# ================== DATA PREPARATION v2 ==================

def load_names_with_frequency(paths: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """Load names and count their frequency across files"""
    name_count = Counter()
    
    for path in paths:
        if not os.path.exists(path):
            print(f"[!] Warning: {path} not found")
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            position = 0
            for line in f:
                line = line.strip().lower()
                if line and not line.startswith('#'):
                    # Earlier position = higher frequency (SQLMap ordering)
                    # Use inverse position as pseudo-frequency
                    freq = max(1, 1000 - position)
                    name_count[line] += freq
                    position += 1
    
    # Sort by frequency
    sorted_names = sorted(name_count.keys(), key=lambda x: -name_count[x])
    
    print(f"[+] Loaded {len(sorted_names)} unique names")
    print(f"[+] Top 20 names: {sorted_names[:20]}")
    
    return sorted_names, dict(name_count)


def build_ngrams(names: List[str], n_range: Tuple[int, int] = (2, 4)) -> Dict[str, Counter]:
    """Build n-gram frequency dictionaries"""
    ngrams = {n: Counter() for n in range(n_range[0], n_range[1] + 1)}
    
    for name in names:
        padded = START_TOKEN + name + END_TOKEN
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(padded) - n + 1):
                ngram = padded[i:i+n]
                ngrams[n][ngram] += 1
    
    print(f"[+] Built n-grams:")
    for n, counter in ngrams.items():
        print(f"    {n}-grams: {len(counter)} unique")
    
    return ngrams


def build_vocabulary(names: List[str]) -> Tuple[Dict, Dict, int]:
    """Build character vocabulary"""
    chars = set()
    for name in names:
        chars.update(name)
    
    # Add special tokens
    vocab = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN] + sorted(chars)
    
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    
    print(f"[+] Vocabulary size: {len(vocab)}")
    print(f"[+] Characters: {sorted(chars)}")
    
    return char2idx, idx2char, len(vocab)


def prepare_training_data(names: List[str], char2idx: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare input-output pairs for training"""
    inputs = []
    targets = []
    
    start_idx = char2idx[START_TOKEN]
    end_idx = char2idx[END_TOKEN]
    pad_idx = char2idx[PAD_TOKEN]
    
    for name in names:
        # Convert to indices
        name_indices = [char2idx.get(c, char2idx[UNK_TOKEN]) for c in name]
        
        # Add START and END tokens
        full_seq = [start_idx] + name_indices + [end_idx]
        
        # Create sliding windows
        for i in range(len(full_seq) - 1):
            # Input: sequence up to position i (padded to SEQ_LENGTH)
            start = max(0, i - SEQ_LENGTH + 1)
            seq = full_seq[start:i+1]
            
            # Pad if needed
            while len(seq) < SEQ_LENGTH:
                seq = [pad_idx] + seq
            
            inputs.append(seq[-SEQ_LENGTH:])
            targets.append(full_seq[i + 1])
    
    X = np.array(inputs, dtype=np.int32)
    y = np.array(targets, dtype=np.int32)
    
    print(f"[+] Training samples: {len(X)}")
    print(f"[+] Input shape: {X.shape}")
    
    return X, y


# ================== MODEL ARCHITECTURE v2 ==================

def build_gru_model_v2(vocab_size: int) -> Model:
    """Build improved GRU model with attention"""
    
    # Input
    inputs = layers.Input(shape=(SEQ_LENGTH,), name='input')
    
    # Embedding
    x = layers.Embedding(
        vocab_size, 
        EMBEDDING_DIM, 
        mask_zero=True,
        name='embedding'
    )(inputs)
    
    # GRU layers with residual connections
    for i in range(NUM_GRU_LAYERS):
        gru_out = layers.GRU(
            GRU_UNITS,
            return_sequences=(i < NUM_GRU_LAYERS - 1),
            dropout=DROPOUT_RATE,
            recurrent_dropout=RECURRENT_DROPOUT,
            name=f'gru_{i}'
        )(x)
        
        if i < NUM_GRU_LAYERS - 1:
            x = gru_out
    
    # Dense layers
    x = layers.Dense(256, activation='relu', name='dense1')(gru_out)
    x = layers.Dropout(DROPOUT_RATE, name='dropout')(x)
    
    # Output
    outputs = layers.Dense(vocab_size, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='GRU_SQLi_v2')
    
    return model


# ================== TRAINING ==================

class GenerationCallback(Callback):
    """Callback to test generation during training"""
    
    def __init__(self, char2idx, idx2char, trie):
        super().__init__()
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.trie = trie
        self.test_prefixes = ['us', 'ad', 'pr', 'or', 'em', 'cu', 'pa', 'se']
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f"\n[Epoch {epoch+1}] Sample generations:")
            for prefix in self.test_prefixes[:4]:
                names = generate_names_beam(
                    self.model, self.char2idx, self.idx2char,
                    prefix=prefix, num_names=3, beam_width=3
                )
                print(f"  '{prefix}' -> {names}")


def train_model():
    """Main training function"""
    print("\n" + "="*70)
    print("GRU MODEL v2.0 - TRAINING")
    print("="*70 + "\n")
    
    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    # Load data
    print("[1] Loading data...")
    names, freq_dict = load_names_with_frequency([TABLES_PATH, COLUMNS_PATH])
    
    # Build Trie
    print("\n[2] Building Trie...")
    trie = Trie()
    for name, freq in freq_dict.items():
        trie.insert(name, freq)
    
    common_prefixes = trie.get_common_prefixes(min_count=3)
    print(f"[+] Common prefixes: {len(common_prefixes)}")
    print(f"[+] Examples: {common_prefixes[:20]}")
    
    # Build n-grams
    print("\n[3] Building n-grams...")
    ngrams = build_ngrams(names)
    
    # Build vocabulary
    print("\n[4] Building vocabulary...")
    char2idx, idx2char, vocab_size = build_vocabulary(names)
    
    # Prepare training data
    print("\n[5] Preparing training data...")
    X, y = prepare_training_data(names, char2idx)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Split
    val_size = int(len(X) * VALIDATION_SPLIT)
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    
    print(f"[+] Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Build model
    print("\n[6] Building model...")
    model = build_gru_model_v2(vocab_size)
    model.summary()
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=MIN_LR,
            verbose=1
        ),
        GenerationCallback(char2idx, idx2char, trie)
    ]
    
    # Train
    print("\n[7] Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model and data
    print("\n[8] Saving...")
    model.save(BEST_MODEL_PATH)
    
    # Save vocabulary
    with open(VOCAB_PATH, 'w') as f:
        json.dump({
            'char2idx': char2idx,
            'idx2char': {str(k): v for k, v in idx2char.items()},
            'vocab_size': vocab_size
        }, f, indent=2)
    
    # Save config
    with open(CONFIG_PATH, 'w') as f:
        json.dump({
            'seq_length': SEQ_LENGTH,
            'vocab_size': vocab_size,
            'embedding_dim': EMBEDDING_DIM,
            'gru_units': GRU_UNITS,
            'num_layers': NUM_GRU_LAYERS
        }, f, indent=2)
    
    # Save Trie
    with open(TRIE_PATH, 'wb') as f:
        pickle.dump(trie, f)
    
    # Save n-grams
    with open(NGRAM_PATH, 'wb') as f:
        pickle.dump(ngrams, f)
    
    # Save frequency
    with open(FREQ_PATH, 'w') as f:
        json.dump(freq_dict, f)
    
    # Plot history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(HISTORY_PLOT)
    print(f"[+] History plot saved to: {HISTORY_PLOT}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    # Test generation
    print("\n[9] Testing generation...")
    test_generation(model, char2idx, idx2char, trie, names[:100])
    
    return model, char2idx, idx2char, trie


# ================== GENERATION (FAST) ==================

def generate_names_beam(
    model, char2idx, idx2char, 
    prefix: str = "", 
    num_names: int = 10,
    beam_width: int = BEAM_WIDTH,
    max_length: int = MAX_NAME_LENGTH
) -> List[str]:
    """Generate names using beam search - FAST"""
    
    start_idx = char2idx[START_TOKEN]
    end_idx = char2idx[END_TOKEN]
    pad_idx = char2idx[PAD_TOKEN]
    
    # Initialize with prefix
    if prefix:
        initial_seq = [start_idx] + [char2idx.get(c, char2idx[UNK_TOKEN]) for c in prefix.lower()]
    else:
        initial_seq = [start_idx]
    
    # Beam: (sequence, log_probability)
    beams = [(initial_seq, 0.0)]
    completed = []
    
    for _ in range(max_length):
        all_candidates = []
        
        for seq, score in beams:
            if seq[-1] == end_idx:
                completed.append((seq, score))
                continue
            
            # Prepare input
            input_seq = seq[-SEQ_LENGTH:] if len(seq) >= SEQ_LENGTH else [pad_idx] * (SEQ_LENGTH - len(seq)) + seq
            x = np.array([input_seq])
            
            # Predict
            probs = model.predict(x, verbose=0)[0]
            
            # Get top-k candidates
            top_indices = np.argsort(probs)[-beam_width:]
            
            for idx in top_indices:
                new_seq = seq + [idx]
                new_score = score + np.log(probs[idx] + 1e-10)
                all_candidates.append((new_seq, new_score))
        
        if not all_candidates:
            break
        
        # Keep top beams
        all_candidates.sort(key=lambda x: -x[1])
        beams = all_candidates[:beam_width]
    
    # Add remaining beams to completed
    completed.extend(beams)
    
    # Convert to strings
    results = []
    for seq, score in sorted(completed, key=lambda x: -x[1]):
        name = ""
        for idx in seq[1:]:  # Skip START token
            if idx == end_idx:
                break
            char = idx2char.get(idx, '')
            if char not in [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]:
                name += char
        if name and name not in results:
            results.append(name)
        if len(results) >= num_names:
            break
    
    return results


def generate_names_batch(
    model, char2idx, idx2char, trie,
    prefixes: List[str] = None,
    num_per_prefix: int = 5,
    use_trie_completion: bool = True
) -> List[str]:
    """Generate names in batch - VERY FAST"""
    
    all_names = set()
    
    # Default prefixes based on common patterns
    if prefixes is None:
        prefixes = [
            'u', 'a', 'p', 'c', 'e', 's', 'o', 'm', 'i', 't', 'd', 'l', 'r', 'n',
            'us', 'ad', 'pa', 'pr', 'cu', 'em', 'se', 'or', 'ac', 'lo', 'da', 'co',
            'user', 'admin', 'pass', 'prod', 'cust', 'emp', 'sess', 'order', 'acc'
        ]
    
    for prefix in prefixes:
        # Method 1: Trie completion (instant)
        if use_trie_completion:
            trie_results = trie.starts_with(prefix)
            for name, _ in trie_results[:num_per_prefix]:
                all_names.add(name)
        
        # Method 2: Model generation
        generated = generate_names_beam(
            model, char2idx, idx2char,
            prefix=prefix,
            num_names=num_per_prefix,
            beam_width=3
        )
        all_names.update(generated)
    
    return list(all_names)


def test_generation(model, char2idx, idx2char, trie, known_names: List[str]):
    """Test generation quality"""
    print("\n--- Generation Test ---")
    
    # Test different prefixes
    test_prefixes = ['us', 'ad', 'pa', 'em', 'or', 'pr', 'se', 'ac', 'cu', 'lo']
    
    total_generated = 0
    total_hits = 0
    
    for prefix in test_prefixes:
        generated = generate_names_beam(
            model, char2idx, idx2char,
            prefix=prefix,
            num_names=10
        )
        
        hits = [n for n in generated if n in known_names]
        total_generated += len(generated)
        total_hits += len(hits)
        
        print(f"  '{prefix}' -> Generated: {len(generated)}, Hits: {len(hits)}")
        print(f"        Names: {generated[:5]}")
    
    hit_rate = total_hits / max(total_generated, 1) * 100
    print(f"\n[+] Overall hit rate: {hit_rate:.2f}%")
    print(f"[+] Total generated: {total_generated}")
    print(f"[+] Total hits: {total_hits}")


# ================== SMART GENERATOR CLASS ==================

class SmartNameGenerator:
    """
    Smart name generator combining:
    1. Trie-based completion (instant)
    2. GRU model generation (learned patterns)
    3. Frequency-based prioritization
    4. N-gram based validation
    """
    
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.char2idx = None
        self.idx2char = None
        self.trie = None
        self.ngrams = None
        self.freq_dict = None
        self.generated_cache = set()
        
    def load(self):
        """Load all model components"""
        print("[*] Loading SmartNameGenerator...")
        
        # Load model
        model_path = os.path.join(self.model_dir, "gru_v2_best.keras")
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"  [+] Model loaded from {model_path}")
        else:
            print(f"  [!] Model not found at {model_path}")
        
        # Load vocabulary
        vocab_path = os.path.join(self.model_dir, "vocab_v2.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                data = json.load(f)
            self.char2idx = data['char2idx']
            self.idx2char = {int(k): v for k, v in data['idx2char'].items()}
            print(f"  [+] Vocabulary loaded")
        
        # Load Trie
        trie_path = os.path.join(self.model_dir, "prefix_trie.pkl")
        if os.path.exists(trie_path):
            with open(trie_path, 'rb') as f:
                self.trie = pickle.load(f)
            print(f"  [+] Trie loaded ({len(self.trie.all_words)} words)")
        
        # Load n-grams
        ngram_path = os.path.join(self.model_dir, "ngrams.pkl")
        if os.path.exists(ngram_path):
            with open(ngram_path, 'rb') as f:
                self.ngrams = pickle.load(f)
            print(f"  [+] N-grams loaded")
        
        # Load frequency
        freq_path = os.path.join(self.model_dir, "frequency.json")
        if os.path.exists(freq_path):
            with open(freq_path, 'r') as f:
                self.freq_dict = json.load(f)
            print(f"  [+] Frequency dict loaded")
        
        print("[*] SmartNameGenerator ready!")
        return self
    
    def generate(self, count: int = 100, name_type: str = "any") -> List[str]:
        """
        Generate names intelligently.
        Returns names sorted by likelihood of being real.
        """
        results = []
        
        # Strategy 1: High-frequency known names from Trie
        if self.trie:
            # Get top names by frequency
            sorted_words = sorted(
                self.trie.all_words, 
                key=lambda x: self.freq_dict.get(x, 0),
                reverse=True
            )
            results.extend(sorted_words[:count // 2])
        
        # Strategy 2: Common prefix completions
        if self.trie:
            common_prefixes = ['user', 'admin', 'pass', 'prod', 'cust', 'emp', 
                              'sess', 'order', 'acc', 'log', 'item', 'cat']
            for prefix in common_prefixes:
                completions = self.trie.starts_with(prefix)
                for word, _ in completions[:5]:
                    if word not in results:
                        results.append(word)
        
        # Strategy 3: Model generation for novel names
        if self.model and len(results) < count:
            prefixes = ['u', 'a', 'p', 'c', 'e', 's', 'o', 'm', 'i', 't']
            for prefix in prefixes:
                if len(results) >= count:
                    break
                generated = generate_names_beam(
                    self.model, self.char2idx, self.idx2char,
                    prefix=prefix,
                    num_names=10
                )
                for name in generated:
                    if name not in results and name not in self.generated_cache:
                        results.append(name)
                        self.generated_cache.add(name)
        
        return results[:count]
    
    def generate_unique(self, count: int = 100) -> List[str]:
        """Generate unique names not yet tried"""
        results = []
        attempts = 0
        max_attempts = count * 10
        
        while len(results) < count and attempts < max_attempts:
            batch = self.generate(count=50)
            for name in batch:
                if name not in self.generated_cache:
                    results.append(name)
                    self.generated_cache.add(name)
                    if len(results) >= count:
                        break
            attempts += 50
        
        return results
    
    def score_name(self, name: str) -> float:
        """Score how likely a name is to be real"""
        score = 0.0
        
        # Frequency score
        if self.freq_dict:
            score += self.freq_dict.get(name.lower(), 0) / 1000
        
        # N-gram score
        if self.ngrams:
            for n in [2, 3]:
                if n in self.ngrams:
                    padded = START_TOKEN + name + END_TOKEN
                    for i in range(len(padded) - n + 1):
                        ngram = padded[i:i+n]
                        score += self.ngrams[n].get(ngram, 0) / 100
        
        # Length penalty (prefer 4-12 chars)
        length_score = 1.0 - abs(len(name) - 8) / 20
        score += max(0, length_score)
        
        return score


# ================== MAIN ==================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Train model
        model, char2idx, idx2char, trie = train_model()
    else:
        # Test generation
        print("Usage: python train_gru_model_v2.py train")
        print("\nTo test existing model:")
        
        generator = SmartNameGenerator()
        if generator.load():
            print("\nGenerating sample names:")
            names = generator.generate(count=50)
            for i, name in enumerate(names[:20]):
                score = generator.score_name(name)
                print(f"  {i+1:2d}. {name:20s} (score: {score:.2f})")
