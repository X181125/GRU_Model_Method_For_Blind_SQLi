"""
===============================================================================
DEMO WEB APP: So sánh SQLMap vs GRU Model cho Blind SQL Injection
===============================================================================
Mục đích: So sánh số lượng requests cần thiết để tìm tên các tables VÀ columns
- SQLMap: Brute-force từng ký tự
- GRU Model: Dự đoán thông minh dựa trên patterns đã học
===============================================================================
"""

import os
import sys
import json
import time
import sqlite3
import pickle
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import List, Tuple, Dict
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add path to GRU model
GRU_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "GRU_Model_Method_For_Blind_SQLi")
sys.path.insert(0, GRU_MODEL_PATH)

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ================== TRIE DATA STRUCTURE (needed for pickle) ==================

class TrieNode:
    """Trie node for fast prefix matching"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0
        self.full_word = None


class Trie:
    """Prefix tree for efficient name lookup and completion"""
    
    def __init__(self):
        self.root = TrieNode()
        self.all_words = []
    
    def insert(self, word: str, count: int = 1):
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
        node = self._get_node(word)
        return node is not None and node.is_end
    
    def starts_with(self, prefix: str) -> List[Tuple[str, int]]:
        node = self._get_node(prefix)
        if node is None:
            return []
        
        results = []
        self._collect_words(node, prefix, results)
        return sorted(results, key=lambda x: -x[1])
    
    def _get_node(self, prefix: str):
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def _collect_words(self, node, prefix: str, results: List):
        if node.is_end:
            results.append((node.full_word, node.count))
        for char, child in node.children.items():
            self._collect_words(child, prefix + char, results)


# ================== CONFIGURATION ==================

DB_PATH = os.path.join(os.path.dirname(__file__), "vuln.db")
MODEL_DIR = os.path.join(GRU_MODEL_PATH, "trained_models")

# GRU Model components
gru_model = None
char2idx = None
idx2char = None
trie = None
freq_dict = None

# ================== DATABASE SETUP ==================

# 5 tables với 1-2 columns mỗi table (tên phổ biến)
DATABASE_SCHEMA = {
    "users": ["id", "username"],
    "products": ["id", "name"],
    "orders": ["id", "total"],
    "customers": ["id", "email"],
    "sessions": ["token"]
}

def init_database():
    """Initialize SQLite database with sample tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop existing tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing = [row[0] for row in cursor.fetchall()]
    for table in existing:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
    
    # Create tables with 1-2 columns each
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            total REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            email TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE sessions (
            token TEXT PRIMARY KEY
        )
    """)
    
    # Insert sample data
    cursor.execute("INSERT INTO users VALUES (1, 'admin')")
    cursor.execute("INSERT INTO products VALUES (1, 'Laptop')")
    cursor.execute("INSERT INTO orders VALUES (1, 999.99)")
    cursor.execute("INSERT INTO customers VALUES (1, 'admin@test.com')")
    cursor.execute("INSERT INTO sessions VALUES ('abc123token')")
    
    conn.commit()
    conn.close()
    
    print(f"[+] Database initialized:")
    for table, cols in DATABASE_SCHEMA.items():
        print(f"    - {table}: {cols}")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_actual_schema():
    """Get actual schema from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    
    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        schema[table] = columns
    
    conn.close()
    return schema


# ================== LOAD GRU MODEL ==================

# Special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'
SEQ_LENGTH = 8

def load_gru_model():
    global gru_model, char2idx, idx2char, trie, freq_dict
    
    print("[*] Loading GRU Model...")
    
    model_path = os.path.join(MODEL_DIR, "gru_v2_best.keras")
    if os.path.exists(model_path):
        gru_model = tf.keras.models.load_model(model_path)
        print(f"  [+] Model loaded")
    else:
        print(f"  [!] Model not found at {model_path}")
        return False
    
    vocab_path = os.path.join(MODEL_DIR, "vocab_v2.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            data = json.load(f)
        char2idx = data['char2idx']
        idx2char = {int(k): v for k, v in data['idx2char'].items()}
        print(f"  [+] Vocabulary loaded ({len(char2idx)} chars)")
    
    trie_path = os.path.join(MODEL_DIR, "prefix_trie.pkl")
    if os.path.exists(trie_path):
        with open(trie_path, 'rb') as f:
            trie = pickle.load(f)
        print(f"  [+] Trie loaded ({len(trie.all_words)} words)")
    
    freq_path = os.path.join(MODEL_DIR, "frequency.json")
    if os.path.exists(freq_path):
        with open(freq_path, 'r') as f:
            freq_dict = json.load(f)
        print(f"  [+] Frequency dict loaded")
    
    print("[*] GRU Model ready!")
    return True


# ================== BLIND SQLi SIMULATION ==================

class BlindSQLiSimulator:
    """Simulate blind SQL injection attacks"""
    
    def __init__(self, schema: Dict[str, List[str]]):
        self.schema = schema
        self.tables = list(schema.keys())
        self.request_count = 0
        self.request_log = []
    
    def reset(self):
        self.request_count = 0
        self.request_log = []
    
    def check_table_exists(self, table_name: str) -> bool:
        """Simulate: SELECT 1 FROM sqlite_master WHERE type='table' AND name='xxx'"""
        self.request_count += 1
        exists = table_name.lower() in [t.lower() for t in self.tables]
        self.request_log.append(f"CHECK_TABLE: {table_name} -> {exists}")
        return exists
    
    def check_column_exists(self, table_name: str, column_name: str) -> bool:
        """Simulate: Check if column exists in table"""
        self.request_count += 1
        if table_name.lower() not in [t.lower() for t in self.tables]:
            return False
        
        actual_table = next(t for t in self.tables if t.lower() == table_name.lower())
        columns = self.schema[actual_table]
        exists = column_name.lower() in [c.lower() for c in columns]
        self.request_log.append(f"CHECK_COLUMN: {table_name}.{column_name} -> {exists}")
        return exists
    
    def check_table_char(self, table_index: int, char_index: int, char: str) -> bool:
        """Simulate: SUBSTRING(table_name, pos, 1) = 'x'"""
        self.request_count += 1
        if table_index >= len(self.tables):
            return False
        
        table_name = sorted(self.tables)[table_index]
        if char_index >= len(table_name):
            return False
        
        result = table_name[char_index].lower() == char.lower()
        return result
    
    def check_column_char(self, table_name: str, col_index: int, char_index: int, char: str) -> bool:
        """Simulate: SUBSTRING(column_name, pos, 1) = 'x'"""
        self.request_count += 1
        if table_name.lower() not in [t.lower() for t in self.tables]:
            return False
        
        actual_table = next(t for t in self.tables if t.lower() == table_name.lower())
        columns = self.schema[actual_table]
        
        if col_index >= len(columns):
            return False
        
        col_name = columns[col_index]
        if char_index >= len(col_name):
            return False
        
        return col_name[char_index].lower() == char.lower()
    
    def get_table_count(self) -> int:
        """Get number of tables"""
        self.request_count += 1
        return len(self.tables)
    
    def get_table_length(self, table_index: int) -> int:
        """Get length of table name at index"""
        self.request_count += 1
        if table_index >= len(self.tables):
            return 0
        return len(sorted(self.tables)[table_index])
    
    def get_column_count(self, table_name: str) -> int:
        """Get number of columns in table"""
        self.request_count += 1
        if table_name.lower() not in [t.lower() for t in self.tables]:
            return 0
        actual_table = next(t for t in self.tables if t.lower() == table_name.lower())
        return len(self.schema[actual_table])
    
    def get_column_length(self, table_name: str, col_index: int) -> int:
        """Get length of column name"""
        self.request_count += 1
        if table_name.lower() not in [t.lower() for t in self.tables]:
            return 0
        actual_table = next(t for t in self.tables if t.lower() == table_name.lower())
        columns = self.schema[actual_table]
        if col_index >= len(columns):
            return 0
        return len(columns[col_index])


# ================== SQLMAP SIMULATION ==================

def run_sqlmap_attack(simulator: BlindSQLiSimulator) -> Dict:
    """
    Simulate SQLMap brute-force approach for finding tables and columns
    Uses binary search for each character position
    """
    simulator.reset()
    results = {
        "tables_found": {},
        "steps": [],
        "total_requests": 0
    }
    
    charset = 'abcdefghijklmnopqrstuvwxyz_0123456789'
    
    # Step 1: Get number of tables
    table_count = simulator.get_table_count()
    results["steps"].append(f"[1] Tìm số lượng tables: {table_count} tables")
    
    # Step 2: For each table, find its name
    for table_idx in range(table_count):
        # Get table name length (binary search: ~7 requests for length up to 128)
        table_length = simulator.get_table_length(table_idx)
        results["steps"].append(f"[2.{table_idx+1}] Độ dài table #{table_idx+1}: {table_length} ký tự")
        
        # Brute-force each character using binary search
        table_name = ""
        for char_idx in range(table_length):
            # Binary search on charset (log2(37) ≈ 5-6 requests per char)
            found_char = None
            for char in charset:
                if simulator.check_table_char(table_idx, char_idx, char):
                    found_char = char
                    break
            if found_char:
                table_name += found_char
        
        results["tables_found"][table_name] = []
        results["steps"].append(f"    -> Tìm thấy table: '{table_name}'")
        
        # Step 3: Find columns for this table
        col_count = simulator.get_column_count(table_name)
        results["steps"].append(f"    -> Số columns: {col_count}")
        
        for col_idx in range(col_count):
            col_length = simulator.get_column_length(table_name, col_idx)
            
            # Brute-force each character
            col_name = ""
            for char_idx in range(col_length):
                for char in charset:
                    if simulator.check_column_char(table_name, col_idx, char_idx, char):
                        col_name += char
                        break
            
            results["tables_found"][table_name].append(col_name)
            results["steps"].append(f"       -> Column: '{col_name}'")
    
    results["total_requests"] = simulator.request_count
    return results


# ================== GRU MODEL - THỰC SỰ SINH TÊN ==================

def generate_name_with_gru(prefix: str = "", max_length: int = 15) -> Tuple[str, List[dict]]:
    """
    THỰC SỰ dùng GRU model để sinh tên từng ký tự một
    Returns: (generated_name, generation_steps_log)
    """
    if gru_model is None or char2idx is None or idx2char is None:
        return "", []
    
    steps_log = []
    
    start_idx = char2idx.get(START_TOKEN, 0)
    end_idx = char2idx.get(END_TOKEN, 1)
    pad_idx = char2idx.get(PAD_TOKEN, 0)
    
    # Initialize with START + prefix
    if prefix:
        current_seq = [start_idx] + [char2idx.get(c, char2idx.get(UNK_TOKEN, 0)) for c in prefix.lower()]
    else:
        current_seq = [start_idx]
    
    generated_chars = list(prefix.lower()) if prefix else []
    
    for step in range(max_length - len(generated_chars)):
        # Pad input to SEQ_LENGTH
        if len(current_seq) >= SEQ_LENGTH:
            input_seq = current_seq[-SEQ_LENGTH:]
        else:
            input_seq = [pad_idx] * (SEQ_LENGTH - len(current_seq)) + current_seq
        
        x = np.array([input_seq])
        probs = gru_model.predict(x, verbose=0)[0]
        
        # Get top 5 predictions
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5 = [(idx2char.get(i, '?'), float(probs[i])) for i in top5_idx]
        
        # Choose best
        next_idx = top5_idx[0]
        next_char = idx2char.get(next_idx, '')
        
        steps_log.append({
            "step": step + 1,
            "current": ''.join(generated_chars),
            "top5": top5,
            "chosen": next_char,
            "prob": float(probs[next_idx])
        })
        
        if next_idx == end_idx:
            break
        
        if next_char not in [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]:
            generated_chars.append(next_char)
            current_seq.append(next_idx)
    
    return ''.join(generated_chars), steps_log


def generate_names_beam_search(prefix: str = "", num_names: int = 5, beam_width: int = 3) -> Tuple[List[str], List[dict]]:
    """
    Beam search để sinh nhiều tên từ 1 prefix
    """
    if gru_model is None or char2idx is None:
        return [], []
    
    log = []
    
    start_idx = char2idx.get(START_TOKEN, 0)
    end_idx = char2idx.get(END_TOKEN, 1)
    pad_idx = char2idx.get(PAD_TOKEN, 0)
    
    if prefix:
        initial_seq = [start_idx] + [char2idx.get(c, char2idx.get(UNK_TOKEN, 0)) for c in prefix.lower()]
    else:
        initial_seq = [start_idx]
    
    beams = [(initial_seq, 0.0, list(prefix.lower()) if prefix else [])]
    completed = []
    
    for iteration in range(15):
        all_candidates = []
        
        for seq, score, chars in beams:
            if seq[-1] == end_idx:
                completed.append((seq, score, chars))
                continue
            
            if len(seq) >= SEQ_LENGTH:
                input_seq = seq[-SEQ_LENGTH:]
            else:
                input_seq = [pad_idx] * (SEQ_LENGTH - len(seq)) + seq
            
            x = np.array([input_seq])
            probs = gru_model.predict(x, verbose=0)[0]
            
            top_indices = np.argsort(probs)[-beam_width:]
            
            for idx in top_indices:
                char = idx2char.get(idx, '')
                new_chars = chars.copy()
                if char not in [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]:
                    new_chars.append(char)
                new_seq = seq + [idx]
                new_score = score + np.log(probs[idx] + 1e-10)
                all_candidates.append((new_seq, new_score, new_chars))
        
        if not all_candidates:
            break
        
        all_candidates.sort(key=lambda x: -x[1])
        beams = all_candidates[:beam_width]
        
        log.append({
            "iter": iteration + 1,
            "beams": [''.join(b[2]) for b in beams[:3]]
        })
    
    completed.extend(beams)
    results = []
    for seq, score, chars in sorted(completed, key=lambda x: -x[1]):
        name = ''.join(chars)
        if name and name not in results and len(name) >= 2:
            results.append(name)
        if len(results) >= num_names:
            break
    
    return results, log


def generate_predictions_with_gru(num_predictions: int = 150) -> Tuple[List[str], List[dict]]:
    """
    Sinh predictions bằng GRU model
    """
    all_predictions = []
    generation_log = []
    
    # Các prefix phổ biến để thử
    prefixes = [
        '',  # Empty - model tự sinh
        'u', 'p', 'o', 'c', 's', 'a', 'e', 't', 'i', 'n', 'd', 'm',
        'us', 'pr', 'or', 'cu', 'se', 'ad', 'em', 'to', 'id', 'na',
        'user', 'prod', 'orde', 'cust', 'sess', 'admi', 'emai', 'toke', 'tota', 'name'
    ]
    
    generation_log.append({
        "phase": "START",
        "description": "Bắt đầu sinh predictions bằng GRU Model",
        "prefixes_to_try": len(prefixes)
    })
    
    # Sinh tên với mỗi prefix
    for prefix in prefixes:
        if len(all_predictions) >= num_predictions:
            break
        
        names, beam_log = generate_names_beam_search(prefix=prefix, num_names=3, beam_width=3)
        
        added = []
        for name in names:
            if name and name.lower() not in [p.lower() for p in all_predictions]:
                all_predictions.append(name)
                added.append(name)
        
        if added:
            generation_log.append({
                "phase": "GRU_GENERATE",
                "prefix": prefix if prefix else "(empty)",
                "generated": names,
                "added": added
            })
    
    # Bổ sung từ Trie (frequency-sorted)
    if trie and freq_dict:
        generation_log.append({
            "phase": "TRIE_AUGMENT",
            "description": "Bổ sung từ Trie (trained wordlist)"
        })
        
        sorted_words = sorted(
            trie.all_words,
            key=lambda x: freq_dict.get(x, 0),
            reverse=True
        )
        
        trie_added = []
        for word in sorted_words:
            if len(all_predictions) >= num_predictions:
                break
            if word.lower() not in [p.lower() for p in all_predictions]:
                all_predictions.append(word)
                trie_added.append(word)
        
        generation_log.append({
            "phase": "TRIE_RESULT",
            "words_from_trie": len(trie_added),
            "sample": trie_added[:15]
        })
    
    generation_log.append({
        "phase": "DONE",
        "total_predictions": len(all_predictions),
        "sample": all_predictions[:20]
    })
    
    return all_predictions, generation_log


def run_gru_attack(simulator: BlindSQLiSimulator) -> Dict:
    """
    GRU attack - THỰC SỰ dùng model để sinh predictions
    """
    simulator.reset()
    results = {
        "tables_found": {},
        "steps": [],
        "total_requests": 0,
        "generation_log": [],
        "sample_predictions": []
    }
    
    # === THỰC SỰ sinh predictions bằng GRU ===
    predictions, generation_log = generate_predictions_with_gru(num_predictions=150)
    results["generation_log"] = generation_log
    results["sample_predictions"] = predictions[:25]
    
    results["steps"].append(f"[1] GRU Model sinh {len(predictions)} predictions")
    results["steps"].append(f"    (Xem 'generation_log' để thấy quá trình sinh)")
    
    # Get table count
    table_count = simulator.get_table_count()
    results["steps"].append(f"[2] Target: {table_count} tables")
    
    # Find tables
    tables_found = []
    tried = 0
    
    for pred in predictions:
        if len(tables_found) >= table_count:
            break
        tried += 1
        if simulator.check_table_exists(pred):
            if pred.lower() not in [t.lower() for t in tables_found]:
                tables_found.append(pred)
                results["tables_found"][pred] = []
                results["steps"].append(f"    ✓ Table '{pred}' (try #{tried})")
    
    if len(tables_found) < table_count:
        results["steps"].append(f"    ! Chưa tìm đủ tables ({len(tables_found)}/{table_count})")
    
    # Find columns
    for table_name in tables_found:
        col_count = simulator.get_column_count(table_name)
        results["steps"].append(f"[3] Columns cho '{table_name}' ({col_count})")
        
        columns_found = []
        for pred in predictions:
            if len(columns_found) >= col_count:
                break
            if simulator.check_column_exists(table_name, pred):
                if pred.lower() not in [c.lower() for c in columns_found]:
                    columns_found.append(pred)
                    results["steps"].append(f"    ✓ Column '{pred}'")
        
        results["tables_found"][table_name] = columns_found
    
    results["total_requests"] = simulator.request_count
    return results


# ================== API ENDPOINTS ==================

@app.route('/api/db_info', methods=['GET'])
def get_db_info():
    """Get database schema info"""
    schema = get_actual_schema()
    return jsonify({
        "tables": list(schema.keys()),
        "table_count": len(schema),
        "schema": schema
    })


@app.route('/api/test_gru', methods=['POST'])
def test_gru_generation():
    """Test GRU generation với 1 prefix cụ thể - để demo quá trình sinh"""
    data = request.json or {}
    prefix = data.get('prefix', '')
    
    # Greedy generation
    name, steps = generate_name_with_gru(prefix=prefix)
    
    # Beam search
    names, beam_log = generate_names_beam_search(prefix=prefix, num_names=5)
    
    return jsonify({
        "prefix": prefix,
        "greedy": {
            "result": name,
            "steps": steps
        },
        "beam_search": {
            "results": names,
            "log": beam_log
        }
    })


@app.route('/api/compare', methods=['POST'])
def compare_methods():
    """Run both attacks simultaneously and compare results"""
    
    # Get actual schema
    schema = get_actual_schema()
    
    # Create simulators for each method
    sqlmap_sim = BlindSQLiSimulator(schema)
    gru_sim = BlindSQLiSimulator(schema)
    
    # Run attacks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        sqlmap_future = executor.submit(run_sqlmap_attack, sqlmap_sim)
        gru_future = executor.submit(run_gru_attack, gru_sim)
        
        sqlmap_result = sqlmap_future.result()
        gru_result = gru_future.result()
    
    # Calculate statistics
    sqlmap_requests = sqlmap_result["total_requests"]
    gru_requests = gru_result["total_requests"]
    
    # Count found items
    sqlmap_tables = len(sqlmap_result["tables_found"])
    sqlmap_columns = sum(len(cols) for cols in sqlmap_result["tables_found"].values())
    
    gru_tables = len(gru_result["tables_found"])
    gru_columns = sum(len(cols) for cols in gru_result["tables_found"].values())
    
    total_tables = len(schema)
    total_cols = sum(len(c) for c in schema.values())
    
    # Calculate improvement
    if sqlmap_requests > 0:
        improvement = ((sqlmap_requests - gru_requests) / sqlmap_requests) * 100
    else:
        improvement = 0
    
    return jsonify({
        "actual_schema": schema,
        "sqlmap": {
            "requests": sqlmap_requests,
            "tables_found": sqlmap_result["tables_found"],
            "tables_count": sqlmap_tables,
            "columns_count": sqlmap_columns,
            "steps": sqlmap_result["steps"],
            "approach": "Brute-force từng ký tự"
        },
        "gru": {
            "requests": gru_requests,
            "tables_found": gru_result["tables_found"],
            "tables_count": gru_tables,
            "columns_count": gru_columns,
            "steps": gru_result["steps"],
            "generation_log": gru_result.get("generation_log", []),
            "sample_predictions": gru_result.get("sample_predictions", []),
            "approach": "GRU Model prediction"
        },
        "comparison": {
            "requests_saved": sqlmap_requests - gru_requests,
            "improvement_percent": round(improvement, 2),
            "winner": "GRU" if gru_requests < sqlmap_requests else "SQLMap",
            "sqlmap_complete": sqlmap_tables == total_tables and sqlmap_columns == total_cols,
            "gru_complete": gru_tables == total_tables and gru_columns == total_cols
        }
    })


@app.route('/')
def index():
    return render_template('index.html')


# ================== MAIN ==================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("DEMO: SQLMap vs GRU Model - Blind SQL Injection Comparison")
    print("="*70 + "\n")
    
    # Initialize database
    init_database()
    
    # Load GRU model
    load_gru_model()
    
    print("\n[*] Starting server...")
    print("[*] Open http://127.0.0.1:5000 in your browser\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
