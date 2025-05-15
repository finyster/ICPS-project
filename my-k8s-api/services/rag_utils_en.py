import numpy as np
import re
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer

from .rag_corpus_en import CORPUS_DOCS

# 初始化嵌入模型
_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
_EMBS = _EMBEDDER.encode(CORPUS_DOCS, normalize_embeddings=True)

# 初始化 FAISS 索引
_DIM = _EMBS.shape[1]
_INDEX = faiss.IndexFlatIP(_DIM)
_INDEX.add(np.asarray(_EMBS, dtype="float32"))

# SQLite 資料庫初始化
def init_database():
    conn = sqlite3.connect('logs.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        function_name TEXT,
        parameters TEXT,
        result TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

def log_to_database(query: str, function_name: str, parameters: str, result: str):
    """記錄執行日誌到資料庫並更新 FAISS 索引"""
    conn = sqlite3.connect('logs.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO logs (query, function_name, parameters, result)
    VALUES (?, ?, ?, ?)
    ''', (query, function_name, parameters, result))
    conn.commit()
    conn.close()
    update_faiss_index()

def load_logs():
    """從資料庫載入日誌並格式化為檢索用文本"""
    conn = sqlite3.connect('logs.db')
    cursor = conn.cursor()
    cursor.execute('SELECT query, function_name, parameters, result, timestamp FROM logs')
    logs = cursor.fetchall()
    conn.close()
    return [f"Query: {log[0]}, Function: {log[1]}, Parameters: {log[2]}, Result: {log[3]}, Timestamp: {log[4]}" for log in logs]

def update_faiss_index():
    """更新 FAISS 索引，包含語料庫和日誌"""
    global _INDEX, CORPUS_AND_LOGS
    log_texts = load_logs()
    CORPUS_AND_LOGS = CORPUS_DOCS + log_texts
    all_embeddings = _EMBEDDER.encode(CORPUS_AND_LOGS, normalize_embeddings=True)
    _INDEX = faiss.IndexFlatIP(_DIM)
    _INDEX.add(np.asarray(all_embeddings, dtype="float32"))

# 初始化資料庫和索引
init_database()
CORPUS_AND_LOGS = CORPUS_DOCS
update_faiss_index()

# 檢索函數
def rag_retrieve(query: str, top_k: int = 3) -> str:
    """
    檢索最相關的語料庫片段和日誌，結果以雙換行符分隔。
    """
    q_vec = _EMBEDDER.encode([query], normalize_embeddings=True)
    D, I = _INDEX.search(np.asarray(q_vec, dtype="float32"), top_k)
    hits = [CORPUS_AND_LOGS[i] for i in I[0] if i != -1]
    return "\n\n".join(hits)

# 語義過濾器
_ALLOWED_RES = r"\b(pod|namespace|node)\b"
_ALLOWED_METRIC = r"\b(cpu|memory|disk|network)\b"
_VALID_RANGE = r"\b(\d+)(m|h|d|mo)\b"

def is_supported(query: str) -> bool:
    """
    簡單的基於規則的過濾器：
    - 必須提到 pod、namespace 或 node
    - 必須提到一個支援的指標（cpu、memory、disk、network）
    - 必須包含合法的時間範圍（30m、1h 等）
    """
    if not re.search(_ALLOWED_RES, query, re.I): return False
    if not re.search(_ALLOWED_METRIC, query, re.I): return False
    if not re.search(_VALID_RANGE, query, re.I): return False
    return True