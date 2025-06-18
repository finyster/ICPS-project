# services/log_handler.py
import sqlite3
import os
from datetime import datetime

DB_PATH = "function_call_logs.db"

def init_db():
    """初始化資料庫和資料表"""
    if os.path.exists(DB_PATH):
        return

    print("Initializing database for function call logs...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE successful_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_query TEXT NOT NULL,
        tool_call TEXT NOT NULL,
        timestamp DATETIME NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def log_successful_call(user_query: str, tool_call: str):
    """記錄一次成功的 Function Calling"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO successful_calls (user_query, tool_call, timestamp) VALUES (?, ?, ?)",
        (user_query, tool_call, datetime.now())
    )
    conn.commit()
    conn.close()

def get_all_logs_as_docs() -> list[str]:
    """從資料庫中獲取所有日誌，並格式化為 RAG 文件"""
    if not os.path.exists(DB_PATH):
        return []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT user_query, tool_call FROM successful_calls ORDER BY timestamp DESC LIMIT 500") # 只取最近500筆以防過多
    logs = cursor.fetchall()
    conn.close()
    
    # 將日誌轉換為對 LLM 有意義的字串格式
    return [f"Example of a successful past request: User asked \"{query}\" and the correct tool call was `<tool_code>{tool}</tool_code>`" for query, tool in logs]

# 在應用程式啟動時，確保資料庫已初始化
init_db()