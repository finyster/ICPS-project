<<<<<<< HEAD
import sqlite3
import json

def log_function_call(query: str, function_name: str, parameters: dict, result: str):
    """
    記錄 function calling 的執行日誌到 SQLite 資料庫。
    
    Parameters:
        - query (str): 使用者的查詢
        - function_name (str): 調用的函數名稱
        - parameters (dict): 函數參數
        - result (str): 函數執行結果
    """
    from .rag_utils_en import log_to_database
    try:
        parameters_str = json.dumps(parameters)
        log_to_database(query, function_name, parameters_str, result)
    except Exception as e:
        print(f"Failed to log function call: {str(e)}")

def validate_log_entry(query: str, function_name: str, parameters: dict, result: str) -> bool:
    """
    驗證日誌條目是否有效。
    
    Returns:
        - bool: True 表示有效，False 表示無效
    """
    if not query or not isinstance(query, str):
        return False
    if not function_name or not isinstance(function_name, str):
        return False
    if not isinstance(parameters, dict):
        return False
    if not isinstance(result, str):
        return False
    return True
=======
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
>>>>>>> my-feature
