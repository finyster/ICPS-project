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