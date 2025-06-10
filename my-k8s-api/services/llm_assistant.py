# -*- coding: utf-8 -*-
import os
import json
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# --- 1. 設定 (Configuration) ---
# 將所有可設定的變數集中管理

load_dotenv()

# LLM 客戶端設定
# 建議使用環境變數來設定模型名稱，增加靈活性
FUNC_MODEL = os.getenv("FUNC_MODEL", "llama3.1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

# Prometheus 設定
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

# 對話歷史與內容長度限制
MAX_HISTORY_LEN = 20
MAX_TOOL_SUMMARY_LEN = 400
MAX_TOTAL_JSON_LEN = 15000

# 初始化 LLM 客戶端
client_fc = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
client_chat = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


# --- 2. Prometheus 服務層 (Service Layer) ---
# 專門負責與 Prometheus 溝通的底層函式

class PrometheusService:
    """封裝所有與 Prometheus API 的互動"""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """統一的請求處理函式"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # 簡化錯誤回報，只回傳重要的資訊
            return {"status": "error", "error": f"Request failed: {e.__class__.__name__}", "message": str(e)}

    def query(self, promql: str) -> Dict[str, Any]:
        """執行即時查詢 (query)"""
        return self._request("/api/v1/query", {"query": promql})

    def query_range(self, promql: str, start: datetime, end: datetime, step: str) -> Dict[str, Any]:
        """執行範圍查詢 (query_range)"""
        params = {
            "query": promql,
            "start": start.isoformat() + "Z",
            "end": end.isoformat() + "Z",
            "step": step,
        }
        return self._request("/api/v1/query_range", params)

# 建立一個全域的 Prometheus 服務實例
prometheus_service = PrometheusService(PROMETHEUS_URL)


# --- 3. 工具函式 (Tool Functions) & 資料模型 ---
# 這是我們要給 LLM 使用的工具箱，並搭配 Pydantic 進行參數驗證

class ToolInputBase(BaseModel):
    """Pydantic 模型基底，方便共用"""
    namespace: str = Field(..., description="Kubernetes namespace, e.g., 'default', 'kube-system'.")

class PodResourceInput(ToolInputBase):
    pod: str = Field(..., description="The name of the pod.")
    duration: str = Field("5m", description="Time duration for the query, e.g., '5m', '1h', '3d'.")

class TopKInput(ToolInputBase):
    k: int = Field(3, description="The number of top pods to return.", ge=1)
    duration: str = Field("5m", description="Time window for rate calculation, e.g., '5m', '1h'.")

class PredictionInput(PodResourceInput):
    future_duration: str = Field("1h", description="The future time window to predict, e.g., '30m', '2h'.")
    step: str = Field("5m", description="The time interval for data points in historical data.")

def _duration_to_seconds(duration: str) -> int:
    """將 '1h', '5m' 等格式的時間字串轉換為秒數"""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    match = re.match(r"(\d+)([smhdw])", duration)
    if not match:
        raise ValueError("Invalid duration format. Use formats like '30s', '10m', '2h', '1d', '1w'.")
    value, unit = match.groups()
    return int(value) * units[unit]

def get_pod_resource_usage(input_data: PodResourceInput, metric: str) -> str:
    """
    【整合工具】取得指定 Pod 的 CPU 或 Memory 使用率。
    - 合併了原本的 get_pod_cpu_usage 和 get_pod_memory_usage。
    """
    if metric not in ["cpu", "memory"]:
        return json.dumps({"error": "Invalid metric specified. Must be 'cpu' or 'memory'."})

    # 根據指標選擇不同的 PromQL
    if metric == "cpu":
        query = f'rate(container_cpu_usage_seconds_total{{namespace="{input_data.namespace}", pod="{input_data.pod}"}}[{input_data.duration}])'
    else: # memory
        query = f'container_memory_usage_bytes{{namespace="{input_data.namespace}", pod="{input_data.pod}"}}'
    
    result = prometheus_service.query(query)
    return json.dumps(result, indent=2)

def get_top_k_pods(input_data: TopKInput, metric: str) -> str:
    """
    【整合工具】取得指定 Namespace 中，CPU 或 Memory 使用率最高的前 K 個 Pod。
    - 合併了 get_top_cpu_pods 和 get_top_memory_pods。
    """
    if metric not in ["cpu", "memory"]:
        return json.dumps({"error": "Invalid metric specified. Must be 'cpu' or 'memory'."})

    if metric == "cpu":
        query = f'topk({input_data.k}, sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{input_data.namespace}"}}[{input_data.duration}])))'
    else: # memory
        query = f'topk({input_data.k}, sum by(pod)(container_memory_usage_bytes{{namespace="{input_data.namespace}"}}))'

    result = prometheus_service.query(query)
    return json.dumps(result, indent=2)

def predict_pod_cpu_usage(input_data: PredictionInput) -> str:
    """
    預測未來 Pod 的 CPU 使用率 (使用簡易的線性回歸)。
    """
    try:
        # 1. 取得歷史資料
        end = datetime.utcnow()
        start = end - timedelta(seconds=_duration_to_seconds(input_data.duration))
        query = f'rate(container_cpu_usage_seconds_total{{namespace="{input_data.namespace}", pod="{input_data.pod}"}}[{input_data.step}])'
        
        result = prometheus_service.query_range(query, start, end, input_data.step)

        if result.get("status") != "success" or not result["data"]["result"]:
            return json.dumps({"error": f"No historical data found for pod '{input_data.pod}' in namespace '{input_data.namespace}'."})

        values = result["data"]["result"][0]["values"]
        if len(values) < 2:
            return json.dumps({"error": "Not enough historical data to make a prediction."})

        # 2. 建立 DataFrame 並訓練模型
        df = pd.DataFrame(values, columns=["timestamp", "value"])
        df["timestamp"] = pd.to_numeric(df["timestamp"])
        df["value"] = pd.to_numeric(df["value"])

        X = df[["timestamp"]]
        y = df["value"]
        
        model = LinearRegression()
        model.fit(X, y)

        # 3. 產生未來時間點並預測
        future_step_seconds = _duration_to_seconds(input_data.step)
        future_steps_count = _duration_to_seconds(input_data.future_duration) // future_step_seconds
        last_timestamp = df["timestamp"].iloc[-1]
        
        future_timestamps = np.arange(1, future_steps_count + 1) * future_step_seconds + last_timestamp
        future_X = pd.DataFrame(future_timestamps, columns=["timestamp"])
        
        predictions = model.predict(future_X)

        # 4. 格式化輸出
        output = {
            "pod": input_data.pod,
            "prediction_window_start": datetime.fromtimestamp(future_timestamps[0]).isoformat(),
            "prediction_window_end": datetime.fromtimestamp(future_timestamps[-1]).isoformat(),
            "predictions": [
                {"timestamp": datetime.fromtimestamp(ts).isoformat(), "predicted_cpu_usage": round(p, 4)}
                for ts, p in zip(future_timestamps, predictions)
            ]
        }
        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred during prediction: {str(e)}"})

# --- 4. 工具定義 (Tool Definitions for LLM) ---
# 這是給 LLM 看的 "API 文件"，描述了每個工具的功能和參數

TOOLS_DEF = [
    {
        "type": "function",
        "function": {
            "name": "get_pod_resource_usage",
            "description": "Fetches the current CPU or Memory usage for a specific pod in a namespace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["cpu", "memory"]},
                    "namespace": {"type": "string"},
                    "pod": {"type": "string"},
                    "duration": {"type": "string", "default": "5m"},
                },
                "required": ["metric", "namespace", "pod"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_k_pods",
            "description": "Finds the top K pods with the highest CPU or Memory usage in a given namespace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["cpu", "memory"]},
                    "namespace": {"type": "string"},
                    "k": {"type": "integer", "default": 3},
                    "duration": {"type": "string", "default": "5m"},
                },
                "required": ["metric", "namespace"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_pod_cpu_usage",
            "description": "Predicts the future CPU usage for a pod based on its historical data using linear regression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "pod": {"type": "string"},
                    "duration": {"type": "string", "default": "1h"},
                    "future_duration": {"type": "string", "default": "1h"},
                    "step": {"type": "string", "default": "5m"},
                },
                "required": ["namespace", "pod"],
            },
        },
    },
]

# 將工具名稱映射到實際的 Python 函式
# 注意：這裡我們需要一個 wrapper 來處理 Pydantic 模型
def _tool_wrapper(func, model_class, **kwargs):
    """一個 wrapper，用於將字典參數轉換為 Pydantic 模型"""
    try:
        # 如果 metric 在 kwargs 中，將其作為獨立參數傳遞
        metric = kwargs.pop("metric", None)
        model_instance = model_class(**kwargs)
        if metric:
            return func(model_instance, metric=metric)
        return func(model_instance)
    except Exception as e:
        return json.dumps({"error": "Parameter validation failed", "details": str(e)})

FUNC_MAP = {
    "get_pod_resource_usage": lambda **kwargs: _tool_wrapper(get_pod_resource_usage, PodResourceInput, **kwargs),
    "get_top_k_pods": lambda **kwargs: _tool_wrapper(get_top_k_pods, TopKInput, **kwargs),
    "predict_pod_cpu_usage": lambda **kwargs: _tool_wrapper(predict_pod_cpu_usage, PredictionInput, **kwargs),
}


# --- 5. 對話核心邏輯 (Core Chat Logic) ---

SYSTEM_PROMPT = """
You are a highly intelligent Kubernetes (k8s) monitoring assistant. Your primary goal is to help users understand the performance and resource consumption of their k8s cluster by calling available tools to fetch real-time data from Prometheus.

**Your Capabilities:**
- Fetch CPU and Memory usage for specific pods.
- Identify the top resource-consuming pods in a namespace.
- Predict future CPU usage for a pod.

**Interaction Rules:**
1.  **Always be helpful and proactive.**
2.  **Clarify Ambiguity:** If a user's request is missing necessary information (e.g., "check cpu usage" without specifying a pod), you MUST ask for clarification. Do NOT guess or call a tool with incomplete parameters. Ask questions like: "For which pod and in which namespace are you interested?"
3.  **Summarize, Don't Just Dump:** When you get data from a tool, do not just dump the raw JSON. Summarize the key information in a clear, human-readable format. Use lists, bold text, and simple language.
4.  **Acknowledge Tool Usage:** Briefly mention what you are doing. For example: "I'm checking the CPU usage for the 'web-server-1' pod..."
5.  **Be concise:** Provide the information directly without unnecessary chatter.
"""

def _is_smalltalk(message: str) -> bool:
    """簡單判斷是否為閒聊"""
    smalltalk_patterns = ["hello", "hi", "who are you", "早安", "你好"]
    return message.lower().strip() in smalltalk_patterns

def _prune_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """裁剪對話歷史，防止內容過長"""
    # 1. 摘要工具回傳的內容
    for msg in history:
        if msg.get("role") == "tool" and len(msg.get("content", "")) > MAX_TOOL_SUMMARY_LEN:
            msg["content"] = msg["content"][:MAX_TOOL_SUMMARY_LEN] + "... (truncated)"
            
    # 2. 如果總長度還是太長，從前面開始刪除舊對話
    # (保留 system prompt 和最後幾輪對話)
    while len(json.dumps(history)) > MAX_TOTAL_JSON_LEN and len(history) > 5:
        # 刪除 system prompt 後的第一則訊息
        history.pop(1)
        
    # 3. 確保歷史訊息不會超過最大輪數
    if len(history) > MAX_HISTORY_LEN * 2: # 乘以2因為包含 user 和 assistant
        history = history[-MAX_HISTORY_LEN*2:]

    return history

def chat_with_llm(user_message: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """主對話流程函式"""
    history = history or []
    
    # 準備 messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_message}]
    
    if _is_smalltalk(user_message):
        # 閒聊直接回覆
        response = client_chat.chat.completions.create(model=CHAT_MODEL, messages=messages)
        assistant_response = response.choices[0].message.content
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_response})
        return {"assistant": assistant_response, "history": history}

    # --- 主要 Function Calling 流程 ---
    # 1. 讓 Function Calling 模型決定是否使用工具
    response_fc = client_fc.chat.completions.create(
        model=FUNC_MODEL,
        messages=messages,
        tools=TOOLS_DEF,
        tool_choice="auto"
    )
    response_message = response_fc.choices[0].message
    tool_calls = response_message.tool_calls

    # 2. 判斷是否需要呼叫工具
    if not tool_calls:
        # 如果模型判斷不需要工具，直接由聊天模型生成回答
        response = client_chat.chat.completions.create(model=CHAT_MODEL, messages=messages)
        assistant_response = response.choices[0].message.content
    else:
        # 3. 執行工具呼叫
        messages.append(response_message) # 將 assistant 的回覆 (包含 tool_calls) 加入歷史
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = FUNC_MAP.get(function_name)
            
            if function_to_call:
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
        
        # 4. 將工具的結果餵給聊天模型，生成最終的自然語言回覆
        final_response = client_chat.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
        )
        assistant_response = final_response.choices[0].message.content

    # 5. 更新對話歷史並回傳
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_response})
    
    # 最後再做一次歷史裁剪
    final_history = _prune_history(history)
    
    return {"assistant": assistant_response, "history": final_history}

# --- 6. 範例使用 ---
if __name__ == "__main__":
    conversation_history = []
    print("Kubernetes Monitoring Assistant. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        result = chat_with_llm(user_input, conversation_history)
        
        print(f"Assistant: {result['assistant']}")
        
        # 更新對話歷史
        conversation_history = result['history']