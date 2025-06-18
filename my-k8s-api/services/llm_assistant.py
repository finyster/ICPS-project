# -*- coding: utf-8 -*-
import os
import json
import re
import time
from datetime import datetime, timedelta
<<<<<<< HEAD
from services.rag_utils_en import rag_retrieve, is_supported
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging
<<<<<<< HEAD
 
=======
from openai import OpenAI

>>>>>>> my-feature
class ChatResponse(BaseModel):
    assistant: Optional[str]
    history: List[dict]
 
=======
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression # 確保匯入 LinearRegression

# --- 關鍵修改：匯入 RAG 系統與日誌處理器 ---
from .rag_utils_dynamic import rag_system
from .log_handler import log_successful_call
# --- 結束修改 ---

# --- 1. 設定 (Configuration) ---
# 將所有可設定的變數集中管理

>>>>>>> my-feature
load_dotenv()
<<<<<<< HEAD
 
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
MODEL          = "llama3-8b-8192"   # 請改成你帳號可用的 Groq 模型名稱
 
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY 未設定！")
 
client = Groq(api_key=GROQ_API_KEY)
 
=======

# LLM 客戶端設定
# 建議使用環境變數來設定模型名稱，增加靈活性
FUNC_MODEL = os.getenv("FUNC_MODEL", "nous-hermes-2-pro")
CHAT_MODEL = os.getenv("CHAT_MODEL", "dolphin-mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

# Prometheus 設定
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

<<<<<<< HEAD
>>>>>>> my-feature
# ------------------------ Prometheus 小工具 ------------------------ #
def _prom_query(q: str) -> dict:
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": q}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}
 
def _prom_query_range(q: str, start: str, end: str, step: str = "1h") -> dict:
    try:
=======
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
>>>>>>> my-feature
        params = {
            "query": promql,
            "start": start.isoformat() + "Z",
            "end": end.isoformat() + "Z",
            "step": step,
        }
<<<<<<< HEAD
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}
def _prom_get(path: str, params: dict) -> dict:
    """
    Prometheus 的 HTTP Get 請求
    """
    try:
        resp = requests.get(f"{PROMETHEUS_URL}{path}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
 
def _duration_to_seconds(duration: str) -> int:
    """
    時間字串轉秒數
    """
    m = re.fullmatch(r"(\d+)([mhdwy])", duration)
    if not m:
        raise ValueError("duration must match e.g. 30m, 4h, 2d, 1w, 1y")
    value, unit = int(m.group(1)), m.group(2)
    factor = {"m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}[unit]
    return value * factor
<<<<<<< HEAD
 
=======

###########################test
def is_smalltalk(msg: str) -> bool:
    # 可以自己加強這個判斷
    return msg.lower().strip() in ["hello", "hi", "who are you", "早安", "你好"]
##############################

>>>>>>> my-feature
def get_pod_cpu_usage(namespace: str, pod: str, range_str: str = "[1h]", **kwargs) -> str:
    q = f'container_cpu_usage_seconds_total{{namespace="{namespace}",pod="{pod}"}}{range_str}'
    return json.dumps(_prom_query(q))
 
def get_pod_memory_usage(namespace: str, pod: str, range_str: str = "[1h]", **kwargs) -> str:
    q = f'container_memory_usage_bytes{{namespace="{namespace}",pod="{pod}"}}{range_str}'
    return json.dumps(_prom_query(q))
 
def get_top_cpu_pods(namespace: str, k: int = 3, **kwargs) -> str:
    q = f'topk({k}, sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[5m])))'
    print(f"[DEBUG] get_top_cpu_pods called with namespace={namespace}, k={k}, extra={kwargs}")
    return json.dumps(_prom_query(q))
 
def get_top_memory_pods(namespace: str, k: int = 3, **kwargs) -> str:
    q = f'topk({k}, sum by(pod)(container_memory_usage_bytes{{namespace="{namespace}"}}))'
    print(f"[DEBUG] get_top_cpu_pods called with namespace={namespace}, k={k}, extra={kwargs}")
    return json.dumps(_prom_query(q))
 
def get_pod_resource_usage_over_time(namespace: str, pod: str, days: int = 7, **kwargs) -> str:
    """Get CPU and memory usage for a pod over a period of days"""
    end = "now()"
    start = f"now()-{days}d"
   
    cpu_query = f'container_cpu_usage_seconds_total{{namespace="{namespace}",pod="{pod}"}}'
    mem_query = f'container_memory_usage_bytes{{namespace="{namespace}",pod="{pod}"}}'
   
    cpu_data = _prom_query_range(cpu_query, start, end, "1h")
    mem_data = _prom_query_range(mem_query, start, end, "1h")
   
    return json.dumps({
        "cpu_usage": cpu_data,
        "memory_usage": mem_data
    })
 
def get_namespace_resource_usage_over_time(namespace: str, days: int = 7, **kwargs) -> str:
    """Get resource usage for all pods in a namespace over time"""
    end = "now()"
    start = f"now()-{days}d"
   
    cpu_query = f'sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[5m]))'
    mem_query = f'sum by(pod)(container_memory_usage_bytes{{namespace="{namespace}"}})'
   
    cpu_data = _prom_query_range(cpu_query, start, end, "1h")
    mem_data = _prom_query_range(mem_query, start, end, "1h")
   
    return json.dumps({
        "cpu_usage": cpu_data,
        "memory_usage": mem_data
    })
 
def query_resource(level: str, target: str, metric: str, duration: str, namespace: str | None = None) -> str:
    # Validate duration format
    match = re.match(r"^(\d+)(m|h|d|mo)$", duration)
    if not match:
        return json.dumps({"error": "Invalid duration format. Use 30m, 1h, 2d, 1mo"})
   
    value, unit = match.groups()
    value = int(value)
    if unit == "mo" and value > 10:
        return json.dumps({"error": "Max supported duration is 10mo"})
 
    # Check for missing namespace in pod-level queries
    if level == "pod":
        if not namespace and "/" not in target:
            return json.dumps({"error": "Namespace is required for pod-level queries. Please specify namespace or use 'namespace/pod' format for target."})
        elif namespace is None and "/" in target:
            namespace, target = target.split("/", 1)
 
    # Rest of the time calculation and query logic remains unchanged
    now = datetime.utcnow()
    delta = {
        "mo": timedelta(days=30 * value),
        "d": timedelta(days=value),
        "h": timedelta(hours=value),
        "m": timedelta(minutes=value),
    }.get(unit)
 
    if not delta:
        return json.dumps({"error": "Unsupported time unit"})
 
    start = int((now - delta).timestamp())
    end = int(now.timestamp())
    step = "1h"
 
    metric = metric.lower().replace(".usage", "")
    if metric not in ("cpu", "memory"):
        return json.dumps({"error": "Metric must be 'cpu' or 'memory'"})
 
    if level == "pod":
        query = f'container_{metric}_usage_seconds_total{{namespace="{namespace}",pod="{target}"}}'
    elif level == "node":
        query = f'node_{metric}_usage_seconds_total{{node="{target}"}}'
    elif level == "namespace":
        query = f'container_{metric}_usage_seconds_total{{namespace="{target}"}}'
    else:
        return json.dumps({"error": f"Unsupported level: {level}"})
 
    result = _prom_query_range(query, start, end, step)
    return json.dumps(result)
 
def top_k_pods(namespace: str, metric: str, k: int = 3, duration: str = "5m") -> str:
=======
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
>>>>>>> my-feature
    """
    【整合工具】取得指定 Pod 的 CPU 或 Memory 使用率。
    - 合併了原本的 get_pod_cpu_usage 和 get_pod_memory_usage。
    """
<<<<<<< HEAD
    if metric not in ("cpu", "memory"):
        return json.dumps({"error": "Invalid metric. Use 'cpu' or 'memory'"})
 
    if metric == "cpu":
        query = f'topk({k}, sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[{duration}])))'
    else:
        query = f'topk({k}, sum by(pod)(container_memory_usage_bytes{{namespace="{namespace}"}}))'
 
    return json.dumps(_prom_query(query))
   
def generate_csv_link(namespace: str, pod: str, range: str = "[1h]") -> str:
    if not range.startswith("["):
        bracketed = f"[{range}]"
    else:
        bracketed = range
    url = f"/api/export_csv?namespace={namespace}&pod={pod}&range={bracketed}"
    return json.dumps({
        "message": f"You can download the CSV from: [Download CSV]({url})"
    })
# ------------------------ 預測 CPU 使用率 ------------------------ #
 
 
def fetch_pod_cpu_metrics(namespace: str, pod: str, duration: str = "1h", step: str = "5m") -> pd.DataFrame:
=======
    if metric not in ["cpu", "memory"]:
        return json.dumps({"error": "Invalid metric specified. Must be 'cpu' or 'memory'."})

    if metric == "cpu":
        query = f'rate(container_cpu_usage_seconds_total{{namespace="{input_data.namespace}", pod="{input_data.pod}"}}[{input_data.duration}])'
    else: # memory
        query = f'container_memory_usage_bytes{{namespace="{input_data.namespace}", pod="{input_data.pod}"}}'
    
    result = prometheus_service.query(query)
    return json.dumps(result, indent=2)

def get_top_k_pods(input_data: TopKInput, metric: str) -> str:
>>>>>>> my-feature
    """
    【整合工具】取得指定 Namespace 中，CPU 或 Memory 使用率最高的前 K 個 Pod。
    - 合併了 get_top_cpu_pods 和 get_top_memory_pods。
    """
<<<<<<< HEAD
    end = int(time.time())
    start = end - _duration_to_seconds(duration)
    query = f'rate(container_cpu_usage_seconds_total{{namespace="{namespace}", pod="{pod}"}}[{step}])'
   
    params = {
        "query": query,
        "start": start,
        "end": end,
        "step": step
    }
    result = _prom_get("/api/v1/query_range", params)
 
    if result["status"] == "error":
        print(f"[ERROR] Prometheus Query Failed: {result['error']}")
        return pd.DataFrame()
 
    if not result["data"]["result"]:
        print(f"[WARNING] No data found for namespace={namespace}, pod={pod}")
        return pd.DataFrame()
 
    # 解析 Prometheus 的資料
    values = result["data"]["result"][0]["values"]
   
    if not values:
        print(f"[WARNING] No values returned from Prometheus for pod '{pod}'")
        return pd.DataFrame()
   
    df = pd.DataFrame(values, columns=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"], unit='s')
    df["y"] = df["y"].astype(float)
 
    return df
 
 
def predict_pod_cpu_usage(namespace: str, pod: str, duration: str = "1h", step: str = "5m", future_duration: str = "1h") -> str:
=======
    if metric not in ["cpu", "memory"]:
        return json.dumps({"error": "Invalid metric specified. Must be 'cpu' or 'memory'."})

    if metric == "cpu":
        query = f'topk({input_data.k}, sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{input_data.namespace}"}}[{input_data.duration}])))'
    else: # memory
        query = f'topk({input_data.k}, sum by(pod)(container_memory_usage_bytes{{namespace="{input_data.namespace}"}}))'

    result = prometheus_service.query(query)
    return json.dumps(result, indent=2)

def predict_pod_cpu_usage(input_data: PredictionInput) -> str:
>>>>>>> my-feature
    """
    預測未來 Pod 的 CPU 使用率 (使用簡易的線性回歸)。
    """
    try:
<<<<<<< HEAD
        df = fetch_pod_cpu_metrics(namespace, pod, duration, step)
       
        # 檢查資料是否存在
        if df.empty:
            return json.dumps({"error": f"No data found for namespace '{namespace}' and pod '{pod}'"})
 
        X = np.array(range(len(df))).reshape(-1, 1)
        y = df["y"].values
 
        if len(X) < 2:  # 需要至少兩個數據點才能進行線性回歸
            return json.dumps({"error": f"Not enough data points for prediction in pod '{pod}'"})
 
        model = LinearRegression()
        model.fit(X, y)
 
        # 計算未來的時間點數量
        future_steps = _duration_to_seconds(future_duration) // _duration_to_seconds(step)
        future_X = np.array(range(len(df), len(df) + future_steps)).reshape(-1, 1)
        predictions = model.predict(future_X)
 
        future_df = pd.DataFrame({
            "timestamp": pd.date_range(start=df["ds"].iloc[-1], periods=future_steps, freq=step),
            "predicted_cpu_usage": predictions
        })
 
        return future_df.to_json(orient='records')
   
    except Exception as e:
        print(f"[ERROR] 預測失敗：{str(e)}")
        return json.dumps({"error": str(e)})
#alan52254改的動態調整
def create_hpa_for_deployment(namespace: str, deployment: str, metric: str = "cpu",
                               min_replicas: int = 1, max_replicas: int = 5,
                               target_utilization: int = 60) -> str:
    from kubernetes import client, config
    config.load_kube_config()
 
    autoscaling_v1 = client.AutoscalingV1Api()
 
    hpa = client.V1HorizontalPodAutoscaler(
        metadata=client.V1ObjectMeta(name=f"{deployment}-hpa"),
        spec=client.V1HorizontalPodAutoscalerSpec(
            scale_target_ref=client.V1CrossVersionObjectReference(
                api_version="apps/v1",
                kind="Deployment",
                name=deployment,
            ),
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            target_cpu_utilization_percentage=target_utilization if metric == "cpu" else None
        )
    )
 
    try:
        autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(namespace=namespace, body=hpa)
        return json.dumps({"status": "success", "message": f"HPA created for {deployment}"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
 
 
TOOLS_DEF: List[dict] = [  
    {
        "type": "function",
        "function": {
            "name": "get_pod_cpu_usage",
            "description": "Get CPU usage for a specific pod in a namespace",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "pod": {"type": "string"},
                    "range_str": {"type": "string", "default": "[1h]"},
                },
                "required": ["namespace", "pod"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pod_memory_usage",
            "description": "Get memory usage for a specific pod in a namespace",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "pod": {"type": "string"},
                    "range_str": {"type": "string", "default": "[1h]"},
                },
                "required": ["namespace", "pod"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_cpu_pods",
            "description": "Top‑K pods by CPU usage in a namespace",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "k": {"type": "integer", "default": 3},
                },
                "required": ["namespace"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_memory_pods",
            "description": "Top‑K pods by memory usage in a namespace",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "k": {"type": "integer", "default": 3},
                },
                "required": ["namespace"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pod_resource_usage_over_time",
            "description": "Get CPU and memory usage for a pod over the past N days",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "pod": {"type": "string"},
                    "days": {"type": "integer"}
                },
                "required": ["namespace", "pod", "days"]
            }
=======
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
>>>>>>> my-feature
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
<<<<<<< HEAD
 
=======

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

>>>>>>> my-feature
FUNC_MAP = {
    "get_pod_resource_usage": lambda **kwargs: _tool_wrapper(get_pod_resource_usage, PodResourceInput, **kwargs),
    "get_top_k_pods": lambda **kwargs: _tool_wrapper(get_top_k_pods, TopKInput, **kwargs),
    "predict_pod_cpu_usage": lambda **kwargs: _tool_wrapper(predict_pod_cpu_usage, PredictionInput, **kwargs),
}
<<<<<<< HEAD
 
SYSTEM_PROMPT = """
You are a Kubernetes observability assistant. Your job is to help users monitor, analyze, and troubleshoot cluster resources in real-time.
 
Your capabilities include:
1. Monitoring CPU, memory, disk, and network usage for specific pods or nodes.
2. Identifying top-K resource-consuming pods by CPU, memory, disk, or network.
3. Checking pod readiness and general health status within a namespace.
4. Analyzing resource usage trends over time (e.g., last 1h, 1d, 1mo).
5. Comparing usage across different namespaces or time intervals.
6. Generating CSV reports for selected pods and time ranges. The report will be returned as a downloadable link.
7. Responding in natural language while calling relevant functions to fetch live data.
8. Format multi-item results using bullet points or line breaks for readability.
 
**Tool Mapping Guidance:**
- For "show" or "get" CPU/memory usage of a pod, use `get_pod_cpu_usage` or `get_pod_memory_usage`.
- For "query" resource usage (pod, node, namespace), use `query_resource`.
- For "predict" or "forecast" CPU usage, use `predict_pod_cpu_usage`. Example: "Predict CPU usage for pod X in namespace Y for next 1h" → `predict_pod_cpu_usage(namespace="Y", pod="X", future_duration="1h")`.
- For "create HPA" or "set up autoscaling", use `create_hpa_for_deployment`. Example: "Create HPA for deployment my-app with CPU 60%" → `create_hpa_for_deployment(namespace="default", deployment="my-app", metric="cpu", min_replicas=1, max_replicas=5, target_utilization=60)`.
 
**Parameter Type Rules:**
- Ensure `min_replicas`, `max_replicas`, and `target_utilization` are integers (e.g., 1, 5, 60), not strings.
- `duration` and `future_duration` must follow formats like "30m", "1h", "2d".
 
**Error Handling:**
- If a tool call fails due to missing or invalid parameters, respond with a helpful message like: "I need the namespace and pod name to proceed. Please specify them, e.g., 'default/my-pod'."
"""
 
MAX_TOOL_SUMMARY = 400        # 單條訊息上限
MAX_TOTAL_CHARS  = 15000      # 整包 JSON 上限（約 6–7k tokens）
 
def _summarize(text: str, limit: int = MAX_TOOL_SUMMARY) -> str:
    """長度>limit 就裁切，避免塞爆 context"""
    return text if len(text) <= limit else text[:limit] + f"...(truncated {len(text)-limit} chars)"
 
def _shrink_until_ok(msgs: list[dict]) -> list[dict]:
    """
    1. 先把 role=='tool' 的 content 做摘要
    2. 若總長度仍超過 MAX_TOTAL_CHARS，往前砍最舊的一句
    """
    for m in msgs:
        if m["role"] == "tool":
            m["content"] = _summarize(m["content"])
    while len(json.dumps(msgs)) > MAX_TOTAL_CHARS:
        for i, m in enumerate(msgs):
            if m["role"] != "system" and i < len(msgs) - 5:
                del msgs[i]
                break
    return msgs
 
MAX_HISTORY_LEN = 20   # 依需要調
 
def prune_history(hist: list[dict]) -> list[dict]:
    """只保留最近 MAX_HISTORY_LEN 句對話"""
    if len(hist) > MAX_HISTORY_LEN:
        return hist[-MAX_HISTORY_LEN:]
    return hist
 
def infer_parameters_from_history(history: list[dict], user_message: str) -> dict:
    inferred_params = {"namespace": None, "pod": None, "range": "[1h]"}  # Default range
 
    if "download csv" in user_message.lower() or "export csv" in user_message.lower():
        for msg in reversed(history):
            content = msg.get("content", "").lower()
            if not inferred_params["namespace"] and "namespace" in content:
                match = re.search(r'namespace\s*[:=]\s*([a-zA-Z0-9_-]+)', content)
                if match:
                    inferred_params["namespace"] = match.group(1)
            if not inferred_params["pod"] and "pod" in content:
                match = re.search(r'pod\s*[:=]\s*([a-zA-Z0-9_-]+)', content)
                if match:
                    inferred_params["pod"] = match.group(1)
            if "range" in content:
                match = re.search(r'range\s*[:=]\s*\[?(\d+[mhd]|1mo)\]?', content)
                if match:
                    inferred_params["range"] = f"[{match.group(1)}]"
 
    return inferred_params
<<<<<<< HEAD
 
def chat_with_llm(user_message: str, history: list | None = None) -> Dict[str, Any]:
=======

def chat_with_llm(user_message: str, history: list | None = None) -> dict:
>>>>>>> my-feature
    history = prune_history(history or [])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message},
    ]
<<<<<<< HEAD
 
    # Handle CSV requests (unchanged logic)
=======
    # --- Step 1. CSV download shortcut (原本邏輯保留) ---
>>>>>>> my-feature
    csv_keywords = ["download csv", "export csv"]
    if any(keyword in user_message.lower() for keyword in csv_keywords):
        inferred_params = infer_parameters_from_history(history, user_message)
        if not inferred_params["namespace"] or not inferred_params["pod"]:
            missing = []
            if not inferred_params["namespace"]:
                missing.append("namespace")
            if not inferred_params["pod"]:
                missing.append("pod")
            return {
                "assistant": f"Please specify the {', '.join(missing)} for the CSV download.",
                "history": history
            }
        result = generate_csv_link(
            namespace=inferred_params["namespace"],
            pod=inferred_params["pod"],
            range=inferred_params["range"]
        )
        messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": "inferred_csv_call"
        })
        resp = client_chat.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=800,
        )
        final_msg = resp.choices[0].message
        messages.append({"role": "assistant", "content": final_msg.content})
        front_history = [m for m in messages if m["role"] != "system"]
        return {"assistant": final_msg.content, "history": front_history}
<<<<<<< HEAD
 
    # First LLM call to decide tool usage
    from groq import BadRequestError
=======

    # --- Step 2. 判斷是否純聊天 (可選) ---
    if is_smalltalk(user_message):
        resp = client_chat.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=400,
        )
        return {"assistant": resp.choices[0].message.content, "history": history}

    # --- Step 3. function calling：Hermes 判斷是否需要 tool-call ---
    from openai import BadRequestError
>>>>>>> my-feature
    try:
        resp_fc = client_fc.chat.completions.create(
            model=FUNC_MODEL,
            messages=messages,
            tools=TOOLS_DEF,
            tool_choice="auto",
            max_tokens=800,
        )
    except BadRequestError as err:
        human = (
            "⚠️ I couldn't execute that request (invalid tool arguments). "
            "Try specifying: namespace, metric (cpu/memory), duration like 30m/2h/1d."
        )
        return {"assistant": human, "history": history or []}
<<<<<<< HEAD
   
    assistant_msg = resp.choices[0].message
=======

    assistant_msg = resp_fc.choices[0].message
>>>>>>> my-feature
    tool_calls = getattr(assistant_msg, "tool_calls", None) or []
    messages.append({
        "role": "assistant",
        "content": assistant_msg.content,
        "tool_calls": tool_calls
    })
<<<<<<< HEAD
 
    # Execute tool calls with enhanced error handling
=======

    # --- Step 4. 執行 tool-calls 並 append 回 messages ---
>>>>>>> my-feature
    for tool_call in tool_calls:
        fn = FUNC_MAP.get(tool_call.function.name)
        if not fn:
            continue
        try:
            args = json.loads(tool_call.function.arguments)
            result = fn(**args)
        except Exception as e:
            assistant_response = f"Error executing {tool_call.function.name}: {str(e)}"
            return {"assistant": assistant_response, "history": history}
        messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id
        })
<<<<<<< HEAD
 
    # Second LLM call for final response
    if tool_calls:
        resp2 = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_DEF,
            tool_choice="none",
            max_tokens=800,
        )
        final_msg = resp2.choices[0].message
        messages.append({"role": "assistant", "content": final_msg.content})
 
    front_history = [m for m in messages if m["role"] != "system"]
    last_msg = messages[-1]
    assistant_content = last_msg.get("content") or "[Error: No response generated from the assistant.]"
   
    return {"assistant": assistant_content, "history": front_history}
=======

    # --- Step 5. 用更會聊天的模型 (如 llama3.1/gpt-4o) 輸出最終回答 ---
    resp2 = client_chat.chat.completions.create(
        model=CHAT_MODEL,
=======


# --- 5. 對話核心邏輯 (Core Chat Logic) ---

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
    while len(json.dumps(history)) > MAX_TOTAL_JSON_LEN and len(history) > 5:
        history.pop(1)
        
    # 3. 確保歷史訊息不會超過最大輪數
    if len(history) > MAX_HISTORY_LEN * 2:
        history = history[-MAX_HISTORY_LEN*2:]

    return history

def chat_with_llm(user_message: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """主對話流程函式 - 已整合 RAG"""
    history = history or []

    # --- 關鍵修改 1: 進行 RAG 檢索 ---
    print("步驟 1: 從 RAG 系統檢索上下文...")
    rag_context = rag_system.retrieve(user_message, top_k=3)
    print(f"檢索到的上下文:\n---\n{rag_context}\n---")

    # --- 關鍵修改 2: 建立包含 RAG 上下文的系統提示 ---
    system_prompt = f"""
    You are a highly intelligent Kubernetes (k8s) monitoring assistant. Your primary goal is to help users by calling available tools to fetch data from Prometheus.

    **Interaction Rules:**
    1.  **Analyze the user's query carefully.**
    2.  **Refer to the 'Context from Knowledge Base' below.** This context contains critical rules and successful examples from the past. Use it to make the best decision for tool calling.
    3.  **Clarify Ambiguity:** If necessary information is missing (e.g., 'check cpu usage' without a pod), you MUST ask for clarification.
    4.  **Summarize, Don't Just Dump:** When you get data from a tool, summarize the key information clearly.
    
    [Context from Knowledge Base]
    {rag_context}
    """

    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_message}]
    
    if _is_smalltalk(user_message):
        response = client_chat.chat.completions.create(model=CHAT_MODEL, messages=messages)
        assistant_response = response.choices[0].message.content
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_response})
        return {"assistant": assistant_response, "history": history}

    # --- 主要 Function Calling 流程 ---
    print("步驟 2: 呼叫 Function-Calling LLM (帶有 RAG 上下文)...")
    response_fc = client_fc.chat.completions.create(
        model=FUNC_MODEL,
>>>>>>> my-feature
        messages=messages,
        tools=TOOLS_DEF,
        tool_choice="auto"
    )
    response_message = response_fc.choices[0].message
    tool_calls = response_message.tool_calls

<<<<<<< HEAD
    front_history = [m for m in messages if m["role"] != "system"]
    assistant_content = final_msg.content or "[Error: No response generated from the assistant.]"
    return {"assistant": assistant_content, "history": front_history}
>>>>>>> my-feature
=======
    if not tool_calls:
        print("步驟 3: LLM 判斷無需呼叫工具，直接回應。")
        response = client_chat.chat.completions.create(model=CHAT_MODEL, messages=messages)
        assistant_response = response.choices[0].message.content
    else:
        print("步驟 3: LLM 決定呼叫工具。")
        messages.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = FUNC_MAP.get(function_name)
            
            if function_to_call:
                function_args = json.loads(tool_call.function.arguments)
                print(f"執行工具: {function_name}，參數: {function_args}")
                function_response_str = function_to_call(**function_args)
                
                # --- 關鍵修改 3: 記錄成功的 Function Call ---
                # 簡單假設只要沒報錯就是成功
                try:
                    response_json = json.loads(function_response_str)
                    if "error" not in response_json:
                        log_successful_call(user_message, f"{function_name}(**{function_args})")
                        print("成功記錄 Function Call。")
                except (json.JSONDecodeError, TypeError):
                    # 如果回傳的不是合法的 JSON，也假設成功並記錄
                    log_successful_call(user_message, f"{function_name}(**{function_args})")
                    print("成功記錄 Function Call (非JSON回傳)。")
                # --- 結束修改 ---

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response_str,
                })
        
        print("步驟 4: 呼叫聊天模型以整理工具回傳結果...")
        final_response = client_chat.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
        )
        assistant_response = final_response.choices[0].message.content

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_response})
    
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
        
        conversation_history = result['history']
>>>>>>> my-feature
