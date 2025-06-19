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
from sklearn.linear_model import LinearRegression # 確保匯入 LinearRegression
import logging

# --- 關鍵修改：匯入 RAG 系統與日誌處理器 ---
from .rag_utils_dynamic import rag_system
from .log_handler import log_successful_call
# --- 結束修改 ---

# --- 1. 設定 (Configuration) ---
# 將所有可設定的變數集中管理

load_dotenv()

# LLM 客戶端設定
# 建議使用環境變數來設定模型名稱，增加靈活性
FUNC_MODEL = os.getenv("FUNC_MODEL", "nous-hermes-2-pro")
CHAT_MODEL = os.getenv("CHAT_MODEL", "dolphin-mistral")
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

logger = logging.getLogger(__name__)

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

# Original Tool Functions (updated to use PrometheusService)

def get_pod_cpu_usage(namespace: str, pod: str, range_str: str = "[1h]", **kwargs) -> str:
    """Get CPU usage for a specific pod in a namespace"""
    # Note: range_str format like [1h] is not directly compatible with query()
    # This function might need adjustment depending on how it's intended to be used
    # For now, assuming it's meant for instant query with a range selector
    q = f'container_cpu_usage_seconds_total{{namespace="{namespace}",pod="{pod}"}}{range_str}'
    return json.dumps(prometheus_service.query(q))

def get_pod_memory_usage(namespace: str, pod: str, range_str: str = "[1h]", **kwargs) -> str:
    """Get memory usage for a specific pod in a namespace"""
    # Note: range_str format like [1h] is not directly compatible with query()
    q = f'container_memory_usage_bytes{{namespace="{namespace}",pod="{pod}"}}{range_str}'
    return json.dumps(prometheus_service.query(q))

def get_top_cpu_pods(namespace: str, k: int = 3, **kwargs) -> str:
    """Top‑K pods by CPU usage in a namespace"""
    q = f'topk({k}, sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[5m])))'
    logger.debug(f"Executing query: {q}")
    return json.dumps(prometheus_service.query(q))

def get_top_memory_pods(namespace: str, k: int = 3, **kwargs) -> str:
    """Top‑K pods by memory usage in a namespace"""
    q = f'topk({k}, sum by(pod)(container_memory_usage_bytes{{namespace="{namespace}"}}))'
    logger.debug(f"Executing query: {q}")
    return json.dumps(prometheus_service.query(q))

def get_pod_resource_usage_over_time(namespace: str, pod: str, days: int = 7, **kwargs) -> str:
    """Get CPU and memory usage for a pod over a period of days"""
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    cpu_query = f'rate(container_cpu_usage_seconds_total{{namespace="{namespace}",pod="{pod}"}}[5m])' # Using rate for CPU
    mem_query = f'container_memory_usage_bytes{{namespace="{namespace}",pod="{pod}"}}'

    # Assuming a step of 1 hour for daily range
    cpu_data = prometheus_service.query_range(cpu_query, start, end, "1h")
    mem_data = prometheus_service.query_range(mem_query, start, end, "1h")

    return json.dumps({
        "cpu_usage": cpu_data,
        "memory_usage": mem_data
    })

def get_namespace_resource_usage_over_time(namespace: str, days: int = 7, **kwargs) -> str:
    """Get resource usage for all pods in a namespace over time"""
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    cpu_query = f'sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[5m]))'
    mem_query = f'sum by(pod)(container_memory_usage_bytes{{namespace="{namespace}"}})'

    # Assuming a step of 1 hour for daily range
    cpu_data = prometheus_service.query_range(cpu_query, start, end, "1h")
    mem_data = prometheus_service.query_range(mem_query, start, end, "1h")

    return json.dumps({
        "cpu_usage": cpu_data,
        "memory_usage": mem_data
    })

def query_resource(level: str, target: str, metric: str = "cpu", duration: str = "1h", namespace: str | None = None) -> str:
    """Query CPU or memory usage of a pod, node, or namespace."""
    try:
        # Validate duration format and convert to seconds
        duration_seconds = _duration_to_seconds(duration)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=duration_seconds)
        step = "1h" # Default step, could be made dynamic

        # Check for missing namespace in pod-level queries
        if level == "pod":
            if not namespace and "/" not in target:
                return json.dumps({"error": "Namespace is required for pod-level queries. Please specify namespace or use 'namespace/pod' format for target."})
            elif namespace is None and "/" in target:
                namespace, target = target.split("/", 1)

        metric = metric.lower().replace(".usage", "")
        if metric not in ("cpu", "memory"):
            return json.dumps({"error": "Metric must be 'cpu' or 'memory'"})

        if level == "pod":
            query = f'rate(container_{metric}_usage_seconds_total{{namespace="{namespace}",pod="{target}"}}[{duration}])' if metric == "cpu" else f'container_memory_usage_bytes{{namespace="{namespace}",pod="{target}"}}'
        elif level == "node":
             query = f'node_{metric}_usage_seconds_total{{node="{target}"}}' # Note: node metrics might have different names
        elif level == "namespace":
            query = f'sum by(pod)(rate(container_{metric}_usage_seconds_total{{namespace="{target}"}}[{duration}]))' if metric == "cpu" else f'sum by(pod)(container_memory_usage_bytes{{namespace="{target}"}})'
        else:
            return json.dumps({"error": f"Unsupported level: {level}"})

        # Use query_range for historical data, query for instant value (if duration is very short or 0)
        # For simplicity, using query_range for any duration > 0
        if duration_seconds > 0:
             result = prometheus_service.query_range(query, start_time, end_time, step)
        else:
             result = prometheus_service.query(query) # Instant query for duration 0 or very small

        return json.dumps(result)

    except ValueError as ve:
        return json.dumps({"error": f"Parameter validation error: {str(ve)}"})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred during query_resource: {str(e)}"})


def top_k_pods(namespace: str, metric: str, k: int = 3, duration: str = "5m") -> str:
    """Return top‑K pods by CPU/Memory in a namespace"""
    if metric not in ("cpu", "memory"):
        return json.dumps({"error": "Invalid metric. Use 'cpu' or 'memory'"})

    if metric == "cpu":
        query = f'topk({k}, sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[{duration}])))'
    else:
        query = f'topk({k}, sum by(pod)(container_memory_usage_bytes{{namespace="{namespace}"}}))'

    return json.dumps(prometheus_service.query(query))

def fetch_pod_cpu_metrics(namespace: str, pod: str, duration: str = "1h", step: str = "5m") -> pd.DataFrame:
    """
    從 Prometheus 取得 Pod 的 CPU 使用率 (用於預測)
    """
    try:
        end = datetime.utcnow()
        start = end - timedelta(seconds=_duration_to_seconds(duration))

        # Use rate for CPU usage over time
        query = f'rate(container_cpu_usage_seconds_total{{namespace="{namespace}", pod="{pod}"}}[{step}])'

        result = prometheus_service.query_range(query, start, end, step)

        if result.get("status") != "success" or not result["data"]["result"]:
            logger.warning(f"No historical data found for pod '{pod}' in namespace '{namespace}'.")
            return pd.DataFrame()

        values = result["data"]["result"][0]["values"]

        if not values:
            logger.warning(f"No values returned from Prometheus for pod '{pod}'")
            return pd.DataFrame()

        df = pd.DataFrame(values, columns=["ds", "y"])
        df["ds"] = pd.to_datetime(df["ds"], unit='s')
        df["y"] = df["y"].astype(float)

        return df
    except Exception as e:
        logger.error(f"Error fetching pod CPU metrics: {str(e)}")
        return pd.DataFrame()


def predict_pod_cpu_usage(namespace: str, pod: str, duration: str = "1h", step: str = "5m", future_duration: str = "1h") -> str:
    """
    預測未來 Pod 的 CPU 使用率，基於歷史數據 (使用簡易的線性回歸)。
    - `duration`: 歷史數據的時間範圍。
    - `step`: 歷史數據的採樣間隔。
    - `future_duration`: 預測的時間範圍。
    """
    try:
        df = fetch_pod_cpu_metrics(namespace, pod, duration, step)

        # 檢查資料是否存在
        if df.empty or len(df) < 2:
            return json.dumps({"error": f"Not enough historical data points ({len(df)}) to make a prediction for pod '{pod}' in namespace '{namespace}'."})

        # 建立 DataFrame 並訓練模型
        # Use numerical index as feature for linear regression
        X = np.array(range(len(df))).reshape(-1, 1)
        y = df["y"].values

        model = LinearRegression()
        model.fit(X, y)

        # 計算未來的時間點數量
        try:
            step_seconds = _duration_to_seconds(step)
            future_duration_seconds = _duration_to_seconds(future_duration)
            if step_seconds <= 0:
                 return json.dumps({"error": f"Invalid step duration: {step}"})
            future_steps = future_duration_seconds // step_seconds
            if future_steps <= 0:
                 return json.dumps({"error": f"Future duration '{future_duration}' is too short relative to step '{step}'."})

        except ValueError as ve:
             return json.dumps({"error": f"Invalid duration format for prediction: {str(ve)}"})


        # 產生未來時間點並預測
        future_X = np.array(range(len(df), len(df) + future_steps)).reshape(-1, 1)
        predictions = model.predict(future_X)

        # Generate future timestamps based on the last historical timestamp and step
        last_timestamp = df["ds"].iloc[-1]
        future_timestamps = [last_timestamp + timedelta(seconds=(i + 1) * step_seconds) for i in range(future_steps)]


        # 格式化輸出
        output = {
            "pod": pod,
            "namespace": namespace,
            "prediction_window_start": future_timestamps[0].isoformat(),
            "prediction_window_end": future_timestamps[-1].isoformat(),
            "predictions": [
                {"timestamp": ts.isoformat(), "predicted_cpu_usage": round(float(p), 4)}
                for ts, p in zip(future_timestamps, predictions)
            ]
        }
        return json.dumps(output, indent=2)

    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {str(e)}")
        return json.dumps({"error": f"An unexpected error occurred during prediction: {str(e)}"})


def generate_csv_link(namespace: str, pod: str, range: str = "[1h]") -> str:
    """Generate a CSV download link for a specific pod's CPU/Memory usage."""
    # This function likely needs to interact with a backend endpoint that generates the CSV.
    # Assuming the endpoint is /api/export_csv
    if not range.startswith("["):
        bracketed = f"[{range}]"
    else:
        bracketed = range
    url = f"/api/export_csv?namespace={namespace}&pod={pod}&range={bracketed}"
    # In a real application, you might generate a temporary link or token
    return json.dumps({
        "message": f"You can download the CSV for pod '{pod}' in namespace '{namespace}' for range '{range}' from: [Download CSV]({url})"
    })

def create_hpa_for_deployment(namespace: str, deployment: str, metric: str = "cpu",
                               min_replicas: int = 1, max_replicas: int = 5,
                               target_utilization: int = 60) -> str:
    """Create a Horizontal Pod Autoscaler (HPA) for a deployment."""
    # This function requires Kubernetes client library and configuration
    try:
        from kubernetes import client, config
        # Load Kubernetes configuration (e.g., from ~/.kube/config)
        config.load_kube_config()

        autoscaling_v1 = client.AutoscalingV1Api()

        # Define the HPA object
        hpa = client.V1HorizontalPodAutoscaler(
            metadata=client.V1ObjectMeta(name=f"{deployment}-hpa", namespace=namespace),
            spec=client.V1HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V1CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=deployment,
                ),
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                target_cpu_utilization_percentage=target_utilization if metric == "cpu" else None
                # For memory, you would use metrics field in autoscaling/v2
                # This example uses autoscaling/v1 which only supports CPU utilization
            )
        )

        # Create the HPA in the specified namespace
        autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(namespace=namespace, body=hpa)

        return json.dumps({"status": "success", "message": f"HPA '{deployment}-hpa' created successfully in namespace '{namespace}'."})

    except ImportError:
         return json.dumps({"status": "error", "message": "Kubernetes client library not installed. Please install it (e.g., `pip install kubernetes`)."})
    except client.ApiException as e:
        return json.dumps({"status": "error", "message": f"Kubernetes API error: {e.status} - {e.reason}"})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"An unexpected error occurred while creating HPA: {str(e)}"})


# --- 4. 工具定義 (Tool Definitions for LLM) ---
# 這是給 LLM 看的 "API 文件"，描述了每個工具的功能和參數

TOOLS_DEF = [
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
                    "days": {"type": "integer", "default": 7}
                },
                "required": ["namespace", "pod"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_namespace_resource_usage_over_time",
            "description": "Get resource usage for all pods in a namespace over time",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "days": {"type": "integer", "default": 7},
                },
                "required": ["namespace"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_resource",
            "description": "Query CPU or memory usage of a pod, node, or namespace. Most fields are optional.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["pod", "node", "namespace"],
                        "description": "Query level. Defaults to 'pod'."
                    },
                    "target": {
                        "type": "string",
                        "description": "The resource name (e.g., pod name, node name)."
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["cpu", "memory"],
                        "description": "Metric to query. Defaults to 'cpu'."
                    },
                    "duration": {
                        "type": "string",
                        "description": "Time range like 1h, 2d, or natural language like 'past 2 hours'."
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Only required for pod-level queries. Can also be inferred from target='namespace/pod'."
                    }
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "top_k_pods",
            "description": "Return top‑K pods by CPU/Memory in a namespace",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Kubernetes namespace"},
                    "metric": {"type": "string", "enum": ["cpu", "memory"]},
                    "k": {"type": "integer", "default": 3, "minimum": 1, "description": "How many pods to return"},
                    "duration": {"type": "string", "default": "5m", "pattern": "^(\\d+)(m|h|d|mo)$", "description": "Look‑back window like 30m, 2h, 1d"}
                },
                "required": ["namespace", "metric"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_csv_link",
            "description": "Generate a CSV download link for a specific pod's CPU/Memory usage. Use context from conversation history to infer namespace, pod, or range if not specified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Kubernetes namespace"},
                    "pod": {"type": "string", "description": "Pod name"},
                    "range": {"type": "string", "default": "[1h]", "description": "Time range like [1h], [2d]"}
                },
                "required": ["namespace", "pod"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_pod_cpu_usage",
            "description": "Predict future CPU usage for a specific pod in a given namespace based on historical data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Kubernetes namespace of the target pod (e.g., 'default')"
                    },
                    "pod": {
                        "type": "string",
                        "description": "Name of the pod to predict CPU usage for"
                    },
                    "duration": {
                        "type": "string",
                        "default": "1h",
                        "description": "Historical data range to train the model, e.g., '1h', '2h', '1d'"
                    },
                    "step": {
                        "type": "string",
                        "default": "5m",
                        "description": "Sampling interval (data granularity), e.g., '5m', '1m'"
                    },
                    "future_duration": {
                        "type": "string",
                        "default": "1h",
                        "description": "Duration of future CPU usage prediction, e.g., '1h', '2h'"
                    }
                },
                "required": ["namespace", "pod"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_hpa_for_deployment",
            "description": "Create a Horizontal Pod Autoscaler (HPA) for a deployment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "deployment": {"type": "string"},
                    "metric": {"type": "string", "enum": ["cpu", "memory"], "default": "cpu"},
                    "min_replicas": {"type": "integer", "default": 1},
                    "max_replicas": {"type": "integer", "default": 5},
                    "target_utilization": {"type": "integer", "default": 60}
                },
                "required": ["namespace", "deployment"]
            }
        }
    }
]

# 將工具名稱映射到實際的 Python 函式
# 注意：這裡我們需要一個 wrapper 來處理 Pydantic 模型 (Optional, can map directly if no Pydantic)
# Re-creating FUNC_MAP with all original functions, using wrapper where appropriate
FUNC_MAP = {
    "get_pod_cpu_usage": get_pod_cpu_usage,
    "get_pod_memory_usage": get_pod_memory_usage,
    "get_top_cpu_pods": get_top_cpu_pods,
    "get_top_memory_pods": get_top_memory_pods,
    "get_pod_resource_usage_over_time": get_pod_resource_usage_over_time,
    "get_namespace_resource_usage_over_time": get_namespace_resource_usage_over_time,
    "query_resource": query_resource,
    "top_k_pods": top_k_pods,
    "generate_csv_link": generate_csv_link,
    "predict_pod_cpu_usage": predict_pod_cpu_usage, # Using the original predict function
    "create_hpa_for_deployment": create_hpa_for_deployment,
}

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

def infer_parameters_from_history(history: list[dict], user_message: str) -> dict:
    """Attempt to infer parameters like namespace, pod, range from chat history for CSV download."""
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
                # Look for range formats like [1h], [2d], 1h, 2d, 1mo
                match = re.search(r'range\s*[:=]\s*\[?(\d+[mhd]|1mo)\]?', content)
                if match:
                    inferred_params["range"] = f"[{match.group(1)}]" # Ensure bracketed format

    return inferred_params


async def chat_with_llm(user_query: str, chat_history: list = None):
    """
    主要對話邏輯，已整合 RAG 系統以增強 Function Calling。
    """
    if chat_history is None:
        chat_history = []

    # --- 關鍵修改 1: RAG 檢索 ---
    print("Step 1: Retrieving context from RAG system...")
    rag_context = rag_system.retrieve(user_query, top_k=3)
    print(f"Retrieved context:\n---\n{rag_context}\n---")
    # --- 結束修改 ---

    # --- 關鍵修改 2: 建立包含 RAG 上下文的系統提示 (System Prompt) ---
    system_prompt = f"""
    You are a professional Kubernetes monitoring assistant. Your task is to help users query monitoring data by calling the appropriate functions.

    Please follow these rules:
    1. First, analyze the user's query.
    2. Refer to the provided "Context from Knowledge Base" to understand successful past examples and rules. This context is highly valuable for making the correct tool call.
    3. Based on the query and the context, decide if a tool call is necessary.
    4. If so, select the correct tool and parameters from the available tools and respond with a tool_calls JSON object.

    [Context from Knowledge Base]
    {rag_context}
    """
    # --- 結束修改 ---

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_query})

    # Handle small talk directly
    if _is_smalltalk(user_query):
        response = client_chat.chat.completions.create(model=CHAT_MODEL, messages=messages)
        assistant_response = response.choices[0].message.content
        # Update history and return
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": assistant_response})
        final_history = _prune_history(chat_history)
        return {"assistant": assistant_response, "history": final_history}

    # Handle CSV download shortcut
    csv_keywords = ["download csv", "export csv"]
    if any(keyword in user_query.lower() for keyword in csv_keywords):
        inferred_params = infer_parameters_from_history(chat_history, user_query)
        if not inferred_params["namespace"] or not inferred_params["pod"]:
            missing = []
            if not inferred_params["namespace"]:
                missing.append("namespace")
            if not inferred_params["pod"]:
                missing.append("pod")
            assistant_response = f"Please specify the {', '.join(missing)} for the CSV download."
            # Update history and return
            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": assistant_response})
            final_history = _prune_history(chat_history)
            return {"assistant": assistant_response, "history": final_history}

        # Call the generate_csv_link function directly
        # Note: This bypasses the LLM's tool calling mechanism for this specific case
        print("Step 2: Handling CSV download shortcut...")
        tool_response_str = generate_csv_link(
            namespace=inferred_params["namespace"],
            pod=inferred_params["pod"],
            range=inferred_params["range"]
        )
        print(f"CSV link generated: {tool_response_str}")

        # Add the tool response to messages for the chat model to synthesize
        messages.append({
            "role": "tool",
            "content": tool_response_str,
            "tool_call_id": "csv_shortcut_call" # Assign a dummy ID
        })

        # Use the chat model to synthesize the response
        print("Step 3: Calling Chat LLM to synthesize CSV link...")
        final_response = client_chat.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
        )
        assistant_response = final_response.choices[0].message.content

        # Update history and return
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": assistant_response})
        final_history = _prune_history(chat_history)
        return {"assistant": assistant_response, "history": final_history}


    # --- Main Function Calling Flow ---
    print("Step 2: Calling Function-Calling LLM with RAG context...")
    response = client_fc.chat.completions.create(
        model=FUNC_MODEL,
        messages=messages,
        tools=TOOLS_DEF,
        tool_choice="auto",
    )
    response_message = response.choices[0].message

    if response_message.tool_calls:
        print("Step 3: LLM decided to call a tool.")
        tool_outputs = []
        messages.append(response_message) # Add LLM's tool call response to messages

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = FUNC_MAP.get(function_name)

            if function_to_call:
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    print(f"Executing tool: {function_name} with args: {function_args}")
                    output = function_to_call(**function_args)

                    # --- 關鍵修改 3: 記錄成功的 Function Call ---
                    # Simple check for success (e.g., no 'error' key in JSON output)
                    is_successful = True
                    try:
                        output_json = json.loads(output)
                        if isinstance(output_json, dict) and "error" in output_json:
                            is_successful = False
                    except (json.JSONDecodeError, TypeError):
                        # If output is not JSON, assume success for logging purposes
                        pass

                    if is_successful:
                         log_successful_call(user_query, f"{function_name}(**{function_args})")
                         print("Successfully logged the function call.")
                         # (可選) 每記錄 N 筆日誌後，刷新一次索引
                         # if should_refresh_index():
                         #     rag_system.refresh_index()
                    else:
                         print(f"Tool execution failed, not logging: {output}")


                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        # Summarize tool output if too long
                        "content": output # Keep full output for chat model
                    })
                except Exception as e:
                    print(f"Error executing tool {function_name}: {e}")
                    # If tool execution fails, add an error message to tool_outputs
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": f"Error executing tool {function_name}: {e}",
                    })
            else:
                print(f"Unknown function requested by LLM: {function_name}")
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": f"Error: Unknown function '{function_name}'",
                })

        messages.extend(tool_outputs)

        print("Step 4: Calling Chat LLM to synthesize tool output...")
        final_response = client_chat.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
        )
        assistant_response = final_response.choices[0].message.content
    else:
        print("Step 3: LLM decided not to call a tool. Responding directly.")
        # If no tool calls, use the chat model to generate a direct response
        final_response = client_chat.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
        )
        assistant_response = final_response.choices[0].message.content


    # Update chat history
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": assistant_response})

    # Prune history before returning
    final_history = _prune_history(chat_history)

    return {"assistant": assistant_response, "history": final_history}

# --- 6. 範例使用 ---
if __name__ == "__main__":
    conversation_history = []
    print("Kubernetes Monitoring Assistant. Type 'exit' to quit.")

    # Initialize the database on startup
    from .log_handler import init_db
    init_db()

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Use asyncio.run to run the async chat_with_llm function
        import asyncio
        result = asyncio.run(chat_with_llm(user_input, conversation_history))

        print(f"Assistant: {result['assistant']}")

        conversation_history = result['history']
