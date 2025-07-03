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
from pydantic import BaseModel, Field, ValidationError
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

# Kubernetes client is required for newly added core tools
try:
    from kubernetes import client as k8s_client, config as k8s_config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False


# --- 1. Configuration ---
# Centralized management of all configurable variables

load_dotenv()

# LLM Client Settings
FUNC_MODEL = os.getenv("FUNC_MODEL", "llama3.1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

# Prometheus Settings
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

# Conversation History and Content Length Limits
MAX_HISTORY_LEN = 20
MAX_TOOL_SUMMARY_LEN = 500
MAX_TOTAL_JSON_LEN = 15000

# Initialize LLM Clients
client_fc = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
client_chat = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


# --- 2. External Services Layer ---
# Encapsulates all interactions with external APIs (Prometheus & Kubernetes)

class PrometheusService:
    """Encapsulates all interactions with the Prometheus API"""
    def __init__(self, base_url: str):
        self.base_url = base_url

    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": f"Request failed: {e.__class__.__name__}", "message": str(e)}

    def query(self, promql: str) -> Dict[str, Any]:
        return self._request("/api/v1/query", {"query": promql})

    def query_range(self, promql: str, start: datetime, end: datetime, step: str) -> Dict[str, Any]:
        params = {"query": promql, "start": start.isoformat()+"Z", "end": end.isoformat()+"Z", "step": step}
        return self._request("/api/v1/query_range", params)

class KubernetesService:
    """Encapsulates all interactions with the Kubernetes API"""
    def __init__(self):
        if not KUBERNETES_AVAILABLE:
            raise ImportError("Kubernetes client library not found. Please run 'pip install kubernetes'.")
        k8s_config.load_kube_config()
        self.core_v1 = k8s_client.CoreV1Api()

    def get_namespaces(self) -> List[str]:
        return [ns.metadata.name for ns in self.core_v1.list_namespace().items]

    def get_pods_in_namespace(self, namespace: str) -> List[str]:
        return [pod.metadata.name for pod in self.core_v1.list_namespaced_pod(namespace).items]

# Initialize service instances
prometheus_service = PrometheusService(PROMETHEUS_URL)
if KUBERNETES_AVAILABLE:
    kubernetes_service = KubernetesService()


# --- 3. Tool Functions & Data Models ---
# This is the toolbox we provide to the LLM, with Pydantic for parameter validation

# --- Pydantic Models ---
class NamespaceInput(BaseModel):
    namespace: str = Field(..., description="The Kubernetes namespace, e.g., 'default', 'kube-system'.")

# NEW: Pydantic model for Namespace-level resource queries
class NamespaceResourceInput(NamespaceInput):
    duration: str = Field("5m", description="Time duration for the query, e.g., '5m', '1h', '2 hours 30 minutes'.")

class PodResourceInput(NamespaceInput):
    pod: str = Field(..., description="The name of the pod.")
    duration: str = Field("5m", description="Time duration for the query, e.g., '5m', '1h'.")

class TopKInput(NamespaceInput):
    k: int = Field(3, description="The number of top pods to return.", ge=1)
    duration: str = Field("5m", description="Time window for rate calculation, e.g., '5m', '1h'.")

class PredictionInput(PodResourceInput):
    future_duration: str = Field("1h", description="The future time window to predict, e.g., '30m', '2h'.")
    step: str = Field("5m", description="The time interval for data points in historical data.")


# --- Helper Functions ---
# MODIFIED: More robust duration parser for PromQL
def parse_duration_to_promql_format(duration_str: str) -> str:
    """
    Parses a human-readable duration string into a PromQL-compatible format (e.g., '1h30m').
    Handles formats like '25"4 hours ago', '1 hour 30 minutes', '2d5h', '10m'.
    """
    # Clean the input string by replacing common separators with spaces
    cleaned_str = re.sub(r'["\',]', ' ', duration_str)
    
    # Regex to find all number-unit pairs
    pattern = re.compile(
        r'(\d+)\s*'
        r'(w|week|weeks|d|day|days|h|hour|hours|m|min|minute|minutes|s|sec|second|seconds)\b',
        re.IGNORECASE
    )
    
    matches = pattern.findall(cleaned_str)
    
    if not matches:
        # Fallback for simple formats like '5m', '1h' if regex fails
        if re.match(r"^\d+[smhdw]$", duration_str):
            return duration_str
        return "5m" # Default fallback

    unit_map = {
        'w': 'w', 'week': 'w', 'weeks': 'w',
        'd': 'd', 'day': 'd', 'days': 'd',
        'h': 'h', 'hour': 'h', 'hours': 'h',
        'm': 'm', 'min': 'm', 'minute': 'm', 'minutes': 'm',
        's': 's', 'sec': 's', 'second': 's', 'seconds': 's',
    }
    
    promql_duration = ""
    for value, unit_long in matches:
        unit_short = unit_map[unit_long.lower()]
        promql_duration += f"{value}{unit_short}"
        
    return promql_duration if promql_duration else "5m"


# MODIFIED: Upgraded _duration_to_seconds to handle complex formats
def _duration_to_seconds(duration_str: str) -> int:
    """
    Calculates the total seconds from a human-readable duration string.
    """
    promql_format = parse_duration_to_promql_format(duration_str)
    
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    total_seconds = 0
    
    # Regex to parse the PromQL format string like '1h30m'
    parts = re.findall(r'(\d+)([smhdw])', promql_format)
    if not parts:
        raise ValueError(f"Invalid duration format after parsing: {promql_format}")

    for value, unit in parts:
        total_seconds += int(value) * units[unit]
        
    return total_seconds

def _format_bytes(byte_val: float) -> str:
    """Converts bytes to a human-readable format (KiB, MiB, GiB)"""
    if byte_val is None: return "N/A"
    power = 1024
    n = 0
    power_labels = {0: 'B', 1: 'KiB', 2: 'MiB', 3: 'GiB', 4: 'TiB'}
    while byte_val >= power and n < len(power_labels) -1 :
        byte_val /= power
        n += 1
    return f"{byte_val:.2f} {power_labels[n]}"

# --- Tool Functions ---

def list_all_namespaces() -> str:
    """Lists all available namespaces in the Kubernetes cluster."""
    if not KUBERNETES_AVAILABLE:
        return json.dumps({"error": "Kubernetes library not available."})
    try:
        namespaces = kubernetes_service.get_namespaces()
        return json.dumps({"namespaces": namespaces})
    except Exception as e:
        return json.dumps({"error": f"Failed to list namespaces: {str(e)}"})

def list_pods_in_namespace(input_data: NamespaceInput) -> str:
    """Lists all pod names in the specified namespace."""
    if not KUBERNETES_AVAILABLE:
        return json.dumps({"error": "Kubernetes library not available."})
    try:
        pods = kubernetes_service.get_pods_in_namespace(input_data.namespace)
        if not pods:
            return json.dumps({"message": f"No pods found in namespace '{input_data.namespace}'."})
        return json.dumps({"namespace": input_data.namespace, "pods": pods})
    except Exception as e:
        return json.dumps({"error": f"Failed to list pods in namespace '{input_data.namespace}': {str(e)}"})

# MODIFIED: Now uses the robust duration parser
def get_pod_resource_usage(input_data: PodResourceInput, metric: str) -> str:
    """Retrieves the CPU or Memory usage for a specified Pod."""
    if metric not in ["cpu", "memory"]:
        return json.dumps({"error": "Invalid metric specified. Must be 'cpu' or 'memory'."})

    # Use the robust parser
    promql_duration = parse_duration_to_promql_format(input_data.duration)

    if metric == "cpu":
        query = f'avg(rate(container_cpu_usage_seconds_total{{namespace="{input_data.namespace}", pod="{input_data.pod}"}}[{promql_duration}]))'
    else: # memory
        query = f'container_memory_usage_bytes{{namespace="{input_data.namespace}", pod="{input_data.pod}"}}'
    
    result = prometheus_service.query(query)
    
    if result.get("status") == "success" and result.get("data", {}).get("result"):
        try:
            value = float(result["data"]["result"][0]["value"][1])
            if metric == "cpu":
                usage_str = f"{value * 1000:.2f}m"
                usage_val = value * 1000
            else:
                usage_str = _format_bytes(value)
                usage_val = value
            return json.dumps({"pod": input_data.pod, "metric": metric, "value": usage_val, "human_readable_value": usage_str})
        except (IndexError, KeyError, ValueError):
            return json.dumps({"error": f"No data found for pod '{input_data.pod}' in namespace '{input_data.namespace}'."})
    return json.dumps(result)

# NEW: Tool to get resource usage for an entire namespace
def get_namespace_resource_usage(input_data: NamespaceResourceInput, metric: str) -> str:
    """Retrieves the total aggregated CPU or Memory usage for an entire Namespace."""
    if metric not in ["cpu", "memory"]:
        return json.dumps({"error": "Invalid metric specified. Must be 'cpu' or 'memory'."})

    # Use the robust parser
    promql_duration = parse_duration_to_promql_format(input_data.duration)
    
    if metric == "cpu":
        # Sum the rate of CPU usage for all pods in the namespace
        query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{input_data.namespace}"}}[{promql_duration}]))'
    else: # memory
        # Sum the memory usage for all pods in the namespace
        query = f'sum(container_memory_usage_bytes{{namespace="{input_data.namespace}"}})'
        
    result = prometheus_service.query(query)
    
    if result.get("status") == "success" and result.get("data", {}).get("result"):
        try:
            value = float(result["data"]["result"][0]["value"][1])
            if metric == "cpu":
                usage_str = f"{value * 1000:.2f}m" # total millicores
                usage_val = value * 1000
            else:
                usage_str = _format_bytes(value)
                usage_val = value
            return json.dumps({"namespace": input_data.namespace, "metric": f"total_{metric}", "value": usage_val, "human_readable_value": usage_str})
        except (IndexError, KeyError, ValueError):
            return json.dumps({"error": f"No data found for namespace '{input_data.namespace}'."})
    return json.dumps(result)


# MODIFIED: Now uses the robust duration parser
def get_top_k_pods(input_data: TopKInput, metric: str) -> str:
    """Finds the top K pods with the highest CPU or Memory usage in the specified Namespace."""
    if metric not in ["cpu", "memory"]:
        return json.dumps({"error": "Invalid metric specified. Must be 'cpu' or 'memory'."})

    # Use the robust parser
    promql_duration = parse_duration_to_promql_format(input_data.duration)

    if metric == "cpu":
        query = f'topk({input_data.k}, sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{input_data.namespace}"}}[{promql_duration}])))'
    else: # memory
        query = f'topk({input_data.k}, sum by(pod)(container_memory_usage_bytes{{namespace="{input_data.namespace}"}}))'

    result = prometheus_service.query(query)

    if result.get("status") == "success" and result.get("data", {}).get("result"):
        pods_data = []
        for res in result["data"]["result"]:
            pod_name = res["metric"]["pod"]
            value = float(res["value"][1])
            if metric == "cpu":
                usage_str = f"{value * 1000:.2f}m"
            else:
                usage_str = _format_bytes(value)
            pods_data.append({"pod": pod_name, "value": value, "human_readable_value": usage_str})
        return json.dumps({"metric": metric, "top_k_pods": pods_data})
    return json.dumps(result)


# predict_pod_cpu_usage (Upgraded Version)

def predict_pod_cpu_usage(input_data: PredictionInput) -> str:
    """
    使用動態選擇的模型預測指定 Pod 的未來 CPU 使用率。
    此函式會優先嘗試使用強大的自我迴歸 (AR) 模型，但在歷史資料有限時，
    會優雅地降級到更簡單的線性迴歸模型。
    輸出結果經過豐富化，包含最後已知值和模型信賴度等情境資訊。
    """
    try:
        # --- 1. 抓取資料 ---
        history_seconds = _duration_to_seconds(input_data.duration)
        future_seconds = _duration_to_seconds(input_data.future_duration)
        step_seconds = _duration_to_seconds(input_data.step)

        end = datetime.utcnow()
        start = end - timedelta(seconds=history_seconds)
        promql_step = parse_duration_to_promql_format(input_data.step)
        
        query = f'rate(container_cpu_usage_seconds_total{{namespace="{input_data.namespace}", pod="{input_data.pod}"}}[{promql_step}])'
        result = prometheus_service.query_range(query, start, end, promql_step)

        if result.get("status") != "success" or not result.get("data", {}).get("result"):
            return json.dumps({
                "status": "failure",
                "error": "No historical data found",
                "message": f"Could not find any historical data for pod '{input_data.pod}' in namespace '{input_data.namespace}'."
            })
        
        values = result["data"]["result"][0]["values"]
        data_points_count = len(values)

        # --- 優雅降級邏輯 ---

        # [C計畫] 最終失敗：資料不足以使用任何模型
        if data_points_count < 2:
            return json.dumps({
                "status": "failure",
                "error": "Insufficient data",
                "message": f"Prediction failed. At least 2 data points are required, but only found {data_points_count}."
            })

        df = pd.DataFrame(values, columns=["timestamp", "value"])
        df["timestamp"] = pd.to_numeric(df["timestamp"])
        df["value"] = pd.to_numeric(df["value"])
        
        last_known_value = df["value"].iloc[-1]

        # [A計畫] 首選策略：資料充足時使用 AR 模型
        if data_points_count >= 10:
            try:
                cpu_usage_series = df["value"]
                lags_selector = ar_select_order(cpu_usage_series, maxlag=10, ic='aic', glob=True)
                best_lags = lags_selector.ar_lags or [1]
                
                model = AutoReg(cpu_usage_series, lags=best_lags, trend='c').fit()
                
                future_steps_count = future_seconds // step_seconds
                predictions = model.predict(start=data_points_count, end=data_points_count + future_steps_count - 1)
                predictions[predictions < 0] = 0
                
                # 更精準的趨勢判斷
                trend = "increasing" if predictions.iloc[-1] > last_known_value else "decreasing"
                
                output = {
                    "status": "success",
                    "pod": input_data.pod,
                    "last_known_cpu": f"{last_known_value * 1000:.2f}m",
                    "prediction": {
                        "window": f"for the next {input_data.future_duration}",
                        "average_cpu": f"{np.mean(predictions) * 1000:.2f}m",
                        "peak_cpu": f"{np.max(predictions) * 1000:.2f}m",
                        "trend": trend,
                    },
                    "model_details": {
                        "model_used": "AutoRegression",
                        "confidence": "High",
                        "message": f"Prediction based on a robust model using {data_points_count} data points (lags: {best_lags})."
                    }
                }
                return json.dumps(output, indent=2)
            except Exception as ar_error:
                # 如果 AR 模型意外失敗，允許降級到 Plan B
                pass

        # [B計畫] 備案策略：資料有限時使用線性迴歸
        if 2 <= data_points_count < 10:
            X, y = df[["timestamp"]], df["value"]
            model = LinearRegression().fit(X, y)
            
            future_steps_count = future_seconds // step_seconds
            last_timestamp = df["timestamp"].iloc[-1]
            future_timestamps = np.arange(1, future_steps_count + 1) * step_seconds + last_timestamp
            predictions = model.predict(pd.DataFrame(future_timestamps, columns=["timestamp"]))
            predictions[predictions < 0] = 0

            trend = "increasing" if model.coef_[0] > 0 else "decreasing"

            output = {
                "status": "success",
                "pod": input_data.pod,
                "last_known_cpu": f"{last_known_value * 1000:.2f}m",
                "prediction": {
                    "window": f"for the next {input_data.future_duration}",
                    "average_cpu": f"{np.mean(predictions) * 1000:.2f}m",
                    "peak_cpu": f"{np.max(predictions) * 1000:.2f}m",
                    "trend": trend,
                },
                "model_details": {
                    "model_used": "Linear Regression",
                    "confidence": "Low",
                    "message": f"Used a basic trend model due to limited data ({data_points_count} points). Forecast indicates a general direction, not precise values."
                }
            }
            return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({"status": "failure", "error": "Prediction function failed", "message": str(e)})

# --- 4. Tool Definitions and Mapping ---

# MODIFIED: Added the new namespace resource usage tool
TOOLS_DEF = [
    {
        "type": "function", "function": {
            "name": "list_all_namespaces",
            "description": "Lists all available namespaces in the Kubernetes cluster. Use when the user wants to know which namespaces are available, or asks about a non-existent namespace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function", "function": {
            "name": "list_pods_in_namespace",
            "description": "Lists all pods in the specified namespace. Use when the user provides a namespace but no pod name, or asks what pods are in a certain namespace.",
            "parameters": {
                "type": "object",
                "properties": {"namespace": {"type": "string"}},
                "required": ["namespace"],
            },
        },
    },
    # NEW TOOL DEFINITION
    {
        "type": "function", "function": {
            "name": "get_namespace_resource_usage",
            "description": "Retrieves the total aggregated CPU or Memory usage for all pods combined within an entire namespace. Use this when the user asks about the resource consumption of a namespace itself, not a specific pod.",
            "parameters": {
                "type": "object", "properties": {
                    "metric": {"type": "string", "enum": ["cpu", "memory"]},
                    "namespace": {"type": "string"},
                    "duration": {"type": "string", "default": "5m", "description": "Time window for the query, e.g., '10m', '1 hour'"},
                }, "required": ["metric", "namespace"],
            },
        },
    },
    {
        "type": "function", "function": {
            "name": "get_pod_resource_usage",
            "description": "Retrieves the average CPU or current Memory usage for a specific pod within a given time range.",
            "parameters": {
                "type": "object", "properties": {
                    "metric": {"type": "string", "enum": ["cpu", "memory"]},
                    "namespace": {"type": "string"}, "pod": {"type": "string"},
                    "duration": {"type": "string", "default": "5m"},
                }, "required": ["metric", "namespace", "pod"],
            },
        },
    },
    {
        "type": "function", "function": {
            "name": "get_top_k_pods",
            "description": "Finds the top K pods with the highest CPU or Memory usage in the specified namespace.",
            "parameters": {
                "type": "object", "properties": {
                    "metric": {"type": "string", "enum": ["cpu", "memory"]},
                    "namespace": {"type": "string"},
                    "k": {"type": "integer", "default": 3},
                    "duration": {"type": "string", "default": "5m"},
                }, "required": ["metric", "namespace"],
            },
        },
    },
    {
        "type": "function", "function": {
            "name": "predict_pod_cpu_usage",
            "description": "Predicts the CPU usage trend, average, and peak for a specific pod over a future period based on historical data.",
            "parameters": {
                "type": "object", "properties": {
                    "namespace": {"type": "string"}, "pod": {"type": "string"},
                    "duration": {"type": "string", "default": "1h"},
                    "future_duration": {"type": "string", "default": "30m"},
                    "step": {"type": "string", "default": "5m"},
                }, "required": ["namespace", "pod"],
            },
        },
    },
]

def _tool_wrapper(func, model_class=None, **kwargs):
    try:
        if model_class:
            metric = kwargs.pop("metric", None)
            model_instance = model_class(**kwargs)
            if metric: return func(model_instance, metric=metric)
            return func(model_instance)
        # For functions with no parameters like list_all_namespaces
        return func()
    except ValidationError as e:
        return json.dumps({"error": "Parameter validation failed", "details": str(e)})
    except Exception as e:
        return json.dumps({"error": "Function execution failed", "details": str(e)})

# MODIFIED: Added the new namespace function to the map
FUNC_MAP = {
    "list_all_namespaces": lambda **kwargs: _tool_wrapper(list_all_namespaces, **kwargs),
    "list_pods_in_namespace": lambda **kwargs: _tool_wrapper(list_pods_in_namespace, NamespaceInput, **kwargs),
    "get_namespace_resource_usage": lambda **kwargs: _tool_wrapper(get_namespace_resource_usage, NamespaceResourceInput, **kwargs), # NEW
    "get_pod_resource_usage": lambda **kwargs: _tool_wrapper(get_pod_resource_usage, PodResourceInput, **kwargs),
    "get_top_k_pods": lambda **kwargs: _tool_wrapper(get_top_k_pods, TopKInput, **kwargs),
    "predict_pod_cpu_usage": lambda **kwargs: _tool_wrapper(predict_pod_cpu_usage, PredictionInput, **kwargs),
}


# --- 5. Core Chat Logic ---

SYSTEM_PROMPT = """
You are a very smart and patient Kubernetes monitoring assistant. Your goal is to help users understand the status of their clusters in the most friendly and human-like way by calling tools.

**Core Interaction Principles:**

1.  **Understand User Intent:** Carefully distinguish whether the user is asking about a specific `pod` or an entire `namespace`. Use the `get_namespace_resource_usage` tool for namespace-level queries.

2.  **Proactive Guidance, Never Leave the User Stuck:**
    * If the information the user wants to query is incomplete (e.g., just says "check pod cpu" without providing a pod name or namespace), **you must never just report an error**.
    * Your standard operating procedure is:
        1.  If the `namespace` is uncertain, first call the `list_all_namespaces` tool, then ask the user: "Okay, which namespace would you like to query? Here are the available options: ...".
        2.  If the user provides a `namespace` but no `pod` name, and their query seems to be about individual pods (e.g., "which pod uses most memory"), call the `list_pods_in_namespace` tool to provide options: "No problem, I found these pods in '{namespace}', which one would you like to check? ...".
        3.  Through a question-and-answer process, guide the user to provide all necessary information, and then call the final query tool.

3.  **Data Interpretation, Not Dumping:**
    * When a tool returns data, your task is to transform it into human-understandable language.
    * For example, convert `{"value": 0.15}` to "CPU usage is approximately 150m (millicores)."
    * Convert `{"value": 524288000}` to "Memory usage is approximately 500 MiB."
    * Present Top-K results in a list, and describe prediction results with summary language (e.g., "It looks like CPU usage will continue to rise over the next 30 minutes, with an estimated average of...").

4.  **Always Friendly and Patient:** Your tone is always helpful. Even if you ask multiple times, maintain patience. Your goal is to make the user feel like they are talking to an expert colleague.
"""

# In services/llm_assistant.py

def chat_with_llm(user_message: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Handles the chat logic with the LLM, including a two-step process for tool use.
    This version corrects the logic flow to prevent errors when no tools are called.
    """
    history = history or []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_message}]
    
    try:
        # --- First Call: Decide on tools ---
        response_fc = client_fc.chat.completions.create(
            model=FUNC_MODEL,
            messages=messages,
            tools=TOOLS_DEF,
            tool_choice="auto"
        )
    except Exception as e:
        error_message = f"An error occurred during the initial tool-call request: {e}"
        history.extend([{"role": "user", "content": user_message}, {"role": "assistant", "content": error_message}])
        return {"assistant": error_message, "history": history}

    response_message = response_fc.choices[0].message
    tool_calls = response_message.tool_calls

    # --- Logic Correction ---
    # Append the assistant's first response (which may include the decision to call tools).
    # We must convert the response object to a dictionary to handle potential None content.
    response_message_dict = response_message.model_dump()
    
    # CRITICAL FIX: Prevent 'content: null' error by replacing None with an empty string.
    # This is the direct cause of the `invalid message content type: <nil>` error.
    if response_message_dict.get("content") is None:
        response_message_dict["content"] = ""
        
    messages.append(response_message_dict)

    # If the model decided to call tools, execute them and append the results.
    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = FUNC_MAP.get(function_name)
            if function_to_call:
                try:
                    function_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    function_response = function_to_call(**function_args)
                except Exception as e:
                    function_response = json.dumps({"error": f"Tool execution error: {e}"})
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                })

    # --- Second Call: Generate Final User-Facing Response ---
    # This call now runs consistently, whether tools were used or not.
    try:
        final_response = client_chat.chat.completions.create(
            model=CHAT_MODEL, 
            messages=messages
        )
        assistant_response = final_response.choices[0].message.content
    except Exception as e:
        error_message = f"An error occurred during the final response generation: {e}"
        # Even if the second call fails, return the error message.
        assistant_response = error_message

    # --- History Management (Unchanged) ---
    final_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response}
    ]
    
    # Prune history if it gets too long
    while len(json.dumps(final_history)) > MAX_TOTAL_JSON_LEN and len(final_history) > 2:
        final_history = final_history[2:]

    return {"assistant": assistant_response, "history": final_history}

# --- 6. Example Usage ---
if __name__ == "__main__":
    if not KUBERNETES_AVAILABLE:
        print("\n⚠️ \033[93mWarning: Kubernetes Python client not installed!\033[0m")
        print("   Some core functionalities (like querying namespaces and pods) will be unavailable.")
        print("   Please run: pip install kubernetes\n")

    conversation_history = []
    print("🤖 Kubernetes Monitoring Assistant (Upgraded Version). Type 'exit' to quit.")
    print("✨ Try asking: 'What's the total CPU usage of the monitoring namespace over the last day?' or 'Check pod cpu since 25\"4 hours ago'")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit": break
        result = chat_with_llm(user_input, conversation_history)
        print(f"\nAssistant: {result['assistant']}\n")
        conversation_history = result['history']