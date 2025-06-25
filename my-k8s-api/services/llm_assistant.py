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
from sklearn.linear_model import LinearRegression
import logging

from .rag_utils_dynamic import rag_system
from .log_handler import log_successful_call, init_db

load_dotenv()

FUNC_MODEL = os.getenv("FUNC_MODEL", "llama3.1:latest")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

MAX_HISTORY_LEN = 20
MAX_TOOL_SUMMARY_LEN = 400
MAX_TOTAL_JSON_LEN = 15000

client_fc = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
client_chat = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

logger = logging.getLogger(__name__)

class PrometheusService:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "error": f"Request failed: {e.__class__.__name__}", "message": str(e)}

    def query(self, promql: str) -> Dict[str, Any]:
        return self._request("/api/v1/query", {"query": promql})

    def query_range(self, promql: str, start: datetime, end: datetime, step: str) -> Dict[str, Any]:
        params = {
            "query": promql,
            "start": start.isoformat() + "Z",
            "end": end.isoformat() + "Z",
            "step": step,
        }
        return self._request("/api/v1/query_range", params)

prometheus_service = PrometheusService(PROMETHEUS_URL)

def _duration_to_seconds(duration: str) -> int:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    match = re.match(r"(\d+)([smhdw])", duration)
    if not match:
        raise ValueError("Invalid duration format. Use formats like '30s', '10m', '2h', '1d', '1w'.")
    value, unit = match.groups()
    return int(value) * units[unit]

def get_pod_cpu_usage(namespace: str, pod: str, range_str: str = "[1h]", **kwargs) -> str:
    clean_range = range_str.strip("[]")
    try:
        duration_seconds = _duration_to_seconds(clean_range)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=duration_seconds)
        step_seconds = max(15, duration_seconds // 200) 
        step = f"{int(step_seconds)}s"
        q = f'rate(container_cpu_usage_seconds_total{{namespace="{namespace}",pod="{pod}"}}[{step}])'
        return json.dumps(prometheus_service.query_range(q, start_time, end_time, step))
    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e)})

def get_pod_memory_usage(namespace: str, pod: str, range_str: str = "[1h]", **kwargs) -> str:
    clean_range = range_str.strip("[]")
    try:
        duration_seconds = _duration_to_seconds(clean_range)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=duration_seconds)
        step_seconds = max(15, duration_seconds // 200)
        step = f"{int(step_seconds)}s"
        q = f'container_memory_usage_bytes{{namespace="{namespace}",pod="{pod}"}}'
        return json.dumps(prometheus_service.query_range(q, start_time, end_time, step))
    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e)})

def get_top_cpu_pods(namespace: str, k: int = 3, **kwargs) -> str:
    q = f'topk({k}, sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[5m])))'
    return json.dumps(prometheus_service.query(q))

def get_top_memory_pods(namespace: str, k: int = 3, **kwargs) -> str:
    q = f'topk({k}, sum by(pod)(container_memory_usage_bytes{{namespace="{namespace}"}}))'
    return json.dumps(prometheus_service.query(q))

def get_pod_resource_usage_over_time(namespace: str, pod: str, days: int = 7, **kwargs) -> str:
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    cpu_query = f'rate(container_cpu_usage_seconds_total{{namespace="{namespace}",pod="{pod}"}}[5m])'
    mem_query = f'container_memory_usage_bytes{{namespace="{namespace}",pod="{pod}"}}'
    cpu_data = prometheus_service.query_range(cpu_query, start, end, "1h")
    mem_data = prometheus_service.query_range(mem_query, start, end, "1h")
    return json.dumps({"cpu_usage": cpu_data, "memory_usage": mem_data})

def get_namespace_resource_usage_over_time(namespace: str, days: int = 7, **kwargs) -> str:
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    cpu_query = f'sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[5m]))'
    mem_query = f'sum by(pod)(container_memory_usage_bytes{{namespace="{namespace}"}})'
    cpu_data = prometheus_service.query_range(cpu_query, start, end, "1h")
    mem_data = prometheus_service.query_range(mem_query, start, end, "1h")
    return json.dumps({"cpu_usage": cpu_data, "memory_usage": mem_data})

def query_resource(level: str, target: str, metric: str = "cpu", duration: str = "1h", namespace: str | None = None) -> str:
    try:
        duration_seconds = _duration_to_seconds(duration)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=duration_seconds)
        step_seconds = max(15, duration_seconds // 200)
        step = f"{int(step_seconds)}s"
        if level == "pod":
            if not namespace and "/" not in target:
                return json.dumps({"error": "Namespace is required for pod-level queries."})
            elif namespace is None and "/" in target:
                namespace, target = target.split("/", 1)
        metric = metric.lower().replace(".usage", "")
        if metric not in ("cpu", "memory"):
            return json.dumps({"error": "Metric must be 'cpu' or 'memory'"})
        if level == "pod":
            query = f'rate(container_cpu_usage_seconds_total{{namespace="{namespace}",pod="{target}"}}[{step}])' if metric == "cpu" else f'container_memory_usage_bytes{{namespace="{namespace}",pod="{target}"}}'
        elif level == "node":
            query = f'node_{metric}_usage_seconds_total{{node="{target}"}}'
        elif level == "namespace":
            query = f'sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{target}"}}[{step}]))' if metric == "cpu" else f'sum by(pod)(container_memory_usage_bytes{{namespace="{target}"}})'
        else:
            return json.dumps({"error": f"Unsupported level: {level}"})
        result = prometheus_service.query_range(query, start_time, end_time, step)
        return json.dumps(result)
    except ValueError as ve:
        return json.dumps({"error": f"Parameter validation error: {str(ve)}"})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred during query_resource: {str(e)}"})

def top_k_pods(namespace: str, metric: str, k: int = 3, duration: str = "5m") -> str:
    if metric not in ("cpu", "memory"):
        return json.dumps({"error": "Invalid metric. Use 'cpu' or 'memory'"})
    if metric == "cpu":
        query = f'topk({k}, sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[{duration}])))'
    else:
        query = f'topk({k}, sum by(pod)(container_memory_usage_bytes{{namespace="{namespace}"}}))'
    return json.dumps(prometheus_service.query(query))

def fetch_pod_cpu_metrics(namespace: str, pod: str, duration: str = "1h", step: str = "5m") -> pd.DataFrame:
    try:
        end = datetime.utcnow()
        start = end - timedelta(seconds=_duration_to_seconds(duration))
        query = f'rate(container_cpu_usage_seconds_total{{namespace="{namespace}", pod="{pod}"}}[{step}])'
        result = prometheus_service.query_range(query, start, end, step)
        if result.get("status") != "success" or not result.get("data", {}).get("result"):
            return pd.DataFrame()
        values = result["data"]["result"][0]["values"]
        if not values:
            return pd.DataFrame()
        df = pd.DataFrame(values, columns=["ds", "y"])
        df["ds"] = pd.to_datetime(df["ds"], unit='s')
        df["y"] = df["y"].astype(float)
        return df
    except Exception as e:
        return pd.DataFrame()

def predict_pod_cpu_usage(namespace: str, pod: str, duration: str = "1h", step: str = "5m", future_duration: str = "1h") -> str:
    try:
        import pmdarima as pm
    except ImportError:
        return json.dumps({"error": "Prediction library 'pmdarima' is not installed. Prediction is unavailable."})
    try:
        df = fetch_pod_cpu_metrics(namespace, pod, duration, step)
        if df.empty or len(df) < 2:
            return json.dumps({"error": f"Not enough historical data points to predict."})
        model = pm.auto_arima(df['y'], seasonal=False, suppress_warnings=True)
        future_steps = _duration_to_seconds(future_duration) // _duration_to_seconds(step)
        predictions, conf_int = model.predict(n_periods=future_steps, return_conf_int=True)
        last_timestamp = df["ds"].iloc[-1]
        step_seconds = _duration_to_seconds(step)
        future_timestamps = [last_timestamp + timedelta(seconds=(i + 1) * step_seconds) for i in range(future_steps)]
        output = {
            "pod": pod, "namespace": namespace,
            "prediction_window_start": future_timestamps[0].isoformat(),
            "prediction_window_end": future_timestamps[-1].isoformat(),
            "predictions": [
                {"timestamp": ts.isoformat(), "predicted_cpu_usage": round(float(p), 4)}
                for ts, p in zip(future_timestamps, predictions)
            ]
        }
        return json.dumps(output, indent=2)
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred during prediction: {str(e)}"})

def generate_csv_link(namespace: str, pod: str, range: str = "[1h]") -> str:
    if not range.startswith("["):
        bracketed = f"[{range}]"
    else:
        bracketed = range
    url = f"/api/export_csv?namespace={namespace}&pod={pod}&range={bracketed}"
    return json.dumps({"message": f"You can download the CSV for pod '{pod}' from: [Download CSV]({url})"})

def create_hpa_for_deployment(namespace: str, deployment: str, metric: str = "cpu", min_replicas: int = 1, max_replicas: int = 5, target_utilization: int = 60) -> str:
    try:
        from kubernetes import client, config
        config.load_kube_config()
        autoscaling_v1 = client.AutoscalingV1Api()
        hpa = client.V1HorizontalPodAutoscaler(
            metadata=client.V1ObjectMeta(name=f"{deployment}-hpa", namespace=namespace),
            spec=client.V1HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V1CrossVersionObjectReference(api_version="apps/v1", kind="Deployment", name=deployment),
                min_replicas=min_replicas, max_replicas=max_replicas,
                target_cpu_utilization_percentage=target_utilization if metric == "cpu" else None
            )
        )
        autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(namespace=namespace, body=hpa)
        return json.dumps({"status": "success", "message": f"HPA '{deployment}-hpa' created successfully."})
    except ImportError:
        return json.dumps({"status": "error", "message": "Kubernetes client library not installed."})
    except client.ApiException as e:
        return json.dumps({"status": "error", "message": f"Kubernetes API error: {e.status} - {e.reason}"})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"An unexpected error occurred: {str(e)}"})

TOOLS_DEF = [
    {"type": "function", "function": {"name": "get_pod_cpu_usage", "description": "Get CPU usage for a specific pod in a namespace over a time range.", "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}, "pod": {"type": "string"}, "range_str": {"type": "string", "default": "[1h]"}}, "required": ["namespace", "pod"]}}},
    {"type": "function", "function": {"name": "get_pod_memory_usage", "description": "Get memory usage for a specific pod in a namespace over a time range.", "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}, "pod": {"type": "string"}, "range_str": {"type": "string", "default": "[1h]"}}, "required": ["namespace", "pod"]}}},
    {"type": "function", "function": {"name": "get_top_cpu_pods", "description": "Top-K pods by current CPU usage in a namespace.", "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}, "k": {"type": "integer", "default": 3}}, "required": ["namespace"]}}},
    {"type": "function", "function": {"name": "get_top_memory_pods", "description": "Top-K pods by current memory usage in a namespace.", "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}, "k": {"type": "integer", "default": 3}}, "required": ["namespace"]}}},
    {"type": "function", "function": {"name": "get_pod_resource_usage_over_time", "description": "Get CPU and memory usage for a pod over the past N days.", "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}, "pod": {"type": "string"}, "days": {"type": "integer", "default": 7}}, "required": ["namespace", "pod"]}}},
    {"type": "function", "function": {"name": "get_namespace_resource_usage_over_time", "description": "Get resource usage for all pods in a namespace over time.", "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}, "days": {"type": "integer", "default": 7}}, "required": ["namespace"]}}},
    {"type": "function", "function": {"name": "query_resource", "description": "A comprehensive tool to query CPU or memory usage of a pod, node, or namespace.", "parameters": {"type": "object", "properties": {"level": {"type": "string", "enum": ["pod", "node", "namespace"]}, "target": {"type": "string"}, "metric": {"type": "string", "enum": ["cpu", "memory"]}, "duration": {"type": "string"}, "namespace": {"type": "string"}}, "required": ["level", "target", "metric", "duration"]}}},
    {"type": "function", "function": {"name": "top_k_pods", "description": "Return top-K pods by CPU/Memory in a namespace, calculated over a duration.", "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}, "metric": {"type": "string", "enum": ["cpu", "memory"]}, "k": {"type": "integer", "default": 3}, "duration": {"type": "string", "default": "5m"}}, "required": ["namespace", "metric"]}}},
    {"type": "function", "function": {"name": "generate_csv_link", "description": "Generate a CSV download link for a specific pod's usage.", "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}, "pod": {"type": "string"}, "range": {"type": "string", "default": "[1h]"}}, "required": ["namespace", "pod"]}}},
    {"type": "function", "function": {"name": "predict_pod_cpu_usage", "description": "Predict future CPU usage for a specific pod.", "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}, "pod": {"type": "string"}, "duration": {"type": "string", "default": "1h"}, "step": {"type": "string", "default": "5m"}, "future_duration": {"type": "string", "default": "1h"}}, "required": ["namespace", "pod"]}}},
    {"type": "function", "function": {"name": "create_hpa_for_deployment", "description": "Create a Horizontal Pod Autoscaler (HPA) for a deployment.", "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}, "deployment": {"type": "string"}, "metric": {"type": "string", "enum": ["cpu"]}, "min_replicas": {"type": "integer", "default": 1}, "max_replicas": {"type": "integer", "default": 5}, "target_utilization": {"type": "integer", "default": 60}}, "required": ["namespace", "deployment"]}}}
]

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
    "predict_pod_cpu_usage": predict_pod_cpu_usage,
    "create_hpa_for_deployment": create_hpa_for_deployment,
}

SYSTEM_PROMPT = """You are a Kubernetes observability assistant. Your job is to help users monitor, analyze, and troubleshoot cluster resources in real-time.

Your capabilities include:
1. Monitoring CPU and memory usage for specific pods, nodes or namespaces.
2. Identifying top-K resource-consuming pods by CPU or memory.
3. Analyzing resource usage trends over time (e.g., last 1h, 1d).
4. Predicting future CPU usage for a specific pod.

When a user asks a question, first refer to the RAG context for successful examples. Based on the user's query and the examples, determine if it requires a tool call. If so, call the most appropriate tool and use the results to provide a concise, helpful, and actionable summary. If the user's request is ambiguous (e.g., "check usage" without specifying a pod), ask clarifying questions to get the necessary details. Avoid guessing if data is unavailable—inform the user clearly and suggest alternatives. For example, if prediction is not possible, offer to show the current top pods instead.
"""

def _prune_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(history) > MAX_HISTORY_LEN * 2:
        return history[-MAX_HISTORY_LEN*2:]
    return history

async def chat_with_llm(user_query: str, chat_history: list = None):
    if chat_history is None:
        chat_history = []
    
    chat_history = _prune_history(chat_history)
    rag_context = rag_system.retrieve(user_query, top_k=3)
    system_prompt_with_rag = f"{SYSTEM_PROMPT}\n\n[RAG Knowledge - Successful Past Examples]\n{rag_context}"

    messages = [{"role": "system", "content": system_prompt_with_rag}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_query})

    print("Step 1: First LLM call (tool decision / conversation)")
    first_response = client_chat.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=TOOLS_DEF,
        tool_choice="auto",
    )
    assistant_message = first_response.choices[0].message
    
    assistant_message_dict = {"role": "assistant", "content": str(assistant_message.content or "")}
    if assistant_message.tool_calls:
        assistant_message_dict["tool_calls"] = json.loads(assistant_message.model_dump_json(exclude_unset=True))['tool_calls']
    
    messages.append(assistant_message_dict)

    if assistant_message.tool_calls:
        print(f"Step 2: LLM decided to call tools: {[tc.function.name for tc in assistant_message.tool_calls]}")
        
        tool_outputs = []
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = FUNC_MAP.get(function_name)
            
            if function_to_call:
                try:
                    args = json.loads(tool_call.function.arguments)
                    result = function_to_call(**args)
                    log_successful_call(user_query, f"{function_name}(**{json.dumps(args)})")
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": result,
                    })
                except Exception as e:
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": f'{{"error": "Failed to execute tool: {str(e)}"}}',
                    })
        messages.extend(tool_outputs)
        
        print("Step 3: Second LLM call (summarizing tool results)")
        second_response = client_chat.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tool_choice="none",
        )
        final_answer = second_response.choices[0].message.content
        assistant_final_message_dict = {"role": "assistant", "content": final_answer}
        messages.append(assistant_final_message_dict)

    else:
        print("Step 2: LLM decided to respond directly.")
        final_answer = assistant_message.content

    final_history = [msg for msg in messages if msg.get("role") != "system"]
    return {"assistant": final_answer, "history": _prune_history(final_history)}


if __name__ == "__main__":
    conversation_history = []
    print("Kubernetes Monitoring Assistant. Type 'exit' to quit.")
    try:
        init_db()
        print("Database initialized.")
    except Exception as e:
        print(f"Warning: Could not initialize database: {e}")

    import asyncio
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        try:
            result = asyncio.run(chat_with_llm(user_input, conversation_history))
            print(f"Assistant: {result['assistant']}")
            conversation_history = result['history']
        except Exception as e:
            print(f"An error occurred in the main chat loop: {e}")