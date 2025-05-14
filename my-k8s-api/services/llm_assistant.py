from __future__ import annotations
import os, json, requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq
from typing import Optional
from pydantic import BaseModel
import re
import time
from datetime import datetime, timedelta
from services.rag_utils_en import rag_retrieve, is_supported
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging

class ChatResponse(BaseModel):
    assistant: Optional[str]
    history: List[dict]

load_dotenv()

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
MODEL          = "llama3-8b-8192"   # 請改成你帳號可用的 Groq 模型名稱

if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY 未設定！")

client = Groq(api_key=GROQ_API_KEY)

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
        params = {
            "query": q,
            "start": start,
            "end": end,
            "step": step
        }
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
    """
    回傳某 namespace 內，CPU 或記憶體使用最高的前 K 個 pod。
    """
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
    """
    從 Prometheus 取得 Pod 的 CPU 使用率
    """
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
    """
    預測未來 Pod 的 CPU 使用率，基於歷史數據。
    - `future_duration`: 預測的時間範圍（例如 "1h", "2h", "1d"）。
    Alan52254改的
    """
    try:
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
            "parameters": {
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
    "create_hpa_for_deployment": create_hpa_for_deployment
}

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

def chat_with_llm(user_message: str, history: list | None = None) -> Dict[str, Any]:
    history = prune_history(history or [])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message},
    ]

    # Handle CSV requests (unchanged logic)
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
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_DEF,
            tool_choice="none",
            max_tokens=800,
        )
        final_msg = resp.choices[0].message
        messages.append({"role": "assistant", "content": final_msg.content})
        front_history = [m for m in messages if m["role"] != "system"]
        return {"assistant": final_msg.content, "history": front_history}

    # First LLM call to decide tool usage
    from groq import BadRequestError
    try:
        resp = client.chat.completions.create(
            model=MODEL,
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
    
    assistant_msg = resp.choices[0].message
    tool_calls = getattr(assistant_msg, "tool_calls", None) or []
    messages.append({
        "role": "assistant",
        "content": assistant_msg.content,
        "tool_calls": tool_calls
    })

    # Execute tool calls with enhanced error handling
    for tool_call in tool_calls:
        fn = FUNC_MAP.get(tool_call.function.name)
        if not fn:
            continue
        try:
            args = json.loads(tool_call.function.arguments)
            result = fn(**args)
        except TypeError as e:
            error_msg = str(e)
            match = re.search(r"missing (\d) required positional argument: '(\w+)'", error_msg)
            if match:
                arg_name = match.group(2)
                assistant_response = f"Error: Missing required argument '{arg_name}' for {tool_call.function.name}. Please provide it."
            else:
                assistant_response = f"Error executing {tool_call.function.name}: {error_msg}"
            return {"assistant": assistant_response, "history": history}
        except Exception as e:
            assistant_response = f"Error executing {tool_call.function.name}: {str(e)}"
            return {"assistant": assistant_response, "history": history}
        messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id
        })

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
