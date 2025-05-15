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
    """
    查詢 pod / node / namespace 的 CPU 或記憶體使用量，支援範圍時間驗證。
    支援格式：30m, 1h, 2d, 1mo（最多 10mo）
    """
    match = re.match(r"^(\d+)(m|h|d|mo)$", duration)
    if not match:
        return json.dumps({"error": "Invalid duration format. Use 30m, 1h, 2d, 1mo"})

    value, unit = match.groups()
    value = int(value)
    if unit == "mo" and value > 10:
        return json.dumps({"error": "Max supported duration is 10mo"})

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
        if not namespace:
            return json.dumps({"error": "Namespace is required for pod-level query"})
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
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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


def predict_pod_cpu_next_hour(namespace: str, pod: str, duration: str = "1h", step: str = "5m") -> str:
    """
    預測下一小時 Pod 的 CPU 使用率，基於過去 1 小時的資料
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

        # 預測未來 12 個 5 分鐘
        future_X = np.array(range(len(df), len(df) + 12)).reshape(-1, 1)
        predictions = model.predict(future_X)

        future_df = pd.DataFrame({
            "timestamp": pd.date_range(start=df["ds"].iloc[-1], periods=12, freq=step),
            "predicted_cpu_usage": predictions
        })

        return future_df.to_json(orient='records')
    
    except Exception as e:
        print(f"[ERROR] 預測失敗：{str(e)}")
        return json.dumps({"error": str(e)})



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
            "description": "Query CPU or Memory usage for pod / node / namespace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "enum": ["pod", "node", "namespace"]},
                    "target": {"type": "string", "description": "pod name, node name, or namespace name"},
                    "metric": {"type": "string", "enum": ["cpu", "memory"]},
                    "duration": {"type": "string", "description": "look‑back window like 30m, 4h, 2d, 1mo (max 10mo)"},
                    "namespace": {"type": "string"}
                },
                "required": ["level", "target", "metric", "duration"]
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
            "name": "predict_pod_cpu_next_hour",
            "description": "Predict the CPU usage for the next hour for a specific pod using historical data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Namespace of the pod"},
                    "pod": {"type": "string", "description": "Pod name"},
                    "duration": {"type": "string", "default": "1h"},
                    "step": {"type": "string", "default": "5m"}
                },
                "required": ["namespace", "pod"]
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
    "predict_pod_cpu_next_hour": predict_pod_cpu_next_hour,
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

When a user asks a question, determine if it requires a tool call (e.g., to Prometheus). If so, call the appropriate tool and use the results to provide a concise, helpful, and actionable summary. Avoid guessing if data is unavailable—inform the user clearly and suggest alternatives.

**Special Instructions for CSV Download:**
- If the user mentions "download CSV", "export CSV", or similar phrases, call the generate_csv_link function.
- If namespace, pod, or range are not provided in the prompt, infer them from the conversation history (e.g., if a previous message mentioned a namespace or pod).
- If you cannot infer the required parameters, respond with a message asking the user to specify the missing information (e.g., "Please specify the namespace and pod for the CSV download.").
- Default range to "[1h]" if not specified.
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
    """
    從對話歷史中推斷 namespace、pod 和 range。
    如果無法推斷，返回 None。
    """
    inferred_params = {"namespace": None, "pod": None, "range": "[1h]"}  # 預設 range 為 [1h]

    # 檢查 user_message 是否有關鍵詞
    if "download csv" in user_message.lower() or "export csv" in user_message.lower():
        # 從歷史訊息中提取 namespace 和 pod
        for msg in reversed(history):
            content = msg.get("content", "").lower()
            # 尋找 namespace
            if not inferred_params["namespace"] and "namespace" in content:
                match = re.search(r'namespace\s*[:=]\s*([a-zA-Z0-9_-]+)', content)
                if match:
                    inferred_params["namespace"] = match.group(1)
            # 尋找 pod
            if not inferred_params["pod"] and "pod" in content:
                match = re.search(r'pod\s*[:=]\s*([a-zA-Z0-9_-]+)', content)
                if match:
                    inferred_params["pod"] = match.group(1)
            # 尋找 range
            if "range" in content:
                match = re.search(r'range\s*[:=]\s*\[?(\d+[mhd]|1mo)\]?', content)
                if match:
                    inferred_params["range"] = f"[{match.group(1)}]"

    return inferred_params

def chat_with_llm(user_message: str, history: list | None = None) -> Dict[str, Any]:
    """同步函式，不要用 await 呼叫"""
    history = prune_history(history or [])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message},
    ]

    # 檢查是否需要生成 CSV 並推斷參數
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
        # 直接模擬 tool call
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
        # 第二次呼叫 LLM，讓它根據結果生成自然語言回應
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

    # 第一次呼叫：LLM 決定是否呼叫工具
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

    # 先取出 tool_calls
    tool_calls = getattr(assistant_msg, "tool_calls", None) or []
    messages.append({
        "role": "assistant",
        "content": assistant_msg.content,
        "tool_calls": tool_calls
    })

    # 如果有 tool_calls，就執行並 append 結果
    for tool_call in tool_calls:
        fn = FUNC_MAP.get(tool_call.function.name)
        if not fn:
            continue
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            err_msg = (
                "⚠️ I generated malformed tool arguments, "
                "please rephrase your question (e.g. fix the time window)."
            )
            return {"assistant": err_msg, "history": history or []}
        result = fn(**args)
        messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id
        })

    # 第二次呼叫：根據工具結果輸出最終回答
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

    # 返回給前端：去除 system，並確保 assistant 回應不為 None
    front_history = [m for m in messages if m["role"] != "system"]
    last_msg = messages[-1]
    assistant_content = last_msg.get("content") or "[Error: No response generated from the assistant.]"
    
    return {"assistant": assistant_content, "history": front_history}
