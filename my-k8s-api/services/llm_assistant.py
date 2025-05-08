# services/llm_assistant.py
from __future__ import annotations
import os, json, requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq
from typing import Optional
from pydantic import BaseModel
import re
from datetime import datetime, timedelta


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

def get_pod_cpu_usage(namespace: str, pod: str, range_str: str = "[1h]",**kwargs) -> str:
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

def get_pod_resource_usage_over_time(namespace: str, pod: str, days: int = 7) -> str:
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

'''
新的兩隻工具
'''
def query_resource(level: str, target: str, metric: str, duration: str) -> str:
    """
    查詢 pod / node / namespace 的 CPU 或記憶體使用量，支援範圍時間驗證。
    支援格式：30m, 1h, 2d, 1mo（最多 10mo）
    """
    # 驗證 duration 格式
    match = re.match(r"^(\d+)(m|h|d|mo)$", duration)
    if not match:
        return json.dumps({"error": "Invalid duration format. Use formats like 30m, 1h, 2d, 1mo"})

    value, unit = match.groups()
    value = int(value)
    
    if unit == "mo" and value > 10:
        return json.dumps({"error": "Max supported duration is 10mo"})

    now = datetime.utcnow()
    if unit == "mo":
        delta = timedelta(days=30 * value)
    elif unit == "d":
        delta = timedelta(days=value)
    elif unit == "h":
        delta = timedelta(hours=value)
    elif unit == "m":
        delta = timedelta(minutes=value)
    else:
        return json.dumps({"error": "Unsupported time unit"})

    start = int((now - delta).timestamp())
    end = int(now.timestamp())
    step = "1h"  # 固定步進

    if level == "pod":
        query = f'container_{metric}_usage_seconds_total{{pod="{target}"}}'
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
                    "pod":       {"type": "string"},
                    "days":      {"type": "integer"}
                },
                "required": ["namespace","pod","days"]
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
        "type": "function",           # 👈 加這行
        "function": {
            "name": "get_pod_resource_usage_over_time",
            "description": "Get CPU and memory usage for a pod over the past N days",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "pod":       {"type": "string"},
                    "days":      {"type": "integer"}
                },
                "required": ["namespace", "pod", "days"]
            }
        }
    },


    {
        "type": "function",
        "function": {
        "name": "query_resource",
        "description": "Query CPU or Memory usage for pod / node / namespace.",
        "parameters": {
            "type": "object",
            "properties": {
            "level":   { "type": "string", "enum": ["pod", "node", "namespace"] },
            "target":  { "type": "string", "description": "pod name, node name, or namespace name" },
            "metric":  { "type": "string", "enum": ["cpu", "memory"] },
            "duration":{ "type": "string", "description": "look‑back window like 30m, 4h, 2d, 1mo (max 10mo)" }
            },
            "required": ["level","target","metric","duration"]
        }
        }
    },
    {
        "type": "function",
        "function": {
        "name": "top_k_pods",
        "description": "Return the Top‑K pods by CPU or Memory in a namespace.",
        "parameters": {
            "type": "object",
            "properties": {
            "namespace": { "type": "string" },
            "metric":    { "type": "string", "enum": ["cpu", "memory"] },
            "k":         { "type": "integer", "default": 3, "minimum": 1 },
            "duration":  { "type": "string", "default": "5m" }
            },
            "required": ["namespace", "metric"]
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
     # ─── 新 2 支 ───
    "query_resource": query_resource,
    "top_k_pods":     top_k_pods,
}

SYSTEM_PROMPT = """You are a Kubernetes monitoring assistant. You can help users understand their cluster's resource usage patterns and trends.

You can:
1. Monitor specific pods' CPU and memory usage
2. Find top resource-consuming pods
3. Analyze namespace-wide resource usage
4. Track resource usage over time
5. Identify resource usage trends and growth rates

Call appropriate functions to gather data, then provide clear, actionable insights based on the results.
"""
MAX_HISTORY_LEN = 20   # 依需要調

def prune_history(hist: list[dict]) -> list[dict]:
    """只保留最近 MAX_HISTORY_LEN 句對話"""
    if len(hist) > MAX_HISTORY_LEN:
        return hist[-MAX_HISTORY_LEN:]
    return hist

def chat_with_llm(user_message: str, history: list | None = None) -> Dict[str, Any]:
    """同步函式，不要用 await 呼叫"""
    history = prune_history(history or [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    messages.append({"role": "user", "content": user_message})

    # 第一次呼叫：LLM 決定是否呼叫工具
    resp = client.chat.completions.create(
        model       = MODEL,
        messages    = messages,
        tools       = TOOLS_DEF,
        tool_choice = "auto",
        max_tokens  = 800,
    )
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
        args = json.loads(tool_call.function.arguments)
        result = fn(**args)
        messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id
        })

    # 第二次呼叫：根據工具結果輸出最終回答
    if tool_calls:
        resp2 = client.chat.completions.create(
            model       = MODEL,
            messages    = messages,
            tools       = TOOLS_DEF,
            tool_choice = "none",
            max_tokens  = 800,
        )
        final_msg = resp2.choices[0].message
        messages.append({"role": "assistant", "content": final_msg.content})

    # 返回給前端：去除 system，並確保 assistant 回應不為 None
    front_history = [m for m in messages if m["role"] != "system"]
    last_msg = messages[-1]
    assistant_content = last_msg.get("content") or "[Error: No response generated from the assistant.]"
    
    return {"assistant": assistant_content, "history": front_history}

