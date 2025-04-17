# services/llm_assistant.py
from __future__ import annotations
import os, json, requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

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

def get_pod_cpu_usage(namespace: str, pod: str, range_str: str = "[1h]") -> str:
    q = f'container_cpu_usage_seconds_total{{namespace="{namespace}",pod="{pod}"}}{range_str}'
    return json.dumps(_prom_query(q))

def get_pod_memory_usage(namespace: str, pod: str, range_str: str = "[1h]") -> str:
    q = f'container_memory_usage_bytes{{namespace="{namespace}",pod="{pod}"}}{range_str}'
    return json.dumps(_prom_query(q))

def get_top_cpu_pods(namespace: str, k: int = 3) -> str:
    q = f'topk({k}, sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[5m])))'
    return json.dumps(_prom_query(q))

def get_top_memory_pods(namespace: str, k: int = 3) -> str:
    q = f'topk({k}, sum by(pod)(container_memory_usage_bytes{{namespace="{namespace}"}}))'
    return json.dumps(_prom_query(q))

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
]

FUNC_MAP = {
    "get_pod_cpu_usage":    get_pod_cpu_usage,
    "get_pod_memory_usage": get_pod_memory_usage,
    "get_top_cpu_pods":     get_top_cpu_pods,
    "get_top_memory_pods":  get_top_memory_pods,
}

SYSTEM_PROMPT = (
    "You are a Kubernetes monitoring assistant. "
    "Call functions when you need Prometheus data, then summarize the results."
)

def chat_with_llm(user_message: str, history: list | None = None) -> Dict[str, Any]:
    """同步函式，不要用 await 呼叫"""
    history = history or []
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
    tool_calls = getattr(assistant_msg, "tool_calls", [])
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

    # 返回給前端：去除 system
    front_history = [m for m in messages if m["role"] != "system"]
    return {"assistant": messages[-1]["content"], "history": front_history}
