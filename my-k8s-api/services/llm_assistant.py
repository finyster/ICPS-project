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
from services.rag_utils_en import rag_retrieve, is_supported

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
def query_resource(level: str, target: str, metric: str, duration: str, namespace: str | None = None) -> str:
    """
    查詢 pod / node / namespace 的 CPU 或記憶體使用量，支援範圍時間驗證。
    支援格式：30m, 1h, 2d, 1mo（最多 10mo）
    """

    # 驗證 duration 格式
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

    # 清理 metric 格式：處理 "cpu.usage" or "memory.usage"
    metric = metric.lower().replace(".usage", "")
    if metric not in ("cpu", "memory"):
        return json.dumps({"error": "Metric must be 'cpu' or 'memory'"})

    # 建立 Prometheus 查詢
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

# ──── CSV 下載連結 ────
# 這邊只是回傳一個下載連結，實際的 CSV 生成在 export_csv.py 裡面
def generate_csv_link(namespace: str, pod: str, range: str = "[1h]") -> str:
    if not range.startswith("["):
        bracketed = f"[{range}]"
    else:
        bracketed = range
    url = f"/api/export_csv?namespace={namespace}&pod={pod}&range={bracketed}"
    return json.dumps({
        "message": f"You can download the CSV from: [Download CSV]({url})"
    })

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
                "duration": { "type": "string", "description": "look‑back window like 30m, 4h, 2d, 1mo (max 10mo)" },
                "namespace":{ "type": "string" }           # ← **注意：前面要有逗號**
                },
                "required": ["level","target","metric","duration"]
            }
        }
    },
    # 只截取 top_k_pods 的定義部分
    {
        "type": "function",
        "function": {
            "name": "top_k_pods",
            "description": "Return top‑K pods by CPU/Memory in a namespace",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string",
                                "description": "Kubernetes namespace" },
                    "metric":    { "type": "string",
                                "enum": ["cpu", "memory"] },
                    "k":         { "type": "integer",
                                "default": 3,
                                "minimum": 1,
                                "description": "How many pods to return" },
                    "duration":  { "type": "string",
                                "default": "5m",
                                "pattern": "^(\\d+)(m|h|d|mo)$",
                                "description": "Look‑back window like 30m, 2h, 1d" }
                },
                "required": ["namespace", "metric"]
            }
        }
    },
    # ─── csv generator ───
    {
        "type": "function",
        "function": {
            "name": "generate_csv_link",
            "description": "Generate a CSV download link for a specific pod's CPU/Memory usage",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string" },
                    "pod":       { "type": "string" },
                    "range":     { "type": "string", "default": "1h" }
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
     # ─── 新 2 支 ───
    "query_resource": query_resource,
    "top_k_pods":     top_k_pods,
    # ─── CSV 下載 ───
    "generate_csv_link": generate_csv_link,
}
SYSTEM_PROMPT = """
You are a Kubernetes observability assistant. Your job is to help users monitor, analyze, and troubleshoot cluster resources in real-time.

Your capabilities include:
1. Monitoring CPU, memory, disk, and network usage for specific pods or nodes
2. Identifying top-K resource-consuming pods by CPU, memory, disk, or network
3. Checking pod readiness and general health status within a namespace
4. Analyzing resource usage trends over time (e.g., last 1h, 1d, 1mo)
5. Comparing usage across different namespaces or time intervals
6. Generating CSV reports for selected pods and time ranges. The report will be returned as a downloadable link.
7. Responding in natural language while calling relevant functions to fetch live data
8. Format multi-item results using bullet points or line breaks for readability.

When a user asks a question, determine if it requires a tool call (e.g., to Prometheus). If so, call the appropriate tool and use the results to provide a concise, helpful, and actionable summary. Avoid guessing if data is unavailable—inform the user clearly and suggest alternatives.
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
        # 保留最後 5 句，從最早開始砍
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

def chat_with_llm(user_message: str, history: list | None = None) -> Dict[str, Any]:
    """同步函式，不要用 await 呼叫"""
        # ---------- Filterv ---------- 下面這邊是阿亨柱解掉的RAG問題
#    if not is_supported(user_message):
#        return {
#            "assistant": (
#                "⚠️  I only answer Kubernetes pod / namespace monitoring "
#                "questions about CPU, Memory, Disk or Network in a valid "
#                "time range (e.g. 30m, 1h, 2d, 1mo). Please rephrase."
#            ),
#            "history": history or []
#        }
    # ---------- Retrieve ----------
    #rag_ctx = rag_retrieve(user_message, top_k=3)

    history = prune_history(history or [])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
       # {"role": "system", "content": f"[RAG Knowledge]\n{rag_ctx}"},   # ← 新增
        *history,
        {"role": "user",   "content": user_message},
    ]
    messages.append({"role": "user", "content": user_message})

    # 第一次呼叫：LLM 決定是否呼叫工具
    from groq import BadRequestError
    try:
        resp = client.chat.completions.create(
            model       = MODEL,
            messages    = messages,
            tools       = TOOLS_DEF,
            tool_choice = "auto",
            max_tokens  = 800,
        )
    except BadRequestError as err:
        # Groq 會回 {"error":{...,"code":"tool_use_failed",...}}
        human = (
            "⚠️  I couldn't execute that request (invalid tool arguments). "
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

    # 如果有 tool_calls，就執行並 append 結果(這邊解決開會出現的問題：Compare the top 3 CPU-heavy pods in the `default` namespace between now and 24 hours ago. What changed?)
    for tool_call in tool_calls:
        fn = FUNC_MAP.get(tool_call.function.name)
        if not fn:
            continue
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            # LLM 產生不合法 JSON → 回溫和錯誤，結束本輪對話
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