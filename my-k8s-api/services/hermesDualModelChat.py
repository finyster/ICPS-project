"""
Hermes Dual‑Stage Chat (v2) – 友善摘要版
========================================
‣ 第一階段：Hermes (`FUNC_MODEL`) 產生 <tool_call>…  
‣ 伺服器執行函式 → 包 <tool_response>…（JSON + markdown）  
‣ 第二階段：`CHAT_MODEL` 讀取 <tool_response>，輸出對人更友善的回答。

改動重點
-----------
1. **更友善的最終回答**  
   在進第 2 階段之前，插入一段額外 `system` 指令 `SUMMARIZER_SYSTEM`，
   告訴第二模型要用 bullet / 表格格式回覆。  
2. **嚴格轉數值型別**  
   `_normalize_args()` 會把 Hermes 回傳的字串數值轉成 `int/float`，
   避免 "k": "3" 這種情況。  
3. **函式回傳轉成 JSON**  
   若函式原本回 str → 先 `json.loads`；最終 `<tool_response>` 會包含
   ➜ 原始 JSON
   ➜ 已排版好的 markdown 區塊，讓第二模型易讀。
"""
from __future__ import annotations

import json, os, re, logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# ---------- 模型設定 ----------
FUNC_MODEL  = "nous-hermes"
CHAT_MODEL  = "dolphin-mistral:latest"      #dolphin-mistral:latest, nous-hermes:latest, llama3.1:latest, llama3:8b    
HF_API      = "http://localhost:11434/v1"   

client_fc   = OpenAI(base_url=HF_API, api_key="ollama")
client_chat = OpenAI(base_url=HF_API, api_key="ollama")

# ---------- 讀取環境 ----------
load_dotenv()
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
logger = logging.getLogger("hermes-assistant")

# ------------------------------------------------------------------
# ▶ Prometheus 工具 (保留原實作，自行 import)
# ------------------------------------------------------------------
from services.prom_tools import (
    list_node_names,
    list_namespace_names,
    list_pod_names,
    get_top_cpu_pods,
    get_top_memory_pods,
    get_pod_cpu_usage,
    get_pod_memory_usage,
    get_pod_resource_usage_over_time,
    get_namespace_resource_usage_over_time,
    query_resource,
    top_k_pods,
    generate_csv_link,
    predict_pod_cpu_usage,
    create_hpa_for_deployment,
)

FUNC_MAP = {
    "list_node_names": list_node_names,
    "list_namespace_names": list_namespace_names,
    "list_pod_names": list_pod_names,
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

# ------------------------------------------------------------------
# ▶  TOOL SCHEMA（集中於 services.tool_defs）
# ------------------------------------------------------------------
from services.tool_defs import TOOLS_DEF  # ↖︎ 你的 JSON Schema 列表

def _build_tools_json() -> str:
    return json.dumps(TOOLS_DEF, ensure_ascii=False)

TOOLS_JSON_TEXT = _build_tools_json()

# ------------------------------------------------------------------
# ▶  PROMPT 模板
# ------------------------------------------------------------------
SYSTEM_PREFIX = f"""
You are a Kubernetes observability assistant that MUST decide whether to call a tool.

<tools>{TOOLS_JSON_TEXT}</tools>

RULES
─────
1. **Decide tool usage**
   • If a tool is needed → output **exactly one** 
     <tool_call>{{"name": ..., "arguments": {{...}}}}</tool_call>
     (no extra text).
   • If NO tool is needed → output <tool_call>{{}}</tool_call>.

2. JSON shape MUST be:
   {{"name": <function-name>, "arguments": <object>}}

3. Argument types must strictly follow the schema:
   • Integers (`k`, `min_replicas`, …) must be int, not string.
   • `duration` / `future_duration` only accept 30m, 2h, 1d, 1mo …

4. DO NOT add any text outside <tool_call> or produce multiple JSON objects.

PHRASE ↔ TOOL MAPPING
─────────────────────
• “show / get CPU (memory) of pod …”     → get_pod_cpu_usage / get_pod_memory_usage
• “top K CPU pods …”                     → get_top_cpu_pods(namespace, k)
• “trend last N days …”                  → get_pod_resource_usage_over_time(days=N)
• “predict next 1h CPU …”                → predict_pod_cpu_usage(future_duration="1h")
• “create HPA …”                         → create_hpa_for_deployment(...)

Follow these rules; all natural-language interaction will be handled by the second-stage model.
"""


SUMMARIZER_SYSTEM = (
    "You are a helpful assistant. Summarize any preceding <tool_response> for the user.\n"
    "• Make the answer concise and friendly.\n"
    "• Use bullet points or tables when listing pods or metrics.\n"
    "• If an error occurred, politely explain and give guidance."
)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

class ChatResponse(BaseModel):
    assistant: str
    history:  List[dict]

# ------------------------------------------------------------------
# ▶  工具與參數助手
# ------------------------------------------------------------------

def _normalize_args(args: dict) -> dict:
    """將 '3' ➜ 3 / '4.5' ➜ 4.5，保證型別正確。"""
    fixed = {}
    for k, v in args.items():
        if isinstance(v, str) and v.isdigit():
            fixed[k] = int(v)
        else:
            try:
                fixed[k] = float(v) if isinstance(v, str) and re.match(r"^-?\d+\.\d+$", v) else v
            except Exception:
                fixed[k] = v
    return fixed


def _extract_tool_call(text: str):
    m = TOOL_CALL_RE.search(text)
    if not m:
        # 沒有 <tool_call> 標籤時，返回 None 並記錄 debug 訊息
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as e:
        # 回傳詳細錯誤訊息，方便 fallback 時 debug
        return {"error": f"JSON decode error in tool_call: {e}", "raw": m.group(1)}


# ------------------------------------------------------------------
# ▶  主邏輯
# ------------------------------------------------------------------

def chat_with_llm(
    user_message: str,
    history: List[dict] | None = None,
    max_depth: int = 3,
) -> Dict[str, Any]:
    history = history or []
    prompt: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PREFIX},
        *history,
        {"role": "user", "content": user_message},
    ]

    depth = 0
    while depth < max_depth:
        resp = client_fc.chat.completions.create(
            model=FUNC_MODEL,
            messages=prompt,
            max_tokens=700,
        )
        assistant_text = resp.choices[0].message.content
        tool_call = _extract_tool_call(assistant_text)

        if tool_call and isinstance(tool_call, dict) and "name" in tool_call:
            fn_name = tool_call.get("name")
            args    = _normalize_args(tool_call.get("arguments", {}))
            fn      = FUNC_MAP.get(fn_name)
            if not fn:
                prompt.append({"role": "assistant", "content": assistant_text})
                prompt.append({"role": "tool", "content": f"<tool_response>\nFunction {fn_name} not found.\n</tool_response>"})
                depth += 1
                continue
            try:
                raw_result = fn(**args)
            except Exception as exc:
                raw_result = json.dumps({"error": str(exc)})
            # 保證結果是 JSON 可解析字串
            try:
                result_json = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
            except Exception:
                result_json = {"raw": raw_result}
            markdown_preview = "\n".join(
                [f"- **{k}**: {v}" for k, v in result_json.items()]
            )
            tool_response_block = (
                "<tool_response>\n" + json.dumps(result_json, ensure_ascii=False) + "\n</tool_response>\n" + markdown_preview
            )
            prompt.append({"role": "assistant", "content": assistant_text})
            prompt.append({"role": "tool", "content": tool_response_block})
            depth += 1
            continue
        else:
            break


    # 加一條系統訊息引導最終摘要
    prompt.insert(1, {"role": "system", "content": SUMMARIZER_SYSTEM})

    final = client_chat.chat.completions.create(
        model=CHAT_MODEL,
        messages=prompt,
        max_tokens=600,
    )
    final_text = final.choices[0].message.content
    prompt.append({"role": "assistant", "content": final_text})

    return {"assistant": final_text, "history": [m for m in prompt if m["role"] != "system"]}
