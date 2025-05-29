# services/prom_tools.py

import requests
import json
import re
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import os

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

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

###########################test
def is_smalltalk(msg: str) -> bool:
    # 可以自己加強這個判斷
    return msg.lower().strip() in ["hello", "hi", "who are you", "早安", "你好"]
##############################

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
    

