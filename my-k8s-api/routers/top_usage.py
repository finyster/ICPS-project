#my-k8s-api/routers/top_usage.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from services.prometheus_utils import prometheus_instant_query

router = APIRouter()

@router.get("/top_cpu")
def get_top_cpu(namespace: str, k: int = 3):
    """
    查詢指定 Namespace 下，前 K 個 CPU 用量最高的 Pods。
    使用 rate(container_cpu_usage_seconds_total{namespace=}[5m]) 來計算。
    """
    query_pod = f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[5m]))'
    pod_result = prometheus_instant_query(query_pod)

    query_total = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[5m]))'
    total_result = prometheus_instant_query(query_total)

    total_usage = 0.0
    if total_result and len(total_result) > 0:
        _, val_str = total_result[0]["value"]
        total_usage = float(val_str)

    pod_usage_list = []
    if pod_result:
        for item in pod_result:
            metric = item.get("metric", {})
            value = item.get("value", [])
            pod_name = metric.get("pod", "N/A")
            if len(value) == 2:
                usage_cores = float(value[1])
                pod_usage_list.append((pod_name, usage_cores))

    pod_usage_list.sort(key=lambda x: x[1], reverse=True)
    top_k = pod_usage_list[:k]

    result_list = []
    for (pod_name, usage_cores) in top_k:
        percent = (usage_cores / total_usage * 100) if total_usage > 0 else 0
        result_list.append({
            "pod": pod_name,
            "cpu_usage_cores": usage_cores,
            "cpu_usage_percent": percent
        })

    return {
        "namespace": namespace,
        "top_k": result_list,
        "total_cpu_cores": total_usage
    }


@router.get("/top_memory")
def get_top_memory(namespace: str, k: int = 3):
    """
    查詢指定 Namespace 下，前 K 個 Memory 用量最高的 Pods。
    使用 container_memory_usage_bytes，並計算百分比。
    """
    query_pod = f'sum by (pod) (container_memory_usage_bytes{{namespace="{namespace}"}})'
    pod_result = prometheus_instant_query(query_pod)

    query_total = f'sum(container_memory_usage_bytes{{namespace="{namespace}"}})'
    total_result = prometheus_instant_query(query_total)

    total_usage = 0.0
    if total_result and len(total_result) > 0:
        _, val_str = total_result[0]["value"]
        total_usage = float(val_str)

    pod_usage_list = []
    if pod_result:
        for item in pod_result:
            metric = item.get("metric", {})
            value = item.get("value", [])
            pod_name = metric.get("pod", "N/A")
            if len(value) == 2:
                usage_bytes = float(value[1])
                pod_usage_list.append((pod_name, usage_bytes))

    pod_usage_list.sort(key=lambda x: x[1], reverse=True)
    top_k = pod_usage_list[:k]

    def bytes_to_human(n: float) -> str:
        if n < 1024:
            return f"{n:.2f} B"
        elif n < 1024**2:
            return f"{n / 1024:.2f} KB"
        elif n < 1024**3:
            return f"{n / 1024**2:.2f} MB"
        else:
            return f"{n / 1024**3:.2f} GB"

    result_list = []
    for (pod_name, usage_bytes) in top_k:
        percent = (usage_bytes / total_usage * 100) if total_usage > 0 else 0
        result_list.append({
            "pod": pod_name,
            "memory_bytes": usage_bytes,
            "memory_usage_percent": percent,
            "memory_human_readable": bytes_to_human(usage_bytes)
        })

    return {
        "namespace": namespace,
        "top_k": result_list,
        "total_memory_bytes": total_usage
    }
