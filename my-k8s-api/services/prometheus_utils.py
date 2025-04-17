import requests
from config import PROMETHEUS_URL

def prometheus_instant_query(query: str):
    """
    使用 Prometheus 的 /api/v1/query 進行 Instant Query。
    例如:
        rate(container_cpu_usage_seconds_total{namespace="xxx"}[5m])
    回傳的資料格式通常為:
        {
          "status": "success",
          "data": {
            "resultType": "vector",
            "result": [
              {
                "metric": {...},
                "value": [<timestamp>, <value>]
              },
              ...
            ]
          }
        }
    """
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "success":
            return data["data"]["result"]
    except Exception as e:
        print(f"[ERROR] Prometheus query failed: {e}")
    return []


def get_prometheus_metric_for_pod(metric_name: str, namespace: str, pod: str, range_str: str = "[1h]"):
    """
    針對單一 Pod，抓取 metric_name{namespace="xxx", pod="yyy"}[range_str] 的數據 (range vector)。
    回傳形如:
      [
        {
          "metric": {...},
          "values": [[<timestamp>, <value>], [<timestamp>, <value>], ...]
        },
        ...
      ]
    """
    query = f'{metric_name}{{namespace="{namespace}", pod="{pod}"}}{range_str}'
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},  # 用 instant query 也可以帶 [range] 抓回 range vector
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ERROR] Prometheus query failed: {e}")
        return []

    if "data" in data and "result" in data["data"]:
        return data["data"]["result"]
    return []
