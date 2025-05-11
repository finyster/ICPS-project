# my‑k8s‑api/routers/metrics.py
from fastapi import APIRouter, HTTPException, Query
from services.prometheus_utils import prometheus_instant_query   # 共用 util
from services.k8s_utils        import list_all_namespaces

router = APIRouter()


# ───── Namespace 下拉用 ──────────────────────────────────────────────
@router.get("/api/namespaces")
def list_namespaces():
    """動態抓 K8s namespace（過濾系統 ns）"""
    return {"namespaces": list_all_namespaces()}


# ───── 主要表格：一次回傳 CPU / Mem / Disk / Net / Ready ────────────
@router.get("/api/pod_metrics")
def pod_metrics(namespace: str = Query(...), k: int = 50):
    """
    回傳指定 namespace 內，前 k 個 (依 CPU) Pod 的：
        ‑ CPU (cores/sec)   5m rate
        ‑ Memory (MiB)      即時 usage
        ‑ Disk  (MiB)       container_fs_usage_bytes
        ‑ NetRx/Tx (KB/s)   5m rate
        ‑ Ready             kube_pod_status_ready{condition="true"}
    """
    try:
        #
        #  1️⃣  CPU & Mem – 先拿來決定要顯示哪幾個 Pod（取前 k 名 CPU）
        #
        q_cpu = f'rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[5m])'
        q_mem = f'container_memory_usage_bytes{{namespace="{namespace}"}}'
        cpu_r = prometheus_instant_query(f'sum by(pod) ({q_cpu})')
        mem_r = prometheus_instant_query(f'sum by(pod) ({q_mem})')

        pods = {}
        for i in cpu_r:
            pod = i["metric"]["pod"]; val=float(i["value"][1])
            pods[pod] = {"pod":pod, "cpu":val, "mem":0,"disk":0,
                         "net_rx":0,"net_tx":0,"ready":0}
        # 取前 k
        top = sorted(pods.values(), key=lambda p: p["cpu"], reverse=True)[:k]
        pod_set = {p["pod"] for p in top}

        #
        #  2️⃣  其它指標
        #
        def _fill(result, key, scale=1):
            for r in result:
                pod=r["metric"].get("pod")
                if pod in pods:
                    pods[pod][key]=float(r["value"][1])*scale

        # Mem‑bytes ➜ MiB
        _fill(mem_r, "mem", scale=1/1024/1024)

        # Disk usage (MiB)
        q_disk = f'sum by(pod)(container_fs_usage_bytes{{namespace="{namespace}"}})'
        _fill(prometheus_instant_query(q_disk), "disk", scale=1/1024/1024)

        # Network RX / TX (rate → KB/s)
        rate = '[5m]'
        _fill(prometheus_instant_query(
            f'sum by(pod)(rate(container_network_receive_bytes_total{{namespace="{namespace}"}}{rate}))'
        ), "net_rx", scale=1/1024)
        _fill(prometheus_instant_query(
            f'sum by(pod)(rate(container_network_transmit_bytes_total{{namespace="{namespace}"}}{rate}))'
        ), "net_tx", scale=1/1024)

        # Ready(0/1)
        _fill(prometheus_instant_query(
            f'max by(pod)(kube_pod_status_ready{{namespace="{namespace}",condition="true"}})'
        ), "ready")

        # 輸出
        rows=[pods[p] for p in pod_set]
        rows.sort(key=lambda p: p["cpu"], reverse=True)
        return rows

    except Exception as e:
        raise HTTPException(500, str(e))


# ───── Top‑K (CPU / Memory) 仍保留給舊前端呼叫 ──────────────────────
@router.get("/api/top_cpu")
def top_cpu(namespace: str, k: int = 3):
    q=f'sum by(pod)(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[5m]))'
    r=prometheus_instant_query(q)
    r.sort(key=lambda x:float(x["value"][1]), reverse=True)
    rows=[{"pod":i["metric"]["pod"],
           "value":float(i["value"][1])} for i in r[:k]]
    return rows

@router.get("/api/top_memory")
def top_memory(namespace: str, k: int = 3):
    q=f'sum by(pod)(container_memory_usage_bytes{{namespace="{namespace}"}})'
    r=prometheus_instant_query(q)
    r.sort(key=lambda x:float(x["value"][1]), reverse=True)
    rows=[{"pod":i["metric"]["pod"],
           "value":float(i["value"][1])} for i in r[:k]]
    return rows
