import csv
import io
from datetime import datetime
from collections import defaultdict
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from services.prometheus_utils import get_prometheus_metric_for_pod

router = APIRouter()

@router.get("/export_csv")
def export_csv(namespace: str, pod: str, range: str = "[1h]"):
    """
    針對特定 namespace + pod，抓取 CPU/Memory (range vector)，合併後輸出 CSV。
    """
    cpu_metric = "container_cpu_usage_seconds_total"
    mem_metric = "container_memory_usage_bytes"

    cpu_data = get_prometheus_metric_for_pod(cpu_metric, namespace, pod, range)
    mem_data = get_prometheus_metric_for_pod(mem_metric, namespace, pod, range)

    combined_data = defaultdict(lambda: {"cpu": 0.0, "memory": 0.0})

    # 處理 CPU
    for series in cpu_data:
        metric = series.get("metric", {})
        podname = metric.get("pod", "N/A")
        if podname == "N/A":
            continue
        container = metric.get("container", "N/A")
        instance = metric.get("instance", "N/A")
        ns = metric.get("namespace", "N/A")

        for ts_str, val_str in series.get("values", []):
            key = (ns, podname, container, instance, ts_str)
            combined_data[key]["cpu"] += float(val_str)

    # 處理 Memory
    for series in mem_data:
        metric = series.get("metric", {})
        podname = metric.get("pod", "N/A")
        if podname == "N/A":
            continue
        container = metric.get("container", "N/A")
        instance = metric.get("instance", "N/A")
        ns = metric.get("namespace", "N/A")

        for ts_str, val_str in series.get("values", []):
            key = (ns, podname, container, instance, ts_str)
            combined_data[key]["memory"] += float(val_str)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "namespace",
        "pod",
        "container",
        "instance",
        "timestamp(UTC)",
        "cpu_usage_seconds",
        "memory_usage_bytes"
    ])

    for (ns, podname, container, instance, ts_str), val_map in sorted(combined_data.items(), key=lambda x: float(x[0][4])):
        ts_human = datetime.utcfromtimestamp(float(ts_str)).strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([
            ns,
            podname,
            container,
            instance,
            ts_human,
            val_map["cpu"],
            val_map["memory"]
        ])

    output.seek(0)
    filename = f"{namespace}-{pod}-metrics-{range.strip('[]')}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(output, media_type="text/csv", headers=headers)
