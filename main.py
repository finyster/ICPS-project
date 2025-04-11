from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import csv
import io
import os
import requests
from collections import defaultdict

# (A) 若需要 K8s 原生操作，保留你的 kube config 程式
try:
    from kubernetes import client, config
    try:
        config.load_incluster_config()
        in_cluster = True
    except:
        config.load_kube_config()
        in_cluster = False
    v1 = client.CoreV1Api()
    HAS_K8S = True
except ImportError:
    # 沒裝 kubernetes 套件就跳過
    in_cluster = False
    v1 = None
    HAS_K8S = False


# --------------------------------------------------------------------
# FastAPI 應用程式主體
# --------------------------------------------------------------------
app = FastAPI(title="K8s Pod Metrics API")

# --------------------------------------------------------------------
# 1. Prometheus URL 請改成你可連線的真實位址 (環境變數 PROMETHEUS_URL)
# --------------------------------------------------------------------
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")


# --------------------------------------------------------------------
# 工具函式：單次 Prometheus Instant Query
# --------------------------------------------------------------------
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


# --------------------------------------------------------------------
# 2. 從 Prometheus 以「range vector」抓取資料的函式 (新增可動態調整 range_str)
# --------------------------------------------------------------------
def get_prometheus_metric_for_pod(metric_name: str, namespace: str, pod: str, range_str: str = "[1h]"):
    """
    針對單一 Pod，抓取 metric_name{namespace="xxx", pod="yyy"}[range_str] 的數據。
    這裡直接用 requests.get() 呼叫 /api/v1/query (instant query 但抓 range vector)。
    回傳形如:
      [
        {
          "metric": {...},
          "values": [[<timestamp>, <value>], [<timestamp>, <value>], ...]
        },
        ...
      ]
    range_str 預設為 "[1h]"，可以像 "[5m]"、"[1h]"、"[2d]" 等。
    """
    # 注意：這裡的 query 直接在 instant query 模式下帶入 [range]，Prometheus 也會回傳 range vector。
    query = f'{metric_name}{{namespace="{namespace}", pod="{pod}"}}{range_str}'
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},  # instant query，但抓 range vector
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


# --------------------------------------------------------------------
# 3. 前端主頁 (Bootstrap)，加入可選擇 Time Range
# --------------------------------------------------------------------
def generate_index_html() -> str:
    """
    產生主頁的 HTML，讓使用者能選擇 Namespace 與 Pod，並選擇時間區間 (range) 後下載 CSV。
    同時包含查詢前 K 名 CPU / Memory 的功能。
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8"/>
        <title>K8s Pod Metrics</title>
        <!-- Bootstrap CSS -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .table-container {
                margin-top: 1rem;
                overflow-x: auto;
            }
        </style>
    </head>
    <body>
        <div class="container my-4">
            <h1 class="text-center mb-4">K8s Pod Metrics Export</h1>

            <!-- Namespace/Pod 選擇區 -->
            <div class="row g-3 mb-4">
                <div class="col-12 col-sm-4">
                    <label for="namespaceSelect" class="form-label fw-bold">Namespace</label>
                    <select id="namespaceSelect" class="form-select" onchange="loadPods()">
                    </select>
                </div>
                <div class="col-12 col-sm-4">
                    <label for="podSelect" class="form-label fw-bold">Pod</label>
                    <select id="podSelect" class="form-select">
                    </select>
                </div>
                <div class="col-12 col-sm-4">
                    <label for="timeRangeSelect" class="form-label fw-bold">Time Range</label>
                    <!-- 預設給幾個常用選項，例如 5m, 1h, 2d 等 -->
                    <select id="timeRangeSelect" class="form-select">
                        <option value="5m">Last 5 min</option>
                        <option value="1h" selected>Last 1 hour</option>
                        <option value="1d">Last 1 day</option>
                        <option value="2d">Last 2 days</option>
                    </select>
                </div>
            </div>

            <!-- 下載 CSV 按鈕 -->
            <div class="mb-4">
                <button class="btn btn-primary" onclick="downloadCSV()">Download CSV</button>
            </div>

            <hr/>

            <!-- 查詢前 K 名 CPU 區域 -->
            <div class="row g-3 mb-4 align-items-end">
                <div class="col-auto">
                    <label for="topKCpu" class="form-label fw-bold">Top K (CPU)</label>
                    <input type="number" class="form-control" id="topKCpu" value="3" min="1" style="width:6em;">
                </div>
                <div class="col-auto">
                    <button class="btn btn-success" onclick="loadTopCpu()">Get Top CPU Pods</button>
                </div>
            </div>

            <!-- 顯示前 K 名 CPU 結果的表格 -->
            <div class="table-container">
                <table class="table table-sm table-bordered table-striped" id="cpuResultTable" style="display:none;">
                    <thead class="table-light">
                        <tr>
                            <th scope="col">#</th>
                            <th scope="col">Pod</th>
                            <th scope="col">CPU Usage (cores)</th>
                            <th scope="col">CPU Usage (%)</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>

            <hr/>

            <!-- 查詢前 K 名 Memory 區域 -->
            <div class="row g-3 mb-4 align-items-end">
                <div class="col-auto">
                    <label for="topKMem" class="form-label fw-bold">Top K (Memory)</label>
                    <input type="number" class="form-control" id="topKMem" value="3" min="1" style="width:6em;">
                </div>
                <div class="col-auto">
                    <button class="btn btn-info" onclick="loadTopMemory()">Get Top Memory Pods</button>
                </div>
            </div>

            <!-- 顯示前 K 名 Memory 結果的表格 -->
            <div class="table-container">
                <table class="table table-sm table-bordered table-striped" id="memResultTable" style="display:none;">
                    <thead class="table-light">
                        <tr>
                            <th scope="col">#</th>
                            <th scope="col">Pod</th>
                            <th scope="col">Memory Usage</th>
                            <th scope="col">Memory Usage (%)</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>

        <!-- Bootstrap Bundle JS -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

        <script>
            /**
             * 載入所有可用的 Namespace，並顯示在下拉選單中
             */
            async function loadNamespaces() {
                try {
                    let res = await fetch('/api/namespaces');
                    let data = await res.json();
                    let nsSelect = document.getElementById('namespaceSelect');
                    nsSelect.innerHTML = '';
                    data.namespaces.forEach(ns => {
                        let opt = document.createElement('option');
                        opt.value = ns;
                        opt.text = ns;
                        nsSelect.appendChild(opt);
                    });
                    // 預設載入後馬上載入該 namespace 下的 pods
                    loadPods();
                } catch (err) {
                    alert('Failed to load namespaces. Check backend or RBAC.');
                }
            }

            /**
             * 根據所選的 namespace，載入該 namespace 的所有 Pods
             */
            async function loadPods() {
                let ns = document.getElementById('namespaceSelect').value;
                if (!ns) return;
                try {
                    let res = await fetch('/api/pods?namespace=' + ns);
                    let data = await res.json();
                    let podSelect = document.getElementById('podSelect');
                    podSelect.innerHTML = '';
                    data.pods.forEach(pod => {
                        let opt = document.createElement('option');
                        opt.value = pod;
                        opt.text = pod;
                        podSelect.appendChild(opt);
                    });
                } catch (err) {
                    alert('Failed to load pods. Check backend or RBAC.');
                }
            }

            /**
             * 下載指定 Pod 的 CSV，時間區間由下拉選單決定 (e.g. [5m], [1h], [2d])
             */
            function downloadCSV() {
                let ns = document.getElementById('namespaceSelect').value;
                let pod = document.getElementById('podSelect').value;
                let rangeValue = document.getElementById('timeRangeSelect').value; // e.g. "5m", "1h"
                if (!ns || !pod) {
                    alert('Please select Namespace and Pod first.');
                    return;
                }
                // 將 rangeValue 包成 Prometheus 需要的 "[5m]"、"[1h]" 等
                let rangeParam = "[" + rangeValue + "]";
                // 導向後端的 /api/export_csv?namespace=xxx&pod=xxx&range=[5m]...
                let url = `/api/export_csv?namespace=${ns}&pod=${pod}&range=${encodeURIComponent(rangeParam)}`;
                window.location.href = url;
            }

            /**
             * 取得指定 Namespace 下前 K 名 CPU 用量最高的 Pods
             */
            async function loadTopCpu() {
                let ns = document.getElementById('namespaceSelect').value;
                let k = document.getElementById('topKCpu').value;
                if (!ns) {
                    alert('Please select a Namespace first.');
                    return;
                }
                if (!k || parseInt(k) < 1) {
                    alert('Top K value must be >= 1');
                    return;
                }
                try {
                    let res = await fetch(`/api/top_cpu?namespace=${ns}&k=${k}`);
                    let data = await res.json();
                    renderCpuTable(data.top_k);
                } catch (err) {
                    alert('Failed to load top CPU usage data.');
                }
            }

            /**
             * 將後端傳回的前 K 名 CPU 用量資料渲染到表格中
             */
            function renderCpuTable(list) {
                let table = document.getElementById('cpuResultTable');
                let tbody = table.querySelector('tbody');
                tbody.innerHTML = ''; // 清空

                table.style.display = list.length > 0 ? '' : 'none';

                list.forEach((item, idx) => {
                    let tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${idx + 1}</td>
                        <td>${item.pod}</td>
                        <td>${item.cpu_usage_cores.toFixed(4)}</td>
                        <td>${item.cpu_usage_percent.toFixed(2)}%</td>
                    `;
                    tbody.appendChild(tr);
                });
            }

            /**
             * 取得指定 Namespace 下前 K 名 Memory 用量最高的 Pods
             */
            async function loadTopMemory() {
                let ns = document.getElementById('namespaceSelect').value;
                let k = document.getElementById('topKMem').value;
                if (!ns) {
                    alert('Please select a Namespace first.');
                    return;
                }
                if (!k || parseInt(k) < 1) {
                    alert('Top K value must be >= 1');
                    return;
                }
                try {
                    let res = await fetch(`/api/top_memory?namespace=${ns}&k=${k}`);
                    let data = await res.json();
                    renderMemTable(data.top_k);
                } catch (err) {
                    alert('Failed to load top Memory usage data.');
                }
            }

            /**
             * 將後端傳回的前 K 名 Memory 用量資料渲染到表格中
             */
            function renderMemTable(list) {
                let table = document.getElementById('memResultTable');
                let tbody = table.querySelector('tbody');
                tbody.innerHTML = ''; // 清空

                table.style.display = list.length > 0 ? '' : 'none';

                list.forEach((item, idx) => {
                    let tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${idx + 1}</td>
                        <td>${item.pod}</td>
                        <td>${item.memory_human_readable}</td>
                        <td>${item.memory_usage_percent.toFixed(2)}%</td>
                    `;
                    tbody.appendChild(tr);
                });
            }

            // 預設頁面載入後，先呼叫 loadNamespaces() 初始化
            window.onload = function() {
                loadNamespaces();
            }
        </script>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def index_page():
    """
    根路徑 ("/") 直接回傳主頁面，讓使用者可在瀏覽器看到 Bootstrap 美化後的介面。
    """
    return generate_index_html()


# --------------------------------------------------------------------
# 4. 列出 Namespace
# --------------------------------------------------------------------
@app.get("/api/namespaces")
def list_namespaces():
    """
    從 Kubernetes 抓取所有可用的 Namespace，過濾掉 kube- 與 openshift- 開頭的系統 Namespace。
    若未安裝 kubernetes 套件，就只回傳 ["default"]。
    """
    if not HAS_K8S:
        return JSONResponse({"namespaces": ["default"]})

    try:
        all_ns = v1.list_namespace()
        ns_names = []
        for item in all_ns.items:
            name = item.metadata.name
            # 過濾掉 kube- 與 openshift- 開頭
            if not name.startswith("kube-") and not name.startswith("openshift-"):
                ns_names.append(name)
        if "default" not in ns_names:
            ns_names.append("default")
        return JSONResponse({"namespaces": ns_names})
    except Exception as e:
        return JSONResponse({"error": str(e), "namespaces": []}, status_code=500)


# --------------------------------------------------------------------
# 5. 列出 Pod
# --------------------------------------------------------------------
@app.get("/api/pods")
def list_pods(namespace: str):
    """
    列出指定 Namespace 中所有 "Running" 狀態的 Pod。
    若未安裝 kubernetes 套件，就回傳一些假資料做示範。
    """
    if not HAS_K8S:
        # 如果沒安裝 kubernetes 套件，就假裝回傳幾個Pod
        return JSONResponse({"pods": ["dummy-pod-1", "dummy-pod-2"]})

    try:
        pod_list = v1.list_namespaced_pod(namespace)
        pod_names = [
            p.metadata.name
            for p in pod_list.items
            if p.status.phase == "Running"  # 只要 Running
        ]
        return JSONResponse({"pods": pod_names})
    except Exception as e:
        return JSONResponse({"error": str(e), "pods": []}, status_code=500)


# --------------------------------------------------------------------
# 6. 匯出「指定單一 Pod」在指定 range (預設 [1h]) 的 CPU/Memory CSV
# --------------------------------------------------------------------
@app.get("/api/export_csv")
def export_csv(namespace: str, pod: str, range: str = "[1h]"):
    """
    以 range vector range (e.g. [5m], [1h], [2d]) 抓取 container_cpu_usage_seconds_total、container_memory_usage_bytes。
    只針對特定 namespace + pod。
    然後合併寫成 CSV，並以 StreamingResponse 形式下載。
    range 參數預設為 [1h]，如需從前端指定，帶入 &range=[5m] 等即可。
    """
    cpu_metric = "container_cpu_usage_seconds_total"
    mem_metric = "container_memory_usage_bytes"

    cpu_data = get_prometheus_metric_for_pod(cpu_metric, namespace, pod, range)
    mem_data = get_prometheus_metric_for_pod(mem_metric, namespace, pod, range)

    # 以 (ns, pod, container, instance, timestamp) 為 key，來做匯整
    combined_data = defaultdict(lambda: {"cpu": 0.0, "memory": 0.0})

    # 處理 CPU
    for series in cpu_data:
        metric = series.get("metric", {})
        ns = metric.get("namespace", "N/A")
        podname = metric.get("pod", "N/A")
        container = metric.get("container", "N/A")
        instance = metric.get("instance", "N/A")

        if podname == "N/A":
            continue

        for ts_str, val_str in series.get("values", []):
            key = (ns, podname, container, instance, ts_str)
            combined_data[key]["cpu"] += float(val_str)

    # 處理 Memory
    for series in mem_data:
        metric = series.get("metric", {})
        ns = metric.get("namespace", "N/A")
        podname = metric.get("pod", "N/A")
        container = metric.get("container", "N/A")
        instance = metric.get("instance", "N/A")

        if podname == "N/A":
            continue

        for ts_str, val_str in series.get("values", []):
            key = (ns, podname, container, instance, ts_str)
            combined_data[key]["memory"] += float(val_str)

    # 寫到 in-memory CSV
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

    for (ns, podname, container, instance, ts_str), val_map in sorted(combined_data.items(),
                                                                     key=lambda x: float(x[0][4])):
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
    # 檔名可帶上 range，以利辨識
    filename = f"{namespace}-{pod}-metrics-{range.strip('[]')}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(output, media_type="text/csv", headers=headers)


# --------------------------------------------------------------------
# 7. 測試用路由：單獨測一個 Pod，將結果寫到本地 CSV (範例)
# --------------------------------------------------------------------
@app.get("/api/test_pod_csv")
def test_pod_csv(namespace: str, pod: str):
    """
    這支 API 和 /api/export_csv 邏輯類似，不同在於「直接寫到本地檔案」，
    再回傳 JSON 告知寫了幾筆。
    (此處固定 range = [1h])
    """
    cpu_metric = "container_cpu_usage_seconds_total"
    mem_metric = "container_memory_usage_bytes"

    cpu_data = get_prometheus_metric_for_pod(cpu_metric, namespace, pod, "[1h]")
    mem_data = get_prometheus_metric_for_pod(mem_metric, namespace, pod, "[1h]")

    combined_data = defaultdict(lambda: {"cpu": 0.0, "memory": 0.0})

    for series in cpu_data:
        metric = series.get("metric", {})
        ns = metric.get("namespace", "N/A")
        podname = metric.get("pod", "N/A")
        container = metric.get("container", "N/A")
        instance = metric.get("instance", "N/A")

        for ts_str, val_str in series.get("values", []):
            key = (ns, podname, container, instance, ts_str)
            combined_data[key]["cpu"] += float(val_str)

    for series in mem_data:
        metric = series.get("metric", {})
        ns = metric.get("namespace", "N/A")
        podname = metric.get("pod", "N/A")
        container = metric.get("container", "N/A")
        instance = metric.get("instance", "N/A")

        for ts_str, val_str in series.get("values", []):
            key = (ns, podname, container, instance, ts_str)
            combined_data[key]["memory"] += float(val_str)

    filename = f"test_{namespace}_{pod}_metrics.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "namespace",
            "pod",
            "container",
            "instance",
            "timestamp(UTC)",
            "cpu_usage_seconds",
            "memory_usage_bytes"
        ])
        for (ns, podname, container, instance, ts_str), val_map in sorted(combined_data.items(),
                                                                         key=lambda x: float(x[0][4])):
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

    return {
        "message": f"Metrics CSV for {namespace}/{pod} written to {filename}",
        "row_count": len(combined_data),
        "file": filename
    }


# --------------------------------------------------------------------
# 8. 新增：取得前 K 個 CPU 用量最高的 Pods (Instant Query)
# --------------------------------------------------------------------
@app.get("/api/top_cpu")
def get_top_cpu(namespace: str, k: int = 3):
    """
    查詢指定 Namespace 下，前 K 個 CPU 用量最高的 Pods。
    使用 rate(container_cpu_usage_seconds_total{namespace=}[5m]) 來計算。
    回傳每個 Pod 的 CPU 用量 (cores) 以及在該 Namespace 的佔比 (百分比)。
    """
    # 1) 查詢每個 Pod 的 CPU 用量
    #    sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="xxx"}[5m]))
    query_pod = f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[5m]))'
    pod_result = prometheus_instant_query(query_pod)

    # 2) 查詢整個 Namespace 的 CPU 總用量 (為了算百分比)
    #    sum(rate(container_cpu_usage_seconds_total{namespace="xxx"}[5m]))
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

    # 依 CPU 用量排序，取前 K 名
    pod_usage_list.sort(key=lambda x: x[1], reverse=True)
    top_k = pod_usage_list[:k]

    # 計算百分比
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


# --------------------------------------------------------------------
# 9. 新增：取得前 K 個 Memory 用量最高的 Pods (Instant Query)
# --------------------------------------------------------------------
@app.get("/api/top_memory")
def get_top_memory(namespace: str, k: int = 3):
    """
    查詢指定 Namespace 下，前 K 個 Memory 用量最高的 Pods。
    直接使用 container_memory_usage_bytes{namespace="xxx"} 即可。
    回傳每個 Pod 的用量 (bytes) 和其佔比 (百分比)。
    並將用量轉為比較好讀的單位 (e.g. KB / MB / GB)。
    """
    # 1) 查詢每個 Pod 的 Memory 用量
    query_pod = f'sum by (pod) (container_memory_usage_bytes{{namespace="{namespace}"}})'
    pod_result = prometheus_instant_query(query_pod)

    # 2) 查詢整個 Namespace 的 Memory 總用量
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

    # 排序後取前 K 名
    pod_usage_list.sort(key=lambda x: x[1], reverse=True)
    top_k = pod_usage_list[:k]

    # 計算百分比 & 轉換為可讀單位
    def bytes_to_human(n: float) -> str:
        """
        將 Bytes 轉成帶單位的字串。
        e.g. 12345678 -> '11.77 MB' (約略計法)
        """
        if n < 1024:
            return f"{n:.2f} B"
        elif n < 1024 * 1024:
            return f"{n / 1024:.2f} KB"
        elif n < 1024 * 1024 * 1024:
            return f"{n / (1024*1024):.2f} MB"
        else:
            return f"{n / (1024*1024*1024):.2f} GB"

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


# --------------------------------------------------------------------
# 10. Health Check
# --------------------------------------------------------------------
@app.get("/health")
def health_check():
    """
    健康檢查，回傳當前服務狀態。
    in_cluster 表示是否在 K8s cluster 中載入 kube_config 成功。
    """
    return {"status": "ok", "in_cluster": in_cluster}


# --------------------------------------------------------------------
# 11. 若要直接啟動，使用 Uvicorn 伺服器
# --------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
