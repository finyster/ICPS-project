import os

# Prometheus URL (預設給 localhost:9090)
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

# 若需要 K8s 原生操作，嘗試載入 kube config
try:
    from kubernetes import client, config
    try:
        config.load_incluster_config()
        IN_CLUSTER = True
    except:
        config.load_kube_config()
        IN_CLUSTER = False

    V1 = client.CoreV1Api()
    HAS_K8S = True
except ImportError:
    IN_CLUSTER = False
    V1 = None
    HAS_K8S = False
