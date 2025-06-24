#my-k8s-api/services/k8s_utils.py
from fastapi.responses import JSONResponse
from config import V1, HAS_K8S

def list_all_namespaces():
    """
    從 Kubernetes 抓取所有可用的 Namespace，過濾掉 kube- 與 openshift- 開頭的系統 Namespace。
    若未安裝 kubernetes 套件，就只回傳 ["default"]。
    """
    if not HAS_K8S:
        return ["default"]

    try:
        all_ns = V1.list_namespace()
        ns_names = []
        for item in all_ns.items:
            name = item.metadata.name
            # 過濾掉 kube- 與 openshift- 開頭
            if not name.startswith("kube-") and not name.startswith("openshift-"):
                ns_names.append(name)
        if "default" not in ns_names:
            ns_names.append("default")
        return ns_names
    except Exception:
        # 真實場景中可打 log
        return []


def list_pods_in_namespace(namespace: str):
    """
    列出指定 Namespace 中所有 "Running" 狀態的 Pod。
    若未安裝 kubernetes 套件，就回傳假資料。
    """
    if not HAS_K8S:
        return ["dummy-pod-1", "dummy-pod-2"]

    try:
        pod_list = V1.list_namespaced_pod(namespace)
        pod_names = [
            p.metadata.name
            for p in pod_list.items
            if p.status.phase == "Running"  # 只要 Running
        ]
        return pod_names
    except Exception:
        # 真實場景中可打 log
        return []