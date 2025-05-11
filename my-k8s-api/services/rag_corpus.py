# services/rag_corpus_en.py
"""
English knowledge base for the Kubernetes‑Prometheus monitoring assistant.
Add / remove sentences here – no code changes needed.
"""

CORPUS_DOCS = [
    # ───────────── Supported queries ─────────────
    "The assistant supports: querying CPU, memory, disk and network usage "
    "for pods or namespaces; retrieving the top‑K resource‑hungry pods; "
    "showing trends for a given time window; and exporting CSV reports.",

    # ───────────── Forbidden queries ─────────────
    "The assistant must refuse requests about Deployments, Services, Jobs, "
    "Ingress resources or any non‑Pod Prometheus data. It must also refuse "
    "application‑level debugging or log inspection.",

    # ───────────── Time window rules ─────────────
    "Valid Prometheus range formats are 30m, 1h, 2d, 1mo. The maximum window "
    "is 10mo (ten months).",

    # ───────────── PromQL mappings ─────────────
    "PromQL mappings: CPU → container_cpu_usage_seconds_total; "
    "Memory → container_memory_usage_bytes; "
    "Disk → container_fs_usage_bytes; "
    "Network RX → container_network_receive_bytes_total; "
    "Network TX → container_network_transmit_bytes_total.",

    # ───────────── Ready state ─────────────
    "Pod readiness can be derived from the metric kube_pod_status_ready.",

    # ───────────── Example questions ─────────────
    "Example: \"Show the top 5 memory‑consuming pods in the monitoring "
    "namespace.\"",
    "Example: \"What was the CPU trend for nginx‑abc in the default namespace "
    "over the past 2 days?\"",
    "Example: \"Give me the aggregated resource usage of the default "
    "namespace during the last 1 hour.\""
]
