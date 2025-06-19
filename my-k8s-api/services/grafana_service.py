# my-k8s-api/services/grafana_service.py
import json
import uuid

def generate_k8s_dashboard_json():
    """
    Generates a Grafana dashboard JSON for basic Kubernetes monitoring.
    Includes panels for CPU, Memory, Network Receive, and Network Transmit.
    """
    dashboard = {
        "apiVersion": 1,
        "title": "Dynamic Kubernetes Monitoring",
        "uid": "k8s-dynamic-default", # Unique identifier for the dashboard
        "timezone": "browser",
        "schemaVersion": 30,
        "version": 1,
        "refresh": "30s",
        "time": {
            "from": "now-1h",
            "to": "now"
        },
        "timepicker": {
            "refresh_intervals": [
                "5s",
                "10s",
                "30s",
                "1m",
                "5m",
                "15m",
                "30m",
                "1h",
                "2h",
                "1d"
            ],
            "time_options": [
                "5m",
                "15m",
                "1h",
                "6h",
                "12h",
                "24h",
                "2d",
                "7d",
                "30d"
            ]
        },
        "panels": [
            # CPU Usage Panel
            {
                "id": 1,
                "title": "Cluster CPU Usage",
                "type": "timeseries",
                "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                "datasource": {"type": "prometheus", "uid": "Prometheus"}, # Ensure this UID matches your Prometheus data source in Grafana
                "targets": [
                    {
                        "expr": "sum(rate(container_cpu_usage_seconds_total{image!=\"\"}[5m]))",
                        "legendFormat": "Total Cluster CPU",
                        "refId": "A"
                    }
                ],
                "options": {
                    "tooltip": {"mode": "single", "sort": "none"},
                    "legend": {"displayMode": "list", "placement": "bottom", "calcs": ["last"]},
                },
                "fieldConfig": {
                    "defaults": {
                        "unit": "short",
                        "custom": {}
                    },
                    "overrides": []
                }
            },
            # Memory Usage Panel
            {
                "id": 2,
                "title": "Cluster Memory Usage",
                "type": "timeseries",
                "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
                "datasource": {"type": "prometheus", "uid": "Prometheus"},
                "targets": [
                    {
                        "expr": "sum(container_memory_usage_bytes{image!=\"\"})",
                        "legendFormat": "Total Cluster Memory",
                        "refId": "A"
                    }
                ],
                 "options": {
                    "tooltip": {"mode": "single", "sort": "none"},
                    "legend": {"displayMode": "list", "placement": "bottom", "calcs": ["last"]},
                },
                 "fieldConfig": {
                    "defaults": {
                        "unit": "bytes",
                         "custom": {}
                    },
                    "overrides": []
                }
            },
             # Network Receive Panel
            {
                "id": 3,
                "title": "Cluster Network Receive",
                "type": "timeseries",
                "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
                "datasource": {"type": "prometheus", "uid": "Prometheus"},
                "targets": [
                    {
                        "expr": "sum(rate(container_network_receive_bytes_total{image!=\"\"}[5m]))",
                        "legendFormat": "Total Cluster Network Receive",
                        "refId": "A"
                    }
                ],
                 "options": {
                    "tooltip": {"mode": "single", "sort": "none"},
                    "legend": {"displayMode": "list", "placement": "bottom", "calcs": ["last"]},
                },
                 "fieldConfig": {
                    "defaults": {
                        "unit": "bps",
                         "custom": {}
                    },
                    "overrides": []
                }
            },
            # Network Transmit Panel
            {
                "id": 4,
                "title": "Cluster Network Transmit",
                "type": "timeseries",
                "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
                "datasource": {"type": "prometheus", "uid": "Prometheus"},
                "targets": [
                    {
                        "expr": "sum(rate(container_network_transmit_bytes_total{image!=\"\"}[5m]))",
                        "legendFormat": "Total Cluster Network Transmit",
                        "refId": "A"
                    }
                ],
                 "options": {
                    "tooltip": {"mode": "single", "sort": "none"},
                    "legend": {"displayMode": "list", "placement": "bottom", "calcs": ["last"]},
                },
                 "fieldConfig": {
                    "defaults": {
                        "unit": "bps",
                         "custom": {}
                    },
                    "overrides": []
                }
            }
        ],
        "templating": {
            "list": []
        },
        "annotations": {
            "list": []
        },
        "links": [],
        "tags": [],
        "variables": {}
    }
    return dashboard