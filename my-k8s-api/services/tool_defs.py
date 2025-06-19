TOOLS_DEF = [   
    {
        "type": "function", #alan52254增加三個新的tools
        "function": {
            "name": "list_node_names",
            "description": "List all node names in the Kubernetes cluster.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_namespace_names",
            "description": "List all namespace names in the Kubernetes cluster.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_pod_names",
            "description": "List all pod names in a specific namespace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Kubernetes namespace"}
                },
                "required": ["namespace"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pod_cpu_usage",
            "description": "Get CPU usage for a specific pod in a namespace",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "pod": {"type": "string"},
                    "range_str": {"type": "string", "default": "[1h]"},
                },
                "required": ["namespace", "pod"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pod_memory_usage",
            "description": "Get memory usage for a specific pod in a namespace",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "pod": {"type": "string"},
                    "range_str": {"type": "string", "default": "[1h]"},
                },
                "required": ["namespace", "pod"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_cpu_pods",
            "description": "Top‑K pods by CPU usage in a namespace",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "k": {"type": "integer", "default": 3},
                },
                "required": ["namespace"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_memory_pods",
            "description": "Top‑K pods by memory usage in a namespace",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "k": {"type": "integer", "default": 3},
                },
                "required": ["namespace"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pod_resource_usage_over_time",
            "description": "Get CPU and memory usage for a pod over the past N days",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "pod": {"type": "string"},
                    "days": {"type": "integer"}
                },
                "required": ["namespace", "pod", "days"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_namespace_resource_usage_over_time",
            "description": "Get resource usage for all pods in a namespace over time",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "days": {"type": "integer", "default": 7},
                },
                "required": ["namespace"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_resource",
            "description": "Query CPU or memory usage of a pod, node, or namespace. Most fields are optional.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["pod", "node", "namespace"],
                        "description": "Query level. Defaults to 'pod'."
                    },
                    "target": {
                        "type": "string",
                        "description": "The resource name (e.g., pod name, node name)."
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["cpu", "memory"],
                        "description": "Metric to query. Defaults to 'cpu'."
                    },
                    "duration": {
                        "type": "string",
                        "description": "Time range like 1h, 2d, or natural language like 'past 2 hours'."
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Only required for pod-level queries. Can also be inferred from target='namespace/pod'."
                    }
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "top_k_pods",
            "description": "Return top‑K pods by CPU/Memory in a namespace",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Kubernetes namespace"},
                    "metric": {"type": "string", "enum": ["cpu", "memory"]},
                    "k": {"type": "integer", "default": 3, "minimum": 1, "description": "How many pods to return"},
                    "duration": {"type": "string", "default": "5m", "pattern": "^(\\d+)(m|h|d|mo)$", "description": "Look‑back window like 30m, 2h, 1d"}
                },
                "required": ["namespace", "metric"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_csv_link",
            "description": "Generate a CSV download link for a specific pod's CPU/Memory usage. Use context from conversation history to infer namespace, pod, or range if not specified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Kubernetes namespace"},
                    "pod": {"type": "string", "description": "Pod name"},
                    "range": {"type": "string", "default": "[1h]", "description": "Time range like [1h], [2d]"}
                },
                "required": ["namespace", "pod"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_pod_cpu_usage",
            "description": "Predict future CPU usage for a specific pod in a given namespace based on historical data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Kubernetes namespace of the target pod (e.g., 'default')"
                    },
                    "pod": {
                        "type": "string",
                        "description": "Name of the pod to predict CPU usage for"
                    },
                    "duration": {
                        "type": "string",
                        "default": "1h",
                        "description": "Historical data range to train the model, e.g., '1h', '2h', '1d'"
                    },
                    "step": {
                        "type": "string",
                        "default": "5m",
                        "description": "Sampling interval (data granularity), e.g., '5m', '1m'"
                    },
                    "future_duration": {
                        "type": "string",
                        "default": "1h",
                        "description": "Duration of future CPU usage prediction, e.g., '1h', '2h'"
                    }
                },
                "required": ["namespace", "pod"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_hpa_for_deployment",
            "parameters": {
                "properties": {
                    "namespace": {"type": "string"},
                    "deployment": {"type": "string"},
                    "metric": {"type": "string", "enum": ["cpu", "memory"], "default": "cpu"},
                    "min_replicas": {"type": "integer", "default": 1},
                    "max_replicas": {"type": "integer", "default": 5},
                    "target_utilization": {"type": "integer", "default": 60}
                },
                "required": ["namespace", "deployment"]
            }
        }
    }
]
