from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from services.k8s_utils import list_pods_in_namespace

router = APIRouter()

@router.get("/pods")
def list_pods(namespace: str):
    """
    以 JSON 回傳指定 namespace 下的 Running Pods
    """
    pods = list_pods_in_namespace(namespace)
    return JSONResponse({"pods": pods})
