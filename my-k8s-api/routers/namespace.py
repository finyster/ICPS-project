#my-k8s-api/routers/namespace.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from services.k8s_utils import list_all_namespaces

router = APIRouter()

@router.get("/namespaces")
def list_namespaces():
    """
    以 JSON 回傳可用的 namespaces
    """
    ns_list = list_all_namespaces()
    return JSONResponse({"namespaces": ns_list})
