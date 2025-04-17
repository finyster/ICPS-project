from fastapi import APIRouter
from config import IN_CLUSTER

router = APIRouter()

@router.get("/health")
def health_check():
    """
    健康檢查
    """
    return {"status": "ok", "in_cluster": IN_CLUSTER}
