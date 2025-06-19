# my-k8s-api/routers/grafana.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ..services.grafana_service import generate_k8s_dashboard_json

router = APIRouter(
    prefix="/api/grafana",
    tags=["grafana"],
)

@router.get("/dashboard")
async def get_k8s_dashboard():
    """
    Generates and returns a Grafana dashboard JSON for Kubernetes monitoring.
    """
    try:
        dashboard_json = generate_k8s_dashboard_json()
        return JSONResponse(content=dashboard_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard JSON: {e}")