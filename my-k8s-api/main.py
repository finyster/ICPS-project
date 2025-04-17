# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routers.home        import router as home_router
from routers.namespace   import router as namespace_router
from routers.pod         import router as pod_router
from routers.export_csv  import router as export_csv_router
from routers.top_usage   import router as top_usage_router
from routers.health      import router as health_router
from routers.llm_chat    import router as llm_chat_router
from routers.chat_ui     import router as chat_ui_router

app = FastAPI(title="K8s Pod Metrics API")

app.include_router(home_router,        tags=["home"])
app.include_router(namespace_router,   prefix="/api", tags=["namespaces"])
app.include_router(pod_router,         prefix="/api", tags=["pods"])
app.include_router(export_csv_router,  prefix="/api", tags=["export"])
app.include_router(top_usage_router,   prefix="/api", tags=["top"])
app.include_router(health_router,      tags=["health"])
app.include_router(llm_chat_router,    prefix="/api", tags=["llm"])
app.include_router(chat_ui_router)

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
