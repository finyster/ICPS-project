from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")
router = APIRouter()

@router.get("/chat", response_class=HTMLResponse, tags=["chat"])
def chat_page(request: Request):
    """
    聊天 UI 頁面 (GET /chat)
    """
    return templates.TemplateResponse("chat.html", {"request": request})
