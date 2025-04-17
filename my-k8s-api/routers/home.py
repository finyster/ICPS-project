from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

# 初始化 Jinja2 模板路徑
templates = Jinja2Templates(directory="templates")


def generate_index_html(request: Request):
    """
    使用 Jinja2 渲染 index.html 模板。
    """
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    """
    GET / -> 回傳主頁 HTML
    """
    return generate_index_html(request)
