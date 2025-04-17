# 使用較小的 Python 基底映像
FROM python:3.9-slim

# 安裝系統套件 (若有需要再添加)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 如果需要一些系統工具, 例如:
    gcc \
    # ----
    && rm -rf /var/lib/apt/lists/*

# 切換工作目錄
WORKDIR /app

# 升級 pip
RUN pip install --no-cache-dir --upgrade pip

# 將 requirements.txt 複製到容器內
COPY requirements.txt .

# 安裝 Python 相依套件
RUN pip install --no-cache-dir -r requirements.txt

# 將專案所有檔案複製進容器 (可視情況優化 .dockerignore)
COPY . .

# 預設執行指令 (以 uvicorn 啟動 FastAPI，綁定 port 8000)
EXPOSE 8000
CMD ["python", "main.py"]
