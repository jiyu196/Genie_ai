# app/main.py
from fastapi import FastAPI
from app.api.v1.api_image import router as image_router

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ai_backbone_python.log", encoding="utf-8")
    ]
)

app = FastAPI(
    title="Webtoon AI Backbone",
    version="1.0.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

# ✅ app 생성 이후에 include_router
app.include_router(
    image_router,
    prefix="/api/v1",
    tags=["image"]
)

# 🔍 디버그용: 라우트 출력
for r in app.routes:
    print(f"[ROUTE] {r.path} -> {r.name}")
