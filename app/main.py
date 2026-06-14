import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.product import router as product_router
from app.core.config import settings

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

app.include_router(product_router, prefix="/api/products", tags=["products"])


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


# Serve built frontend in production (Docker).  In dev the Vite server handles it.
_candidates = [
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"),
    "/app/frontend/dist",
]
_frontend_dist = next((p for p in _candidates if os.path.isdir(p)), None)
if _frontend_dist:
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
