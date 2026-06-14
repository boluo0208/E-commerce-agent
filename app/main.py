from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.product import router as product_router
from app.core.config import settings

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(product_router, prefix="/api/products", tags=["products"])


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
