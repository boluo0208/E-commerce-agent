"""Lightweight FastAPI test shell for the image-translate module.

Run::

    cd test/image_translate_module
    pip install fastapi uvicorn python-multipart
    python api_server.py

Then open http://127.0.0.1:8001/docs for the interactive Swagger UI.

This is a **testing convenience** – the module itself has zero FastAPI dependency.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Support both `python api_server.py` and `python -m image_translate_module.api_server`.
try:
    from .config import ModuleConfig, get_config, set_config
    from .pipeline import process_image, run_pipeline
    from .schemas import PipelineInput
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import ModuleConfig, get_config, set_config
    from pipeline import process_image, run_pipeline
    from schemas import PipelineInput


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cfg = ModuleConfig.from_env()
    set_config(cfg)
    yield


app = FastAPI(
    title="Image Translate Module — Test Server",
    description="Standalone test API for OCR → translate → render pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "module": "image-translate"}


@app.post("/translate")
async def translate_image(
    image: UploadFile = File(...),
    resize: bool = Form(False),
    split_composite: bool = Form(False),
) -> JSONResponse:
    """Upload one product image; return the translated image + debug info."""
    ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}
    ext = Path(image.filename or "img.jpg").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported format: {ext}")

    job_id = uuid4().hex
    job_dir = Path("test_outputs") / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"input{ext}"
    output_path = job_dir / f"output.jpg"
    input_path.write_bytes(await image.read())

    cfg = get_config()
    cfg.resize_enabled = resize
    cfg.auto_split_composite = split_composite

    pipeline_input = PipelineInput(
        image_path=input_path,
        output_path=output_path,
        apply_resize=resize,
        apply_split=split_composite,
    )

    result = await run_pipeline(pipeline_input, cfg)

    return JSONResponse(content=result.to_dict())


@app.post("/translate-download")
async def translate_image_download(
    image: UploadFile = File(...),
    resize: bool = Form(False),
    split_composite: bool = Form(False),
) -> FileResponse:
    """Same as /translate, but returns the processed image file directly."""
    ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}
    ext = Path(image.filename or "img.jpg").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported format: {ext}")

    job_id = uuid4().hex
    job_dir = Path("test_outputs") / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"input{ext}"
    output_path = job_dir / f"output.jpg"
    input_path.write_bytes(await image.read())

    cfg = get_config()
    cfg.resize_enabled = resize
    cfg.auto_split_composite = split_composite

    pipeline_input = PipelineInput(
        image_path=input_path,
        output_path=output_path,
        apply_resize=resize,
        apply_split=split_composite,
    )

    result = await run_pipeline(pipeline_input, cfg)

    if not result.success:
        raise HTTPException(500, detail=result.errors)

    return FileResponse(
        path=result.processed_path,
        filename="translated.jpg",
        media_type="image/jpeg",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
