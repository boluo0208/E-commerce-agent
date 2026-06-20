import asyncio
import json
import re
from pathlib import Path
from uuid import uuid4

from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.schemas.product import ProductExportRow, VisionResult
from app.services.deepseek_service import generate_content_with_deepseek
from app.services.export_service import create_export_zip, export_to_excel
from app.services.image_translate import (
    ModuleConfig,
    PipelineInput,
    run_pipeline,
)
from app.services.image_translate.processor import (
    resize_with_white_background,
    split_composite_image,
)
from app.services.mimo_vision_service import analyze_product_image_with_mimo

router = APIRouter()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
INVALID_FILENAME_CHARS = r'<>:"/\|?*'


def _safe_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG, and WEBP images are supported.",
        )
    return extension


def _safe_download_name(title: str) -> str:
    cleaned = "".join("_" if char in INVALID_FILENAME_CHARS else char for char in title.strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = "product_export"
    return cleaned[:80]


def _combine_vision_results(vision_results: list[VisionResult]) -> VisionResult:
    first = vision_results[0]
    categories = sorted({item.category for item in vision_results if item.category and item.category != "unknown"})
    colors = sorted({item.color for item in vision_results if item.color and item.color != "unknown"})
    styles = sorted({item.style for item in vision_results if item.style and item.style != "unknown"})
    visible_features: list[str] = []

    for index, result in enumerate(vision_results, start=1):
        prefix = f"Image {index}"
        if result.visible_features:
            visible_features.extend(f"{prefix}: {feature}" for feature in result.visible_features)
        else:
            visible_features.append(f"{prefix}: no extra visible features returned")

    return VisionResult(
        category=", ".join(categories) if categories else first.category,
        color=", ".join(colors) if colors else first.color,
        style=", ".join(styles) if styles else first.style,
        visible_features=visible_features,
        image_width=first.image_width,
        image_height=first.image_height,
        note=f"Combined vision result from {len(vision_results)} image(s) of the same product.",
    )


# Words that are almost certainly background, skin, or accessory colors — not
# the main product color.  Filtered conservatively so real product colors like
# "Black" or "White" are kept when they are the *only* signal.
_NON_PRODUCT_COLOR_WORDS: set[str] = {
    "skin", "nude", "flesh", "tan",
}


def _compute_overall_color(vision_results: list[VisionResult]) -> str:
    """Aggregate product color across multiple images of the same product.

    Deduplicates, removes ``"unknown"``, filters obvious non-product color
    words (background / skin-tone), and joins the rest with ``/``.
    """
    all_parts: list[str] = []
    for vr in vision_results:
        raw = (vr.color or "").strip()
        if not raw or raw.lower() == "unknown":
            continue
        # Split on common delimiters.
        for part in re.split(r"[,/、]", raw):
            part = part.strip()
            if not part or part.lower() == "unknown":
                continue
            if part.lower() in _NON_PRODUCT_COLOR_WORDS:
                continue
            all_parts.append(part)

    if not all_parts:
        # Fall back to raw colors, even if they might include unknowns.
        fallback: list[str] = []
        for vr in vision_results:
            raw = (vr.color or "").strip()
            if raw and raw.lower() != "unknown":
                fallback.append(raw)
        return "/".join(dict.fromkeys(fallback)) if fallback else "unknown"

    # Deduplicate case-insensitively, preserving first-seen order.
    seen: set[str] = set()
    unique: list[str] = []
    for part in all_parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            unique.append(part)

    return "/".join(unique)


def _compute_color_label(vision_results: list[VisionResult]) -> str:
    """Return the darkest product colour across all images."""
    # Ordered from darkest → lightest.
    _DARKNESS_ORDER: dict[str, int] = {
        "黑色": 0, "紫色": 1, "蓝色": 2, "棕色": 3,
        "绿色": 4, "红色": 5, "灰色": 6, "银色": 7,
        "橙色": 8, "金色": 9, "粉色": 10, "黄色": 11,
        "米色": 12, "白色": 13,
    }
    best_rank = 999
    best_label = "未知"
    for vr in vision_results:
        label = (vr.color_label or "").strip()
        rank = _DARKNESS_ORDER.get(label, 999)
        if rank < best_rank:
            best_rank = rank
            best_label = label
    return best_label


def _compute_title_anchored_color_label(vision_results: list[VisionResult]) -> str:
    """Return one normalized product color label from title-anchored vision results."""
    darkness_order: dict[str, int] = {
        "黑色": 0, "紫色": 1, "蓝色": 2, "棕色": 3,
        "绿色": 4, "红色": 5, "灰色": 6, "银色": 7,
        "橙色": 8, "金色": 9, "粉色": 10, "黄色": 11,
        "米色": 12, "白色": 13,
    }
    best_rank = 999
    best_label = "unknown"
    for vision_result in vision_results:
        label = (vision_result.color_label or "").strip()
        rank = darkness_order.get(label, 999)
        if rank < best_rank:
            best_rank = rank
            best_label = label
    return best_label


def _build_image_translate_config() -> ModuleConfig:
    """Map the main app's Settings to the image-translate module's config."""
    return ModuleConfig(
        translate_api_key=settings.deepseek_api_key,
        translate_base_url=settings.deepseek_base_url,
        translate_model=settings.deepseek_model,
        mimo_api_key=settings.mimo_api_key,
        mimo_base_url=settings.mimo_base_url,
        mimo_model=settings.mimo_model,
        ark_api_key=settings.ark_api_key,
        ark_base_url=settings.ark_base_url,
        ark_seedream_model=settings.ark_seedream_model,
        use_seedream_erase=settings.use_seedream_erase,
        seedream_watermark=settings.seedream_watermark,
        seedream_timeout=settings.seedream_timeout,
        mock_translate_when_no_key=settings.mock_llm_when_no_key,
        mock_vision_when_no_key=settings.mock_vision_when_no_key,
        translate_image_text=settings.translate_image_text,
        translate_image_text_min_confidence=settings.translate_image_text_min_confidence,
        auto_split_composite=settings.auto_split_composite_images,
        resize_enabled=True,
        resize_target_size=(660, 900),
    )


@router.post("/generate")
async def generate_product_export(
    chinese_title: str = Form(..., min_length=1),
    images: list[UploadFile] = File(...),
) -> FileResponse:
    if not images:
        raise HTTPException(status_code=400, detail="Please upload at least one image.")

    job_id = uuid4().hex
    upload_dir = settings.upload_dir / job_id
    job_output_dir = settings.output_dir / job_id
    processed_image_dir = job_output_dir / "images"
    excel_path = job_output_dir / "products.xlsx"
    download_name = _safe_download_name(chinese_title)
    zip_path = job_output_dir / f"{download_name}.zip"

    upload_dir.mkdir(parents=True, exist_ok=True)
    processed_image_dir.mkdir(parents=True, exist_ok=True)

    uploaded_images: list[tuple[Path, Path]] = []
    processed_image_paths_extra: list[Path] = []
    split_upload_dir = upload_dir / "splits"
    translated_upload_dir = upload_dir / "translated"

    # ---- build module config -------------------------------------------------
    module_config = _build_image_translate_config()

    # ---- debug accumulator --------------------------------------------------
    debug_images: list[dict] = []

    for index, image in enumerate(images, start=1):
        extension = _safe_extension(image.filename or "")
        image_id = f"product_{index:03d}"
        upload_path = upload_dir / f"{image_id}{extension}"

        upload_path.write_bytes(await image.read())

        if settings.auto_split_composite_images:
            split_paths = split_composite_image(upload_path, split_upload_dir / image_id)
        else:
            split_paths = [upload_path]

        for split_path in split_paths:
            output_index = len(uploaded_images) + 1
            processed_image_path = processed_image_dir / f"product_{output_index:03d}.jpg"
            translated_path = translated_upload_dir / f"product_{output_index:03d}.jpg"

            # --- new pipeline: OCR → merge → translate → erase → redraw ------
            pipeline_input = PipelineInput(
                image_path=split_path,
                output_path=translated_path,
                apply_resize=False,
                apply_split=False,
            )
            pipeline_result = await run_pipeline(pipeline_input, module_config)
            translated_source_path = pipeline_result.processed_path
            ocr_regions = [r.to_dict() for r in pipeline_result.regions]

            with Image.open(split_path) as pil_split:
                split_w, split_h = pil_split.size

            debug_images.append({
                "image_id": f"product_{output_index:03d}",
                "source_upload": f"product_{index:03d}{extension}",
                "is_from_split": len(split_paths) > 1,
                "split_source": upload_path.name if len(split_paths) > 1 else None,
                "original_dimensions": {"width": split_w, "height": split_h},
                "ocr_regions": ocr_regions,
                "mimo_vision": None,   # filled after vision analysis
                "processed_dimensions": None,  # filled after resize
            })

            # ---- collect seedream intermediate if present -----------------
            seedream_candidate = translated_source_path.parent / f"{translated_source_path.stem}_seedream_clean.jpg"
            if seedream_candidate.exists():
                seedream_dest = processed_image_dir / seedream_candidate.name
                seedream_dest.write_bytes(seedream_candidate.read_bytes())
                processed_image_paths_extra.append(seedream_dest)

            uploaded_images.append((translated_source_path, processed_image_path))

    semaphore = asyncio.Semaphore(settings.max_concurrent_images)

    async def process_one(
        idx: int,
        upload_path: Path,
        processed_image_path: Path,
    ) -> VisionResult:
        async with semaphore:
            vision_result = await analyze_product_image_with_mimo(
                upload_path,
                product_title=chinese_title,
            )

        await asyncio.to_thread(
            resize_with_white_background,
            upload_path,
            processed_image_path,
        )
        # Record dimensions after processing.
        with Image.open(processed_image_path) as pil_processed:
            pw, ph = pil_processed.size
        debug_images[idx]["processed_dimensions"] = {"width": pw, "height": ph}
        debug_images[idx]["mimo_vision"] = vision_result.model_dump()

        return vision_result

    vision_results = await asyncio.gather(
        *(process_one(i, up, pip) for i, (up, pip) in enumerate(uploaded_images))
    )
    processed_image_paths = [processed_image_path for _, processed_image_path in uploaded_images]
    processed_image_paths += processed_image_paths_extra
    image_files = [f"images/{processed_image_path.name}" for processed_image_path in processed_image_paths]
    combined_vision_result = _combine_vision_results(vision_results)
    overall_color = _compute_title_anchored_color_label(vision_results)
    content = await generate_content_with_deepseek(chinese_title, combined_vision_result)

    rows = [
        ProductExportRow(
            chinese_title=chinese_title,
            english_title=content.english_title,
            arabic_title=content.arabic_title,
            overall_color=overall_color,
            chinese_description=content.chinese_description,
            english_description=content.english_description,
            image_file="; ".join(image_files),
        )
    ]

    # ---- debug.json ---------------------------------------------------------
    debug_path = job_output_dir / "debug.json"
    debug_payload = {
        "job_id": job_id,
        "chinese_title": chinese_title,
        "original_image_count": len(images),
        "images_after_split": len(uploaded_images),
        "overall_color": overall_color,
        "images": debug_images,
        "combined_vision_result": combined_vision_result.model_dump(),
        "deepseek_response": content.model_dump(),
    }
    debug_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    export_to_excel(rows, excel_path)
    create_export_zip(excel_path, processed_image_paths, zip_path, extra_files=[debug_path])

    return FileResponse(
        path=zip_path,
        filename=f"{download_name}.zip",
        media_type="application/zip",
    )
