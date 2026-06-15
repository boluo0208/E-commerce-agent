"""Main pipeline: OCR → translate → erase → redraw.

This is the top-level orchestrator.  It ties together ocr.py, translator.py,
renderer.py, and (optionally) the Mimo vision fallback for Chinese-text
detection when RapidOCR misses characters.
"""

import asyncio
from pathlib import Path

from PIL import Image, ImageOps

from .config import ModuleConfig, get_config
from .ocr import (
    classify_text_role,
    contains_chinese,
    get_bbox_rect,
    get_ocr_threshold,
    is_compact_label,
    is_invalid_ocr_text,
    is_large_title,
    merge_ocr_lines_into_paragraphs,
    run_ocr_on_image,
    scale_box_back,
    _normalized_bbox_to_bounds,
)
from .renderer import (
    detect_original_text_color,
    erase_and_draw_translations,
    erase_and_draw_merged_paragraphs,
    estimate_original_font_size,
    estimate_surrounding_background,
)
from .schemas import (
    MergedParagraph,
    ModuleError,
    OCRLine,
    PipelineInput,
    PipelineOutput,
    RegionDebug,
    SplitInfo,
)
from .translator import translate_texts_to_english


# ---------------------------------------------------------------------------
# Mimo helpers (optional – only used when Mimo credentials are configured)
# ---------------------------------------------------------------------------


def _mimo_is_configured(config: ModuleConfig) -> bool:
    return bool(config.mimo_api_key and config.mimo_base_url and config.mimo_model)


async def _mimo_detect_chinese_regions(
    image_path: Path,
    config: ModuleConfig,
) -> list[dict]:
    """Ask Mimo to find Chinese-text bounding boxes on the full image."""
    import base64
    import json

    import httpx

    suffix = image_path.suffix.lower().lstrip(".")
    media_type = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    data_url = f"data:image/{media_type};base64,{encoded}"

    url = f"{config.mimo_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": config.mimo_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Find visible Chinese text regions in this product image. "
                    "Return JSON only: "
                    '{"items":[{"text":"Chinese text","bbox":[left,top,right,bottom]}]} '
                    "bbox values must be normalized coordinates from 0 to 1. "
                    "Only include text printed on the image. Skip model names, English text, logos, and uncertain text. "
                    "If there is no Chinese text, return {\"items\":[]}."
                )},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {config.mimo_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=config.mimo_timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
    except httpx.HTTPError:
        return []

    try:
        data = json.loads(resp.json()["choices"][0]["message"]["content"])
    except (KeyError, json.JSONDecodeError):
        return []

    items = data.get("items")
    return items if isinstance(items, list) else []


async def _mimo_recognize_region(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    config: ModuleConfig,
) -> str:
    """Crop *image* to *bbox* and ask Mimo to read any Chinese text inside."""
    import base64
    import io
    import json

    import httpx

    left, top, right, bottom = bbox
    if right <= left or bottom <= top:
        return ""

    crop = image.crop((left, top, right, bottom))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=95)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")

    url = f"{config.mimo_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": config.mimo_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Read the Chinese text visible in this image region. "
                    "Return valid JSON only: {\"text\": \"the Chinese text\"}. "
                    "If there is no Chinese text, return {\"text\": \"\"}."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ],
        }],
        "temperature": 0.05,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {config.mimo_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
    except httpx.HTTPError:
        return ""

    try:
        data = json.loads(resp.json()["choices"][0]["message"]["content"])
        return str(data.get("text") or "").strip()
    except (KeyError, json.JSONDecodeError):
        return ""


# ---------------------------------------------------------------------------
# region-debug helpers
# ---------------------------------------------------------------------------


def _make_skip_rd(
    original_text: str,
    left: int, top: int, right: int, bottom: int,
    confidence: float,
    threshold: float,
    bg_luminance: float | None = None,
    detected_by: str = "rapidocr",
    skip_reason: str = "",
) -> RegionDebug:
    return RegionDebug(
        original_text=original_text,
        bbox=[left, top, right, bottom],
        confidence=round(confidence, 3),
        confidence_threshold=threshold,
        estimated_original_font_size=estimate_original_font_size((left, top, right, bottom)),
        background_luminance=round(bg_luminance, 1) if bg_luminance is not None else 0.0,
        detected_by=detected_by,
        skip_reason=skip_reason,
    )


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(
    input_: PipelineInput,
    config: ModuleConfig | None = None,
) -> PipelineOutput:
    """Run the full image-translate pipeline on a single image.

    1. Open image.
    2. OCR on 3× upscaled image → collect Chinese-text candidates.
    3. Optional Mimo fallback for missed regions.
    4. Batch-translate all Chinese texts via LLM.
    5. Erase Chinese regions and redraw English text.
    6. Save output JPEG.

    Args:
        input_: PipelineInput with image_path, output_path, and optional overrides.
        config: ModuleConfig.  Uses ``get_config()`` when omitted.

    Returns:
        PipelineOutput with the processed path, region debug list, and errors.
    """
    cfg = config or get_config()
    region_debug: list[RegionDebug] = []
    errors: list[str] = []

    if not cfg.translate_image_text:
        return PipelineOutput(
            processed_path=input_.image_path,
            regions=[],
            errors=[],
            original_size=(0, 0),
            processed_size=(0, 0),
        )

    # --- open original -------------------------------------------------------
    try:
        with Image.open(input_.image_path) as source:
            original_image = ImageOps.exif_transpose(source).convert("RGB")
    except Exception as exc:
        return PipelineOutput(
            processed_path=input_.image_path,
            regions=[],
            errors=[f"Cannot open image: {exc}"],
        )

    img_w, img_h = original_image.size

    # --- OCR on upscaled image -----------------------------------------------
    ocr_items = run_ocr_on_image(original_image, scale_factor=cfg.ocr_scale_factor)

    # --- collect candidates with metadata -----------------------------------
    candidates: list[tuple[tuple[int, int, int, int], str]] = []
    # Per-candidate metadata (parallel lists).
    cand_font_sizes: list[int] = []
    cand_bg_luminances: list[float] = []
    cand_is_light_bgs: list[bool] = []
    cand_confidences: list[float] = []
    cand_detected_bys: list[str] = []

    ocr_has_chinese = False
    ocr_has_large_garbled = False
    # Deferred Mimo tasks for regions where OCR found a bbox but garbled text.
    mimo_tasks: list[tuple[tuple[int, int, int, int], asyncio.Task]] = []

    for box, text, confidence in ocr_items:
        original_box = scale_box_back(box, cfg.ocr_scale_factor)
        bbox = get_bbox_rect(original_box)
        left, top, right, bottom = bbox
        if right <= left or bottom <= top:
            continue

        threshold = get_ocr_threshold(original_box, img_w, img_h)
        est_font = estimate_original_font_size(bbox)

        if contains_chinese(text):
            ocr_has_chinese = True

        # ---- low confidence -------------------------------------------------
        if confidence < threshold:
            region_debug.append(_make_skip_rd(
                text, left, top, right, bottom,
                confidence, threshold,
                skip_reason=f"low_confidence:{confidence:.3f}",
            ))
            continue

        # ---- invalid / no Chinese → maybe Mimo on this region ---------------
        if is_invalid_ocr_text(text) or not contains_chinese(text):
            is_large = is_large_title(bbox, img_w, img_h)
            if is_large and is_invalid_ocr_text(text):
                ocr_has_large_garbled = True

            padded = (
                max(0, left - 4), max(0, top - 4),
                min(img_w, right + 4), min(img_h, bottom + 4),
            )
            bg_info = estimate_surrounding_background(original_image, padded)
            if not bg_info["is_light"]:
                region_debug.append(_make_skip_rd(
                    text, left, top, right, bottom,
                    confidence, threshold,
                    bg_luminance=bg_info["mean_luminance"],
                    skip_reason="background_not_light",
                ))
                continue

            if _mimo_is_configured(cfg):
                async def _mimo_task(_bbox: tuple, _padded: tuple):
                    mimo_text = await _mimo_recognize_region(original_image, _padded, cfg)
                    return (_bbox, _padded, mimo_text)
                loop = asyncio.get_running_loop()
                task = loop.create_task(_mimo_task(bbox, padded))
                mimo_tasks.append((bbox, task))
            else:
                region_debug.append(_make_skip_rd(
                    text, left, top, right, bottom,
                    confidence, threshold,
                    skip_reason="no_chinese_in_ocr",
                ))
            continue

        # ---- background check -----------------------------------------------
        padded = (
            max(0, left - 4), max(0, top - 4),
            min(img_w, right + 4), min(img_h, bottom + 4),
        )
        bg_info = estimate_surrounding_background(original_image, padded)
        if not bg_info["is_light"]:
            # Check if this is white text on a dark background (hero headline).
            tc_label, _, _ = detect_original_text_color(original_image, padded)
            is_white_on_dark = (tc_label == "white")
            if not is_white_on_dark:
                region_debug.append(_make_skip_rd(
                    text, left, top, right, bottom,
                    confidence, threshold,
                    bg_luminance=bg_info["mean_luminance"],
                    skip_reason="background_not_light",
                ))
                continue
            # White text on dark bg — accept but mark as dark background.
            # The renderer will handle white text styling.

        # ---- candidate accepted ---------------------------------------------
        candidates.append((padded, text))
        cand_font_sizes.append(est_font)
        cand_bg_luminances.append(round(bg_info["mean_luminance"], 1))
        cand_is_light_bgs.append(bg_info["is_light"])
        cand_confidences.append(confidence)
        cand_detected_bys.append("rapidocr")

    # --- resolve deferred Mimo tasks -----------------------------------------
    if mimo_tasks:
        await asyncio.gather(*(t for _, t in mimo_tasks), return_exceptions=True)
    for (orig_bbox, task) in mimo_tasks:
        try:
            result = task.result()
        except Exception:
            result = None
        if result is None:
            continue
        _b, padded, mimo_text = result
        if mimo_text and contains_chinese(mimo_text) and not is_invalid_ocr_text(mimo_text):
            candidates.append((padded, mimo_text))
            est_font = estimate_original_font_size(padded)
            cand_font_sizes.append(est_font)
            cand_bg_luminances.append(0.0)
            cand_is_light_bgs.append(True)
            cand_confidences.append(0.0)
            cand_detected_bys.append("mimo_crop")
            region_debug.append(RegionDebug(
                original_text=mimo_text,
                bbox=list(padded),
                detected_by="mimo_crop",
            ))
        else:
            region_debug.append(RegionDebug(
                original_text=mimo_text or "",
                bbox=list(padded),
                detected_by="mimo_crop",
                skip_reason="mimo_crop_no_valid_chinese",
            ))

    # --- Mimo full-image fallback --------------------------------------------
    should_use_mimo = (not ocr_has_chinese) or ocr_has_large_garbled
    if not candidates and should_use_mimo and _mimo_is_configured(cfg):
        mimo_items = await _mimo_detect_chinese_regions(input_.image_path, cfg)
        for item in mimo_items:
            text = str(item.get("text") or "").strip()
            if not contains_chinese(text):
                region_debug.append(RegionDebug(
                    original_text=text,
                    bbox=item.get("bbox"),
                    detected_by="mimo",
                    skip_reason="mimo_no_chinese",
                ))
                continue
            bounds = _normalized_bbox_to_bounds(item.get("bbox") or [], img_w, img_h)
            if bounds is None:
                region_debug.append(RegionDebug(
                    original_text=text,
                    bbox=item.get("bbox"),
                    detected_by="mimo",
                    skip_reason="mimo_invalid_bbox",
                ))
                continue
            left, top, right, bottom = bounds
            padded = (
                max(0, left - 4), max(0, top - 4),
                min(img_w, right + 4), min(img_h, bottom + 4),
            )
            bg_info = estimate_surrounding_background(original_image, padded)
            if not bg_info["is_light"]:
                region_debug.append(RegionDebug(
                    original_text=text,
                    bbox=[left, top, right, bottom],
                    estimated_original_font_size=estimate_original_font_size(bounds),
                    background_luminance=round(bg_info["mean_luminance"], 1),
                    detected_by="mimo",
                    skip_reason="mimo_background_not_light",
                ))
                continue
            candidates.append((padded, text))
            cand_font_sizes.append(estimate_original_font_size(bounds))
            cand_bg_luminances.append(round(bg_info["mean_luminance"], 1))
            cand_is_light_bgs.append(bg_info["is_light"])
            cand_confidences.append(0.0)
            cand_detected_bys.append("mimo")

    # --- no candidates → return original ------------------------------------
    if not candidates:
        return PipelineOutput(
            processed_path=input_.image_path,
            regions=region_debug,
            errors=errors,
            original_size=(img_w, img_h),
            processed_size=(img_w, img_h),
        )

    # --- sort candidates by y (top → bottom) for merging --------------------
    sorted_indices = sorted(range(len(candidates)), key=lambda i: candidates[i][0][1])
    candidates = [candidates[i] for i in sorted_indices]
    cand_font_sizes = [cand_font_sizes[i] for i in sorted_indices]
    cand_bg_luminances = [cand_bg_luminances[i] for i in sorted_indices]
    cand_is_light_bgs = [cand_is_light_bgs[i] for i in sorted_indices]
    cand_confidences = [cand_confidences[i] for i in sorted_indices]
    cand_detected_bys = [cand_detected_bys[i] for i in sorted_indices]

    # --- merge OCR lines → paragraphs (pre-translation step) ------------------
    merged_paragraphs, merge_debug_log = merge_ocr_lines_into_paragraphs(
        candidates,
        (img_w, img_h),
        font_sizes=cand_font_sizes,
        bg_luminances=cand_bg_luminances,
        is_light_bgs=cand_is_light_bgs,
        confidences=cand_confidences,
        detected_bys=cand_detected_bys,
    )

    # Emit merge debug info into region_debug as a structured note.
    for entry in merge_debug_log:
        if entry["action"] == "merge":
            region_debug.append(RegionDebug(
                original_text=f"[MERGE] {entry['line_count']} lines → 1 paragraph",
                full_translation=entry.get("merged_text", "")[:80],
                bbox=entry.get("merged_bbox", []),
                detected_by="merge_engine",
                skip_reason=f"reason: {entry.get('reason', '')}",
            ))

    # --- translate merged paragraphs -----------------------------------------
    # Detect text types for targeted translation.
    texts_to_translate = [p.merged_text for p in merged_paragraphs]
    text_types: list[str] = []
    _TITLE_FS = 36
    for p in merged_paragraphs:
        is_comp, _ = is_compact_label(p.merged_bbox, (img_w, img_h), p.merged_text)
        # Detect original text colour for classification.
        _tc_label, _, _ = detect_original_text_color(
            original_image, p.merged_bbox,
        )
        role = classify_text_role(
            p.merged_bbox, (img_w, img_h), p.merged_text,
            font_size=p.estimated_font_size,
            is_white_text=(_tc_label == "white"),
            is_compact=is_comp,
        )
        if role == "hero_headline":
            text_types.append("hero_headline")
        elif role == "bottom_hero_headline":
            text_types.append("bottom_hero_headline")
        elif is_comp:
            text_types.append("compact_label")
        elif p.estimated_font_size >= _TITLE_FS:
            text_types.append("title")
        else:
            text_types.append("body")

    try:
        translation_results = await translate_texts_to_english(
            texts_to_translate, cfg, text_types=text_types,
        )
    except Exception as exc:
        errors.append(f"Translation failed: {exc}")
        return PipelineOutput(
            processed_path=input_.image_path,
            regions=region_debug,
            errors=errors,
            original_size=(img_w, img_h),
            processed_size=(img_w, img_h),
        )

    # --- erase & redraw on original (using merged paragraphs) -----------------
    input_.output_path.parent.mkdir(parents=True, exist_ok=True)
    modified_image, draw_rd = erase_and_draw_merged_paragraphs(
        original_image,
        merged_paragraphs,
        translation_results,
        str(input_.output_path),
        jpeg_quality=cfg.output_jpeg_quality,
    )
    region_debug.extend(draw_rd)

    return PipelineOutput(
        processed_path=input_.output_path,
        regions=region_debug,
        errors=errors,
        original_size=(img_w, img_h),
        processed_size=modified_image.size,
    )


# ---------------------------------------------------------------------------
# convenience: process with pre-processing
# ---------------------------------------------------------------------------


async def process_image(
    image_path: Path,
    output_dir: Path,
    config: ModuleConfig | None = None,
    apply_resize: bool | None = None,
    apply_split: bool | None = None,
) -> list[PipelineOutput]:
    """Convenience wrapper: pre-process (split + resize) then run the pipeline.

    Returns a list of PipelineOutput — one per image after any splitting.
    """
    from .processor import preprocess_images

    cfg = config or get_config()
    output_dir.mkdir(parents=True, exist_ok=True)

    prepped = preprocess_images(
        image_path, output_dir, cfg,
        apply_split=apply_split,
        apply_resize=apply_resize,
    )

    results: list[PipelineOutput] = []
    for idx, (img_path, split_info) in enumerate(prepped, start=1):
        out_name = f"{image_path.stem}_{idx:02d}_translated.jpg"
        output_path = output_dir / out_name

        input_ = PipelineInput(
            image_path=img_path,
            output_path=output_path,
            apply_resize=False,   # already done in pre-process
            apply_split=False,    # already done in pre-process
        )
        result = await run_pipeline(input_, cfg)
        result.split_info = split_info
        results.append(result)

    return results
