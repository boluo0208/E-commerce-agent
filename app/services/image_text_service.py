import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from app.core.config import settings
from app.services.deepseek_service import translate_texts_to_english_with_deepseek
from app.services.mimo_vision_service import detect_chinese_text_regions_with_mimo

CHINESE_PATTERN = re.compile(r"[一-鿿]")
FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/arialbd.ttf")
OCR_SCALE_FACTOR = 3


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def contains_chinese(text: str) -> bool:
    return bool(CHINESE_PATTERN.search(text or ""))


def is_invalid_ocr_text(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if set(cleaned) <= {"?", "？", "�", " "}:
        return True
    replacement_count = cleaned.count("�")
    question_count = cleaned.count("?") + cleaned.count("？")
    return replacement_count >= 1 or question_count >= max(2, len(cleaned) // 2)


@lru_cache
def _get_ocr():
    try:
        from rapidocr import RapidOCR
    except ImportError:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            return None
    return RapidOCR()


def _as_confidence(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_ocr_items(
    result: object,
) -> list[tuple[list[list[float]], str, float]]:
    if result is None:
        return []
    if hasattr(result, "boxes") and hasattr(result, "txts") and hasattr(result, "scores"):
        boxes = result.boxes if result.boxes is not None else []
        texts = result.txts if result.txts is not None else []
        scores = result.scores if result.scores is not None else []
        return [
            (
                box.tolist() if hasattr(box, "tolist") else box,
                str(text),
                _as_confidence(score),
            )
            for box, text, score in zip(boxes, texts, scores)
        ]
    if isinstance(result, tuple):
        result = result[0]
    items = []
    for item in result or []:
        if len(item) >= 3:
            box, text, confidence = item[:3]
            items.append((box, str(text), _as_confidence(confidence)))
    return items


def scale_box_back(
    box: list[list[float]],
    scale_factor: float,
) -> list[list[int]]:
    return [
        [round(point[0] / scale_factor), round(point[1] / scale_factor)]
        for point in box
    ]


def get_bbox_rect(box: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def get_ocr_threshold(
    box: list[list[int]],
    image_width: int,
    image_height: int,
) -> float:
    left, top, right, bottom = get_bbox_rect(box)
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)
    width_ratio = box_width / image_width
    height_ratio = box_height / image_height
    # Large title text — tolerate lower OCR confidence.
    if box_height >= 36 or width_ratio >= 0.18 or height_ratio >= 0.04:
        return 0.35
    return 0.60


def _normalized_bbox_to_bounds(
    bbox: list[object],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if max(left, top, right, bottom) <= 1.2:
        left, right = left * width, right * width
        top, bottom = top * height, bottom * height
    bounds = (
        max(0, int(left)),
        max(0, int(top)),
        min(width, int(right)),
        min(height, int(bottom)),
    )
    return bounds if bounds[2] > bounds[0] and bounds[3] > bounds[1] else None


def _is_large_title(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> bool:
    left, top, right, bottom = bbox
    h = bottom - top
    w = right - left
    return h >= 36 or (w / img_w) >= 0.18 or (h / img_h) >= 0.04


# ---------------------------------------------------------------------------
# background detection  (Section 5)
# ---------------------------------------------------------------------------


def estimate_surrounding_background(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    padding_ratio: float = 0.35,
) -> dict:
    """Sample the outer ring around *bbox*, not the text interior."""
    rgb = image.convert("RGB")
    array = np.asarray(rgb)
    image_height, image_width = array.shape[:2]
    left, top, right, bottom = bbox

    box_width = max(1, right - left)
    box_height = max(1, bottom - top)

    pad_x = max(4, round(box_width * padding_ratio))
    pad_y = max(4, round(box_height * padding_ratio))

    outer_left = max(0, left - pad_x)
    outer_top = max(0, top - pad_y)
    outer_right = min(image_width, right + pad_x)
    outer_bottom = min(image_height, bottom + pad_y)

    outer = array[outer_top:outer_bottom, outer_left:outer_right]

    if outer.size == 0:
        return {"is_light": True, "mean_luminance": 255.0}

    mask = np.ones(outer.shape[:2], dtype=bool)

    inner_left = max(0, left - outer_left)
    inner_top = max(0, top - outer_top)
    inner_right = min(outer.shape[1], right - outer_left)
    inner_bottom = min(outer.shape[0], bottom - outer_top)

    mask[inner_top:inner_bottom, inner_left:inner_right] = False

    ring_pixels = outer[mask]

    if ring_pixels.size == 0:
        ring_pixels = outer.reshape(-1, 3)

    luminance = (
        0.2126 * ring_pixels[:, 0]
        + 0.7152 * ring_pixels[:, 1]
        + 0.0722 * ring_pixels[:, 2]
    )

    mean_luminance = float(np.median(luminance))

    return {
        "is_light": mean_luminance >= 170,
        "mean_luminance": mean_luminance,
    }


# ---------------------------------------------------------------------------
# font / text sizing  (Sections 6, 7, 9)
# ---------------------------------------------------------------------------


def estimate_original_font_size(bbox: tuple[int, int, int, int]) -> int:
    left, top, right, bottom = bbox
    box_height = max(1, bottom - top)
    # Chinese glyph height ≈ 82 % of bbox height.
    estimated_size = round(box_height * 0.82)
    return max(10, estimated_size)


def estimate_final_text_height(
    original_text_height: int,
    original_image_size: tuple[int, int],
    target_size: tuple[int, int] = (660, 900),
) -> float:
    original_width, original_height = original_image_size
    target_width, target_height = target_size
    scale = min(target_width / original_width, target_height / original_height)
    return original_text_height * scale


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int = 2,
) -> list[str] | None:
    words = text.split()
    if not words:
        return None

    lines: list[str] = []
    current_line = ""

    for word in words:
        candidate = word if not current_line else f"{current_line} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        candidate_width = bbox[2] - bbox[0]

        if candidate_width <= max_width:
            current_line = candidate
            continue

        if current_line:
            lines.append(current_line)
            current_line = word
        else:
            return None

        if len(lines) >= max_lines:
            return None

    if current_line:
        lines.append(current_line)

    if len(lines) > max_lines:
        return None

    return lines


# ---------------------------------------------------------------------------
# debug record builder
# ---------------------------------------------------------------------------


def _make_debug(
    *,
    original_text: str = "",
    full_translation: str = "",
    image_translation: str = "",
    bbox: list[int] | None = None,
    confidence: float = 0,
    confidence_threshold: float = 0,
    estimated_original_font_size: int = 0,
    used_font_size: int = 0,
    final_estimated_text_height: float = 0,
    background_luminance: float = 0,
    text_color: str = "",
    detected_by: str = "paddleocr",
    translated: bool = False,
    replaced: bool = False,
    skip_reason: str | None = None,
) -> dict:
    return {
        "original_text": original_text,
        "full_translation": full_translation,
        "image_translation": image_translation,
        "bbox": bbox or [],
        "confidence": confidence,
        "confidence_threshold": confidence_threshold,
        "estimated_original_font_size": estimated_original_font_size,
        "used_font_size": used_font_size,
        "final_estimated_text_height": final_estimated_text_height,
        "background_luminance": background_luminance,
        "text_color": text_color,
        "detected_by": detected_by,
        "translated": translated,
        "replaced": replaced,
        "skip_reason": skip_reason,
    }


# ---------------------------------------------------------------------------
# main pipeline  (Sections 1-9)
# ---------------------------------------------------------------------------


async def translate_chinese_text_on_image(
    image_path: Path,
    output_path: Path,
) -> tuple[Path, list[dict]]:
    """Detect Chinese text, translate, erase and redraw on the ORIGINAL image.

    Processing order:
    1. OCR on 3× upscaled image → map bboxes back to original.
    2. Erase Chinese on the original high-res image.
    3. Draw English on the original high-res image.
    4. Save JPEG **once** (quality=95, subsampling=0).

    Returns ``(output_path, region_debug_list)``.
    """
    region_debug: list[dict] = []

    if not settings.translate_image_text:
        return image_path, region_debug

    ocr = _get_ocr()
    if ocr is None:
        return image_path, region_debug

    # --- open original -------------------------------------------------------
    with Image.open(image_path) as source:
        original_image = ImageOps.exif_transpose(source).convert("RGB")
    img_w, img_h = original_image.size

    # --- OCR on 3× upscaled image -------------------------------------------
    ocr_image = original_image.resize(
        (img_w * OCR_SCALE_FACTOR, img_h * OCR_SCALE_FACTOR),
        Image.Resampling.LANCZOS,
    )
    ocr_result = ocr(np.array(ocr_image))
    ocr_items = _extract_ocr_items(ocr_result)

    # --- collect candidates --------------------------------------------------
    candidates: list[tuple[tuple[int, int, int, int], str]] = []
    ocr_has_chinese = False
    ocr_has_large_garbled = False

    for box, text, confidence in ocr_items:
        original_box = scale_box_back(box, OCR_SCALE_FACTOR)
        bbox = get_bbox_rect(original_box)
        left, top, right, bottom = bbox

        if right <= left or bottom <= top:
            continue

        threshold = get_ocr_threshold(original_box, img_w, img_h)

        if contains_chinese(text):
            ocr_has_chinese = True

        # ---- low confidence -------------------------------------------------
        if confidence < threshold:
            region_debug.append(_make_debug(
                original_text=text,
                bbox=[left, top, right, bottom],
                confidence=round(confidence, 3),
                confidence_threshold=threshold,
                estimated_original_font_size=estimate_original_font_size(bbox),
                skip_reason=f"low_confidence:{confidence:.3f}",
            ))
            continue

        # ---- invalid / no Chinese -------------------------------------------
        if is_invalid_ocr_text(text) or not contains_chinese(text):
            is_large = _is_large_title(bbox, img_w, img_h)
            if is_large and is_invalid_ocr_text(text):
                ocr_has_large_garbled = True
            region_debug.append(_make_debug(
                original_text=text,
                bbox=[left, top, right, bottom],
                confidence=round(confidence, 3),
                confidence_threshold=threshold,
                estimated_original_font_size=estimate_original_font_size(bbox),
                skip_reason="no_chinese_or_invalid_ocr",
            ))
            continue

        # ---- background check -----------------------------------------------
        padded = (
            max(0, left - 4), max(0, top - 4),
            min(img_w, right + 4), min(img_h, bottom + 4),
        )
        bg_info = estimate_surrounding_background(original_image, padded)
        if not bg_info["is_light"]:
            region_debug.append(_make_debug(
                original_text=text,
                bbox=[left, top, right, bottom],
                confidence=round(confidence, 3),
                confidence_threshold=threshold,
                estimated_original_font_size=estimate_original_font_size(bbox),
                background_luminance=round(bg_info["mean_luminance"], 1),
                skip_reason="background_not_light",
            ))
            continue

        candidates.append((padded, text))

    # --- Mimo fallback -------------------------------------------------------
    should_use_mimo = (not ocr_has_chinese) or ocr_has_large_garbled
    if not candidates and should_use_mimo:
        mimo_items = await detect_chinese_text_regions_with_mimo(image_path)
        for item in mimo_items:
            text = str(item.get("text") or "").strip()
            if not contains_chinese(text):
                region_debug.append(_make_debug(
                    original_text=text,
                    bbox=item.get("bbox"),
                    detected_by="mimo",
                    skip_reason="mimo_no_chinese",
                ))
                continue

            bounds = _normalized_bbox_to_bounds(
                item.get("bbox") or [], img_w, img_h,
            )
            if bounds is None:
                region_debug.append(_make_debug(
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
                region_debug.append(_make_debug(
                    original_text=text,
                    bbox=[left, top, right, bottom],
                    estimated_original_font_size=estimate_original_font_size(bounds),
                    background_luminance=round(bg_info["mean_luminance"], 1),
                    detected_by="mimo",
                    skip_reason="mimo_background_not_light",
                ))
                continue

            candidates.append((padded, text))

    # --- no candidates → return original ------------------------------------
    if not candidates:
        return image_path, region_debug

    # --- translate -----------------------------------------------------------
    translation_results = await translate_texts_to_english_with_deepseek(
        [text for _, text in candidates]
    )

    # --- erase & redraw on original ------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draw = ImageDraw.Draw(original_image)
    font_path = str(FONT_PATH) if FONT_PATH.exists() else None

    for idx, ((left, top, right, bottom), original_text) in enumerate(candidates):
        trans = translation_results[idx] if idx < len(translation_results) else {}
        full_trans = trans.get("full_translation", original_text)
        image_trans = trans.get("image_translation", full_trans)

        text_height = bottom - top
        final_h = estimate_final_text_height(text_height, (img_w, img_h))
        bbox = (left, top, right, bottom)
        orig_font_size = estimate_original_font_size(bbox)
        bg_info = estimate_surrounding_background(original_image, bbox)

        # --- too small at final output? --------------------------------------
        if final_h < 14:
            region_debug.append(_make_debug(
                original_text=original_text,
                full_translation=full_trans,
                image_translation=image_trans,
                bbox=[left, top, right, bottom],
                estimated_original_font_size=orig_font_size,
                final_estimated_text_height=round(final_h, 1),
                background_luminance=round(bg_info["mean_luminance"], 1),
                text_color="black" if bg_info["is_light"] else "white",
                translated=True,
                replaced=False,
                skip_reason="final_text_too_small",
            ))
            continue

        # --- erase colour ----------------------------------------------------
        if bg_info["is_light"]:
            text_color = (20, 20, 20)
            erase_color = "white"
        else:
            text_color = (255, 255, 255)
            erase_color = (20, 20, 20)

        draw.rectangle((left, top, right, bottom), fill=erase_color)

        is_large_title = _is_large_title(bbox, img_w, img_h)
        draw_text = full_trans if is_large_title else image_trans

        # --- typesetting (Section 7) -----------------------------------------
        box_w = max(10, right - left - 4)
        box_h = max(10, bottom - top - 4)
        expanded_box_w = round((right - left) * (1.35 if is_large_title else 1.2))
        min_font_size = max(10, round(orig_font_size * (0.72 if is_large_title else 0.85)))

        lines_to_draw: list[str] | None = None
        final_font_size = 0

        def _make_font(size: int) -> ImageFont.FreeTypeFont:
            selected_font = FONT_BOLD_PATH if is_large_title and FONT_BOLD_PATH.exists() else FONT_PATH
            font_file = str(selected_font) if selected_font.exists() else None
            return (
                ImageFont.truetype(font_file, size=size)
                if font_file
                else ImageFont.load_default()
            )

        # 1) original size, single line
        font = _make_font(orig_font_size)
        test_w = draw.textbbox((0, 0), draw_text, font=font)[2]
        if test_w <= box_w:
            lines_to_draw = [draw_text]
            final_font_size = orig_font_size

        # 2) original size, two lines
        if lines_to_draw is None:
            lines = wrap_text_to_width(draw, draw_text, font, box_w, max_lines=2)
            if lines is not None:
                lines_to_draw = lines
                final_font_size = orig_font_size

        # 3) expand width up to 20 %
        if lines_to_draw is None:
            test_w = draw.textbbox((0, 0), draw_text, font=font)[2]
            if test_w <= expanded_box_w:
                lines_to_draw = [draw_text]
                final_font_size = orig_font_size
            else:
                lines = wrap_text_to_width(draw, draw_text, font, expanded_box_w, max_lines=2)
                if lines is not None:
                    lines_to_draw = lines
                    final_font_size = orig_font_size

        # 4) shrink font (max → 85 %)
        if lines_to_draw is None:
            for fs in range(orig_font_size - 1, min_font_size - 1, -1):
                font = _make_font(fs)
                test_w = draw.textbbox((0, 0), draw_text, font=font)[2]
                if test_w <= box_w:
                    lines_to_draw = [draw_text]
                    final_font_size = fs
                    break
                lines = wrap_text_to_width(draw, draw_text, font, box_w, max_lines=2)
                if lines is not None:
                    lines_to_draw = lines
                    final_font_size = fs
                    break

        # 5) give up
        if lines_to_draw is None:
            region_debug.append(_make_debug(
                original_text=original_text,
                full_translation=full_trans,
                image_translation=image_trans,
                bbox=[left, top, right, bottom],
                estimated_original_font_size=orig_font_size,
                final_estimated_text_height=round(final_h, 1),
                background_luminance=round(bg_info["mean_luminance"], 1),
                text_color="black" if bg_info["is_light"] else "white",
                translated=True,
                replaced=False,
                skip_reason="text_too_long_for_box",
            ))
            continue

        # --- draw ------------------------------------------------------------
        font = _make_font(final_font_size)
        font_bbox = draw.textbbox((0, 0), "Ag", font=font)
        line_height = max(1, round((font_bbox[3] - font_bbox[1]) * 1.15))
        total_text_height = line_height * len(lines_to_draw)
        y_start = top + max(2, (box_h + 4 - total_text_height) // 2)

        for li, line in enumerate(lines_to_draw):
            line_w = draw.textbbox((0, 0), line, font=font)[2]
            x = left + 2 + max(0, (box_w - line_w) // 2)
            draw.text((x, y_start + li * line_height), line, fill=text_color, font=font)

        region_debug.append(_make_debug(
            original_text=original_text,
            full_translation=full_trans,
            image_translation=image_trans,
            bbox=[left, top, right, bottom],
            estimated_original_font_size=orig_font_size,
            used_font_size=final_font_size,
            final_estimated_text_height=round(final_h, 1),
            background_luminance=round(bg_info["mean_luminance"], 1),
            text_color="black" if bg_info["is_light"] else "white",
            translated=True,
            replaced=True,
            skip_reason=None,
        ))

    # --- save JPEG once ------------------------------------------------------
    original_image.save(
        output_path,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=True,
    )
    return output_path, region_debug
