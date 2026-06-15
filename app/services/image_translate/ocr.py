"""Local OCR via RapidOCR + Chinese-text detection helpers.

Extracted from app/services/image_text_service.py – zero coupling to the main app.
"""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

# Matches any CJK Unified Ideograph (U+4E00 – U+9FFF).
CHINESE_PATTERN = re.compile(r"[一-鿿]")
OCR_SCALE_FACTOR_DEFAULT = 3


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------


def contains_chinese(text: str) -> bool:
    """Return True when *text* contains at least one Chinese character."""
    return bool(CHINESE_PATTERN.search(text or ""))


def is_invalid_ocr_text(text: str) -> bool:
    """Heuristic to reject garbled OCR output."""
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if set(cleaned) <= {"?", "？", "�", " "}:
        return True
    replacement_count = cleaned.count("�")
    question_count = cleaned.count("?") + cleaned.count("？")
    return replacement_count >= 1 or question_count >= max(2, len(cleaned) // 2)


# ---------------------------------------------------------------------------
# OCR engine (lazy singleton)
# ---------------------------------------------------------------------------


@lru_cache
def _get_ocr():
    """Return a RapidOCR instance or None if the library is unavailable."""
    try:
        from rapidocr import RapidOCR
    except ImportError:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            return None
    return RapidOCR()


# ---------------------------------------------------------------------------
# OCR result parsing
# ---------------------------------------------------------------------------


def _as_confidence(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_ocr_items(
    result: object,
) -> list[tuple[list[list[float]], str, float]]:
    """Normalise RapidOCR result → [(box, text, confidence), ...]."""
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


def run_ocr_on_image(image: Image.Image, scale_factor: int = OCR_SCALE_FACTOR_DEFAULT):
    """Run RapidOCR on a 3× upscaled PIL image.  Returns parsed items."""
    ocr = _get_ocr()
    if ocr is None:
        return []
    w, h = image.size
    ocr_image = image.resize(
        (w * scale_factor, h * scale_factor),
        Image.Resampling.LANCZOS,
    )
    ocr_result = ocr(np.array(ocr_image))
    return _extract_ocr_items(ocr_result)


# ---------------------------------------------------------------------------
# bbox utilities
# ---------------------------------------------------------------------------


def scale_box_back(
    box: list[list[float]],
    scale_factor: float,
) -> list[list[int]]:
    return [
        [round(point[0] / scale_factor), round(point[1] / scale_factor)]
        for point in box
    ]


def get_bbox_rect(box: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
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
        left, top, right, bottom = [float(v) for v in bbox]
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


def is_large_title(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> bool:
    left, top, right, bottom = bbox
    h = bottom - top
    w = right - left
    return h >= 36 or (w / img_w) >= 0.18 or (h / img_h) >= 0.04


# ---------------------------------------------------------------------------
# OCR line → paragraph merging  (pre-translation step)
# ---------------------------------------------------------------------------

# Punctuation that indicates a sentence / clause break.
_SENTENCE_END = frozenset({"。", "！", "？", "；", "…", "）", "」", "》", "”", "　"})
_SENTENCE_PAUSE = frozenset({"，", "、", "：", "（", "「", "《", "“"})


def _text_ends_open(text: str) -> bool:
    """Return True when *text* does NOT end with a sentence-ending punctuation.

    A line ending with '。', '！', '？' etc. is less likely to continue
    into the next line.  A line ending with nothing or a pause (，、：…)
    is very likely to continue.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    last_char = stripped[-1]
    return last_char not in _SENTENCE_END


def _is_title_line(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    font_size: int,
    median_font_size: int,
    median_line_height: int = 0,
) -> bool:
    """Heuristic title detection for paragraph merging.

    A line is considered a title (and thus NOT merged with the next line) when:
    - Its font is significantly larger than the median (≥ 1.4×), OR
    - Its line height is ≥ 1.5× the median line height, OR
    - Its line height is ≥ 52 px (absolute fallback).

    We deliberately do NOT use bbox width — a full-width body line is body text.
    """
    left, top, right, bottom = bbox
    h = bottom - top

    # Tall line relative to peers → title.
    if median_line_height > 0 and h >= median_line_height * 1.5:
        return True
    # Tall absolute (only when no median info).
    if median_line_height == 0 and h >= 52:
        return True
    # Font significantly larger than median.
    if median_font_size > 0 and font_size >= median_font_size * 1.4:
        return True
    return False


# ---------------------------------------------------------------------------
# compact-label detection
# ---------------------------------------------------------------------------

_PROMO_KEYWORDS: frozenset[str] = frozenset({
    "赠品", "会员", "满", "下单", "优惠", "折扣", "包邮",
    "买", "送", "立减", "券", "专享", "限时", "抢", "秒杀",
    "特价", "免费", "新品", "首发",
})

_HERO_BADGE_KEYWORDS: frozenset[str] = frozenset({
    "\u8fdb\u53e3", "\u5f39\u529b", "\u4e0d\u52d2", "\u4e0d\u6389",
    "\u6389\u8ddf", "\u9002\u5408", "\u53ef\u7a7f", "\u65a4",
    "\u9632\u6ed1", "\u56de\u7f29", "\u6a61\u7b4b",
})


def is_compact_label(
    bbox: tuple[int, int, int, int],
    img_size: tuple[int, int],
    text: str = "",
) -> tuple[bool, list[str]]:
    """Determine whether an OCR region is a small promo label / badge.

    Two "weak" signals (area, tall) need a second signal to confirm.
    """
    left, top, right, bottom = bbox
    w = right - left
    h = bottom - top
    area = w * h
    img_w, img_h = img_size
    img_area = img_w * img_h

    reasons: list[str] = []
    sig_narrow = w < 180
    sig_tiny = img_area > 0 and area < img_area * 0.02
    sig_tall = h > 0 and w > 0 and h / w > 1.1
    sig_promo = any(kw in (text or "").strip() for kw in _PROMO_KEYWORDS)

    if sig_narrow:
        reasons.append(f"narrow(w={w})")
    if sig_tiny:
        reasons.append(f"tiny_area({area / img_area:.4f})")
    if sig_tall:
        reasons.append(f"tall_ratio({h / w:.2f})")
    if sig_promo:
        reasons.append("promo_keyword")

    # "narrow" and "promo" are strong signals by themselves.
    # "tiny_area" and "tall_ratio" need a second signal to avoid false positives
    # on wide-but-small areas like short titles.
    signal_count = sum([sig_narrow, sig_promo])
    weak_count = sum([sig_tiny, sig_tall])
    if signal_count > 0:
        return True, reasons
    if weak_count >= 2:
        return True, reasons
    if weak_count >= 1 and signal_count >= 1:
        return True, reasons

    return False, reasons


def classify_text_role(
    bbox: tuple[int, int, int, int],
    img_size: tuple[int, int],
    text: str = "",
    font_size: int = 0,
    is_white_text: bool = False,
    is_compact: bool = False,
) -> str:
    """Classify text as hero_headline, bottom_hero_headline, title, compact_label, or body.

    hero_headline: large white text in top-left, often product badge/callout.
    bottom_hero_headline: large black headline near the lower edge.
    """
    left, top, right, bottom = bbox
    w = right - left
    h = bottom - top
    img_w, img_h = img_size

    # Compact labels override all.
    if is_compact:
        return "compact_label"

    # Hero/product badge: top-left large product selling-point copy. Prefer a
    # hard route here because these blocks need short white badge styling, not
    # normal paragraph rendering.
    top_ratio = top / img_h if img_h else 0
    left_ratio = left / img_w if img_w else 0
    is_top_left = top_ratio < 0.25 and left_ratio < 0.25
    is_large = font_size >= 28 or h >= 30
    has_badge_keyword = any(kw in (text or "") for kw in _HERO_BADGE_KEYWORDS)
    is_wide_enough = img_w > 0 and w >= img_w * 0.35

    if is_top_left and is_large and (is_white_text or has_badge_keyword or is_wide_enough):
        return "hero_headline"

    # Bottom hero headline: large, wide selling-point text near the bottom.
    # It must not be treated as normal body copy because it needs short
    # headline translation and a single large redraw pass.
    bottom_ratio = bottom / img_h if img_h else 0
    top_ratio_for_bottom = top / img_h if img_h else 0
    is_bottom_band = bottom_ratio > 0.72 or top_ratio_for_bottom > 0.62
    is_wide_bottom = img_w > 0 and w >= img_w * 0.45
    if is_bottom_band and is_large and is_wide_bottom and not is_white_text:
        return "bottom_hero_headline"

    # Title: large font but not hero (may be centered or not white).
    if font_size >= 36:
        return "title"

    return "body"


def merge_ocr_lines_into_paragraphs(
    candidates: list[tuple[tuple[int, int, int, int], str]],
    img_size: tuple[int, int],
    font_sizes: list[int] | None = None,
    bg_luminances: list[float] | None = None,
    is_light_bgs: list[bool] | None = None,
    confidences: list[float] | None = None,
    detected_bys: list[str] | None = None,
) -> tuple[list["MergedParagraph"], list[dict]]:
    """Merge OCR lines that belong to the same paragraph.

    Args:
        candidates: ``[((left,top,right,bottom), text), ...]``, sorted by y.
        img_size: ``(width, height)`` of the original image.
        font_sizes: Estimated font size per line (optional).
        bg_luminances: Background luminance per line (optional).
        is_light_bgs: Whether each line is on a light background (optional).
        confidences: OCR confidence per line (optional).
        detected_bys: Detection method per line (optional).

    Returns:
        ``(merged_paragraphs, merge_debug_log)`` where *merged_paragraphs* is a
        list of ``MergedParagraph`` and *merge_debug_log* is a list of dicts
        suitable for debug output.
    """
    # Lazy import to avoid circular dependency at module level.
    from .schemas import MergedParagraph, OCRLine  # noqa: F811

    if not candidates:
        return [], []

    n = len(candidates)
    img_w, img_h = img_size

    # --- compute per-line metadata -------------------------------------------
    median_font = 0
    median_line_h = 0
    if font_sizes and len(font_sizes) == n:
        sorted_sizes = sorted(font_sizes)
        median_font = sorted_sizes[len(sorted_sizes) // 2]
    if n > 0:
        heights = sorted([c[0][3] - c[0][1] for c in candidates])
        median_line_h = heights[len(heights) // 2]

    lines: list[OCRLine] = []
    for i, ((left, top, right, bottom), text) in enumerate(candidates):
        lines.append(OCRLine(
            text=text,
            bbox=(left, top, right, bottom),
            confidence=confidences[i] if confidences and i < len(confidences) else 0.0,
            estimated_font_size=font_sizes[i] if font_sizes and i < len(font_sizes) else 0,
            background_luminance=bg_luminances[i] if bg_luminances and i < len(bg_luminances) else 0.0,
            is_light_background=is_light_bgs[i] if is_light_bgs and i < len(is_light_bgs) else True,
            detected_by=detected_bys[i] if detected_bys and i < len(detected_bys) else "rapidocr",
        ))

    # --- group into paragraphs -----------------------------------------------
    paragraphs: list[MergedParagraph] = []
    debug_log: list[dict] = []
    current_lines: list[OCRLine] = [lines[0]]
    reasons: list[str] = []

    for i in range(1, n):
        prev = lines[i - 1]
        curr = lines[i]

        prev_left, prev_top, prev_right, prev_bottom = prev.bbox
        curr_left, curr_top, curr_right, curr_bottom = curr.bbox

        prev_h = prev_bottom - prev_top
        curr_h = curr_bottom - curr_top
        prev_w = prev_right - prev_left

        merge = True
        merge_reasons: list[str] = []

        # 1) x-alignment: left edges within 20 px.
        x_diff = abs(curr_left - prev_left)
        if x_diff <= 20:
            merge_reasons.append(f"x_aligned(Δ{x_diff})")
        else:
            merge = False
            debug_log.append({
                "action": "split", "line": curr.text[:30],
                "reason": f"x_misaligned: prev_left={prev_left} curr_left={curr_left} diff={x_diff}",
            })

        # 2) y-gap: line spacing should be 0.5×–2.5× the previous line height.
        if merge:
            y_gap = curr_top - prev_bottom
            min_gap = prev_h * -0.2   # allow slight overlap
            max_gap = prev_h * 2.5
            if min_gap <= y_gap <= max_gap:
                merge_reasons.append(f"line_spacing(gap={y_gap},prev_h={prev_h})")
            else:
                merge = False
                debug_log.append({
                    "action": "split", "line": curr.text[:30],
                    "reason": f"y_gap_out_of_range: gap={y_gap} prev_h={prev_h}",
                })

        # 3) font-size similarity: within 25 %.
        if merge and prev.estimated_font_size > 0 and curr.estimated_font_size > 0:
            fs_ratio = max(prev.estimated_font_size, curr.estimated_font_size) / max(1, min(prev.estimated_font_size, curr.estimated_font_size))
            if fs_ratio <= 1.25:
                merge_reasons.append(f"font_size(ratio={fs_ratio:.2f})")
            else:
                merge = False
                debug_log.append({
                    "action": "split", "line": curr.text[:30],
                    "reason": f"font_size_mismatch: prev={prev.estimated_font_size} curr={curr.estimated_font_size}",
                })

        # 4) width similarity OR last line can be shorter.
        if merge:
            width_ratio = max(prev_w, curr_right - curr_left) / max(1, min(prev_w, curr_right - curr_left))
            if width_ratio <= 1.8:
                merge_reasons.append(f"width_similar(ratio={width_ratio:.2f})")
            else:
                merge = False
                debug_log.append({
                    "action": "split", "line": curr.text[:30],
                    "reason": f"width_mismatch: prev_w={prev_w} curr_w={curr_right - curr_left}",
                })

        # 5) Previous line is NOT a title.
        if merge:
            if _is_title_line(prev.bbox, img_w, img_h, prev.estimated_font_size, median_font, median_line_h):
                merge = False
                debug_log.append({
                    "action": "split", "line": curr.text[:30],
                    "reason": "prev_is_title",
                })

        # 6) Text continuity: previous line ends open (no 。！？etc).
        if merge:
            if _text_ends_open(prev.text):
                merge_reasons.append("text_continuity")
            # Even if prev ends with sentence-end, still merge when other
            # signals are strong (x_aligned + line_spacing + font_size).
            # We add a note but don't block.

        if merge:
            current_lines.append(curr)
            reasons.append(" + ".join(merge_reasons))
        else:
            # Finalise current paragraph.
            para = _build_paragraph(current_lines, reasons)
            paragraphs.append(para)
            debug_log.append({
                "action": "merge",
                "merged_text": para.merged_text[:60],
                "line_count": para.line_count,
                "lines": [l.text[:40] for l in current_lines],
                "merged_bbox": list(para.merged_bbox),
                "reason": para.merge_reason,
            })
            # Start new paragraph.
            current_lines = [curr]
            reasons = []

    # Final paragraph.
    para = _build_paragraph(current_lines, reasons)
    paragraphs.append(para)
    debug_log.append({
        "action": "merge",
        "merged_text": para.merged_text[:60],
        "line_count": para.line_count,
        "lines": [l.text[:40] for l in current_lines],
        "merged_bbox": list(para.merged_bbox),
        "reason": para.merge_reason,
    })

    return paragraphs, debug_log


def _build_paragraph(
    lines: list["OCRLine"],
    reasons: list[str],
) -> "MergedParagraph":
    """Create a MergedParagraph from a group of OCRLine objects."""
    from .schemas import MergedParagraph  # noqa: F811

    if not lines:
        return MergedParagraph()

    # Concatenate: remove spaces between lines (Chinese doesn't use inter-word spaces).
    merged_text = "".join(line.text.strip() for line in lines)

    xs = [l.bbox[0] for l in lines] + [l.bbox[2] for l in lines]
    ys = [l.bbox[1] for l in lines] + [l.bbox[3] for l in lines]
    merged_bbox = (min(xs), min(ys), max(xs), max(ys))

    font_sizes = [l.estimated_font_size for l in lines if l.estimated_font_size > 0]
    # Use max of individual line font sizes — the merged paragraph should
    # be rendered at least as large as its largest constituent line.
    est_font = max(font_sizes) if font_sizes else 0

    # Build merge reason string.
    if reasons:
        # Count occurrences of each reason type across all merge decisions.
        reason_types: dict[str, int] = {}
        for r in reasons:
            for part in r.split(" + "):
                key = part.split("(")[0]
                reason_types[key] = reason_types.get(key, 0) + 1
        merge_reason = ", ".join(f"{k}×{v}" for k, v in reason_types.items())
    else:
        merge_reason = "single_line"

    return MergedParagraph(
        merged_text=merged_text,
        lines=lines,
        merged_bbox=merged_bbox,
        estimated_font_size=est_font,
        merge_reason=merge_reason,
    )
