"""Render: erase Chinese text from image and draw English replacement.

Extracted from app/services/image_text_service.py – zero coupling to the main app.

Typesetting algorithm (revised):
  A. Start at 0.88× (body) or 0.92× (title) of original font size.
  B. Auto-wrap English to up to 5-6 lines at that size.
  C. If height fits → use it.
  D. If not → shrink one step, re-wrap (allow more lines).
  E. If MIN_FONT reached and still doesn't fit → expand draw area downward.
  F. Never shrink below min(18, orig*0.72) for body / max(22, orig*0.78) for title.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .schemas import MergedParagraph, RegionDebug


# ---------------------------------------------------------------------------
# draw-text validation
# ---------------------------------------------------------------------------

# Characters that indicate the text is NOT ready for English rendering.
_RE_NON_ENGLISH = re.compile(r"[一-鿿㐀-䶿＀-￯]")


def validate_draw_text(text: str, expected_lang: str = "en") -> tuple[bool, str]:
    """Check whether *text* is safe to draw as English replacement.

    Returns ``(valid, reason)``.
    """
    if not text or not text.strip():
        return False, "empty"
    if _RE_NON_ENGLISH.search(text):
        return False, f"contains_non_english_chars"
    if len(text) > 500:
        return False, f"too_long({len(text)})"
    return True, "ok"


def _font_supports_text(text: str, font: ImageFont.FreeTypeFont) -> bool:
    """Quick check: does *font* have glyphs for every character in *text*?"""
    try:
        # PIL doesn't have a direct has_glyph, but getmask will raise or return
        # empty for missing glyphs (tofu).  We check a representative sample.
        for ch in set(text[:20]):
            mask = font.getmask(ch)
            if mask is None or mask.size == 0:
                return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# background colour extraction (median of border pixels, excludes text)
# ---------------------------------------------------------------------------


def estimate_background_color(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    padding: int = 10,
) -> tuple[int, int, int]:
    """Estimate the true background colour around *bbox*.

    Samples the border ring, excludes dark pixels (text), and returns the
    **median RGB**.  Falls back to white ``(255, 255, 255)`` on failure.
    """
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    img_h, img_w = arr.shape[:2]
    left, top, right, bottom = bbox

    mask = np.zeros((img_h, img_w), dtype=bool)
    ol = max(0, left - padding)
    ot = max(0, top - padding)
    or_ = min(img_w, right + padding)
    ob = min(img_h, bottom + padding)

    if ot < top:
        mask[ot:top, ol:or_] = True
    if bottom < ob:
        mask[bottom:ob, ol:or_] = True
    if ol < left:
        mask[top:bottom, ol:left] = True
    if right < or_:
        mask[top:bottom, right:or_] = True

    ring = arr[mask]
    if ring.size == 0:
        big_l = max(0, left - padding * 3)
        big_t = max(0, top - padding * 3)
        big_r = min(img_w, right + padding * 3)
        big_b = min(img_h, bottom + padding * 3)
        expanded = arr[big_t:big_b, big_l:big_r]
        if expanded.size == 0:
            return (255, 255, 255)
        ring = expanded.reshape(-1, 3)

    lum = 0.2126 * ring[:, 0] + 0.7152 * ring[:, 1] + 0.0722 * ring[:, 2]
    bright = ring[lum >= 100]
    if bright.size == 0:
        bright = ring

    r = int(np.median(bright[:, 0]))
    g = int(np.median(bright[:, 1]))
    b = int(np.median(bright[:, 2]))
    return (r, g, b)


# ---------------------------------------------------------------------------
# background detection (luminance)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# background complexity grading
# ---------------------------------------------------------------------------


def _bbox_to_slice(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> tuple[slice, slice]:
    """Convert a bbox to NumPy slices, clamped to image bounds."""
    left, top, right, bottom = bbox
    return (
        slice(max(0, top), min(img_h, bottom)),
        slice(max(0, left), min(img_w, right)),
    )


def _bbox_roi(
    arr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Extract the region-of-interest from *arr* (H×W or H×W×C)."""
    img_h, img_w = arr.shape[:2]
    row_sl, col_sl = _bbox_to_slice(bbox, img_w, img_h)
    return arr[row_sl, col_sl]


def estimate_background_complexity(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    padding: int = 12,
) -> dict:
    """Grade the background complexity around *bbox*.

    Returns a dict with keys:
      - level: ``"simple"`` | ``"medium"`` | ``"complex"``
      - color_variance: float
      - edge_density: float
      - texture_score: float
      - brightness_variance: float
      - recommended_erase: ``"color_fill"`` | ``"blur_patch"`` | ``"inpaint"``
    """
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float64)
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    img_h, img_w = gray.shape

    left, top, right, bottom = bbox

    # --- sample region: border ring around bbox (excludes text interior) ----
    ol = max(0, left - padding)
    ot = max(0, top - padding)
    or_ = min(img_w, right + padding)
    ob = min(img_h, bottom + padding)

    # Build a mask of just the border ring (top/bottom/left/right strips).
    ring_mask = np.zeros((ob - ot, or_ - ol), dtype=bool)
    # Top strip
    if ot < top:
        ring_mask[0:top - ot, :] = True
    # Bottom strip
    if bottom < ob:
        ring_mask[bottom - ot:, :] = True
    # Left strip (between top and bottom)
    inner_top = max(0, top - ot)
    inner_bottom = min(ob - ot, bottom - ot)
    if ol < left:
        ring_mask[inner_top:inner_bottom, 0:left - ol] = True
    # Right strip
    if right < or_:
        ring_mask[inner_top:inner_bottom, right - ol:] = True

    sample_full = _bbox_roi(arr, (ol, ot, or_, ob))  # H×W×3
    sample_gray_full = _bbox_roi(gray, (ol, ot, or_, ob))  # H×W

    if sample_full.size == 0 or ring_mask.sum() == 0:
        return {
            "level": "simple",
            "color_variance": 0.0,
            "edge_density": 0.0,
            "texture_score": 0.0,
            "brightness_variance": 0.0,
            "recommended_erase": "color_fill",
        }

    # Extract border-ring pixels only.
    sample = sample_full[ring_mask]  # N×3
    sample_gray = sample_gray_full[ring_mask]  # N

    if sample.size == 0 or sample_gray.size == 0 or ring_mask.sum() == 0:
        return {
            "level": "simple",
            "color_variance": 0.0,
            "edge_density": 0.0,
            "texture_score": 0.0,
            "brightness_variance": 0.0,
            "recommended_erase": "color_fill",
        }

    # --- 1) color_variance --------------------------------------------------
    # Per-channel std on ring pixels, averaged.
    ch_std = np.std(sample, axis=0)
    color_variance = float(np.mean(ch_std))

    # --- 2) brightness_variance ---------------------------------------------
    brightness_variance = float(np.std(sample_gray))

    # --- 3) edge_density (Sobel on 2D ring region) -------------------------
    try:
        import cv2
        ring_2d = np.zeros_like(sample_gray_full, dtype=np.uint8)
        ring_2d[ring_mask] = np.clip(sample_gray, 0, 255).astype(np.uint8)
        grad_x = cv2.Sobel(ring_2d, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(ring_2d, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_pixels = edge_mag[ring_mask]
        edge_density = float(np.count_nonzero(edge_pixels > 40) / max(1, edge_pixels.size))
    except ImportError:
        if sample_gray_full.shape[0] > 1 and sample_gray_full.shape[1] > 1:
            dy = np.abs(np.diff(sample_gray_full, axis=0))
            dx = np.abs(np.diff(sample_gray_full, axis=1))
            em = (dy[:, :-1] + dy[:, 1:]) / 2 + (dx[:-1, :] + dx[1:, :]) / 2
            ring_em = em[:ring_mask.shape[0], :ring_mask.shape[1]][ring_mask[:em.shape[0], :em.shape[1]]]
            edge_density = float(np.count_nonzero(ring_em > 25) / max(1, ring_em.size))
        else:
            edge_density = 0.0

    # --- 4) texture_score (Laplacian variance on ring region) --------------
    try:
        import cv2
        lap = cv2.Laplacian(ring_2d, cv2.CV_64F)
        lap_ring = lap[ring_mask]
        texture_score = float(np.var(lap_ring))
    except ImportError:
        texture_score = 0.0

    # --- classify -----------------------------------------------------------
    level: str
    recommended: str

    # Uniform colour → definitely simple (ignore edge_density which may
    # include mask-boundary artifacts).
    if color_variance < 8 and brightness_variance < 10:
        level = "simple"
        recommended = "color_fill"
    elif color_variance < 20 and edge_density < 0.06 and brightness_variance < 40:
        level = "simple"
        recommended = "color_fill"
    elif color_variance < 45 and edge_density < 0.15:
        level = "medium"
        recommended = "blur_patch"
    else:
        level = "complex"
        recommended = "inpaint"

    return {
        "level": level,
        "color_variance": round(color_variance, 2),
        "edge_density": round(edge_density, 4),
        "texture_score": round(texture_score, 2),
        "brightness_variance": round(brightness_variance, 2),
        "recommended_erase": recommended,
    }


# ---------------------------------------------------------------------------
# text mask generation
# ---------------------------------------------------------------------------


def create_text_mask(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    text_color: str = "black",
) -> np.ndarray:
    """Create a binary mask of text pixels within *bbox*.

    Args:
        text_color: ``"black"``, ``"white"``, or ``"colored"``.
            Drives the luminance threshold and dilation strategy.

    Returns a uint8 mask (255 = text pixel) sized to the bbox region.
    """
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float64)
    img_h, img_w = arr.shape[:2]

    left, top, right, bottom = bbox
    left = max(0, left); top = max(0, top)
    right = min(img_w, right); bottom = min(img_h, bottom)

    roi = arr[top:bottom, left:right]
    if roi.size == 0:
        return np.zeros((max(1, bottom - top), max(1, right - left)), dtype=np.uint8)

    lum = 0.2126 * roi[:, :, 0] + 0.7152 * roi[:, :, 1] + 0.0722 * roi[:, :, 2]
    local_median = float(np.median(lum))

    if text_color == "white":
        # White/light text can sit on bright product photos, so a plain
        # absolute threshold misses the strokes or selects too much background.
        # Combine core highlights with local contrast against a blurred bg.
        threshold = max(185.0, min(235.0, float(np.percentile(lum, 82))))
        try:
            import cv2
            lum_u8 = np.clip(lum, 0, 255).astype(np.uint8)
            k = max(15, min(51, ((min(lum_u8.shape) // 2) * 2 + 1)))
            if k % 2 == 0:
                k += 1
            local_bg = cv2.medianBlur(lum_u8, k)
            contrast = lum - local_bg.astype(np.float64)
            grad_x = cv2.Sobel(lum_u8, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(lum_u8, cv2.CV_64F, 0, 1, ksize=3)
            edge_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            mask = (
                ((lum >= threshold) & ((contrast >= 8) | (edge_mag >= 22))) |
                ((lum >= local_median + 34) & (lum >= 188) & (edge_mag >= 14))
            ).astype(np.uint8) * 255

            # Remove broad bright product/background patches. Text strokes tend
            # to be many small/medium connected components; a large smooth
            # clothing highlight should not be sent to inpainting.
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            filtered = np.zeros_like(mask)
            roi_area = max(1, mask.shape[0] * mask.shape[1])
            for label in range(1, num):
                x, y, w, h, area = stats[label]
                if area < 3:
                    continue
                if area > roi_area * 0.08:
                    continue
                if w > mask.shape[1] * 0.45 and h > mask.shape[0] * 0.45:
                    continue
                filtered[labels == label] = 255
            mask = filtered

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        except ImportError:
            mask = (((lum >= threshold) & (lum >= local_median + 20)) | ((lum >= local_median + 34) & (lum >= 188))).astype(np.uint8) * 255

    elif text_color == "black":
        # Dark text on light bg.
        mask = (lum < local_median - 20).astype(np.uint8) * 255
        try:
            import cv2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            mask = cv2.dilate(mask, kernel, iterations=1)
        except ImportError:
            pass

    else:  # colored
        # Color-distance based: text pixels differ from local median in any channel.
        r_med = float(np.median(roi[:, :, 0]))
        g_med = float(np.median(roi[:, :, 1]))
        b_med = float(np.median(roi[:, :, 2]))
        diff = np.abs(roi[:, :, 0] - r_med) + np.abs(roi[:, :, 1] - g_med) + np.abs(roi[:, :, 2] - b_med)
        mask = (diff > 80).astype(np.uint8) * 255
        try:
            import cv2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            mask = cv2.dilate(mask, kernel, iterations=1)
        except ImportError:
            pass

    return mask


# ---------------------------------------------------------------------------
# adaptive erase + inpainting
# ---------------------------------------------------------------------------


def inpaint_text_region(
    image: Image.Image,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    inpaint_radius: int = 3,
) -> Image.Image:
    """Inpaint text pixels within *bbox* using OpenCV TELEA.

    Only the bbox ROI is processed; the rest of the image is untouched.
    Modifies *image* in-place.
    """
    try:
        import cv2
    except ImportError:
        # Fallback: use median blur on the bbox region.
        return _blur_patch_erase(image, bbox)

    arr = np.array(image.convert("RGB"))
    img_h, img_w = arr.shape[:2]

    left, top, right, bottom = bbox
    left = max(0, left)
    top = max(0, top)
    right = min(img_w, right)
    bottom = min(img_h, bottom)

    # Expand ROI slightly for better inpainting context.
    ctx_pad = inpaint_radius + 2
    ctx_l = max(0, left - ctx_pad)
    ctx_t = max(0, top - ctx_pad)
    ctx_r = min(img_w, right + ctx_pad)
    ctx_b = min(img_h, bottom + ctx_pad)

    roi = arr[ctx_t:ctx_b, ctx_l:ctx_r].copy()

    # Build mask positioned within the ROI.
    full_mask = np.zeros((ctx_b - ctx_t, ctx_r - ctx_l), dtype=np.uint8)
    mask_h, mask_w = mask.shape
    # Place mask at the correct offset within ROI.
    m_t = top - ctx_t
    m_l = left - ctx_l
    m_b = m_t + mask_h
    m_r = m_l + mask_w
    full_mask[m_t:m_b, m_l:m_r] = mask

    # Inpaint the ROI.
    flags = cv2.INPAINT_TELEA
    inpainted_roi = cv2.inpaint(roi, full_mask, inpaint_radius, flags)

    # Write back.
    arr[ctx_t:ctx_b, ctx_l:ctx_r] = inpainted_roi

    return Image.fromarray(arr)


def _blur_patch_erase(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
) -> Image.Image:
    """Medium-complexity erase: apply median blur to the bbox region."""
    try:
        import cv2
    except ImportError:
        return image

    arr = np.array(image.convert("RGB"))
    img_h, img_w = arr.shape[:2]

    left, top, right, bottom = bbox
    left = max(0, left)
    top = max(0, top)
    right = min(img_w, right)
    bottom = min(img_h, bottom)

    roi = arr[top:bottom, left:right]
    if roi.size == 0:
        return image

    # Median blur preserves edges better than Gaussian for textures.
    kernel = min(7, min(roi.shape[0] // 3, roi.shape[1] // 3))
    if kernel % 2 == 0:
        kernel += 1
    if kernel < 3:
        kernel = 3
    blurred = cv2.medianBlur(roi, kernel)
    arr[top:bottom, left:right] = blurred

    return Image.fromarray(arr)


def erase_text_region(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    complexity: dict,
    text_color: str = "black",
    erase_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[Image.Image, dict]:
    """Adaptively erase text from *bbox* based on background complexity.

    For white text: always uses mask-based inpainting (even on simple bg)
    to properly remove glow edges.
    """
    level = complexity.get("level", "simple")
    recommended = complexity.get("recommended_erase", "color_fill")
    erase_target = erase_bbox or bbox
    left, top, right, bottom = erase_target
    bbox_area = max(1, (right - left) * (bottom - top))

    debug: dict = {
        "bbox": list(erase_target),
        "level": level,
        "color_variance": complexity.get("color_variance"),
        "edge_density": complexity.get("edge_density"),
        "strategy": recommended,
        "mask_mode": f"{text_color}_text",
        "bbox_area": bbox_area,
    }

    # --- white text: always use mask+inpaint to remove glow edges ---------
    if text_color == "white":
        mask = create_text_mask(image, erase_target, text_color="white")
        mask_pixels = int(np.count_nonzero(mask))
        debug["mask_pixels"] = mask_pixels
        debug["mask_ratio"] = round(mask_pixels / bbox_area, 4)
        debug["dilation_px"] = "3+1close"

        min_mask = max(10, int(bbox_area * 0.005))
        if mask_pixels >= min_mask:
            bbox_h = bottom - top
            radius = 3 if bbox_h >= 30 else 2
            image = inpaint_text_region(image, mask, erase_target, inpaint_radius=radius)
            debug["inpaint_radius"] = radius
            debug["strategy"] = "inpaint"
            debug["inpaint_applied"] = True
            debug["fallback_used"] = False
        else:
            # Mask too sparse — try looser threshold.
            debug["fallback_reason"] = f"mask_too_sparse({mask_pixels})"
            mask2 = _create_white_text_mask_fallback(image, erase_target)
            mp2 = int(np.count_nonzero(mask2))
            debug["mask_pixels_fallback"] = mp2
            if mp2 >= min_mask:
                image = inpaint_text_region(image, mask2, erase_target, inpaint_radius=3)
                debug["inpaint_applied"] = True
                debug["fallback_used"] = True
                debug["strategy"] = "inpaint_fallback"
                debug["inpaint_radius"] = 3
            else:
                # Last resort: color-fill the bbox.
                bg_rgb = estimate_background_color(image, erase_target)
                draw = ImageDraw.Draw(image)
                draw.rectangle((left, top, right, bottom), fill=bg_rgb)
                debug["strategy"] = "color_fill_last_resort"
                debug["inpaint_applied"] = False
                debug["fallback_used"] = True
                debug["bg_color"] = list(bg_rgb)

    elif level == "simple":
        # Strategy A: sampled color fill (dark text on simple bg).
        bg_rgb = estimate_background_color(image, erase_target)
        draw = ImageDraw.Draw(image)
        draw.rectangle((left, top, right, bottom), fill=bg_rgb)
        debug["bg_color"] = list(bg_rgb)
        debug["inpaint_radius"] = 0
        debug["mask_pixels"] = 0
        debug["inpaint_applied"] = False

    elif level == "medium":
        # Strategy B: blur patch.
        image = _blur_patch_erase(image, erase_target)
        debug["inpaint_radius"] = 0
        debug["mask_pixels"] = 0
        debug["inpaint_applied"] = False
        debug["fallback_reason"] = None

    else:  # complex
        mask = create_text_mask(image, erase_target, text_color=text_color)
        mask_pixels = int(np.count_nonzero(mask))
        debug["mask_pixels"] = mask_pixels

        if mask_pixels > 0:
            bbox_h = bottom - top
            radius = 6 if bbox_h >= 36 else (4 if bbox_h >= 24 else 3)
            image = inpaint_text_region(image, mask, erase_target, inpaint_radius=radius)
            debug["inpaint_radius"] = radius
            debug["inpaint_applied"] = True
            debug["fallback_reason"] = None
        else:
            image = _blur_patch_erase(image, erase_target)
            debug["strategy"] = "blur_patch_fallback"
            debug["inpaint_radius"] = 0
            debug["inpaint_applied"] = False
            debug["fallback_reason"] = "empty_mask"

    return image, debug


def _create_white_text_mask_fallback(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Looser white-text mask — lower luminance threshold, more dilation."""
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float64)
    img_h, img_w = arr.shape[:2]
    left, top, right, bottom = bbox
    left = max(0, left); top = max(0, top)
    right = min(img_w, right); bottom = min(img_h, bottom)
    roi = arr[top:bottom, left:right]
    if roi.size == 0:
        return np.zeros((max(1, bottom - top), max(1, right - left)), dtype=np.uint8)
    lum = 0.2126 * roi[:, :, 0] + 0.7152 * roi[:, :, 1] + 0.0722 * roi[:, :, 2]
    # Very loose: luminance > 150 OR top 40% brightest in bbox.
    threshold = max(150, np.percentile(lum, 60))
    mask = (lum > threshold).astype(np.uint8) * 255
    try:
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    except ImportError:
        pass
    return mask


def estimate_surrounding_background(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    padding_ratio: float = 0.35,
) -> dict:
    """Sample the outer ring around *bbox* to determine background colour.

    Returns ``{"is_light": bool, "mean_luminance": float}``.
    """
    rgb = image.convert("RGB")
    array = np.asarray(rgb)
    img_h, img_w = array.shape[:2]
    left, top, right, bottom = bbox

    box_w = max(1, right - left)
    box_h = max(1, bottom - top)
    pad_x = max(4, round(box_w * padding_ratio))
    pad_y = max(4, round(box_h * padding_ratio))

    outer_left = max(0, left - pad_x)
    outer_top = max(0, top - pad_y)
    outer_right = min(img_w, right + pad_x)
    outer_bottom = min(img_h, bottom + pad_y)

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
    mean_lum = float(np.median(luminance))
    return {"is_light": mean_lum >= 170, "mean_luminance": mean_lum}


# ---------------------------------------------------------------------------
# font / text sizing
# ---------------------------------------------------------------------------


def estimate_original_font_size(bbox: tuple[int, int, int, int]) -> int:
    left, top, right, bottom = bbox
    box_height = max(1, bottom - top)
    # Chinese glyph ≈ 82% of bbox height; for English we're more generous.
    return max(10, round(box_height * 0.90))


def estimate_final_text_height(
    original_text_height: int,
    original_image_size: tuple[int, int],
    target_size: tuple[int, int] = (660, 900),
) -> float:
    ow, oh = original_image_size
    tw, th = target_size
    scale = min(tw / ow, th / oh)
    return original_text_height * scale


# ---------------------------------------------------------------------------
# font loading (cached)
# ---------------------------------------------------------------------------

# Try common font paths — prefer sans-serif for English readability.
_FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/msyh.ttc",     # Microsoft YaHei (fallback)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

_cached_font_path: str | None = None


def _find_system_font() -> str | None:
    """Locate a TrueType font file on the system (cached)."""
    global _cached_font_path
    if _cached_font_path is not None:
        return _cached_font_path if _cached_font_path else None

    import os
    for path in _FONT_PATHS:
        if os.path.isfile(path):
            _cached_font_path = path
            return path
    _cached_font_path = ""
    return None


def _make_font(size: int) -> ImageFont.FreeTypeFont:
    """Create a TrueType font at *size*. Falls back to default bitmap."""
    path = _find_system_font()
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# word-wrap (unlimited lines)
# ---------------------------------------------------------------------------


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int = 5,
) -> list[str] | None:
    """Word-wrap *text* to fit within *max_width*.

    Returns None when a single word is wider than *max_width* or *max_lines*
    is exceeded.
    """
    if "\n" in text:
        out: list[str] = []
        for segment in text.splitlines():
            segment = segment.strip()
            if not segment:
                continue
            wrapped = wrap_text_to_width(draw, segment, font, max_width, max_lines=max_lines - len(out))
            if wrapped is None:
                return None
            out.extend(wrapped)
            if len(out) > max_lines:
                return None
        return out or None

    words = text.split()
    if not words:
        return None

    lines: list[str] = []
    current_line = ""

    for word in words:
        candidate = word if not current_line else f"{current_line} {word}"
        w = draw.textbbox((0, 0), candidate, font=font)[2]
        if w <= max_width:
            current_line = candidate
            continue
        if current_line:
            lines.append(current_line)
            current_line = word
        else:
            # Single word too wide → can't fit at this font size.
            return None
        if len(lines) >= max_lines:
            return None

    if current_line:
        lines.append(current_line)
    if len(lines) > max_lines:
        return None
    return lines


# ---------------------------------------------------------------------------
# typesetting engine (the core rewrite)
# ---------------------------------------------------------------------------

# Whether a paragraph is a "title" vs "body" for sizing purposes.
# Use the original font size (from OCR) rather than bbox height, because
# a merged body paragraph can be quite tall (120-200 px) while still
# having normal body font sizes (24-34 px).
_TITLE_FONT_THRESHOLD = 36   # px — larger font → title sizing rules


def detect_original_text_color(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
) -> tuple[str, tuple[int, int, int], float]:
    """Detect the original text colour by sampling dark/bright pixels in *bbox*.

    Returns ``(label, rgb, confidence)`` where label is ``"white"``, ``"black"``,
    or ``"colored"``.
    """
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float64)
    img_h, img_w = arr.shape[:2]
    left, top, right, bottom = bbox
    left = max(0, left); top = max(0, top)
    right = min(img_w, right); bottom = min(img_h, bottom)

    roi = arr[top:bottom, left:right]
    if roi.size == 0:
        return ("black", (0, 0, 0), 1.0)

    lum = 0.2126 * roi[:, :, 0] + 0.7152 * roi[:, :, 1] + 0.0722 * roi[:, :, 2]

    # Sample the darkest 15% and brightest 15% of pixels.
    flat_lum = lum.ravel()
    n = max(1, len(flat_lum) // 7)
    darkest_idx = np.argpartition(flat_lum, n)[:n]
    brightest_idx = np.argpartition(flat_lum, -n)[-n:]

    dark_lum = float(np.median(flat_lum[darkest_idx]))
    bright_lum = float(np.median(flat_lum[brightest_idx]))
    median_all = float(np.median(flat_lum))

    # If the bright pixels are very bright and have any meaningful local
    # contrast, the text is likely white/light. The contrast threshold is kept
    # deliberately lower than the old value because many product photos place
    # white headline text over pale skin/beige backgrounds.
    if bright_lum > 218 and (bright_lum - median_all) > 28:
        bright_pixels = roi.reshape(-1, 3)[brightest_idx]
        r = int(np.median(bright_pixels[:, 0]))
        g = int(np.median(bright_pixels[:, 1]))
        b = int(np.median(bright_pixels[:, 2]))
        return ("white", (r, g, b), bright_lum / 255.0)

    # If the dark pixels are very dark AND significantly darker than median,
    # the text is likely black/dark.
    if dark_lum < 80 and (median_all - dark_lum) > 50:
        dark_pixels = roi.reshape(-1, 3)[darkest_idx]
        r = int(np.median(dark_pixels[:, 0]))
        g = int(np.median(dark_pixels[:, 1]))
        b = int(np.median(dark_pixels[:, 2]))
        return ("black", (r, g, b), 1.0 - dark_lum / 255.0)

    return ("colored", (128, 128, 128), 0.5)


def detect_text_alignment(
    lines: "list[MergedParagraph | OCRLine]",
    container_width: int,
) -> str:
    """Detect original text alignment from OCR bboxes.

    Returns ``"left"``, ``"center"``, or ``"right"``.  Defaults to ``"left"``.
    """
    if not lines:
        return "left"
    bboxes = []
    for item in lines:
        if hasattr(item, "merged_bbox"):
            bboxes.append(item.merged_bbox)
        elif hasattr(item, "bbox"):
            bboxes.append(item.bbox)
    if len(bboxes) < 2:
        return "left"

    lefts = [b[0] for b in bboxes]
    rights = [b[2] for b in bboxes]

    # Check center first (most specific), then right, then left as default.
    centers = [(b[0] + b[2]) / 2 for b in bboxes]
    center_spread = max(centers) - min(centers)
    avg_center = sum(centers) / len(centers)
    if center_spread <= 25 and abs(avg_center - container_width / 2) < container_width * 0.12:
        return "center"

    right_spread = max(rights) - min(rights)
    if right_spread <= 20:
        return "right"

    # Left is the default.
    return "left"


def _line_height_for_font(font_size: int) -> int:
    """Line height at 1.25 × cap-height."""
    return max(font_size + 2, round(font_size * 1.25))


def _typeset_english(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_w: int,
    box_h: int,
    orig_font_size: int,
    img_w: int,
    img_h: int,
    img_bottom: int,     # bottom edge of the image (for area expansion)
    is_title: bool | None = None,
    title_width_factor: float = 1.50,
) -> tuple[list[str] | None, int, int, int, dict]:
    """Find the best font size and line count to fit *text* inside the box.

    For titles: strongly prefers single line.
    """
    if is_title is None:
        is_title = orig_font_size >= _TITLE_FONT_THRESHOLD

    # --- sizing rules --------------------------------------------------------
    if is_title:
        start_font = max(22, round(orig_font_size * 0.92))
        min_font = max(22, round(orig_font_size * 0.78))
        max_lines_try = 2
    else:
        start_font = max(18, round(orig_font_size * 0.88))
        min_font = max(18, round(orig_font_size * 0.72))
        max_lines_try = 6

    debug = {
        "orig_font_size": orig_font_size,
        "start_font": start_font,
        "min_font": min_font,
        "is_title": is_title,
        "box_w": box_w,
        "box_h": box_h,
        "text_len": len(text),
        "title_width_factor": title_width_factor,
    }

    title_expanded_w = round(box_w * title_width_factor) if is_title else box_w

    # --- Phase 0 (title): prefer single line with wider tolerance -------------
    if is_title and "\n" not in text:
        for fs in range(start_font, min_font - 1, -1):
            font = _make_font(fs)
            if draw.textbbox((0, 0), text, font=font)[2] <= title_expanded_w:
                lh = _line_height_for_font(fs)
                if lh <= box_h:
                    debug.update(final_font=fs, lines=1, line_height=lh,
                                 total_h=lh, phase="fit_single", expanded=False)
                    return [text], fs, box_h, 0, debug

    # --- Phase 1: start_font → min_font, allow up to max_lines_try -----------
    for fs in range(start_font, min_font - 1, -1):
        font = _make_font(fs)
        test_w = title_expanded_w if is_title else box_w
        for max_ln in range(2, max_lines_try + 1):
            lines = wrap_text_to_width(draw, text, font, test_w, max_lines=max_ln)
            if lines is None:
                continue
            lh = _line_height_for_font(fs)
            total_h = lh * len(lines)
            if total_h <= box_h:
                debug["final_font"] = fs
                debug["lines"] = len(lines)
                debug["line_height"] = lh
                debug["total_h"] = total_h
                debug["phase"] = "fit"
                debug["expanded"] = False
                return lines, fs, box_h, 0, debug

    # --- Phase 2: at min_font with max lines, check if it fits ---------------
    font = _make_font(min_font)
    lines = wrap_text_to_width(draw, text, font, box_w, max_lines=max_lines_try)
    if lines is not None:
        lh = _line_height_for_font(min_font)
        total_h = lh * len(lines)
        if total_h <= box_h:
            debug["final_font"] = min_font
            debug["lines"] = len(lines)
            debug["line_height"] = lh
            debug["total_h"] = total_h
            debug["phase"] = "fit_at_min"
            debug["expanded"] = False
            return lines, min_font, box_h, 0, debug

    # --- Phase 3: expand draw area downward -----------------------------------
    # Use whitespace between the current bbox bottom and the image bottom.
    # Expand by up to 2× the original box height, but not past the image edge.
    # Font is kept at min_font — we expand the area instead of shrinking further.
    if lines is not None:
        font_min = _make_font(min_font)
        for max_ln3 in range(max_lines_try, max_lines_try + 4):
            lines3 = wrap_text_to_width(draw, text, font_min, box_w, max_lines=max_ln3)
            if lines3 is None:
                continue
            lh3 = _line_height_for_font(min_font)
            needed_h = lh3 * len(lines3)
            for expand_ratio in (1.2, 1.4, 1.7, 2.0, 2.5):
                expanded_h = round(box_h * expand_ratio)
                available = img_h - (img_bottom - box_h)  # total available from paragraph top
                if expanded_h > available:
                    expanded_h = available
                if expanded_h <= box_h:
                    continue
                if needed_h <= expanded_h:
                    expanded_by = expanded_h - box_h
                    debug["final_font"] = min_font
                    debug["lines"] = len(lines3)
                    debug["line_height"] = lh3
                    debug["total_h"] = needed_h
                    debug["phase"] = "expanded"
                    debug["expanded"] = True
                    debug["expanded_by_px"] = expanded_by
                    debug["expanded_ratio"] = expand_ratio
                    return lines3, min_font, expanded_h, expanded_by, debug

    # --- Phase 4: last resort — min_font with whatever fits, allow overflow ---
    if lines is not None:
        lh = _line_height_for_font(min_font)
        total_h = lh * len(lines)
        debug["final_font"] = min_font
        debug["lines"] = len(lines)
        debug["line_height"] = lh
        debug["total_h"] = total_h
        debug["phase"] = "overflow"
        debug["expanded"] = False
        return lines, min_font, box_h, 0, debug

    # --- Phase 5: even shorter version of the text (image_translation) -------
    # The caller should pass the shorter `image_translation` text for drawing.
    # If we're here with the full text, this shouldn't happen normally.
    debug["phase"] = "failed"
    return None, 0, 0, 0, debug


def _expand_hero_erase_bbox(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Expand a large white product badge bbox enough to catch glow remnants."""
    left, top, right, bottom = bbox
    h = max(1, bottom - top)
    pad_x = max(4, round(h * 0.08))
    pad_top = max(3, round(h * 0.06))
    pad_bottom = max(5, round(h * 0.10))
    return (
        max(0, left - pad_x),
        max(0, top - pad_top),
        min(img_w, right + pad_x),
        min(img_h, bottom + pad_bottom),
    )


def _expand_bottom_hero_erase_bbox(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Expand bottom headline erase enough to cover bold antialiasing."""
    left, top, right, bottom = bbox
    h = max(1, bottom - top)
    return (
        max(0, left - max(6, round(h * 0.08))),
        max(0, top - max(5, round(h * 0.08))),
        min(img_w, right + max(8, round(h * 0.10))),
        min(img_h, bottom + max(6, round(h * 0.08))),
    )


def _expand_feature_label_erase_bbox(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Slightly expand small feature-label text to clear antialias/gray text."""
    left, top, right, bottom = bbox
    h = max(1, bottom - top)
    return (
        max(0, left - max(4, round(h * 0.08))),
        max(0, top - max(3, round(h * 0.06))),
        min(img_w, right + max(6, round(h * 0.10))),
        min(img_h, bottom + max(4, round(h * 0.08))),
    )


def _extract_kg_range(text: str) -> str | None:
    """Return a normalized kg range from text containing kg or Chinese jin."""
    if not text:
        return None
    kg_match = re.search(r"(\d+(?:\.\d+)?)\s*[-~–]\s*(\d+(?:\.\d+)?)\s*kg", text, re.I)
    if kg_match:
        a = float(kg_match.group(1))
        b = float(kg_match.group(2))
        return f"{a:g}-{b:g} kg"
    jin_match = re.search(r"(\d+(?:\.\d+)?)\s*[-~–]\s*(\d+(?:\.\d+)?)\s*(?:jin|\u65a4)", text, re.I)
    if jin_match:
        a = float(jin_match.group(1)) / 2
        b = float(jin_match.group(2)) / 2
        return f"{a:g}-{b:g} kg"
    return None


def _shorten_hero_badge_text(
    original_text: str,
    full_translation: str,
    image_translation: str,
) -> str:
    """Force top-left product badges into short e-commerce label copy."""
    source = " ".join(str(x or "") for x in (original_text, full_translation, image_translation))
    kg_range = _extract_kg_range(source)
    lowered = source.lower()

    has_elastic = ("elastic" in lowered) or ("\u5f39\u529b" in source) or ("\u6a61\u7b4b" in source)
    has_no_slip = (
        "slip" in lowered or "non-slip" in lowered or "no-slip" in lowered
        or "\u4e0d\u6389" in source or "\u6389\u8ddf" in source
    )
    has_snug = "snug" in lowered or "pinch" in lowered or "\u4e0d\u52d2" in source
    has_imported = "import" in lowered or "\u8fdb\u53e3" in source

    if has_elastic or has_no_slip or has_snug or has_imported:
        first = "Imported Elastic, No-Slip Fit" if has_imported or has_elastic else "Snug No-Slip Fit"
        if kg_range:
            return f"{first}\nFits {kg_range}"
        return first

    candidate = (image_translation or full_translation or original_text or "").replace("\n", " ").strip()
    candidate = re.sub(r"\s+", " ", candidate)
    words = candidate.split()
    if len(words) > 10 or "." in candidate:
        candidate = " ".join(words[:10]).rstrip(".,;:")
    return candidate


def _shorten_bottom_hero_text(
    original_text: str,
    full_translation: str,
    image_translation: str,
) -> str:
    """Force bottom hero copy into a short 2-line product headline."""
    source = " ".join(str(x or "") for x in (original_text, full_translation, image_translation))
    lowered = source.lower()

    has_grip = any(x in lowered for x in ("grip", "anti-slip", "non-slip", "silicone")) or "\u9632\u6ed1" in source
    has_cushion = any(x in lowered for x in ("cushion", "shock", "protect", "support")) or any(
        x in source for x in ("\u7f13\u51b2", "\u51cf\u9707", "\u5b88\u62a4")
    )
    if has_grip and has_cushion:
        return "Full-Sole Grip\nCushions Every Step"
    if has_grip:
        return "Full-Sole Grip\nNo-Slip Support"
    if has_cushion:
        return "Cushioning Support\nFor Every Move"

    candidate = (image_translation or full_translation or original_text or "").replace("\n", " ").strip()
    candidate = re.sub(r"\s+", " ", candidate)
    words = candidate.split()
    if len(words) > 10 or "." in candidate:
        candidate = " ".join(words[:10]).rstrip(".,;:")
    return candidate


def _fallback_english_text(
    original_text: str,
    full_translation: str,
    image_translation: str,
    text_role: str = "",
) -> str:
    """Deterministic fallback when model output is Chinese/garbled/too long."""
    source = " ".join(str(x or "") for x in (original_text, full_translation, image_translation))
    lowered = source.lower()

    if text_role == "hero_headline":
        return _shorten_hero_badge_text(original_text, full_translation, image_translation)
    if text_role == "bottom_hero_headline":
        return _shorten_bottom_hero_text(original_text, full_translation, image_translation)

    # Common product feature-card labels.
    if "\u4e94\u8dbe" in source or "five-toe" in lowered or "toe" in lowered:
        return "Five-Toe Design\nFlexible Toe Comfort"
    if "\u889c\u53e3" in source or "\u5f39\u529b" in source or "elastic" in lowered:
        kg_range = _extract_kg_range(source) or "42-80 kg"
        return f"Elastic Cuff\nFits {kg_range}"
    if "\u68c9" in source or "\u900f\u6c14" in source or "\u67d4\u8f6f" in source or "cotton" in lowered:
        return "Premium Cotton\nBreathable and Soft"
    if "\u9632\u6ed1\u70b9" in source or "grip dot" in lowered or "anti-slip dot" in lowered:
        return "Weighted Grip Dots\nFirm Grip, No Shifting"
    if "\u540e\u8ddf" in source or "heel" in lowered:
        return "Comfort Heel\nReinforced, No Restriction"
    if "\u9632\u6ed1" in source or "grip" in lowered or "slip" in lowered:
        return "Added Grip\nStable, No Slipping"

    # Last resort: keep only a compact ASCII-ish fragment if available.
    candidate = (image_translation or full_translation or "").replace("\n", " ").strip()
    candidate = re.sub(r"[^\x20-\x7E]+", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    words = candidate.split()
    if words:
        return " ".join(words[:8]).rstrip(".,;:")
    return "Product Feature"


# ---------------------------------------------------------------------------
# debug record builder
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# compact-label typesetting helpers
# ---------------------------------------------------------------------------

# Short words that should never appear alone on a line.
_ORPHAN_WORDS: frozenset[str] = frozenset({
    "with", "for", "on", "over", "the", "a", "an", "of", "in", "at", "to",
    "by", "or", "and", "is", "it", "as", "be", "no", "so", "we", "he",
})

_COMPACT_LABEL_MIN_FONT = 14
_COMPACT_LABEL_MAX_LINES = 3


def _prevent_orphan_words(lines: list[str]) -> list[str]:
    """Re-flow short orphan words so no line has a single short word alone."""
    if not lines or len(lines) < 2:
        return lines
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        words_in_line = line.split()
        if len(words_in_line) == 1 and words_in_line[0].lower() in _ORPHAN_WORDS:
            # Merge this orphan with the previous line if possible.
            if result:
                result[-1] = f"{result[-1]} {words_in_line[0]}"
            elif i + 1 < len(lines):
                # Merge forward into next line.
                lines[i + 1] = f"{words_in_line[0]} {lines[i + 1]}"
            else:
                result.append(line)
        else:
            result.append(line)
        i += 1
    return result


def _wrap_compact_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int = _COMPACT_LABEL_MAX_LINES,
) -> list[str] | None:
    """Word-wrap for compact labels — prevents orphan words."""
    lines = wrap_text_to_width(draw, text, font, max_width, max_lines=max_lines)
    if lines is None:
        return None
    return _prevent_orphan_words(lines)


def _typeset_compact_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_w: int,
    box_h: int,
    orig_font_size: int,
    img_w: int,
    img_h: int,
    img_bottom: int,
) -> tuple[list[str] | None, int, int, int, dict]:
    """Typesetting for compact promo labels: max 2-3 lines, high min font."""
    start_fs = max(_COMPACT_LABEL_MIN_FONT, round(orig_font_size * 0.85))
    min_fs = max(_COMPACT_LABEL_MIN_FONT, round(orig_font_size * 0.65))

    debug: dict = {
        "orig_font_size": orig_font_size,
        "start_font": start_fs,
        "min_font": min_fs,
        "is_compact": True,
        "box_w": box_w,
        "box_h": box_h,
        "text_len": len(text),
    }

    for fs in range(start_fs, min_fs - 1, -1):
        font = _make_font(fs)
        for max_ln in range(1, _COMPACT_LABEL_MAX_LINES + 1):
            lines = _wrap_compact_label(draw, text, font, box_w, max_lines=max_ln)
            if lines is None:
                continue
            lh = _line_height_for_font(fs)
            if lh * len(lines) <= box_h:
                debug.update(final_font=fs, lines=len(lines), line_height=lh,
                             total_h=lh * len(lines), phase="fit", expanded=False)
                return lines, fs, box_h, 0, debug

    # Try expanded area.
    font = _make_font(min_fs)
    for max_ln in range(1, _COMPACT_LABEL_MAX_LINES + 1):
        lines = _wrap_compact_label(draw, text, font, box_w, max_lines=max_ln)
        if lines is None:
            continue
        lh = _line_height_for_font(min_fs)
        for expand_ratio in (1.3, 1.6, 2.0):
            expanded_h = round(box_h * expand_ratio)
            if img_bottom + expanded_h > img_h:
                expanded_h = img_h - img_bottom
            if expanded_h <= box_h:
                continue
            if lh * len(lines) <= expanded_h:
                expanded_by = expanded_h - box_h
                debug.update(final_font=min_fs, lines=len(lines), line_height=lh,
                             total_h=lh * len(lines), phase="expanded", expanded=True,
                             expanded_by_px=expanded_by)
                return lines, min_fs, expanded_h, expanded_by, debug

    # Overflow.
    if lines is not None:
        lh = _line_height_for_font(min_fs)
        debug.update(final_font=min_fs, lines=len(lines), line_height=lh,
                     total_h=lh * len(lines), phase="overflow", expanded=False)
        return lines, min_fs, box_h, 0, debug

    debug["phase"] = "failed"
    return None, 0, 0, 0, debug


# ---------------------------------------------------------------------------
# debug record builder
# ---------------------------------------------------------------------------


def _make_rd(
    original_text: str,
    full_translation: str,
    image_translation: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
    estimated_original_font_size: int = 0,
    bg_info: dict | None = None,
    used_font_size: int = 0,
    final_h: float = 0,
    detected_by: str = "rapidocr",
    translated: bool = False,
    replaced: bool = False,
    skip_reason: str | None = None,
    confidence: float = 0,
    confidence_threshold: float = 0,
    typeset_debug: dict | None = None,
    bg_color: tuple[int, int, int] | None = None,
) -> RegionDebug:
    bg = bg_info or {}
    is_light = bg.get("is_light", True)

    # Fold typesetting debug into skip_reason for successful renders.
    reason = skip_reason
    if typeset_debug and skip_reason is None:
        td = typeset_debug
        parts = [
            f"typeset: orig_fs={td.get('orig_font_size','?')}",
            f"start={td.get('start_font','?')}",
            f"final={td.get('final_font','?')}",
            f"min={td.get('min_font','?')}",
            f"lines={td.get('lines','?')}",
            f"phase={td.get('phase','?')}",
        ]
        if td.get("hero_headline"):
            parts.append("HERO")
        if td.get("text_role") and td["text_role"] not in ("body",):
            if not td.get("hero_headline"):
                parts.append(f"role={td['text_role']}")
        if td.get("is_compact"):
            parts.append("COMPACT")
            cr = td.get("compact_reasons", [])
            if cr:
                parts.append(f"reasons=[{','.join(cr)}]")
        if td.get("alignment") and td.get("alignment") != "left":
            parts.append(f"align={td['alignment']}")
        if td.get("expanded"):
            parts.append(f"EXPANDED+{td.get('expanded_by_px','?')}px")
        reason = " ".join(parts)

    # Append bg colour to reason if available.
    if bg_color and reason and "bg=" not in reason:
        reason += f" bg=({bg_color[0]},{bg_color[1]},{bg_color[2]})"

    # Append erase strategy info.
    if typeset_debug and skip_reason is None:
        td = typeset_debug
        erase_parts = []
        if td.get("erase_level"):
            erase_parts.append(f"erase={td['erase_level']}({td.get('erase_strategy','?')})")
        if td.get("inpaint_r"):
            erase_parts.append(f"ir={td['inpaint_r']}")
        if td.get("mask_px"):
            erase_parts.append(f"mask={td['mask_px']}")
        render_parts = []
        if td.get("shadow_enabled"):
            render_parts.append("shadow")
        if td.get("stroke_width", 0) > 0:
            render_parts.append(f"stroke={td['stroke_width']}")
        if td.get("glow_enabled"):
            render_parts.append(f"glow_r={td['glow_radius']}")
        if not td.get("text_layer_blurred", True):
            render_parts.append("crisp")
        all_parts = erase_parts + render_parts
        if all_parts:
            reason += " " + " ".join(all_parts)

    debug_text_color = None
    if typeset_debug:
        debug_text_color = typeset_debug.get("final_text_color")

    return RegionDebug(
        original_text=original_text,
        full_translation=full_translation,
        image_translation=image_translation,
        bbox=[left, top, right, bottom],
        confidence=confidence,
        confidence_threshold=confidence_threshold,
        estimated_original_font_size=estimated_original_font_size,
        used_font_size=used_font_size,
        final_estimated_text_height=round(final_h, 1),
        background_luminance=round(bg.get("mean_luminance", 0), 1),
        text_color=debug_text_color or ("black" if is_light else "white"),
        detected_by=detected_by,
        translated=translated,
        replaced=replaced,
        skip_reason=reason,
    )


# ---------------------------------------------------------------------------
# legacy per-line renderer (kept for backward compat / non-merge paths)
# ---------------------------------------------------------------------------


def erase_and_draw_translations(
    image: Image.Image,
    candidates: list[tuple[tuple[int, int, int, int], str]],
    translation_results: list[dict],
    output_path: str,
    jpeg_quality: int = 95,
) -> tuple[Image.Image, list[RegionDebug]]:
    """Legacy per-line renderer.  Prefer ``erase_and_draw_merged_paragraphs``."""
    img_w, img_h = image.size
    draw = ImageDraw.Draw(image)
    region_debug: list[RegionDebug] = []

    for idx, ((left, top, right, bottom), original_text) in enumerate(candidates):
        trans = translation_results[idx] if idx < len(translation_results) else {}
        full_trans = trans.get("full_translation", original_text)
        image_trans = trans.get("image_translation", full_trans)

        box_w = max(10, right - left - 4)
        box_h = max(10, bottom - top - 4)
        orig_fs = estimate_original_font_size((left, top, right, bottom))
        bg_info = estimate_surrounding_background(image, (left, top, right, bottom))

        is_title = orig_fs >= _TITLE_FONT_THRESHOLD
        draw_text = full_trans if is_title else image_trans
        if is_title:
            start_fs = max(22, round(orig_fs * 0.92))
            min_fs = max(22, round(orig_fs * 0.78))
            max_ln = 3
        else:
            start_fs = max(18, round(orig_fs * 0.88))
            min_fs = max(18, round(orig_fs * 0.72))
            max_ln = 5

        if bg_info["is_light"]:
            text_color = (20, 20, 20)
            erase_color = "white"
        else:
            text_color = (255, 255, 255)
            erase_color = (20, 20, 20)

        draw.rectangle((left, top, right, bottom), fill=erase_color)

        lines, final_fs, _draw_h, _exp, ts_debug = _typeset_english(
            draw, draw_text, box_w, box_h, orig_fs,
            img_w, img_h, img_bottom=bottom,
        )

        if lines is None:
            region_debug.append(_make_rd(
                original_text, full_trans, image_trans,
                left, top, right, bottom, orig_fs,
                bg_info, translated=True, replaced=False,
                skip_reason=f"text_too_long: {ts_debug.get('phase','?')}",
            ))
            continue

        font = _make_font(final_fs)
        lh = _line_height_for_font(final_fs)
        total_text_h = lh * len(lines)
        y_start = top + max(2, (box_h + 4 - total_text_h) // 2)

        for li, line in enumerate(lines):
            line_w = draw.textbbox((0, 0), line, font=font)[2]
            x = left + 2 + max(0, (box_w - line_w) // 2)
            draw.text((x, y_start + li * lh), line, fill=text_color, font=font)

        region_debug.append(_make_rd(
            original_text, full_trans, image_trans,
            left, top, right, bottom, orig_fs,
            bg_info, used_font_size=final_fs, final_h=0,
            translated=True, replaced=True, skip_reason=None,
            typeset_debug=ts_debug,
        ))

    image.save(output_path, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)
    return image, region_debug


# ---------------------------------------------------------------------------
# merged-paragraph rendering (Plan A: erase merged_bbox, layout English)
# ---------------------------------------------------------------------------


def erase_and_draw_merged_paragraphs(
    image: Image.Image,
    merged_paragraphs: list[MergedParagraph],
    translation_results: list[dict],
    output_path: str,
    jpeg_quality: int = 95,
) -> tuple[Image.Image, list[RegionDebug]]:
    """Erase each merged paragraph's bounding box and draw the English translation.

    This implements **Plan A**: the entire paragraph region is erased as one
    rectangle, then the English text is laid out inside it using the revised
    typesetting engine (start large, allow many lines, shrink only as needed,
    expand area as last resort).

    Args:
        image: PIL Image (RGB) to modify in-place.
        merged_paragraphs: List of ``MergedParagraph``.
        translation_results: List of ``{original, full_translation, image_translation}``,
            one per merged paragraph.
        output_path: Where to save the final JPEG.
        jpeg_quality: JPEG quality (1–100).

    Returns:
        ``(modified_image, region_debug_list)``.
    """
    img_w, img_h = image.size
    draw = ImageDraw.Draw(image)
    region_debug: list[RegionDebug] = []
    # Track drawn bboxes to prevent overlapping/repeated draws.
    drawn_bboxes: list[tuple[int, int, int, int]] = []

    for idx, para in enumerate(merged_paragraphs):
        trans = translation_results[idx] if idx < len(translation_results) else {}
        full_trans = trans.get("full_translation", para.merged_text)
        image_trans = trans.get("image_translation", full_trans)

        left, top, right, bottom = para.merged_bbox
        box_w = max(20, right - left - 6)
        box_h = max(10, bottom - top - 4)
        orig_fs = para.estimated_font_size or estimate_original_font_size(para.merged_bbox)

        # --- background check on merged bbox ---------------------------------
        bg_info = estimate_surrounding_background(image, para.merged_bbox)

        # --- text-height sanity check -----------------------------------------
        if box_h < 14:
            region_debug.append(_make_rd(
                para.merged_text, full_trans, image_trans,
                left, top, right, bottom, orig_fs,
                bg_info, translated=True, replaced=False,
                skip_reason="final_text_too_small",
            ))
            continue

        # --- adaptive erase based on background complexity -----------------
        complexity = estimate_background_complexity(image, para.merged_bbox)
        bg_rgb = estimate_background_color(image, para.merged_bbox)
        bg_lum = 0.2126 * bg_rgb[0] + 0.7152 * bg_rgb[1] + 0.0722 * bg_rgb[2]
        is_light_bg = bg_lum >= 150

        # --- detect text colour + role BEFORE erase ---------------------------
        ts_debug: dict = {}
        from .ocr import is_compact_label, classify_text_role  # noqa: F811
        is_compact, compact_reasons = is_compact_label(
            para.merged_bbox, (img_w, img_h), para.merged_text,
        )
        orig_tc, orig_tc_rgb, orig_tc_conf = detect_original_text_color(
            image, para.merged_bbox,
        )
        is_white_text = orig_tc == "white"

        text_role = classify_text_role(
            para.merged_bbox, (img_w, img_h), para.merged_text,
            font_size=orig_fs,
            is_white_text=is_white_text,
            is_compact=is_compact,
        )
        is_hero = text_role == "hero_headline"
        is_bottom_hero = text_role == "bottom_hero_headline"
        # Small/medium product detail captions in card grids. These are often
        # black title + gray subtitle, and gray text is hard to mask cleanly.
        para_w = max(1, right - left)
        para_h = max(1, bottom - top)
        is_feature_label = (
            not is_hero and not is_bottom_hero and not is_compact
            and top > img_h * 0.05
            and para_h <= img_h * 0.16
            and para_w <= img_w * 0.55
        )
        ts_debug["text_role"] = text_role
        if is_feature_label:
            ts_debug["feature_label"] = True

        # --- overlap check BEFORE erase/draw -------------------------------
        # If a later OCR fragment overlaps an already-rendered text block, do
        # not erase it. Erasing after a previous draw is what creates black
        # smears and missing redraws on large bottom headlines.
        overlap = False
        for prev_bbox in drawn_bboxes:
            pl, pt, pr, pb = prev_bbox
            il = max(left, pl); it = max(top, pt)
            ir = min(right, pr); ib = min(bottom, pb)
            if il < ir and it < ib:
                iou_area = (ir - il) * (ib - it)
                this_area = max(1, (right - left) * (bottom - top))
                if iou_area / this_area > 0.5:
                    overlap = True
                    break
        if overlap:
            region_debug.append(_make_rd(
                para.merged_text, full_trans, image_trans,
                left, top, right, bottom, orig_fs,
                bg_info, translated=True, replaced=False,
                skip_reason="overlap_with_previous_draw",
            ))
            continue

        # --- now erase (pass detected text color for mask strategy) ----------
        erase_tc = "white" if is_hero else ("black" if is_bottom_hero else orig_tc)
        if is_hero:
            erase_bbox = _expand_hero_erase_bbox(para.merged_bbox, img_w, img_h)
        elif is_bottom_hero:
            erase_bbox = _expand_bottom_hero_erase_bbox(para.merged_bbox, img_w, img_h)
            complexity = {
                "level": "simple",
                "recommended_erase": "color_fill",
                "color_variance": complexity.get("color_variance"),
                "edge_density": complexity.get("edge_density"),
            }
        elif is_feature_label:
            erase_bbox = _expand_feature_label_erase_bbox(para.merged_bbox, img_w, img_h)
            complexity = {
                "level": "simple",
                "recommended_erase": "color_fill",
                "color_variance": complexity.get("color_variance"),
                "edge_density": complexity.get("edge_density"),
            }
        else:
            erase_bbox = para.merged_bbox
        image, erase_debug = erase_text_region(
            image, para.merged_bbox, complexity,
            text_color=erase_tc,
            erase_bbox=erase_bbox,
        )
        draw = ImageDraw.Draw(image)

        # --- choose text colour: inherit original for hero headlines --------
        if is_hero:
            text_color = (255, 255, 255)  # force white
            ts_debug["text_color_inherited"] = "white_from_product_badge"
        elif is_bottom_hero:
            text_color = (12, 12, 12)
            ts_debug["text_color_inherited"] = "black_from_bottom_hero"
        elif is_light_bg:
            text_color = (20, 20, 20)
        else:
            text_color = (255, 255, 255)

        ts_debug["original_text_color"] = orig_tc
        ts_debug["final_text_color"] = "white" if text_color == (255, 255, 255) else "black"

        # --- choose text -----------------------------------------------------
        is_title = orig_fs >= _TITLE_FONT_THRESHOLD or is_hero or is_bottom_hero
        draw_text = full_trans
        if is_hero:
            draw_text = _shorten_hero_badge_text(para.merged_text, full_trans, image_trans)
            image_trans = draw_text
            full_trans = draw_text
            # Hero badges must stay inside the original left/top visual lane.
            box_w = max(20, min(box_w, img_w - left - 8))
            ts_debug["product_badge_shortened"] = True
            ts_debug["erase_bbox"] = list(erase_bbox)
        elif is_bottom_hero:
            draw_text = _shorten_bottom_hero_text(para.merged_text, full_trans, image_trans)
            image_trans = draw_text
            full_trans = draw_text
            box_w = max(20, min(box_w, img_w - left - 8))
            ts_debug["bottom_hero_shortened"] = True
            ts_debug["erase_bbox"] = list(erase_bbox)
        elif is_feature_label:
            fallback = _fallback_english_text(para.merged_text, full_trans, image_trans, text_role)
            # Feature cards need short label copy; use deterministic fallback
            # when model output is long, non-English, or paragraph-like.
            valid_now, _reason_now = validate_draw_text(image_trans or full_trans)
            candidate_words = (image_trans or full_trans or "").replace("\n", " ").split()
            if (not valid_now) or len(candidate_words) > 10 or "." in (image_trans or full_trans):
                draw_text = fallback
                image_trans = fallback
                full_trans = fallback
                ts_debug["feature_fallback_used"] = True
            else:
                draw_text = image_trans or full_trans
            box_w = max(20, min(box_w, img_w - left - 8))
            ts_debug["erase_bbox"] = list(erase_bbox)

        # --- detect alignment -------------------------------------------------
        if is_compact:
            alignment = "center"
        else:
            alignment = detect_text_alignment(para.lines, img_w)
        if is_hero or is_bottom_hero:
            alignment = "left"  # hero headlines are always left-aligned

        # --- typeset ---------------------------------------------------------
        if is_compact:
            lines, final_fs, draw_h, expanded_by, _ts = _typeset_compact_label(
                draw, draw_text, box_w, box_h, orig_fs,
                img_w, img_h, img_bottom=bottom,
            )
            ts_debug.update(_ts)
            ts_debug["compact_reasons"] = compact_reasons
        elif is_hero or is_bottom_hero:
            lines, final_fs, draw_h, expanded_by, _ts = _typeset_english(
                draw, draw_text, box_w, box_h, orig_fs,
                img_w, img_h, img_bottom=bottom, is_title=True,
                title_width_factor=1.0,
            )
            ts_debug.update(_ts)
            ts_debug["hero_headline"] = is_hero
            ts_debug["bottom_hero_headline"] = is_bottom_hero
        else:
            lines, final_fs, draw_h, expanded_by, _ts = _typeset_english(
                draw, draw_text, box_w, box_h, orig_fs,
                img_w, img_h, img_bottom=bottom,
            )
            ts_debug.update(_ts)

        # --- fallback ---------------------------------------------------------
        if lines is None and image_trans != full_trans:
            if is_compact:
                lines, final_fs, draw_h, expanded_by, _ts = _typeset_compact_label(
                    draw, image_trans, box_w, box_h, orig_fs,
                    img_w, img_h, img_bottom=bottom,
                )
                ts_debug.update(_ts)
            elif is_hero or is_bottom_hero:
                lines, final_fs, draw_h, expanded_by, _ts = _typeset_english(
                    draw, image_trans, box_w, box_h, orig_fs,
                    img_w, img_h, img_bottom=bottom, is_title=True,
                    title_width_factor=1.0,
                )
                ts_debug.update(_ts)
            else:
                lines, final_fs, draw_h, expanded_by, _ts = _typeset_english(
                    draw, image_trans, box_w, box_h, orig_fs,
                    img_w, img_h, img_bottom=bottom,
                )
                ts_debug.update(_ts)
            if lines is not None:
                draw_text = image_trans
                ts_debug.update(_ts)

        if lines is None:
            region_debug.append(_make_rd(
                para.merged_text, full_trans, image_trans,
                left, top, right, bottom, orig_fs,
                bg_info, translated=True, replaced=False,
                skip_reason=f"typeset_failed: {ts_debug.get('phase','?')}",
            ))
            continue

        # --- validate draw text before rendering ---------------------------
        draw_valid, draw_valid_reason = validate_draw_text(draw_text)
        if not draw_valid:
            fallback = _fallback_english_text(para.merged_text, full_trans, image_trans, text_role)
            fallback_valid, fallback_reason = validate_draw_text(fallback)
            if fallback_valid:
                draw_text = fallback
                image_trans = fallback
                full_trans = fallback
                ts_debug["draw_text_recovered"] = draw_valid_reason
                # Re-typeset after fallback so a skipped label still appears.
                if is_compact:
                    lines, final_fs, draw_h, expanded_by, _ts = _typeset_compact_label(
                        draw, draw_text, box_w, box_h, orig_fs,
                        img_w, img_h, img_bottom=bottom,
                    )
                else:
                    lines, final_fs, draw_h, expanded_by, _ts = _typeset_english(
                        draw, draw_text, box_w, box_h, orig_fs,
                        img_w, img_h, img_bottom=bottom,
                        is_title=(is_title or is_feature_label),
                        title_width_factor=1.0,
                    )
                ts_debug.update(_ts)
                if lines is None:
                    region_debug.append(_make_rd(
                        para.merged_text, full_trans, image_trans,
                        left, top, right, bottom, orig_fs,
                        bg_info, translated=True, replaced=False,
                        skip_reason=f"fallback_typeset_failed:{ts_debug.get('phase','?')}",
                    ))
                    continue
            else:
                region_debug.append(_make_rd(
                    para.merged_text, full_trans, image_trans,
                    left, top, right, bottom, orig_fs,
                    bg_info, translated=True, replaced=False,
                    skip_reason=f"draw_text_invalid:{draw_valid_reason};fallback:{fallback_reason}",
                ))
                continue

        # --- overlap check: skip if this area already drawn -----------------
        overlap = False
        for prev_bbox in drawn_bboxes:
            pl, pt, pr, pb = prev_bbox
            il = max(left, pl); it = max(top, pt)
            ir = min(right, pr); ib = min(bottom, pb)
            if il < ir and it < ib:
                iou_area = (ir - il) * (ib - it)
                this_area = max(1, (right - left) * (bottom - top))
                if iou_area / this_area > 0.5:
                    overlap = True
                    break
        if overlap:
            region_debug.append(_make_rd(
                para.merged_text, full_trans, image_trans,
                left, top, right, bottom, orig_fs,
                bg_info, translated=True, replaced=False,
                skip_reason="overlap_with_previous_draw",
            ))
            continue
        drawn_bboxes.append((left, top, right, bottom))

        ts_debug["draw_count"] = len(drawn_bboxes)
        ts_debug["draw_overlap"] = False
        ts_debug["draw_valid"] = True
        ts_debug["validated_text"] = draw_text[:60]

        # --- crisp text rendering: shadow + stroke + main text --------------
        font = _make_font(final_fs)
        font_path = _find_system_font() or "default"
        ts_debug["font_path"] = font_path
        # Warn if font may not support the text (e.g. Chinese in Arial).
        font_ok = _font_supports_text(draw_text, font)
        ts_debug["font_supports_text"] = font_ok

        lh = _line_height_for_font(final_fs)
        total_text_h = lh * len(lines)

        effective_bottom = bottom + expanded_by
        draw_area_h = effective_bottom - top
        y_start = top + max(2, (draw_area_h - total_text_h) // 2)

        ts_debug["alignment"] = alignment

        # Rendering strategy: subtle shadow/stroke → crisp main text on top.
        # Hero headlines always use shadow+stroke (white text on photos).
        shadow_enabled = (not is_light_bg) or is_hero
        shadow_offset = (1, 1)
        shadow_fill = (40, 40, 40, 100) if shadow_enabled else None
        stroke_width = 1 if (shadow_enabled and final_fs >= 20) else 0
        stroke_fill = (60, 60, 60, 80) if stroke_width > 0 else None

        ts_debug["font_size"] = final_fs
        ts_debug["fill_alpha"] = 255
        ts_debug["stroke_width"] = stroke_width
        ts_debug["glow_enabled"] = False
        ts_debug["glow_radius"] = 0
        ts_debug["shadow_enabled"] = shadow_enabled
        ts_debug["shadow_offset"] = str(shadow_offset) if shadow_enabled else "none"
        ts_debug["text_layer_blurred"] = False
        ts_debug["output_format"] = "JPEG"
        ts_debug["jpeg_quality"] = jpeg_quality

        for li, line in enumerate(lines):
            line_w = draw.textbbox((0, 0), line, font=font)[2]
            if alignment == "left":
                x = left + 2
            elif alignment == "right":
                x = right - 2 - line_w
            else:
                x = left + 2 + max(0, (box_w - line_w) // 2)
            y = y_start + li * lh

            # Layer 1: shadow (dark, offset, low opacity).
            if shadow_enabled:
                draw.text(
                    (x + shadow_offset[0], y + shadow_offset[1]),
                    line, fill=(40, 40, 40), font=font,
                )

            # Layer 2: stroke (thin outline).
            if stroke_width > 0:
                for dx in (-1, 1):
                    draw.text((x + dx, y), line, fill=(50, 50, 50), font=font)
                for dy in (-1, 1):
                    draw.text((x, y + dy), line, fill=(50, 50, 50), font=font)

            # Layer 3: CRISP main text — last layer, no blur, full opacity.
            draw.text((x, y), line, fill=text_color, font=font)

        # --- per-line debug entries for each original OCR line ---------------
        for ocl in para.lines:
            l_left, l_top, l_right, l_bottom = ocl.bbox
            # Merge erase debug into typeset debug.
            combined_debug = dict(ts_debug)
            combined_debug["erase_level"] = erase_debug.get("level", "?")
            combined_debug["erase_strategy"] = erase_debug.get("strategy", "?")
            if erase_debug.get("inpaint_radius", 0) > 0:
                combined_debug["inpaint_r"] = erase_debug["inpaint_radius"]
            if erase_debug.get("mask_pixels", 0) > 0:
                combined_debug["mask_px"] = erase_debug["mask_pixels"]
            if erase_debug.get("fallback_reason"):
                combined_debug["erase_fallback"] = erase_debug["fallback_reason"]

            region_debug.append(_make_rd(
                ocl.text, full_trans, image_trans,
                l_left, l_top, l_right, l_bottom,
                ocl.estimated_font_size or orig_fs,
                {
                    "is_light": ocl.is_light_background,
                    "mean_luminance": ocl.background_luminance,
                },
                used_font_size=final_fs,
                final_h=0,
                detected_by=ocl.detected_by,
                translated=True,
                replaced=True,
                skip_reason=None,
                typeset_debug=combined_debug,
                bg_color=bg_rgb,
            ))

    # --- save once -----------------------------------------------------------
    image.save(output_path, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)
    return image, region_debug


# ---------------------------------------------------------------------------
# draw-only renderer (for use AFTER Seedream or other external erase)
# ---------------------------------------------------------------------------


def draw_merged_paragraph_translations(
    image: Image.Image,
    merged_paragraphs: list[MergedParagraph],
    translation_results: list[dict],
    output_path: str,
    jpeg_quality: int = 95,
) -> tuple[Image.Image, list[RegionDebug]]:
    """Draw English translations onto an already-clean image (no erase step).

    This is the draw-only counterpart of
    :func:`erase_and_draw_merged_paragraphs`.  It assumes Chinese text has
    already been removed (e.g. by Seedream) and only handles PIL text
    rendering — typesetting, font sizing, alignment, colour, and shadow.

    Args:
        image: PIL Image (RGB) with Chinese text already erased.
        merged_paragraphs: List of ``MergedParagraph``.
        translation_results: List of ``{original, full_translation, image_translation}``,
            one per merged paragraph.
        output_path: Where to save the final JPEG.
        jpeg_quality: JPEG quality (1–100).

    Returns:
        ``(modified_image, region_debug_list)``.
    """
    img_w, img_h = image.size
    draw = ImageDraw.Draw(image)
    region_debug: list[RegionDebug] = []
    drawn_bboxes: list[tuple[int, int, int, int]] = []

    for idx, para in enumerate(merged_paragraphs):
        trans = translation_results[idx] if idx < len(translation_results) else {}
        full_trans = trans.get("full_translation", para.merged_text)
        image_trans = trans.get("image_translation", full_trans)

        left, top, right, bottom = para.merged_bbox
        box_w = max(20, right - left - 6)
        box_h = max(10, bottom - top - 4)
        orig_fs = para.estimated_font_size or estimate_original_font_size(para.merged_bbox)

        # --- background check on merged bbox ---------------------------------
        bg_info = estimate_surrounding_background(image, para.merged_bbox)
        bg_rgb = estimate_background_color(image, para.merged_bbox)
        bg_lum = 0.2126 * bg_rgb[0] + 0.7152 * bg_rgb[1] + 0.0722 * bg_rgb[2]
        is_light_bg = bg_lum >= 150

        # --- text-height sanity check -----------------------------------------
        if box_h < 14:
            region_debug.append(_make_rd(
                para.merged_text, full_trans, image_trans,
                left, top, right, bottom, orig_fs,
                bg_info, translated=True, replaced=False,
                skip_reason="final_text_too_small",
            ))
            continue

        # --- detect text colour + role (image is already clean) -----------------
        ts_debug: dict = {}
        from .ocr import is_compact_label, classify_text_role  # noqa: F811
        is_compact, compact_reasons = is_compact_label(
            para.merged_bbox, (img_w, img_h), para.merged_text,
        )
        orig_tc, orig_tc_rgb, orig_tc_conf = detect_original_text_color(
            image, para.merged_bbox,
        )
        is_white_text = orig_tc == "white"

        text_role = classify_text_role(
            para.merged_bbox, (img_w, img_h), para.merged_text,
            font_size=orig_fs,
            is_white_text=is_white_text,
            is_compact=is_compact,
        )
        is_hero = text_role == "hero_headline"
        is_bottom_hero = text_role == "bottom_hero_headline"
        para_w = max(1, right - left)
        para_h = max(1, bottom - top)
        is_feature_label = (
            not is_hero and not is_bottom_hero and not is_compact
            and top > img_h * 0.05
            and para_h <= img_h * 0.16
            and para_w <= img_w * 0.55
        )
        ts_debug["text_role"] = text_role

        # --- overlap check BEFORE draw -----------------------------------------
        overlap = False
        for prev_bbox in drawn_bboxes:
            pl, pt, pr, pb = prev_bbox
            il = max(left, pl); it = max(top, pt)
            ir = min(right, pr); ib = min(bottom, pb)
            if il < ir and it < ib:
                iou_area = (ir - il) * (ib - it)
                this_area = max(1, (right - left) * (bottom - top))
                if iou_area / this_area > 0.5:
                    overlap = True
                    break
        if overlap:
            region_debug.append(_make_rd(
                para.merged_text, full_trans, image_trans,
                left, top, right, bottom, orig_fs,
                bg_info, translated=True, replaced=False,
                skip_reason="overlap_with_previous_draw",
            ))
            continue

        # --- choose text colour ------------------------------------------------
        if is_hero:
            text_color = (255, 255, 255)
            ts_debug["text_color_inherited"] = "white_from_product_badge"
        elif is_bottom_hero:
            text_color = (12, 12, 12)
            ts_debug["text_color_inherited"] = "black_from_bottom_hero"
        elif is_light_bg:
            text_color = (20, 20, 20)
        else:
            text_color = (255, 255, 255)

        ts_debug["original_text_color"] = orig_tc
        ts_debug["final_text_color"] = "white" if text_color == (255, 255, 255) else "black"

        # --- choose text -------------------------------------------------------
        is_title = orig_fs >= _TITLE_FONT_THRESHOLD or is_hero or is_bottom_hero
        draw_text = full_trans
        if is_hero:
            draw_text = _shorten_hero_badge_text(para.merged_text, full_trans, image_trans)
            image_trans = draw_text
            full_trans = draw_text
            box_w = max(20, min(box_w, img_w - left - 8))
            ts_debug["product_badge_shortened"] = True
        elif is_bottom_hero:
            draw_text = _shorten_bottom_hero_text(para.merged_text, full_trans, image_trans)
            image_trans = draw_text
            full_trans = draw_text
            box_w = max(20, min(box_w, img_w - left - 8))
            ts_debug["bottom_hero_shortened"] = True
        elif is_feature_label:
            fallback = _fallback_english_text(para.merged_text, full_trans, image_trans, text_role)
            valid_now, _reason_now = validate_draw_text(image_trans or full_trans)
            candidate_words = (image_trans or full_trans or "").replace("\n", " ").split()
            if (not valid_now) or len(candidate_words) > 10 or "." in (image_trans or full_trans):
                draw_text = fallback
                image_trans = fallback
                full_trans = fallback
                ts_debug["feature_fallback_used"] = True
            else:
                draw_text = image_trans or full_trans
            box_w = max(20, min(box_w, img_w - left - 8))

        # --- detect alignment ---------------------------------------------------
        if is_compact:
            alignment = "center"
        else:
            alignment = detect_text_alignment(para.lines, img_w)
        if is_hero or is_bottom_hero:
            alignment = "left"

        # --- typeset -----------------------------------------------------------
        if is_compact:
            lines, final_fs, draw_h, expanded_by, _ts = _typeset_compact_label(
                draw, draw_text, box_w, box_h, orig_fs,
                img_w, img_h, img_bottom=bottom,
            )
            ts_debug.update(_ts)
            ts_debug["compact_reasons"] = compact_reasons
        elif is_hero or is_bottom_hero:
            lines, final_fs, draw_h, expanded_by, _ts = _typeset_english(
                draw, draw_text, box_w, box_h, orig_fs,
                img_w, img_h, img_bottom=bottom, is_title=True,
                title_width_factor=1.0,
            )
            ts_debug.update(_ts)
            ts_debug["hero_headline"] = is_hero
            ts_debug["bottom_hero_headline"] = is_bottom_hero
        else:
            lines, final_fs, draw_h, expanded_by, _ts = _typeset_english(
                draw, draw_text, box_w, box_h, orig_fs,
                img_w, img_h, img_bottom=bottom,
            )
            ts_debug.update(_ts)

        # --- fallback to image_translation ------------------------------------
        if lines is None and image_trans != full_trans:
            if is_compact:
                lines, final_fs, draw_h, expanded_by, _ts = _typeset_compact_label(
                    draw, image_trans, box_w, box_h, orig_fs,
                    img_w, img_h, img_bottom=bottom,
                )
                ts_debug.update(_ts)
            elif is_hero or is_bottom_hero:
                lines, final_fs, draw_h, expanded_by, _ts = _typeset_english(
                    draw, image_trans, box_w, box_h, orig_fs,
                    img_w, img_h, img_bottom=bottom, is_title=True,
                    title_width_factor=1.0,
                )
                ts_debug.update(_ts)
            else:
                lines, final_fs, draw_h, expanded_by, _ts = _typeset_english(
                    draw, image_trans, box_w, box_h, orig_fs,
                    img_w, img_h, img_bottom=bottom,
                )
                ts_debug.update(_ts)
            if lines is not None:
                draw_text = image_trans
                ts_debug.update(_ts)

        if lines is None:
            region_debug.append(_make_rd(
                para.merged_text, full_trans, image_trans,
                left, top, right, bottom, orig_fs,
                bg_info, translated=True, replaced=False,
                skip_reason=f"typeset_failed: {ts_debug.get('phase','?')}",
            ))
            continue

        # --- validate draw text ------------------------------------------------
        draw_valid, draw_valid_reason = validate_draw_text(draw_text)
        if not draw_valid:
            fallback = _fallback_english_text(para.merged_text, full_trans, image_trans, text_role)
            fallback_valid, fallback_reason = validate_draw_text(fallback)
            if fallback_valid:
                draw_text = fallback
                image_trans = fallback
                full_trans = fallback
                ts_debug["draw_text_recovered"] = draw_valid_reason
                if is_compact:
                    lines, final_fs, draw_h, expanded_by, _ts = _typeset_compact_label(
                        draw, draw_text, box_w, box_h, orig_fs,
                        img_w, img_h, img_bottom=bottom,
                    )
                else:
                    lines, final_fs, draw_h, expanded_by, _ts = _typeset_english(
                        draw, draw_text, box_w, box_h, orig_fs,
                        img_w, img_h, img_bottom=bottom,
                        is_title=(is_title or is_feature_label),
                        title_width_factor=1.0,
                    )
                ts_debug.update(_ts)
                if lines is None:
                    region_debug.append(_make_rd(
                        para.merged_text, full_trans, image_trans,
                        left, top, right, bottom, orig_fs,
                        bg_info, translated=True, replaced=False,
                        skip_reason=f"fallback_typeset_failed:{ts_debug.get('phase','?')}",
                    ))
                    continue
            else:
                region_debug.append(_make_rd(
                    para.merged_text, full_trans, image_trans,
                    left, top, right, bottom, orig_fs,
                    bg_info, translated=True, replaced=False,
                    skip_reason=f"draw_text_invalid:{draw_valid_reason};fallback:{fallback_reason}",
                ))
                continue

        # --- overlap recheck --------------------------------------------------
        overlap = False
        for prev_bbox in drawn_bboxes:
            pl, pt, pr, pb = prev_bbox
            il = max(left, pl); it = max(top, pt)
            ir = min(right, pr); ib = min(bottom, pb)
            if il < ir and it < ib:
                iou_area = (ir - il) * (ib - it)
                this_area = max(1, (right - left) * (bottom - top))
                if iou_area / this_area > 0.5:
                    overlap = True
                    break
        if overlap:
            region_debug.append(_make_rd(
                para.merged_text, full_trans, image_trans,
                left, top, right, bottom, orig_fs,
                bg_info, translated=True, replaced=False,
                skip_reason="overlap_with_previous_draw",
            ))
            continue
        drawn_bboxes.append((left, top, right, bottom))

        ts_debug["draw_count"] = len(drawn_bboxes)
        ts_debug["draw_overlap"] = False
        ts_debug["draw_valid"] = True
        ts_debug["validated_text"] = draw_text[:60]

        # --- crisp text rendering ---------------------------------------------
        font = _make_font(final_fs)
        font_path = _find_system_font() or "default"
        ts_debug["font_path"] = font_path
        font_ok = _font_supports_text(draw_text, font)
        ts_debug["font_supports_text"] = font_ok

        lh = _line_height_for_font(final_fs)
        total_text_h = lh * len(lines)

        effective_bottom = bottom + expanded_by
        draw_area_h = effective_bottom - top
        y_start = top + max(2, (draw_area_h - total_text_h) // 2)

        ts_debug["alignment"] = alignment

        shadow_enabled = (not is_light_bg) or is_hero
        shadow_offset = (1, 1)
        stroke_width = 1 if (shadow_enabled and final_fs >= 20) else 0

        ts_debug["font_size"] = final_fs
        ts_debug["fill_alpha"] = 255
        ts_debug["stroke_width"] = stroke_width
        ts_debug["glow_enabled"] = False
        ts_debug["glow_radius"] = 0
        ts_debug["shadow_enabled"] = shadow_enabled
        ts_debug["shadow_offset"] = str(shadow_offset) if shadow_enabled else "none"
        ts_debug["text_layer_blurred"] = False
        ts_debug["output_format"] = "JPEG"
        ts_debug["jpeg_quality"] = jpeg_quality

        for li, line in enumerate(lines):
            line_w = draw.textbbox((0, 0), line, font=font)[2]
            if alignment == "left":
                x = left + 2
            elif alignment == "right":
                x = right - 2 - line_w
            else:
                x = left + 2 + max(0, (box_w - line_w) // 2)
            y = y_start + li * lh

            if shadow_enabled:
                draw.text(
                    (x + shadow_offset[0], y + shadow_offset[1]),
                    line, fill=(40, 40, 40), font=font,
                )

            if stroke_width > 0:
                for dx in (-1, 1):
                    draw.text((x + dx, y), line, fill=(50, 50, 50), font=font)
                for dy in (-1, 1):
                    draw.text((x, y + dy), line, fill=(50, 50, 50), font=font)

            draw.text((x, y), line, fill=text_color, font=font)

        # --- per-line debug entries for each original OCR line ---------------
        for ocl in para.lines:
            l_left, l_top, l_right, l_bottom = ocl.bbox
            region_debug.append(_make_rd(
                ocl.text, full_trans, image_trans,
                l_left, l_top, l_right, l_bottom,
                ocl.estimated_font_size or orig_fs,
                {
                    "is_light": ocl.is_light_background,
                    "mean_luminance": ocl.background_luminance,
                },
                used_font_size=final_fs,
                final_h=0,
                detected_by=ocl.detected_by,
                translated=True,
                replaced=True,
                skip_reason=None,
                typeset_debug=ts_debug,
                bg_color=bg_rgb,
            ))

    # --- save once -----------------------------------------------------------
    image.save(output_path, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)
    return image, region_debug
