import base64
import json
from pathlib import Path

import httpx
from fastapi import HTTPException
from PIL import Image

from app.core.config import settings
from app.schemas.product import VisionResult


VISION_PROMPT = """Identify only visible product information in this image.
Return JSON only with these fields:
category, color, color_label, style, visible_features.

For "color": Return ONLY the main product/item color(s). IGNORE background color, model skin tone, mannequin color, and accessory colors. If the product has multiple colors (gradient, color-block, multi-panel), list them separated by "/" (e.g. "Purple/Pink/White"). If you cannot determine the product color, return "unknown".

For "color_label": You MUST return exactly ONE single color label from the list below, in Chinese.
  This is the colour of the PRODUCT ITSELF — NOT the background, NOT the model's skin, NOT a handbag or shoe the model is wearing, NOT any accessory.
  Look at the clothing/item that is the main subject of the image and pick its dominant colour.
  Allowed labels: 黑色, 白色, 灰色, 银色, 红色, 粉色, 橙色, 黄色, 绿色, 蓝色, 紫色, 棕色, 米色, 金色.
  If the product has multiple colours (e.g. stripes, colour-block, gradient), pick the single most dominant one. Do NOT return any other value.

For "category": Use a standard e-commerce category like "T-Shirt", "Dress", "Sneakers", "Backpack", etc. Return "unknown" if unclear.

For "style": Describe the visual style concisely (e.g. "Casual", "Sporty", "Formal", "Streetwear").

For "visible_features": List visible design elements (e.g. "V-neck", "logo on chest", "striped pattern").

Do not guess material, dimensions, brand, quantity, or hidden features."""

TEXT_REGION_PROMPT = """Find visible Chinese text regions in this product image.
Return JSON only:
{"items":[{"text":"Chinese text","bbox":[left,top,right,bottom]}]}
bbox values must be normalized coordinates from 0 to 1.
Only include text printed on the image. Skip model names, English text, logos, and uncertain text.
If there is no Chinese text, return {"items":[]}."""


def _image_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".")
    media_type = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/{media_type};base64,{encoded}"


def _mock_vision_result(image_path: Path) -> VisionResult:
    with Image.open(image_path) as image:
        width, height = image.size

    return VisionResult(
        category="unknown",
        color="unknown",
        color_label="未知",
        style="unknown",
        visible_features=[
            "Product image was uploaded successfully.",
            "Mimo vision model is not connected yet.",
        ],
        image_width=width,
        image_height=height,
        note="Mock Mimo result. Configure MIMO_API_KEY, MIMO_BASE_URL, and MIMO_MODEL to enable real vision analysis.",
    )


def _parse_mimo_json(raw_content: str) -> dict:
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Mimo returned invalid JSON") from exc


async def analyze_product_image_with_mimo(image_path: Path) -> VisionResult:
    if not settings.mimo_api_key or not settings.mimo_base_url or not settings.mimo_model:
        if settings.mock_vision_when_no_key:
            return _mock_vision_result(image_path)
        raise HTTPException(status_code=500, detail="Mimo vision settings are not configured")

    with Image.open(image_path) as image:
        width, height = image.size

    url = f"{settings.mimo_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.mimo_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.mimo_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image_path)},
                    },
                ],
            }
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Mimo vision request failed: {exc}") from exc

    raw_content = response.json()["choices"][0]["message"]["content"]
    data = _parse_mimo_json(raw_content)

    # Normalise color_label to the allowed set.
    _ALLOWED_LABELS = {
        "黑色", "白色", "灰色", "银色", "红色", "粉色", "橙色",
        "黄色", "绿色", "蓝色", "紫色", "棕色", "米色", "金色",
    }
    raw_label = str(data.get("color_label") or "未知").strip()
    if raw_label not in _ALLOWED_LABELS:
        raw_label = "unknown"

    return VisionResult(
        category=str(data.get("category") or "unknown"),
        color=str(data.get("color") or "unknown"),
        color_label=raw_label,
        style=str(data.get("style") or "unknown"),
        visible_features=list(data.get("visible_features") or []),
        image_width=width,
        image_height=height,
        note="Analyzed by Mimo vision model.",
    )


async def detect_chinese_text_regions_with_mimo(image_path: Path) -> list[dict]:
    if not settings.mimo_api_key or not settings.mimo_base_url or not settings.mimo_model:
        return []

    url = f"{settings.mimo_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.mimo_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.mimo_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TEXT_REGION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image_path)},
                    },
                ],
            }
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPError:
        return []

    try:
        raw_content = response.json()["choices"][0]["message"]["content"]
        data = json.loads(raw_content)
    except (KeyError, json.JSONDecodeError):
        return []

    items = data.get("items")
    return items if isinstance(items, list) else []
