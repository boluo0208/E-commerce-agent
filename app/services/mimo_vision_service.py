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

You will receive a product title before the image. First infer what the MAIN
PRODUCT is from that title, then inspect the image and find that same product.
Use the title only to decide which object is the product; do not invent facts
that are not visible in the image.

Color extraction rules:
1. Return ONLY the color of the title-matched main product.
2. Ignore background color, packaging, cards, text, props, model skin tone,
   mannequin color, hands, legs, shoes, bags, jewelry, and other accessories.
3. If the image contains several objects, choose the object that matches the
   title. Example: if the title says socks, inspect the socks only; ignore feet,
   skin, floor, and anti-slip dots unless the dots are the dominant visible
   surface of the socks.
4. If the title says clothing, inspect the garment, not the model or styling
   accessories.
5. If the product has multiple visible colors, return them in "color" separated
   by "/" (e.g. "Pink/White"). For "color_label", choose the single dominant
   base color of the product itself.
6. If the product cannot be found or its color is unclear, return "unknown".

For "color": Return the visible main product color(s) in English, such as
"Pink", "Black/White", or "Purple/Pink/White".

For "color_label": You MUST return exactly ONE single normalized color label
from the list below, in Chinese. Do not return any other value.
Allowed labels: 黑色, 白色, 灰色, 银色, 红色, 粉色, 橙色, 黄色, 绿色, 蓝色, 紫色, 棕色, 米色, 金色.

For "category": Use a standard e-commerce category like "T-Shirt", "Dress",
"Socks", "Sneakers", "Backpack", etc. Prefer the category implied by the title
when it matches a visible product in the image. Return "unknown" if unclear.

For "style": Describe the visible style concisely (e.g. "Casual", "Sporty",
"Formal", "Streetwear").

For "visible_features": List visible design elements of the title-matched
product only (e.g. "V-neck", "logo on chest", "striped pattern").

Do not guess material, dimensions, brand, quantity, or hidden features."""

ALLOWED_COLOR_LABELS = {
    "黑色", "白色", "灰色", "银色", "红色", "粉色", "橙色",
    "黄色", "绿色", "蓝色", "紫色", "棕色", "米色", "金色",
}


def _image_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".")
    media_type = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/{media_type};base64,{encoded}"


def _build_vision_prompt(product_title: str) -> str:
    title = (product_title or "").strip() or "unknown"
    return (
        f"Product title: {title}\n\n"
        "Use this title to identify the main product in the image, then follow "
        "the instructions below.\n\n"
        f"{VISION_PROMPT}"
    )


def _mock_vision_result(image_path: Path) -> VisionResult:
    with Image.open(image_path) as image:
        width, height = image.size

    return VisionResult(
        category="unknown",
        color="unknown",
        color_label="unknown",
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


async def analyze_product_image_with_mimo(
    image_path: Path,
    product_title: str = "",
) -> VisionResult:
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
                    {"type": "text", "text": _build_vision_prompt(product_title)},
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

    raw_label = str(data.get("color_label") or "unknown").strip()
    if raw_label not in ALLOWED_COLOR_LABELS:
        raw_label = "unknown"

    return VisionResult(
        category=str(data.get("category") or "unknown"),
        color=str(data.get("color") or "unknown"),
        color_label=raw_label,
        style=str(data.get("style") or "unknown"),
        visible_features=list(data.get("visible_features") or []),
        image_width=width,
        image_height=height,
        note="Analyzed by Mimo vision model using the product title as the main-product anchor.",
    )
