"""Seedream image-editing service: erase Chinese text from product images.

Uses the Volcengine Ark Seedream API (``/images/generations``) for
image-to-image inpainting.  Seedream is ONLY asked to erase Chinese text
and restore backgrounds — it must NOT generate any new text.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import httpx
from PIL import Image

from .config import ModuleConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# prompt builder
# ---------------------------------------------------------------------------


def _build_erase_prompt(text_regions: list[dict]) -> str:
    """Build a Seedream prompt that asks to erase Chinese text only.

    The prompt includes approximate bounding-box descriptions so the model
    knows *where* to erase, without asking it to draw any replacement text.
    """
    region_descriptions: list[str] = []
    for i, region in enumerate(text_regions, start=1):
        bbox = region.get("bbox", [])
        original = region.get("original_text", "")
        if len(bbox) == 4:
            left, top, right, bottom = bbox
            region_descriptions.append(
                f"  Region {i}: [{left}, {top}, {right}, {bottom}] "
                f"— original text: 「{original[:30]}」"
            )
        elif original:
            region_descriptions.append(
                f"  Region {i}: (no bbox) — original text: 「{original[:30]}」"
            )

    region_block = "\n".join(region_descriptions) if region_descriptions else (
        "  (detect and erase all Chinese text automatically)"
    )

    return (
        "请基于原图进行局部修复，只擦除图片中的中文文字，不要生成任何新文字。\n"
        "\n"
        "任务：\n"
        "删除图片中所有中文文字以及文字边缘残影，保留文字下方原本的背景、产品、"
        "卡片、光影、纹理、材质和颜色。\n"
        "\n"
        "严格要求：\n"
        "1. 只擦除中文文字。\n"
        "2. 不要翻译。\n"
        "3. 不要生成英文。\n"
        "4. 不要添加任何新文字。\n"
        "5. 非中文文字保持原样。\n"
        "6. 不要改变产品、背景、构图、卡片、圆角、阴影、光线、颜色和材质。\n"
        "7. 擦除区域要与周围背景自然融合，看不出原来有文字。\n"
        "8. 不要改变图片比例和清晰度。\n"
        "9. 输出一张无中文文字的干净底图。\n"
        "\n"
        "需要擦除的文字区域坐标如下：\n"
        f"{region_block}"
    )


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------


def _image_to_data_url(image_path: Path) -> str:
    """Read a local image and return a ``data:image/...;base64,...`` URL."""
    suffix = image_path.suffix.lower().lstrip(".")
    media_type = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/{media_type};base64,{encoded}"



# ---------------------------------------------------------------------------
# main API
# ---------------------------------------------------------------------------


async def erase_chinese_text_with_seedream(
    image_path: Path,
    text_regions: list[dict],
    output_path: Path,
    config: ModuleConfig,
) -> Path:
    """Erase Chinese text from *image_path* using the Ark Seedream API.

    Args:
        image_path: Path to the source product image.
        text_regions: List of dicts, each with at least ``bbox`` (4 ints) and
            ``original_text`` (str).  Used to build the erase prompt.
        output_path: Where to save the cleaned image.
        config: ModuleConfig with Ark / Seedream settings.

    Returns:
        *output_path* pointing to the cleaned image.

    Raises:
        ValueError: If Ark API key is missing.
        RuntimeError: If the Seedream API call or download fails.
    """
    if not config.ark_api_key:
        raise ValueError("ARK_API_KEY is not configured — cannot call Seedream.")

    # --- build prompt ---------------------------------------------------------
    prompt = _build_erase_prompt(text_regions)

    # --- build request --------------------------------------------------------
    url = f"{config.ark_base_url.rstrip('/')}/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.ark_api_key}",
    }
    payload: dict = {
        "model": config.ark_seedream_model,
        "prompt": prompt,
        "image": _image_to_data_url(image_path),
        "response_format": "url",
        "size": "2K",
        "stream": False,
        "watermark": config.seedream_watermark,
    }

    logger.info(
        "Calling Seedream erase — model=%s image=%s regions=%d",
        config.ark_seedream_model,
        image_path.name,
        len(text_regions),
    )

    # --- call Ark API ---------------------------------------------------------
    try:
        async with httpx.AsyncClient(timeout=config.seedream_timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Seedream API request failed: {exc}") from exc

    # --- parse response -------------------------------------------------------
    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Seedream returned non-JSON response: {response.text[:500]}"
        ) from exc

    # The Ark / OpenAI-images shape: {"data": [{"url": "..."}]}
    image_data = data.get("data")
    if not isinstance(image_data, list) or len(image_data) == 0:
        raise RuntimeError(
            f"Seedream response missing 'data' array: {json.dumps(data, ensure_ascii=False)[:500]}"
        )

    result_url = image_data[0].get("url") or image_data[0].get("b64_json")
    if not result_url:
        raise RuntimeError(
            f"Seedream response has no image url: {json.dumps(image_data[0], ensure_ascii=False)[:500]}"
        )

    logger.info("Seedream erase complete — downloading result from %s...", result_url[:80])

    # --- download result ------------------------------------------------------
    await _download_result_async(result_url, output_path, timeout=config.seedream_timeout)

    return output_path


async def _download_result_async(
    url: str,
    output_path: Path,
    timeout: int = 60,
) -> Path:
    """Async wrapper around the download logic."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Failed to download Seedream result image: {exc}") from exc

    output_path.write_bytes(response.content)

    # Validate image.
    try:
        with Image.open(output_path) as img:
            img.verify()
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Seedream result is not a valid image: {exc}"
        ) from exc

    return output_path
