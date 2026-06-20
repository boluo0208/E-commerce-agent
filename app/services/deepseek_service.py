import json

import httpx
from fastapi import HTTPException

from app.core.config import settings
from app.schemas.product import ProductContent, VisionResult


SYSTEM_PROMPT = """You are a professional cross-border e-commerce product copy editor.
Generate concise, natural, customer-friendly product content for global marketplaces
such as Amazon, AliExpress, Noon, and similar platforms.

Rules:
1. English title: use natural English word order. Preferred structure:
   {Brand} {Model/Series} {Gender/Audience} {Product Type} {Season/Year}.
   Example: "On Court-T Fade Men's Tennis T-Shirt Spring/Summer 2026".
2. Treat short leading words such as "On", "In", and "Off" as possible brand
   names, not as prepositions. Do not translate official brand names.
3. Do not repeat brand, series, gender, or product type.
4. Arabic title: write a natural Arabic product title matching the same product
   information. Keep brand names in English unless there is a standard Arabic name.
5. Chinese description: write 3-6 natural Chinese e-commerce sentences based only
   on the title and vision result.  Do NOT mention colour — colour is already listed
   in a separate field.
6. English description: write 3-6 natural English sentences with the same facts.
   Do NOT mention colour.
7. Do not invent material, functions, certifications, colors, sizes, dimensions,
   weight, origins, compatibility, or selling points that are not in the input.
8. Avoid exaggerated claims such as "best", "No.1", "premium", or "guaranteed".
9. Return valid JSON only, with no Markdown fences or extra text."""


def _mock_content(chinese_title: str, vision_result: VisionResult) -> ProductContent:
    visible = "; ".join(vision_result.visible_features)
    return ProductContent(
        english_title=f"{chinese_title} - Product Listing",
        arabic_title=f"{chinese_title} - Product Title",
        chinese_description=(
            f"商品标题：{chinese_title}。图片已上传并完成基础检查。"
            f"可见信息：{visible}"
        ),
        english_description=(
            f"Product: {chinese_title}. The image has been uploaded and processed. "
            f"Visible information: {visible}"
        ),
        safety_notes=[
            "Mock content was generated because DEEPSEEK_API_KEY is not configured.",
            "No material, size, or functional claims were added.",
        ],
    )


def _build_user_prompt(chinese_title: str, vision_result: VisionResult) -> str:
    payload = {
        "chinese_title": chinese_title,
        "vision_result": vision_result.model_dump(),
        "instructions": {
            "english_title": (
                "Create one natural marketplace title. Keep brand/model terms from "
                "the Chinese title. If the brand is On/昂跑, use 'On' as the brand. "
                "Do not produce awkward phrases such as 'On Men's Court-T...'; prefer "
                "'On Court-T Fade Men's Tennis T-Shirt Spring/Summer 2026'."
            ),
            "arabic_title": "Natural Arabic title for Middle Eastern e-commerce.",
            "chinese_description": (
                "3-6 natural Chinese e-commerce sentences. Factual, visible, no hype. "
                "Do NOT mention colour."
            ),
            "english_description": (
                "3-6 natural English sentences matching the same facts. "
                "Do NOT mention colour."
            ),
        },
        "required_json_schema": {
            "english_title": "string",
            "arabic_title": "string",
            "chinese_description": "string",
            "english_description": "string",
            "safety_notes": ["string"],
        },
    }
    return (
        "Generate multilingual product content from this JSON. "
        "Return JSON only.\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


async def generate_content_with_deepseek(
    chinese_title: str,
    vision_result: VisionResult,
) -> ProductContent:
    if not settings.deepseek_api_key:
        if settings.mock_llm_when_no_key:
            return _mock_content(chinese_title, vision_result)
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY is not configured")

    url = f"{settings.deepseek_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.deepseek_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.deepseek_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(chinese_title, vision_result)},
        ],
        "temperature": 0.25,
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"DeepSeek request failed: {exc}") from exc

    raw_content = response.json()["choices"][0]["message"]["content"]
    try:
        return ProductContent.model_validate_json(raw_content)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="DeepSeek returned invalid JSON") from exc
