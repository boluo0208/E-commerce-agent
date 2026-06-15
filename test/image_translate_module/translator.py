"""Translation: calls an OpenAI-compatible LLM API to translate Chinese→English.

Extracted from app/services/deepseek_service.py – zero coupling to the main app.
Config is passed explicitly; no dependency on app.core.config.
"""

import json

import httpx

from .config import ModuleConfig
from .schemas import TranslateError

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

TRANSLATION_SYSTEM_PROMPT = (
    "You translate Chinese text printed on product images into natural "
    "English for the same image. Preserve the original meaning and tone. "
    "Do not summarize a headline into disconnected keywords. "
    "Do not split one phrase into separate concepts. "
    "Use natural e-commerce English, not literal word-for-word translation. "
    "For slogans/headlines, prefer a compact natural phrase such as "
    "'Move Freely on the Court' rather than 'Court Freedom'. "
    "For feature labels, use concise labels such as 'Side Slits for Motion'. "
    "Do not invent material, functions, numbers, or claims. "
    "Keep brand names, model names, years, sizes, and units unchanged. "
    "Return exactly one translation object for each input item, in order. "
    "Return valid JSON only."
)


def _build_translation_user_message(
    texts: list[str],
    text_types: list[str] | None = None,
) -> str:
    """Build the user message, tagging compact-label items for short-translation mode."""
    items: list[dict] = []
    for i, t in enumerate(texts):
        entry: dict = {"text": t}
        if text_types and i < len(text_types):
            entry["type"] = text_types[i]
        items.append(entry)

    return json.dumps(
        {
            "items": items,
            "rules": {
                "full_translation": (
                    "Complete natural English translation. For headlines, "
                    "make it a natural headline phrase."
                ),
                "image_translation": (
                    "English to draw back on the image. Keep it natural. "
                    "Use 2-8 words for labels; 3-10 words for headlines. "
                    "Do not shorten so much that meaning becomes awkward."
                ),
            },
            "hero_headline_rules": {
                "applies_to": "items where type='hero_headline'",
                "full_translation": (
                    "Short punchy English for a hero product badge at the top "
                    "of a product image. 6-12 words max. Use natural line breaks. "
                    "Preserve key numbers but convert Chinese units: "
                    "'斤' (jin) → kg (divide by 2). "
                    "'XL' stays 'XL'. Keep weight ranges like '40-80 kg'. "
                    "Write compact, natural e-commerce badge copy. "
                    "Do NOT produce long sentences. "
                    "Examples: 'Imported Elastic, Snug Fit / Fits 40-80 kg'"
                ),
                "image_translation": (
                    "Same as full_translation — keep it 6-10 words, "
                    "2 lines max. E-commerce badge style."
                ),
            },
            "bottom_hero_headline_rules": {
                "applies_to": "items where type='bottom_hero_headline'",
                "full_translation": (
                    "Short bold English for the main bottom headline on a "
                    "product image. 5-10 words max, preferably 2 lines. "
                    "Keep the key selling point, but do NOT produce a full "
                    "sentence or paragraph. Examples: 'Full-Sole Grip / "
                    "Cushions Every Step', 'No-Slip Support / For Every Move'."
                ),
                "image_translation": (
                    "Same as full_translation. Use punchy headline copy, "
                    "2 lines max, 5-10 words."
                ),
            },
            "compact_label_rules": {
                "applies_to": "items where type='compact_label'",
                "full_translation": (
                    "Short marketing English for a small promo badge. "
                    "2-5 words maximum. Keep numbers (¥1600, 50% etc). "
                    "Natural e-commerce badge copy. "
                    "Examples: 'Member Gift', 'Free Gift Over ¥1600', "
                    "'Buy 2 Get 1 Free', 'Limited Offer'."
                ),
                "image_translation": (
                    "Same as full_translation for compact labels — "
                    "keep it very short, 2-4 words. "
                    "Examples: 'Member Gift', 'Over ¥1600 Gift'."
                ),
            },
            "required_json_schema": {
                "translations": [
                    {
                        "original": "string",
                        "full_translation": "string",
                        "image_translation": "string",
                    }
                ]
            },
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# main API
# ---------------------------------------------------------------------------


def _mock_translations(texts: list[str]) -> list[dict]:
    """Return identity translations when no API key is configured."""
    return [
        {"original": t, "full_translation": t, "image_translation": t}
        for t in texts
    ]


async def translate_texts_to_english(
    texts: list[str],
    config: ModuleConfig,
    text_types: list[str] | None = None,
) -> list[dict]:
    """Translate a batch of Chinese strings to English via LLM.

    Args:
        texts: List of Chinese text strings to translate.
        config: Module configuration.
        text_types: Optional per-item type hints: ``"title"``, ``"body"``,
            ``"compact_label"``.  Affects the translation prompt.

    Returns:
        List of dicts: ``[{original, full_translation, image_translation}, ...]``

    Raises:
        TranslateError: When the API call or response parsing fails.
    """
    if not texts:
        return []

    if not config.translate_api_key:
        if config.mock_translate_when_no_key:
            return _mock_translations(texts)
        raise TranslateError("TRANSLATE_API_KEY is not configured")

    url = f"{config.translate_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.translate_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.translate_model,
        "messages": [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": _build_translation_user_message(texts, text_types)},
        ],
        "temperature": config.translate_temperature,
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=config.translate_timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise TranslateError(f"Translation API request failed: {exc}") from exc

    try:
        data = json.loads(response.json()["choices"][0]["message"]["content"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise TranslateError("Translation API returned invalid JSON") from exc

    # Handle both {"translations": [...]} and bare [...] responses.
    if isinstance(data, list):
        translations = data
    elif isinstance(data, dict):
        translations = data.get("translations")
    else:
        translations = None

    if not isinstance(translations, list):
        raise TranslateError("Translation JSON missing 'translations' key")

    # Pad or trim to match input length.
    if len(translations) < len(texts):
        translations = list(translations) + [
            texts[i] for i in range(len(translations), len(texts))
        ]
    elif len(translations) > len(texts):
        translations = translations[: len(texts)]

    result: list[dict] = []
    for idx, item in enumerate(translations):
        if isinstance(item, str):
            result.append({
                "original": texts[idx],
                "full_translation": item.strip(),
                "image_translation": item.strip(),
            })
        elif isinstance(item, dict):
            ft = str(item.get("full_translation") or item.get("image_translation") or texts[idx]).strip()
            it = str(item.get("image_translation") or item.get("full_translation") or texts[idx]).strip()
            result.append({
                "original": str(item.get("original") or texts[idx]),
                "full_translation": ft,
                "image_translation": it,
            })
        else:
            val = str(item).strip()
            result.append({
                "original": texts[idx],
                "full_translation": val,
                "image_translation": val,
            })

    return result
