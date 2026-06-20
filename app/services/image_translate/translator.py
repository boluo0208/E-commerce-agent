"""Translation: call an OpenAI-compatible LLM API for image text copy."""

from __future__ import annotations

import json

import httpx

from .config import ModuleConfig
from .schemas import TranslateError


TRANSLATION_SYSTEM_PROMPT = (
    "You translate Chinese text printed on product images into natural English "
    "for the same product image. Preserve the original meaning, selling point, "
    "numbers, and tone. Use natural cross-border e-commerce English, not "
    "literal word-for-word translation. For slogans and headlines, write a "
    "compact natural phrase, not disconnected keywords. For feature labels and "
    "product-detail captions, use short marketplace copy that can fit back into "
    "a small image text box. Do not invent material, functions, numbers, sizes, "
    "certifications, claims, or product benefits that are not in the source. "
    "Keep brand names, model names, years, sizes, and units unchanged unless a "
    "unit conversion is explicitly requested by the rules. Use standard English "
    "spelling only. Do not output pseudo-English, misspelled words, random "
    "words, mixed Chinese-English fragments, or unreadable abbreviations. "
    "Return exactly one translation object for each input item, in order. "
    "Return valid JSON only."
)


def _build_translation_user_message(
    texts: list[str],
    text_types: list[str] | None = None,
) -> str:
    """Build the JSON user prompt for batch translation."""
    items: list[dict] = []
    for index, text in enumerate(texts):
        entry: dict = {"text": text}
        if text_types and index < len(text_types):
            entry["type"] = text_types[index]
        items.append(entry)

    return json.dumps(
        {
            "items": items,
            "rules": {
                "full_translation": (
                    "Complete natural English translation. For headlines, make "
                    "it a natural product-image headline phrase. Preserve all "
                    "numbers and product facts."
                ),
                "image_translation": (
                    "English to draw back on the image as real text. Keep it "
                    "short, natural, and easy to read. Use 2-8 words for labels "
                    "and 3-10 words for headlines. Prefer common e-commerce "
                    "phrases. Do not shorten so much that meaning becomes "
                    "awkward. Do not use rare abbreviations."
                ),
            },
            "type_rules": {
                "hero_headline": {
                    "applies_to": "items where type='hero_headline'",
                    "full_translation": (
                        "Short punchy English for a hero product badge at the "
                        "top of a product image. Use 6-12 words max and 2 lines "
                        "max. Preserve key numbers. Convert Chinese weight unit "
                        "'jin' to kg only when the source clearly uses jin: "
                        "kg = jin / 2. Keep 'XL' as 'XL'. Keep weight ranges in "
                        "clear forms like '40-80 kg' or '85-160 lb'. Do not "
                        "produce long sentences."
                    ),
                    "image_translation": (
                        "Same as full_translation, but keep it extra compact: "
                        "6-10 words, 2 lines max, e-commerce badge style."
                    ),
                    "examples": [
                        "Imported Elastic, Snug Fit",
                        "Fits 40-80 kg",
                        "Stretchy Cuff Design",
                    ],
                },
                "bottom_hero_headline": {
                    "applies_to": "items where type='bottom_hero_headline'",
                    "full_translation": (
                        "Short bold English for the main bottom headline on a "
                        "product image. Use 5-10 words max, preferably 2 lines. "
                        "Keep the key selling point, but do not produce a full "
                        "sentence or paragraph."
                    ),
                    "image_translation": (
                        "Same as full_translation. Use punchy headline copy, "
                        "2 lines max, 5-10 words."
                    ),
                    "examples": [
                        "Full-Sole Grip",
                        "Cushions Every Step",
                        "No-Slip Support",
                    ],
                },
                "compact_label": {
                    "applies_to": "items where type='compact_label'",
                    "full_translation": (
                        "Short marketing English for a small promo badge. Use "
                        "2-5 words maximum. Keep numbers and promotion "
                        "thresholds such as 1600 or 50% when present."
                    ),
                    "image_translation": (
                        "Same as full_translation for compact labels. Keep it "
                        "very short, 2-4 words."
                    ),
                    "examples": [
                        "Member Gift",
                        "Free Gift Over 1600",
                        "Buy 2 Get 1 Free",
                        "Limited Offer",
                    ],
                },
                "title": {
                    "applies_to": "items where type='title'",
                    "full_translation": (
                        "Natural English feature title. Use title case when it "
                        "looks like a heading. Keep it concise and commercially "
                        "clear."
                    ),
                    "image_translation": (
                        "Use 2-6 words when possible. Keep title meaning clear."
                    ),
                    "examples": [
                        "Five-Toe Design",
                        "Elastic Cuff Design",
                        "Premium Cotton Fabric",
                    ],
                },
                "body": {
                    "applies_to": "items where type='body'",
                    "full_translation": (
                        "Natural English explanatory copy. Preserve facts, "
                        "materials, numbers, and warnings."
                    ),
                    "image_translation": (
                        "Use a short readable phrase. Prefer one line; use two "
                        "lines only when needed."
                    ),
                    "examples": [
                        "Flexible & Comfortable",
                        "Breathable & Soft",
                        "Secure Grip, No Slipping",
                    ],
                },
            },
            "quality_rules": [
                "Every English word must be correctly spelled.",
                "Do not output random English words.",
                "Do not output invented words.",
                "Do not leave Chinese characters in image_translation.",
                "Do not add new product claims absent from the source.",
                "Keep image_translation shorter than full_translation when space is tight.",
            ],
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


def _mock_translations(texts: list[str]) -> list[dict]:
    """Return identity translations when no API key is configured."""
    return [
        {"original": text, "full_translation": text, "image_translation": text}
        for text in texts
    ]


async def translate_texts_to_english(
    texts: list[str],
    config: ModuleConfig,
    text_types: list[str] | None = None,
) -> list[dict]:
    """Translate a batch of Chinese strings to English via an LLM."""
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

    if isinstance(data, list):
        translations = data
    elif isinstance(data, dict):
        translations = data.get("translations")
    else:
        translations = None

    if not isinstance(translations, list):
        raise TranslateError("Translation JSON missing 'translations' key")

    if len(translations) < len(texts):
        translations = list(translations) + [
            texts[index] for index in range(len(translations), len(texts))
        ]
    elif len(translations) > len(texts):
        translations = translations[: len(texts)]

    result: list[dict] = []
    for index, item in enumerate(translations):
        if isinstance(item, str):
            full_translation = item.strip()
            image_translation = full_translation
            original = texts[index]
        elif isinstance(item, dict):
            full_translation = str(
                item.get("full_translation") or item.get("image_translation") or texts[index]
            ).strip()
            image_translation = str(
                item.get("image_translation") or item.get("full_translation") or texts[index]
            ).strip()
            original = str(item.get("original") or texts[index])
        else:
            full_translation = str(item).strip()
            image_translation = full_translation
            original = texts[index]

        result.append({
            "original": original,
            "full_translation": full_translation,
            "image_translation": image_translation,
        })

    return result
