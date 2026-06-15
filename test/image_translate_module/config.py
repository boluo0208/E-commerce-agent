"""Independent configuration for the image-translate module.

Every value can be set via constructor – no dependency on the main app's Settings.
Environment variables are read as defaults when the constructor is not used.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModuleConfig:
    """All knobs for the image → OCR → translate → render pipeline."""

    # -- LLM / translation API ------------------------------------------------
    translate_api_key: str = ""
    translate_base_url: str = "https://api.deepseek.com"
    translate_model: str = "deepseek-chat"
    translate_temperature: float = 0.05
    translate_timeout: int = 60

    # -- Mimo vision (optional fallback for Chinese-text detection) -----------
    mimo_api_key: str = ""
    mimo_base_url: str = ""
    mimo_model: str = ""
    mimo_timeout: int = 60

    # -- Mock switches (work without real API keys) ---------------------------
    mock_translate_when_no_key: bool = True
    mock_vision_when_no_key: bool = True

    # -- Pipeline behaviour ---------------------------------------------------
    translate_image_text: bool = True
    translate_image_text_min_confidence: float = 0.55
    ocr_scale_factor: int = 3

    # -- Image preprocessing --------------------------------------------------
    resize_enabled: bool = False
    resize_target_size: tuple[int, int] = (660, 900)
    resize_quality: int = 95

    auto_split_composite: bool = False
    split_min_parts: int = 2
    split_max_parts: int = 6

    # -- Output ---------------------------------------------------------------
    output_jpeg_quality: int = 95

    @classmethod
    def from_env(cls) -> "ModuleConfig":
        """Build a config from the standard environment variables."""
        import os

        return cls(
            translate_api_key=os.getenv("TRANSLATE_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")),
            translate_base_url=os.getenv("TRANSLATE_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")),
            translate_model=os.getenv("TRANSLATE_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
            translate_temperature=float(os.getenv("TRANSLATE_TEMPERATURE", "0.05")),
            translate_timeout=int(os.getenv("TRANSLATE_TIMEOUT", "60")),
            mimo_api_key=os.getenv("MIMO_API_KEY", ""),
            mimo_base_url=os.getenv("MIMO_BASE_URL", ""),
            mimo_model=os.getenv("MIMO_MODEL", ""),
            mimo_timeout=int(os.getenv("MIMO_TIMEOUT", "60")),
            mock_translate_when_no_key=os.getenv("MOCK_TRANSLATE_WHEN_NO_KEY", "true").lower() == "true",
            mock_vision_when_no_key=os.getenv("MOCK_VISION_WHEN_NO_KEY", "true").lower() == "true",
            translate_image_text=os.getenv("TRANSLATE_IMAGE_TEXT", "true").lower() == "true",
            translate_image_text_min_confidence=float(os.getenv("TRANSLATE_IMAGE_TEXT_MIN_CONFIDENCE", "0.55")),
            ocr_scale_factor=int(os.getenv("OCR_SCALE_FACTOR", "3")),
            resize_enabled=os.getenv("RESIZE_ENABLED", "false").lower() == "true",
            auto_split_composite=os.getenv("AUTO_SPLIT_COMPOSITE", "false").lower() == "true",
        )


# Module-level default (lazy – created on first access via get_config).
_default_config: ModuleConfig | None = None


def get_config() -> ModuleConfig:
    global _default_config
    if _default_config is None:
        _default_config = ModuleConfig.from_env()
    return _default_config


def set_config(config: ModuleConfig) -> None:
    global _default_config
    _default_config = config
