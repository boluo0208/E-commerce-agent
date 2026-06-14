from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "E-commerce Product Content Agent"
    app_env: str = "development"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    mimo_api_key: str = ""
    mimo_base_url: str = ""
    mimo_model: str = ""
    mock_llm_when_no_key: bool = True
    mock_vision_when_no_key: bool = True
    max_concurrent_images: int = 4
    auto_split_composite_images: bool = True
    translate_image_text: bool = True
    translate_image_text_min_confidence: float = 0.55
    upload_dir: Path = Field(default=Path("uploads"))
    output_dir: Path = Field(default=Path("outputs"))
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()
